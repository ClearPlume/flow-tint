from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.flow_tint.model.component.color_model import ColorModel
from src.flow_tint.model.component.model_config import ModelConfig
from src.flow_tint.model.component.tokenizer import Tokenizer
from src.flow_tint.path import get_models_dir, get_data_dir


class Tester:
    """统一颜色语义模型的交互式测试器"""

    def __init__(
            self,
            model_path: Path,
            tokenizer_path: str,
            device: str = 'cuda'
    ):
        self.device = device
        self.tokenizer = Tokenizer(tokenizer_path)

        # 加载模型
        self.model = ColorModel(ModelConfig.small()).to(device)

        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

        print("✓ 模型加载成功")
        print(f"模型参数量：{sum(param.numel() for param in self.model.parameters()) / 1e6:.1f}M")

    @torch.no_grad()
    def generate(
            self,
            text: str,
            max_length: int = 128,
            temperature: float = 1.0,
            top_p: float = 0.9,
            do_sample: bool = True
    ) -> str:
        """生成操作序列"""

        # 编码输入
        src_tokens = self.tokenizer.encode(text)
        src_ids = torch.tensor([src_tokens]).to(self.device)
        src_mask = torch.zeros(src_ids.shape, dtype=torch.bool).to(self.device)

        # 编码器处理
        memory = self.model.encode(src_ids, src_mask)

        # 初始化解码序列 [BOS]
        generated = [self.tokenizer.bos_token_id]

        for _ in range(max_length):
            # 当前解码序列
            tgt_ids = torch.tensor([generated]).to(self.device)

            # 解码一步
            logits = self.model.decode(memory, src_mask, tgt_ids)

            # 采样策略
            if do_sample:
                # Temperature scaling
                logits = logits / temperature

                # Top-p sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # 移除累积概率超过top_p的tokens
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices,
                                                                         src=sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')

                # 采样
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
            else:
                # 贪心解码
                next_token = logits.argmax(dim=-1).item()

            generated.append(next_token)

            # 检查终止条件
            if next_token == self.tokenizer.eos_token_id:
                break

        # 解码生成的序列（去掉BOS/EOS）
        output_tokens = [t for t in generated[1:] if t != self.tokenizer.eos_token_id]
        result = self.tokenizer.decode(output_tokens)

        return result

    def parse_operations(self, generated_text: str) -> dict:
        """解析生成的操作序列"""
        lines = generated_text.strip().split('\n')
        operations = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith('color '):
                # 基础颜色坐标
                parts = line.split()
                if len(parts) == 4:
                    try:
                        l_val = int(parts[1]) / 1000.0
                        c_val = int(parts[2]) / 1000.0
                        h_val = int(parts[3]) / 1000.0
                        operations.append({
                            'type': 'color',
                            'values': [l_val, c_val, h_val]
                        })
                    except ValueError:
                        operations.append({
                            'type': 'invalid',
                            'raw': line
                        })
            elif '$p' in line:
                # 操作函数
                parts = line.split()
                if len(parts) >= 3:
                    func_name = parts[0]
                    try:
                        param = float(parts[2])
                        operations.append({
                            'type': 'function',
                            'function': func_name,
                            'parameter': param
                        })
                    except ValueError:
                        operations.append({
                            'type': 'invalid',
                            'raw': line
                        })
            else:
                operations.append({
                    'type': 'unknown',
                    'raw': line
                })

        return {
            'operations': operations,
            'valid': all(op['type'] in ['color', 'function'] for op in operations),
            'raw_text': generated_text
        }

    def analyze_generation(self, input_text: str, expected_output: str = None):
        """详细分析生成结果"""
        print("\n" + "=" * 60)
        print(f"输入文本: {repr(input_text)}")
        print("=" * 60)

        # 生成结果
        generated = self.generate(input_text, do_sample=False)  # 贪心解码获得稳定结果
        print(f"\n【生成结果】")
        print(f"原始输出: {repr(generated)}")

        # 解析操作
        parsed = self.parse_operations(generated)
        print(f"\n【操作解析】")
        print(f"有效性: {'✓' if parsed['valid'] else '✗'}")
        print(f"操作数量: {len(parsed['operations'])}")

        for i, op in enumerate(parsed['operations']):
            if op['type'] == 'color':
                print(f"  {i + 1}. 颜色坐标: L={op['values'][0]:.3f}, C={op['values'][1]:.3f}, H={op['values'][2]:.3f}")
            elif op['type'] == 'function':
                print(f"  {i + 1}. 函数调用: {op['function']}(参数={op['parameter']})")
            else:
                print(f"  {i + 1}. 无效操作: {op.get('raw', op)}")

        # 与期望结果对比
        if expected_output:
            print(f"\n【预期对比】")
            print(f"期望输出: {repr(expected_output)}")

            # 简单的文本匹配
            normalized_generated = ' '.join(generated.split())
            normalized_expected = ' '.join(expected_output.split())

            match = normalized_generated == normalized_expected
            print(f"完全匹配: {'✓' if match else '✗'}")

            if not match:
                # 计算编辑距离或其他相似度指标
                import difflib
                diff = list(difflib.unified_diff(
                    normalized_expected.split(),
                    normalized_generated.split(),
                    fromfile='expected',
                    tofile='generated',
                    lineterm=''
                ))
                if diff:
                    print("差异:")
                    for line in diff[:10]:  # 只显示前10行差异
                        print(f"  {line}")

    def interactive_test(self):
        """交互式测试循环"""
        print("\n" + "=" * 60)
        print("统一颜色语义模型交互式测试")
        print("=" * 60)
        print("支持的输入类型:")
        print("- 基础颜色: 红、蓝、朱红、天青等")
        print("- 修饰颜色: 深红、亮一些的蓝、偏红的紫等")
        print("- 对话序列: 包含历史上下文的多轮对话")
        print("-" * 60)
        print("命令: quit-退出, help-示例, analyze <文本>-详细分析")
        print("生成参数: temp <值>-设置temperature, sample-切换采样模式")
        print("=" * 60)

        # 生成参数
        temperature = 1.0
        do_sample = True

        examples = [
            "红",
            "深红",
            "亮一些的蓝",
            "偏红的紫色",
            "红色\ncolor 628 258 29227\n深一点"
        ]

        while True:
            try:
                user_input = input(f"\n[T={temperature:.1f},Sample={do_sample}]> ").strip()

                if user_input.lower() == 'quit':
                    print("再见!")
                    break

                if user_input.lower() == 'help':
                    print("\n示例输入:")
                    for example in examples:
                        print(f"  {repr(example)}")
                        result = self.generate(example.replace('\\n', '\n'),
                                               temperature=temperature, do_sample=do_sample)
                        print(f"  → {repr(result)}\n")
                    continue

                if user_input.startswith('analyze '):
                    text = user_input[8:].replace('\\n', '\n')
                    self.analyze_generation(text)
                    continue

                if user_input.startswith('temp '):
                    try:
                        temperature = float(user_input[5:])
                        temperature = max(0.1, min(2.0, temperature))
                        print(f"Temperature设置为: {temperature}")
                        continue
                    except ValueError:
                        print("无效的temperature值")
                        continue

                if user_input == 'sample':
                    do_sample = not do_sample
                    print(f"采样模式: {'开启' if do_sample else '关闭(贪心)'}")
                    continue

                # 处理转义字符
                text = user_input.replace('\\n', '\n')

                # 快速生成
                result = self.generate(text, temperature=temperature, do_sample=do_sample)
                print(f"→ {repr(result)}")

            except KeyboardInterrupt:
                print("\n\n中断!")
                break
            except Exception as e:
                print(f"错误: {e}")
                import traceback
                traceback.print_exc()

    def batch_test(self, test_cases: list[tuple[str, str]]):
        """批量测试"""
        print("\n批量测试结果:")
        print("-" * 80)

        correct = 0
        total = len(test_cases)

        for i, (input_text, expected_output) in enumerate(test_cases):
            print(f"\n[{i + 1}/{total}] 测试: {repr(input_text)}")

            generated = self.generate(input_text, do_sample=False)  # 使用贪心解码确保一致性

            # 标准化比较（去除多余空格和换行）
            normalized_generated = ' '.join(generated.split())
            normalized_expected = ' '.join(expected_output.split())

            match = normalized_generated == normalized_expected

            if match:
                correct += 1
                print(f"✓ 生成: {repr(generated)}")
            else:
                print(f"✗ 生成: {repr(generated)}")
                print(f"  期望: {repr(expected_output)}")

        print(f"\n批量测试总结:")
        print(f"总数: {total}, 正确: {correct}, 准确率: {correct / total:.2%}")

    def evaluate_model(self, eval_cases: list[tuple[str, str]], show_progress: bool = True):
        """模型性能评估"""
        total_cases = len(eval_cases)
        exact_matches = 0
        valid_parses = 0

        if show_progress:
            progress = tqdm(eval_cases, desc="评估中")
        else:
            progress = eval_cases

        for input_text, expected_output in progress:
            generated = self.generate(input_text, do_sample=False)

            # 精确匹配
            if generated.strip() == expected_output.strip():
                exact_matches += 1

            # 解析有效性
            parsed = self.parse_operations(generated)
            if parsed['valid']:
                valid_parses += 1

        print(f"\n模型评估结果:")
        print(f"总样本数: {total_cases}")
        print(f"精确匹配: {exact_matches}/{total_cases} ({exact_matches / total_cases:.2%})")
        print(f"有效解析: {valid_parses}/{total_cases} ({valid_parses / total_cases:.2%})")

        return {
            'exact_match_rate': exact_matches / total_cases,
            'valid_parse_rate': valid_parses / total_cases,
            'total_cases': total_cases
        }


if __name__ == '__main__':
    # 初始化测试器
    tester = Tester(
        model_path=get_models_dir() / 'checkpoints' / 'color_semantic_20251006_194128.pt',
        tokenizer_path=f"{get_data_dir()}/vocab/color_semantic.model",
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # 方式1: 单个分析
    tester.analyze_generation("深红")

    # 方式2: 批量测试
    test_cases = [
        ("红", "color 628 258 29227"),
        ("深红", "color 400 164 29227"),
        ("偏红的蓝", "color 452 313 264059<sep>color 628 258 29227<sep>mix $p $p 0.109"),
    ]
    tester.batch_test(test_cases)

    # 方式3: 交互式测试
    tester.interactive_test()
