#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三阶段颜色语料生成器（改进版）
新增hue（色相）和temperature（色温）维度
基于一、二阶段数据，生成多重修饰的复杂颜色表达
支持读取真实的一阶段语料文件
"""

import random
from pathlib import Path
from typing import Dict, List, Tuple

from src.flow_tint.path import get_data_dir
from src.flow_tint.core.utils import add_fluctuation


def check_semantic_conflict(color_name: str, dimension: str, operation: str) -> bool:
    """
    检查是否存在严重语义冲突
    - hue维度不检查冲突

    :param color_name: 红/绿
    :param dimension: lightness/chroma/hue/mix/temperature
    :param operation: 红_mix/绿_mix
    """

    # hue维度是纯粹的数学调整，不检查语义冲突
    if dimension == 'hue':
        return False

    # 只检查mix维度的冲突
    if dimension == 'mix':
        target = operation.replace('_mix', '')
        return bool(set(color_name) & set(target))

    return False


class Stage3Generator:
    base_colors: dict[str, str]

    def __init__(self):
        self.data_dir = get_data_dir()

        # 基础颜色列表
        self.base_colors = {}
        self.parse_stage1_data()

        # 扩展的修饰词体系：新增hue和temperature维度
        self.modifiers = {
            'lightness': {
                'darken': ['深', '暗', '浓'],
                'brighten': ['浅', '淡', '亮', '明']
            },
            'chroma': {
                'enhance': ['鲜', '艳', '纯', '正'],
                'reduce': ['灰', '浊', '素']
            },
            'hue': {
                'hue_shift_positive': ['色相偏移', 'H值增加', '色相旋转'],
                'hue_shift_negative': ['色相回调', 'H值减少', '色相反转']
            },
            'temperature': {
                'warmer': ['暖调', '温暖', '热情'],
                'cooler': ['冷调', '清冷', '沉静']
            },
            'mix': self.generate_mix_modifiers()
        }

        # 强度词分配策略：为新维度添加配置
        self.intensity_levels = {
            'lightness': {
                'light': {
                    'prefixes': ['稍微', '一点', '略'],
                    'factors': [0.02, 0.03, 0.04, 0.05, 0.06]
                },
                'medium': {
                    'prefixes': [''],
                    'factors': [0.08, 0.09, 0.1, 0.11, 0.12]
                },
                'strong': {
                    'prefixes': ['很', '比较'],
                    'factors': [0.15, 0.16, 0.17, 0.18, 0.19]
                }
            },

            'chroma': {
                'light': {
                    'prefixes': ['稍微', '一点', '略'],
                    'factors': [0.02, 0.03, 0.04, 0.05, 0.06]
                },
                'medium': {
                    'prefixes': [''],
                    'factors': [0.08, 0.09, 0.1, 0.11, 0.12]
                },
                'strong': {
                    'prefixes': ['很', '比较'],
                    'factors': [0.15, 0.16, 0.17, 0.18, 0.19]
                }
            },

            'hue': {
                'light': {
                    'prefixes': ['稍微', '一点', '略'],
                    'factors': [12.0, 13.5, 15.0, 16.5, 18.0]
                },
                'medium': {
                    'prefixes': [''],
                    'factors': [27.0, 28.5, 30.0, 31.5, 33.0]
                },
                'strong': {
                    'prefixes': ['很', '比较'],
                    'factors': [42.0, 43.5, 45.0, 46.5, 48.0]
                }
            },

            'temperature': {
                'light': {
                    'prefixes': ['稍微', '一点', '略'],
                    'factors': [0.03, 0.04, 0.05, 0.06, 0.07]
                },
                'medium': {
                    'prefixes': [''],
                    'factors': [0.08, 0.09, 0.1, 0.11, 0.12]
                },
                'strong': {
                    'prefixes': ['很', '比较'],
                    'factors': [0.13, 0.14, 0.15, 0.16, 0.17]
                }
            },

            'mix': {
                'light': {
                    'prefixes': ['稍微', '一点', '略'],
                    'factors': [0.03, 0.04, 0.05, 0.06, 0.07]
                },
                'medium': {
                    'prefixes': [''],
                    'factors': [0.08, 0.09, 0.1, 0.11, 0.12]
                }
            }
        }

        # 色温调整参考色（用于temperature混合）
        self.temperature_colors = {
            'warmer': 'color 950 20 85000',  # 暖色调参考（暖白）
            'cooler': 'color 950 20 240000'  # 冷色调参考（冷白）
        }

    def generate_mix_modifiers(self) -> dict[str, str]:
        """从基础颜色动态生成mix修饰词"""
        mix_dict = {}

        for color in self.base_colors:
            key = f'{color}_mix'
            mix_dict[key] = [f'偏{color}', f'泛{color}', f'带{color}', ]

        return mix_dict

    def parse_stage1_data(self):
        """解析一阶段数据，提取颜色名称和坐标"""

        # 读取一阶段数据
        try:
            with open(self.data_dir / 'samples/stage1_samples.txt', 'r', encoding='utf-8') as f:
                stage1_text = f.read()
        except FileNotFoundError:
            raise "请确保 stage1_samples.txt 文件存在"

        color_samples = stage1_text.strip().split('\n\n')

        for color_sample in color_samples:
            color_name, coords = color_sample.strip().split('\n')
            self.base_colors[color_name] = coords

    def check_multi_modifier_conflict(self, color_name: str, modifiers_list: List[Tuple[str, str]]) -> bool:
        """检查多个修饰词之间的冲突 - 扩展版"""

        # 1. 同维度重复检查
        dimensions = [dim for dim, _ in modifiers_list]
        if len(set(dimensions)) != len(dimensions):
            return True

        # 2. 对立操作检查 - 扩展版
        operations = [op for _, op in modifiers_list]
        conflicts = [
            ('darken', 'brighten'),
            ('enhance', 'reduce'),
            ('warmer', 'cooler'),  # 新增：色温冲突
        ]

        for op1, op2 in conflicts:
            if op1 in operations and op2 in operations:
                return True

        # 3. hue维度内部的对立操作检查
        hue_ops = [op for dim, op in modifiers_list if dim == 'hue']
        if 'hue_shift_positive' in hue_ops and 'hue_shift_negative' in hue_ops:
            return True  # 不允许同时正向和负向调整色相

        # 4. 语义冲突检查
        for dimension, operation in modifiers_list:
            return check_semantic_conflict(color_name, dimension, operation)

        return False

    def generate_function_sequence(self, modifiers_list: List[Tuple[str, str, float]],
                                   original_order: List[Tuple[str, str]]) -> List[str]:
        """根据语言描述的自然顺序生成函数调用序列 - 扩展版"""

        function_calls = []

        # 按照原始语言顺序处理
        for dimension, operation in original_order:
            # 找到对应的factor
            factor = None
            for dim, op, f in modifiers_list:
                if dim == dimension and op == operation:
                    factor = f
                    break

            if factor is None:
                continue

            if dimension == 'mix':
                # 颜色混合：需要先添加目标颜色
                mix_color = self.base_colors[operation.removesuffix('_mix')]
                function_calls.append(mix_color)
                function_calls.append(f"mix $p $p {factor}")

            elif dimension == 'lightness':
                sign = '-' if operation == 'darken' else ''
                function_calls.append(f"lightness $p {sign}{factor}")

            elif dimension == 'chroma':
                sign = '' if operation == 'enhance' else '-'
                function_calls.append(f"chroma $p {sign}{factor}")

            elif dimension == 'hue':
                # 色相调整：纯粹的H值角度调整
                if operation == 'hue_shift_positive':
                    function_calls.append(f"hue $p {factor}")
                elif operation == 'hue_shift_negative':
                    function_calls.append(f"hue $p -{factor}")

            elif dimension == 'temperature':
                # 色温调整：混合暖色或冷色参考
                temp_color = self.temperature_colors[operation]
                function_calls.append(temp_color)
                function_calls.append(f"temperature $p $p {factor}")

        return function_calls

    def generate_input_text(self, color_name: str, modifiers_info: List[Tuple[str, str, str, float]]) -> Tuple[
        str, List[Tuple[str, str]]]:
        """生成自然的输入文本 - 改进版：hue调整使用半连接形式"""

        # 按维度分类修饰词
        hue_modifiers = []  # hue单独处理
        prefix_words = []  # 前置修饰（mix, temperature）
        core_modifiers = []  # 核心修饰（lightness, chroma）

        # 记录对应的操作信息
        hue_ops = []
        prefix_ops = []
        intensity_word = ""

        for dim, op, intensity, factor in modifiers_info:
            if dim == 'hue':
                # hue调整：生成"色相旋转±X度，"格式
                if op == 'hue_shift_positive':
                    hue_text = f"色相旋转+{factor}度，"
                elif op == 'hue_shift_negative':
                    hue_text = f"色相旋转-{factor}度，"
                hue_modifiers.append(hue_text)
                hue_ops.append((dim, op))

            elif dim in ['mix', 'temperature']:
                word = random.choice(self.modifiers[dim][op])
                prefix_words.append(word)
                prefix_ops.append((dim, op))

            elif dim in ['lightness', 'chroma']:
                word = random.choice(self.modifiers[dim][op])
                core_modifiers.append((word, dim, op))

            # 获取强度词（排除hue维度）
            if not intensity_word and intensity != 'medium' and dim != 'hue':
                intensity_data = self.intensity_levels[dim][intensity]
                prefixes = intensity_data['prefixes']
                if prefixes and prefixes[0]:
                    intensity_word = random.choice(prefixes)

        # 按语法顺序组合文本
        result_parts = []

        # 1. hue调整前缀（技术性描述，用逗号分隔）
        if hue_modifiers:
            result_parts.extend(hue_modifiers)

        # 2. 强度词
        if intensity_word:
            result_parts.append(intensity_word)

        # 3. 前置修饰词
        if prefix_words:
            if len(prefix_words) == 1:
                result_parts.append(prefix_words[0])
            else:
                prefix_part = prefix_words[0] + "的" + "".join(prefix_words[1:])
                result_parts.append(prefix_part)

        # 4. 添加"的"字
        need_de = prefix_words and core_modifiers
        if need_de:
            result_parts.append("的")

        # 5. 核心修饰词
        if core_modifiers:
            core_words = [word for word, _, _ in core_modifiers]
            core_part = "".join(core_words)
            result_parts.append(core_part)

        # 6. 颜色名称
        result_parts.append(color_name)

        # 生成文本
        input_text = "".join(result_parts).replace("的的", "的")

        # 确定处理顺序：由内向外（从右到左）
        processing_order = []

        # 核心修饰词：倒序处理（离被修饰词越近越先处理）
        core_ops = [(dim, op) for _, dim, op in core_modifiers]
        core_ops.reverse()
        processing_order.extend(core_ops)

        # 前置修饰：也要倒序处理（离被修饰词越近越先处理）
        prefix_ops.reverse()
        processing_order.extend(prefix_ops)

        # hue调整：最后处理（最外层）
        processing_order.extend(hue_ops)

        return input_text, processing_order

    def generate_multi_modifier_sample(self, color_name: str, coords: str, modifier_count: int = 2) -> Tuple[str, str]:
        """生成多重修饰的单个样本 - 按实际使用频率分配权重"""

        # 定义各维度的使用权重（符合日常使用频率）
        dimension_weights = {
            'lightness': 40,  # 最常用：明暗调整
            'mix': 35,  # 很常用：颜色混合
            'chroma': 15,  # 中等：饱和度调整
            'temperature': 8,  # 较少：色温调整
            'hue': 2  # 最少：技术性色相调整
        }

        # 收集各维度的可用修饰词
        available_by_dimension = {}
        for dim in self.modifiers:
            available_modifiers = []
            for op in self.modifiers[dim]:
                if not check_semantic_conflict(color_name, dim, op):
                    available_modifiers.append((dim, op))
            if available_modifiers:
                available_by_dimension[dim] = available_modifiers

        # 按权重随机选择维度
        available_dimensions = list(available_by_dimension.keys())
        weights = [dimension_weights.get(dim, 1) for dim in available_dimensions]

        # 尝试生成有效组合
        max_attempts = 50
        for attempt in range(max_attempts):
            # 按权重选择维度
            if len(available_dimensions) < modifier_count:
                raise ValueError(f"可用维度不足: {len(available_dimensions)} < {modifier_count}")

            # 使用加权随机选择
            selected_dims = []
            temp_dims = available_dimensions.copy()
            temp_weights = weights.copy()

            for _ in range(modifier_count):
                # 按权重选择一个维度
                import random
                selected_dim = random.choices(temp_dims, weights=temp_weights, k=1)[0]
                selected_dims.append(selected_dim)

                # 移除已选择的维度，避免重复
                dim_index = temp_dims.index(selected_dim)
                temp_dims.pop(dim_index)
                temp_weights.pop(dim_index)

                if not temp_dims:  # 如果维度用完了就停止
                    break

            # 为每个选中的维度随机选择具体操作
            selected = []
            for dim in selected_dims:
                available_ops = available_by_dimension[dim]
                selected.append(random.choice(available_ops))

            if not self.check_multi_modifier_conflict(color_name, selected):
                break
        else:
            raise ValueError(f"经过{max_attempts}次尝试仍无法生成有效组合")

        # 为每个修饰词分配强度和因子
        modifiers_with_factors = []
        global_intensity = random.choice(['light', 'medium', 'strong']) if random.random() < 0.6 else 'medium'
        intensity_assigned = False

        for i, (dim, op) in enumerate(selected):
            if not intensity_assigned and global_intensity != 'medium':
                intensity = global_intensity
                intensity_assigned = True
            else:
                intensity = 'medium'

            # 计算因子
            available_intensities = list(self.intensity_levels[dim].keys())
            if intensity not in available_intensities:
                intensity = 'medium'

            factor = add_fluctuation(random.choice(self.intensity_levels[dim][intensity]['factors']))
            modifiers_with_factors.append((dim, op, intensity, factor))

        # 生成输入文本
        input_text, language_order = self.generate_input_text(color_name, modifiers_with_factors)

        # 生成输出序列
        factor_data = [(dim, op, factor) for dim, op, _, factor in modifiers_with_factors]
        function_calls = self.generate_function_sequence(factor_data, language_order)
        output_text = f"{coords}\n" + "\n".join(function_calls)

        return input_text, output_text

    def generate_stage3_corpus(self, samples_per_color: int = 20) -> str:
        """生成完整的三阶段语料"""
        print(f"解析到 {len(self.base_colors)} 个颜色")

        all_samples = []
        successful_colors = 0

        for color_name, coords in self.base_colors.items():
            color_samples = []
            target_samples = samples_per_color

            # 2重修饰（70%）
            two_mod_target = int(target_samples * 0.7)
            for _ in range(two_mod_target):
                try:
                    sample = self.generate_multi_modifier_sample(color_name, coords, 2)
                    color_samples.append(sample)
                except Exception:
                    continue

            # 3重修饰（30%）
            three_mod_target = target_samples - len(color_samples)
            for _ in range(three_mod_target):
                try:
                    sample = self.generate_multi_modifier_sample(color_name, coords, 3)
                    color_samples.append(sample)
                except Exception:
                    continue

            if len(color_samples) > 0:
                all_samples.extend(color_samples)
                successful_colors += 1

                if successful_colors % 50 == 0:
                    print(f"已处理 {successful_colors} 个颜色，生成样本 {len(all_samples)} 条")

        print(f"成功处理 {successful_colors}/{len(self.base_colors)} 个颜色")
        print(f"总共生成 {len(all_samples)} 条三阶段语料")

        # 随机打乱
        random.shuffle(all_samples)

        # 格式化输出
        corpus_lines = []
        for input_text, output_text in all_samples:
            corpus_lines.append(input_text)
            corpus_lines.append(output_text)
            corpus_lines.append("")

        return '\n'.join(corpus_lines)

    def analyze_corpus(self, corpus: str) -> Dict:
        """分析生成的语料统计信息 - 扩展版"""
        samples = corpus.split('\n\n')

        stats = {
            'total_samples': len(samples),
            'dimensions': {'lightness': 0, 'chroma': 0, 'hue': 0, 'temperature': 0, 'mix': 0},
            'complexity': {'two_modifiers': 0, 'three_plus_modifiers': 0}
        }

        for sample in samples:
            text = sample.split('\n')[0]
            modifier_count = 0

            # 统计各维度分布
            if any(kw in text for kw in ['深', '浅', '暗', '亮', '淡', '明']):
                stats['dimensions']['lightness'] += 1
                modifier_count += 1

            if any(kw in text for kw in ['鲜', '艳', '灰', '浊', '纯', '正', '素']):
                stats['dimensions']['chroma'] += 1
                modifier_count += 1

            if any(kw in text for kw in ['色相偏移', '色相旋转', '色相回调', 'H值增加', 'H值减少', '色相反转']):
                stats['dimensions']['hue'] += 1
                modifier_count += 1

            if any(kw in text for kw in ['暖调', '冷调', '温暖', '清冷', '热情', '沉静']):
                stats['dimensions']['temperature'] += 1
                modifier_count += 1

            if any(kw in text for kw in ['偏', '泛', '带']):
                stats['dimensions']['mix'] += 1
                modifier_count += 1

            # 统计复杂度
            if modifier_count >= 3:
                stats['complexity']['three_plus_modifiers'] += 1
            elif modifier_count == 2:
                stats['complexity']['two_modifiers'] += 1

        return stats

    def save_corpus(self, corpus: str, filename: Path):
        """保存语料到文件"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(corpus)
        print(f"语料已保存到 {filename}")


def main():
    """主函数"""
    # 创建生成器
    generator = Stage3Generator()

    # 生成三阶段语料
    print("开始生成三阶段语料...")
    corpus = generator.generate_stage3_corpus(samples_per_color=30)

    # 分析统计信息
    stats = generator.analyze_corpus(corpus)
    print("\n语料统计:")
    print(f"总样本数: {stats['total_samples']}")
    print(f"维度分布: {stats['dimensions']}")
    print(f"复杂度分布: {stats['complexity']}")

    # 显示前20个样本
    print("\n前20个样本:")
    lines = corpus.split('\n')
    sample_count = 0
    for line in lines:
        print(line)
        if len(line) == 0:
            sample_count += 1
        if sample_count >= 20:
            break

    # 保存语料
    generator.save_corpus(corpus, generator.data_dir / 'samples/stage2_complex_samples.txt')


if __name__ == "__main__":
    main()
