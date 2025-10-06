#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
二阶段颜色语料生成器（改进版）
基于一阶段颜色数据，生成修饰词 + 颜色的函数调用语料
新增权重分配，修正hue和temperature函数映射
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


class Stage2Generator:
    base_colors: dict[str, str]

    def __init__(self):
        self.data_dir = get_data_dir()

        # 基础颜色列表
        self.base_colors = {}
        self.parse_stage1_data()

        # 修饰词分类体系
        self.modifiers = {
            'lightness': {
                'darken': ['深', '暗', '浓'],
                'brighten': ['浅', '淡', '亮', '明']
            },
            'chroma': {
                'enhance': ['鲜', '艳', '纯', '正'],
                'reduce': ['灰', '浊', '素']
            },
            'hue': {  # 修正：使用技术术语
                'hue_shift_positive': ['色相偏移', 'H值增加', '色相旋转'],
                'hue_shift_negative': ['色相回调', 'H值减少', '色相反转']
            },
            'mix': self.generate_mix_modifiers(),  # 扩展：包含所有基本颜色
            'temperature': {  # 修正：使用色温描述
                'warmer': ['暖调', '温暖', '热情'],
                'cooler': ['冷调', '清冷', '沉静']
            }
        }

        # 各维度的使用权重（符合日常使用频率）
        self.dimension_weights = {
            'lightness': 40,  # 最常用：明暗调整
            'mix': 35,  # 很常用：颜色混合
            'chroma': 15,  # 中等：饱和度调整
            'temperature': 8,  # 较少：色温调整
            'hue': 2  # 最少：技术性色相调整
        }

        # 强度量化策略（修正版）
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
                },
                'extreme': {
                    'prefixes': ['非常', '特别', '极'],
                    'factors': [0.22, 0.24, 0.25, 0.27, 0.29]
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
                },
                'extreme': {
                    'prefixes': ['非常', '特别'],
                    'factors': [0.22, 0.24, 0.25, 0.27, 0.29]
                }
            },

            'hue': {
                'light': {
                    'prefixes': ['稍微', '一点', '略'],
                    'factors': [12.0, 14.0, 15.0, 16.0, 18.0]
                },
                'medium': {
                    'prefixes': [''],
                    'factors': [27.0, 29.0, 30.0, 31.0, 33.0]
                },
                'strong': {
                    'prefixes': ['比较'],
                    'factors': [42.0, 44.0, 45.0, 47.0, 49.0]
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
            }
        }

        # 函数映射（修正版）
        self.function_map = {
            'lightness': {
                'darken': 'lightness',
                'brighten': 'lightness'
            },
            'chroma': {
                'enhance': 'chroma',
                'reduce': 'chroma'
            },
            'hue': {
                'hue_shift_positive': 'hue',
                'hue_shift_negative': 'hue'
            },
            'mix': self.generate_mix_function_map(),
            'temperature': {
                'warmer': 'temperature',
                'cooler': 'temperature'
            }
        }

        # 色温参考颜色（更新版）
        self.temperature_colors = {
            'warmer': 'color 950 20 85000',  # 暖白
            'cooler': 'color 950 20 240000'  # 冷白
        }

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

    def generate_mix_modifiers(self) -> dict[str, list[str]]:
        """从基础颜色动态生成mix修饰词"""
        mix_dict = {}

        for color in self.base_colors:
            key = f'{color}_mix'
            mix_dict[key] = [f'偏{color}', f'泛{color}', f'带{color}', ]

        return mix_dict

    def generate_mix_function_map(self) -> dict[str, str]:
        """从基础颜色动态生成mix修饰词"""
        mix_dict = {}

        for color in self.base_colors:
            key = f'{color}_mix'
            mix_dict[key] = "mix"

        return mix_dict

    def generate_modified_color(self, color_name: str, coords: str,
                                dimension: str, operation: str, intensity: str) -> Tuple[str, str]:
        """生成修饰后的颜色语料 - 改进版"""

        # 检查语义冲突
        if check_semantic_conflict(color_name, dimension, operation):
            raise ValueError(f"语义冲突: {color_name} + {operation}")

        intensity_data = self.intensity_levels[dimension][intensity]
        factor = add_fluctuation(random.choice(intensity_data['factors']))

        # 构建输入文本（hue维度特殊处理）
        if dimension == 'hue':
            # hue调整使用半连接形式
            if operation == 'hue_shift_positive':
                input_text = f"色相旋转+{factor}度，{color_name}"
            else:  # hue_shift_negative
                input_text = f"色相旋转-{factor}度，{color_name}"
        else:
            # 其他维度正常处理
            modifier_words = self.modifiers[dimension][operation]
            modifier = random.choice(modifier_words)
            prefix = random.choice(intensity_data['prefixes'])

            if prefix:
                input_text = f"{prefix}{modifier}的{color_name}"
            else:
                input_text = f"{modifier}{color_name}"

        # 构建输出（函数调用）
        function_name = self.function_map[dimension][operation]

        if dimension == 'lightness':
            if operation == 'darken':
                value = f"-{factor}"
            else:  # brighten
                value = f"{factor}"
            function_call = f"{function_name} $p {value}"
            output_text = f"{coords}\n{function_call}"

        elif dimension == 'chroma':
            if operation == 'enhance':
                value = f"{factor}"
            else:  # reduce
                value = f"-{factor}"
            function_call = f"{function_name} $p {value}"
            output_text = f"{coords}\n{function_call}"

        elif dimension == 'hue':  # 修正：色相调整
            if operation == 'hue_shift_positive':
                function_call = f"{function_name} $p {factor}"
            else:  # hue_shift_negative
                function_call = f"{function_name} $p -{factor}"
            output_text = f"{coords}\n{function_call}"

        elif dimension == 'mix':
            # 颜色混合：原色 + 目标色
            mix_color = self.base_colors[operation.removesuffix('_mix')]
            function_call = f"{function_name} $p $p {factor}"
            output_text = f"{coords}\n{mix_color}\n{function_call}"

        elif dimension == 'temperature':  # 修正：色温调整
            # 色温调整：原色 + 色温参考色混合
            temp_color = self.temperature_colors[operation]
            function_call = f"temperature $p $p {factor}"
            output_text = f"{coords}\n{temp_color}\n{function_call}"

        return input_text, output_text

    def generate_weighted_combinations(self, color_name: str, coords: str, target_samples: int = 15) -> List[
        Tuple[str, str]]:
        """为单个颜色生成加权的修饰组合"""
        combinations = []

        # 收集各维度的可用操作——dimension，五大基本方法之一
        available_by_dimension = {}
        for dimension in self.modifiers:
            available_ops = []
            # operation: 红_mix/绿_mix
            for operation in self.modifiers[dimension]:
                if not check_semantic_conflict(color_name, dimension, operation):
                    available_ops.append(operation)
            if available_ops:
                available_by_dimension[dimension] = available_ops

        # 按权重分配样本数量（确保每个维度至少1个样本）
        available_dimensions = list(available_by_dimension.keys())
        total_weight = sum(self.dimension_weights.get(dim, 1) for dim in available_dimensions)

        # 先给每个维度分配1个样本（保底）
        samples_per_dimension = {dim: 1 for dim in available_dimensions}
        remaining_samples = target_samples - len(available_dimensions)

        # 如果还有剩余样本，按权重分配
        if remaining_samples > 0:
            for i, dim in enumerate(available_dimensions):
                weight = self.dimension_weights.get(dim, 1)
                if i == len(available_dimensions) - 1:  # 最后一个维度分配剩余样本
                    samples_per_dimension[dim] += remaining_samples
                    remaining_samples = 0
                else:
                    additional = int(remaining_samples * weight / total_weight)
                    samples_per_dimension[dim] += additional
                    remaining_samples -= additional

        # 为每个维度生成对应数量的样本
        for dimension, sample_count in samples_per_dimension.items():
            if sample_count <= 0:
                continue

            available_ops = available_by_dimension[dimension]
            available_intensities = list(self.intensity_levels[dimension].keys())

            # 生成所有可能的组合
            dim_combinations = []
            for operation in available_ops:
                for intensity in available_intensities:
                    try:
                        input_text, output_text = self.generate_modified_color(
                            color_name, coords, dimension, operation, intensity
                        )
                        dim_combinations.append((input_text, output_text))
                    except Exception:
                        continue

            # 随机选择指定数量的样本
            if len(dim_combinations) > sample_count:
                selected = random.sample(dim_combinations, sample_count)
            else:
                selected = dim_combinations

            combinations.extend(selected)

        return combinations

    def generate_stage2_corpus(self, samples_per_color: int = 15) -> str:
        """生成完整的二阶段语料 - 使用权重分配"""
        # 解析一阶段数据
        print(f"解析到 {len(self.base_colors)} 个颜色")

        all_combinations = []

        # 为每个颜色生成加权的修饰变体
        for i, (color_name, coords) in enumerate(self.base_colors.items()):
            combinations = self.generate_weighted_combinations(color_name, coords, samples_per_color)
            all_combinations.extend(combinations)

            # 显示进度
            if (i + 1) % 100 == 0:
                print(f"已处理 {i + 1} 个颜色，累计生成 {len(all_combinations)} 条语料")

        print(f"总共生成 {len(all_combinations)} 条二阶段语料")

        # 随机打乱顺序
        random.shuffle(all_combinations)

        # 格式化输出
        corpus_lines = []
        for input_text, output_text in all_combinations:
            corpus_lines.append(input_text)
            corpus_lines.append(output_text)
            corpus_lines.append("")  # 空行分隔

        return '\n'.join(corpus_lines)

    def save_corpus(self, corpus: str, filename: Path):
        """保存语料到文件"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(corpus)
        print(f"语料已保存到 {filename}")

    def analyze_corpus(self, corpus: str) -> Dict:
        """分析生成的语料统计信息 - 更新版"""
        samples = corpus.split('\n\n')

        stats = {
            'total_samples': len(samples),
            'dimensions': {'lightness': 0, 'chroma': 0, 'hue': 0, 'mix': 0, 'temperature': 0},
            'intensities': {'light': 0, 'medium': 0, 'strong': 0, 'extreme': 0}
        }

        for sample in samples:
            text = sample.split('\n')[0]

            # 统计维度分布（更新关键词）
            if any(kw in text for kw in ['深', '浅', '暗', '亮', '淡', '明']):
                stats['dimensions']['lightness'] += 1

            if any(kw in text for kw in ['鲜', '艳', '灰', '浊', '纯', '正', '素']):
                stats['dimensions']['chroma'] += 1

            if any(kw in text for kw in ['色相偏移', '色相旋转', '色相回调', 'H值增加', 'H值减少', '色相反转']):
                stats['dimensions']['hue'] += 1

            if any(kw in text for kw in ['偏', '泛', '带']):
                stats['dimensions']['mix'] += 1

            if any(kw in text for kw in ['暖调', '温暖', '热情', '冷调', '清冷', '沉静']):
                stats['dimensions']['temperature'] += 1

            # 统计强度分布
            if any(prefix in text for prefix in ['非常', '特别', '极']):
                stats['intensities']['extreme'] += 1
            elif any(prefix in text for prefix in ['很', '比较']):
                stats['intensities']['strong'] += 1
            elif any(prefix in text for prefix in ['稍微', '一点', '略']):
                stats['intensities']['light'] += 1
            else:
                stats['intensities']['medium'] += 1

        return stats


def main():
    """主函数"""
    # 创建生成器
    generator = Stage2Generator()

    # 生成二阶段语料
    print("开始生成二阶段语料...")
    corpus = generator.generate_stage2_corpus(samples_per_color=30)

    # 分析统计信息
    stats = generator.analyze_corpus(corpus)
    print("\n语料统计:")
    print(f"总样本数: {stats['total_samples']}")
    print(f"维度分布: {stats['dimensions']}")
    print(f"强度分布: {stats['intensities']}")

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
    generator.save_corpus(corpus, generator.data_dir / 'samples/stage2_easy_samples.txt')


if __name__ == "__main__":
    main()
