#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
四阶段颜色语料生成器（修复版）
基于一阶段颜色数据，生成多轮对话形式的颜色调整语料
支持2-5轮对话，维护完整的操作历史
新增完整的五维度修饰体系和权重分配
修复：在生成时进行统计，而不是事后文本分析
"""

import random
from dataclasses import dataclass
from functools import reduce
from operator import add
from pathlib import Path
from typing import Dict, List, Tuple

from src.flow_tint.core.utils import add_fluctuation
from src.flow_tint.path import get_data_dir


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


@dataclass
class DialogueTurn:
    """单轮对话数据"""
    input_text: str
    output_operations: List[str]
    is_final: bool = False
    is_complex: bool = False  # 新增：标记是否为复合指令
    dimensions_used: List[str] = None  # 新增：记录使用的维度


@dataclass
class DialogueSession:
    """完整对话会话"""
    turns: List[DialogueTurn]
    base_color: str
    base_coords: str


@dataclass
class GenerationStats:
    """生成统计信息"""

    def __init__(self):
        self.total_sessions = 0
        self.total_turns = 0
        self.turns_distribution = {2: 0, 3: 0, 4: 0, 5: 0}
        self.operation_types = {'lightness': 0, 'chroma': 0, 'hue': 0, 'mix': 0, 'temperature': 0}
        self.simple_instructions = 0
        self.complex_instructions = 0
        self.complex_types = {
            'lightness+temperature': 0,
            'lightness+chroma': 0,
            'mix+chroma': 0,
            'mix+lightness': 0,
            'temperature+chroma': 0,
            'mix+temperature': 0
        }
        # 复合指令生成尝试统计
        self.complex_attempts = 0
        self.complex_failures = 0
        # 新增：详细的轮次统计
        self.first_turns = 0  # 第一轮（颜色查询）
        self.adjustment_turns = 0  # 调整轮次
        # 新增：复合指令类型选择统计
        self.complex_type_attempts = {
            'lightness+temperature': 0,
            'lightness+chroma': 0,
            'mix+chroma': 0,
            'mix+lightness': 0,
            'temperature+chroma': 0,
            'mix+temperature': 0
        }

    def add_complex_attempt(self, complex_type: str = None):
        """记录复合指令尝试"""
        self.complex_attempts += 1
        if complex_type and complex_type in self.complex_type_attempts:
            self.complex_type_attempts[complex_type] += 1

    def add_complex_failure(self):
        """记录复合指令失败"""
        self.complex_failures += 1

    def add_session(self, session: DialogueSession):
        """添加会话统计"""
        self.total_sessions += 1
        turn_count = len(session.turns)
        self.total_turns += turn_count

        if turn_count in self.turns_distribution:
            self.turns_distribution[turn_count] += 1

    def get_complex_key(self, dimensions_used: List[str]) -> str | None:
        """根据使用的维度生成复合指令类型key - 智能匹配"""
        if len(dimensions_used) < 2:
            return None

        # 转换为set进行匹配，避免顺序问题
        dims_set = set(dimensions_used)

        # 预定义的维度组合映射（语义逻辑顺序）
        complex_mappings = {
            frozenset(['lightness', 'temperature']): 'lightness+temperature',
            frozenset(['lightness', 'chroma']): 'lightness+chroma',
            frozenset(['mix', 'chroma']): 'mix+chroma',
            frozenset(['mix', 'lightness']): 'mix+lightness',
            frozenset(['temperature', 'chroma']): 'temperature+chroma',
            frozenset(['mix', 'temperature']): 'mix+temperature'
        }

        return complex_mappings.get(frozenset(dims_set))

    def add_turn(self, turn: DialogueTurn):
        """添加轮次统计"""
        if turn.dimensions_used:
            for dim in turn.dimensions_used:
                if dim in self.operation_types:
                    self.operation_types[dim] += 1

        # 统计轮次类型
        if turn.output_operations:  # 有输出操作的是调整轮次
            self.adjustment_turns += 1
            if turn.is_complex:
                self.complex_instructions += 1
                # 统计具体的复合类型 - 修复版
                if turn.dimensions_used and len(turn.dimensions_used) >= 2:
                    complex_key = self.get_complex_key(turn.dimensions_used)
                    if complex_key and complex_key in self.complex_types:
                        self.complex_types[complex_key] += 1
            else:
                self.simple_instructions += 1
        else:  # 没有输出操作的是第一轮（颜色查询）
            self.first_turns += 1

    def get_summary(self) -> Dict:
        """获取统计摘要"""
        total_instructions = self.simple_instructions + self.complex_instructions
        complex_ratio = (self.complex_instructions / total_instructions * 100) if total_instructions > 0 else 0
        success_rate = ((
                                self.complex_attempts - self.complex_failures) / self.complex_attempts * 100) if self.complex_attempts > 0 else 0
        expected_complex = self.adjustment_turns * 0.1  # 期望的复合指令数量

        return {
            'total_sessions': self.total_sessions,
            'average_turns': self.total_turns / self.total_sessions if self.total_sessions > 0 else 0,
            'turns_distribution': self.turns_distribution,
            'operation_types': self.operation_types,
            'instruction_complexity': {
                'simple': self.simple_instructions,
                'complex': self.complex_instructions,
                'complex_ratio': f"{complex_ratio:.1f}%"
            },
            'complex_types': self.complex_types,
            'complex_generation': {
                'attempts': self.complex_attempts,
                'failures': self.complex_failures,
                'success_rate': f"{success_rate:.1f}%"
            },
            'turn_details': {
                'first_turns': self.first_turns,
                'adjustment_turns': self.adjustment_turns,
                'total_instructions': total_instructions,
                'expected_complex': f"{expected_complex:.0f}",
                'actual_complex': self.complex_instructions
            },
            'complex_type_attempts': self.complex_type_attempts
        }


class Stage4Generator:
    base_colors: dict[str, str]

    def __init__(self):
        self.data_dir = get_data_dir()

        # 基础颜色列表
        self.base_colors = {}
        self.parse_stage1_data()

        # 完整的五维度修饰词体系（与二三阶段统一）
        self.modifiers = {
            'lightness': {
                'darken': ['深', '暗', '浓'],
                'brighten': ['浅', '淡', '亮', '明']
            },
            'chroma': {
                'enhance': ['鲜', '艳', '纯', '正'],
                'reduce': ['灰', '浊', '素']
            },
            'hue': {  # 新增：色相角度调整
                'hue_shift_positive': ['色相偏移', 'H值增加', '色相旋转'],
                'hue_shift_negative': ['色相回调', 'H值减少', '色相反转']
            },
            'mix': self.generate_mix_modifiers(),
            'temperature': {  # 新增：色温调整
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

        # 对话强度设计：后续轮次使用较小调整
        self.dialogue_intensity = {
            'lightness': {
                'light': [0.01, 0.02, 0.03, 0.04, 0.05],
                'medium': [0.06, 0.07, 0.08, 0.09, 0.10],
                'strong': [0.13, 0.14, 0.15, 0.16, 0.17]
            },
            'chroma': {
                'light': [0.01, 0.02, 0.03, 0.04, 0.05],
                'medium': [0.06, 0.07, 0.08, 0.09, 0.10],
                'strong': [0.13, 0.14, 0.15, 0.16, 0.17]
            },
            'hue': {
                'light': [8.0, 9.0, 10.0, 11.0, 12.0],
                'medium': [18.0, 19.0, 20.0, 21.0, 22.0],
                'strong': [28.0, 29.0, 30.0, 31.0, 32.0]
            },
            'mix': {
                'light': [0.03, 0.04, 0.05, 0.06, 0.07],
                'medium': [0.08, 0.09, 0.10, 0.11, 0.12]
            },
            'temperature': {
                'light': [0.03, 0.04, 0.05, 0.06, 0.07],
                'medium': [0.08, 0.09, 0.10, 0.11, 0.12]
            }
        }

        # 色温参考颜色（与二三阶段统一）
        self.temperature_colors = {
            'warmer': 'color 950 20 85000',  # 暖白
            'cooler': 'color 950 20 240000'  # 冷白
        }

        # 对话模式模板（扩展版 + 复合指令）
        self.dialogue_patterns = {
            'progressive': {
                'description': '渐进调整',
                'flow': ['lightness', 'chroma', 'mix'],
                'phrases': {
                    'lightness': ['深一点', '浅一点', '亮一些', '暗一些'],
                    'chroma': ['艳一点', '素一些', '鲜艳一些', '灰一点'],
                    'mix': ['偏{color}一点', '带点{color}', '泛{color}一些'],
                    'temperature': ['暖一些', '冷一点', '温暖一些'],
                    'hue': ['色相调一下', '转一点角度']
                }
            },
            'correction': {
                'description': '修正调整',
                'flow': ['mix', 'lightness'],
                'phrases': {
                    'mix': ['不，偏{color}', '改成{color}一点', '要{color}一些'],
                    'lightness': ['再深一点', '再浅一些', '明度调一下'],
                    'temperature': ['不对，要暖调', '偏冷一些'],
                    'chroma': ['饱和度调一下', '纯度高一些']
                }
            },
            'refinement': {
                'description': '精细调节',
                'flow': ['chroma', 'mix', 'lightness'],
                'phrases': {
                    'chroma': ['更纯一点', '降低饱和度', '色彩浓一些'],
                    'mix': ['加点{color}', '混入{color}'],
                    'lightness': ['微调明度', '稍微深一点'],
                    'temperature': ['调暖一些', '色温冷一点'],
                    'hue': ['色相偏移一下']
                }
            }
        }

        # 复合指令模板（10%概率使用）- 修复版：消除语义歧义
        self.complex_phrases = {
            'lightness+temperature': [
                '暖调浅一点', '冷调深一些', '暖色亮一些', '冷调暗一点'
            ],
            'lightness+chroma': [
                '深一些但鲜艳', '亮一些且纯正', '浅色但艳丽', '暗一些却鲜明'
            ],
            'mix+chroma': [
                '偏{color}且鲜艳', '带{color}但要纯', '泛{color}色且艳丽'
            ],
            'mix+lightness': [
                '偏{color}但要亮', '带{color}且深沉', '泛{color}但浅淡'
            ],
            'temperature+chroma': [
                '暖调且鲜艳', '冷色但要纯', '温暖又艳丽'
            ],
            'mix+temperature': [
                '偏{color}且暖调', '带{color}但冷色', '泛{color}又温暖'
            ]
        }

        # 简化颜色名称到mix操作的映射
        self.color_to_mix = self.generate_color_to_mix()

        # 新增：统计对象
        self.stats = GenerationStats()

    def generate_mix_modifiers(self) -> dict[str, list[str]]:
        """从基础颜色动态生成mix修饰词"""
        mix_dict = {}

        for color in self.base_colors.keys():
            key = f'{color}_mix'
            mix_dict[key] = [f'偏{color}', f'泛{color}', f'带{color}', ]

        return mix_dict

    def generate_color_to_mix(self) -> dict[str, str]:
        mix_dict = {}

        for color in self.base_colors.keys():
            mix_dict[color] = f"{color}_mix"

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

    def select_weighted_dimension(self, excluded_dims: List[str] = None) -> str:
        """按权重选择维度"""
        if excluded_dims is None:
            excluded_dims = []

        available_dims = [dim for dim in self.dimension_weights.keys() if dim not in excluded_dims]
        if not available_dims:
            return random.choice(list(self.dimension_weights.keys()))

        weights = [self.dimension_weights[dim] for dim in available_dims]
        return random.choices(available_dims, weights=weights, k=1)[0]

    def analyze_semantic_intent(self, text: str, dimension: str) -> Tuple[bool, bool]:
        """分析语义意图：返回(是否匹配该维度, 是否为正向操作) - 调试版"""

        # 检测否定词（注意：不包含转折词）
        negation_words = ['不要', '不', '别', '减少', '降低']
        has_negation = any(neg in text for neg in negation_words)

        if dimension == 'lightness':
            darken_words = ['深', '暗', '浓']
            brighten_words = ['浅', '淡', '亮', '明']

            has_darken = any(word in text for word in darken_words)
            has_brighten = any(word in text for word in brighten_words)

            if has_darken:
                # 检测到变深的词：正常为负向操作（变深），有否定时为正向（变浅）
                return True, has_negation
            elif has_brighten:
                # 检测到变亮的词：正常为正向操作（变亮），有否定时为负向（变深）
                return True, not has_negation

        elif dimension == 'chroma':
            enhance_words = ['艳', '鲜', '纯', '正', '浓', '明']  # 增加'明'
            reduce_words = ['灰', '浊', '素', '淡']

            has_enhance = any(word in text for word in enhance_words)
            has_reduce = any(word in text for word in reduce_words)

            if has_enhance:
                # 检测到增强词：正常为正向操作（增加饱和度），有否定时为负向
                return True, not has_negation
            elif has_reduce:
                # 检测到减少词：正常为负向操作（减少饱和度），有否定时为正向
                return True, has_negation

        elif dimension == 'temperature':
            warm_words = ['暖', '温暖', '热情']
            cool_words = ['冷', '清冷', '沉静']

            has_warm = any(word in text for word in warm_words)
            has_cool = any(word in text for word in cool_words)

            if has_warm:
                # 检测到变暖词：正常为正向操作（变暖），有否定时为负向（变冷）
                return True, not has_negation
            elif has_cool:
                # 检测到变冷词：正常为负向操作（变冷），有否定时为正向（变暖）
                return True, has_negation

        elif dimension == 'mix':
            mix_words = ['偏', '泛', '带', '混入', '加点']
            has_mix = any(word in text for word in mix_words)
            # mix操作：检测到混合词就匹配，方向始终为正向（添加颜色）
            if has_mix:
                return True, True  # 简化：mix总是正向的

        return False, False

    def generate_complex_operation(self, complex_type: str, base_color_name: str) -> Tuple[str, List[str], List[str]]:
        """生成复合指令操作 - 调试版：找出异常原因"""
        dimensions = complex_type.split('+')
        operations = []
        target_color = None
        dimensions_used = []

        # 获取复合模板
        templates = self.complex_phrases[complex_type]

        # 检查是否需要颜色替换（检查所有模板，不只是第一个）
        needs_color = any('{color}' in template for template in templates)

        if needs_color:
            # 选择可用的颜色（避免语义冲突）
            available_colors = []
            for color in self.base_colors.keys():
                if color in self.color_to_mix:
                    mix_op = self.color_to_mix[color]
                    if not check_semantic_conflict(base_color_name, 'mix', mix_op):
                        available_colors.append(color)

            if available_colors:
                target_color = random.choice(available_colors)
                # 从包含{color}的模板中选择
                color_templates = [t for t in templates if '{color}' in t]
                template = random.choice(color_templates)
                input_text = template.format(color=target_color)
            else:
                # 调试：记录颜色选择失败
                raise Exception(f"复合指令生成失败：无可用颜色 - {complex_type}, base_color={base_color_name}")
        else:
            input_text = random.choice(templates)

        # 为每个维度生成操作 - 使用改进的语义分析
        for dimension in dimensions:
            matches, is_positive = self.analyze_semantic_intent(input_text, dimension)

            if matches:
                dimensions_used.append(dimension)

                if dimension == 'lightness':
                    factor = add_fluctuation(random.choice(self.dialogue_intensity[dimension]['light']))
                    if is_positive:  # 变亮
                        operations.append(f"lightness $p {factor}")
                    else:  # 变暗
                        operations.append(f"lightness $p -{factor}")

                elif dimension == 'chroma':
                    factor = add_fluctuation(random.choice(self.dialogue_intensity[dimension]['light']))
                    if is_positive:  # 增加饱和度
                        operations.append(f"chroma $p {factor}")
                    else:  # 降低饱和度
                        operations.append(f"chroma $p -{factor}")

                elif dimension == 'mix':
                    # 修复：移除过严的 is_positive 条件，mix操作通常都是添加颜色
                    if target_color and target_color in self.color_to_mix:
                        mix_op = self.color_to_mix[target_color]
                        mix_coord = self.base_colors[mix_op.removesuffix("_mix")]
                        factor = add_fluctuation(random.choice(self.dialogue_intensity[dimension]['light']))
                        operations.extend([mix_coord, f"mix $p $p {factor}"])
                    else:
                        # 调试：记录mix操作生成失败的原因
                        raise Exception(
                            f"mix操作失败：target_color={target_color}, in_map={target_color in self.color_to_mix if target_color else False}")

                elif dimension == 'temperature':
                    factor = add_fluctuation(random.choice(self.dialogue_intensity[dimension]['light']))
                    if is_positive:  # 变暖
                        temp_color = self.temperature_colors['warmer']
                        operations.extend([temp_color, f"temperature $p $p {factor}"])
                    else:  # 变冷
                        temp_color = self.temperature_colors['cooler']
                        operations.extend([temp_color, f"temperature $p $p {factor}"])
            else:
                # 调试：记录维度匹配失败
                raise Exception(f"维度匹配失败：{dimension} 在 '{input_text}' 中无匹配")

        # 如果没有生成任何操作或维度不完整，抛出异常
        if not operations:
            raise Exception(f"复合指令生成失败：无任何操作 - {input_text}")

        if len(dimensions_used) < len(dimensions):
            missing = set(dimensions) - set(dimensions_used)
            raise Exception(f"复合指令不完整：缺少维度 {missing} - {input_text}")

        return input_text, operations, dimensions_used

    def generate_dialogue_turn(self, turn_index: int, pattern: str,
                               base_color_name: str, operation_history: List[str],
                               used_dimensions: List[str] = None) -> DialogueTurn | None:
        """生成单轮对话 - 改进版：支持10%复合指令并记录统计"""

        if used_dimensions is None:
            used_dimensions = []

        if turn_index == 0:
            # 第一轮：颜色名称查询
            turn = DialogueTurn(
                input_text=f'{base_color_name}',
                output_operations=[],
                is_final=False,
                is_complex=False,
                dimensions_used=[]
            )
            self.stats.add_turn(turn)
            return turn

        # 10%概率生成复合指令，90%生成简单指令
        is_complex = random.random() < 0.1

        if is_complex:
            # 生成复合指令
            complex_types = list(self.complex_phrases.keys())
            complex_type = random.choice(complex_types)
            self.stats.add_complex_attempt(complex_type)  # 记录尝试及类型

            try:
                input_text, operations, dimensions_used_this_turn = self.generate_complex_operation(complex_type,
                                                                                                    base_color_name)
                turn = DialogueTurn(
                    input_text=input_text,
                    output_operations=operations,
                    is_final=False,
                    is_complex=True,
                    dimensions_used=dimensions_used_this_turn
                )
                self.stats.add_turn(turn)
                return turn
            except Exception as e:
                # 复合指令生成失败，回退到简单指令
                self.stats.add_complex_failure()  # 记录失败
                is_complex = False

        if not is_complex:
            # 生成简单指令（原有逻辑）
            pattern_config = self.dialogue_patterns[pattern]
            flow = pattern_config['flow']
            phrases = pattern_config['phrases']

            # 选择这一轮要做的调整类型（优先使用权重，其次使用流程）
            if turn_index - 1 < len(flow):
                dimension = flow[turn_index - 1]
                # 如果该维度已使用过，按权重重新选择
                if dimension in used_dimensions:
                    dimension = self.select_weighted_dimension(used_dimensions)
            else:
                # 超出预定义流程，按权重选择未使用的维度
                dimension = self.select_weighted_dimension(used_dimensions)

            # 生成对应的输入文本和操作
            dimensions_used_this_turn = [dimension]

            if dimension == 'mix':
                # 颜色混合：从实际可用颜色中选择
                available_colors = []
                for color in self.base_colors:
                    if color in self.color_to_mix:
                        mix_op = self.color_to_mix[color]
                        if not check_semantic_conflict(base_color_name, 'mix', mix_op):
                            available_colors.append(color)

                if available_colors:
                    target_color = random.choice(available_colors)
                    template = random.choice(phrases[dimension])
                    input_text = template.format(color=target_color)

                    # 生成对应的操作
                    intensity = random.choice(['light', 'medium'])
                    factor = add_fluctuation(random.choice(self.dialogue_intensity[dimension][intensity]))
                    mix_op = self.color_to_mix[target_color]
                    mix_coord = self.base_colors[mix_op]
                    operations = [mix_coord, f"mix $p $p {factor}"]
                else:
                    # 回退到lightness
                    dimension = 'lightness'
                    dimensions_used_this_turn = [dimension]
                    input_text = random.choice(phrases.get('lightness', ['深一点']))
                    operations = self.generate_simple_operation(dimension, input_text)

            elif dimension == 'temperature':
                # 色温调整
                input_text = random.choice(phrases[dimension])
                intensity = 'light' if any(word in input_text for word in ['一点', '一些', '稍微']) else 'medium'
                factor = add_fluctuation(random.choice(self.dialogue_intensity[dimension][intensity]))

                if any(word in input_text for word in ['暖', '温暖']):
                    temp_color = self.temperature_colors['warmer']
                else:
                    temp_color = self.temperature_colors['cooler']

                operations = [temp_color, f"temperature $p $p {factor}"]

            elif dimension == 'hue':
                # 色相调整：使用半连接形式
                intensity = random.choice(['light', 'medium'])
                factor = add_fluctuation(random.choice(self.dialogue_intensity[dimension][intensity]))

                if random.random() < 0.5:
                    input_text = f"色相旋转+{factor}度"
                    operations = [f"hue $p {factor}"]
                else:
                    input_text = f"色相旋转-{factor}度"
                    operations = [f"hue $p -{factor}"]

            else:
                # lightness或chroma调整
                input_text = random.choice(phrases.get(dimension, ['调整一下']))
                operations = self.generate_simple_operation(dimension, input_text)

            turn = DialogueTurn(
                input_text=input_text,
                output_operations=operations,
                is_final=False,
                is_complex=False,
                dimensions_used=dimensions_used_this_turn
            )
            self.stats.add_turn(turn)
            return turn

    def generate_simple_operation(self, dimension: str, input_text: str) -> List[str]:
        """生成简单的单一维度调整操作 - 扩展版"""
        intensity = 'light' if any(word in input_text for word in ['一点', '一些', '稍微']) else 'medium'
        factor = add_fluctuation(random.choice(self.dialogue_intensity[dimension][intensity]))

        if dimension == 'lightness':
            if any(word in input_text for word in ['深', '暗']):
                return [f"lightness $p -{factor}"]
            else:
                return [f"lightness $p {factor}"]
        elif dimension == 'chroma':
            if any(word in input_text for word in ['艳', '鲜', '纯']):
                return [f"chroma $p {factor}"]
            else:
                return [f"chroma $p -{factor}"]

        return []

    def generate_dialogue_session(self, base_color_name: str, base_coords: str,
                                  turns_count: int = None) -> DialogueSession:
        """生成完整的对话会话 - 改进版：跟踪已使用维度"""

        if turns_count is None:
            turns_count = random.choice([2, 3, 4, 5])

        # 随机选择对话模式
        pattern = random.choice(list(self.dialogue_patterns.keys()))

        turns = []
        operation_history = []
        used_dimensions = []

        for turn_idx in range(turns_count):
            turn = self.generate_dialogue_turn(turn_idx, pattern, base_color_name, operation_history, used_dimensions)

            if turn_idx == turns_count - 1:
                turn.is_final = True

            turns.append(turn)
            operation_history.extend(turn.output_operations)

            # 跟踪已使用的维度
            if turn.dimensions_used:
                used_dimensions.extend(turn.dimensions_used)

        session = DialogueSession(
            turns=turns,
            base_color=base_color_name,
            base_coords=base_coords
        )

        # 添加会话统计
        self.stats.add_session(session)
        return session

    def format_dialogue_session(self, session: DialogueSession) -> str:
        """格式化DialogueSession为语料格式 - 完整历史上下文版"""
        dialogue = []
        # 基本思路
        # 一组对话有N轮，遍历N次处理每轮，每轮遍历N次处理历史轮
        #     遍历第N轮时，如果当前是最后一个turn，添加‘###’作为输入输出分隔
        #     轮遍历结束时，添加‘---’
        for idx, turn in enumerate(session.turns):
            for i in range(idx + 1):
                turn_i = session.turns[i]
                # 当前turn的输入文本
                dialogue.append(turn_i.input_text)

                # 如果当前是最后一个turn
                if i == idx:
                    dialogue.append('###')

                # 当前turn的输出
                if i == 0:
                    dialogue.append(session.base_coords)
                else:
                    operations = reduce(
                        add,
                        map(lambda t: t.output_operations, session.turns[:i + 1]),
                        [session.base_coords],
                    )

                    for operation in operations:
                        dialogue.append(operation)

            # 如果当前不是最后一轮对话
            if idx < len(session.turns) - 1:
                dialogue.append('---')

        return '\n'.join(dialogue)

    def generate_stage4_corpus(self, sessions_per_color: int = 5) -> str:
        """生成完整的四阶段语料"""

        # 重置统计
        self.stats = GenerationStats()

        # 解析一阶段数据并构建颜色词库
        self.parse_stage1_data()
        print(f"解析到 {len(self.base_colors)} 个颜色")
        print(f"可用基础颜色: {self.base_colors.keys()}")

        all_sessions = []
        successful_colors = 0

        for color_name, coords in self.base_colors.items():
            color_sessions = []

            # 为每个颜色生成多个对话会话
            for _ in range(sessions_per_color):
                try:
                    # 随机选择对话轮数（权重偏向中等长度）
                    turns_count = random.choices(
                        [2, 3, 4, 5],
                        weights=[0.3, 0.4, 0.2, 0.1]
                    )[0]

                    session = self.generate_dialogue_session(color_name, coords, turns_count)
                    color_sessions.append(session)

                except Exception as e:
                    # 跳过生成失败的会话
                    continue

            if color_sessions:
                all_sessions.extend(color_sessions)
                successful_colors += 1

                # 显示进度
                if successful_colors % 100 == 0:
                    print(f"已处理 {successful_colors} 个颜色，生成会话 {len(all_sessions)} 个")

        print(f"成功处理 {successful_colors}/{len(self.base_colors)} 个颜色")
        print(f"总共生成 {len(all_sessions)} 个对话会话")

        # 随机打乱
        random.shuffle(all_sessions)

        # 格式化输出
        corpus_parts = []
        for i, session in enumerate(all_sessions):
            session_text = self.format_dialogue_session(session)
            corpus_parts.append(session_text)

            # 在会话之间添加空行分隔（除了最后一个）
            if i < len(all_sessions) - 1:
                corpus_parts.append('')

        return '\n'.join(corpus_parts)

    def get_stats(self) -> Dict:
        """获取生成统计信息"""
        return self.stats.get_summary()

    def save_corpus(self, corpus: str, filename: Path):
        """保存语料到文件"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(corpus)
        print(f"语料已保存到 {filename}")


def main():
    """主函数"""
    # 创建生成器
    generator = Stage4Generator()

    # 生成四阶段语料
    print("开始生成四阶段语料...")
    corpus = generator.generate_stage4_corpus(sessions_per_color=16)

    # 获取统计信息（现在是实时统计，准确可靠）
    stats = generator.get_stats()
    print("\n语料统计:")
    print(f"总会话数: {stats['total_sessions']}")
    print(f"平均轮数: {stats['average_turns']:.1f}")
    print(f"轮数分布: {stats['turns_distribution']}")
    print(f"操作类型: {stats['operation_types']}")
    print(f"指令复杂度: {stats['instruction_complexity']}")
    print(f"复合指令生成: {stats['complex_generation']}")
    print(f"轮次详情: {stats['turn_details']}")
    print(f"复合指令类型尝试: {stats['complex_type_attempts']}")
    if any(count > 0 for count in stats['complex_types'].values()):
        print(f"复合指令类型成功: {stats['complex_types']}")

    # 显示前几个完整会话
    print("\n前3个完整会话:")
    sessions = [s.strip() for s in corpus.split('\n\n') if s.strip()]
    for i, session in enumerate(sessions[:3]):
        print(f"\n=== 会话 {i + 1} ===")
        print(session.strip())
        if i < 2:
            print("\n" + "=" * 50)

    # 保存语料
    generator.save_corpus(corpus, generator.data_dir / 'samples/stage3_samples.txt')


if __name__ == "__main__":
    main()
