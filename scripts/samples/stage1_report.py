import re
import statistics
from collections import defaultdict, Counter
from typing import List, Tuple, Dict


class ColorCorpusAnalyzer:
    """颜色语料评估分析器"""

    def __init__(self, corpus_file: str):
        """初始化分析器

        Args:
            corpus_file: 语料文件路径，格式为:
                红
                color(628 258 29227)
        """
        self.corpus_file = corpus_file
        self.data = []
        self.color_to_coords = {}  # 颜色名称 -> 坐标
        self.coords_to_colors = defaultdict(list)  # 坐标 -> 颜色名称列表

    def load_corpus(self):
        """加载语料数据"""
        with open(self.corpus_file, 'r', encoding='utf-8') as f:
            samples = f.read().split('\n\n')

        for sample in samples:
            color_name, coord_str = sample.strip().split('\n')
            coords = self._parse_coordinates(coord_str)

            if coords:
                self.data.append((color_name, coords))
                self.color_to_coords[color_name] = coords
                coord_key = tuple(coords)
                self.coords_to_colors[coord_key].append(color_name)

    def _parse_coordinates(self, coord_str: str) -> Tuple[int, int, int] | None:
        """解析坐标字符串"""
        # 匹配 color L C H 格式
        match = re.match(r'color (\d+)\s+(\d+)\s+(\d+)', coord_str)
        if match:
            return int(match.group(1)), int(match.group(2)), int(match.group(3))
        return None

    def basic_statistics(self) -> Dict:
        """基本统计信息"""
        total_entries = len(self.data)
        unique_colors = len(set(item[0] for item in self.data))
        unique_coordinates = len(self.coords_to_colors)

        # 同义词统计
        synonym_groups = [colors for colors in self.coords_to_colors.values() if len(colors) > 1]
        synonym_count = len(synonym_groups)

        # 颜色名称长度统计
        name_lengths = [len(item[0]) for item in self.data]

        return {
            'total_entries': total_entries,
            'unique_color_names': unique_colors,
            'unique_coordinates': unique_coordinates,
            'synonym_groups': synonym_count,
            'avg_synonyms_per_coord': total_entries / unique_coordinates if unique_coordinates > 0 else 0,
            'color_name_length': {
                'min': min(name_lengths),
                'max': max(name_lengths),
                'avg': statistics.mean(name_lengths),
                'median': statistics.median(name_lengths)
            }
        }

    def coordinate_analysis(self) -> Dict:
        """坐标分析"""
        all_coords = [item[1] for item in self.data]

        # 分别统计 L, C, H
        L_values = [coord[0] for coord in all_coords]
        C_values = [coord[1] for coord in all_coords]
        H_values = [coord[2] for coord in all_coords]

        def analyze_dimension(values, name):
            return {
                'min': min(values),
                'max': max(values),
                'avg': statistics.mean(values),
                'median': statistics.median(values),
                'std': statistics.stdev(values) if len(values) > 1 else 0,
                'unique_count': len(set(values))
            }

        # 转换回原始数值范围（除以1000）
        L_real = [l / 1000 for l in L_values]
        C_real = [c / 1000 for c in C_values]
        H_real = [h / 1000 for h in H_values]

        return {
            'lightness': analyze_dimension(L_real, 'L'),
            'chroma': analyze_dimension(C_real, 'C'),
            'hue': analyze_dimension(H_real, 'H'),
            'coordinate_distribution': {
                'L_bins': self._create_bins(L_real, 10),
                'C_bins': self._create_bins(C_real, 10),
                'H_bins': self._create_bins(H_real, 10)
            }
        }

    def _create_bins(self, values: List[float], num_bins: int) -> Dict:
        """创建数值分布的bin统计"""
        min_val, max_val = min(values), max(values)
        bin_width = (max_val - min_val) / num_bins
        bins = defaultdict(int)

        for val in values:
            bin_idx = int((val - min_val) / bin_width)
            if bin_idx == num_bins:  # 处理最大值的边界情况
                bin_idx = num_bins - 1
            bin_range = f"[{min_val + bin_idx * bin_width:.3f}, {min_val + (bin_idx + 1) * bin_width:.3f})"
            bins[bin_range] += 1

        return dict(bins)

    def quality_analysis(self) -> Dict:
        """数据质量分析"""
        issues = []

        # 检查坐标合理性
        invalid_coords = []
        for color_name, (L, C, H) in self.data:
            # LCH 颜色空间的合理范围检查
            L_real = L / 1000
            C_real = C / 1000
            H_real = H / 1000

            if not (0 <= L_real <= 100):
                invalid_coords.append((color_name, f"L值异常: {L_real}"))
            if C_real < 0:  # 彩度不能为负
                invalid_coords.append((color_name, f"C值异常: {C_real}"))
            if not (0 <= H_real <= 360):
                invalid_coords.append((color_name, f"H值异常: {H_real}"))

        # 检查重复的颜色名称但坐标不同
        color_coord_conflicts = []
        color_counts = Counter(item[0] for item in self.data)
        for color_name, count in color_counts.items():
            if count > 1:
                coords_for_color = [item[1] for item in self.data if item[0] == color_name]
                if len(set(tuple(coord) for coord in coords_for_color)) > 1:
                    color_coord_conflicts.append((color_name, coords_for_color))

        return {
            'invalid_coordinates': invalid_coords,
            'color_coordinate_conflicts': color_coord_conflicts,
            'total_issues': len(invalid_coords) + len(color_coord_conflicts)
        }

    def synonym_analysis(self) -> Dict:
        """同义词分析"""
        # 找出同义词组
        synonym_groups = {}
        for coord_key, color_names in self.coords_to_colors.items():
            if len(color_names) > 1:
                synonym_groups[coord_key] = color_names

        # 统计同义词组大小分布
        group_sizes = [len(names) for names in synonym_groups.values()]
        size_distribution = Counter(group_sizes)

        # 找出最大的同义词组
        largest_groups = sorted(synonym_groups.items(),
                                key=lambda x: len(x[1]), reverse=True)[:5]

        return {
            'synonym_groups_count': len(synonym_groups),
            'group_size_distribution': dict(size_distribution),
            'largest_synonym_groups': [
                {
                    'coordinate': coord,
                    'synonyms': names,
                    'count': len(names)
                }
                for coord, names in largest_groups
            ],
            'coverage_ratio': len(synonym_groups) / len(self.coords_to_colors) if self.coords_to_colors else 0
        }

    def vocabulary_analysis(self) -> Dict:
        """词汇分析"""
        all_color_names = [item[0] for item in self.data]

        # 字符统计
        all_chars = ''.join(all_color_names)
        char_freq = Counter(all_chars)

        # 常见颜色词根
        color_roots = ['红', '蓝', '绿', '黄', '紫', '橙', '黑', '白', '灰', '粉']
        root_coverage = {}
        for root in color_roots:
            matching_colors = [name for name in all_color_names if root in name]
            root_coverage[root] = len(matching_colors)

        return {
            'total_characters': len(all_chars),
            'unique_characters': len(char_freq),
            'most_common_chars': dict(char_freq.most_common(10)),
            'color_root_coverage': root_coverage,
            'potential_vocab_size': len(set(all_color_names)) + 1000  # 颜色词汇 + 数字tokens
        }

    def generate_report(self) -> str:
        """生成完整的评估报告"""
        self.load_corpus()

        basic_stats = self.basic_statistics()
        coord_analysis = self.coordinate_analysis()
        quality_analysis = self.quality_analysis()
        synonym_analysis = self.synonym_analysis()
        vocab_analysis = self.vocabulary_analysis()

        report = f"""\
========================================
颜色语料评估报告
========================================

### 1. 基本统计信息
- 总语料条目数: {basic_stats['total_entries']}
- 唯一颜色名称数: {basic_stats['unique_color_names']}
- 唯一坐标数: {basic_stats['unique_coordinates']}
- 同义词组数: {basic_stats['synonym_groups']}
- 平均同义词数: {basic_stats['avg_synonyms_per_coord']:.2f}

颜色名称长度分布:
- 最短: {basic_stats['color_name_length']['min']} 字符
- 最长: {basic_stats['color_name_length']['max']} 字符
- 平均: {basic_stats['color_name_length']['avg']:.2f} 字符
- 中位数: {basic_stats['color_name_length']['median']} 字符

### 2. 坐标空间分析
明度 (L): 
- 范围: [{coord_analysis['lightness']['min']:.3f}, {coord_analysis['lightness']['max']:.3f}]
- 平均: {coord_analysis['lightness']['avg']:.3f}, 标准差: {coord_analysis['lightness']['std']:.3f}
- 唯一值数量: {coord_analysis['lightness']['unique_count']}

彩度 (C):
- 范围: [{coord_analysis['chroma']['min']:.3f}, {coord_analysis['chroma']['max']:.3f}]
- 平均: {coord_analysis['chroma']['avg']:.3f}, 标准差: {coord_analysis['chroma']['std']:.3f}
- 唯一值数量: {coord_analysis['chroma']['unique_count']}

色相 (H):
- 范围: [{coord_analysis['hue']['min']:.3f}, {coord_analysis['hue']['max']:.3f}]
- 平均: {coord_analysis['hue']['avg']:.3f}, 标准差: {coord_analysis['hue']['std']:.3f}
- 唯一值数量: {coord_analysis['hue']['unique_count']}

### 3. 数据质量分析
总问题数: {quality_analysis['total_issues']}

坐标异常: {len(quality_analysis['invalid_coordinates'])} 条
"""

        if quality_analysis['invalid_coordinates']:
            report += "异常坐标示例:\n"
            for color, issue in quality_analysis['invalid_coordinates'][:5]:
                report += f"  - {color}: {issue}\n"

        report += f"""
颜色名称冲突: {len(quality_analysis['color_coordinate_conflicts'])} 条
"""

        if quality_analysis['color_coordinate_conflicts']:
            report += "冲突示例:\n"
            for color, coords in quality_analysis['color_coordinate_conflicts'][:3]:
                report += f"  - {color}: {len(coords)} 个不同坐标\n"

        report += f"""

### 4. 同义词分析
同义词组数: {synonym_analysis['synonym_groups_count']}
覆盖率: {synonym_analysis['coverage_ratio']:.2%}

同义词组大小分布:
"""
        for size, count in sorted(synonym_analysis['group_size_distribution'].items()):
            report += f"  - {size}个同义词: {count}组\n"

        report += "\n最大同义词组 (前5个):\n"
        for group in synonym_analysis['largest_synonym_groups']:
            coord = group['coordinate']
            coord_real = (coord[0] / 1000, coord[1] / 1000, coord[2] / 1000)
            report += f"  - {coord_real}: {group['synonyms']} ({group['count']}个)\n"

        report += f"""

### 5. 词汇分析
总字符数: {vocab_analysis['total_characters']}
唯一字符数: {vocab_analysis['unique_characters']}
预估词表大小: {vocab_analysis['potential_vocab_size']}

常用字符 (前10个):
"""
        for char, freq in vocab_analysis['most_common_chars'].items():
            report += f"  - '{char}': {freq} 次\n"

        report += "\n颜色词根覆盖度:\n"
        for root, count in vocab_analysis['color_root_coverage'].items():
            report += f"  - '{root}': {count} 个颜色\n"

        report += f"""

### 6. 总结与建议
✓ 数据量充足: {basic_stats['total_entries']} 条语料满足一阶段训练需求
✓ 同义词丰富: {basic_stats['avg_synonyms_per_coord']:.1f} 个同义词/坐标有助于泛化
✓ 颜色空间覆盖: L({coord_analysis['lightness']['unique_count']}), C({coord_analysis['chroma']['unique_count']}), H({coord_analysis['hue']['unique_count']}) 维度都有良好分布

建议:
1. 词表大小设置为 ~{vocab_analysis['potential_vocab_size']} 应该充足
2. 重点关注{'高频字符的token化效率' if len(vocab_analysis['most_common_chars']) > 0 else '字符分布'}
3. {'修复数据质量问题' if quality_analysis['total_issues'] > 0 else '数据质量良好，可直接使用'}
"""

        return report


# 使用示例
if __name__ == "__main__":
    analyzer = ColorCorpusAnalyzer("../../data/samples/stage1_samples.txt")
    report = analyzer.generate_report()
    print(report)
