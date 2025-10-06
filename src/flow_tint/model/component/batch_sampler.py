import random
from typing import cast

from torch.utils.data import Subset

from src.flow_tint.model.component.color_dataset import ColorDataset, TokenizedColorSample


class LengthGroupedBatchSampler:
    """按长度分组的批次采样器，减少padding开销"""

    def __init__(self, dataset: Subset[ColorDataset], batch_size: int, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        # 计算每个样本的长度并分组
        self.length_groups = self._group_by_length()

    def _group_by_length(self) -> list[list[int]]:
        """按序列长度对样本进行分组"""
        # 获取所有样本的长度和索引
        length_index_pairs = [(len(cast(TokenizedColorSample, cast(object, self.dataset[i])).input), i)
                              for i in range(len(self.dataset))]

        # 按长度排序
        length_index_pairs.sort(key=lambda x: x[0])

        # 按长度分组
        groups: list[list[int]] = []
        for i in range(0, len(length_index_pairs), self.batch_size):
            group = [pair[1] for pair in length_index_pairs[i:i + self.batch_size]]
            groups.append(group)

        return groups

    def __iter__(self):
        """生成批次索引"""
        batches = []

        # 为每个长度组创建批次
        for indices in self.length_groups:
            # 随机打乱同组内的样本
            random.shuffle(indices)

            # 创建批次
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)

        # 随机打乱所有批次
        random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        """返回批次总数"""
        total_batches = 0
        for indices in self.length_groups:
            group_batches = len(indices) // self.batch_size
            if len(indices) % self.batch_size != 0 and not self.drop_last:
                group_batches += 1
            total_batches += group_batches
        return total_batches
