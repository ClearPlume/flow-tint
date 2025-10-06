import os
from dataclasses import dataclass
from typing import Self

from torch import Tensor, cat, zeros, ones, stack, Generator
from torch import tensor
from torch.nn import functional as F
from torch.utils.data import Dataset, random_split, Subset

from src.flow_tint.core.constants import SEP_TOKEN, PAD_TOKEN_ID
from src.flow_tint.model.component.tokenizer import Tokenizer


@dataclass
class ColorSample:
    """
    单个从语料直接提取的原始样本

    Attributes:
        input: 输入文本
        output: 输入文本
    """
    input: str
    output: str


@dataclass
class TokenizedColorSample:
    """
    单个样本，已tokenized

    Attributes:
        input: 编码后的输入tokens
        target: 编码后的目标tokens
    """
    input: Tensor
    target: Tensor


def _load_sample_file(file_path: str, stage_idx: int) -> list[ColorSample]:
    """
    加载单个文件，直接返回ColorSample列表
    ```
    ins风珊瑚红
    color 820 54 31263

    ins珊瑚红
    color 820 54 31263

    ins风青
    color 822 29 193998
    ```
    """
    samples = []

    with open(file_path, "r", encoding="utf-8") as f:
        for sample in map(str.strip, f.read().split("\n\n")):
            if sample:  # 跳过空样本
                samples.extend(_parse_stage(sample, stage_idx))

    return samples


def _parse_stage(sample: str, stage: int) -> list[ColorSample]:
    if stage == 2:
        samples = []
        dialogue_list = sample.split("\n---\n")

        for dialogue in dialogue_list:
            start, end = dialogue.split("\n###\n")
            samples.append(ColorSample(start.replace("\n", SEP_TOKEN), end.replace("\n", SEP_TOKEN)))

        return samples
    else:
        sample = sample.split("\n")
        return [ColorSample(sample[0], SEP_TOKEN.join(sample[1:]))]


def collate_fn(batch: list[TokenizedColorSample]) -> dict[str, Tensor]:
    """
    对批次内的token序列进行padding对齐，并生成attention_mask

    :param batch: 一个批次中，元素为单个token的列表
    :return 经过了padding，设置了mask，重组为张量的训练语料
    """

    # 分离源序列与目标序列
    src = [batch.input for batch in batch]
    tgt = [batch.target for batch in batch]

    # Padding源序列
    src_max_len = max(len(s) for s in src)
    padded_src = []
    src_padding_mask = []

    for s in src:
        padding_len = src_max_len - len(s)
        padded = F.pad(s, (0, padding_len), value=PAD_TOKEN_ID)
        mask = cat([zeros(len(s)), ones(padding_len)]).bool()

        padded_src.append(padded)
        src_padding_mask.append(mask)

    # Padding目标序列，需分为输入和输出两部分
    tgt_max_len = max(len(t) for t in tgt) - 1  # 去除BOS或者EOS之后的最大长度
    padded_tgt_in = []
    padded_tgt_out = []
    tgt_padding_mask = []

    for t in tgt:
        # 去掉最后一个token作为解码器输入 [BOS] + tokens[:-1]
        tgt_in = t[:-1]
        # 去掉第一个token作为目标输出 tokens + [EOS]
        tgt_out = t[1:]

        # Padding
        in_padding_len = tgt_max_len - len(tgt_in)
        out_padding_len = tgt_max_len - len(tgt_out)

        padded_in = F.pad(tgt_in, (0, in_padding_len), value=PAD_TOKEN_ID)
        padded_out = F.pad(tgt_out, (0, out_padding_len), value=PAD_TOKEN_ID)

        # 目标序列mask
        mask = cat([zeros(len(tgt_out)), ones(out_padding_len)]).bool()

        padded_tgt_in.append(padded_in)
        padded_tgt_out.append(padded_out)
        tgt_padding_mask.append(mask)

    return {
        "src": stack(padded_src),
        "tgt_in": stack(padded_tgt_in),
        "tgt_out": stack(padded_tgt_out),
        "src_padding_mask": stack(src_padding_mask),
        "tgt_padding_mask": stack(tgt_padding_mask),
    }


class ColorDataset(Dataset):
    """颜色语义数据集"""
    samples: list[ColorSample]
    tokenizer: Tokenizer

    def __init__(self, data_files: dict[int, str], tokenizer: Tokenizer):
        """加载所有语料"""

        self.samples = []
        # self.samples = [ColorSample("红", "color 628 258 29227")]
        self.tokenizer = tokenizer
        self._prepare_samples(data_files)

    def _prepare_samples(self, data_files: dict[int, str]):
        """准备训练样本"""
        for stage, file in data_files.items():
            if os.path.exists(file):
                self.samples.extend(_load_sample_file(file, stage))
            else:
                print(f"Warning: File not found: {file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> TokenizedColorSample:
        """
        dataloader中简单分批后，会通过索引获取语料
        我选择在这里进行语料的token化和初步张量
        """

        sample: ColorSample = self.samples[idx]

        return TokenizedColorSample(
            tensor(self.tokenizer.encode(sample.input)),
            tensor(
                [self.tokenizer.bos_token_id] + self.tokenizer.encode(sample.output) + [self.tokenizer.eos_token_id]
            ),
        )

    def split(self, val_ratio: float = 0.1) -> list[Subset[Self]]:
        total_size = len(self)
        val_size = int(total_size * val_ratio)
        train_size = total_size - val_size

        return random_split(self, [train_size, val_size], generator=Generator().manual_seed(42))
