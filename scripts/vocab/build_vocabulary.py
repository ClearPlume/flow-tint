import os

import sentencepiece as spm

from src.flow_tint.core.constants import PAD_TOKEN_ID, PAD_TOKEN, VOCAB_SIZE, SEP_TOKEN, BOS_TOKEN, EOS_TOKEN
from src.flow_tint.path import get_data_dir

data_dir = get_data_dir()
(data_dir / "vocab").mkdir(parents=True, exist_ok=True)

# 四个阶段的样本文件
stage_files = [
    "stage1_samples.txt",
    "stage2_samples.txt",
    "stage3_samples.txt",
]

# 默认保护token列表
protected_tokens = [
    # 多字符函数名
    "color", "lightness", "chroma", "hue", "mix", "temperature",

    # 特殊符号
    "$p", "ins", SEP_TOKEN, BOS_TOKEN, EOS_TOKEN,
]


def train():
    sample_dir = f"{data_dir}/samples"
    sample_file = f"{sample_dir}/samples.txt"

    if os.path.exists(sample_file):
        os.remove(sample_file)

    for stage_file in stage_files:
        with open(f"{sample_dir}/{stage_file}", "r", encoding="utf-8") as stage_sample:
            with open(sample_file, "a", encoding="utf-8") as f:
                f.write(stage_sample.read())
                f.write("\n")

    spm.SentencePieceTrainer.Train(
        input=sample_file,
        model_prefix=f"{data_dir}/vocab/color_semantic",
        model_type="unigram",
        vocab_size=VOCAB_SIZE,
        # 我选择单字分词，是因为unigram并非基于语义的分词，实践之后，发现在颜色领域，概率并非最佳选择——会把语义错误拆分，不如直接单字，让模型自己学习语义
        max_sentencepiece_length=1,
        pad_id=PAD_TOKEN_ID,
        pad_piece=PAD_TOKEN,
        shuffle_input_sentence=True,
        character_coverage=0.9999,
        hard_vocab_limit=False,
        user_defined_symbols=protected_tokens,
        normalization_rule_name="identity",
        remove_extra_whitespaces=False,
        split_digits=True,
        add_dummy_prefix=False,
    )


def verify():
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(f"{data_dir}/vocab/color_semantic.model")

    test_cases = [
        "红",
        "深红",
        "亮一些的蓝",
        "亮一些泛蓝的浅红",
        "赤金色<sep>color 842 173 84578<sep>赤金色<sep>color 842 173 84578<sep>浅一点",
        "克莱因蓝",
        "偏钴蓝丝绸黄",
    ]

    for i, test_case in enumerate(test_cases):
        encoded = tokenizer.Encode(test_case)
        print(test_case)
        print(encoded)
        token_num = len(encoded)

        encoded_str = "["
        for j, encoded in enumerate(encoded):
            encoded_str += "\""
            encoded_str += tokenizer.Decode(encoded)
            encoded_str += "\""

            if j != token_num - 1:
                encoded_str += ", "

        encoded_str += "]"

        print(encoded_str)

        if i < len(test_cases) - 1:
            print("=" * 50)


if __name__ == "__main__":
    train()
    verify()
