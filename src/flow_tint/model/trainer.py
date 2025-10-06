import logging
from datetime import datetime
from pathlib import Path

import torch
from torch import device as device_type
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.flow_tint.core.constants import VOCAB_SIZE
from src.flow_tint.model.component.batch_sampler import LengthGroupedBatchSampler
from src.flow_tint.model.component.color_dataset import ColorDataset, collate_fn
from src.flow_tint.model.component.color_model import ColorModel
from src.flow_tint.model.component.dropout_scheduler import DropoutScheduler
from src.flow_tint.model.component.model_config import ModelConfig
from src.flow_tint.model.component.tokenizer import Tokenizer
from src.flow_tint.model.component.trainer import Trainer
from src.flow_tint.path import get_data_dir, get_logs_dir


def setup_logger(log_file: Path = None) -> logging.Logger:
    """设置日志记录器"""
    logger = logging.getLogger('color_semantic_training')
    logger.setLevel(logging.INFO)

    # 避免重复添加handler
    if logger.handlers:
        return logger

    # 创建格式器
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件处理器
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def main():
    # 设置日志
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = get_logs_dir() / f"training_{timestamp}.log"
    logger = setup_logger(log_file)

    logger.info("=" * 60)
    logger.info("色彩语义模型训练")
    logger.info("=" * 60)

    # 初始化
    data_dir = get_data_dir()
    tokenizer = Tokenizer(f"{data_dir}/vocab/color_semantic.model")
    logger.info(f"数据目录: {data_dir}")
    logger.info(f"词汇表大小: {VOCAB_SIZE}")

    # 设定语料文件
    data_files = {
        0: f"{data_dir}/samples/stage1_samples.txt",
        1: f"{data_dir}/samples/stage2_samples.txt",
        2: f"{data_dir}/samples/stage3_samples.txt",
    }
    logger.info(f"训练数据文件: {list(data_files.values())}")

    # 基于语料创建数据集，这里会对原始语料进行token化
    dataset = ColorDataset(data_files, tokenizer)
    logger.info(f"数据集大小: {len(dataset)} 样本")

    # 数据分割，使用90%的数据集训练，10%的数据集用于验证训练结果
    train_dataset, val_dataset = dataset.split()

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=LengthGroupedBatchSampler(
            train_dataset,
            batch_size=32,
            drop_last=True,
        ),
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_sampler=LengthGroupedBatchSampler(
            val_dataset,
            batch_size=32,
            drop_last=True,
        ),
        collate_fn=collate_fn,
        num_workers=4,
    )

    # 模型初始化
    model_config = ModelConfig.small(learning_rate=5e-5, dropout=0.2)

    model = ColorModel(model_config)
    param_count = sum(param.numel() for param in model.parameters()) / 1e6

    logger.info(f"模型配置: {model_config}")
    logger.info(f"模型参数量: {param_count:.1f}M")

    # 训练器
    device = device_type("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = AdamW(model.parameters(), lr=model_config.learning_rate, betas=(0.9, 0.999), weight_decay=3e-4)
    lr_scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.9,
        patience=3,
        min_lr=1e-10,
        threshold_mode="abs",
        threshold=0.005,
    )
    dropout_scheduler = DropoutScheduler(
        model.named_modules(),
        initial_dropout=model_config.dropout,
        factor=0.3,
        patience=2,
    )

    trainer = Trainer.from_scratch(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        tokenizer=tokenizer,
        optimizer=optimizer,
        dropout_scheduler=dropout_scheduler,
        lr_scheduler=lr_scheduler,
        log_interval=50,
        logger=logger,
    )

    trainer = Trainer.from_checkpoint(
        checkpoint_path=Path("../../../models/checkpoints/color_semantic_20251006_190933.pt"),
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        tokenizer=tokenizer,
        log_interval=50,
        logger=logger,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        dropout_scheduler=dropout_scheduler,
        load_optimizer=False,
        load_lr_scheduler=False,
        load_dp_scheduler=False,
    )

    trainer.train()


if __name__ == '__main__':
    main()
    # Trainer.save_checkpoint("color_semantic_20250929_105649")
