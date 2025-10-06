import signal
import sys
from datetime import datetime
from importlib.metadata import version
from logging import Logger
from pathlib import Path
from typing import cast

import torch
from torch import device as device_type, Tensor
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.flow_tint.path import get_models_dir, get_logs_dir
from src.flow_tint.model.component.batch_sampler import LengthGroupedBatchSampler
from src.flow_tint.model.component.color_model import ColorModel
from src.flow_tint.model.component.dropout_scheduler import DropoutScheduler
from src.flow_tint.model.component.history import History
from src.flow_tint.model.component.tokenizer import Tokenizer


def compute_loss(logits: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """
    计算序列生成损失

    :param logits [batch_size, seq_len, vocab_size]
    :param target [batch_size, seq_len]
    :param mask [batch_size, seq_len] True为Padding位置
    """

    batch_size, seq_len, vocab_size = logits.shape

    # 重新指定数据读取维度
    logits_view = logits.view(-1, vocab_size)
    target_view = target.view(-1)
    mask_view = mask.view(-1)

    # 计算损失
    loss = F.cross_entropy(logits_view, target_view, reduction='none', label_smoothing=0.05)

    # 忽略Padding
    logs = loss * (~mask_view).float()

    # 平均损失
    valid_tokens = (~mask_view).sum()
    if valid_tokens > 0:
        return logs.sum() / valid_tokens
    else:
        return logs.sum()


def compute_metrics(logits: Tensor, target: Tensor, mask: Tensor) -> dict[str, float]:
    """计算评估指标"""

    # 获取预测
    predictions = logits.argmax(dim=-1)

    # 忽略padding位置
    valid_mask = ~mask

    # Token级别准确率
    token_correct = (predictions == target) & valid_mask
    token_accuracy = token_correct.sum().float() / valid_mask.sum().float()

    # 序列级别准确率
    seq_correct = (token_correct | mask).all(dim=1)
    seq_accuracy = seq_correct.float().mean()

    return {
        "token_accuracy": token_accuracy.item(),
        "sequence_accuracy": seq_accuracy.item(),
        "valid_tokens": valid_mask.sum().item(),
    }


class Trainer:
    # stateless
    model: ColorModel
    train_loader: DataLoader
    val_loader: DataLoader
    device: device_type
    tokenizer: Tokenizer
    log_interval: int
    logger: Logger

    # stateful
    optimizer: Optimizer
    lr_scheduler: LRScheduler
    dropout_scheduler: DropoutScheduler
    history: History
    best_val_loss: float
    start_epoch: int

    # fixed
    epochs = 50
    early_stopping_counter = 0
    early_stopping_patience = 5
    model_dir = get_models_dir()
    checkpoint_dir = model_dir / "checkpoints"
    log_dir = get_logs_dir()

    def __init__(self):
        raise RuntimeError("请使用 from_scratch() 或 from_checkpoint() 创建 Trainer 实例")

    @staticmethod
    def _init_base(
            model: ColorModel,
            train_loader: DataLoader,
            val_loader: DataLoader,
            device: device_type,
            tokenizer: Tokenizer,
            log_interval: int,
            logger: Logger,
    ) -> "Trainer":
        trainer = object.__new__(Trainer)

        trainer.model = model.to(device)
        trainer.train_loader = train_loader
        trainer.val_loader = val_loader
        trainer.device = device
        trainer.tokenizer = tokenizer
        trainer.log_interval = log_interval
        trainer.logger = logger

        return trainer

    @staticmethod
    def from_scratch(
            model: ColorModel,
            train_loader: DataLoader,
            val_loader: DataLoader,
            device: device_type,
            tokenizer: Tokenizer,
            optimizer: Optimizer,
            lr_scheduler: LRScheduler,
            dropout_scheduler: DropoutScheduler,
            log_interval: int,
            logger: Logger,
    ) -> "Trainer":
        trainer = Trainer._init_base(model, train_loader, val_loader, device, tokenizer, log_interval, logger)

        trainer.optimizer = optimizer
        trainer.lr_scheduler = lr_scheduler
        trainer.dropout_scheduler = dropout_scheduler
        trainer.history = History()
        trainer.best_val_loss = float("inf")
        trainer.start_epoch = 1

        trainer.logger.info("训练器初始化完成")
        trainer._log_basic()

        return trainer

    @staticmethod
    def from_checkpoint(
            checkpoint_path: Path,
            model: ColorModel,
            train_loader: DataLoader,
            val_loader: DataLoader,
            device: device_type,
            tokenizer: Tokenizer,
            log_interval: int,
            logger: Logger,
            optimizer: Optimizer,
            lr_scheduler: LRScheduler,
            dropout_scheduler: DropoutScheduler,
            load_optimizer: bool = True,
            load_lr_scheduler: bool = True,
            load_dp_scheduler: bool = True,
    ) -> "Trainer":
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint文件不存在: {checkpoint_path.absolute()}")

        trainer = Trainer._init_base(model, train_loader, val_loader, device, tokenizer, log_interval, logger)
        trainer.optimizer = optimizer
        trainer.lr_scheduler = lr_scheduler
        trainer.dropout_scheduler = dropout_scheduler

        checkpoint = torch.load(checkpoint_path)

        trainer.model.load_state_dict(checkpoint["model_state"])

        if load_optimizer:
            try:
                trainer.optimizer.load_state_dict(checkpoint["optimizer_state"])
            except Exception as e:
                logger.warning(f"无法加载优化器状态: {e}")

        if load_lr_scheduler:
            try:
                trainer.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state"])
            except Exception as e:
                logger.warning(f"无法加载学习率调度器状态: {e}")

        if load_dp_scheduler:
            try:
                trainer.dropout_scheduler.load_state_dict(checkpoint["dp_scheduler_state"])
            except Exception as e:
                logger.warning(f"无法加载丢弃率调度器状态: {e}")

        trainer.history = History.from_dict(checkpoint["history"])
        trainer.best_val_loss = checkpoint["best_val_loss"]
        trainer.start_epoch = checkpoint["start_epoch"] + 1

        trainer.logger.info("训练器恢复完成")
        trainer._log_basic()

        return trainer

    @staticmethod
    def save_checkpoint(checkpoint_name: str):
        trainer = object.__new__(Trainer)
        checkpoint_path = Trainer.checkpoint_dir / f"{checkpoint_name}.pt"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint文件不存在: {checkpoint_path.absolute()}")

        checkpoint = torch.load(checkpoint_path)
        model_path = trainer.model_dir / f"color_semantic_{version("flow-tint")}.pt"

        torch.save(checkpoint["model_state"], model_path)

    def _log_basic(self):
        self.logger.info(f"学习率: {self.optimizer.param_groups[0]['lr']}")
        self.logger.info(f"批次大小: {cast(LengthGroupedBatchSampler, self.train_loader.batch_sampler).batch_size}")
        self.logger.info(f"训练批次数: {len(self.train_loader)}")
        self.logger.info(f"验证批次数: {len(self.val_loader)}")

    def train_epoch(self, epoch: int):
        """一个训练epoch"""

        self.model.train()
        total_loss = 0.0
        total_norm = 0.0
        all_token_acc = []
        all_seq_acc = []

        progress = tqdm(self.train_loader, f"Epoch {epoch} [Train]")

        for batch_idx, batch in enumerate(progress):
            batch: dict[str, Tensor]

            src = batch["src"].to(self.device)
            tgt_in = batch["tgt_in"].to(self.device)
            tgt_out = batch["tgt_out"].to(self.device)
            src_padding_mask = batch["src_padding_mask"].to(self.device)
            tgt_padding_mask = batch["tgt_padding_mask"].to(self.device)

            # 第一个和最后一个epoch输出详细信息
            if epoch == 1 and batch_idx == 0 or epoch == 50 and batch_idx == len(progress):
                print(f"\n=== 训练数据诊断 ===")
                print(f"Batch shapes - src: {src.shape}, tgt_in: {tgt_in.shape}, tgt_out: {tgt_out.shape}")
                print(f"Source tokens: {src[0].tolist()}")
                print(f"Target input: {tgt_in[0].tolist()}")
                print(f"Target output: {tgt_out[0].tolist()}")
                print(f"Source text: {self.tokenizer.decode(src[0].tolist())}")
                print(f"Target text: {self.tokenizer.decode(tgt_out[0].tolist())}")
                print(f"Padding masks - src: {src_padding_mask[0]}, tgt: {tgt_padding_mask[0]}")

            # 前向传播
            logits = self.model(
                src,
                tgt_in,
                src_key_padding_mask=src_padding_mask,
                tgt_key_padding_mask=tgt_padding_mask,
            )

            # 第一个和最后一个epoch输出模型输出诊断
            if epoch == 1 and batch_idx == 0 or epoch == 50 or batch_idx == len(progress):
                print(f"\n=== 模型输出诊断 ===")
                print(f"Logits shape: {logits.shape}")
                print(f"Logits range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
                print(f"Logits mean: {logits.mean().item():.3f}, std: {logits.std().item():.3f}")

                # 预测的token
                predictions = logits.argmax(dim=-1)
                print(f"Predicted tokens: {predictions[0].tolist()}")
                print(f"Predicted text: {self.tokenizer.decode(predictions[0].tolist())}")

                # 输出概率分布
                probs = F.softmax(logits[0, 0], dim=-1)
                top_probs, top_indices = torch.topk(probs, 5)
                print(f"Top 5 probabilities for first token:")
                for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                    token_text = self.tokenizer.decode([idx.item()])
                    print(f"  {i + 1}. {token_text} (id={idx.item()}): {prob.item():.4f}")

            # 计算损失
            loss = compute_loss(logits, tgt_out, tgt_padding_mask)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            total_norm = clip_grad_norm_(self.model.parameters(), 1).item()

            # 第一个和最后一个epoch输出梯度诊断
            if epoch == 1 and batch_idx == 0 or epoch == 50 or batch_idx == len(progress):
                # 梯度诊断
                print(f"\n=== 梯度诊断 ===")
                print(f"Gradient norm before clipping: {total_norm:.4f}")

                # 检查各层梯度
                grad_norms = {}
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        grad_norms[name] = param.grad.norm().item()

                print("Gradient norms by layer:")
                for name, norm in sorted(grad_norms.items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"  {name}: {norm:.4f}")

            total_loss += loss.item()

            self.optimizer.step()

            metrics = compute_metrics(logits, tgt_out, tgt_padding_mask)
            all_token_acc.append(metrics["token_accuracy"])
            all_seq_acc.append(metrics["sequence_accuracy"])

            # 更新进度条
            if batch_idx % self.log_interval == 0:
                progress.set_postfix({
                    "loss": f"{loss.item():.5f}",
                    "token_acc": f"{metrics['token_accuracy']:.3f}",
                    "seq_acc": f"{metrics['sequence_accuracy']:.3f}",
                    "grad_norm": f"{total_norm:.3f}",
                })

        # 计算epoch指标
        avg_loss = total_loss / len(self.train_loader)
        avg_token_acc = sum(all_token_acc) / len(all_token_acc)
        avg_seq_acc = sum(all_seq_acc) / len(all_seq_acc)

        epoch_metrics = {
            "avg_token_accuracy": avg_token_acc,
            "avg_sequence_accuracy": avg_seq_acc,
        }

        return avg_loss, total_norm, epoch_metrics

    @torch.no_grad()
    def validate(self, epoch: int):
        """验证模型"""

        self.model.eval()
        total_loss = 0
        all_token_acc = []
        all_seq_acc = []

        progress = tqdm(self.val_loader, f"Epoch {epoch} [Val]")

        for batch in progress:
            batch: dict[str, Tensor]

            src = batch["src"].to(self.device)
            tgt_in = batch["tgt_in"].to(self.device)
            tgt_out = batch["tgt_out"].to(self.device)
            src_padding_mask = batch["src_padding_mask"].to(self.device)
            tgt_padding_mask = batch["tgt_padding_mask"].to(self.device)

            logits = self.model(
                src,
                tgt_in,
                src_key_padding_mask=src_padding_mask,
                tgt_key_padding_mask=tgt_padding_mask,
            )

            loss = compute_loss(logits, tgt_out, tgt_padding_mask)
            total_loss += loss.item()

            metrics = compute_metrics(logits, tgt_out, tgt_padding_mask)
            all_token_acc.append(metrics["token_accuracy"])
            all_seq_acc.append(metrics["sequence_accuracy"])

        avg_loss = total_loss / len(self.val_loader)
        avg_token_acc = sum(all_token_acc) / len(all_token_acc)
        avg_seq_acc = sum(all_seq_acc) / len(all_seq_acc)

        epoch_metrics = {
            "avg_token_accuracy": avg_token_acc,
            "avg_sequence_accuracy": avg_seq_acc,
        }

        return avg_loss, epoch_metrics

    def _save_emergency_report(self, reason: str = "手动中断", save_checkpoint: bool = True):
        """保存紧急训练报告"""
        emergency_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            # 保存History
            self.history.save_to_file(self.log_dir / f"training_{emergency_timestamp}_emergency_history.json")

            # 尝试保存当前模型状态 (即使不完整)
            if save_checkpoint and hasattr(self, 'optimizer') and hasattr(self, 'scheduler'):
                current_epoch = len(self.history.epochs)

                torch.save(
                    {
                        "model_state": self.model.state_dict(),
                        "optimizer_state": self.optimizer.state_dict(),
                        "lr_scheduler_state": self.lr_scheduler.state_dict(),
                        "dp_scheduler_state": self.dropout_scheduler.state_dict(),
                        "history": self.history.to_dict(),
                        "best_val_loss": self.best_val_loss,
                        "start_epoch": current_epoch,
                        "is_emergency_save": True,
                        "emergency_reason": reason,
                    },
                    self.checkpoint_dir / f"emergency_{emergency_timestamp}.pt",
                )

                print(f"⚠️  紧急保存模型checkpoint (可能不完整): checkpoints/emergency_{emergency_timestamp}.pt")

            # 生成紧急报告
            emergency_report = f"训练{reason}报告\n"
            emergency_report += f"中断时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            emergency_report += f"中断原因: {reason}\n"
            emergency_report += f"已完成epoch数: {len(self.history.epochs)}\n"
            emergency_report += f"最后保存的最佳损失: {self.best_val_loss:.5f}\n"
            emergency_report += "⚠️  注意: 紧急保存的模型可能包含不完整的训练状态\n"
            emergency_report += "=" * 60 + "\n\n"
            emergency_report += self.history.generate_training_report()

            with open(
                    self.log_dir / f"training_{emergency_timestamp}_emergency_report.txt",
                    'w',
                    encoding='utf-8',
            ) as f:
                f.write(emergency_report)

            print(f"\n💾 紧急保存完成: logs/training_{emergency_timestamp}_emergency_*")
            if self.logger:
                self.logger.info(f"紧急保存训练数据: {reason} | 已完成epoch: {len(self.history.epochs)}")

        except Exception as e:
            print(f"❌ 紧急保存失败: {e}")
            # 至少尝试保存History
            try:
                self.history.save_to_file(self.log_dir / f"training_{emergency_timestamp}_history_only.json")
                print(f"📝 至少保存了训练历史数据")
            except:
                print(f"💥 连历史数据都无法保存")

    def train(self):
        """完整训练流程"""

        # 设置信号处理器
        def signal_handler(signum, _):
            print(f"\n🛑 接收到中断信号 ({signum})")
            self._save_emergency_report("手动中断")
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # IDE停止按钮

        if self.logger:
            self.logger.info(f"开始训练，共{self.epochs - (self.start_epoch - 1)}个epoch")

        try:
            for epoch in range(self.start_epoch, self.epochs + 1):
                epoch_start_time = datetime.now()

                print(f"\n{"=" * 60}")
                print(f"Epoch {epoch}/{self.epochs}")
                print(f"=" * 60)

                train_loss, total_norm, train_metrics = self.train_epoch(epoch)
                val_loss, val_metrics = self.validate(epoch)

                # 调用学习率调度器
                old_lr = self.optimizer.param_groups[0]['lr']
                if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                    self.lr_scheduler.step(val_loss)
                else:
                    self.lr_scheduler.step()
                new_lr = self.optimizer.param_groups[0]['lr']

                epoch_time = datetime.now() - epoch_start_time

                print(f"\nTrain Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}")
                print(
                    f"Train Token Accuracy: {train_metrics["avg_token_accuracy"]:.5f}, ",
                    f"Train Sequence Accuracy: {train_metrics["avg_sequence_accuracy"]:.5f}",
                )
                print(
                    f"Val Token Accuracy: {val_metrics["avg_token_accuracy"]:.5f}, ",
                    f"Val Sequence Accuracy: {val_metrics["avg_sequence_accuracy"]:.5f}",
                )

                # 调用dropout调度器
                old_dp = self.dropout_scheduler.current_dropout
                train_decreasing, val_increasing = self.history.detect_overfitting()
                self.dropout_scheduler.step(train_decreasing, val_increasing)
                new_dp = self.dropout_scheduler.current_dropout

                self.history.add_epoch(
                    epoch=epoch,
                    epoch_time=epoch_time.total_seconds(),
                    train_loss=train_loss,
                    val_loss=val_loss,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    learning_rate=new_lr,
                    dropout=new_dp,
                    gradient_norm=total_norm,
                    early_stopping_counter=self.early_stopping_counter,
                )

                # 详细日志记录
                if self.logger:
                    lr = f"学习率: {new_lr:.2e}" + \
                         (f" (从 {old_lr:.2e} 调整)" if old_lr != new_lr else "")
                    dp = f"丢弃率: {new_dp:.2f}" + \
                         (f" (从 {old_dp:.2f} 调整)" if old_dp != new_dp else "")

                    self.logger.info(
                        f"Epoch {epoch:2d}/{self.epochs} | "
                        f"时间: {epoch_time.total_seconds():.1f}s | "
                        f"训练损失: {train_loss:.5f} | "
                        f"验证损失: {val_loss:.5f} | "
                        f"训练token准确率: {train_metrics['avg_token_accuracy']:.4f} | "
                        f"训练序列准确率: {train_metrics['avg_sequence_accuracy']:.4f} | "
                        f"验证token准确率: {val_metrics['avg_token_accuracy']:.4f} | "
                        f"验证序列准确率: {val_metrics['avg_sequence_accuracy']:.4f} | "
                        f"梯度: {total_norm} | "
                        f"{lr} | "
                        f"{dp}"
                    )

                # 保存最佳模型和早停检查
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.early_stopping_counter = 0

                    now = datetime.now().strftime("%Y%m%d_%H%M%S")

                    # 保存checkpoint
                    torch.save(
                        {
                            "model_state": self.model.state_dict(),
                            "optimizer_state": self.optimizer.state_dict(),
                            "lr_scheduler_state": self.lr_scheduler.state_dict(),
                            "dp_scheduler_state": self.dropout_scheduler.state_dict(),
                            "history": self.history.to_dict(),
                            "best_val_loss": val_loss,
                            "start_epoch": epoch,
                        },
                        self.checkpoint_dir / f"color_semantic_{now}.pt",
                    )

                    # 同步保存History文件
                    self.history.save_to_file(self.log_dir / f"training_{now}_history.json")

                    print(f"✓ Saved best model(val loss: {self.best_val_loss:.5f})")
                    if self.logger:
                        self.logger.info(f"保存最佳模型 | epoch: {epoch} | 验证损失: {self.best_val_loss:.5f}")
                else:
                    self.early_stopping_counter += 1
                    if self.logger:
                        self.logger.info(
                            f"验证损失未改善 | 早停计数: {self.early_stopping_counter}/{self.early_stopping_patience}"
                        )

                    if self.early_stopping_counter >= self.early_stopping_patience:
                        if self.logger:
                            self.logger.info(f"早停触发 | 最佳验证损失: {self.best_val_loss:.5f}")
                        print(f"Early stopping at epoch {epoch}")
                        break

        except KeyboardInterrupt:
            print(f"\n⚠️  训练被手动中断 (Ctrl+C)")
            self._save_emergency_report("Ctrl+C中断")
            return
        except Exception as e:
            print(f"\n❌ 训练过程中发生异常: {e}")
            self._save_emergency_report(f"异常中断: {e}")
            raise

        # 训练正常结束，保存最终报告
        final_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.history.save_training_report(self.log_dir / f"training_{final_timestamp}_report.txt")

        if self.logger:
            self.logger.info(f"训练完成 | 最佳验证损失: {self.best_val_loss:.5f}")
            self.logger.info(f"训练报告已保存: logs/training_{final_timestamp}_report.txt")
