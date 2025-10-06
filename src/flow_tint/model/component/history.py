import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Self


class History:
    def __init__(self):
        # 基础训练信息
        self.epochs: list[int] = []
        self.epoch_times: list[float] = []

        # 损失指标
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.loss_improvements: list[float] = []

        # 准确率指标
        self.train_token_accuracies: list[float] = []
        self.train_sequence_accuracies: list[float] = []
        self.val_token_accuracies: list[float] = []
        self.val_sequence_accuracies: list[float] = []

        # 优化器状态
        self.learning_rates: list[float] = []
        self.dropouts: list[float] = []
        self.gradient_norms: list[float] = []

        # 早停相关
        self.early_stopping_counters: list[int] = []

    def add_epoch(self, epoch: int, epoch_time: float,
                  train_loss: float, val_loss: float,
                  train_metrics: dict[str, Any], val_metrics: dict[str, Any],
                  learning_rate: float, dropout: float, gradient_norm: float,
                  early_stopping_counter: int):
        """添加一个epoch的完整记录"""
        self.epochs.append(epoch)
        self.epoch_times.append(epoch_time)

        # 损失
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        # 计算loss improvement (相对于上一个epoch)
        if len(self.val_losses) > 1:
            improvement = self.val_losses[-2] - val_loss
            self.loss_improvements.append(improvement)
        else:
            self.loss_improvements.append(0.0)

        # 准确率
        self.train_token_accuracies.append(train_metrics["avg_token_accuracy"])
        self.train_sequence_accuracies.append(train_metrics["avg_sequence_accuracy"])
        self.val_token_accuracies.append(val_metrics["avg_token_accuracy"])
        self.val_sequence_accuracies.append(val_metrics["avg_sequence_accuracy"])

        # 优化器状态
        self.learning_rates.append(learning_rate)
        self.dropouts.append(dropout)
        self.gradient_norms.append(gradient_norm)

        # 早停
        self.early_stopping_counters.append(early_stopping_counter)

    def get_latest(self) -> dict[str, Any]:
        """获取最新一个epoch的所有数据"""
        if not self.epochs:
            return {}

        return {
            "epoch": self.epochs[-1],
            "epoch_time": self.epoch_times[-1],
            "train_loss": self.train_losses[-1],
            "val_loss": self.val_losses[-1],
            "loss_improvement": self.loss_improvements[-1],
            "train_token_accuracy": self.train_token_accuracies[-1],
            "train_sequence_accuracy": self.train_sequence_accuracies[-1],
            "val_token_accuracy": self.val_token_accuracies[-1],
            "val_sequence_accuracy": self.val_sequence_accuracies[-1],
            "learning_rate": self.learning_rates[-1],
            "dropout": self.dropouts[-1],
            "gradient_norm": self.gradient_norms[-1],
            "early_stopping_counter": self.early_stopping_counters[-1],
        }

    def get_best_epoch(self) -> dict[str, Any]:
        """获取验证损失最低的epoch信息"""
        if not self.val_losses:
            return {}

        best_idx = self.val_losses.index(min(self.val_losses))
        return {
            "epoch": self.epochs[best_idx],
            "epoch_time": self.epoch_times[best_idx],
            "train_loss": self.train_losses[best_idx],
            "val_loss": self.val_losses[best_idx],
            "loss_improvement": self.loss_improvements[best_idx],
            "train_token_accuracy": self.train_token_accuracies[best_idx],
            "train_sequence_accuracy": self.train_sequence_accuracies[best_idx],
            "val_token_accuracy": self.val_token_accuracies[best_idx],
            "val_sequence_accuracy": self.val_sequence_accuracies[best_idx],
            "learning_rate": self.learning_rates[best_idx],
            "dropout": self.dropouts[best_idx],
            "gradient_norm": self.gradient_norms[best_idx],
            "early_stopping_counter": self.early_stopping_counters[best_idx],
        }

    def detect_overfitting(self, window=3) -> tuple[bool, bool]:
        if len(self.epochs) < window:
            return False, False

        train_losses = self.train_losses[-window:]
        val_losses = self.val_losses[-window:]

        train_decreasing = all(train_losses[i] > train_losses[i + 1] for i in range(window - 1))
        val_increasing = all(val_losses[i] < val_losses[i + 1] for i in range(window - 1))

        return train_decreasing, val_increasing

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式，用于保存到checkpoint"""
        return {
            "epochs": self.epochs,
            "epoch_times": self.epoch_times,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "loss_improvements": self.loss_improvements,
            "train_token_accuracies": self.train_token_accuracies,
            "train_sequence_accuracies": self.train_sequence_accuracies,
            "val_token_accuracies": self.val_token_accuracies,
            "val_sequence_accuracies": self.val_sequence_accuracies,
            "learning_rates": self.learning_rates,
            "dropouts": self.dropouts,
            "gradient_norms": self.gradient_norms,
            "early_stopping_counters": self.early_stopping_counters,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """从字典恢复History对象，用于从checkpoint加载"""
        history = cls()
        history.epochs = data.get("epochs", [])
        history.epoch_times = data.get("epoch_times", [])
        history.train_losses = data.get("train_losses", [])
        history.val_losses = data.get("val_losses", [])
        history.loss_improvements = data.get("loss_improvements", [])
        history.train_token_accuracies = data.get("train_token_accuracies", [])
        history.train_sequence_accuracies = data.get("train_sequence_accuracies", [])
        history.val_token_accuracies = data.get("val_token_accuracies", [])
        history.val_sequence_accuracies = data.get("val_sequence_accuracies", [])
        history.learning_rates = data.get("learning_rates", [])
        history.dropouts = data.get("dropouts", [])
        history.gradient_norms = data.get("gradient_norms", [])
        history.early_stopping_counters = data.get("early_stopping_counters", [])
        return history

    def save_to_file(self, path: Path):
        """保存History到独立的JSON文件"""

        path.parent.mkdir(parents=True, exist_ok=True)

        # 添加元数据
        data = self.to_dict()
        data["metadata"] = {
            "created_at": datetime.now().isoformat(),
            "total_epochs": len(self.epochs),
            "best_epoch": self.get_best_epoch().get("epoch", 0) if self.epochs else 0,
            "best_val_loss": min(self.val_losses) if self.val_losses else sys.float_info.max,
            "final_val_loss": self.val_losses[-1] if self.val_losses else sys.float_info.max,
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_file(cls, path: Path) -> Self:
        """从JSON文件加载History"""

        if not path.exists():
            raise FileNotFoundError(f"History文件不存在: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return cls.from_dict(data)

    def save_training_report(self, path: Path):
        """保存训练报告到文本文件"""

        path.parent.mkdir(parents=True, exist_ok=True)

        report = self.generate_training_report()

        with open(path, 'w', encoding='utf-8') as f:
            f.write(report)

    def generate_event_timeline(self) -> str:
        """生成训练事件时间线"""
        if not self.epochs:
            return "无训练数据"

        events = [f"Epoch {self.epochs[0]:2d}  : 训练开始 | Loss={self.train_losses[0]:.3f}"]

        # 最佳验证损失
        best_val_idx = self.val_losses.index(min(self.val_losses))
        events.append(
            f"Epoch {self.epochs[best_val_idx]:2d}  : 最佳验证损失 | Val_Loss={min(self.val_losses):.3f} (BEST)")

        # 学习率变化
        for i in range(1, len(self.learning_rates)):
            if self.learning_rates[i] != self.learning_rates[i - 1]:
                events.append(
                    f"Epoch {self.epochs[i]:2d}  : 学习率调整 | LR={self.learning_rates[i - 1]:.1e} → {self.learning_rates[i]:.1e}"
                )

        # 丢弃率变化
        for i in range(len(self.dropouts)):
            if self.dropouts[i] != self.dropouts[i - 1]:
                events.append(
                    f"Epoch {self.epochs[i]:2d}  : 丢弃率调整 | DP={self.dropouts[i - 1]:.2f} → {self.dropouts[i]:.2f}"
                )

        # 早停警告
        for i, counter in enumerate(self.early_stopping_counters):
            if counter >= 3:
                events.append(f"Epoch {self.epochs[i]:2d}  : 早停警告 | 连续{counter}个epoch无改善")

        # 训练结束
        if len(self.epochs) > 1:
            events.append(f"Epoch {self.epochs[-1]:2d}  : 训练结束 | 最终Val_Loss={self.val_losses[-1]:.3f}")

        return "训练事件时间线:\n" + "\n".join(events)

    def generate_data_overview(self) -> str:
        """生成数据概览表格"""
        if not self.epochs:
            return "无训练数据"

        header = "Epoch | Train_Loss | Val_Loss | Token_Acc | Seq_Acc | LR      | Dropout | Grad_Norm"
        separator = "------|------------|----------|-----------|---------|---------|---------|----------"

        lines = [header, separator]

        # 关键epoch: 首、末、最佳、每10个
        key_epochs = [0, -1]
        if len(self.epochs) > 20:
            key_epochs.extend([i for i in range(9, len(self.epochs), 10)])
        key_epochs.append(self.val_losses.index(min(self.val_losses)))

        for idx in sorted(set(key_epochs)):
            if 0 <= idx < len(self.epochs):
                line = (f"{self.epochs[idx]:5d} | "
                        f"{self.train_losses[idx]:10.4f} | "
                        f"{self.val_losses[idx]:8.4f} | "
                        f"{self.val_token_accuracies[idx]:9.4f} | "
                        f"{self.val_sequence_accuracies[idx]:7.4f} | "
                        f"{self.learning_rates[idx]:7.1e} | "
                        f"{self.dropouts[idx]:9.2f} | "
                        f"{self.gradient_norms[idx]:9.2f}")
                lines.append(line)

        return "\n".join(lines)

    def generate_ascii_chart(self, data_key: str, height: int = 10, width: int = 50) -> str:
        """生成ASCII趋势图"""
        if not hasattr(self, data_key):
            return f"数据字段 {data_key} 不存在"

        data = getattr(self, data_key)
        if not data:
            return f"{data_key} 无数据"

        # 数据采样 (如果数据点太多)
        if len(data) > width:
            step = len(data) // width
            sampled_data = [data[i] for i in range(0, len(data), step)]
        else:
            sampled_data = data

        # 归一化
        min_val, max_val = min(sampled_data), max(sampled_data)
        range_val = max_val - min_val if max_val != min_val else 1

        lines = []
        for row in range(height, 0, -1):
            threshold = min_val + (row - 0.5) * range_val / height
            line = f"{min_val + (row - 1) * range_val / height:5.2f} ┤"

            for val in sampled_data:
                line += "█" if val >= threshold else " "
            lines.append(line)

        # 坐标轴
        axis = "      ┴" + "─" * len(sampled_data)
        lines.append(axis)

        title = data_key.replace('_', ' ').title() + " 趋势:"
        return title + "\n" + "\n".join(lines)

    def generate_training_report(self) -> str:
        """生成完整训练报告"""
        if not self.epochs:
            return "无训练数据可分析"

        report = list()
        report.append("=" * 60)
        report.append("训练分析报告")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 60)
        report.append("")

        # 基本统计
        report.append("训练统计:")
        report.append(f"  总epoch数: {len(self.epochs)}")
        report.append(
            f"  最佳验证损失: {min(self.val_losses):.4f} (Epoch {self.epochs[self.val_losses.index(min(self.val_losses))] + 1})")
        report.append(f"  最终验证损失: {self.val_losses[-1]:.4f}")
        report.append(f"  最终Token准确率: {self.val_token_accuracies[-1]:.4f}")
        report.append(f"  最终序列准确率: {self.val_sequence_accuracies[-1]:.4f}")
        report.append("")

        # 事件时间线
        report.append(self.generate_event_timeline())
        report.append("")

        # 数据概览
        report.append(self.generate_data_overview())
        report.append("")

        # 趋势图
        report.append(self.generate_ascii_chart("val_losses"))
        report.append("")
        report.append(self.generate_ascii_chart("val_token_accuracies"))
        report.append("")

        return "\n".join(report)
