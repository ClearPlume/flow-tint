from typing import Iterator, Any

from torch.nn import Module, Dropout


class DropoutScheduler:
    dropouts: dict[str, Dropout]
    initial_dropout: float
    factor: float
    patience: int
    max_dropout: float
    overfitting_counter: int
    current_dropout: float

    def __init__(
            self,
            modules: Iterator[tuple[str, Module]],
            initial_dropout: float,
            factor: float,
            patience: int,
            max_dropout: float = 0.5,
    ):
        self.dropouts = {}

        for name, module in modules:
            if isinstance(module, Dropout):
                self.dropouts[name] = module

        self.initial_dropout = initial_dropout
        self.factor = factor
        self.patience = patience
        self.max_dropout = max_dropout

        self.overfitting_counter = 0
        self.current_dropout = initial_dropout

        # 设置所有dropout模块的初始值
        self._adjust_dropout(initial_dropout)

    def step(self, train_loss_decreasing: bool, val_loss_increasing: bool) -> float:
        if train_loss_decreasing and val_loss_increasing:
            self.overfitting_counter += 1

            if self.overfitting_counter >= self.patience:
                new_dropout = min(self.current_dropout * (1 + self.factor), self.max_dropout)

                if new_dropout != self.current_dropout:
                    self._adjust_dropout(new_dropout)
                    self.current_dropout = new_dropout
                    self.overfitting_counter = 0

        else:
            self.overfitting_counter = 0

        return self.current_dropout

    def _adjust_dropout(self, dropout):
        for _, module in self.dropouts.items():
            module.p = dropout

    def state_dict(self) -> dict[str, Any]:
        return {
            'initial_dropout': self.initial_dropout,
            'factor': self.factor,
            'patience': self.patience,
            'max_dropout': self.max_dropout,
            'overfitting_counter': self.overfitting_counter,
            'current_dropout': self.current_dropout,
        }

    def load_state_dict(self, state_dict: dict[str, Any]):
        self.initial_dropout = state_dict['initial_dropout']
        self.factor = state_dict['factor']
        self.patience = state_dict['patience']
        self.max_dropout = state_dict['max_dropout']
        self.overfitting_counter = state_dict['overfitting_counter']
        self.current_dropout = state_dict['current_dropout']

        # 恢复dropout模块的状态
        self._adjust_dropout(self.current_dropout)
