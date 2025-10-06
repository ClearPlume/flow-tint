from dataclasses import dataclass


@dataclass
class ModelConfig:
    """模型配置"""

    d_model: int
    n_head: int
    d_ff: int
    n_encoder_layers: int
    n_decoder_layers: int
    learning_rate: float
    dropout: float

    max_len: int = 2048

    @staticmethod
    def micro(learning_rate=1e-4, dropout=0.1):
        return ModelConfig(256, 8, 1024, 6, 6, learning_rate, dropout)

    @staticmethod
    def small(learning_rate=8e-5, dropout=0.15):
        return ModelConfig(512, 8, 2048, 8, 8, learning_rate, dropout)

    @staticmethod
    def medium(learning_rate=5e-5, dropout=0.2):
        return ModelConfig(768, 12, 3072, 8, 8, learning_rate, dropout)

    @staticmethod
    def large(learning_rate=3e-5, dropout=0.25):
        return ModelConfig(1024, 16, 4096, 16, 16, learning_rate, dropout)
