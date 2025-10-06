import math

import torch
from torch import Tensor
from torch.nn import Module, Embedding, TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer, \
    TransformerDecoder, Linear
from torch.nn.init import xavier_normal_

from src.flow_tint.core.constants import VOCAB_SIZE, PAD_TOKEN_ID
from src.flow_tint.model.component.model_config import ModelConfig
from src.flow_tint.model.component.positional_encoding import positional_encoding


class ColorModel(Module):
    """颜色语义理解与生成模型"""

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.d_model = config.d_model

        n_head = config.n_head
        n_encoder_layers = config.n_encoder_layers
        n_decoder_layers = config.n_decoder_layers
        d_ff = config.d_ff
        dropout = config.dropout
        max_len = config.max_len

        # 词嵌入
        self.embedding = Embedding(VOCAB_SIZE, self.d_model, padding_idx=PAD_TOKEN_ID)

        # 位置编码
        self.pos_encoding = positional_encoding(max_len, self.d_model)

        # Transformer编码器
        encoder_layer = TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=n_head,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,

        )
        self.encoder = TransformerEncoder(encoder_layer, n_encoder_layers, enable_nested_tensor=False)

        # Transformer解码器
        decoder_layer = TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=n_head,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,

        )
        self.decoder = TransformerDecoder(decoder_layer, n_decoder_layers)

        # 输出投影层
        self.output_projection = Linear(self.d_model, VOCAB_SIZE)

        # 词嵌入和输出投影共享权重
        # self.output_projection.weight = self.embedding.weight

        # 初始化模型参数
        self.init_parameters()

    def init_parameters(self):
        """初始化模型参数"""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_normal_(p)

    def forward(
            self,
            src: Tensor,
            tgt: Tensor,
            src_key_padding_mask: Tensor = None,
            tgt_key_padding_mask: Tensor = None,
    ) -> Tensor:
        """
        前向传播
        
        :param src: 源序列 [batch_size, src_len]
        :param src_key_padding_mask: 源序列padding_mask
        :param tgt: 目标序列 [batch_size, tgt_len]
        :param tgt_key_padding_mask: 目标序列padding_mask
        :return: 词汇表上的原始分数 [batch_size, tgt_len, vocab_size]
        """

        # 编码
        src_embedded = self.embedding(src) * math.sqrt(self.d_model)
        src_pe = self.pos_encoding[:, :src.shape[1], :]
        src_embedded = src_embedded + src_pe

        memory = self.encoder(src_embedded, src_key_padding_mask=src_key_padding_mask)

        if tgt is None:
            return memory

        # 解码器
        tgt_embedded = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_pe = self.pos_encoding[:, :tgt.shape[1], :]
        tgt_embedded = tgt_embedded + tgt_pe

        tgt_len = tgt.shape[1]
        # 目标因果mask
        tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool().to(tgt.device)

        decoder_output = self.decoder(
            tgt_embedded,
            memory,
            tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

        # 输出投影
        return self.output_projection(decoder_output)

    def encode(self, src: Tensor, src_key_padding_mask: Tensor) -> Tensor:
        """推理时的编码器独立调用"""

        src_embedded = self.embedding(src) * math.sqrt(self.d_model)
        src_embedded = src_embedded + self.pos_encoding[:, :src.shape[1], :]

        return self.encoder(src_embedded, src_key_padding_mask=src_key_padding_mask)

    def decode(self, memory: Tensor, memory_key_padding_mask: Tensor, tgt: Tensor) -> Tensor:
        """推理时的单步解码器独立调用"""

        tgt_len = tgt.shape[1]

        # 因果mask
        tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool().to(tgt.device)

        # 目标序列
        tgt_embedded = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_embedded = tgt_embedded + self.pos_encoding[:, :tgt_len, :]

        # 解码
        decoder_output = self.decoder(
            tgt_embedded,
            memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        # 输出投影，只输出最后一个位置
        return self.output_projection(decoder_output[:, -1, :])
