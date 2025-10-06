import os

from sentencepiece import SentencePieceProcessor

from src.flow_tint import BOS_TOKEN, EOS_TOKEN


class Tokenizer:
    """词表处理器"""
    tokenizer: SentencePieceProcessor

    bos_token_id: int
    eos_token_id: int

    def __init__(self, vocab_file: str):
        if os.path.exists(vocab_file):
            self.tokenizer = SentencePieceProcessor()
            self.tokenizer.Load(vocab_file)
        else:
            raise FileNotFoundError(f"Vocab file not found: {vocab_file}")

        self.bos_token_id = self.piece_to_id(BOS_TOKEN)
        self.eos_token_id = self.piece_to_id(EOS_TOKEN)

    def piece_to_id(self, text: str) -> int:
        """编码单个token为id"""
        return self.tokenizer.PieceToId(text)

    def encode(self, text: str) -> list[int]:
        """编码文本为token ids"""
        return self.tokenizer.Encode(text)

    def decode(self, token_ids: list[int]) -> str:
        """解码token ids为文本"""
        return self.tokenizer.Decode(token_ids)

    @property
    def vocab_size(self):
        return self.tokenizer.GetPieceSize()
