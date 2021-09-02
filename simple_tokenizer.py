from collections import Counter
from itertools import chain
from typing import List
import re

WHITE_SPACE_TOKENIZER = lambda string: re.split(r"\s+", string)

class SimpleTokenizer:
    """A mapping from token to token index and vice versa"""
    def __init__(self, token2idx, idx2token, pad, unknown, tokenizer=WHITE_SPACE_TOKENIZER):
        self.token2idx = token2idx
        self.idx2token = idx2token
        self.pad = pad
        self.unknown = unknown
        self.dismissed_tokens = {self.pad}
        self.tokenizer = tokenizer
    
    def encode(self, string):
        """Tokenize a string and mapping each token to its index in vocab"""
        return [self.token2idx.get(token, self.unknown) for token in self.tokenizer(string) if token]
    
    def decode(self, indices: List[int]):
        return ' '.join(
            self.idx2token.get(idx, '') for idx in indices if idx not in self.dismissed_tokens
        )
    
    def pad_sequence(self, sequence: List[int], max_len: int = 256):
        stripped_sequence = sequence[:max_len]
        nbr_pads = max_len - len(stripped_sequence)
        return stripped_sequence + [self.pad] * nbr_pads
    
    def texts_to_sequences(self, lines: List[str], max_len: int = 256):
        return [self.pad_sequence(self.encode(line), max_len) for line in lines]