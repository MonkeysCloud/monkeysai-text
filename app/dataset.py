"""app/dataset.py — production-ready
• Caches tokenised + padded tensors to avoid re-encoding in DataLoader workers
• Disables Hugging Face tokenizers parallelism warning
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # silent & safe after fork

import pandas as pd
import torch
from torch.utils.data import Dataset
from tokenizers import ByteLevelBPETokenizer

TOKENIZER_VOCAB = "services/text/data/tokenizer-vocab.json"
TOKENIZER_MERGE = "services/text/data/tokenizer-merges.txt"
PAD_TOKEN       = "<PAD>"

class TextDataset(Dataset):
    """Pre-tokenised, padded language-model dataset.

    Each item returns (input_ids, target_ids) where target == input (next-token LM).
    All heavy work (BPE encode, pad/truncate) is done once in __init__, so
    DataLoader workers merely index an in-memory tensor list — no HF fork warning.
    """

    def __init__(self, parquet_path: str, seq_len: int):
        df = pd.read_parquet(parquet_path, columns=["text"])
        texts = df["text"].dropna().tolist()

        self.tokenizer = ByteLevelBPETokenizer(TOKENIZER_VOCAB, TOKENIZER_MERGE)
        pad_id = self.tokenizer.token_to_id(PAD_TOKEN)
        self.seq_len = seq_len

        self.samples: list[torch.Tensor] = []
        for txt in texts:
            ids = self.tokenizer.encode(txt).ids[: seq_len]
            ids += [pad_id] * (seq_len - len(ids))  # right-pad
            self.samples.append(torch.tensor(ids, dtype=torch.long))

    # dataset protocol ---------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        ids = self.samples[idx]
        return ids, ids  # (input, target)
