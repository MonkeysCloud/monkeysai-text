import torch
from torch.utils.data import Dataset
from app.tokenizer import SimpleTokenizer

import pandas as pd, torch
class TextDataset(Dataset):
    def __init__(self, parquet_path="services/text/data/latest.parquet", seq_len=128):
        df = pd.read_parquet(parquet_path, columns=["text"])
        lines = df["text"].tolist()

        # build a tokenizer on-the-fly
        self.tokenizer = SimpleTokenizer(lines)
        self.seq_len    = seq_len
        self.ids        = [ torch.tensor(
                                self.tokenizer.encode(line)[: seq_len+1],
                                dtype=torch.long
                             )
                             for line in lines ]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        x = self.ids[idx]
        tgt_len = self.seq_len + 1
        # pad or truncate to tgt_len
        if x.size(0) < tgt_len:
            pad = torch.zeros(tgt_len - x.size(0), dtype=torch.long)
            x = torch.cat([x, pad])
        else:
            x = x[:tgt_len]

        # input = all but last, target = all but first
        return x[:-1], x[1:]