import torch
from torch.utils.data import Dataset
from app.tokenizer import SimpleTokenizer

class TextDataset(Dataset):
    """
    Loads lines from a text file and turns them into (input, target) ID tensors.
    """
    def __init__(self, path: str, seq_len: int = 128):
        # read your corpus
        with open(path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

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