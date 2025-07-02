import pandas as pd
import torch
from torch.utils.data import Dataset
from tokenizers import ByteLevelBPETokenizer

class TextDataset(Dataset):
    def __init__(self, parquet_path: str, seq_len: int):
        # load your scraped text
        df = pd.read_parquet(parquet_path, columns=["text"])
        self.texts = df["text"].dropna().tolist()

        # load the trained BPE tokenizer
        self.tokenizer = ByteLevelBPETokenizer(
            "services/text/data/tokenizer-vocab.json",
            "services/text/data/tokenizer-merges.txt",
        )
        # pad/truncate to fixed length
        self.tokenizer.enable_truncation(max_length=seq_len)
        self.tokenizer.enable_padding(
            pad_id=self.tokenizer.token_to_id("<PAD>"),
            pad_token="<PAD>",
            length=seq_len
        )
        self.seq_len = seq_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer.encode(self.texts[idx])
        ids = torch.tensor(enc.ids, dtype=torch.long)
        # for language modeling, input and target are the same
        return ids, ids