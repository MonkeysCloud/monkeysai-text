import re
from collections import Counter

class SimpleTokenizer:
    """
    A toy word-punctuation tokenizer:
     - Split on word characters or single punctuation.
     - Builds a vocab from your corpus.
     - Reserves 0 for <PAD>, 1 for <UNK>.
    """
    def __init__(self, texts, min_freq: int = 1):
        # flatten all texts, lowercase
        tokens = re.findall(r"\w+|[^\w\s]", " ".join(texts).lower())
        counts = Counter(tokens)
        # only keep tokens that appear >= min_freq
        vocab = [tok for tok, freq in counts.items() if freq >= min_freq]
        # build mappings
        self.token2id = {"<PAD>": 0, "<UNK>": 1}
        for idx, tok in enumerate(vocab, start=2):
            self.token2id[tok] = idx
        self.id2token = {i: t for t, i in self.token2id.items()}

    def encode(self, text: str) -> list[int]:
        toks = re.findall(r"\w+|[^\w\s]", text.lower())
        return [ self.token2id.get(t, 1) for t in toks ]  # unknown → 1

    def decode(self, ids: list[int]) -> str:
        return " ".join(self.id2token.get(i, "<UNK>") for i in ids)