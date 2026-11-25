"""HDDN - Hybrid DGA DefenseNet (Computers & Security, 2024).

Reconstruction: FastText-style input - each character position is represented by its character
embedding plus the averaged embeddings of the hashed character n-grams (n = 3, 4, 5) that start there,
with FastText's '<' '>' word-boundary markers. Then CNN (local patterns) -> LSTM (sequence) -> dense.
Implemented directly in PyTorch, so no FastText library is needed."""
import zlib

import torch
from torch import nn

from common.data import MAXLEN, VOCAB, encode
from common.layers import mask_of, masked_max, run_rnn

N_VALUES = (3, 4, 5)
BUCKETS = 20000


def subword_ids(labels):
    x = torch.zeros(len(labels), MAXLEN, len(N_VALUES), dtype=torch.int32)
    for i, s in enumerate(labels):
        s = s[:MAXLEN]
        w = "<" + s + ">"
        for j in range(len(s)):  # n-grams starting at character j (offset by the '<' marker)
            for k, n in enumerate(N_VALUES):
                if j + n <= len(w):
                    x[i, j, k] = zlib.crc32(w[j:j + n].encode()) % BUCKETS + 1
    return x


class HDDN(nn.Module):
    def __init__(self, dim=64, filters=128, hidden=128):
        super().__init__()
        self.char_emb = nn.Embedding(VOCAB, dim, padding_idx=0)
        self.sub_emb = nn.Embedding(BUCKETS + 1, dim, padding_idx=0)
        self.conv = nn.Conv1d(dim, filters, 3, padding="same")
        self.lstm = nn.LSTM(filters, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1))

    def prepare(self, labels):
        return encode(labels), subword_ids(labels)

    def forward(self, chars, subs):
        mask = mask_of(chars)
        count = (subs != 0).sum(-1, keepdim=True).clamp(min=1)
        x = self.char_emb(chars) + self.sub_emb(subs).sum(2) / count
        x = torch.relu(self.conv(x.transpose(1, 2))).transpose(1, 2)
        h, _ = run_rnn(self.lstm, x, mask)
        return self.head(masked_max(h, mask)).squeeze(-1)
