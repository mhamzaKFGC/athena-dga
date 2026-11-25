"""n-CBDC: n-gram combined character-based domain classification (Xu, Shen, Du, Computers & Security 2019).

Reconstruction: three parallel embedding streams (characters, hashed bigrams, hashed trigrams),
each passed through multi-width 1-D CNNs and max-pooled, then fused by dense layers."""
import zlib

import torch
from torch import nn

from common.data import MAXLEN, VOCAB, encode
from common.layers import MultiKernelCNN, mask_of

BUCKETS = {2: 4096, 3: 8192}


def ngram_ids(labels, n):
    x = torch.zeros(len(labels), MAXLEN, dtype=torch.int32)
    for i, s in enumerate(labels):
        s = s[:MAXLEN]
        for j in range(len(s) - n + 1):
            x[i, j] = zlib.crc32(s[j:j + n].encode()) % BUCKETS[n] + 1
    return x


class NCBDC(nn.Module):
    def __init__(self, dim=64, filters=64, kernels=(2, 3, 4)):
        super().__init__()
        self.char_emb = nn.Embedding(VOCAB, dim, padding_idx=0)
        self.bi_emb = nn.Embedding(BUCKETS[2] + 1, dim, padding_idx=0)
        self.tri_emb = nn.Embedding(BUCKETS[3] + 1, dim, padding_idx=0)
        self.cnns = nn.ModuleList(MultiKernelCNN(dim, filters, kernels) for _ in range(3))
        width = 3 * self.cnns[0].out_dim
        self.head = nn.Sequential(nn.Linear(width, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))

    def prepare(self, labels):
        return encode(labels), ngram_ids(labels, 2), ngram_ids(labels, 3)

    def forward(self, chars, bi, tri):
        mask = mask_of(chars)
        streams = (self.char_emb(chars), self.bi_emb(bi), self.tri_emb(tri))
        feats = torch.cat([cnn(e, mask) for cnn, e in zip(self.cnns, streams)], -1)
        return self.head(feats).squeeze(-1)
