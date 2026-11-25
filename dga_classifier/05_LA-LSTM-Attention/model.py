"""LA_Bin07 / LA_Mul07 (Tuan, Long, Taniar - Computers & Security 113, 2022).

Reconstruction: character embedding -> LSTM -> attention layer -> dense.
LA_Bin07 has one output (DGA or not); LA_Mul07 has one output per class (legit + each DGA family)."""
from torch import nn

from common.data import VOCAB, encode
from common.layers import AdditiveAttention, mask_of, run_rnn


class LALSTM(nn.Module):
    def __init__(self, n_out=1, dim=64, hidden=128):
        super().__init__()
        self.n_out = n_out
        self.emb = nn.Embedding(VOCAB, dim, padding_idx=0)
        self.lstm = nn.LSTM(dim, hidden, batch_first=True)
        self.att = AdditiveAttention(hidden)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden, n_out))

    def prepare(self, labels):
        return (encode(labels),)

    def forward(self, chars):
        mask = mask_of(chars)
        h, _ = run_rnn(self.lstm, self.emb(chars), mask)
        out = self.head(self.att(h, mask))
        return out.squeeze(-1) if self.n_out == 1 else out
