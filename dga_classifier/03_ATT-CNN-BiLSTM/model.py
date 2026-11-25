"""ATT-CNN-BiLSTM (Ren, Jiang, Wang, Liu - Cybersecurity 2020).

Reconstruction: character embedding -> 1-D CNN (local features) -> BiLSTM (both directions)
-> attention pooling that weights the most informative positions -> dense output."""
import torch
from torch import nn

from common.data import VOCAB, encode
from common.layers import AdditiveAttention, mask_of, run_rnn


class ATTCNNBiLSTM(nn.Module):
    def __init__(self, dim=64, filters=128, hidden=64):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, dim, padding_idx=0)
        self.conv = nn.Conv1d(dim, filters, 3, padding="same")
        self.lstm = nn.LSTM(filters, hidden, batch_first=True, bidirectional=True)
        self.att = AdditiveAttention(2 * hidden)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * hidden, 64), nn.ReLU(), nn.Linear(64, 1))

    def prepare(self, labels):
        return (encode(labels),)

    def forward(self, chars):
        mask = mask_of(chars)
        x = torch.relu(self.conv(self.emb(chars).transpose(1, 2))).transpose(1, 2)
        h, _ = run_rnn(self.lstm, x, mask)
        return self.head(self.att(h, mask)).squeeze(-1)
