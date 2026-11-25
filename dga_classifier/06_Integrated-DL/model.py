"""Integrated deep learning detector (Ravi, Alazab, Srinivasan, Arunachalam, Soman -
IEEE Transactions on Engineering Management 70(1), 2023).

Reconstruction: a CNN branch (local character patterns) and a BiLSTM branch (whole-sequence
context) run in parallel on the same character embedding and are fused before the output.
Robustness: train.py adds CharBot-style perturbed legit names as DGA examples (adversarial training)."""
import torch
from torch import nn

from common.data import VOCAB, encode
from common.layers import MultiKernelCNN, mask_of, masked_max, masked_mean, run_rnn


class IntegratedDL(nn.Module):
    def __init__(self, dim=64, filters=64, kernels=(2, 3, 4, 5), hidden=64):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, dim, padding_idx=0)
        self.cnn = MultiKernelCNN(dim, filters, kernels)
        self.lstm = nn.LSTM(dim, hidden, batch_first=True, bidirectional=True)
        width = self.cnn.out_dim + 4 * hidden
        self.head = nn.Sequential(nn.Linear(width, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))

    def prepare(self, labels):
        return (encode(labels),)

    def forward(self, chars):
        mask = mask_of(chars)
        e = self.emb(chars)
        h, _ = run_rnn(self.lstm, e, mask)
        fused = torch.cat([self.cnn(e, mask), masked_max(h, mask), masked_mean(h, mask)], -1)
        return self.head(fused).squeeze(-1)
