"""Building blocks shared by the neural models."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def mask_of(chars):
    return chars != 0


def run_rnn(rnn, x, mask):
    """Run an (Bi)LSTM over right-padded input so padding never enters the recurrence."""
    lengths = mask.sum(1).clamp(min=1).cpu()
    packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
    out, (h, _) = rnn(packed)
    out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
    return out, h


def masked_max(h, mask):
    return h.masked_fill(~mask.unsqueeze(-1), -1e4).max(1).values


def masked_mean(h, mask):
    m = mask.unsqueeze(-1).float()
    return (h * m).sum(1) / m.sum(1).clamp(min=1)


class AdditiveAttention(nn.Module):
    """Bahdanau-style attention pooling over time steps."""

    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.score = nn.Linear(dim, 1, bias=False)

    def forward(self, h, mask):
        s = self.score(torch.tanh(self.proj(h))).squeeze(-1).masked_fill(~mask, -1e4)
        w = s.softmax(-1)
        self.last_weights = w.detach()  # kept for XAI (attention maps)
        return (w.unsqueeze(-1) * h).sum(1)


class MultiKernelCNN(nn.Module):
    """Parallel 1-D convolutions with different widths, each max-pooled over valid positions."""

    def __init__(self, dim, filters, kernels):
        super().__init__()
        self.convs = nn.ModuleList(nn.Conv1d(dim, filters, k, padding="same") for k in kernels)
        self.out_dim = filters * len(kernels)

    def forward(self, x, mask):  # x (B, L, D)
        x = x.transpose(1, 2)
        return torch.cat([masked_max(torch.relu(c(x)).transpose(1, 2), mask) for c in self.convs], -1)
