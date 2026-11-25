"""MaskDGA (Sidi, Nadler, Shabtai, 2019 / arXiv:1902.08909; IEEE Access 2020).

"MaskDGA: An Evasion Attack Against DGA Classifiers and Adversarial Defenses."

MaskDGA is a PERTURBATION attack, not a from-scratch generator: it takes an existing DGA name and
rewrites about half of its characters so a detector scores it as benign. It is crafted on a local
SUBSTITUTE model (kept separate from the targets), which makes it black-box to the real classifiers.

Substitute (paper Sec. 5.3): one-hot character input -> 1D CNN with filter widths 2,3,4,5 ->
max-pooling that halves the length -> two fully-connected layers -> class output. Trained 5 epochs,
batch 128, Adam, lr 0.01.

Attack (paper Sec. 4.4-4.5, Algorithm 1):
  1. One-hot the domain X. Feed forward, take the benign-class score.
  2. Backpropagate to the input -> Jacobian saliency map S[i,j] (sensitivity of benignness to putting
     symbol j at position i).
  3. For each position i, s*_i = max_j S[i,j]; threshold Theta = median of {s*_i}.
  4. For every position with s*_i >= Theta, replace its character with argmax_j S[i,j].
     => exactly half the characters change, each position at most once, and never to a dot
        (here the label alphabet has no dot, so that constraint is automatic).

Sign convention: we differentiate the benign-class LOGIT (higher = more benign), so argmax_j S[i,j]
is the symbol that most increases benignness at position i -- the change the attack wants.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from common_gen import ALPHABET, collect, label_of, legit_labels

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = Path(__file__).resolve().parent / "maskdga_substitute.pt"
VOCAB = len(ALPHABET)
CHAR2ID = {c: i for i, c in enumerate(ALPHABET)}
MAXLEN = 40


def onehot(label, maxlen=MAXLEN):
    x = torch.zeros(maxlen, VOCAB)
    for i, c in enumerate(label[:maxlen]):
        x[i, CHAR2ID[c]] = 1.0
    return x


class Substitute(nn.Module):
    """1D-CNN substitute detector (binary: benign vs DGA)."""

    def __init__(self, widths=(2, 3, 4, 5), filters=128, fc=128, maxlen=MAXLEN):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(VOCAB, filters, w, padding=w // 2) for w in widths])
        self.pool = nn.MaxPool1d(2)                         # halves the domain length
        flat = filters * len(widths) * (maxlen // 2)
        self.fc1 = nn.Linear(flat, fc)
        self.fc2 = nn.Linear(fc, 2)

    def forward(self, x):                                   # x: (B, maxlen, VOCAB)
        h = x.transpose(1, 2)                               # (B, VOCAB, maxlen)
        h = torch.cat([self.pool(F.relu(c(h)))[..., :MAXLEN // 2] for c in self.convs], dim=1)
        h = h.flatten(1)
        return self.fc2(F.relu(self.fc1(h)))                # logits (B, 2); class 0 = benign, 1 = DGA


# --------------------------------------------------------------------- data
def _dga_labels(limit=60000):
    """DGA labels from the UMUDGA family lists (full domains -> leftmost label)."""
    out, seen = [], set()
    folder = ROOT / "dga_classifier" / "data" / "umudga"
    for p in sorted(folder.glob("*.txt")):
        if p.stem == "legit":
            continue
        for line in p.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
            s = label_of(line)
            if 3 <= len(s) <= MAXLEN and all(c in ALPHABET for c in s) and s not in seen:
                seen.add(s)
                out.append(s)
    import random
    random.Random(0).shuffle(out)
    return out[:limit]


# --------------------------------------------------------------------- training
def train_substitute(epochs=5, batch=128, lr=0.01, n_each=30000, seed=0, quick=False):
    torch.manual_seed(seed)
    if quick:
        epochs, n_each = 1, 2000
    legit = legit_labels()[:n_each]
    dga = _dga_labels()[:n_each]
    labels = legit + dga
    y = torch.tensor([0] * len(legit) + [1] * len(dga))
    X = torch.stack([onehot(s) for s in labels])
    model = Substitute()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    idx = torch.randperm(len(X))
    for ep in range(epochs):
        idx = idx[torch.randperm(len(idx))]
        tot = 0.0
        for b in range(0, len(idx), batch):
            j = idx[b:b + batch]
            opt.zero_grad()
            loss = F.cross_entropy(model(X[j]), y[j])
            loss.backward()
            opt.step()
            tot += loss.item() * len(j)
        print(f"  [MaskDGA substitute] epoch {ep + 1}/{epochs}  loss {tot / len(idx):.4f}", flush=True)
    torch.save(model.state_dict(), WEIGHTS)
    print(f"  saved {WEIGHTS}", flush=True)
    return model


def load_substitute():
    model = Substitute()
    if not WEIGHTS.exists():
        raise FileNotFoundError(f"{WEIGHTS} missing - run train_all.py (or train_substitute()) first")
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    model.eval()
    return model


# --------------------------------------------------------------------- attack
def mask_perturb(label, model):
    """Rewrite ~half of `label` toward the benign class, per the JSM algorithm."""
    L = min(len(label), MAXLEN)
    x = onehot(label).clone().requires_grad_(True)
    logit_benign = model(x.unsqueeze(0))[0, 0]              # benign-class logit
    model.zero_grad()
    logit_benign.backward()
    S = x.grad[:L]                                          # (L, VOCAB) saliency
    best_sym = S.argmax(dim=1)                              # argmax_j per position
    best_val = S.max(dim=1).values
    theta = best_val.median()
    chars = list(label[:L])
    for i in range(L):
        if best_val[i] >= theta:
            c = ALPHABET[best_sym[i].item()]
            if c != "-" or (0 < i < L - 1):                 # never leading/trailing hyphen (dot N/A)
                chars[i] = c
    return "".join(chars)


def generate(n=1000, seed=2026, quick=False):
    """n evasive names: sample DGA labels, perturb each with the substitute's saliency map."""
    model = load_substitute()
    dga = _dga_labels()
    import random
    rng0 = random.Random(seed)
    pool = dga[:]
    rng0.shuffle(pool)
    it = iter(pool)

    def step(rng):
        try:
            base = next(it)
        except StopIteration:
            return None
        return mask_perturb(base, model)

    return collect(step, n, seed, max_tries=len(pool))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train_substitute(quick="--quick" in sys.argv)
    else:
        n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
        for name in generate(n):
            print(name)
