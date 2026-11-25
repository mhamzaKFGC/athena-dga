"""Khaos (Yun, Huang, et al., IEEE Transactions on Information Forensics and Security, 2020).

"Khaos: An Adversarial Neural Network DGA With High Anti-Detection Ability."

Khaos builds domain names from the n-grams of real domains using a Wasserstein GAN with gradient
penalty (WGAN-GP). The paper's four components:
  1. n-gram Generator  - a WGAN-GP that emits sequences of n-gram tokens.
  2. n-gram Tokenizer   - mines frequent n-grams from real domains and segments domains into them.
  3. Domain Synthesizer - concatenates the generated n-gram tokens into a domain string.
  4. Domain Filter      - keeps only valid, novel names.
The premise is that real domains are built from readable syllables/acronyms, so recombining their
frequent n-grams yields names that look benign to a lexical detector.

Fidelity notes: the IEEE TIFS full text is not open-access, so exact layer sizes are not published.
This follows the described 4-component pipeline with a standard char/token WGAN-GP (Gulrajani et al.,
2017: 1-D residual-conv generator and critic, gradient penalty lambda=10, n_critic=5). Token vocab =
the most frequent n-grams of the Majestic-200k labels. Sizes are CPU-scaled. Treat Khaos's exact
numbers as an approximation; the mechanism (n-gram WGAN-GP) matches the paper.
"""
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from common_gen import collect, legit_labels

HERE = Path(__file__).resolve().parent
WEIGHTS = HERE / "khaos_wgan.pt"
SEQ = 8             # max n-gram tokens per domain
DIM = 128          # conv channels
NOISE = 128
LAMBDA = 10.0      # gradient-penalty weight (WGAN-GP)
N_CRITIC = 5       # critic steps per generator step


# --------------------------------------------------------------------- tokenizer (component 2)
def build_vocab(labels, top_k=500, ns=(2, 3)):
    """Most frequent n-grams across the corpus, plus single chars (fallback) and PAD."""
    cnt = Counter()
    for s in labels:
        for n in ns:
            for i in range(len(s) - n + 1):
                cnt[s[i:i + n]] += 1
    grams = [g for g, _ in cnt.most_common(top_k)]
    singles = sorted({c for s in labels for c in s})
    vocab = ["<pad>"] + singles + grams                 # id 0 = pad
    return vocab


def segment(label, tok2id, vocab):
    """Greedy longest-match segmentation of `label` into n-gram token ids."""
    ids, i = [], 0
    while i < len(label) and len(ids) < SEQ:
        for n in (3, 2):                                # prefer longer n-grams
            if label[i:i + n] in tok2id:
                ids.append(tok2id[label[i:i + n]])
                i += n
                break
        else:
            if label[i] in tok2id:
                ids.append(tok2id[label[i]])
            i += 1
    ids = ids[:SEQ] + [0] * (SEQ - len(ids))
    return ids


# --------------------------------------------------------------------- WGAN-GP (component 1)
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.c1 = nn.Conv1d(dim, dim, 3, padding=1)
        self.c2 = nn.Conv1d(dim, dim, 3, padding=1)

    def forward(self, x):
        h = self.c2(F.relu(self.c1(F.relu(x))))
        return x + 0.3 * h


class Generator(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.fc = nn.Linear(NOISE, DIM * SEQ)
        self.blocks = nn.Sequential(*[ResBlock(DIM) for _ in range(3)])
        self.out = nn.Conv1d(DIM, vocab, 1)

    def forward(self, z):
        h = self.fc(z).view(-1, DIM, SEQ)
        h = self.out(self.blocks(h))                    # (B, vocab, SEQ)
        return F.softmax(h.transpose(1, 2), dim=-1)     # (B, SEQ, vocab)


class Critic(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.inp = nn.Conv1d(vocab, DIM, 1)
        self.blocks = nn.Sequential(*[ResBlock(DIM) for _ in range(3)])
        self.fc = nn.Linear(DIM * SEQ, 1)

    def forward(self, x):                               # x: (B, SEQ, vocab)
        h = self.blocks(self.inp(x.transpose(1, 2)))
        return self.fc(h.flatten(1))


def gradient_penalty(critic, real, fake):
    eps = torch.rand(real.size(0), 1, 1)
    mix = (eps * real + (1 - eps) * fake).requires_grad_(True)
    d = critic(mix)
    g = torch.autograd.grad(d, mix, torch.ones_like(d), create_graph=True)[0]
    return ((g.flatten(1).norm(2, dim=1) - 1) ** 2).mean()


def train(iters=1500, batch=128, lr=1e-4, seed=0, quick=False):
    torch.manual_seed(seed)
    if quick:
        iters = 60
    labels = legit_labels()[:60000]
    vocab = build_vocab(labels)
    tok2id = {t: i for i, t in enumerate(vocab)}
    V = len(vocab)
    seqs = torch.tensor([segment(s, tok2id, vocab) for s in labels])
    real_oh = F.one_hot(seqs, V).float()

    gen, crit = Generator(V), Critic(V)
    g_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.9))
    c_opt = torch.optim.Adam(crit.parameters(), lr=lr, betas=(0.5, 0.9))

    for it in range(iters):
        for _ in range(N_CRITIC):
            real = real_oh[torch.randint(0, len(real_oh), (batch,))]
            fake = gen(torch.randn(batch, NOISE)).detach()
            c_opt.zero_grad()
            c_loss = crit(fake).mean() - crit(real).mean() + LAMBDA * gradient_penalty(crit, real, fake)
            c_loss.backward()
            c_opt.step()
        g_opt.zero_grad()
        g_loss = -crit(gen(torch.randn(batch, NOISE))).mean()
        g_loss.backward()
        g_opt.step()
        if (it + 1) % max(1, iters // 10) == 0:
            print(f"  [Khaos WGAN-GP] iter {it + 1}/{iters}  c_loss {c_loss.item():.3f}  g_loss {g_loss.item():.3f}", flush=True)

    torch.save({"gen": gen.state_dict(), "vocab": vocab}, WEIGHTS)
    print(f"  saved {WEIGHTS}", flush=True)


# --------------------------------------------------------------------- synthesize (components 3 & 4)
def _load():
    ck = torch.load(WEIGHTS, map_location="cpu")
    vocab = ck["vocab"]
    gen = Generator(len(vocab))
    gen.load_state_dict(ck["gen"])
    gen.eval()
    return gen, vocab


def generate(n=1000, seed=2026, quick=False):
    if not WEIGHTS.exists():
        raise FileNotFoundError("Khaos weights missing - run train_all.py first")
    gen, vocab = _load()
    g = torch.Generator().manual_seed(seed)

    @torch.no_grad()
    def step(rng):
        probs = gen(torch.randn(1, NOISE, generator=g))[0]       # (SEQ, vocab)
        toks = torch.multinomial(probs, 1).squeeze(1).tolist()   # sample a token per slot
        return "".join(vocab[t] for t in toks if t != 0)         # drop PAD, concatenate n-grams

    return collect(step, n, seed, max_tries=n * 60)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train(quick="--quick" in sys.argv)
    else:
        k = int(sys.argv[1]) if len(sys.argv) > 1 else 20
        for name in generate(k):
            print(name)
