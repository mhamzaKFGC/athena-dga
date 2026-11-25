"""DeepDGA (Anderson, Woodbridge, Filar, 2016 / arXiv:1610.01969).

"DeepDGA: Adversarially-Tuned Domain Generation and Detection."

A character autoencoder trained on legit domains, then repurposed as a GAN that turns a random seed
into an Alexa-like domain name. Verified architecture from the paper (Sec. III, V):

  Encoder : embedding dim d=20 -> 20 conv filters width 2 + 10 width 3 -> pooling (32-dim/timestep)
            -> highway layer (shared over time) -> LSTM -> final state = domain embedding.
  Decoder : embedding repeated over max length -> LSTM -> highway -> same conv widths
            -> time-distributed softmax, SAMPLED per character.
  Pretrain: Alexa 1M, 300 epochs, batch 128, Adam (here: Majestic-200k, fewer epochs, CPU).
  GAN     : freeze the autoencoder. Generator = dense(20 uniform noise -> embedding) + BOX LAYER
            (clamps to the min/max corner of real-domain embeddings). Discriminator = encoder +
            logistic regression on the embedding. Vanilla GAN loss; only the generator dense layer and
            the logistic layer are trained. HISTORY REGULARIZATION (train the discriminator on current
            + past generated batches) prevents the mode collapse the paper reports. 3 adversarial rounds.

Fidelity notes: the paper's exact max-over-time / max-over-filters pooling is only loosely specified;
here a per-timestep linear stands in for the "32 dimensions per time step" it describes. The GAN is
trained in embedding space (the paper "append[s] a logistic regression layer to the domain embedding"),
which avoids feeding soft outputs back through the embedding lookup. Sizes are CPU-scaled.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from common_gen import ALPHABET, collect, legit_labels

HERE = Path(__file__).resolve().parent
AE_WEIGHTS = HERE / "deepdga_ae.pt"
GEN_WEIGHTS = HERE / "deepdga_gan.pt"
STOP = len(ALPHABET)                                # explicit end/pad symbol the decoder learns to emit
VOCAB = len(ALPHABET) + 1
CHAR2ID = {c: i for i, c in enumerate(ALPHABET)}
MAXLEN = 25
EMB = 20            # character embedding dim d (paper: 20)
HID = 64            # domain-embedding / LSTM hidden size
NOISE = 20          # uniform seed dimension (paper: 20)


def encode_ids(label, maxlen=MAXLEN):
    ids = [CHAR2ID[c] for c in label[:maxlen] if c in CHAR2ID]
    ids += [STOP] * (maxlen - len(ids))             # pad with STOP so the model learns where names end
    return torch.tensor(ids)


class Highway(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.t = nn.Linear(dim, dim)
        self.h = nn.Linear(dim, dim)

    def forward(self, x):
        t = torch.sigmoid(self.t(x))
        return t * F.relu(self.h(x)) + (1 - t) * x


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, EMB)
        self.c2 = nn.Conv1d(EMB, 20, 2, padding=1)      # 20 bigram filters
        self.c3 = nn.Conv1d(EMB, 10, 3, padding=1)      # 10 trigram filters
        self.proj = nn.Linear(30, 32)                   # -> "32 dimensions for each time step"
        self.hw = Highway(32)
        self.lstm = nn.LSTM(32, HID, batch_first=True)

    def forward(self, ids):                             # ids: (B, L)
        e = self.emb(ids).transpose(1, 2)               # (B, EMB, L)
        h = torch.cat([self.c2(e)[..., :MAXLEN], self.c3(e)[..., :MAXLEN]], dim=1)  # (B,30,L)
        h = self.hw(self.proj(h.transpose(1, 2)))       # (B, L, 32)
        _, (hn, _) = self.lstm(h)
        return hn[-1]                                   # (B, HID) domain embedding


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(HID, HID, batch_first=True)
        self.hw = Highway(HID)
        self.out = nn.Linear(HID, VOCAB)

    def forward(self, z):                               # z: (B, HID)
        seq = z.unsqueeze(1).repeat(1, MAXLEN, 1)       # repeat embedding over time
        h, _ = self.lstm(seq)
        return self.out(self.hw(h))                     # (B, L, VOCAB) logits


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder()
        self.dec = Decoder()

    def forward(self, ids):
        return self.dec(self.enc(ids))


class BoxGenerator(nn.Module):
    """Dense(noise -> [0,1]^HID) then box layer y = a*vmax + (1-a)*vmin (paper Sec. III-B)."""

    def __init__(self, vmin, vmax):
        super().__init__()
        self.fc = nn.Linear(NOISE, HID)
        self.register_buffer("vmin", vmin)
        self.register_buffer("vmax", vmax)

    def forward(self, noise):
        a = torch.sigmoid(self.fc(noise))
        return a * self.vmax + (1 - a) * self.vmin


# --------------------------------------------------------------------- pretraining
def pretrain_ae(epochs=12, batch=128, lr=1e-3, n=120000, seed=0, quick=False):
    torch.manual_seed(seed)
    if quick:
        epochs, n = 1, 3000
    data = legit_labels()[:n]
    X = torch.stack([encode_ids(s) for s in data])
    ae = AutoEncoder()
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    for ep in range(epochs):
        idx = torch.randperm(len(X))
        tot = 0.0
        for b in range(0, len(idx), batch):
            j = idx[b:b + batch]
            opt.zero_grad()
            logits = ae(X[j])                            # (B, L, VOCAB)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), X[j].reshape(-1))
            loss.backward()
            opt.step()
            tot += loss.item() * len(j)
        print(f"  [DeepDGA AE] epoch {ep + 1}/{epochs}  recon-loss {tot / len(idx):.4f}", flush=True)
    torch.save(ae.state_dict(), AE_WEIGHTS)
    print(f"  saved {AE_WEIGHTS}", flush=True)
    return ae


# --------------------------------------------------------------------- GAN rounds
def train_gan(rounds=3, steps=200, batch=128, lr=1e-3, seed=0, quick=False):
    torch.manual_seed(seed)
    if quick:
        rounds, steps = 1, 20
    ae = AutoEncoder()
    ae.load_state_dict(torch.load(AE_WEIGHTS, map_location="cpu"))
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)

    data = legit_labels()[:60000]
    X = torch.stack([encode_ids(s) for s in data])
    with torch.no_grad():
        emb = torch.cat([ae.enc(X[b:b + 512]) for b in range(0, len(X), 512)])   # real embeddings
    vmin, vmax = emb.min(0).values, emb.max(0).values

    gen = BoxGenerator(vmin, vmax)
    disc = nn.Linear(HID, 1)                              # logistic on embedding; 1 = "generated"
    g_opt = torch.optim.Adam(gen.parameters(), lr=lr)
    d_opt = torch.optim.Adam(disc.parameters(), lr=lr)
    history = []                                          # past generated batches (history regularization)

    for rnd in range(rounds):
        for step in range(steps):
            real = emb[torch.randint(0, len(emb), (batch,))]
            noise = torch.rand(batch, NOISE)
            fake = gen(noise).detach()
            pool = torch.cat([fake] + history[-5:]) if history else fake   # current + past fakes
            # discriminator: real -> 0, generated -> 1
            d_opt.zero_grad()
            d_loss = (F.binary_cross_entropy_with_logits(disc(real).squeeze(1), torch.zeros(len(real)))
                      + F.binary_cross_entropy_with_logits(disc(pool).squeeze(1), torch.ones(len(pool))))
            d_loss.backward()
            d_opt.step()
            # generator: wants disc(fake) -> 0 (valid)
            g_opt.zero_grad()
            noise = torch.rand(batch, NOISE)
            g_loss = F.binary_cross_entropy_with_logits(disc(gen(noise)).squeeze(1), torch.zeros(batch))
            g_loss.backward()
            g_opt.step()
        history.append(gen(torch.rand(batch, NOISE)).detach())
        print(f"  [DeepDGA GAN] round {rnd + 1}/{rounds}  d_loss {d_loss.item():.3f}  g_loss {g_loss.item():.3f}", flush=True)

    torch.save({"gen": gen.state_dict(), "vmin": vmin, "vmax": vmax}, GEN_WEIGHTS)
    print(f"  saved {GEN_WEIGHTS}", flush=True)


# --------------------------------------------------------------------- generation
def _load():
    ae = AutoEncoder()
    ae.load_state_dict(torch.load(AE_WEIGHTS, map_location="cpu"))
    ae.eval()
    ck = torch.load(GEN_WEIGHTS, map_location="cpu")
    gen = BoxGenerator(ck["vmin"], ck["vmax"])
    gen.load_state_dict(ck["gen"])
    gen.eval()
    return ae, gen


@torch.no_grad()
def _decode_sample(ae, z, rng):
    logits = ae.dec(z)[0]                                 # (L, VOCAB)
    probs = F.softmax(logits, dim=-1)
    chars = []
    for t in range(MAXLEN):
        j = torch.multinomial(probs[t], 1).item()        # paper: numpy.random.multinomial sampling
        if j == STOP:
            break
        chars.append(ALPHABET[j])
    return "".join(chars).strip("-")


def generate(n=1000, seed=2026, quick=False):
    if not (AE_WEIGHTS.exists() and GEN_WEIGHTS.exists()):
        raise FileNotFoundError("DeepDGA weights missing - run train_all.py first")
    ae, gen = _load()
    g = torch.Generator().manual_seed(seed)

    def step(rng):
        z = gen(torch.rand(1, NOISE, generator=g))
        return _decode_sample(ae, z, rng)

    return collect(step, n, seed, max_tries=n * 60)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        q = "--quick" in sys.argv
        pretrain_ae(quick=q)
        train_gan(quick=q)
    else:
        k = int(sys.argv[1]) if len(sys.argv) > 1 else 20
        for name in generate(k):
            print(name)
