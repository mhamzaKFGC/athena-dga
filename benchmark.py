"""Benchmark Athena v1 (previous-node tanh) vs v2 (dilated gated residual + prefix node)
on the same real-domain split. All local. Run: python benchmark.py"""
import random, re, time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).parent
EPOCHS, BATCH, WIDTH, SEED = 25, 128, 64, 42
N_GEN, TEMP, TOPK, PREFIX_LEN = 1000, 0.8, 8, 3

alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-"
start, stop = len(alphabet), len(alphabet) + 1
VOCAB = stop + 1
valid = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")
idx = {c: i for i, c in enumerate(alphabet)}

# ---------- data ----------
known = set()
for line in (HERE / "data" / "opendns-top-domains.txt").read_text(encoding="utf-8").splitlines():
    p = line.strip().lower().split(".")[0]
    if valid.fullmatch(p) and 3 <= len(p) <= 24:
        known.add(p)
labels = sorted(known)
random.Random(SEED).shuffle(labels)
h = len(labels) // 10
test, val, train = labels[:h], labels[h:2 * h], labels[2 * h:]
print(f"unique labels {len(labels)} | train {len(train)} val {len(val)} test {len(test)}")


def encode(batch):
    rows = [[start] + [idx[c] for c in s] + [stop] for s in batch]
    L = max(map(len, rows)) - 1
    x = torch.full((len(rows), L), start)
    y = torch.full_like(x, -100)
    for i, r in enumerate(rows):
        x[i, :len(r) - 1] = torch.tensor(r[:-1])
        y[i, :len(r) - 1] = torch.tensor(r[1:])
    return x, y


# ---------- models ----------
class V1(nn.Module):
    def __init__(self, w=WIDTH, n=3):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, w)
        self.layers = nn.ModuleList(nn.Linear(2 * w, w) for _ in range(n))
        self.out = nn.Linear(w, VOCAB)

    def forward(self, x):
        s = self.emb(x)
        for l in self.layers:
            prev = F.pad(s[:, :-1], (0, 0, 1, 0))
            s = l(torch.cat((s, prev), -1)).tanh()
        return self.out(s)


class V2(nn.Module):
    def __init__(self, w=WIDTH, dil=(1, 2, 4)):
        super().__init__()
        self.dil = dil
        self.emb = nn.Embedding(VOCAB, w)
        self.upd = nn.ModuleList(nn.Linear(3 * w, w) for _ in dil)
        self.gate = nn.ModuleList(nn.Linear(3 * w, w) for _ in dil)
        self.norm = nn.ModuleList(nn.LayerNorm(w) for _ in dil)
        self.out = nn.Linear(w, VOCAB)

    def forward(self, x):
        h = self.emb(x)
        L = x.shape[1]
        # prefix node = mean embedding of the first PREFIX_LEN chars, visible only after they were read
        p = self.emb(x[:, 1:1 + PREFIX_LEN]).mean(1)[:, None, :]
        p = p * (torch.arange(L) >= PREFIX_LEN).float()[None, :, None]
        for d, upd, gate, norm in zip(self.dil, self.upd, self.gate, self.norm):
            prev = torch.zeros_like(h)
            if d < L:
                prev[:, d:] = h[:, :-d]
            m = torch.cat((h, prev, p.expand_as(h)), -1)
            h = norm(h + torch.sigmoid(gate(m)) * torch.tanh(upd(m)))
        return self.out(h)


# ---------- train / eval ----------
def loss_fn(model, batch, after_prefix=False):
    x, y = encode(batch)
    logits = model(x)
    if after_prefix:  # only score tokens the model has to generate after the prefix
        y = y.clone()
        y[:, :PREFIX_LEN] = -100
    return F.cross_entropy(logits.transpose(1, 2), y, reduction="sum").item(), int((y != -100).sum())


@torch.no_grad()
def evaluate(model, split, after_prefix=True):
    model.eval()
    tot = n = 0
    for k in range(0, len(split), 512):
        a, b = loss_fn(model, split[k:k + 512], after_prefix)
        tot += a
        n += b
    return tot / n


def fit(model, tag):
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
    steps = EPOCHS * ((len(train) + BATCH - 1) // BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps)
    data = list(train)
    best, ckpt, t0 = 1e9, None, time.time()
    for ep in range(EPOCHS):
        model.train()
        rng.shuffle(data)
        for k in range(0, len(data), BATCH):
            x, y = encode(data[k:k + BATCH])
            opt.zero_grad()
            F.cross_entropy(model(x).transpose(1, 2), y).backward()
            opt.step()
            sched.step()
        v = evaluate(model, val)
        if v < best:
            best, ckpt = v, {k: t.clone() for k, t in model.state_dict().items()}
    secs = time.time() - t0
    model.load_state_dict(ckpt)
    print(f"[{tag}] trained {secs:.1f}s | best val {best:.3f}")
    return secs


# ---------- generation + quality ----------
@torch.no_grad()
def generate(model, prefixes):
    model.eval()
    x = torch.tensor([[start] + [idx[c] for c in p] for p in prefixes])
    for _ in range(24 - PREFIX_LEN):
        lg = model(x)[:, -1].clone()
        lg[:, start] = -torch.inf
        kth = lg.topk(TOPK).values[:, -1:]
        lg = lg.masked_fill(lg < kth, -torch.inf)
        tok = torch.multinomial((lg / TEMP).softmax(-1), 1)
        x = torch.cat((x, tok), 1)
    names = []
    for row in x[:, 1:].tolist():
        s = ""
        for t in row:
            if t == stop:
                break
            if t < len(alphabet):
                s += alphabet[t]
        names.append(s)
    return names


def one_edit_neighbors(s):
    out = set()
    for i in range(len(s) + 1):
        for c in alphabet:
            out.add(s[:i] + c + s[i:])
            if i < len(s):
                out.add(s[:i] + c + s[i + 1:])
        if i < len(s):
            out.add(s[:i] + s[i + 1:])
    out.discard(s)
    return out


train_tri = {s[i:i + 3] for s in train for i in range(len(s) - 2)}


def quality(names):
    ok = [n for n in names if valid.fullmatch(n) and len(n) >= 4]
    uniq = set(ok)
    novel = {n for n in uniq if n not in known}
    wordlike = [n for n in novel if all(n[i:i + 3] in train_tri for i in range(len(n) - 2))]
    look = [n for n in novel if one_edit_neighbors(n) & known]
    safe = [n for n in wordlike if n not in set(look)]
    f = lambda a: 100 * len(a) / len(names)
    return dict(valid=f(ok), unique_valid=f(uniq), novel=f(novel), wordlike=f(wordlike),
                lookalike_of_known=f(look), usable=f(safe))


rows = {}
test_prefixes = [s[:PREFIX_LEN] for s in (test * (N_GEN // len(test) + 1))[:N_GEN]]
for tag, ctor in (("v1", V1), ("v2", V2)):
    torch.manual_seed(SEED)
    m = ctor()
    secs = fit(m, tag)
    params = sum(p.numel() for p in m.parameters())
    t0 = time.time()
    names = generate(m, test_prefixes)
    gsecs = time.time() - t0
    q = quality(names)
    rows[tag] = dict(params=params, train_s=secs, gen_s=gsecs,
                     test_loss=evaluate(m, test), **q, sample=names[:8])

print("\n=== RESULTS (same split, same seeds) ===")
keys = ["params", "train_s", "gen_s", "test_loss", "valid", "unique_valid", "novel",
        "wordlike", "lookalike_of_known", "usable"]
print(f"{'metric':22}{'v1':>12}{'v2':>12}")
for k in keys:
    print(f"{k:22}{rows['v1'][k]:>12.3f}{rows['v2'][k]:>12.3f}")
for t in rows:
    print(t, "samples:", rows[t]["sample"])
