"""Explainability (XAI) for all seven classifiers. No extra libraries needed.

Local, per character (works for EVERY model, neural or forest):
  char_shapley     Monte-Carlo Shapley values: each character's share of the score, found by removing
                   characters in random orders. Sum of values = score(name) - score(empty name).
Local, neural models only:
  integrated_gradients  gradients along a path from an all-zero embedding to the real one (Sundararajan
                   et al. 2017), summed per character position; works on the logit.
  attention_weights     the attention layer's weights (models 03 and 05).
Global:
  feature_shapley  Monte-Carlo Shapley values over the hand-made features of models 02 and 04.
  summarize        averages character attributions by character type, position and n-gram.
Faithfulness:
  deletion_test    remove the k most important characters vs k random ones and compare the score drop.
"""
import random
from collections import defaultdict

import numpy as np
import torch
from torch import nn

VOWELS, DIGITS = set("aeiou"), set("0123456789")


def char_type(c):
    return "digit" if c in DIGITS else "vowel" if c in VOWELS else "hyphen" if c in "-_" else "consonant"


# ------------------------------------------------------------------ model-agnostic Shapley
def char_shapley(score_fn, labels, n_perm=16, seed=0):
    """Returns (list of per-character arrays, base score of the empty name, full scores)."""
    rng = random.Random(seed)
    plans, needed = [], {""}
    for s in labels:
        perms = []
        for _ in range(n_perm):
            order = list(range(len(s)))
            rng.shuffle(order)
            keep, chain = set(), []
            for i in order:
                keep.add(i)
                chain.append("".join(s[j] for j in range(len(s)) if j in keep))
            perms.append((order, chain))
            needed.update(chain)
        plans.append(perms)
    strings = sorted(needed)
    values = dict(zip(strings, map(float, score_fn(strings))))
    base = values[""]
    out = []
    for s, perms in zip(labels, plans):
        phi = np.zeros(len(s))
        for order, chain in perms:
            prev = base
            for i, sub in zip(order, chain):
                phi[i] += values[sub] - prev
                prev = values[sub]
        out.append(phi / n_perm)
    return out, base, [values[s] for s in labels]


# ------------------------------------------------------------------ neural-only
def integrated_gradients(model, labels, steps=32):
    """Per-character IG attributions on the logit (baseline: all-zero embeddings)."""
    model.eval()
    embeddings = [m for m in model.modules() if isinstance(m, nn.Embedding)]
    state = {"alpha": 1.0, "outs": []}

    def hook(_, __, out):
        out = out * state["alpha"]
        out.retain_grad()
        state["outs"].append(out)
        return out

    handles = [e.register_forward_hook(hook) for e in embeddings]
    try:
        inputs = model.prepare(labels)
        length = max(int((inputs[0] != 0).sum(1).max()), 1)
        inputs = [t[:, :length].long() for t in inputs]
        total, full = None, None
        for k in range(1, steps + 1):
            state["alpha"], state["outs"] = k / steps, []
            model.zero_grad()
            logit = model(*inputs)
            logit.sum().backward()
            grads = [o.grad.detach() for o in state["outs"]]
            if total is None:
                total = [torch.zeros_like(g) for g in grads]
            for t, g in zip(total, grads):
                t += g
            if k == steps:
                full = [o.detach() for o in state["outs"]]  # alpha = 1 -> the real embeddings
        attr = None
        for x, g in zip(full, total):
            a = (x * g / steps).sum(-1)          # (B, L) or (B, L, n) for subword ids
            while a.dim() > 2:
                a = a.sum(-1)
            attr = a if attr is None else attr + a
        attr = attr.numpy()
    finally:
        for h in handles:
            h.remove()
    return [attr[i, :len(s)] for i, s in enumerate(labels)]


@torch.no_grad()
def attention_weights(model, labels):
    att = [m for m in model.modules() if m.__class__.__name__ == "AdditiveAttention"]
    if not att:
        return None
    model.eval()
    inputs = model.prepare(labels)
    length = max(int((inputs[0] != 0).sum(1).max()), 1)
    model(*[t[:, :length].long() for t in inputs])
    w = att[0].last_weights.numpy()
    return [w[i, :len(s)] for i, s in enumerate(labels)]


# ------------------------------------------------------------------ feature models
def feature_shapley(predict, X, background, n_perm=16, seed=0):
    """Monte-Carlo Shapley values over tabular features (missing feature = a background row's value)."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    phi = np.zeros((n, d))
    for _ in range(n_perm):
        order = rng.permutation(d)
        ref = background[rng.integers(0, len(background), n)]
        cur = ref.copy()
        prev = predict(cur)
        for j in order:
            cur[:, j] = X[:, j]
            now = predict(cur)
            phi[:, j] += now - prev
            prev = now
    return phi / n_perm


# ------------------------------------------------------------------ summaries
def summarize(labels, attributions, top=15):
    by_type, by_pos, grams = defaultdict(list), defaultdict(list), defaultdict(list)
    for s, a in zip(labels, attributions):
        for i, (c, v) in enumerate(zip(s, a)):
            by_type[char_type(c)].append(v)
            by_pos[min(int(i / max(len(s), 1) * 5), 4)].append(v)
        for n in (2, 3):
            for i in range(len(s) - n + 1):
                grams[s[i:i + n]].append(float(np.sum(a[i:i + n])))
    common = {g: np.mean(v) for g, v in grams.items() if len(v) >= 5}
    ranked = sorted(common.items(), key=lambda kv: kv[1])
    return {
        "mean_by_char_type": {k: float(np.mean(v)) for k, v in sorted(by_type.items())},
        "mean_by_position_fifth": {f"{k * 20}-{k * 20 + 20}%": float(np.mean(v)) for k, v in sorted(by_pos.items())},
        "ngrams_toward_dga": [[g, float(v)] for g, v in ranked[::-1][:top]],
        "ngrams_toward_legit": [[g, float(v)] for g, v in ranked[:top]],
    }


def deletion_test(score_fn, labels, attributions, k=2, seed=0):
    """Score drop after deleting the k highest-attributed characters vs k random characters."""
    rng = random.Random(seed)
    top, rand, keep = [], [], []
    for s, a in zip(labels, attributions):
        if len(s) <= k:
            continue
        idx = set(np.argsort(-np.asarray(a))[:k])
        top.append("".join(c for i, c in enumerate(s) if i not in idx))
        r = set(rng.sample(range(len(s)), k))
        rand.append("".join(c for i, c in enumerate(s) if i not in r))
        keep.append(s)
    if not keep:
        return None
    full, t, r = (np.asarray(score_fn(x), dtype=float) for x in (keep, top, rand))
    return {"n": len(keep), "k": k, "mean_drop_top": float((full - t).mean()),
            "mean_drop_random": float((full - r).mean())}


def rank_agreement(a, b):
    """Mean Spearman correlation between two attribution sets over the same names (length >= 4)."""
    vals = []
    for x, y in zip(a, b):
        if len(x) >= 4:
            rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
            c = np.corrcoef(rx, ry)[0, 1]
            if not np.isnan(c):
                vals.append(c)
    return float(np.mean(vals)) if vals else float("nan")
