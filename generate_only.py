"""Generate names from the saved v1 weights without retraining.
Needs athena_v1_weights.pt (written by Athena_GNN.py) and the same data file for the novelty/lookalike filter.
Example: python generate_only.py --prefix sun --count 30 --seed 7
Use the same --width and --layers that the weights were trained with (defaults match Athena_GNN.py)."""
import argparse
import csv
import re
import secrets
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).parent
ap = argparse.ArgumentParser()
ap.add_argument("--prefix", default="abc")
ap.add_argument("--count", type=int, default=20)
ap.add_argument("--seed", type=int, default=None, help="default: random, printed so the run can be repeated")
ap.add_argument("--max-len", type=int, default=12)
ap.add_argument("--temperature", type=float, default=0.8)
ap.add_argument("--top-k", type=int, default=8)
ap.add_argument("--tlds", default="com,net,org,io,co,app,dev,xyz,online,site,info,ai")
ap.add_argument("--data", default=str(HERE / "data" / "majestic" / "domains_200k.txt"))
ap.add_argument("--weights", default=str(HERE / "athena_v1_weights.pt"))
ap.add_argument("--out", default=str(HERE / "athena_generated.csv"))
ap.add_argument("--width", type=int, default=128)
ap.add_argument("--layers", type=int, default=3)
a = ap.parse_args()

alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-"
start, stop = len(alphabet), len(alphabet) + 1
valid = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")
prefix = a.prefix.lower()
if not re.fullmatch(r"(?:[a-z0-9][a-z0-9-]{0,61})?", prefix):
    raise SystemExit("Enter a valid prefix.")
if a.max_len < len(prefix) + 2:
    raise SystemExit("--max-len must be at least the prefix length plus 2.")
seed = a.seed if a.seed is not None else secrets.randbelow(900000) + 100000
tlds = [t.strip().lower() for t in a.tlds.split(",") if t.strip()]
print(f"seed {seed} | prefix {prefix} | max_len {a.max_len}")

known = set()
for line in Path(a.data).read_text(encoding="utf-8-sig").splitlines():
    parts = line.strip().lower().removesuffix(".").split(".")
    if len(parts) <= 2 and all(valid.fullmatch(p) for p in parts):
        known.add(parts[0])


class CharacterGraph(nn.Module):
    def __init__(self, width, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(stop + 1, width)
        self.layers = nn.ModuleList(nn.Linear(2 * width, width) for _ in range(n_layers))
        self.readout = nn.Linear(width, stop + 1)

    def forward(self, tokens):
        states = self.embedding(tokens)
        for layer in self.layers:
            previous = F.pad(states[:, :-1], (0, 0, 1, 0))
            states = layer(torch.cat((states, previous), dim=-1)).tanh()
        return self.readout(states)


def near_known(label):
    for i in range(len(label) + 1):
        for c in alphabet:
            if label[:i] + c + label[i:] in known:
                return True
            if i < len(label) and c != label[i] and label[:i] + c + label[i + 1:] in known:
                return True
        if i < len(label) and label[:i] + label[i + 1:] in known:
            return True
    return False


model = CharacterGraph(a.width, a.layers)
model.load_state_dict(torch.load(a.weights, map_location="cpu"))
model.eval()
torch.manual_seed(seed)
limit = a.max_len + 1
output = {}
with torch.no_grad():
    for _ in range(max(100, a.count * 100)):
        label = prefix
        while len(label) < limit:
            tail = ([start] + [alphabet.index(c) for c in label])[-(a.layers + 1):]
            scores = model(torch.tensor([tail]))[0, -1]
            scores[start] = -torch.inf
            if label == prefix or label.endswith("-"):
                scores[stop] = -torch.inf
            if not label or len(label) == limit - 1:
                scores[alphabet.index("-")] = -torch.inf
            cutoff = scores.topk(min(a.top_k, scores.numel())).values[-1]
            scores[scores < cutoff] = -torch.inf
            token = int(torch.multinomial((scores / a.temperature).softmax(0), 1).item())
            if token == stop:
                break
            label += alphabet[token]
        if (len(label) <= a.max_len and label not in known and valid.fullmatch(label)
                and not near_known(label)):
            output[label] = None
        if len(output) == a.count:
            break

rows = [(f"{n}.{tlds[i % len(tlds)]}", "synthetic") for i, n in enumerate(output)]
with open(a.out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["domain", "label"])
    w.writerows(rows)
print(f"Saved {len(rows)} of {a.count} names to {a.out}")
for d, _ in rows:
    print(d)
