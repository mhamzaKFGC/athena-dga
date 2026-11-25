# %% [markdown]
# # Athena: character-graph domain name generator
#
# Trains a character-level graph neural network on legitimate domain labels and samples new labels
# from it. Each name is a chain graph: one node per character, one edge from each position to the
# next. Three message-passing layers give the final node a four-character receptive field, and the
# readout predicts the next character.
#
# Generated names are kept only if they are absent from the training corpus, are not a one-edit
# neighbour of any corpus label, and fit the length limit. Requires Python 3.9+ and PyTorch.

# %%
import csv
import random
import re
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

# %% [markdown]
# ## 1. Data and hyperparameters
# `input_file` is a UTF-8 text file with one label or two-part domain per line; the suffix is
# discarded, so `abc.com` becomes `abc`. Setting it to `None` substitutes a small built-in corpus,
# which is only useful for checking that the pipeline runs. All reported results use the
# Majestic top-200k list.

# %%
input_file = Path("data/majestic/domains_200k.txt")   # run from the AthenaExp folder; None = toy demo
batch_size = 256
prefix = "abc"
count = 20
epochs = 10           # ~700 batches per epoch on 200k labels (25 for the 10k list, ~300 for the toy demo)
width = 128           # feature size per character node
n_layers = 3          # each layer lets a node see one more previous position
temperature = 0.8     # lower = safer names, higher = more variety
top_k = 8             # sample only from the k most likely next characters
max_len = 12          # longest generated label; longer samples are rejected, not cut off
seed = 464393         # randomly chosen; controls split, init, training and sampling
tlds = ["com", "net", "org", "io", "co", "app", "dev", "xyz", "online", "site", "info", "ai"]  # cycled across names; use ["test"] for the safe demo
output_file = Path("athena_synthetic.csv")

if input_file is None:
    first = ["amber", "cedar", "lotus", "olive", "maple", "ravi", "indus", "lunar"]
    last = ["lab", "field", "brook", "leaf", "hill", "spring", "grove", "vale"]
    lines = [a + b + ".test" for a in first for b in last]
else:
    if Path(input_file).resolve() == output_file.resolve():
        raise ValueError("Choose a separate output path.")
    lines = Path(input_file).read_text(encoding="utf-8-sig").splitlines()

# %% [markdown]
# ## 2. Clean the labels and split the data

# %%
alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-"
start, stop = len(alphabet), len(alphabet) + 1
valid = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")


prefix = prefix.lower()
if not re.fullmatch(r"(?:[a-z0-9][a-z0-9-]{0,61})?", prefix):
    raise ValueError("Enter a valid prefix up to 62 characters.")
if count < 1 or epochs < 1:
    raise ValueError("Count and epochs must be positive.")
known = set()
for line in lines:
    parts = line.strip().lower().removesuffix(".").split(".")
    if len(parts) <= 2 and all(valid.fullmatch(p) for p in parts):
        known.add(parts[0])
if len(known) < 10:
    raise ValueError("Supply at least 10 distinct labels for this demo split.")
labels = sorted(known)
random.Random(seed).shuffle(labels)
hold = max(1, len(labels)//10)
test, validation, train = labels[:hold], labels[hold:2*hold], labels[2*hold:]
print(f"Training: {len(train)}, validation: {len(validation)}, test: {len(test)}")

# %% [markdown]
# ## 3. Model
# Each layer concatenates a node's state with the state of the preceding position and projects the
# pair back to `width`. Stacking `n_layers` of these widens the receptive field by one character per
# layer, so the readout at position t predicts token t+1 from the preceding `n_layers` characters.

# %%
class CharacterGraph(nn.Module):
    """Each character position receives a message from the previous position."""
    def __init__(self, width=32, n_layers=2):
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

def batch_loss(model, labels):
    rows = [[start] + [alphabet.index(c) for c in label] + [stop] for label in labels]
    length = max(map(len, rows)) - 1
    inputs = torch.full((len(rows), length), start)
    targets = torch.full_like(inputs, -100)
    for i, row in enumerate(rows):
        inputs[i, :len(row)-1] = torch.tensor(row[:-1])
        targets[i, :len(row)-1] = torch.tensor(row[1:])
    logits = model(inputs)
    return F.cross_entropy(logits.transpose(1, 2), targets)

# %% [markdown]
# ## 4. Training
# Adam on next-character cross-entropy. The checkpoint with the lowest validation loss is kept, and
# held-out loss is reported once, after model selection.

# %%
torch.manual_seed(seed)
rng = random.Random(seed)
train = list(labels[2*hold:])
model = CharacterGraph(width, n_layers)
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
@torch.no_grad()
def evaluate(split):
    total = tokens = 0
    for k in range(0, len(split), batch_size):
        batch = split[k:k+batch_size]
        size = sum(len(s)+1 for s in batch)
        total += batch_loss(model, batch).item() * size
        tokens += size
    return total / tokens
best = float("inf")
for epoch in range(epochs):
    rng.shuffle(train)
    for k in range(0, len(train), batch_size):
        optimizer.zero_grad()
        batch_loss(model, train[k:k+batch_size]).backward()
        optimizer.step()
    score = evaluate(validation)
    print(f"Epoch {epoch+1}: validation loss {score:.3f}")
    if score < best:
        best = score
        checkpoint = {k:v.detach().clone() for k,v in model.state_dict().items()}
model.load_state_dict(checkpoint)
torch.save(checkpoint, "athena_v1_weights.pt")
print("Held-out loss per token:", evaluate(test))

print("Trainable parameters:", sum(p.numel() for p in model.parameters()))

# %% [markdown]
# ## 5. Sampling
# The sampler extends the prefix one character at a time, restricted to the `top_k` most likely
# continuations and scaled by `temperature`. Corpus labels, duplicates and one-edit neighbours of a
# corpus label are rejected, so a run can return fewer names than requested once the attempt limit
# is reached. Rerunning overwrites the output CSV.

# %%
def near_known(label, known):
    """True if label is exactly one insertion, deletion or substitution from a known label."""
    for i in range(len(label) + 1):
        for c in alphabet:
            if label[:i] + c + label[i:] in known:
                return True
            if i < len(label) and label[:i] + c + label[i + 1:] in known and c != label[i]:
                return True
        if i < len(label) and label[:i] + label[i + 1:] in known:
            return True
    return False


@torch.no_grad()
def generate(model, prefix, known, count, limit):
    output = {}
    for _ in range(max(100, count * 100)):
        label = prefix
        while len(label) < limit:
            tail = ([start] + [alphabet.index(c) for c in label])[-(n_layers + 1):]
            scores = model(torch.tensor([tail]))[0, -1]
            scores[start] = -torch.inf
            if label == prefix or label.endswith("-"):
                scores[stop] = -torch.inf
            if not label or len(label) == limit - 1:
                scores[alphabet.index("-")] = -torch.inf
            cutoff = scores.topk(min(top_k, scores.numel())).values[-1]
            scores[scores < cutoff] = -torch.inf
            token = int(torch.multinomial((scores / temperature).softmax(0), 1).item())
            if token == stop:
                break
            label += alphabet[token]
        if (len(label) <= max_len and label not in known and valid.fullmatch(label)
                and not near_known(label, known)):
            output[label] = None
        if len(output) == count:
            break
    return output


model.eval()
torch.manual_seed(seed)
if max_len < len(prefix) + 2:
    raise ValueError("max_len must be at least the prefix length plus 2.")
limit = max_len + 1  # a sample that reaches this length never emitted a stop token, so it is rejected
output = generate(model, prefix, known, count, limit)
with output_file.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream)
    writer.writerow(["domain", "label"])
    writer.writerows((f"{name}.{tlds[i % len(tlds)]}", "synthetic") for i, name in enumerate(output))
print(f"Saved {len(output)} of {count} requested names to {output_file.resolve()}")
for name in list(output)[:10]:
    print(f"{name}.{tlds[list(output).index(name) % len(tlds)]}")

# %% [markdown]
# ## Notes
# The reported loss is next-character cross-entropy per token; novelty is measured against the
# supplied corpus only, so a wider corpus gives a stricter filter. Training cost per epoch scales
# with the number of padded character positions. When varying the training set size, fix the
# validation and test splits first so the loss stays comparable across runs.
