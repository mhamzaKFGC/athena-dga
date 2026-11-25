"""Train v1 and v2 (definitions reused from benchmark.py) and print names generated from the same seed and prefixes."""
from pathlib import Path

HERE = Path(__file__).parent
exec(compile((HERE / "benchmark.py").read_text(encoding="utf-8").split("rows = {}")[0], "benchmark_defs", "exec"))

PREFIXES, PER_PREFIX = ["abc", "sun", "tec"], 8
models = {}
for tag, ctor in (("v1", V1), ("v2", V2)):
    torch.manual_seed(SEED)
    m = ctor()
    fit(m, tag)
    models[tag] = m

results = {}
for tag, m in models.items():
    for p in PREFIXES:
        torch.manual_seed(SEED)  # identical random seed for both models
        out, seen = [], set()
        for n in generate(m, [p] * 300):
            if (n in seen or n in known or not valid.fullmatch(n) or len(n) < 5
                    or one_edit_neighbors(n) & known):
                continue
            seen.add(n)
            out.append(n)
            if len(out) == PER_PREFIX:
                break
        results[(tag, p)] = out

print("\n| prefix | # | v1 | v2 |\n|---|---|---|---|")
for p in PREFIXES:
    for i in range(PER_PREFIX):
        a = results[("v1", p)]
        b = results[("v2", p)]
        print(f"| {p} | {i + 1} | {a[i] + '.test' if i < len(a) else '-'} | {b[i] + '.test' if i < len(b) else '-'} |")
