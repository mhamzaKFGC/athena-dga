"""Shared scaffolding for the four adversarial-DGA baselines (CharBot, DeepDGA, MaskDGA, Khaos).

Every generator exposes `generate(n, seed) -> list[str]` returning bare DNS labels (no TLD), so they
drop straight into the project's scorers and the Athena-vs-classifiers comparison. Names are filtered
to be *novel* the same way Athena's are: not an exact known legit label, and not one edit away from one
(near_known), so a baseline cannot "win" just by echoing real domains.
"""
import random
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # AthenaExp
LEGIT_FILE = ROOT / "data" / "majestic" / "domains_200k.txt"
ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789-"      # valid DNS label characters (same as Athena)
LETTERS = "abcdefghijklmnopqrstuvwxyz0123456789"        # no hyphen, for interior substitutions
TLDS = ["com", "net", "org", "info", "biz", "ru", "co.uk", "xyz", "online", "top"]


def label_of(domain):
    """Leftmost DNS label, lowercased (google.com -> google)."""
    parts = [p for p in domain.strip().lower().rstrip(".").split(".") if p]
    if parts and parts[0] == "www" and len(parts) > 2:
        parts = parts[1:]
    return parts[0] if parts else ""


@lru_cache(maxsize=4)
def legit_labels(limit=200000):
    """List of distinct legit labels from the Majestic top-200k, in rank order."""
    out, seen = [], set()
    for line in LEGIT_FILE.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        s = label_of(line)
        if 3 <= len(s) <= 63 and all(c in ALPHABET for c in s) and s not in seen:
            seen.add(s)
            out.append(s)
            if len(out) >= limit:
                break
    return out


@lru_cache(maxsize=1)
def known_set():
    """Set of all legit labels, for novelty / lookalike rejection."""
    return frozenset(legit_labels())


def near_known(label, known):
    """True if `label` is exactly one insertion, deletion or substitution from a known label.
    Same rule Athena uses, so baselines are held to the same novelty bar."""
    if label in known:
        return True
    for i in range(len(label) + 1):
        for c in ALPHABET:
            if label[:i] + c + label[i:] in known:                       # insertion
                return True
            if i < len(label) and c != label[i] and label[:i] + c + label[i + 1:] in known:  # substitution
                return True
        if i < len(label) and label[:i] + label[i + 1:] in known:        # deletion
            return True
    return False


def valid_label(s):
    """A syntactically valid DNS label: 3-63 chars, allowed alphabet, no leading/trailing hyphen."""
    return 3 <= len(s) <= 63 and all(c in ALPHABET for c in s) and not s.startswith("-") and not s.endswith("-")


def novel(s, known, extra=None):
    """Valid, not a known legit label or a 1-edit lookalike, and not already produced this run."""
    return valid_label(s) and not near_known(s, known) and (extra is None or s not in extra)


def collect(step, n, seed, max_tries=None):
    """Call `step(rng) -> candidate-or-None` until `n` novel distinct labels are gathered.
    `step` proposes one name; collect() enforces novelty and de-duplication."""
    rng = random.Random(seed)
    known = known_set()
    out, seen = [], set()
    tries = 0
    cap = max_tries or n * 200
    while len(out) < n and tries < cap:
        tries += 1
        s = step(rng)
        if s and novel(s, known, seen):
            seen.add(s)
            out.append(s)
    return out
