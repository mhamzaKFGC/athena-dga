"""Load the saved models (no training) and return scoring functions: labels -> P(DGA)."""
import importlib.util
import json
import pickle
import warnings
from pathlib import Path

import torch

from .features import lexical_features, ngram_kl_jaccard_features
from .train_eval import predict_torch

warnings.filterwarnings("ignore", message="Using padding='same'")

ROOT = Path(__file__).resolve().parent.parent
TORCH_MODELS = {"01_n-CBDC": ("NCBDC", {}), "03_ATT-CNN-BiLSTM": ("ATTCNNBiLSTM", {}),
                "05_LA-LSTM-Attention": ("LALSTM", {"n_out": 1}), "06_Integrated-DL": ("IntegratedDL", {}),
                "07_HDDN": ("HDDN", {})}
FEATURE_MODELS = {"02_MaldomDetector": lexical_features, "04_NGram-KL-Jaccard": ngram_kl_jaccard_features}


def model_folders():
    return sorted(p for p in ROOT.iterdir() if p.is_dir() and p.name[:2].isdigit())


def load_torch_model(folder):
    weights = folder / "weights.pt"
    if folder.name not in TORCH_MODELS or not weights.exists():
        return None
    spec = importlib.util.spec_from_file_location(f"model_{folder.name[:2]}", folder / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls, kwargs = TORCH_MODELS[folder.name]
    model = getattr(module, cls)(**kwargs)
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.eval()
    return model


def load_feature_model(folder):
    """(classifier, n-gram reference, feature function) for models 02/04, or None."""
    path = folder / "model.pkl"
    if folder.name not in FEATURE_MODELS or not path.exists():
        return None
    with open(path, "rb") as f:
        saved = pickle.load(f)
    return saved["clf"], saved["ref"], FEATURE_MODELS[folder.name]


def load_scorer(folder):
    """Scoring function for one trained model folder, or None if it has no saved model."""
    model = load_torch_model(folder)
    if model is not None:
        return lambda labels: predict_torch(model, labels)
    fm = load_feature_model(folder)
    if fm is not None:
        clf, ref, fn = fm
        return lambda labels: clf.predict_proba(fn(labels, ref))[:, 1]
    return None


def model_info(folder):
    """Name and whether the saved model came from a --quick smoke test."""
    res = folder / "results.json"
    if not res.exists():
        return {"name": folder.name[3:], "quick": None}
    r = json.loads(res.read_text(encoding="utf-8"))
    return {"name": r.get("model", folder.name[3:]), "quick": bool(r.get("settings", {}).get("quick"))}


def load_all(only=None):
    """{folder_name: (display name, scorer, info)} for every folder with a saved model."""
    out = {}
    for folder in model_folders():
        if only and folder.name[:2] not in only:
            continue
        scorer = load_scorer(folder)
        if scorer is not None:
            info = model_info(folder)
            out[folder.name] = (info["name"], scorer, info)
    return out
