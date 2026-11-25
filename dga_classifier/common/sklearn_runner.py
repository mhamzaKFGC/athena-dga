"""Standard path for the two feature-based models (scikit-learn random forest)."""
import pickle
import time
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier

from .features import NgramReference
from .train_eval import binary_metrics, column, robustness, save_results, timed_predict


def run_feature_model(feature_fn, ds, a, folder, info, feature_names):
    # The n-gram reference is built from one half of the training rows and the forest is fitted on
    # the other half; otherwise every training name would look "already seen" and the model overfits.
    half = len(ds["train"]) // 2
    ref_rows, train = ds["train"][:half], ds["train"][half:]
    ref = NgramReference([r[0] for r in ref_rows if r[1] == 0], [r[0] for r in ref_rows if r[1] == 1])
    t0 = time.time()
    x = feature_fn(column(train, 0), ref)
    print(f"features for {len(train)} training rows in {time.time() - t0:.0f}s", flush=True)
    clf = RandomForestClassifier(n_estimators=40 if a.quick else (200 if ds.get("full") else 300),
                                 min_samples_leaf=5 if ds.get("full") else 2, n_jobs=-1, random_state=a.seed)
    clf.fit(x, column(train, 1))
    train_s = time.time() - t0
    predict = lambda labels: clf.predict_proba(feature_fn(labels, ref))[:, 1]
    scores, ms = timed_predict(predict, column(ds["test"], 0))
    info.update(params=sum(e.tree_.node_count for e in clf.estimators_), params_note="total tree nodes",
                train_seconds=round(train_s, 1), ms_per_1000=round(ms, 2),
                training_samples_seen=len(train), reference_rows=len(ref_rows),
                data="full UMUDGA" if ds.get("full") else "UMUDGA 10k lists",
                test=binary_metrics(scores, ds["test"]), robustness=robustness(predict, ds),
                feature_importance=dict(sorted(zip(feature_names, map(float, clf.feature_importances_)),
                                               key=lambda kv: -kv[1])),
                settings=vars(a), tag=a.tag, holdout=a.holdout)
    with open(Path(folder) / f"model{a.tag}.pkl", "wb") as f:
        pickle.dump({"clf": clf, "ref": ref}, f)
    save_results(folder, info)
