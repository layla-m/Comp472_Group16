"""
Helpers shared by the Decision Tree / Random Forest scripts.

"""

import json
import os
import random
import time

import numpy as np
import pandas as pd

SEED = 42
N_CLASSES = {"image": 50, "sentiment": 3}
SENTIMENT_NAMES = ["Negative", "Neutral", "Positive"]
RESULTS = os.path.join("results", "trees")


# ------------------------------------------------------------------ data
def load_split(task, quick=False):
    """Return X_train, y_train, X_val, y_val from the shared split.
    quick=True keeps 25% of the training rows."""
    path = os.path.join("data", "processed", f"{task}_split.npz")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found: run  python prepare_data.py  first.")
    d = np.load(path)
    X_tr, y_tr, X_va, y_va = d["X_train"], d["y_train"], d["X_val"], d["y_val"]
    if quick:
        keep = np.random.default_rng(0).permutation(len(y_tr))[: len(y_tr) // 4]
        X_tr, y_tr = X_tr[keep], y_tr[keep]
    return X_tr, y_tr, X_va, y_va


def load_full_train_and_test(task):
    """All labelled rows (train + validation) and the test rows, from the CSVs."""
    train = pd.read_csv(os.path.join("data", f"{task}_train.csv"))
    test = pd.read_csv(os.path.join("data", f"{task}_test.csv"))
    cols = [c for c in train.columns if c.startswith("feature_")]
    return (train[cols].to_numpy(dtype=np.float32), train["label"].to_numpy(dtype=np.int64),
            test["id"], test[cols].to_numpy(dtype=np.float32))


# ---------------------------------------------------------------- misc
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)


def out_dir(task):
    path = os.path.join(RESULTS, task)
    os.makedirs(path, exist_ok=True)
    return path


def timed_fit(model, X, y):
    start = time.perf_counter()
    model.fit(X, y)
    return time.perf_counter() - start


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def best_params_path(task, model):
    return os.path.join(out_dir(task), f"best_params_{model}.json")


def tasks_from_arg(arg):
    return ["image", "sentiment"] if arg == "both" else [arg]


# ------------------------------------------------------------- metrics
def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def confusion_matrix(y_true, y_pred, n_classes):
    """cm[i, j] = number of samples of true class i predicted as class j."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm


def per_class_report(y_true, y_pred, n_classes, names=None):
    """Precision, recall, F1 and support for every class."""
    cm = confusion_matrix(y_true, y_pred, n_classes)
    tp = np.diag(cm).astype(float)
    pred_count = cm.sum(axis=0)
    true_count = cm.sum(axis=1)
    precision = np.divide(tp, pred_count, out=np.zeros_like(tp), where=pred_count > 0)
    recall = np.divide(tp, true_count, out=np.zeros_like(tp), where=true_count > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return pd.DataFrame({
        "class": names if names is not None else np.arange(n_classes),
        "precision": precision, "recall": recall, "f1": f1, "support": true_count,
    })


def most_confused_pairs(cm, top=10):
    """Largest off-diagonal entries of the confusion matrix."""
    off = cm.copy()
    np.fill_diagonal(off, 0)
    rows = []
    for f in np.argsort(off, axis=None)[::-1][:top]:
        i, j = np.unravel_index(f, off.shape)
        if off[i, j] == 0:
            break
        rows.append({"true": int(i), "predicted": int(j), "count": int(off[i, j]),
                     "share_of_true_class": off[i, j] / max(cm[i].sum(), 1)})
    return pd.DataFrame(rows)
