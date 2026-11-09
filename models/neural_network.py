"""
COMP472 Project - Fall 2026 : Feed-forward Neural Network (PyTorch)

Trains one MLP per task (image: 50 classes, sentiment: 3 classes),
validates on a stratified 80/20 split of the training data, then retrains on
all training data and writes `submission.csv` (10,000 rows: id,label).

Run from the project root (data/*.csv must exist):
    python -m models.neural_network

Outputs: results/nn_submission.csv, results/nn/history_<task>.csv
"""

import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# ------------------------------------------------------------------ setup
SEED = 42
N_SEEDS = 5  # final models trained with seeds 42..46, probabilities averaged
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
torch.set_num_threads(min(4, os.cpu_count() or 1))


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# Hyperparameters per task (chosen on the validation split)
CONFIG = {
    "image": dict(
        hidden=[1024, 512],   # Input(500) -> 1024 -> 512 -> 50
        dropout=0.5,
        lr=1e-3,
        weight_decay=1e-2,
        batch_size=128,
        max_epochs=60,
        patience=10,
        sqrt_features=True,   # Hellinger transform for histogram-like features
    ),
    "sentiment": dict(
        hidden=[512, 128],    # Input(768) -> 512 -> 128 -> 3
        dropout=0.5,
        lr=5e-4,
        weight_decay=1e-2,
        batch_size=128,
        max_epochs=60,
        patience=10,
        sqrt_features=False,
    ),
}


# ------------------------------------------------------------------ data
def load_task(task):
    train = pd.read_csv(os.path.join(DATA_DIR, f"{task}_train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, f"{task}_test.csv"))
    feats = [c for c in train.columns if c.startswith("feature_")]
    X = train[feats].to_numpy(np.float32)
    y = train["label"].to_numpy(np.int64)
    X_test = test[feats].to_numpy(np.float32)
    return X, y, X_test, test["id"]


class Standardizer:
    """Fit mean/std on the training subset only (no leakage)."""

    def __init__(self, sqrt=False):
        self.sqrt = sqrt

    def _pre(self, X):
        return np.sqrt(np.clip(X, 0, None)) if self.sqrt else X

    def fit(self, X):
        X = self._pre(X)
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-6
        return self

    def transform(self, X):
        return ((self._pre(X) - self.mu) / self.sd).astype(np.float32)


# ------------------------------------------------------------------ model
class MLP(nn.Module):
    def __init__(self, n_in, hidden, n_out, dropout):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, n_out))  # logits (softmax is inside the loss)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def predict_proba(model, X):
    model.eval()
    return torch.softmax(model(torch.from_numpy(X)), 1).numpy()


@torch.no_grad()
def predict(model, X):
    model.eval()
    return model(torch.from_numpy(X)).argmax(1).numpy()


def train_model(X_tr, y_tr, n_classes, cfg, X_va=None, y_va=None, epochs=None, verbose=True, seed=SEED):
    """Train an MLP. With a validation set: early stopping, returns best epoch.
    Without: trains for exactly `epochs` epochs."""
    set_seed(seed)
    model = MLP(X_tr.shape[1], cfg["hidden"], n_classes, cfg["dropout"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                        batch_size=cfg["batch_size"], shuffle=True)
    n_epochs = epochs or cfg["max_epochs"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    best_acc, best_epoch, best_state, wait = -1, 0, None, 0
    history = []
    for ep in range(1, n_epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        sched.step()

        if X_va is None:
            continue
        tr_acc = (predict(model, X_tr) == y_tr).mean()
        va_acc = (predict(model, X_va) == y_va).mean()
        history.append((ep, total / len(X_tr), tr_acc, va_acc))
        if verbose:
            print(f"  epoch {ep:3d}  loss={total / len(X_tr):.4f}  train_acc={tr_acc:.4f}  val_acc={va_acc:.4f}")
        if va_acc > best_acc:
            best_acc, best_epoch, wait = va_acc, ep, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg["patience"]:
                if verbose:
                    print(f"  early stopping (best epoch {best_epoch})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, best_acc, history


# ------------------------------------------------------------------ pipeline
def run_task(task):
    cfg = CONFIG[task]
    print(f"\n========== {task.upper()} ==========")
    X, y, X_test, test_ids = load_task(task)
    n_classes = len(np.unique(y))
    print(f"train {X.shape}, test {X_test.shape}, classes={n_classes}")

    # 1) validation: stratified 80/20 split, preprocessing fitted on train part only
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    scaler = Standardizer(cfg["sqrt_features"]).fit(X_tr)
    t0 = time.time()
    model, best_epoch, best_acc, history = train_model(
        scaler.transform(X_tr), y_tr, n_classes, cfg, scaler.transform(X_va), y_va)
    print(f"[{task}] best validation accuracy = {best_acc:.4f} at epoch {best_epoch} "
          f"({time.time() - t0:.1f}s)")

    y_pred = predict(model, scaler.transform(X_va))
    if task == "sentiment":
        print(classification_report(y_va, y_pred, target_names=["Negative", "Neutral", "Positive"], digits=4))
    else:
        cm = confusion_matrix(y_va, y_pred)
        np.fill_diagonal(cm, 0)
        pairs = np.dstack(np.unravel_index(np.argsort(cm.ravel())[::-1][:10], cm.shape))[0]
        print("most confused (true -> predicted : count):")
        for t, p in pairs:
            print(f"  {t:2d} -> {p:2d} : {cm[t, p]}")
    (RESULTS_DIR / "nn").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history, columns=["epoch", "train_loss", "train_acc", "val_acc"]).to_csv(
        RESULTS_DIR / "nn" / f"history_{task}.csv", index=False)

    # 2) final model: retrain on ALL training data for the best number of epochs,
    #    with N_SEEDS different seeds, and average their softmax probabilities.
    scaler_full = Standardizer(cfg["sqrt_features"]).fit(X)
    X_full, X_te = scaler_full.transform(X), scaler_full.transform(X_test)
    probs = np.zeros((len(X_te), n_classes))
    for s in range(N_SEEDS):
        m, *_ = train_model(X_full, y, n_classes, cfg, epochs=best_epoch, verbose=False, seed=SEED + s)
        probs += predict_proba(m, X_te)
    test_pred = probs.argmax(1)
    return pd.DataFrame({"id": test_ids, "label": test_pred}), best_acc


def main():
    img_res, img_acc = run_task("image")
    sen_res, sen_acc = run_task("sentiment")
    print(f"\nValidation: image={img_acc:.4f}  sentiment={sen_acc:.4f}  "
          f"overall={(img_acc + sen_acc) / 2:.4f}")

    sample = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
    preds = pd.concat([img_res, sen_res], ignore_index=True)
    sub = sample[["id"]].merge(preds, on="id", how="left", validate="one_to_one")

    assert len(sub) == 10000
    assert sub["label"].notna().all()
    sub["label"] = sub["label"].astype(int)
    assert sub.loc[sub["id"].str.startswith("img_"), "label"].between(0, 49).all()
    assert sub.loc[sub["id"].str.startswith("text_"), "label"].between(0, 2).all()

    out = RESULTS_DIR / "nn_submission.csv"
    sub.to_csv(out, index=False)
    print(f"Saved {out} ({len(sub)} rows)")


if __name__ == "__main__":
    main()
