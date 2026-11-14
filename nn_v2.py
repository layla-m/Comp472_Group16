"""
COMP472 - Neural network, iteration 2.

Same pipeline as neural_network.py (stratified 80/20 validation split, then
retrain on all data for best_epoch with several seeds, average probabilities),
with extra regularisation options explored by hp_search2.py:
  - row_norm     : L1-normalise each histogram row before the sqrt (true Hellinger)
  - in_dropout   : dropout on the input features
  - mixup        : mixup alpha (0 = off)
  - smoothing    : label smoothing

Usage:
    python nn_v2.py            -> writes iter2.csv
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from neural_network import SEED, load_task, predict, predict_proba, set_seed

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
N_SEEDS = 5
OUT_NAME = "iter2.csv"

BASE = dict(hidden=[512, 128], dropout=0.5, in_dropout=0.0, lr=1e-3, weight_decay=1e-2,
            batch_size=128, max_epochs=60, patience=10, sqrt_features=False,
            row_norm=False, l2_norm=False, mixup=0.0, smoothing=0.1)

# One config per submitted iteration (chosen from hp_search2.py results), selected by the
# output file name: `python nn_v2.py iter3.csv` uses ITERS["iter3"].
SENTIMENT = {**BASE, "dropout": 0.7, "lr": 5e-4}
IMAGE_ITER2 = {**BASE, "hidden": [4096], "sqrt_features": True, "in_dropout": 0.5, "dropout": 0.3}
ITERS = {
    # in_dropout 0.5 ([2048]: 0.481), [4096] (0.496), + dropout 0.3 -> val 0.509, Kaggle 0.609
    "iter2": {"image": IMAGE_ITER2, "sentiment": SENTIMENT},
    # piste 2: + tfidf weighting of the visual words (val 0.510)
    "iter3": {"image": {**IMAGE_ITER2, "tfidf": True}, "sentiment": SENTIMENT},
    # piste 1: wider layer ([8192] alone: 0.514) + up to 100 epochs (helped [4096]: 0.496 -> 0.508)
    "iter4": {"image": {**IMAGE_ITER2, "hidden": [8192], "max_epochs": 100, "patience": 15},
              "sentiment": SENTIMENT},
}
# iter7: same network as iter2, re-run to save its test probabilities for the NN + k-NN ensemble
ITERS["iter7_nn"] = ITERS["iter2"]
# iter8: iter4 network ([8192], 100 epochs), re-run to save its probabilities for the ensemble
ITERS["iter8_nn"] = ITERS["iter4"]
# iter10: diverse networks for the ensemble (a weaker but different network helped: +0.5 pt)
ITERS["iter10_a"] = {"image": {**IMAGE_ITER2, "row_norm": True, "std": False, "in_dropout": 0.3},
                     "sentiment": SENTIMENT}   # Hellinger features, no standardisation
ITERS["iter10_b"] = {"image": {**IMAGE_ITER2, "hidden": [2048, 1024]}, "sentiment": SENTIMENT}
ITERS["iter10_c"] = {"image": {**IMAGE_ITER2, "in_dropout": 0.7}, "sentiment": SENTIMENT}
CONFIG = ITERS["iter2"]  # default, also used as base by hp_search2.py (round 4)


class Prep:
    """Row-wise transforms + standardisation fitted on the training subset only."""

    def __init__(self, cfg):
        self.cfg = cfg

    def _pre(self, X):
        X = np.asarray(X, np.float32)
        if self.cfg.get("tfidf"):
            X = np.clip(X, 0, None) * self.idf
        if self.cfg["row_norm"]:
            X = np.clip(X, 0, None)
            X = X / (X.sum(1, keepdims=True) + 1e-8)
        if self.cfg["sqrt_features"]:
            X = np.sqrt(np.clip(X, 0, None))
        if self.cfg["l2_norm"]:
            X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        return X

    def fit(self, X):
        if self.cfg.get("tfidf"):
            # idf of each "visual word": rare words weigh more (smooth idf, as in sklearn)
            df = (np.asarray(X) > 0).sum(0)
            self.idf = (np.log((1 + len(X)) / (1 + df)) + 1).astype(np.float32)
        X = self._pre(X)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        if self.cfg.get("std", True) is False:
            # no per-feature standardisation (keeps the Hellinger geometry): only a global
            # scale so that the mean squared value of a feature is about 1
            scale = self.cfg.get("scale") or np.sqrt(X.shape[1])
            self.mu = np.zeros(X.shape[1], np.float32)
            self.sd = np.full(X.shape[1], 1.0 / scale, np.float32)
        if self.cfg.get("pca"):
            # PCA whitening on the standardised features, fitted on the training subset only
            from sklearn.decomposition import PCA
            self.pca = PCA(n_components=self.cfg["pca"], whiten=True, random_state=SEED)
            self.pca.fit((X - self.mu) / self.sd)
        return self

    def transform(self, X):
        Z = ((self._pre(X) - self.mu) / self.sd).astype(np.float32)
        if self.cfg.get("pca"):
            Z = self.pca.transform(Z).astype(np.float32)
        return Z


class MLP(nn.Module):
    def __init__(self, n_in, hidden, n_out, dropout, in_dropout):
        super().__init__()
        layers, d = ([nn.Dropout(in_dropout)] if in_dropout > 0 else []), n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def to_counts(X):
    """Image features are L2-normalised integer counts of visual words: the smallest
    non-zero value of a row is 1/||counts||, so counts = X / min_nonzero(X)."""
    X = np.asarray(X, np.float32)
    m = np.where(X > 0, X, np.inf).min(1, keepdims=True)
    m[~np.isfinite(m)] = 1.0
    return np.round(X / m).astype(np.float32)


class TorchPrep:
    """Same transform as Prep (L2 norm of counts -> sqrt -> standardise), in torch, so that
    augmented counts can be transformed batch by batch during training."""

    def __init__(self, prep):
        cfg = prep.cfg
        assert not (cfg.get("tfidf") or cfg.get("pca") or cfg["row_norm"] or cfg["l2_norm"])
        self.sqrt = cfg["sqrt_features"]
        self.mu, self.sd = torch.from_numpy(prep.mu), torch.from_numpy(prep.sd)

    def __call__(self, K):
        x = K / (K.norm(dim=1, keepdim=True) + 1e-8)
        if self.sqrt:
            x = torch.sqrt(x)
        return (x - self.mu) / self.sd


@torch.no_grad()
def update_bn(model, X, batch_size=512):
    """Recompute BatchNorm statistics of the averaged (SWA) model on the training data."""
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None  # cumulative average
    model.train()
    for i in range(0, len(X), batch_size):
        model(X[i:i + batch_size])


def train_model(X_tr, y_tr, n_classes, cfg, X_va=None, y_va=None, epochs=None, seed=SEED, verbose=False,
                counts=None, tprep=None):
    """counts/tprep: raw visual-word counts of X_tr and the torch transform, needed for the
    `thin` augmentation (each patch kept with probability 1 - thin, then re-normalised)."""
    set_seed(seed)
    model = MLP(X_tr.shape[1], cfg["hidden"], n_classes, cfg["dropout"], cfg["in_dropout"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg["smoothing"])
    thin, noise, swa_start = cfg.get("thin", 0.0), cfg.get("noise", 0.0), cfg.get("swa_start", 0)
    inputs = torch.from_numpy(counts) if thin > 0 else torch.from_numpy(X_tr)
    loader = DataLoader(TensorDataset(inputs, torch.from_numpy(y_tr)),
                        batch_size=cfg["batch_size"], shuffle=True)
    X_tr_t = torch.from_numpy(X_tr)
    swa = torch.optim.swa_utils.AveragedModel(model) if swa_start else None
    n_epochs = epochs or cfg["max_epochs"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    beta = torch.distributions.Beta(cfg["mixup"], cfg["mixup"]) if cfg["mixup"] > 0 else None

    best_acc, best_epoch, best_state, wait = -1, 0, None, 0
    for ep in range(1, n_epochs + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            if thin > 0:
                xb = tprep(torch.binomial(xb, torch.full_like(xb, 1 - thin)))
            if noise > 0:
                xb = xb + noise * torch.randn_like(xb)
            if beta is not None:
                lam = beta.sample().item()
                perm = torch.randperm(len(xb))
                out = model(lam * xb + (1 - lam) * xb[perm])
                loss = lam * loss_fn(out, yb) + (1 - lam) * loss_fn(out, yb[perm])
            else:
                loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()
        # SWA: average the weights of every epoch from swa_start on
        use_swa = swa is not None and ep >= swa_start
        if use_swa:
            swa.update_parameters(model)
            if X_va is not None or ep == n_epochs:
                update_bn(swa, X_tr_t)
        if X_va is None:
            continue
        net = swa if use_swa else model
        va_acc = (predict(net, X_va) == y_va).mean()
        if verbose:
            print(f"  epoch {ep:3d} val_acc={va_acc:.4f}", flush=True)
        if va_acc > best_acc:
            best_acc, best_epoch, wait = va_acc, ep, 0
            best_state = (use_swa, {k: v.clone() for k, v in net.state_dict().items()})
        else:
            wait += 1
            if wait >= cfg["patience"]:
                break
    if X_va is None:
        return (swa if swa is not None and n_epochs >= swa_start else model), n_epochs, None
    if best_state is not None:
        was_swa, state = best_state
        net = swa if was_swa else model
        net.load_state_dict(state)
        return net, best_epoch, best_acc
    return model, best_epoch, best_acc


def aug_args(cfg, X_raw, prep):
    """Extra train_model arguments for the `thin` augmentation (raw counts + torch transform)."""
    if cfg.get("thin", 0) > 0:
        return dict(counts=to_counts(X_raw), tprep=TorchPrep(prep))
    return {}


def run_task(task, config, tag):
    cfg = config[task]
    X, y, X_test, ids = load_task(task)
    n_classes = len(np.unique(y))
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    p = Prep(cfg).fit(X_tr)
    t0 = time.time()
    m_val, best_epoch, best_acc = train_model(p.transform(X_tr), y_tr, n_classes, cfg, p.transform(X_va), y_va,
                                          **aug_args(cfg, X_tr, p))
    print(f"[{task}] val_acc={best_acc:.4f} best_epoch={best_epoch} ({time.time() - t0:.0f}s)", flush=True)
    # class probabilities saved for ensembles (ens_combine.py)
    os.makedirs(os.path.join(DATA_DIR, "ens"), exist_ok=True)
    np.save(os.path.join(DATA_DIR, "ens", f"{tag}_{task}_val.npy"), predict_proba(m_val, p.transform(X_va)))

    p = Prep(cfg).fit(X)
    X_full, X_te = p.transform(X), p.transform(X_test)
    probs = np.zeros((len(X_te), n_classes))
    for s in range(N_SEEDS):
        m, *_ = train_model(X_full, y, n_classes, cfg, epochs=best_epoch, seed=SEED + s,
                            **aug_args(cfg, X, p))
        probs += predict_proba(m, X_te)
    np.save(os.path.join(DATA_DIR, "ens", f"{tag}_{task}_test.npy"), probs / N_SEEDS)
    return pd.DataFrame({"id": ids, "label": probs.argmax(1)}), best_acc


def main():
    out_name = sys.argv[1] if len(sys.argv) > 1 else OUT_NAME
    config = ITERS.get(os.path.splitext(out_name)[0], CONFIG)
    print(f"config: {config}", flush=True)
    tag = os.path.splitext(out_name)[0]
    img, a1 = run_task("image", config, tag)
    sen, a2 = run_task("sentiment", config, tag)
    print(f"Validation: image={a1:.4f} sentiment={a2:.4f} overall={(a1 + a2) / 2:.4f}")
    sample = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
    sub = sample[["id"]].merge(pd.concat([img, sen]), on="id", how="left", validate="one_to_one")
    assert len(sub) == 10000 and sub["label"].notna().all()
    sub["label"] = sub["label"].astype(int)
    assert sub.loc[sub["id"].str.startswith("img_"), "label"].between(0, 49).all()
    assert sub.loc[sub["id"].str.startswith("text_"), "label"].between(0, 2).all()
    out = os.path.join(DATA_DIR, out_name)
    sub.to_csv(out, index=False)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
