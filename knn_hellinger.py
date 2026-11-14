"""
Test A: k-NN with the Hellinger distance on the image features.

Hellinger distance between two histograms = Euclidean distance between the square roots
of their L1-normalised versions. No per-feature standardisation.

On the same 80/20 split as the MLP: grid over k and the neighbour weighting, results in
knn_results_image.csv. The best setting is then used to save class probabilities:
  ens/knn_image_val.npy   (model fitted on the 80 %, predicted on the 20 %)
  ens/knn_image_test.npy  (model fitted on 100 % of train, predicted on test)

Usage: python knn_hellinger.py
"""
import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from neural_network import SEED, load_task

HERE = os.path.dirname(os.path.abspath(__file__))
K_MAX = 50
KS = [1, 3, 5, 7, 10, 15, 20, 30, 50]
WEIGHTS = ["uniform", "inv", "exp0.02", "exp0.05", "exp0.1"]


def hellinger(X):
    X = np.clip(np.asarray(X, np.float32), 0, None)
    return np.sqrt(X / (X.sum(1, keepdims=True) + 1e-12)).astype(np.float32)


def neighbours(A, B, k=K_MAX, batch=1000):
    """k nearest rows of B for every row of A (Euclidean): distances and indices."""
    B_t = torch.from_numpy(B)
    ds, idx = [], []
    for i in range(0, len(A), batch):
        d = torch.cdist(torch.from_numpy(A[i:i + batch]), B_t)
        v, j = d.topk(k, dim=1, largest=False)
        ds.append(v.numpy())
        idx.append(j.numpy())
    return np.vstack(ds), np.vstack(idx)


def knn_proba(dist, idx, y_ref, n_classes, k, weight):
    d, lab = dist[:, :k], y_ref[idx[:, :k]]
    if weight == "uniform":
        w = np.ones_like(d)
    elif weight == "inv":
        w = 1.0 / (d + 1e-6)
    else:  # exp<tau>: closer neighbours count exponentially more
        tau = float(weight[3:])
        w = np.exp(-(d - d[:, :1]) / tau)
    P = np.zeros((len(d), n_classes))
    for c in range(k):
        P[np.arange(len(d)), lab[:, c]] += w[:, c]
    return P / P.sum(1, keepdims=True)


def main():
    torch.set_num_threads(int(os.environ.get("NTHREADS", 4)))
    X, y, X_test, _ = load_task("image")
    nc = len(np.unique(y))
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    H_tr, H_va = hellinger(X_tr), hellinger(X_va)
    dist, idx = neighbours(H_va, H_tr)

    rows = []
    for k in KS:
        for w in WEIGHTS:
            if k == 1 and w != "uniform":
                continue
            acc = (knn_proba(dist, idx, y_tr, nc, k, w).argmax(1) == y_va).mean()
            rows.append(dict(k=k, weight=w, val_acc=round(float(acc), 4)))
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows).sort_values("val_acc", ascending=False)
    df.to_csv(os.path.join(HERE, "knn_results_image.csv"), index=False)
    best = df.iloc[0]
    k, w = int(best["k"]), best["weight"]
    print(f"BEST k={k} weight={w} val_acc={best['val_acc']}", flush=True)

    os.makedirs(os.path.join(HERE, "ens"), exist_ok=True)
    np.save(os.path.join(HERE, "ens", "knn_image_val.npy"), knn_proba(dist, idx, y_tr, nc, k, w))
    dist_t, idx_t = neighbours(hellinger(X_test), hellinger(X))
    np.save(os.path.join(HERE, "ens", "knn_image_test.npy"), knn_proba(dist_t, idx_t, y, nc, k, w))
    print("saved ens/knn_image_val.npy and ens/knn_image_test.npy", flush=True)


if __name__ == "__main__":
    main()
