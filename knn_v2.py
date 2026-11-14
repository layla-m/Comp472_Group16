"""
iter9 (pistes 1 + 2): better k-NN for the image task, judged directly inside the ensemble
P = w * mean(MLP[4096], MLP[8192]) + (1 - w) * kNN   (the iter8 ensemble, val 0.6298).

Step 1 - tune the k-NN: feature transform (power p of the L1-normalised counts, then L2
         normalisation; p=0.5 is the Hellinger distance used so far; optional idf weighting),
         k up to 200 and the exponential weight temperature tau.
Step 2 - pseudo-labelling: the unlabelled points whose ensemble prediction is confident are
         added to the k-NN reference set with their predicted label (weight pw). On validation
         this is simulated with the validation points themselves as the unlabelled pool
         (leave-one-out: a point is never its own neighbour; their true labels are never used).
         Pool / train ratio is the same as for test (4000/16000 = 5000/20000).

Writes knn2_results_image.csv and, if the validation improves on iter8, iter9.csv.
Usage: python knn_v2.py
"""
import itertools
import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from neural_network import SEED, load_task

K_MAX = 200
KS = [20, 30, 50, 75, 100, 150, 200]
TAUS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.1]
POWERS = [0.3, 0.4, 0.5, 0.7]
WS = np.round(np.arange(0, 1.001, 0.05), 2)
ITER8_VAL = 0.6298


def to_counts(X):
    X = np.clip(np.asarray(X, np.float64), 0, None)
    m = np.where(X > 0, X, np.inf).min(1, keepdims=True)
    return np.round(X / m)


def transform(C, p, idf):
    Z = C * idf if idf is not None else C
    Z = (Z / (Z.sum(1, keepdims=True) + 1e-12)) ** p
    return (Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)).astype(np.float32)


def neighbours(A, B, k, exclude_self=False, batch=1000):
    B_t = torch.from_numpy(B)
    ds, idx = [], []
    for i in range(0, len(A), batch):
        d = torch.cdist(torch.from_numpy(A[i:i + batch]), B_t)
        if exclude_self:
            r = torch.arange(d.shape[0])
            d[r, r + i] = float("inf")
        v, j = d.topk(k, dim=1, largest=False)
        ds.append(v.numpy())
        idx.append(j.numpy())
    return np.vstack(ds), np.vstack(idx)


def knn_proba(d, lab, wt, k, tau, nc):
    """d, lab, wt: (n, >=k) sorted neighbour distances, labels, extra weights."""
    d, lab, wt = d[:, :k], lab[:, :k], wt[:, :k]
    w = np.exp(-(d - d[:, :1]) / tau) * wt
    w[~np.isfinite(d)] = 0
    P = np.zeros((len(d), nc))
    np.add.at(P, (np.repeat(np.arange(len(d)), k), lab.ravel()), w.ravel())
    return P / (P.sum(1, keepdims=True) + 1e-12)


def best_w(N, K, y):
    accs = [((w * N + (1 - w) * K).argmax(1) == y).mean() for w in WS]
    i = int(np.argmax(accs))
    return WS[i], accs[i]


def merged(dA, iA, labA, dB, iB, labB, conf, pw, k):
    """Merge train neighbours (A) with pseudo-labelled pool neighbours (B), keep k nearest."""
    okB = conf[iB]
    dB = np.where(okB, dB, np.inf)
    d = np.hstack([dA, dB])
    lab = np.hstack([labA[iA], labB[iB]])
    wt = np.hstack([np.ones_like(dA), np.full_like(dB, pw)])
    o = np.argsort(d, axis=1)[:, :k]
    r = np.arange(len(d))[:, None]
    return d[r, o], lab[r, o], wt[r, o]


def main():
    torch.set_num_threads(int(os.environ.get("NTHREADS", 8)))
    X, y, X_test, test_ids = load_task("image")
    nc = len(np.unique(y))
    C, C_te = to_counts(X), to_counts(X_test)
    i_tr, i_va = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED)
    y_tr, y_va = y[i_tr], y[i_va]
    load = lambda tag, part: np.load(f"ens/{tag}_image_{part}.npy")
    N_va = (load("iter7_nn", "val") + load("iter8_nn", "val")) / 2
    N_te = (load("iter7_nn", "test") + load("iter8_nn", "test")) / 2

    def idf_of(Cref):
        df = (Cref > 0).sum(0)
        return np.log((1 + len(Cref)) / (1 + df)) + 1

    # ---------------- step 1: k-NN tuning ----------------
    rows, cache = [], {}
    for p, use_idf in itertools.product(POWERS, [False, True]):
        idf = idf_of(C[i_tr]) if use_idf else None
        H_tr, H_va = transform(C[i_tr], p, idf), transform(C[i_va], p, idf)
        d, j = neighbours(H_va, H_tr, K_MAX)
        lab, one = y_tr[j], np.ones_like(d)
        for k, tau in itertools.product(KS, TAUS):
            K = knn_proba(d, lab, one, k, tau, nc)
            w, acc = best_w(N_va, K, y_va)
            rows.append(dict(step=1, p=p, idf=use_idf, k=k, tau=tau, knn_acc=(K.argmax(1) == y_va).mean(),
                             w_nn=w, ens_acc=acc))
        r = max((r for r in rows if r["p"] == p and r["idf"] == use_idf), key=lambda r: r["ens_acc"])
        print(f"p={p} idf={use_idf}: best k={r['k']} tau={r['tau']} knn={r['knn_acc']:.4f} "
              f"ens={r['ens_acc']:.4f} (w={r['w_nn']})", flush=True)
    s1 = max(rows, key=lambda r: r["ens_acc"])
    p, use_idf, k, tau = s1["p"], s1["idf"], s1["k"], s1["tau"]
    print(f"STEP1 BEST p={p} idf={use_idf} k={k} tau={tau} knn={s1['knn_acc']:.4f} "
          f"ens={s1['ens_acc']:.4f} w={s1['w_nn']}", flush=True)

    # ---------------- step 2: pseudo-labelling ----------------
    idf = idf_of(C[i_tr]) if use_idf else None
    H_tr, H_va = transform(C[i_tr], p, idf), transform(C[i_va], p, idf)
    dA, iA = neighbours(H_va, H_tr, K_MAX)
    dB, iB = neighbours(H_va, H_va, K_MAX, exclude_self=True)
    K1 = knn_proba(dA, y_tr[iA], np.ones_like(dA), k, tau, nc)
    P1 = s1["w_nn"] * N_va + (1 - s1["w_nn"]) * K1
    pseudo, pconf = P1.argmax(1), P1.max(1)
    s2 = None
    for thr, pw, k2 in itertools.product([0.0, 0.3, 0.5, 0.7, 0.9], [0.5, 1.0], sorted({k, 50, 100, 200})):
        d, lab, wt = merged(dA, iA, y_tr, dB, iB, pseudo, pconf >= thr, pw, k2)
        for tau2 in TAUS:
            K2 = knn_proba(d, lab, wt, k2, tau2, nc)
            w, acc = best_w(N_va, K2, y_va)
            r = dict(step=2, p=p, idf=use_idf, k=k2, tau=tau2, thr=thr, pw=pw,
                     knn_acc=(K2.argmax(1) == y_va).mean(), w_nn=w, ens_acc=acc)
            rows.append(r)
            if s2 is None or acc > s2["ens_acc"]:
                s2 = r
        print(f"thr={thr} pw={pw} k={k2}: best ens so far {s2['ens_acc']:.4f}", flush=True)
    print(f"STEP2 BEST thr={s2['thr']} pw={s2['pw']} k={s2['k']} tau={s2['tau']} "
          f"knn={s2['knn_acc']:.4f} ens={s2['ens_acc']:.4f} w={s2['w_nn']}", flush=True)
    pd.DataFrame(rows).sort_values("ens_acc", ascending=False).round(4).to_csv(
        "knn2_results_image.csv", index=False)

    # ---------------- test predictions ----------------
    best = s2 if s2["ens_acc"] > s1["ens_acc"] else s1
    print(f"USING step {best['step']} val ens={best['ens_acc']:.4f} (iter8 val {ITER8_VAL})", flush=True)
    if best["ens_acc"] <= ITER8_VAL:
        print("No gain over iter8: iter9.csv not written.")
        return
    idf = idf_of(C) if use_idf else None
    H, H_te = transform(C, p, idf), transform(C_te, p, idf)
    dA, iA = neighbours(H_te, H, K_MAX)
    K1 = knn_proba(dA, y[iA], np.ones_like(dA), k, tau, nc)
    np.save("ens/knn2_step1_image_test.npy", K1)
    K = K1
    if best["step"] == 2:
        P1 = s1["w_nn"] * N_te + (1 - s1["w_nn"]) * K1
        dB, iB = neighbours(H_te, H_te, K_MAX, exclude_self=True)
        d, lab, wt = merged(dA, iA, y, dB, iB, P1.argmax(1), P1.max(1) >= best["thr"], best["pw"], best["k"])
        K = knn_proba(d, lab, wt, best["k"], best["tau"], nc)
        np.save("ens/knn2_step2_image_test.npy", K)
    w = best["w_nn"]
    img = pd.DataFrame({"id": test_ids, "label": (w * N_te + (1 - w) * K).argmax(1)})
    nn = pd.read_csv("iter8_nn.csv")
    sen = nn[nn["id"].str.startswith("text_")]
    sample = pd.read_csv("sample_submission.csv")
    sub = sample[["id"]].merge(pd.concat([img, sen]), on="id", how="left", validate="one_to_one")
    assert len(sub) == 10000 and sub["label"].notna().all()
    sub["label"] = sub["label"].astype(int)
    assert sub.loc[sub["id"].str.startswith("img_"), "label"].between(0, 49).all()
    assert sub.loc[sub["id"].str.startswith("text_"), "label"].between(0, 2).all()
    sub.to_csv("iter9.csv", index=False)
    it8 = pd.read_csv("iter8.csv").merge(sub, on="id")
    print(f"Saved iter9.csv (agreement with iter8: {(it8.label_x == it8.label_y).mean():.4f})")


if __name__ == "__main__":
    main()
