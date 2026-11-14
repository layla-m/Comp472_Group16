"""
iter10.csv = image: average of a subset of MLPs + tuned k-NN with pseudo-labelling (iter9 recipe).

Networks available (probabilities in ens/): iter7_nn ([4096]), iter8_nn ([8192]),
iter10_a (Hellinger features, no standardisation), iter10_b ([2048, 1024]), iter10_c (in_dropout 0.7).
Every non-empty subset (equal weights) is evaluated on the 80/20 validation split with the full
iter9 pipeline: k-NN p=0.4, k=50, tau=0.02 -> ensemble -> pseudo-labels (thr 0.3, weight 0.5)
-> k-NN k=100, tau=0.02 -> ensemble. The best subset is used for test. Sentiment: iter8_nn.

Usage: python make_iter10.py
"""
import itertools
import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from knn_v2 import K_MAX, best_w, knn_proba, merged, neighbours, to_counts, transform
from neural_network import SEED, load_task

P_POW, K1, TAU1 = 0.4, 50, 0.02          # step 1 (knn_v2.py)
THR, PW, K2, TAU2 = 0.3, 0.5, 100, 0.02  # step 2 (knn_v2.py)
ITER9_VAL = 0.6370
NETS = ["iter7_nn", "iter8_nn", "iter10_a", "iter10_b", "iter10_c"]


def pipeline(N, H_q, H_ref, y_ref, nc, y_q=None, w1=None, w2=None):
    """k-NN + ensemble + pseudo-labelling on query set H_q. With y_q, weights are tuned."""
    dA, iA = neighbours(H_q, H_ref, K_MAX)
    dB, iB = neighbours(H_q, H_q, K_MAX, exclude_self=True)
    return dA, iA, dB, iB


def run(N, nb, y_ref, nc, y_q=None, w1=None, w2=None):
    dA, iA, dB, iB = nb
    Ka = knn_proba(dA, y_ref[iA], np.ones_like(dA), K1, TAU1, nc)
    if y_q is not None:
        w1, _ = best_w(N, Ka, y_q)
    P1 = w1 * N + (1 - w1) * Ka
    d, lab, wt = merged(dA, iA, y_ref, dB, iB, P1.argmax(1), P1.max(1) >= THR, PW, K2)
    Kb = knn_proba(d, lab, wt, K2, TAU2, nc)
    acc = None
    if y_q is not None:
        w2, acc = best_w(N, Kb, y_q)
    return w2 * N + (1 - w2) * Kb, w1, w2, acc


def main():
    torch.set_num_threads(int(os.environ.get("NTHREADS", 8)))
    X, y, X_test, test_ids = load_task("image")
    nc = len(np.unique(y))
    C, C_te = to_counts(X), to_counts(X_test)
    i_tr, i_va = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED)
    nets = [n for n in NETS if os.path.exists(f"ens/{n}_image_test.npy")]  # finished networks only
    out_name = os.environ.get("OUT", "iter10")
    V = {n: np.load(f"ens/{n}_image_val.npy") for n in nets}
    T = {n: np.load(f"ens/{n}_image_test.npy") for n in nets}
    for n in nets:
        print(f"{n:10s} alone {(V[n].argmax(1) == y[i_va]).mean():.4f}", flush=True)

    H_tr, H_va = transform(C[i_tr], P_POW, None), transform(C[i_va], P_POW, None)
    nb_va = pipeline(None, H_va, H_tr, y[i_tr], nc)
    res = []
    for r in range(1, len(nets) + 1):
        for sub in itertools.combinations(nets, r):
            N = np.mean([V[n] for n in sub], axis=0)
            _, w1, w2, acc = run(N, nb_va, y[i_tr], nc, y_q=y[i_va])
            res.append((acc, sub, w1, w2, (N.argmax(1) == y[i_va]).mean()))
    res.sort(key=lambda t: -t[0])
    for acc, sub, w1, w2, nacc in res[:10]:
        print(f"ens {acc:.4f}  nn-mean {nacc:.4f}  w1={w1} w2={w2}  {'+'.join(sub)}", flush=True)
    acc, sub, w1, w2, _ = res[0]
    print(f"BEST {'+'.join(sub)} val_acc={acc:.4f} (iter9 val {ITER9_VAL})", flush=True)
    pd.DataFrame([dict(val_acc=a, nets="+".join(s), w1=x, w2=z, nn_mean=n) for a, s, x, z, n in res]
                 ).round(4).to_csv(f"{out_name}_subsets.csv", index=False)

    H, H_te = transform(C, P_POW, None), transform(C_te, P_POW, None)
    N_te = np.mean([T[n] for n in sub], axis=0)
    P, *_ = run(N_te, pipeline(None, H_te, H, y, nc), y, nc, w1=w1, w2=w2)
    img = pd.DataFrame({"id": test_ids, "label": P.argmax(1)})
    nn = pd.read_csv("iter8_nn.csv")
    sen = nn[nn["id"].str.startswith("text_")]
    sample = pd.read_csv("sample_submission.csv")
    out = sample[["id"]].merge(pd.concat([img, sen]), on="id", how="left", validate="one_to_one")
    assert len(out) == 10000 and out["label"].notna().all()
    out["label"] = out["label"].astype(int)
    assert out.loc[out["id"].str.startswith("img_"), "label"].between(0, 49).all()
    assert out.loc[out["id"].str.startswith("text_"), "label"].between(0, 2).all()
    out.to_csv(f"{out_name}.csv", index=False)
    it9 = pd.read_csv("iter9.csv").merge(out, on="id")
    print(f"Saved {out_name}.csv (agreement with iter9: {(it9.label_x == it9.label_y).mean():.4f})")


if __name__ == "__main__":
    main()
