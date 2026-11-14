"""
Measure on the 80/20 validation split how much seed / architecture ensembling
helps. Saves val probabilities of each (config, seed) to ens_<task>/ and prints
single-model vs ensemble accuracies.

Usage: python ens_eval.py sentiment|image
"""
import os
import sys
import itertools

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from neural_network import SEED, load_task, predict_proba
from nn_v2 import CONFIG, Prep, train_model

CONFIGS = {
    "sentiment": {
        "A_base": {},
        "B_[256,64]": {"hidden": [256, 64]},
        "C_[1024]": {"hidden": [1024]},
        "D_mixup0.2": {"mixup": 0.2},
    },
    "image": {
        "A_[2048,1024]_rn": {"row_norm": True},
        "B_[2048]_rn": {"row_norm": True, "hidden": [2048]},
    },
}
SEEDS = [0, 1, 2]


def main(task):
    torch.set_num_threads(int(os.environ.get("NTHREADS", 4)))
    X, y, _, _ = load_task(task)
    nc = len(np.unique(y))
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    d = f"ens_{task}"
    os.makedirs(d, exist_ok=True)
    probs = {}
    for name, ch in CONFIGS[task].items():
        cfg = {**CONFIG[task], **ch}
        p = Prep(cfg).fit(X_tr)
        Xt, Xv = p.transform(X_tr), p.transform(X_va)
        for s in SEEDS:
            f = f"{d}/{name}_s{s}.npy"
            if os.path.exists(f):
                probs[(name, s)] = np.load(f)
                continue
            m, ep, acc = train_model(Xt, y_tr, nc, cfg, Xv, y_va, seed=SEED + s)
            probs[(name, s)] = predict_proba(m, Xv)
            np.save(f, probs[(name, s)])
            print(f"{name} seed{s}: val={acc:.4f} ep={ep}", flush=True)
    acc = lambda P: (P.argmax(1) == y_va).mean()
    names = list(CONFIGS[task])
    for n in names:
        ens = sum(probs[(n, s)] for s in SEEDS)
        print(f"ENS {n} x{len(SEEDS)} seeds: {acc(ens):.4f}", flush=True)
    for k in range(2, len(names) + 1):
        for combo in itertools.combinations(names, k):
            ens = sum(probs[(n, s)] for n in combo for s in SEEDS)
            print(f"ENS {'+'.join(combo)}: {acc(ens):.4f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
