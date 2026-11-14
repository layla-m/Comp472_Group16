"""
Hyperparameter search, iteration 2 (uses nn_v2.py). Same 80/20 split.
Each finished variant is appended to hp2_results_<task>.csv.

Usage:
    python hp_search2.py image [round]
    python hp_search2.py sentiment [round]
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from neural_network import SEED, load_task, predict_proba
from nn_v2 import CONFIG, Prep, aug_args, train_model

IMG = {**CONFIG["image"], "hidden": [2048]}   # cheaper base for screening
SEN = CONFIG["sentiment"]

VARIANTS = {
    ("image", "1"): [
        ("base[2048]", {}),
        ("row_norm", {"row_norm": True}),
        ("in_drop=0.2", {"in_dropout": 0.2}),
        ("in_drop=0.4", {"in_dropout": 0.4}),
        ("mixup=0.2", {"mixup": 0.2, "max_epochs": 100, "patience": 15}),
        ("mixup=0.4", {"mixup": 0.4, "max_epochs": 100, "patience": 15}),
        ("smooth=0.2", {"smoothing": 0.2}),
        ("hidden=[4096]", {"hidden": [4096]}),
        ("bs=256,lr=2e-3", {"batch_size": 256, "lr": 2e-3}),
    ],
    ("image", "2"): [
        ("in_drop=0.5", {"in_dropout": 0.5}),
        ("in_drop=0.6", {"in_dropout": 0.6}),
        ("in_drop=0.7", {"in_dropout": 0.7}),
        ("in_drop=0.5,rn", {"in_dropout": 0.5, "row_norm": True}),
        ("in_drop=0.5,[2048,1024]", {"in_dropout": 0.5, "hidden": [2048, 1024]}),
        ("in_drop=0.5,[4096]", {"in_dropout": 0.5, "hidden": [4096]}),
        ("in_drop=0.5,d=0.3", {"in_dropout": 0.5, "dropout": 0.3}),
    ],
    ("image", "3a"): [
        ("[4096],in=0.5,d=0.3", {"hidden": [4096], "in_dropout": 0.5, "dropout": 0.3}),
        ("[8192],in=0.5", {"hidden": [8192], "in_dropout": 0.5}),
        ("[4096],in=0.4", {"hidden": [4096], "in_dropout": 0.4}),
    ],
    ("image", "3b"): [
        ("[4096],in=0.5,ep=100", {"hidden": [4096], "in_dropout": 0.5, "max_epochs": 100, "patience": 15}),
        ("[4096],in=0.5,mixup=0.2", {"hidden": [4096], "in_dropout": 0.5, "mixup": 0.2,
                                      "max_epochs": 100, "patience": 15}),
        ("[4096],in=0.5,wd=0.05", {"hidden": [4096], "in_dropout": 0.5, "weight_decay": 5e-2}),
    ],
    # piste 2: transform the image features (base = iter2 config: [4096], in_dropout 0.5, dropout 0.3)
    ("image", "4"): [
        ("tfidf", {"tfidf": True}),
        ("tfidf+row_norm", {"tfidf": True, "row_norm": True}),
        ("pca_whiten_0.99", {"pca": 0.99}),
        ("pca_whiten_0.99,in=0.2", {"pca": 0.99, "in_dropout": 0.2}),
    ],
    # iter5: data augmentation on the visual-word counts + SWA (base = iter2 config)
    ("image", "5"): [
        ("thin=0.3", {"thin": 0.3}),
        ("thin=0.5", {"thin": 0.5}),
        ("thin=0.5,in=0.3", {"thin": 0.5, "in_dropout": 0.3}),
        ("noise=0.3", {"noise": 0.3}),
        ("swa_from_40", {"swa_start": 40, "patience": 60}),
    ],
    # test B: Hellinger features (L1 norm + sqrt) WITHOUT per-feature standardisation
    ("image", "6"): [
        ("hellinger_nostd", {"row_norm": True, "std": False}),
        ("hellinger_nostd,in=0.3", {"row_norm": True, "std": False, "in_dropout": 0.3}),
        ("sqrtL2_nostd", {"std": False}),
    ],
    ("sentiment", "1"): [
        ("base", {}),
        ("in_drop=0.1", {"in_dropout": 0.1}),
        ("in_drop=0.2", {"in_dropout": 0.2}),
        ("l2_norm", {"l2_norm": True}),
        ("mixup=0.2", {"mixup": 0.2}),
        ("mixup=0.4", {"mixup": 0.4}),
        ("smooth=0.2", {"smoothing": 0.2}),
        ("smooth=0.0", {"smoothing": 0.0}),
        ("[256,64],d=0.7", {"hidden": [256, 64]}),
        ("[1024],d=0.7", {"hidden": [1024]}),
        ("wd=0.05", {"weight_decay": 5e-2}),
        ("lr=2e-4", {"lr": 2e-4}),
    ],
}


def main(task, rnd):
    torch.set_num_threads(int(os.environ.get("NTHREADS", 4)))
    base = IMG if task == "image" else SEN
    if (task, rnd) in {("image", "4"), ("image", "5"), ("image", "6")}:
        base = CONFIG["image"]
    X, y, _, _ = load_task(task)
    n_classes = len(np.unique(y))
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    out = f"hp2_results_{task}.csv"
    for name, change in VARIANTS[(task, rnd)]:
        cfg = {**base, **change}
        p = Prep(cfg).fit(X_tr)
        t0 = time.time()
        m, ep, acc = train_model(p.transform(X_tr), y_tr, n_classes, cfg, p.transform(X_va), y_va,
                                 **aug_args(cfg, X_tr, p))
        row = dict(round=rnd, variant=name, val_acc=round(float(acc), 4), best_epoch=ep,
                   seconds=round(time.time() - t0, 1), cfg=str(change))
        print(row, flush=True)
        os.makedirs("ens", exist_ok=True)
        np.save(f"ens/hp_{task}_{rnd}_{name}.npy", predict_proba(m, p.transform(X_va)))
        pd.DataFrame([row]).to_csv(out, mode="a", header=not os.path.exists(out), index=False)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "1")
