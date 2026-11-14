"""
Hyperparameter search for the MLP (one factor at a time around the base config).
Same stratified 80/20 split as neural_network.py. Results -> hp_results_<task>.csv

Usage:
    python hp_search.py image
    python hp_search.py sentiment
"""

import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from neural_network import CONFIG, SEED, Standardizer, load_task, predict, train_model

VARIANTS = {
    "image": [
        ("base", {}),
        ("no_sqrt", {"sqrt_features": False}),
        ("hidden=[1024]", {"hidden": [1024]}),
        ("hidden=[2048]", {"hidden": [2048]}),
        ("hidden=[2048,1024]", {"hidden": [2048, 1024]}),
        ("dropout=0.3", {"dropout": 0.3}),
        ("dropout=0.7", {"dropout": 0.7}),
        ("wd=0.05", {"weight_decay": 5e-2}),
        ("lr=3e-3", {"lr": 3e-3}),
        ("lr=3e-4", {"lr": 3e-4}),
    ],
    "sentiment": [
        ("base", {}),
        ("hidden=[256]", {"hidden": [256]}),
        ("hidden=[256,64]", {"hidden": [256, 64]}),
        ("hidden=[1024,256]", {"hidden": [1024, 256]}),
        ("hidden=[1024,512,128]", {"hidden": [1024, 512, 128]}),
        ("dropout=0.3", {"dropout": 0.3}),
        ("dropout=0.7", {"dropout": 0.7}),
        ("lr=1e-3", {"lr": 1e-3}),
        ("lr=1e-4", {"lr": 1e-4}),
        ("wd=0.1", {"weight_decay": 1e-1}),
        ("batch=64", {"batch_size": 64}),
        ("batch=256", {"batch_size": 256}),
    ],
}


def main(task):
    X, y, _, _ = load_task(task)
    n_classes = len(np.unique(y))
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    rows = []
    for name, change in VARIANTS[task]:
        cfg = {**CONFIG[task], **change}
        sc = Standardizer(cfg["sqrt_features"]).fit(X_tr)
        Xt, Xv = sc.transform(X_tr), sc.transform(X_va)
        t0 = time.time()
        model, best_ep, best_acc, _ = train_model(Xt, y_tr, n_classes, cfg, Xv, y_va, verbose=False)
        train_acc = (predict(model, Xt) == y_tr).mean()
        rows.append(dict(variant=name, val_acc=round(best_acc, 4), train_acc=round(train_acc, 4),
                         best_epoch=best_ep, seconds=round(time.time() - t0, 1)))
        print(rows[-1], flush=True)

    df = pd.DataFrame(rows).sort_values("val_acc", ascending=False)
    df.to_csv(f"hp_results_{task}.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
