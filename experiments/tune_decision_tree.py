"""
Hyperparameter study for our Decision Tree.

Outputs (results/trees/<task>/):
    dt_tuning.csv            every configuration tried
    dt_depth.png             train vs validation accuracy vs max_depth
    dt_min_samples_leaf.png  same for min_samples_leaf
    dt_min_samples_split.png same for min_samples_split
    best_params_dt.json      chosen configuration
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from models.decision_tree import DecisionTreeClassifier
from experiments.tree_utils import (best_params_path, load_split, out_dir, save_json, set_seed,
                    tasks_from_arg, timed_fit, SEED)

DEPTHS = [2, 4, 6, 8, 10, 12, 15, 20, None]
LEAVES = [1, 2, 5, 10, 20, 50, 100]
SPLITS = [2, 10, 20, 50, 100, 200]
QUICK_DEPTHS = [2, 5, 10, None]
QUICK_LEAVES = [1, 10, 50]
QUICK_SPLITS = [2, 50]


def evaluate(params, data, experiment, rows):
    X_tr, y_tr, X_va, y_va = data
    tree = DecisionTreeClassifier(random_state=SEED, **params)
    fit_time = timed_fit(tree, X_tr, y_tr)
    row = {"experiment": experiment, **params,
           "train_acc": tree.score(X_tr, y_tr), "val_acc": tree.score(X_va, y_va),
           "n_leaves": tree.n_leaves, "actual_depth": tree.depth,
           "fit_time_s": round(fit_time, 2)}
    row["gap"] = row["train_acc"] - row["val_acc"]
    rows.append(row)
    print(f"  {experiment:<18} {str(params):<85} train={row['train_acc']:.3f} "
          f"val={row['val_acc']:.3f} leaves={row['n_leaves']} ({fit_time:.1f}s)")
    return row


def plot_curve(df, x, title, path, log=False):
    labels = ["None" if pd.isna(v) else str(int(v)) for v in df[x]]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(labels, df["train_acc"], "o-", label="train accuracy", color="#1f77b4")
    ax.plot(labels, df["val_acc"], "s-", label="validation accuracy", color="#d62728")
    ax.set_xlabel(x + (" (None = unlimited)" if x == "max_depth" else ""))
    ax.set_ylabel("accuracy")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(task, quick):
    print(f"\n=== Decision Tree tuning: {task} {'(QUICK)' if quick else ''}")
    set_seed()
    data = load_split(task, quick)
    out = out_dir(task)
    rows = []
    base = dict(max_depth=None, min_samples_split=2, min_samples_leaf=1, criterion="gini")

    # 1) depth: the main under/overfitting knob
    for d in (QUICK_DEPTHS if quick else DEPTHS):
        evaluate({**base, "max_depth": d}, data, "max_depth", rows)
    df = pd.DataFrame(rows)
    depth_df = df[df.experiment == "max_depth"]
    best_depth = depth_df.loc[depth_df.val_acc.idxmax(), "max_depth"]
    best_depth = None if pd.isna(best_depth) else int(best_depth)

    # 2) splitting criterion, at the best depth
    for crit in ("gini", "entropy"):
        evaluate({**base, "max_depth": best_depth, "criterion": crit}, data, "criterion", rows)
    df = pd.DataFrame(rows)
    crit_df = df[df.experiment == "criterion"]
    best_crit = crit_df.loc[crit_df.val_acc.idxmax(), "criterion"]

    # 3) min_samples_leaf and 4) min_samples_split on an unlimited-depth tree
    for leaf in (QUICK_LEAVES if quick else LEAVES):
        evaluate({**base, "criterion": best_crit, "min_samples_leaf": leaf}, data,
                 "min_samples_leaf", rows)
    for split in (QUICK_SPLITS if quick else SPLITS):
        evaluate({**base, "criterion": best_crit, "min_samples_split": split}, data,
                 "min_samples_split", rows)
    df = pd.DataFrame(rows)
    leaf_df = df[df.experiment == "min_samples_leaf"]
    split_df = df[df.experiment == "min_samples_split"]
    best_leaf = int(leaf_df.loc[leaf_df.val_acc.idxmax(), "min_samples_leaf"])
    best_split = int(split_df.loc[split_df.val_acc.idxmax(), "min_samples_split"])

    # 5) combine the best values
    combos = {(best_depth, best_leaf, best_split)}
    if best_depth is not None:
        combos.add((None, best_leaf, best_split))
    for d, leaf, split in combos:
        evaluate(dict(max_depth=d, min_samples_split=split, min_samples_leaf=leaf,
                      criterion=best_crit), data, "combined", rows)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "dt_tuning.csv"), index=False)
    best = df.loc[df.val_acc.idxmax()]
    best_params = {k: (None if pd.isna(best[k]) else
                       (str(best[k]) if k == "criterion" else int(best[k])))
                   for k in ("max_depth", "min_samples_split", "min_samples_leaf", "criterion")}
    save_json({"params": best_params, "train_acc": best.train_acc, "val_acc": best.val_acc,
               "quick_run": quick}, best_params_path(task, "dt"))

    plot_curve(depth_df, "max_depth", f"Decision Tree ({task}): effect of max_depth",
               os.path.join(out, "dt_depth.png"))
    plot_curve(leaf_df, "min_samples_leaf", f"Decision Tree ({task}): effect of min_samples_leaf",
               os.path.join(out, "dt_min_samples_leaf.png"))
    plot_curve(split_df, "min_samples_split",
               f"Decision Tree ({task}): effect of min_samples_split",
               os.path.join(out, "dt_min_samples_split.png"))
    print(f"best DT for {task}: {best_params}  val_acc={best.val_acc:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["image", "sentiment", "both"], default="both")
    p.add_argument("--quick", action="store_true", help="small fast run to test the code")
    a = p.parse_args()
    for t in tasks_from_arg(a.task):
        run(t, a.quick)
