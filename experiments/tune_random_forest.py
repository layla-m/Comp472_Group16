"""
Hyperparameter study for our Random Forest.

Outputs (results/trees/<task>/):
    rf_n_trees.csv / rf_n_trees.png
    rf_tuning.csv
    rf_max_features.png, rf_max_depth.png
    best_params_rf.json
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models.random_forest import RandomForestClassifier
from experiments.tree_utils import (best_params_path, load_split, load_json, out_dir, save_json,
                    set_seed, tasks_from_arg, timed_fit, SEED)


def evaluate(params, data, experiment, rows, jobs):
    X_tr, y_tr, X_va, y_va = data
    rf = RandomForestClassifier(random_state=SEED, n_jobs=jobs, oob_score=True, **params)
    fit_time = timed_fit(rf, X_tr, y_tr)
    row = {"experiment": experiment, **{k: str(v) for k, v in params.items()},
           "train_acc": rf.score(X_tr, y_tr), "val_acc": rf.score(X_va, y_va),
           "oob_acc": rf.oob_score_, "fit_time_s": round(fit_time, 1),
           "avg_leaves_per_tree": float(np.mean([t.n_leaves for t in rf.estimators_]))}
    rows.append(row)
    print(f"  {experiment:<16} {str(params):<110} train={row['train_acc']:.3f} "
          f"val={row['val_acc']:.3f} oob={row['oob_acc']:.3f} ({fit_time:.0f}s)")
    return row


def bar_plot(df, x, title, path):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    pos = np.arange(len(df))
    ax.bar(pos - 0.2, df["train_acc"], 0.4, label="train", color="#1f77b4")
    ax.bar(pos + 0.2, df["val_acc"], 0.4, label="validation", color="#d62728")
    ax.plot(pos + 0.2, df["oob_acc"], "k^", label="out-of-bag")
    ax.set_xticks(pos, df[x])
    ax.set_xlabel(x)
    ax.set_ylabel("accuracy")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def n_trees_curve(task, data, out, jobs, n_max, single_tree_val):
    X_tr, y_tr, X_va, y_va = data
    rf = RandomForestClassifier(n_estimators=n_max, random_state=SEED, n_jobs=jobs)
    fit_time = timed_fit(rf, X_tr, y_tr)
    ks, accs = [], []
    for k, proba in rf.staged_tree_probas(X_va):
        ks.append(k)
        accs.append(float(np.mean(np.argmax(proba, 1) == y_va)))
    pd.DataFrame({"n_trees": ks, "val_acc": accs}).to_csv(
        os.path.join(out, "rf_n_trees.csv"), index=False)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(ks, accs, color="#d62728", label="Random Forest (validation)")
    if single_tree_val is not None:
        ax.axhline(single_tree_val, ls="--", color="gray", label="best single Decision Tree")
    ax.set_xlabel("number of trees")
    ax.set_ylabel("validation accuracy")
    ax.set_title(f"Random Forest ({task}): accuracy vs number of trees")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "rf_n_trees.png"), dpi=150)
    plt.close(fig)
    for k in sorted({1, 10, 50, 100, n_max}):
        if k <= n_max:
            print(f"  n_trees={k:<4} val_acc={accs[k - 1]:.3f}")
    print(f"  (200-tree forest trained in {fit_time:.0f}s)")
    return accs


def run(task, quick, jobs):
    print(f"\n=== Random Forest tuning: {task} {'(QUICK)' if quick else ''}")
    set_seed()
    data = load_split(task, quick)
    out = out_dir(task)
    T = 30 if quick else 100           # trees per forest for experiments 2-4
    n_max = 50 if quick else 200

    try:
        single_tree_val = load_json(best_params_path(task, "dt"))["val_acc"]
    except FileNotFoundError:
        single_tree_val = None

    # 1) number of trees
    accs = n_trees_curve(task, data, out, jobs, n_max, single_tree_val)

    rows = []
    base = dict(n_estimators=T, max_depth=None, min_samples_leaf=1, max_features="sqrt")
    # 2) max_features
    for mf in ["log2", "sqrt", 0.1, 0.2]:
        evaluate({**base, "max_features": mf}, data, "max_features", rows, jobs)
    df = pd.DataFrame(rows)
    mf_df = df[df.experiment == "max_features"]
    best_mf = mf_df.loc[mf_df.val_acc.idxmax(), "max_features"]
    best_mf = float(best_mf) if best_mf.replace(".", "").isdigit() else best_mf

    # 3) max_depth
    for d in [5, 10, 20, None]:
        evaluate({**base, "max_features": best_mf, "max_depth": d}, data, "max_depth", rows, jobs)
    df = pd.DataFrame(rows)
    d_df = df[df.experiment == "max_depth"]
    best_d = d_df.loc[d_df.val_acc.idxmax(), "max_depth"]
    best_d = None if best_d == "None" else int(best_d)

    # 4) min_samples_leaf
    for leaf in [1, 5]:
        evaluate({**base, "max_features": best_mf, "max_depth": best_d,
                  "min_samples_leaf": leaf}, data, "min_samples_leaf", rows, jobs)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "rf_tuning.csv"), index=False)
    best = df.loc[df.val_acc.idxmax()]

    # number of trees for the final model: smallest k within 0.2% of the best
    accs = np.array(accs)
    n_final = int(np.argmax(accs >= accs.max() - 0.002) + 1)
    n_final = max(n_final, 50 if not quick else 10)

    def parse(v):
        if v == "None":
            return None
        try:
            f = float(v)
            return int(f) if f.is_integer() and "." not in v else f
        except ValueError:
            return v

    params = {k: parse(best[k]) for k in ("max_depth", "min_samples_leaf", "max_features")}
    params["n_estimators"] = n_final
    save_json({"params": params, "val_acc_at_tuning": best.val_acc, "oob_acc": best.oob_acc,
               "quick_run": quick}, best_params_path(task, "rf"))

    bar_plot(mf_df, "max_features", f"Random Forest ({task}): max_features",
             os.path.join(out, "rf_max_features.png"))
    bar_plot(d_df, "max_depth", f"Random Forest ({task}): max_depth of each tree",
             os.path.join(out, "rf_max_depth.png"))
    print(f"best RF for {task}: {params}  val_acc={best.val_acc:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["image", "sentiment", "both"], default="both")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--jobs", type=int, default=-1)
    a = p.parse_args()
    for t in tasks_from_arg(a.task):
        run(t, a.quick, a.jobs)
