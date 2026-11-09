"""
Error analysis for the Decision Tree (DT) and Random Forest (RF).

"""

import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models.decision_tree import DecisionTreeClassifier
from models.random_forest import RandomForestClassifier
from experiments.tree_utils import (N_CLASSES, SENTIMENT_NAMES, SEED, accuracy,
                                    best_params_path, confusion_matrix, load_json,
                                    load_split, most_confused_pairs, out_dir,
                                    per_class_report, set_seed)


def plot_confusion(cm, names, title, path, annotate):
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)   # row = true class
    size = 5 if annotate else 9
    fig, ax = plt.subplots(figsize=(size, size * 0.9))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_title(title)
    if annotate:
        ax.set_xticks(range(len(names)), names)
        ax.set_yticks(range(len(names)), names)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{norm[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center",
                        color="white" if norm[i, j] > 0.5 else "black", fontsize=9)
    else:
        ticks = range(0, len(names), 5)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
    fig.colorbar(im, ax=ax, fraction=0.046, label="share of the true class")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(task, jobs):
    print(f"\n=== Error analysis: {task}")
    set_seed()
    out = out_dir(task)
    C = N_CLASSES[task]
    names = SENTIMENT_NAMES if task == "sentiment" else [str(i) for i in range(C)]
    X_tr, y_tr, X_va, y_va = load_split(task)
    os.makedirs("results", exist_ok=True)

    dt_params = load_json(best_params_path(task, "dt"))["params"]
    rf_params = load_json(best_params_path(task, "rf"))["params"]
    models = {
        "dt": DecisionTreeClassifier(random_state=SEED, **dt_params),
        "rf": RandomForestClassifier(random_state=SEED, n_jobs=jobs, oob_score=True,
                                     **rf_params),
    }

    res, lines = {}, [f"# Trees error analysis - {task}\n",
                      f"DT params: `{dt_params}`  ", f"RF params: `{rf_params}`\n"]
    for key, model in models.items():
        t0 = time.perf_counter()
        model.fit(X_tr, y_tr)
        fit_t = time.perf_counter() - t0
        t0 = time.perf_counter()
        proba = model.predict_proba(X_va)
        pred_t = time.perf_counter() - t0
        pred = np.argmax(proba, axis=1)
        train_acc = model.score(X_tr, y_tr)
        res[key] = dict(pred=pred, conf=proba.max(axis=1), proba=proba)

        # val_index = row number in the shared validation split (same order as k-NN)
        pd.DataFrame({"val_index": np.arange(len(y_va)), "true_label": y_va,
                      "pred_label": pred, "confidence": proba.max(axis=1).round(4)}) \
            .to_csv(os.path.join(out, f"val_predictions_{key}.csv"), index=False)
        np.save(os.path.join("results", f"{task}_{key}_predictions.npy"), pred)
        rep = per_class_report(y_va, pred, C, names)
        rep.to_csv(os.path.join(out, f"per_class_{key}.csv"), index=False)
        cm = confusion_matrix(y_va, pred, C)
        np.save(os.path.join("results", f"{task}_{key}_confusion_matrix.npy"), cm)
        plot_confusion(cm, names, f"{key.upper()} ({task}) - normalised confusion matrix",
                       os.path.join(out, f"confusion_{key}.png"), annotate=(C <= 10))
        pairs = most_confused_pairs(cm, top=10)
        if task == "sentiment" and len(pairs):
            pairs["true"] = [names[i] for i in pairs["true"]]
            pairs["predicted"] = [names[i] for i in pairs["predicted"]]
        pairs.to_csv(os.path.join(out, f"confused_pairs_{key}.csv"), index=False)

        if key == "dt":
            size = f"{model.n_nodes} nodes, {model.n_leaves} leaves, depth {model.depth}"
        else:
            n_nodes = sum(t.n_nodes for t in model.estimators_)
            size = (f"{len(model.estimators_)} trees, {n_nodes} nodes in total, "
                    f"OOB accuracy {model.oob_score_:.4f}")
        val_acc = accuracy(y_va, pred)
        lines += [f"## {key.upper()}",
                  f"- train accuracy: **{train_acc:.4f}**, validation accuracy: "
                  f"**{val_acc:.4f}** (gap {train_acc - val_acc:+.4f})",
                  f"- macro F1 (validation): {rep.f1.mean():.4f}",
                  f"- fit time {fit_t:.1f}s, predict time {pred_t:.2f}s "
                  f"for {len(y_va)} samples; size: {size}",
                  "- worst classes by recall: " + ", ".join(
                      f"{r['class']} ({r.recall:.2f})"
                      for _, r in rep.sort_values("recall").head(5).iterrows()),
                  "- best classes by recall: " + ", ".join(
                      f"{r['class']} ({r.recall:.2f})"
                      for _, r in rep.sort_values("recall", ascending=False).head(5).iterrows()),
                  "- most confused pairs (true -> predicted, count): " + ", ".join(
                      f"{r['true']}->{r['predicted']} ({r['count']})"
                      for _, r in pairs.head(5).iterrows()), ""]
        print(f"  {key}: train={train_acc:.4f} val={val_acc:.4f}")
        if task == "sentiment":
            lines.insert(-1, "- per-class recall: " + ", ".join(
                f"{r['class']} {r.recall:.3f}" for _, r in rep.iterrows()))
        res[key]["report"] = rep
        res[key]["model"] = model

    # ---- DT vs RF agreement
    dt_ok, rf_ok = res["dt"]["pred"] == y_va, res["rf"]["pred"] == y_va
    both_wrong = ~dt_ok & ~rf_ok
    same_mistake = both_wrong & (res["dt"]["pred"] == res["rf"]["pred"])
    agree = pd.DataFrame({
        "case": ["both correct", "only RF correct", "only DT correct", "both wrong",
                 "both wrong with the SAME wrong label"],
        "count": [int((dt_ok & rf_ok).sum()), int((~dt_ok & rf_ok).sum()),
                  int((dt_ok & ~rf_ok).sum()), int(both_wrong.sum()), int(same_mistake.sum())]})
    agree["share"] = (agree["count"] / len(y_va)).round(4)
    agree.to_csv(os.path.join(out, "agreement.csv"), index=False)
    table = ["| case | count | share |", "|---|---|---|"] + [
        f"| {r['case']} | {r['count']} | {r['share']:.2%} |" for _, r in agree.iterrows()]
    lines += ["## DT vs RF"] + table + [
              f"\nOf the samples the DT gets wrong, the RF fixes "
              f"{(~dt_ok & rf_ok).sum() / max((~dt_ok).sum(), 1):.1%}.", ""]

    # ---- comparison with k-NN (Person A), if its predictions are available
    knn_path = os.path.join("results", f"{task}_knn_predictions.npy")
    if os.path.exists(knn_path):
        knn = np.load(knn_path)
        if len(knn) == len(y_va):
            knn_ok = knn == y_va
            rows = []
            for key in ("dt", "rf"):
                ok = res[key]["pred"] == y_va
                wrong_both = ~ok & ~knn_ok
                rows.append({
                    "tree_model": key.upper(), "knn_val_acc": round(float(knn_ok.mean()), 4),
                    "tree_val_acc": round(float(ok.mean()), 4),
                    "both_correct": int((ok & knn_ok).sum()),
                    "only_knn_correct": int((~ok & knn_ok).sum()),
                    "only_tree_correct": int((ok & ~knn_ok).sum()),
                    "both_wrong": int(wrong_both.sum()),
                    "both_wrong_same_label": int((wrong_both & (res[key]["pred"] == knn)).sum())})
            kdf = pd.DataFrame(rows)
            kdf.to_csv(os.path.join(out, "agreement_with_knn.csv"), index=False)
            lines += ["## Comparison with k-NN",
                      f"k-NN validation accuracy: {knn_ok.mean():.4f}  "]
            for r in rows:
                lines.append(
                    f"- {r['tree_model']}: both correct {r['both_correct']}, only k-NN correct "
                    f"{r['only_knn_correct']}, only {r['tree_model']} correct "
                    f"{r['only_tree_correct']}, both wrong {r['both_wrong']} "
                    f"(same wrong label: {r['both_wrong_same_label']})")
            lines.append("")
            print("  compared with k-NN predictions")
        else:
            print(f"  {knn_path} has a different length than the validation set: skipped")
    else:
        print("  (no k-NN predictions found in results/: comparison with k-NN skipped)")

    # ---- per-class recall DT vs RF
    fig, ax = plt.subplots(figsize=(9 if C > 10 else 5, 4))
    pos = np.arange(C)
    ax.bar(pos - 0.2, res["dt"]["report"].recall, 0.4, label="Decision Tree", color="#ff7f0e")
    ax.bar(pos + 0.2, res["rf"]["report"].recall, 0.4, label="Random Forest", color="#2ca02c")
    ax.set_xticks(pos if C <= 10 else pos[::5], names if C <= 10 else pos[::5])
    ax.set_xlabel("class")
    ax.set_ylabel("recall (validation)")
    ax.set_title(f"Per-class recall ({task})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "per_class_recall_dt_vs_rf.png"), dpi=150)
    plt.close(fig)

    # ---- confidence of correct vs wrong predictions (RF)
    conf = res["rf"]["conf"]
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, 1, 26)
    ax.hist(conf[rf_ok], bins, alpha=0.6, label="correct", color="#2ca02c", density=True)
    ax.hist(conf[~rf_ok], bins, alpha=0.6, label="wrong", color="#d62728", density=True)
    ax.set_xlabel("RF confidence = share of trees' probability for the predicted class")
    ax.set_ylabel("density")
    ax.set_title(f"Random Forest confidence ({task})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "confidence.png"), dpi=150)
    plt.close(fig)
    lines += ["## Confidence (RF)",
              f"- mean confidence when correct: {conf[rf_ok].mean():.3f}, "
              f"when wrong: {conf[~rf_ok].mean():.3f}", ""]

    # ---- feature importance
    lines.append("## Feature importance")
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, color in (("dt", "#ff7f0e"), ("rf", "#2ca02c")):
        imp = np.sort(res[key]["model"].feature_importances_)[::-1]
        cum = np.cumsum(imp)
        ax.plot(np.arange(1, len(imp) + 1), cum, color=color,
                label="Decision Tree" if key == "dt" else "Random Forest")
        n50, n90 = int(np.searchsorted(cum, 0.5) + 1), int(np.searchsorted(cum, 0.9) + 1)
        used = int((imp > 0).sum())
        top = np.argsort(res[key]["model"].feature_importances_)[::-1][:5]
        lines.append(f"- {key.upper()}: {used}/{len(imp)} features used at least once; "
                     f"{n50} features give 50% of the importance, {n90} give 90%. "
                     f"Top 5 features: {', '.join('feature_' + str(i) for i in top)}")
    ax.set_xlabel("number of features (sorted by importance)")
    ax.set_ylabel("cumulative importance")
    ax.set_title(f"How many features do the trees rely on? ({task})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "feature_importance.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  written: {os.path.join(out, 'summary.md')}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["image", "sentiment", "both"], default="both")
    p.add_argument("--jobs", type=int, default=-1)
    a = p.parse_args()
    for t in (["image", "sentiment"] if a.task == "both" else [a.task]):
        run(t, a.jobs)
