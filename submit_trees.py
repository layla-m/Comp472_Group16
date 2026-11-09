"""
Kaggle submission with the tuned tree models.

Output: results/<model>_submission.csv  (e.g. results/rf_submission.csv)
"""

import argparse
import os

import pandas as pd

from models.decision_tree import DecisionTreeClassifier
from models.random_forest import RandomForestClassifier
from experiments.tree_utils import (SEED, best_params_path, load_full_train_and_test,
                                    load_json, set_seed)


def main(model_name, jobs):
    set_seed()
    parts = []
    for task in ("image", "sentiment"):
        params = load_json(best_params_path(task, model_name))["params"]
        X, y, test_ids, X_test = load_full_train_and_test(task)
        if model_name == "dt":
            model = DecisionTreeClassifier(random_state=SEED, **params)
        else:
            model = RandomForestClassifier(random_state=SEED, n_jobs=jobs, **params)
        print(f"Training {model_name.upper()} on all {len(y)} {task} rows {params} ...")
        model.fit(X, y)
        parts.append(pd.DataFrame({"id": test_ids, "label": model.predict(X_test).astype(int)}))

    # Same merging and checks 
    sample = pd.read_csv(os.path.join("data", "sample_submission.csv"))
    preds = pd.concat(parts, ignore_index=True)
    sub = sample[["id"]].merge(preds, on="id", how="left", validate="one_to_one")
    assert len(sub) == len(sample)
    assert sub["label"].notna().all(), "some test ids have no prediction"
    sub["label"] = sub["label"].astype(int)
    assert sub.loc[sub["id"].str.startswith("img_"), "label"].between(0, 49).all()
    assert sub.loc[sub["id"].str.startswith("text_"), "label"].between(0, 2).all()

    os.makedirs("results", exist_ok=True)
    path = os.path.join("results", f"{model_name}_submission.csv")
    sub.to_csv(path, index=False)
    print(f"Kaggle submission saved: {path} ({len(sub)} predictions)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["dt", "rf"], default="rf")
    p.add_argument("--jobs", type=int, default=-1, help="CPU processes for the forest")
    a = p.parse_args()
    main(a.model, a.jobs)
