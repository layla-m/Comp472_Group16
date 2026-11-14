"""
Reproducibility check: regenerate everything that does not need to retrain a network and
compare it with the saved results (probabilities, search tables, submissions).

The networks themselves take hours to retrain; their equivalence with the original scripts is
covered by tests/test_equivalence.py (same weights after a few epochs).
"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import SEARCH_DIR
from .recipes import RECIPES, RecipeContext, make_submission
from .search import knn_grid, knn_search
from .store import ProbabilityStore
from .submission import agreement, load_submission


def _report(name, ok, detail=""):
    print(f"  [{'OK ' if ok else 'DIFF'}] {name} {detail}", flush=True)
    return ok


def verify_knn_grid(store):
    with tempfile.TemporaryDirectory() as tmp:
        tmp_store = ProbabilityStore(Path(tmp))
        df = knn_grid(tmp_store, save=False)
        ok = _report("knn_results_image.csv", df.reset_index(drop=True).equals(
            pd.read_csv(SEARCH_DIR / "knn_results_image.csv")))
        for part in ("val", "test"):
            same = np.array_equal(tmp_store.load("knn", "image", part), store.load("knn", "image", part))
            ok &= _report(f"knn_image_{part}.npy", same)
    return ok


def verify_knn_search():
    df = knn_search(save=False).reset_index(drop=True)
    ref = pd.read_csv(SEARCH_DIR / "knn2_results_image.csv")
    cols = ["step", "p", "idf", "k", "tau", "knn_acc", "w_nn", "ens_acc", "thr", "pw"]
    same = np.allclose(df[cols].fillna(-1).astype(float).values, ref[cols].fillna(-1).astype(float).values)
    return _report("knn2_results_image.csv", same)


def verify_recipes(names=None):
    ctx, ok = RecipeContext(log=lambda *a: None), True
    for name in names or RECIPES:
        sub, res = make_submission(name, ctx, save=False)
        a = agreement(sub, load_submission(name))
        ok &= _report(f"{name}.csv", a == 1.0, f"(identical labels: {a:.4%}, val image {res.val_acc:.4f})")
    return ok


def verify_all(fast=False):
    store = ProbabilityStore()
    print("Submissions rebuilt from the saved probabilities:")
    ok = verify_recipes()
    if not fast:
        print("First Hellinger k-NN (grid):")
        ok &= verify_knn_grid(store)
        print("Tuned k-NN search:")
        ok &= verify_knn_search()
    print("ALL IDENTICAL" if ok else "SOME RESULTS DIFFER")
    return ok
