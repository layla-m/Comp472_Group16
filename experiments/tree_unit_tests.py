"""
Sanity tests for our Decision Tree and Random Forest.

"""

import time
import numpy as np

from models.decision_tree import DecisionTreeClassifier, gini_from_counts, entropy_from_counts
from models.random_forest import RandomForestClassifier


def brute_force_best_split(X, y, criterion):
    """Exhaustive search over every feature and every midpoint threshold."""
    imp = gini_from_counts if criterion == "gini" else entropy_from_counts
    C = y.max() + 1
    n = len(y)
    parent = imp(np.bincount(y, minlength=C)[None].astype(float), np.array([n]))[0]
    best = -np.inf
    for j in range(X.shape[1]):
        for t in np.unique(X[:, j])[:-1]:
            L, R = y[X[:, j] <= t], y[X[:, j] > t]
            cl = np.bincount(L, minlength=C)[None].astype(float)
            cr = np.bincount(R, minlength=C)[None].astype(float)
            child = (len(L) * imp(cl, np.array([len(L)]))[0]
                     + len(R) * imp(cr, np.array([len(R)]))[0]) / n
            best = max(best, parent - child)
    return best


def test_root_split_matches_brute_force():
    rng = np.random.default_rng(0)
    # few distinct values per feature -> binning is exact
    X = rng.integers(0, 20, size=(300, 6)).astype(float)
    y = (X[:, 2] + rng.integers(0, 6, 300) > 14).astype(int) + (X[:, 4] > 12)
    for crit in ("gini", "entropy"):
        tree = DecisionTreeClassifier(max_depth=1, criterion=crit, n_bins=64).fit(X, y)
        ours = tree.impurity_[0] - (
            tree.n_node_samples_[1] * tree.impurity_[1]
            + tree.n_node_samples_[2] * tree.impurity_[2]) / tree.n_node_samples_[0]
        brute = brute_force_best_split(X, y, crit)
        assert abs(ours - brute) < 1e-9, (crit, ours, brute)
    print("ok  root split = exhaustive search (gini and entropy)")


def test_full_tree_memorises_training_data():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 10))
    y = rng.integers(0, 5, 500)            # random labels: pure memorisation
    tree = DecisionTreeClassifier(n_bins=256).fit(X, y)
    acc = tree.score(X, y)
    assert acc > 0.95, acc
    print(f"ok  unlimited-depth tree fits random labels (train acc = {acc:.3f})")


def test_stopping_rules():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(1000, 8))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    t = DecisionTreeClassifier(max_depth=3).fit(X, y)
    assert t.depth <= 3
    t = DecisionTreeClassifier(min_samples_leaf=50).fit(X, y)
    assert t.n_node_samples_[t.feature_ < 0].min() >= 50
    t = DecisionTreeClassifier(max_depth=0).fit(X, y)
    assert t.n_nodes == 1
    print("ok  max_depth / min_samples_leaf / depth-0 respected")


def test_forest_beats_single_tree():
    rng = np.random.default_rng(3)
    d, C = 50, 5
    centers = rng.normal(size=(C, d))
    y = rng.integers(0, C, 3000)
    X = centers[y] + rng.normal(0, 2.0, size=(3000, d))
    Xtr, ytr, Xva, yva = X[:2000], y[:2000], X[2000:], y[2000:]
    tree = DecisionTreeClassifier(random_state=0).fit(Xtr, ytr)
    forest = RandomForestClassifier(n_estimators=50, oob_score=True,
                                    random_state=0, n_jobs=2).fit(Xtr, ytr)
    t_acc, f_acc = tree.score(Xva, yva), forest.score(Xva, yva)
    assert f_acc > t_acc
    print(f"ok  forest beats tree on validation ({f_acc:.3f} vs {t_acc:.3f}), "
          f"OOB = {forest.oob_score_:.3f}")


def test_reproducible():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(600, 20))
    y = rng.integers(0, 3, 600)
    a = RandomForestClassifier(n_estimators=10, random_state=7).fit(X, y).predict(X)
    b = RandomForestClassifier(n_estimators=10, random_state=7, n_jobs=2).fit(X, y).predict(X)
    assert np.array_equal(a, b)
    print("ok  same seed -> same forest (also with multiprocessing)")


if __name__ == "__main__":
    start = time.time()
    test_root_split_matches_brute_force()
    test_full_tree_memorises_training_data()
    test_stopping_rules()
    test_forest_beats_single_tree()
    test_reproducible()
    print(f"all tests passed in {time.time() - start:.1f}s")
