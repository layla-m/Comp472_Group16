"""
Random Forest classifier
"""

import os
import numpy as np
from multiprocessing import Pool

from models.decision_tree import DecisionTreeClassifier, FeatureBinner


_SHARED = {}


def _init_worker(Xb, y, binner):
    _SHARED.update(Xb=Xb, y=y, binner=binner)


def _fit_one_tree(args):
    seed, tree_params, n_classes, bootstrap, max_samples = args
    Xb, y, binner = _SHARED["Xb"], _SHARED["y"], _SHARED["binner"]
    n = Xb.shape[0]
    rng = np.random.default_rng(seed)
    m = n if max_samples is None else max(1, int(max_samples * n))
    if bootstrap:
        idx = rng.integers(0, n, size=m)                 # with replacement
    else:
        idx = rng.permutation(n)[:m]                     # without replacement
    tree = DecisionTreeClassifier(random_state=seed, **tree_params)
    tree.fit(None, y, n_classes=n_classes, binner=binner, Xb=Xb, sample_indices=idx)
    tree.binner_ = None          # the forest keeps the single shared binner
    in_bag = np.zeros(n, dtype=bool)
    in_bag[idx] = True
    return tree, in_bag


class RandomForestClassifier:
    """
    Parameters
    ----------
    n_estimators : number of trees.
    max_depth, min_samples_split, min_samples_leaf, criterion, n_bins :
        passed to every DecisionTreeClassifier.
    max_features : features tried per split ("sqrt", "log2", int, float, None).
    bootstrap : draw each tree's training set with replacement.
    max_samples : fraction of n drawn for each tree (None = n samples).
    oob_score : compute the out-of-bag accuracy (a free validation estimate).
    n_jobs : number of processes used to build trees (-1 = all CPUs).
    random_state : seed for reproducibility.
    """

    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, criterion="gini", max_features="sqrt",
                 bootstrap=True, max_samples=None, oob_score=False, n_bins=64,
                 n_jobs=1, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        self.oob_score = oob_score
        self.n_bins = n_bins
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).astype(np.int64)
        self.n_classes_ = int(y.max()) + 1
        self.n_features_ = X.shape[1]

        # Bin once, share with all trees 
        self.binner_ = FeatureBinner(self.n_bins).fit(X)
        Xb = self.binner_.transform(X)
        _SHARED.update(Xb=Xb, y=y, binner=self.binner_)

        tree_params = dict(max_depth=self.max_depth,
                           min_samples_split=self.min_samples_split,
                           min_samples_leaf=self.min_samples_leaf,
                           criterion=self.criterion,
                           max_features=self.max_features,
                           n_bins=self.n_bins)
        seeds = np.random.SeedSequence(self.random_state).generate_state(self.n_estimators)
        jobs = [(int(s), tree_params, self.n_classes_, self.bootstrap, self.max_samples)
                for s in seeds]

        n_jobs = os.cpu_count() if self.n_jobs in (-1, None) else self.n_jobs
        if n_jobs > 1:
           
            with Pool(n_jobs, initializer=_init_worker,
                      initargs=(Xb, y, self.binner_)) as pool:
                results = pool.map(_fit_one_tree, jobs)
        else:
            results = [_fit_one_tree(j) for j in jobs]
        _SHARED.clear()

        self.estimators_ = [t for t, _ in results]
        in_bag = np.stack([b for _, b in results])            # (T, n)

        importances = np.mean([t.feature_importances_ for t in self.estimators_], axis=0)
        self.feature_importances_ = importances / max(importances.sum(), 1e-12)

        if self.oob_score:
            proba = np.zeros((X.shape[0], self.n_classes_))
            votes = np.zeros(X.shape[0])
            for tree, bag in zip(self.estimators_, in_bag):
                oob = ~bag
                if oob.any():
                    proba[oob] += tree.predict_proba(X[oob])
                    votes[oob] += 1
            has = votes > 0
            self.oob_score_ = float(np.mean(np.argmax(proba[has], 1) == y[has]))
        return self

    def predict_proba(self, X, n_trees=None):
        """Average class probabilities of the first `n_trees` trees (all by default)."""
        X = np.asarray(X, dtype=np.float64)
        trees = self.estimators_ if n_trees is None else self.estimators_[:n_trees]
        proba = np.zeros((X.shape[0], self.n_classes_))
        for tree in trees:
            proba += tree.predict_proba(X)
        return proba / len(trees)

    def staged_tree_probas(self, X):
        """Yield (k, averaged proba of the first k trees) for k = 1..T.
        Lets us plot accuracy vs number of trees from ONE trained forest."""
        X = np.asarray(X, dtype=np.float64)
        total = np.zeros((X.shape[0], self.n_classes_))
        for k, tree in enumerate(self.estimators_, start=1):
            total += tree.predict_proba(X)
            yield k, total / k

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))
