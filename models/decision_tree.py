"""
Decision Tree classifier
"""

import numpy as np


# ----------------------------------------------------------------------------
# Feature binning
# ----------------------------------------------------------------------------
class FeatureBinner:
    """Cuts every feature into (at most) n_bins quantile bins.

    bin(x) = number of edges strictly smaller than x, so that
        bin(x) <= t   <=>   x <= edges[t]
    which lets us turn a split on bins back into a real threshold.
    """

    def __init__(self, n_bins=64):
        if not 2 <= n_bins <= 256:
            raise ValueError("n_bins must be between 2 and 256")
        self.n_bins = n_bins

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        qs = np.linspace(0.0, 1.0, self.n_bins + 1)[1:-1]
        # edges has shape (n_features, n_bins - 1)
        self.edges_ = np.quantile(X, qs, axis=0).T.copy()
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        Xb = np.empty(X.shape, dtype=np.uint8)
        for j in range(X.shape[1]):
            Xb[:, j] = np.searchsorted(self.edges_[j], X[:, j], side="left")
        return Xb

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# ----------------------------------------------------------------------------
# Impurity functions (vectorised over many candidate splits at once)
# ----------------------------------------------------------------------------
def gini_from_counts(counts, n):
    """Gini impurity = 1 - sum_k p_k^2 ; counts shape (..., C), n shape (...)."""
    n_safe = np.maximum(n, 1)
    p = counts / n_safe[..., None]
    return 1.0 - np.sum(p * p, axis=-1)


def entropy_from_counts(counts, n):
    """Entropy = - sum_k p_k log2 p_k (0 log 0 is treated as 0)."""
    n_safe = np.maximum(n, 1)
    p = counts / n_safe[..., None]
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(p > 0, np.log2(p), 0.0)
    return -np.sum(p * logp, axis=-1)


IMPURITY = {"gini": gini_from_counts, "entropy": entropy_from_counts}


# ----------------------------------------------------------------------------
# Decision tree
# ----------------------------------------------------------------------------
class DecisionTreeClassifier:
    """CART-style decision tree for multi-class classification.

    Parameters
    ----------
    max_depth : int or None
        Maximum depth of the tree (root = depth 0). None = grow until the
        other stopping rules apply.
    min_samples_split : int
        A node needs at least this many samples to be split.
    min_samples_leaf : int
        Each child created by a split must keep at least this many samples.
    criterion : {"gini", "entropy"}
        Impurity measure used to score splits.
    min_impurity_decrease : float
        A split is kept only if (weighted) impurity drops by at least this.
    max_features : None, int, float, "sqrt" or "log2"
        Number of features examined at each split. None = all features
        (classic decision tree). Random Forest uses a random subset.
    n_bins : int
        Number of quantile bins per feature for the histogram split search.
    random_state : int or None
        Seed for the random feature subsets (only matters if max_features).
    """

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 criterion="gini", min_impurity_decrease=0.0, max_features=None,
                 n_bins=64, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.n_bins = n_bins
        self.random_state = random_state

    # ------------------------------------------------------------------ fit
    def fit(self, X, y, n_classes=None, binner=None, Xb=None, sample_indices=None):
        """Train the tree.

        X : (n_samples, n_features) float array
        y : (n_samples,) integer labels 0..C-1
        n_classes : total number of classes (default: max(y) + 1)
        binner, Xb, sample_indices : used by RandomForest so that the binning
            is computed once and shared by all trees, and so that a bootstrap
            sample can be given as indices (with repeats) instead of a copy.
        """
        if self.criterion not in IMPURITY:
            raise ValueError("criterion must be 'gini' or 'entropy'")
        y = np.asarray(y).astype(np.int64)
        self.n_classes_ = int(n_classes) if n_classes is not None else int(y.max()) + 1
        self.classes_ = np.arange(self.n_classes_)

        if binner is None:
            binner = FeatureBinner(self.n_bins).fit(X)
            Xb = binner.transform(X)
        self.binner_ = binner
        self.n_features_ = Xb.shape[1]
        self._impurity = IMPURITY[self.criterion]
        self._rng = np.random.default_rng(self.random_state)
        self._k_features = self._resolve_max_features(self.n_features_)

        if sample_indices is None:
            sample_indices = np.arange(Xb.shape[0])

        # Tree stored as parallel lists (converted to arrays at the end).
        self._feature, self._threshold_bin, self._left, self._right = [], [], [], []
        self._value, self._n_node, self._impurity_node, self._depth = [], [], [], []
        self.feature_importances_ = np.zeros(self.n_features_)

        self._build(Xb, y, sample_indices)

        self.feature_ = np.array(self._feature, dtype=np.int64)
        tb = np.array(self._threshold_bin, dtype=np.int64)
        # Real-valued threshold: x <= edges[f, t]  <=>  bin(x) <= t
        self.threshold_ = np.where(
            self.feature_ >= 0,
            binner.edges_[np.maximum(self.feature_, 0), np.maximum(tb, 0)],
            np.nan)
        self.children_left_ = np.array(self._left, dtype=np.int64)
        self.children_right_ = np.array(self._right, dtype=np.int64)
        self.value_ = np.array(self._value)              # class probabilities
        self.n_node_samples_ = np.array(self._n_node)
        self.impurity_ = np.array(self._impurity_node)
        self.node_depth_ = np.array(self._depth)
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total
        # free the temporary lists
        del (self._feature, self._threshold_bin, self._left, self._right,
             self._value, self._n_node, self._impurity_node, self._depth)
        return self

    def _resolve_max_features(self, d):
        mf = self.max_features
        if mf is None:
            return d
        if mf == "sqrt":
            return max(1, int(np.sqrt(d)))
        if mf == "log2":
            return max(1, int(np.log2(d)))
        if isinstance(mf, float):
            return max(1, int(mf * d))
        return max(1, min(d, int(mf)))

    def _new_node(self, counts, n, depth):
        node_id = len(self._feature)
        self._feature.append(-1)
        self._threshold_bin.append(-1)
        self._left.append(-1)
        self._right.append(-1)
        self._value.append(counts / n)
        self._n_node.append(n)
        self._impurity_node.append(float(self._impurity(counts[None, :].astype(float),
                                                         np.array([n]))[0]))
        self._depth.append(depth)
        return node_id

    def _build(self, Xb, y, root_idx):
        """Grow the tree with an explicit stack (avoids Python recursion limits)."""
        C = self.n_classes_
        root_counts = np.bincount(y[root_idx], minlength=C).astype(float)
        root = self._new_node(root_counts, len(root_idx), 0)
        stack = [(root, root_idx, 0)]

        while stack:
            node, idx, depth = stack.pop()
            n = len(idx)
            counts = np.bincount(y[idx], minlength=C)

            # ---- stopping rules
            if (self.max_depth is not None and depth >= self.max_depth) \
                    or n < self.min_samples_split \
                    or n < 2 * self.min_samples_leaf \
                    or np.count_nonzero(counts) <= 1:
                continue

            split = self._best_split(Xb, y, idx, counts)
            if split is None:
                continue
            feat, t_bin, gain = split
            if gain < self.min_impurity_decrease or gain <= 1e-12:
                continue

            go_left = Xb[idx, feat] <= t_bin
            left_idx, right_idx = idx[go_left], idx[~go_left]

            self._feature[node] = feat
            self._threshold_bin[node] = t_bin
            self.feature_importances_[feat] += gain * n

            lc = np.bincount(y[left_idx], minlength=C).astype(float)
            rc = np.bincount(y[right_idx], minlength=C).astype(float)
            left = self._new_node(lc, len(left_idx), depth + 1)
            right = self._new_node(rc, len(right_idx), depth + 1)
            self._left[node], self._right[node] = left, right
            stack.append((right, right_idx, depth + 1))
            stack.append((left, left_idx, depth + 1))

    def _best_split(self, Xb, y, idx, counts):
        """Return (feature, threshold_bin, impurity_decrease) or None."""
        B = self.binner_.n_bins
        n = len(idx)

        if self._k_features < self.n_features_:
            feats = self._rng.choice(self.n_features_, self._k_features, replace=False)
        else:
            feats = np.arange(self.n_features_)
        f = len(feats)

    
        present = np.flatnonzero(counts)
        C = len(present)
        local = np.zeros(self.n_classes_, dtype=np.int64)
        local[present] = np.arange(C)
        counts = counts[present]

        # Histogram: hist[j, b, k] = #samples of class k in bin b of feature j
        sub = Xb[idx][:, feats].astype(np.int64)                 # (n, f)
        flat = (np.arange(f, dtype=np.int64) * (B * C))[None, :] + sub * C \
            + local[y[idx]][:, None]
        hist = np.bincount(flat.ravel(), minlength=f * B * C).reshape(f, B, C)

        # Left counts for threshold "bin <= t", for t = 0 .. B-2
        left = np.cumsum(hist, axis=1)[:, :-1, :].astype(np.float64)  # (f, B-1, C)
        right = counts[None, None, :] - left
        n_left = left.sum(axis=2)
        n_right = n - n_left

        valid = (n_left >= self.min_samples_leaf) & (n_right >= self.min_samples_leaf)
        if not valid.any():
            return None

        parent = self._impurity(counts[None, :].astype(float), np.array([n]))[0]
        child = (n_left * self._impurity(left, n_left)
                 + n_right * self._impurity(right, n_right)) / n
        gain = np.where(valid, parent - child, -np.inf)

        j, t = np.unravel_index(np.argmax(gain), gain.shape)
        return int(feats[j]), int(t), float(gain[j, t])

    # ------------------------------------------------------------- predict
    def apply(self, X):
        """Index of the leaf each sample ends up in."""
        X = np.asarray(X, dtype=np.float64)
        node = np.zeros(X.shape[0], dtype=np.int64)
        rows = np.arange(X.shape[0])
        while True:
            internal = self.feature_[node] >= 0
            if not internal.any():
                return node
            r = rows[internal]
            nd = node[internal]
            go_left = X[r, self.feature_[nd]] <= self.threshold_[nd]
            node[internal] = np.where(go_left, self.children_left_[nd],
                                      self.children_right_[nd])

    def predict_proba(self, X):
        return self.value_[self.apply(X)]

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))

    # --------------------------------------------------------------- info
    @property
    def n_nodes(self):
        return len(self.feature_)

    @property
    def n_leaves(self):
        return int(np.sum(self.feature_ < 0))

    @property
    def depth(self):
        return int(self.node_depth_.max())
