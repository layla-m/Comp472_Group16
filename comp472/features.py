"""
Feature transforms.

- `FeaturePipeline`: input of the MLP (row-wise transforms + standardisation fitted on the
  training subset only), driven by an `MLPConfig`.
- `TorchFeaturePipeline`: the same transform in torch, for augmentations applied batch by batch.
- `Hellinger`, `PowerNormalize`: geometry of the k-NN (interchangeable strategies).
"""
import numpy as np
import torch

from .config import SEED
from .data import to_counts


def smooth_idf(X):
    """idf of each visual word, smooth version as in scikit-learn: rare words weigh more."""
    df = (np.asarray(X) > 0).sum(0)
    return np.log((1 + len(X)) / (1 + df)) + 1


class FeaturePipeline:
    """tfidf -> row_norm -> sqrt -> l2_norm -> standardisation (or global scale) -> PCA."""

    def __init__(self, cfg):
        self.cfg = cfg

    def _rowwise(self, X):
        cfg = self.cfg
        X = np.asarray(X, np.float32)
        if cfg.tfidf:
            X = np.clip(X, 0, None) * self.idf
        if cfg.row_norm:
            X = np.clip(X, 0, None)
            X = X / (X.sum(1, keepdims=True) + 1e-8)
        if cfg.sqrt_features:
            X = np.sqrt(np.clip(X, 0, None))
        if cfg.l2_norm:
            X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        return X

    def fit(self, X):
        cfg = self.cfg
        if cfg.tfidf:
            self.idf = smooth_idf(X).astype(np.float32)
        Z = self._rowwise(X)
        self.mu, self.sd = Z.mean(0), Z.std(0) + 1e-6
        if not cfg.standardize:
            # only a global scale so that the mean squared value of a feature is about 1
            scale = cfg.scale or np.sqrt(Z.shape[1])
            self.mu = np.zeros(Z.shape[1], np.float32)
            self.sd = np.full(Z.shape[1], 1.0 / scale, np.float32)
        if cfg.pca:
            from sklearn.decomposition import PCA
            self.pca = PCA(n_components=cfg.pca, whiten=True, random_state=SEED)
            self.pca.fit((Z - self.mu) / self.sd)
        return self

    def transform(self, X):
        Z = ((self._rowwise(X) - self.mu) / self.sd).astype(np.float32)
        if self.cfg.pca:
            Z = self.pca.transform(Z).astype(np.float32)
        return Z

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class TorchFeaturePipeline:
    """L2 norm of the counts -> sqrt -> standardise, in torch (for the `thin` augmentation:
    counts are thinned batch by batch, then transformed like the clean features)."""

    def __init__(self, pipeline):
        cfg = pipeline.cfg
        assert not (cfg.tfidf or cfg.pca or cfg.row_norm or cfg.l2_norm)
        self.sqrt = cfg.sqrt_features
        self.mu, self.sd = torch.from_numpy(pipeline.mu), torch.from_numpy(pipeline.sd)

    def __call__(self, K):
        x = K / (K.norm(dim=1, keepdim=True) + 1e-8)
        if self.sqrt:
            x = torch.sqrt(x)
        return (x - self.mu) / self.sd


def augmentation_inputs(cfg, X_raw, pipeline):
    """Extra trainer inputs needed by the `thin` augmentation (raw counts + torch transform)."""
    if cfg.thin > 0:
        return dict(counts=to_counts(X_raw).astype(np.float32), tprep=TorchFeaturePipeline(pipeline))
    return {}


# ------------------------------------------------------------------ k-NN geometries (strategies)
class Hellinger:
    """sqrt of the L1-normalised features: Euclidean distance = Hellinger distance.
    Used by the first k-NN (iter6-8), applied directly to the raw features."""

    def fit(self, X):
        return self

    def __call__(self, X):
        X = np.clip(np.asarray(X, np.float32), 0, None)
        return np.sqrt(X / (X.sum(1, keepdims=True) + 1e-12)).astype(np.float32)

    def __repr__(self):
        return "Hellinger()"


class PowerNormalize:
    """(counts [* idf]) L1-normalised, raised to `power`, then L2-normalised (power 0.5 with
    no idf = Hellinger). Used by the tuned k-NN (iter9+). idf is fitted on the reference set."""

    def __init__(self, power, use_idf=False):
        self.power, self.use_idf, self.idf = power, use_idf, None

    def fit(self, X):
        self.idf = smooth_idf(to_counts(X)) if self.use_idf else None
        return self

    def __call__(self, X):
        Z = to_counts(X)
        if self.idf is not None:
            Z = Z * self.idf
        Z = (Z / (Z.sum(1, keepdims=True) + 1e-12)) ** self.power
        return (Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)).astype(np.float32)

    def __repr__(self):
        return f"PowerNormalize(power={self.power}, use_idf={self.use_idf})"
