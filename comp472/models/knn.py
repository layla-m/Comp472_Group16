"""
k-nearest neighbours on visual-word histograms.

The geometry is a strategy object (`features.Hellinger` or `features.PowerNormalize`), the
neighbour weighting is a string: "uniform", "inv" (1 / d) or "exp<tau>" (exp(-(d - d_min) / tau)).
Neighbours are computed once (up to k_max) and every k <= k_max reuses them.
"""
from dataclasses import dataclass

import numpy as np
import torch


def neighbours(A, B, k, exclude_self=False, batch=1000):
    """k nearest rows of B for every row of A (Euclidean): sorted distances and indices.
    exclude_self: A is B, a point is never its own neighbour (leave-one-out)."""
    B_t = torch.from_numpy(B)
    ds, idx = [], []
    for i in range(0, len(A), batch):
        d = torch.cdist(torch.from_numpy(A[i:i + batch]), B_t)
        if exclude_self:
            r = torch.arange(d.shape[0])
            d[r, r + i] = float("inf")
        v, j = d.topk(k, dim=1, largest=False)
        ds.append(v.numpy())
        idx.append(j.numpy())
    return np.vstack(ds), np.vstack(idx)


def neighbour_weights(d, weighting):
    if weighting == "uniform":
        return np.ones_like(d)
    if weighting == "inv":
        return 1.0 / (d + 1e-6)
    if weighting.startswith("exp"):  # closer neighbours count exponentially more
        return np.exp(-(d - d[:, :1]) / float(weighting[3:]))
    raise ValueError(f"unknown weighting {weighting!r}")


def vote(d, labels, k, weighting, n_classes, extra=None, eps=0.0):
    """Class probabilities from the k first (sorted) neighbours.
    extra: per-neighbour weight (pseudo-labelled points); infinite distances get weight 0."""
    d, labels = d[:, :k], labels[:, :k]
    w = neighbour_weights(d, weighting)
    if extra is not None:
        w = w * extra[:, :k]
    w[~np.isfinite(d)] = 0
    P = np.zeros((len(d), n_classes))
    np.add.at(P, (np.repeat(np.arange(len(d)), k), labels.ravel()), w.ravel())
    return P / (P.sum(1, keepdims=True) + eps)


@dataclass
class Neighbourhood:
    """Precomputed neighbours of a query set in a labelled reference set."""
    dist: np.ndarray
    idx: np.ndarray
    y_ref: np.ndarray

    @property
    def labels(self):
        return self.y_ref[self.idx]

    def proba(self, k, weighting, n_classes, eps=0.0):
        return vote(self.dist, self.labels, k, weighting, n_classes, eps=eps)


class KNNClassifier:
    """k-NN with a pluggable geometry. `fit` stores the transformed reference set."""

    def __init__(self, geometry, k_max):
        self.geometry, self.k_max = geometry, k_max

    def fit(self, X, y):
        self.geometry.fit(X)
        self.H_ref, self.y_ref = self.geometry(X), y
        return self

    def neighbourhood(self, X):
        dist, idx = neighbours(self.geometry(X), self.H_ref, self.k_max)
        return Neighbourhood(dist, idx, self.y_ref)


def with_pseudo_labels(ref, pool_dist, pool_idx, pool_labels, confident, pseudo_weight, k):
    """Merge the neighbours in the labelled set (`ref`) with the neighbours in the unlabelled
    pool (leave-one-out), keeping only confident pool points, weighted by `pseudo_weight`.
    Returns the k nearest (distances, labels, extra weights)."""
    dB = np.where(confident[pool_idx], pool_dist, np.inf)
    d = np.hstack([ref.dist, dB])
    lab = np.hstack([ref.labels, pool_labels[pool_idx]])
    wt = np.hstack([np.ones_like(ref.dist), np.full_like(dB, pseudo_weight)])
    o = np.argsort(d, axis=1)[:, :k]
    r = np.arange(len(d))[:, None]
    return d[r, o], lab[r, o], wt[r, o]
