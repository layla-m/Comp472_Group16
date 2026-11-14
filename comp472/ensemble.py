"""
Combining models on the image task.

blend: P = w * P_nn + (1 - w) * P_knn, w chosen on the validation split (grid 0, 0.05, ..., 1).
PseudoLabelEnsemble: iter9 recipe = tuned k-NN -> blend -> confident predictions of the
unlabelled points added as extra neighbours -> k-NN again -> blend.
"""
from dataclasses import dataclass

import numpy as np

from .models.knn import neighbours, vote, with_pseudo_labels, Neighbourhood

WEIGHT_GRID = np.round(np.arange(0, 1.001, 0.05), 2)


def accuracy(P, y):
    return (P.argmax(1) == y).mean()


def blend(N, K, w):
    return w * N + (1 - w) * K


def weight_curve(N, K, y, grid=WEIGHT_GRID):
    return [accuracy(blend(N, K, w), y) for w in grid]


def best_weight(N, K, y, grid=WEIGHT_GRID):
    """Weight of the network with the best validation accuracy (first one on ties)."""
    accs = weight_curve(N, K, y, grid)
    i = int(np.argmax(accs))
    return grid[i], accs[i]


@dataclass
class QueryNeighbourhoods:
    """Neighbours of the query points among the labelled points (`ref`) and among the query
    points themselves (`pool`, leave-one-out), computed once for every setting."""
    ref: Neighbourhood
    pool_dist: np.ndarray
    pool_idx: np.ndarray

    @classmethod
    def compute(cls, H_query, H_ref, y_ref, k_max):
        d, i = neighbours(H_query, H_ref, k_max)
        dB, iB = neighbours(H_query, H_query, k_max, exclude_self=True)
        return cls(Neighbourhood(d, i, y_ref), dB, iB)


@dataclass
class EnsembleResult:
    proba: np.ndarray
    w1: float            # network weight of the first blend (used for the pseudo-labels)
    w2: float            # network weight of the final blend
    val_acc: float = None


class PseudoLabelEnsemble:
    """iter9 recipe for fixed `KNNParams`. On validation (y given) both weights are tuned;
    on test they are passed in."""

    def __init__(self, params, n_classes):
        self.p, self.nc = params, n_classes

    def first_knn(self, nb):
        return vote(nb.ref.dist, nb.ref.labels, self.p.k, f"exp{self.p.tau}", self.nc,
                    extra=np.ones_like(nb.ref.dist), eps=1e-12)

    def second_knn(self, nb, P1, k=None, tau=None, threshold=None, pseudo_weight=None):
        p = self.p
        k = k or p.k_pseudo
        d, lab, wt = with_pseudo_labels(nb.ref, nb.pool_dist, nb.pool_idx, P1.argmax(1),
                                        P1.max(1) >= (p.threshold if threshold is None else threshold),
                                        p.pseudo_weight if pseudo_weight is None else pseudo_weight, k)
        return vote(d, lab, k, f"exp{tau or p.tau_pseudo}", self.nc, extra=wt, eps=1e-12)

    def run(self, N, nb, y=None, w1=None, w2=None):
        Ka = self.first_knn(nb)
        if y is not None:
            w1, _ = best_weight(N, Ka, y)
        Kb = self.second_knn(nb, blend(N, Ka, w1))
        acc = None
        if y is not None:
            w2, acc = best_weight(N, Kb, y)
        return EnsembleResult(blend(N, Kb, w2), w1, w2, acc)
