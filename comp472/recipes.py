"""
Submission recipes for the ensembles (iter6 ... iter10), registered by name.

Each recipe only uses saved probabilities (results/probabilities) and the k-NN, so it runs in
seconds to minutes. Weights are always chosen on the 80/20 validation split. The sentiment
part of every ensemble is the sentiment MLP (identical in every experiment since iter2).
"""
import itertools
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from .config import KNN_ITER9, LS_ITER11
from .data import load_task
from .ensemble import (PseudoLabelEnsemble, QueryNeighbourhoods, accuracy, best_weight,
                       label_spreading)
from .features import PowerNormalize
from .store import ProbabilityStore
from .submission import build_submission, predictions, save_submission

SENTIMENT_SOURCE = "iter8_nn"
RECIPES = {}


def recipe(name, description):
    def register(fn):
        RECIPES[name] = (fn, description)
        return fn
    return register


@dataclass
class RecipeResult:
    image_proba: np.ndarray
    val_acc: float
    details: dict = field(default_factory=dict)


class RecipeContext:
    """Lazy access to the data, the split and the saved probabilities."""

    def __init__(self, store=None, log=print):
        self.store, self.log = store or ProbabilityStore(), log

    @cached_property
    def image(self):
        return load_task("image")

    @property
    def y_va(self):
        return self.image.split.y_va

    def nets(self, tags, part):
        return self.store.mean(list(tags), "image", part)

    @cached_property
    def neighbourhoods(self):
        """Tuned k-NN neighbours, on validation (ref = 80 %) and on test (ref = 100 %)."""
        img, s, p = self.image, self.image.split, KNN_ITER9
        geo = PowerNormalize(p.power, p.use_idf)
        va = QueryNeighbourhoods.compute(geo.fit(s.X_tr)(s.X_va), geo(s.X_tr), s.y_tr, p.k_max)
        te = QueryNeighbourhoods.compute(geo.fit(img.X)(img.X_test), geo(img.X), img.y, p.k_max)
        return va, te

    def pseudo_label_ensemble(self, tags):
        """iter9 recipe for the average of the networks `tags`: (validation result, test proba)."""
        nb_va, nb_te = self.neighbourhoods
        ens = PseudoLabelEnsemble(KNN_ITER9, self.image.n_classes)
        val = ens.run(self.nets(tags, "val"), nb_va, y=self.y_va)
        test = ens.run(self.nets(tags, "test"), nb_te, w1=val.w1, w2=val.w2)
        return val, test.proba


# ---------------------------------------------------------------------------------- recipes
@recipe("iter6", "kNN Hellinger seul pour l'image")
def iter6(ctx):
    K_va = ctx.store.load("knn", "image", "val")
    return RecipeResult(ctx.store.load("knn", "image", "test"), accuracy(K_va, ctx.y_va))


@recipe("iter7", "w * réseau [4096] + (1 - w) * kNN Hellinger")
def iter7(ctx):
    N_va, K_va = ctx.store.load("iter7_nn"), ctx.store.load("knn")
    w, acc = best_weight(N_va, K_va, ctx.y_va)
    ctx.log(f"best w_nn={w:.2f} val_acc={acc:.4f}")
    N_te, K_te = ctx.store.load("iter7_nn", part="test"), ctx.store.load("knn", part="test")
    return RecipeResult(w * N_te + (1 - w) * K_te, acc, {"w_nn": w})


@recipe("iter8", "moyenne des réseaux [4096] et [8192] (ou [8192] seul) + kNN Hellinger")
def iter8(ctx):
    options = {"[8192]": ("iter8_nn",), "mean([4096],[8192])": ("iter7_nn", "iter8_nn")}
    K_va, K_te = ctx.store.load("knn"), ctx.store.load("knn", part="test")
    best = None
    for name, tags in options.items():
        w, acc = best_weight(ctx.nets(tags, "val"), K_va, ctx.y_va)
        ctx.log(f"{name:22s} best w_nn={w:.2f} -> {acc:.4f}")
        if best is None or acc > best[0]:
            best = (acc, name, w)
    acc, name, w = best
    N_te = ctx.nets(options[name], "test")
    return RecipeResult(w * N_te + (1 - w) * K_te, acc, {"option": name, "w_nn": w})


@recipe("iter9", "réseaux [4096] + [8192], kNN affiné (p=0.4) et pseudo-étiquetage")
def iter9(ctx):
    val, P = ctx.pseudo_label_ensemble(("iter7_nn", "iter8_nn"))
    ctx.log(f"val_acc={val.val_acc:.4f} w1={val.w1} w2={val.w2}")
    return RecipeResult(P, val.val_acc, {"w1": val.w1, "w2": val.w2})


def best_subset(ctx, nets):
    """Every non-empty subset of the networks, scored with the iter9 recipe."""
    nb_va, _ = ctx.neighbourhoods
    ens = PseudoLabelEnsemble(KNN_ITER9, ctx.image.n_classes)
    rows = []
    for r in range(1, len(nets) + 1):
        for sub in itertools.combinations(nets, r):
            N = ctx.nets(sub, "val")
            res = ens.run(N, nb_va, y=ctx.y_va)
            rows.append(dict(val_acc=res.val_acc, nets="+".join(sub), w1=res.w1, w2=res.w2,
                             nn_mean=accuracy(N, ctx.y_va)))
    rows.sort(key=lambda r: -r["val_acc"])
    for r in rows[:10]:
        ctx.log(f"ens {r['val_acc']:.4f}  nn-mean {r['nn_mean']:.4f}  w1={r['w1']} w2={r['w2']}  {r['nets']}")
    return rows


def _subset_recipe(ctx, nets):
    rows = best_subset(ctx, nets)
    tags = tuple(rows[0]["nets"].split("+"))
    val, P = ctx.pseudo_label_ensemble(tags)
    return RecipeResult(P, val.val_acc, {"nets": rows[0]["nets"], "w1": val.w1, "w2": val.w2,
                                         "subsets": rows})


@recipe("iter10a", "meilleur sous-ensemble de {iter7, iter8, iter10_a} + kNN affiné + pseudo-étiquetage")
def iter10a(ctx):
    return _subset_recipe(ctx, ("iter7_nn", "iter8_nn", "iter10_a"))


@recipe("iter10", "meilleur sous-ensemble des 5 réseaux + kNN affiné + pseudo-étiquetage")
def iter10(ctx):
    return _subset_recipe(ctx, ("iter7_nn", "iter8_nn", "iter10_a", "iter10_b", "iter10_c"))


@recipe("iter11", "iter9 + label spreading sur le graphe kNN (train + test non étiqueté)")
def iter11(ctx):
    img, s = ctx.image, ctx.image.split
    val, P_te = ctx.pseudo_label_ensemble(("iter7_nn", "iter8_nn"))
    geo = PowerNormalize(KNN_ITER9.power, KNN_ITER9.use_idf)
    # validation: labelled = 80 %, unlabelled = the 20 % (labels never used); test: 100 % + test
    F_va = label_spreading(geo.fit(s.X_tr)(s.X_tr), s.y_tr, geo(s.X_va), LS_ITER11, img.n_classes)
    F_te = label_spreading(geo.fit(img.X)(img.X), img.y, geo(img.X_test), LS_ITER11, img.n_classes)
    w, acc = best_weight(val.proba, F_va, ctx.y_va)
    ctx.log(f"iter9 val={val.val_acc:.4f}  label spreading alone={accuracy(F_va, ctx.y_va):.4f}  "
            f"blend w_iter9={w:.2f} -> {acc:.4f}")
    return RecipeResult(w * P_te + (1 - w) * F_te, acc, {"w_iter9": w})


# ---------------------------------------------------------------------------------- running
def make_submission(name, ctx=None, save=True):
    ctx = ctx or RecipeContext()
    fn, description = RECIPES[name]
    ctx.log(f"recipe {name}: {description}")
    res = fn(ctx)
    image = predictions(ctx.image.test_ids, res.image_proba.argmax(1))
    sen_P = ctx.store.load(SENTIMENT_SOURCE, "sentiment", "test")
    sentiment = predictions(load_task("sentiment").test_ids, sen_P.argmax(1))
    sub = build_submission(image, sentiment)
    if save:
        ctx.log(f"Saved {save_submission(sub, name)} (val image {res.val_acc:.4f})")
    return sub, res
