"""
Searches on the 80/20 validation split.

- hyperparameter searches of the MLP, one factor at a time around a base config (`SEARCHES`);
- seed / architecture ensemble study on sentiment;
- first Hellinger k-NN grid (k x weighting) -> probabilities "knn" used by iter6-8;
- in-ensemble k-NN search (geometry, k, tau, then pseudo-labelling) -> settings of iter9.
"""
import itertools
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import (IMAGE_ITER1, IMAGE_ITER2, IMAGE_SCREENING, KNN_GRID, KNN_SEARCH, SEED, SENTIMENT,
                     SENTIMENT_ITER1, KNNParams)
from .data import load_task
from .ensemble import PseudoLabelEnsemble, QueryNeighbourhoods, best_weight
from .features import FeaturePipeline, Hellinger, PowerNormalize, augmentation_inputs
from .models.knn import KNNClassifier
from .models.mlp import predict, predict_proba
from .models.trainer import MLPTrainer
from .paths import SEARCH_DIR, SEED_STUDY_DIR
from .store import ProbabilityStore


@dataclass(frozen=True)
class Search:
    task: str
    base: object                # MLPConfig
    variants: tuple             # (name, {field: value}) pairs
    table: str                  # results/search/<table>.csv (rows appended)
    save_proba: bool = True     # validation probabilities saved as hp_<task>_<round>_<name>.npy


def _v(*pairs):
    return tuple(pairs)


SEARCHES = {
    # first search (between iter1 and iter2)
    "image-v1": Search("image", IMAGE_ITER1, _v(
        ("base", {}), ("no_sqrt", {"sqrt_features": False}), ("hidden=[1024]", {"hidden": [1024]}),
        ("hidden=[2048]", {"hidden": [2048]}), ("hidden=[2048,1024]", {"hidden": [2048, 1024]}),
        ("dropout=0.3", {"dropout": 0.3}), ("dropout=0.7", {"dropout": 0.7}),
        ("wd=0.05", {"weight_decay": 5e-2}), ("lr=3e-3", {"lr": 3e-3}), ("lr=3e-4", {"lr": 3e-4}),
    ), "hp_results_image", save_proba=False),
    "sentiment-v1": Search("sentiment", SENTIMENT_ITER1, _v(
        ("base", {}), ("hidden=[256]", {"hidden": [256]}), ("hidden=[256,64]", {"hidden": [256, 64]}),
        ("hidden=[1024,256]", {"hidden": [1024, 256]}),
        ("hidden=[1024,512,128]", {"hidden": [1024, 512, 128]}),
        ("dropout=0.3", {"dropout": 0.3}), ("dropout=0.7", {"dropout": 0.7}),
        ("lr=1e-3", {"lr": 1e-3}), ("lr=1e-4", {"lr": 1e-4}), ("wd=0.1", {"weight_decay": 1e-1}),
        ("batch=64", {"batch_size": 64}), ("batch=256", {"batch_size": 256}),
    ), "hp_results_sentiment", save_proba=False),
    # second search (iter2 -> iter5), rounds 1-3 on a cheaper [2048] base
    "image-1": Search("image", IMAGE_SCREENING, _v(
        ("base[2048]", {}), ("row_norm", {"row_norm": True}),
        ("in_drop=0.2", {"in_dropout": 0.2}), ("in_drop=0.4", {"in_dropout": 0.4}),
        ("mixup=0.2", {"mixup": 0.2, "max_epochs": 100, "patience": 15}),
        ("mixup=0.4", {"mixup": 0.4, "max_epochs": 100, "patience": 15}),
        ("smooth=0.2", {"smoothing": 0.2}), ("hidden=[4096]", {"hidden": [4096]}),
        ("bs=256,lr=2e-3", {"batch_size": 256, "lr": 2e-3}),
    ), "hp2_results_image"),
    "image-2": Search("image", IMAGE_SCREENING, _v(
        ("in_drop=0.5", {"in_dropout": 0.5}), ("in_drop=0.6", {"in_dropout": 0.6}),
        ("in_drop=0.7", {"in_dropout": 0.7}), ("in_drop=0.5,rn", {"in_dropout": 0.5, "row_norm": True}),
        ("in_drop=0.5,[2048,1024]", {"in_dropout": 0.5, "hidden": [2048, 1024]}),
        ("in_drop=0.5,[4096]", {"in_dropout": 0.5, "hidden": [4096]}),
        ("in_drop=0.5,d=0.3", {"in_dropout": 0.5, "dropout": 0.3}),
    ), "hp2_results_image"),
    "image-3a": Search("image", IMAGE_SCREENING, _v(
        ("[4096],in=0.5,d=0.3", {"hidden": [4096], "in_dropout": 0.5, "dropout": 0.3}),
        ("[8192],in=0.5", {"hidden": [8192], "in_dropout": 0.5}),
        ("[4096],in=0.4", {"hidden": [4096], "in_dropout": 0.4}),
    ), "hp2_results_image"),
    "image-3b": Search("image", IMAGE_SCREENING, _v(
        ("[4096],in=0.5,ep=100", {"hidden": [4096], "in_dropout": 0.5, "max_epochs": 100, "patience": 15}),
        ("[4096],in=0.5,mixup=0.2", {"hidden": [4096], "in_dropout": 0.5, "mixup": 0.2,
                                      "max_epochs": 100, "patience": 15}),
        ("[4096],in=0.5,wd=0.05", {"hidden": [4096], "in_dropout": 0.5, "weight_decay": 5e-2}),
    ), "hp2_results_image"),
    # piste 2 (iter3): transform the image features, base = iter2 network
    "image-4": Search("image", IMAGE_ITER2, _v(
        ("tfidf", {"tfidf": True}), ("tfidf+row_norm", {"tfidf": True, "row_norm": True}),
        ("pca_whiten_0.99", {"pca": 0.99}), ("pca_whiten_0.99,in=0.2", {"pca": 0.99, "in_dropout": 0.2}),
    ), "hp2_results_image"),
    # iter5: augmentation of the visual-word counts + SWA
    "image-5": Search("image", IMAGE_ITER2, _v(
        ("thin=0.3", {"thin": 0.3}), ("thin=0.5", {"thin": 0.5}),
        ("thin=0.5,in=0.3", {"thin": 0.5, "in_dropout": 0.3}), ("noise=0.3", {"noise": 0.3}),
        ("swa_from_40", {"swa_start": 40, "patience": 60}),
    ), "hp2_results_image"),
    # test B: Hellinger features without per-feature standardisation
    "image-6": Search("image", IMAGE_ITER2, _v(
        ("hellinger_nostd", {"row_norm": True, "standardize": False}),
        ("hellinger_nostd,in=0.3", {"row_norm": True, "standardize": False, "in_dropout": 0.3}),
        ("sqrtL2_nostd", {"standardize": False}),
    ), "hp2_results_image"),
    "sentiment-1": Search("sentiment", SENTIMENT, _v(
        ("base", {}), ("in_drop=0.1", {"in_dropout": 0.1}), ("in_drop=0.2", {"in_dropout": 0.2}),
        ("l2_norm", {"l2_norm": True}), ("mixup=0.2", {"mixup": 0.2}), ("mixup=0.4", {"mixup": 0.4}),
        ("smooth=0.2", {"smoothing": 0.2}), ("smooth=0.0", {"smoothing": 0.0}),
        ("[256,64],d=0.7", {"hidden": [256, 64]}), ("[1024],d=0.7", {"hidden": [1024]}),
        ("wd=0.05", {"weight_decay": 5e-2}), ("lr=2e-4", {"lr": 2e-4}),
    ), "hp2_results_sentiment"),
}


def _append(row, table):
    SEARCH_DIR.mkdir(parents=True, exist_ok=True)
    path = SEARCH_DIR / f"{table}.csv"
    df = pd.DataFrame([row])
    if path.exists():  # older tables may have fewer columns: realign instead of blind append
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    df.to_csv(path, index=False)


def run_search(name, store=None):
    """Train every variant on the 80 % and append its validation accuracy to the table."""
    search, store = SEARCHES[name], store or ProbabilityStore()
    data = load_task(search.task)
    s = data.split
    rnd = name.split("-", 1)[1]
    for variant, change in search.variants:
        cfg = search.base.but(**change)
        pipe = FeaturePipeline(cfg).fit(s.X_tr)
        X_tr, X_va = pipe.transform(s.X_tr), pipe.transform(s.X_va)
        t0 = time.time()
        res = MLPTrainer(cfg, seed=SEED).fit(X_tr, s.y_tr, data.n_classes, X_va, s.y_va,
                                             **augmentation_inputs(cfg, s.X_tr, pipe))
        row = dict(round=rnd, variant=variant, val_acc=round(float(res.best_acc), 4),
                   train_acc=round(float((predict(res.model, X_tr) == s.y_tr).mean()), 4),
                   best_epoch=res.best_epoch, seconds=round(time.time() - t0, 1), cfg=str(change))
        print(row, flush=True)
        if search.save_proba:
            store.save_named(predict_proba(res.model, X_va), f"hp_{search.task}_{rnd}_{variant}")
        _append(row, search.table)


# ------------------------------------------------------------------ seed / architecture study
SEED_STUDY = {"A_base": {}, "B_[256,64]": {"hidden": [256, 64]}, "C_[1024]": {"hidden": [1024]},
              "D_mixup0.2": {"mixup": 0.2}}
SEED_OFFSETS = (0, 1, 2)


def seed_study(task="sentiment"):
    """Validation accuracy of single models vs ensembles of seeds / architectures (sentiment)."""
    data = load_task(task)
    s = data.split
    SEED_STUDY_DIR.mkdir(parents=True, exist_ok=True)
    probs = {}
    for name, change in SEED_STUDY.items():
        cfg = SENTIMENT.but(**change)
        pipe = FeaturePipeline(cfg).fit(s.X_tr)
        Xt, Xv = pipe.transform(s.X_tr), pipe.transform(s.X_va)
        for k in SEED_OFFSETS:
            f = SEED_STUDY_DIR / f"{name}_s{k}.npy"
            if not f.exists():
                res = MLPTrainer(cfg, seed=SEED + k).fit(Xt, s.y_tr, data.n_classes, Xv, s.y_va)
                np.save(f, predict_proba(res.model, Xv))
                print(f"{name} seed{k}: val={res.best_acc:.4f} ep={res.best_epoch}", flush=True)
            probs[(name, k)] = np.load(f)
    acc = lambda P: (P.argmax(1) == s.y_va).mean()
    names = list(SEED_STUDY)
    for r in range(1, len(names) + 1):
        for combo in itertools.combinations(names, r):
            P = sum(probs[(n, k)] for n in combo for k in SEED_OFFSETS)
            print(f"ENS {'+'.join(combo)}: {acc(P):.4f}", flush=True)
    return probs


# ------------------------------------------------------------------ first k-NN grid (iter6-8)
def knn_grid(store=None, grid=KNN_GRID, save=True):
    """Hellinger k-NN, grid over k and the weighting. The best setting is saved as the
    probabilities "knn" (val: fitted on 80 %, test: fitted on 100 %)."""
    store = store or ProbabilityStore()
    data = load_task("image")
    s, nc = data.split, data.n_classes
    nb = KNNClassifier(Hellinger(), grid.k_max).fit(s.X_tr, s.y_tr).neighbourhood(s.X_va)
    rows = []
    for k in grid.ks:
        for w in grid.weightings:
            if k == 1 and w != "uniform":
                continue
            acc = (nb.proba(k, w, nc).argmax(1) == s.y_va).mean()
            rows.append(dict(k=k, weight=w, val_acc=round(float(acc), 4)))
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows).sort_values("val_acc", ascending=False)
    if save:
        SEARCH_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(SEARCH_DIR / "knn_results_image.csv", index=False)
    k, w = int(df.iloc[0]["k"]), df.iloc[0]["weight"]
    print(f"BEST k={k} weight={w} val_acc={df.iloc[0]['val_acc']}", flush=True)
    store.save(nb.proba(k, w, nc), "knn", "image", "val")
    nb_te = KNNClassifier(Hellinger(), grid.k_max).fit(data.X, data.y).neighbourhood(data.X_test)
    store.save(nb_te.proba(k, w, nc), "knn", "image", "test")
    return df


# ------------------------------------------------------------------ in-ensemble k-NN search (iter9)
def knn_search(store=None, space=KNN_SEARCH, save=True):
    """Step 1: geometry (power, idf), k and tau of the k-NN, judged by the accuracy of the blend
    with the average of the networks `space.nets`. Step 2: pseudo-labelling (threshold, weight,
    k, tau), the validation points playing the unlabelled pool (leave-one-out)."""
    store = store or ProbabilityStore()
    data = load_task("image")
    s, nc = data.split, data.n_classes
    N_va = store.mean(list(space.nets), "image", "val")
    rows = []
    for p, use_idf in itertools.product(space.powers, space.idf):
        geo = PowerNormalize(p, use_idf).fit(s.X_tr)
        nb = KNNClassifier(geo, space.k_max).fit(s.X_tr, s.y_tr).neighbourhood(s.X_va)
        for k, tau in itertools.product(space.ks, space.taus):
            K = PseudoLabelEnsemble(KNNParams(power=p, use_idf=use_idf, k=k, tau=tau), nc).first_knn(
                QueryNeighbourhoods(nb, None, None))
            w, acc = best_weight(N_va, K, s.y_va)
            rows.append(dict(step=1, p=p, idf=use_idf, k=k, tau=tau, knn_acc=(K.argmax(1) == s.y_va).mean(),
                             w_nn=w, ens_acc=acc))
        r = max((r for r in rows if r["p"] == p and r["idf"] == use_idf), key=lambda r: r["ens_acc"])
        print(f"p={p} idf={use_idf}: best k={r['k']} tau={r['tau']} knn={r['knn_acc']:.4f} "
              f"ens={r['ens_acc']:.4f} (w={r['w_nn']})", flush=True)
    s1 = max(rows, key=lambda r: r["ens_acc"])
    print(f"STEP1 BEST p={s1['p']} idf={s1['idf']} k={s1['k']} tau={s1['tau']} knn={s1['knn_acc']:.4f} "
          f"ens={s1['ens_acc']:.4f} w={s1['w_nn']}", flush=True)

    params = KNNParams(power=s1["p"], use_idf=s1["idf"], k=s1["k"], tau=s1["tau"], k_max=space.k_max)
    geo = PowerNormalize(params.power, params.use_idf).fit(s.X_tr)
    nb = QueryNeighbourhoods.compute(geo(s.X_va), geo(s.X_tr), s.y_tr, space.k_max)
    ens = PseudoLabelEnsemble(params, nc)
    K1 = ens.first_knn(nb)
    P1 = s1["w_nn"] * N_va + (1 - s1["w_nn"]) * K1
    s2 = None
    for thr, pw, k2 in itertools.product(space.thresholds, space.pseudo_weights,
                                         sorted({params.k, *space.pseudo_ks})):
        for tau2 in space.taus:
            K2 = ens.second_knn(nb, P1, k=k2, tau=tau2, threshold=thr, pseudo_weight=pw)
            w, acc = best_weight(N_va, K2, s.y_va)
            r = dict(step=2, p=params.power, idf=params.use_idf, k=k2, tau=tau2, thr=thr, pw=pw,
                     knn_acc=(K2.argmax(1) == s.y_va).mean(), w_nn=w, ens_acc=acc)
            rows.append(r)
            if s2 is None or acc > s2["ens_acc"]:
                s2 = r
        print(f"thr={thr} pw={pw} k={k2}: best ens so far {s2['ens_acc']:.4f}", flush=True)
    print(f"STEP2 BEST thr={s2['thr']} pw={s2['pw']} k={s2['k']} tau={s2['tau']} "
          f"knn={s2['knn_acc']:.4f} ens={s2['ens_acc']:.4f} w={s2['w_nn']}", flush=True)
    df = pd.DataFrame(rows).sort_values("ens_acc", ascending=False).round(4)
    if save:
        SEARCH_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(SEARCH_DIR / "knn2_results_image.csv", index=False)
    return df
