"""
Equivalence of the refactored package with the original scripts (branch `sauvegarde-originale`).

The original scripts are extracted from git into a temporary folder, then for the same inputs:
  - feature pipelines give identical arrays (Prep / Standardizer vs FeaturePipeline);
  - the 80/20 split gives identical indices;
  - training gives bit-identical weights (a few epochs on a subset, every training option:
    validation + early stopping, fixed epochs, mixup, thinning, noise, SWA, iter1 trainer).

Run: python -m pytest tests  (or python tests/test_equivalence.py)
"""
import importlib
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from comp472 import config as C  # noqa: E402
from comp472.data import load_task  # noqa: E402
from comp472.features import FeaturePipeline, augmentation_inputs  # noqa: E402
from comp472.models.trainer import MLPTrainer  # noqa: E402

BACKUP_BRANCH = "sauvegarde-originale"
LEGACY_FILES = ("neural_network.py", "nn_v2.py")
_TMP = None


def legacy():
    """Import the original neural_network.py and nn_v2.py from the backup branch."""
    global _TMP
    if _TMP is None:
        _TMP = tempfile.TemporaryDirectory()
        tmp = Path(_TMP.name)
        for f in LEGACY_FILES:
            src = subprocess.run(["git", "-C", str(ROOT), "show", f"{BACKUP_BRANCH}:{f}"],
                                 check=True, capture_output=True, text=True).stdout
            (tmp / f).write_text(src)
        for csv in (ROOT / "data").glob("*.csv"):
            (tmp / csv.name).symlink_to(csv)
        sys.path.insert(0, str(tmp))
    threads = torch.get_num_threads()
    nn1, nn2 = importlib.import_module("neural_network"), importlib.import_module("nn_v2")
    torch.set_num_threads(threads)  # neural_network.py changes it at import
    return nn1, nn2


def as_dict(cfg):
    """MLPConfig -> dict understood by the original nn_v2.py."""
    d = dict(hidden=list(cfg.hidden), dropout=cfg.dropout, in_dropout=cfg.in_dropout, lr=cfg.lr,
             weight_decay=cfg.weight_decay, batch_size=cfg.batch_size, max_epochs=cfg.max_epochs,
             patience=cfg.patience, sqrt_features=cfg.sqrt_features, row_norm=cfg.row_norm,
             l2_norm=cfg.l2_norm, mixup=cfg.mixup, smoothing=cfg.smoothing)
    for key, default in (("tfidf", False), ("pca", None), ("thin", 0.0), ("noise", 0.0), ("swa_start", 0)):
        if getattr(cfg, key) != default:
            d[key] = getattr(cfg, key)
    if not cfg.standardize:
        d["std"] = False
    return d


def same_state(m1, m2):
    s1, s2 = m1.state_dict(), m2.state_dict()
    return s1.keys() == s2.keys() and all(torch.equal(s1[k], s2[k]) for k in s1)


def test_split():
    nn1, _ = legacy()
    from sklearn.model_selection import train_test_split
    for task in ("image", "sentiment"):
        d = load_task(task)
        X_tr, X_va, y_tr, y_va = train_test_split(d.X, d.y, test_size=0.2, stratify=d.y, random_state=nn1.SEED)
        s = d.split
        assert np.array_equal(X_tr, s.X_tr) and np.array_equal(X_va, s.X_va)
        assert np.array_equal(y_tr, s.y_tr) and np.array_equal(y_va, s.y_va)


PIPELINE_CONFIGS = [C.IMAGE_ITER2, C.IMAGE_ITER2.but(tfidf=True), C.IMAGE_ITER2.but(tfidf=True, row_norm=True),
                    C.IMAGE_ITER2.but(row_norm=True, standardize=False), C.IMAGE_ITER2.but(standardize=False),
                    C.IMAGE_ITER2.but(l2_norm=True), C.IMAGE_ITER2.but(pca=0.99)]


def test_feature_pipeline():
    nn1, nn2 = legacy()
    s = load_task("image").split
    for cfg in PIPELINE_CONFIGS:
        old = nn2.Prep(as_dict(cfg)).fit(s.X_tr)
        new = FeaturePipeline(cfg).fit(s.X_tr)
        if cfg.pca:  # PCA (randomised solver) is compared on the first rows only to stay fast
            assert np.array_equal(old.transform(s.X_va[:500]), new.transform(s.X_va[:500])), cfg
        else:
            assert np.array_equal(old.transform(s.X_va), new.transform(s.X_va)), cfg
    sen = load_task("sentiment").split
    assert np.array_equal(nn2.Prep(as_dict(C.SENTIMENT)).fit(sen.X_tr).transform(sen.X_va),
                          FeaturePipeline(C.SENTIMENT).fit(sen.X_tr).transform(sen.X_va))
    # iter1: Standardizer of neural_network.py
    for task, cfg in (("image", C.IMAGE_ITER1), ("sentiment", C.SENTIMENT_ITER1)):
        sp = load_task(task).split
        old = nn1.Standardizer(cfg.sqrt_features).fit(sp.X_tr).transform(sp.X_va)
        assert np.array_equal(old, FeaturePipeline(cfg).fit(sp.X_tr).transform(sp.X_va)), task


def _subset(task, n_tr=3000, n_va=800):
    s = load_task(task).split
    return s.X_tr[:n_tr], s.y_tr[:n_tr], s.X_va[:n_va], s.y_va[:n_va], load_task(task).n_classes


SHORT = dict(max_epochs=3, patience=2)
TRAIN_CONFIGS = [
    ("image", C.IMAGE_ITER2.but(**SHORT)),
    ("image", C.IMAGE_ITER2.but(hidden=(1024, 256), mixup=0.2, **SHORT)),
    ("image", C.IMAGE_ITER2.but(hidden=(1024,), thin=0.3, **SHORT)),
    ("image", C.IMAGE_ITER2.but(hidden=(1024,), noise=0.3, **SHORT)),
    ("image", C.IMAGE_ITER2.but(hidden=(1024,), swa_start=2, **SHORT)),
    ("image", C.IMAGE_ITER2.but(hidden=(1024,), row_norm=True, standardize=False, in_dropout=0.3, **SHORT)),
    ("sentiment", C.SENTIMENT.but(**SHORT)),
]


def test_trainer_matches_nn_v2():
    _, nn2 = legacy()
    for task, cfg in TRAIN_CONFIGS:
        X_tr, y_tr, X_va, y_va, nc = _subset(task)
        old_p, new_p = nn2.Prep(as_dict(cfg)).fit(X_tr), FeaturePipeline(cfg).fit(X_tr)
        A, V = new_p.transform(X_tr), new_p.transform(X_va)
        # with validation (early stopping)
        m_old, ep_old, acc_old = nn2.train_model(A, y_tr, nc, as_dict(cfg), V, y_va, **nn2.aug_args(as_dict(cfg), X_tr, old_p))
        res = MLPTrainer(cfg).fit(A, y_tr, nc, V, y_va, **augmentation_inputs(cfg, X_tr, new_p))
        assert (ep_old, acc_old) == (res.best_epoch, res.best_acc), (task, cfg)
        assert same_state(m_old, res.model), (task, cfg)
        # history recording must not change the weights
        res_h = MLPTrainer(cfg, record_history=True).fit(A, y_tr, nc, V, y_va, **augmentation_inputs(cfg, X_tr, new_p))
        assert same_state(m_old, res_h.model) and len(res_h.history) == 3, (task, cfg)
        # final model: fixed number of epochs, another seed
        m_old, *_ = nn2.train_model(A, y_tr, nc, as_dict(cfg), epochs=2, seed=43, **nn2.aug_args(as_dict(cfg), X_tr, old_p))
        res = MLPTrainer(cfg, seed=43).fit(A, y_tr, nc, epochs=2, **augmentation_inputs(cfg, X_tr, new_p))
        assert same_state(m_old, res.model), (task, cfg, "final")
        print(f"  ok {task} {cfg.hidden} mixup={cfg.mixup} thin={cfg.thin} noise={cfg.noise} swa={cfg.swa_start} "
              f"std={cfg.standardize}", flush=True)


def test_trainer_matches_iter1():
    nn1, _ = legacy()
    for task, cfg in (("image", C.IMAGE_ITER1.but(**SHORT)), ("sentiment", C.SENTIMENT_ITER1.but(**SHORT))):
        X_tr, y_tr, X_va, y_va, nc = _subset(task)
        p = FeaturePipeline(cfg).fit(X_tr)
        A, V = p.transform(X_tr), p.transform(X_va)
        old_cfg = dict(hidden=list(cfg.hidden), dropout=cfg.dropout, lr=cfg.lr, weight_decay=cfg.weight_decay,
                       batch_size=cfg.batch_size, max_epochs=cfg.max_epochs, patience=cfg.patience)
        m_old, ep, acc, hist = nn1.train_model(A, y_tr, nc, old_cfg, V, y_va, verbose=False)
        res = MLPTrainer(cfg, record_history=True).fit(A, y_tr, nc, V, y_va)
        assert (ep, acc) == (res.best_epoch, res.best_acc) and same_state(m_old, res.model), task
        assert np.allclose(np.array(hist, float), np.array(res.history, float)), task
        print(f"  ok iter1 {task}", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(8)
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            print(name, flush=True)
            fn()
    print("ALL EQUIVALENT")
