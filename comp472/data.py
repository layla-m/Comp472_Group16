"""Loading the two Kaggle tasks and the common 80/20 validation split."""
from dataclasses import dataclass
from functools import cached_property, lru_cache

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import SEED, VAL_FRACTION
from .paths import DATA_DIR

TASKS = ("image", "sentiment")
ID_PREFIX = {"image": "img_", "sentiment": "text_"}
SENTIMENT_NAMES = ("Negative", "Neutral", "Positive")


@dataclass(frozen=True)
class Split:
    """Stratified 80/20 split (same indices for every model of the project)."""
    idx_tr: np.ndarray
    idx_va: np.ndarray
    X_tr: np.ndarray
    X_va: np.ndarray
    y_tr: np.ndarray
    y_va: np.ndarray


@dataclass(frozen=True, eq=False)
class TaskData:
    name: str
    X: np.ndarray          # (n_train, n_features) float32
    y: np.ndarray          # (n_train,) int64
    X_test: np.ndarray     # (n_test, n_features) float32
    test_ids: pd.Series

    @property
    def n_classes(self):
        return len(np.unique(self.y))

    @cached_property
    def split(self):
        idx_tr, idx_va = train_test_split(np.arange(len(self.y)), test_size=VAL_FRACTION,
                                          stratify=self.y, random_state=SEED)
        return Split(idx_tr, idx_va, self.X[idx_tr], self.X[idx_va], self.y[idx_tr], self.y[idx_va])


@lru_cache(maxsize=None)
def load_task(task):
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}, expected one of {TASKS}")
    train = pd.read_csv(DATA_DIR / f"{task}_train.csv")
    test = pd.read_csv(DATA_DIR / f"{task}_test.csv")
    feats = [c for c in train.columns if c.startswith("feature_")]
    return TaskData(task, train[feats].to_numpy(np.float32), train["label"].to_numpy(np.int64),
                    test[feats].to_numpy(np.float32), test["id"])


def to_counts(X):
    """Image features are L2-normalised integer counts of visual words: the smallest non-zero
    value of a row is 1/||counts||, so counts = X / min_nonzero(X). Returns float64 integers."""
    X = np.clip(np.asarray(X, np.float64), 0, None)
    m = np.where(X > 0, X, np.inf).min(1, keepdims=True)
    return np.round(X / m)
