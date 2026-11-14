"""Saved class probabilities (results/probabilities/<tag>_<task>_<part>.npy)."""
import numpy as np

from .paths import PROBA_DIR


class ProbabilityStore:
    """part = "val" (model trained on the 80 %, predicted on the 20 %) or
    "test" (average of the final models trained on 100 %, predicted on test)."""

    def __init__(self, root=PROBA_DIR):
        self.root = root

    def path(self, tag, task, part):
        return self.root / f"{tag}_{task}_{part}.npy"

    def exists(self, tag, task="image", part="test"):
        return self.path(tag, task, part).exists()

    def load(self, tag, task="image", part="val"):
        return np.load(self.path(tag, task, part))

    def save(self, P, tag, task, part):
        self.root.mkdir(parents=True, exist_ok=True)
        np.save(self.path(tag, task, part), P)

    def mean(self, tags, task="image", part="val"):
        """Equal-weight average of several models."""
        return np.mean([self.load(t, task, part) for t in tags], axis=0)

    def save_named(self, P, name):
        self.root.mkdir(parents=True, exist_ok=True)
        np.save(self.root / f"{name}.npy", P)

    def load_named(self, name):
        return np.load(self.root / f"{name}.npy")
