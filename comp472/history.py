"""Scores of every iteration (validation 80/20 and Kaggle), as reported in COMPTE_RENDU_NN.md."""
from dataclasses import dataclass


@dataclass(frozen=True)
class IterationScore:
    name: str
    method: str
    val_image: float
    val_sentiment: float = None
    kaggle: float = None          # None: not submitted

    @property
    def val_mean(self):
        return None if self.val_sentiment is None else (self.val_image + self.val_sentiment) / 2


ITERATIONS = (
    IterationScore("iter1", "MLP de base", 0.397, 0.673, 0.539),
    IterationScore("iter2", "in_dropout 0.5, [4096]", 0.5085, 0.6817, 0.609),
    IterationScore("iter3", "+ tf-idf", 0.5075, 0.6817, 0.608),
    IterationScore("iter4", "[8192], 100 epochs", 0.5155, 0.6817, 0.619),
    IterationScore("iter5", "augmentation / SWA", 0.511, None, None),
    IterationScore("iter6", "kNN Hellinger", 0.589, 0.6817, 0.637),
    IterationScore("iter7", "réseau + kNN", 0.627, 0.6817, 0.657),
    IterationScore("iter8", "2 réseaux + kNN", 0.6298, 0.6817, 0.659),
    IterationScore("iter9", "kNN affiné + pseudo-étiq.", 0.637, 0.6817, 0.661),
    IterationScore("iter10a", "+ réseau Hellinger", 0.6372, 0.6817, 0.660),
    IterationScore("iter10", "5 réseaux", 0.6385, 0.6817, None),
    IterationScore("iter11", "+ label spreading", 0.6515, 0.6817, 0.667),
)

# Kaggle scores of the other models of the group (no combination unless stated)
GROUP_KAGGLE = {
    "Random forest": 0.478,
    "kNN": 0.544,
    "Réseau (Layla)": 0.558,
    "Hybride kNN + RF": 0.562,
    "Meilleur hybride (Layla)": 0.575,
}
KAGGLE_TARGET = 0.651
KAGGLE_TOP = 0.671
