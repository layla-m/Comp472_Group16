"""
Hyperparameters.

`MLPConfig` describes one network (architecture, optimisation, feature pipeline, training
tricks). `Experiment` pairs one config per task. `EXPERIMENTS` is the registry of every
network that produced a submission or saved probabilities (iter1 ... iter10_c).
"""
from dataclasses import dataclass, field, replace

SEED = 42          # seed of the 80/20 split and of the first model
N_SEEDS = 5        # final models are trained with seeds 42..46 and their probabilities averaged
VAL_FRACTION = 0.2


@dataclass(frozen=True)
class MLPConfig:
    # architecture
    hidden: tuple = (512, 128)
    dropout: float = 0.5          # after every hidden layer
    in_dropout: float = 0.0       # on the input features (the key lever for the image task)
    # optimisation (AdamW + cosine learning rate, early stopping on validation accuracy)
    lr: float = 1e-3
    weight_decay: float = 1e-2
    batch_size: int = 128
    max_epochs: int = 60
    patience: int = 10
    smoothing: float = 0.1        # label smoothing
    # feature pipeline (see features.FeaturePipeline)
    tfidf: bool = False
    row_norm: bool = False        # L1-normalise each row (histogram -> distribution)
    sqrt_features: bool = False   # square root (Hellinger)
    l2_norm: bool = False
    standardize: bool = True      # False: only a global scale, keeps the Hellinger geometry
    scale: float = None           # global scale when standardize=False (default sqrt(n_features))
    pca: float = None             # PCA whitening, fraction of variance kept
    # training tricks
    mixup: float = 0.0            # mixup alpha (0 = off)
    thin: float = 0.0             # binomial thinning of the visual-word counts
    noise: float = 0.0            # Gaussian noise on the standardised features
    swa_start: int = 0            # stochastic weight averaging from this epoch (0 = off)

    def __post_init__(self):
        object.__setattr__(self, "hidden", tuple(self.hidden))

    def but(self, **changes):
        """Copy with some fields changed: `IMAGE_ITER2.but(hidden=[8192])`."""
        return replace(self, **changes)


@dataclass(frozen=True)
class Experiment:
    name: str
    image: MLPConfig
    sentiment: MLPConfig
    description: str = ""

    def config(self, task):
        return getattr(self, task)


# ---------------------------------------------------------------------------- base configs
# iter1: starting configuration (neural_network.py before any search)
IMAGE_ITER1 = MLPConfig(hidden=(1024, 512), dropout=0.5, lr=1e-3, sqrt_features=True)
SENTIMENT_ITER1 = MLPConfig(hidden=(512, 128), dropout=0.5, lr=5e-4)

# from iter2 on (hp_search2.py): dropout 0.7 for sentiment, input dropout + wide layer for image
SENTIMENT = MLPConfig(hidden=(512, 128), dropout=0.7, lr=5e-4)
IMAGE_ITER2 = MLPConfig(hidden=(4096,), dropout=0.3, in_dropout=0.5, sqrt_features=True)
IMAGE_ITER4 = IMAGE_ITER2.but(hidden=(8192,), max_epochs=100, patience=15)

# cheaper base used to screen image variants in rounds 1-3 of the second search
IMAGE_SCREENING = MLPConfig(hidden=(2048,), dropout=0.5, sqrt_features=True)


def _register(*experiments):
    return {e.name: e for e in experiments}


EXPERIMENTS = _register(
    Experiment("iter1", IMAGE_ITER1, SENTIMENT_ITER1, "configuration de départ, sans recherche"),
    Experiment("iter2", IMAGE_ITER2, SENTIMENT, "image : in_dropout 0.5, [4096], dropout 0.3"),
    Experiment("iter3", IMAGE_ITER2.but(tfidf=True), SENTIMENT, "iter2 + pondération tf-idf"),
    Experiment("iter4", IMAGE_ITER4, SENTIMENT, "image : [8192], 100 epochs"),
    # re-runs that also save probabilities for the ensembles (identical predictions)
    Experiment("iter7_nn", IMAGE_ITER2, SENTIMENT, "réseau d'iter2, probabilités sauvegardées"),
    Experiment("iter8_nn", IMAGE_ITER4, SENTIMENT, "réseau d'iter4, probabilités sauvegardées"),
    # iter10: different networks for the ensemble
    Experiment("iter10_a", IMAGE_ITER2.but(row_norm=True, standardize=False, in_dropout=0.3),
               SENTIMENT, "features Hellinger sans standardisation"),
    Experiment("iter10_b", IMAGE_ITER2.but(hidden=(2048, 1024)), SENTIMENT, "deux couches [2048, 1024]"),
    Experiment("iter10_c", IMAGE_ITER2.but(in_dropout=0.7), SENTIMENT, "in_dropout 0.7"),
)


# ---------------------------------------------------------------------------- k-NN settings
@dataclass(frozen=True)
class KNNParams:
    """Tuned k-NN + pseudo-labelling recipe (knn-search, iter9)."""
    power: float = 0.4        # power of the L1-normalised counts (0.5 = Hellinger)
    use_idf: bool = False
    k: int = 50
    tau: float = 0.02         # temperature of the exponential neighbour weights
    threshold: float = 0.3    # min. ensemble confidence for a pseudo-label
    pseudo_weight: float = 0.5
    k_pseudo: int = 100
    tau_pseudo: float = 0.02
    k_max: int = 200          # neighbours computed once, then truncated


KNN_ITER9 = KNNParams()

# first Hellinger k-NN (iter6-8): grid over k and the weighting
@dataclass(frozen=True)
class KNNGrid:
    ks: tuple = (1, 3, 5, 7, 10, 15, 20, 30, 50)
    weightings: tuple = ("uniform", "inv", "exp0.02", "exp0.05", "exp0.1")
    k_max: int = 50


KNN_GRID = KNNGrid()


@dataclass(frozen=True)
class KNNSearchSpace:
    """Grid of the in-ensemble k-NN search (knn-search)."""
    powers: tuple = (0.3, 0.4, 0.5, 0.7)
    idf: tuple = (False, True)
    ks: tuple = (20, 30, 50, 75, 100, 150, 200)
    taus: tuple = (0.005, 0.01, 0.02, 0.03, 0.05, 0.1)
    thresholds: tuple = (0.0, 0.3, 0.5, 0.7, 0.9)
    pseudo_weights: tuple = (0.5, 1.0)
    pseudo_ks: tuple = (50, 100, 200)
    k_max: int = 200
    nets: tuple = field(default=("iter7_nn", "iter8_nn"))


KNN_SEARCH = KNNSearchSpace()
