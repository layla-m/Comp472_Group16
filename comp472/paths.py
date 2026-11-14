"""Every filesystem location used by the project, in one place."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"                       # Kaggle CSVs (not versioned)
RESULTS_DIR = ROOT / "results"
PROBA_DIR = RESULTS_DIR / "probabilities"      # saved class probabilities (val / test)
SEED_STUDY_DIR = PROBA_DIR / "sentiment_seeds"  # seed / architecture study on sentiment
SUBMISSIONS_DIR = RESULTS_DIR / "submissions"  # iterN.csv files sent to Kaggle
SEARCH_DIR = RESULTS_DIR / "search"            # hyperparameter / k-NN search tables
LOGS_DIR = RESULTS_DIR / "logs"                # run logs and training histories
FIGURES_DIR = RESULTS_DIR / "figures"          # matplotlib figures


def ensure_dirs():
    for d in (PROBA_DIR, SEED_STUDY_DIR, SUBMISSIONS_DIR, SEARCH_DIR, LOGS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
