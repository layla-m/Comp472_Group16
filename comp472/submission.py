"""Building, checking and saving a Kaggle submission (10,000 rows: id,label)."""
import pandas as pd

from .data import ID_PREFIX
from .paths import DATA_DIR, SUBMISSIONS_DIR

N_ROWS = 10000
LABEL_RANGE = {"image": (0, 49), "sentiment": (0, 2)}


def predictions(test_ids, labels):
    return pd.DataFrame({"id": test_ids, "label": labels})


def build_submission(image, sentiment):
    """image, sentiment: DataFrames (id, label). Rows follow sample_submission.csv."""
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    sub = sample[["id"]].merge(pd.concat([image, sentiment]), on="id", how="left",
                               validate="one_to_one")
    check_submission(sub)
    sub["label"] = sub["label"].astype(int)
    return sub


def check_submission(sub):
    assert len(sub) == N_ROWS, f"{len(sub)} rows instead of {N_ROWS}"
    assert sub["label"].notna().all(), "missing labels"
    for task, (lo, hi) in LABEL_RANGE.items():
        rows = sub["id"].str.startswith(ID_PREFIX[task])
        assert sub.loc[rows, "label"].between(lo, hi).all(), f"{task} label out of range"


def save_submission(sub, name):
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = SUBMISSIONS_DIR / f"{name}.csv"
    sub.to_csv(path, index=False)
    return path


def load_submission(name):
    return pd.read_csv(SUBMISSIONS_DIR / f"{name}.csv")


def agreement(a, b, task=None):
    """Fraction of identical labels between two submissions (optionally one task only)."""
    m = a.merge(b, on="id")
    if task:
        m = m[m["id"].str.startswith(ID_PREFIX[task])]
    return (m["label_x"] == m["label_y"]).mean()
