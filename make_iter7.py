"""
iter7.csv = image: weighted average of the MLP probabilities (iter7_nn, iter2 config) and of
the Hellinger k-NN probabilities (knn_hellinger.py); sentiment: MLP (iter7_nn).
The weight is chosen on the 80/20 validation split (val probabilities of both models).

Requires: python nn_v2.py iter7_nn.csv  and  python knn_hellinger.py
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from neural_network import SEED, load_task

X, y, _, test_ids = load_task("image")
_, _, _, y_va = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
N_va, K_va = np.load("ens/iter7_nn_image_val.npy"), np.load("ens/knn_image_val.npy")
N_te, K_te = np.load("ens/iter7_nn_image_test.npy"), np.load("ens/knn_image_test.npy")

ws = np.round(np.arange(0, 1.001, 0.05), 2)
accs = [((w * N_va + (1 - w) * K_va).argmax(1) == y_va).mean() for w in ws]
for w, a in zip(ws, accs):
    print(f"w_nn={w:.2f}  val_acc={a:.4f}")
w = ws[int(np.argmax(accs))]
print(f"BEST w_nn={w:.2f} val_acc={max(accs):.4f}  (nn alone {accs[-1]:.4f}, knn alone {accs[0]:.4f})")

img = pd.DataFrame({"id": test_ids, "label": (w * N_te + (1 - w) * K_te).argmax(1)})
nn = pd.read_csv("iter7_nn.csv")
sen = nn[nn["id"].str.startswith("text_")]
sample = pd.read_csv("sample_submission.csv")
sub = sample[["id"]].merge(pd.concat([img, sen]), on="id", how="left", validate="one_to_one")
assert len(sub) == 10000 and sub["label"].notna().all()
sub["label"] = sub["label"].astype(int)
assert sub.loc[sub["id"].str.startswith("img_"), "label"].between(0, 49).all()
assert sub.loc[sub["id"].str.startswith("text_"), "label"].between(0, 2).all()
sub.to_csv("iter7.csv", index=False)
print("Saved iter7.csv")
