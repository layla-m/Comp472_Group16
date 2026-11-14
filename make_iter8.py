"""
iter8.csv = image: MLP probabilities + Hellinger k-NN probabilities, like iter7, but with the
iter4 network ([8192], 100 epochs). Two MLP options are compared on the 80/20 validation split:
  - iter8_nn alone ([8192])
  - mean of iter7_nn ([4096]) and iter8_nn ([8192])
and the k-NN weight is chosen on the same split. Sentiment: MLP (iter8_nn, same as before).

Requires: nn_v2.py iter7_nn.csv, nn_v2.py iter8_nn.csv, knn_hellinger.py
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from neural_network import SEED, load_task

X, y, _, test_ids = load_task("image")
_, _, _, y_va = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
load = lambda tag, part: np.load(f"ens/{tag}_image_{part}.npy")
K_va, K_te = np.load("ens/knn_image_val.npy"), np.load("ens/knn_image_test.npy")
options = {
    "[8192]": (load("iter8_nn", "val"), load("iter8_nn", "test")),
    "mean([4096],[8192])": ((load("iter7_nn", "val") + load("iter8_nn", "val")) / 2,
                            (load("iter7_nn", "test") + load("iter8_nn", "test")) / 2),
}
ws = np.round(np.arange(0, 1.001, 0.05), 2)
best = (-1, None, None)
for name, (N_va, _) in options.items():
    accs = [((w * N_va + (1 - w) * K_va).argmax(1) == y_va).mean() for w in ws]
    i = int(np.argmax(accs))
    print(f"{name:22s} nn alone {accs[-1]:.4f}  best w_nn={ws[i]:.2f} -> {accs[i]:.4f}   "
          + " ".join(f"{w:.2f}:{a:.4f}" for w, a in zip(ws, accs) if 0.3 <= w <= 0.8))
    if accs[i] > best[0]:
        best = (accs[i], name, ws[i])
acc, name, w = best
print(f"BEST {name} w_nn={w:.2f} val_acc={acc:.4f}")

N_te = options[name][1]
img = pd.DataFrame({"id": test_ids, "label": (w * N_te + (1 - w) * K_te).argmax(1)})
nn = pd.read_csv("iter8_nn.csv")
sen = nn[nn["id"].str.startswith("text_")]
sample = pd.read_csv("sample_submission.csv")
sub = sample[["id"]].merge(pd.concat([img, sen]), on="id", how="left", validate="one_to_one")
assert len(sub) == 10000 and sub["label"].notna().all()
sub["label"] = sub["label"].astype(int)
assert sub.loc[sub["id"].str.startswith("img_"), "label"].between(0, 49).all()
assert sub.loc[sub["id"].str.startswith("text_"), "label"].between(0, 2).all()
sub.to_csv("iter8.csv", index=False)
print("Saved iter8.csv")
