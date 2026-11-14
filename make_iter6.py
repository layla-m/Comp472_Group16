"""
iter6.csv = image predicted by the k-NN with the Hellinger distance (knn_hellinger.py,
k=50, exp weights tau=0.02, val 0.589) + sentiment taken from iter4.csv (MLP, val 0.682).

Requires: python knn_hellinger.py (writes ens/knn_image_test.npy) and iter4.csv.
"""
import numpy as np
import pandas as pd

P = np.load("ens/knn_image_test.npy")
test_ids = pd.read_csv("image_test.csv", usecols=["id"])["id"]
img = pd.DataFrame({"id": test_ids, "label": P.argmax(1)})

it4 = pd.read_csv("iter4.csv")
sen = it4[it4["id"].str.startswith("text_")]

sample = pd.read_csv("sample_submission.csv")
sub = sample[["id"]].merge(pd.concat([img, sen]), on="id", how="left", validate="one_to_one")
assert len(sub) == 10000 and sub["label"].notna().all()
sub["label"] = sub["label"].astype(int)
assert sub.loc[sub["id"].str.startswith("img_"), "label"].between(0, 49).all()
assert sub.loc[sub["id"].str.startswith("text_"), "label"].between(0, 2).all()
sub.to_csv("iter6.csv", index=False)
print("Saved iter6.csv")
