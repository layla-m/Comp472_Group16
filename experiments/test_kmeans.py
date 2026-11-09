import numpy as np
from models.kmeans import kmeans

# Load the sentiment training data
data = np.load("data/processed/sentiment_split.npz")

X_train = data["X_train"]
y_train = data["y_train"]

# Test k-Means with and without normalization
for normalized in [False, True]:

    if normalized:
        norms = np.linalg.norm(X_train, axis=1, keepdims=True)
        norms[norms == 0] = 1
        X = X_train / norms
    else:
        X = X_train

    # Run k-Means with 3 clusters
    clusters, centroids = kmeans(X, k=3)

    print("\nNormalized:", normalized)
    print("Cluster | Negative | Neutral | Positive")

    # Compare clusters with the actual sentiment labels
    for i in range(3):
        actual_labels = y_train[clusters == i]
        counts = np.bincount(actual_labels, minlength=3)

        print(i, "     |", counts[0], "     |",
              counts[1], "    |", counts[2])
