import numpy as np


def kmeans(X, k=3, max_iterations=100, random_state=42):

    # Choose k random examples as the initial centroids
    rng = np.random.default_rng(random_state)
    indices = rng.choice(len(X), size=k, replace=False)
    centroids = X[indices].copy()

    for _ in range(max_iterations):

        # Calculate distances between examples and centroids
        distances = np.linalg.norm(
            X[:, None, :] - centroids[None, :, :],
            axis=2
        )

        # Assign each example to its nearest centroid
        labels = np.argmin(distances, axis=1)

        # Calculate the new centroid of each cluster
        new_centroids = centroids.copy()

        for i in range(k):
            points = X[labels == i]

            if len(points) > 0:
                new_centroids[i] = np.mean(points, axis=0)

        # Stop if the centroids are no longer changing
        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    # Assign examples to the final centroids
    distances = np.linalg.norm(
        X[:, None, :] - centroids[None, :, :],
        axis=2
    )
    labels = np.argmin(distances, axis=1)

    return labels, centroids
