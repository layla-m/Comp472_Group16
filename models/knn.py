import numpy as np

# Function to predict the class of one example using k-NN
def predict_one(X_train, y_train, x, k=3, distance_type="euclidean", weighted=False):

    # Calculate the distance from x to every training example
    if distance_type == "euclidean":
        distances = np.sqrt(np.sum((X_train - x) ** 2, axis=1))

    elif distance_type == "manhattan":
        distances = np.sum(np.abs(X_train - x), axis=1)

    elif distance_type == "cosine":
        # Calculate cosine similarity between x and the training examples
        dot_products = np.sum(X_train * x, axis=1)
        train_norms = np.linalg.norm(X_train, axis=1)
        x_norm = np.linalg.norm(x)

        # Convert cosine similarity into distance
        similarities = dot_products / (train_norms * x_norm + 1e-8)
        distances = 1 - similarities

    # Find the indices of the k closest training examples
    nearest_indices = np.argsort(distances)[:k]

    # Get the labels and distances of the closest examples
    nearest_labels = y_train[nearest_indices]
    nearest_distances = distances[nearest_indices]

    if weighted:
        # Give closer neighbours more weight
        weights = 1 / (nearest_distances + 1e-8)

        # Add the weights for each class
        counts = np.bincount(nearest_labels, weights=weights)

    else:
        # Count how many times each class appears
        counts = np.bincount(nearest_labels)

    # Predict the class with the most votes or highest weight
    prediction = np.argmax(counts)

    return prediction


# Function to predict the classes of multiple examples
def predict(X_train, y_train, X_val, k=3, distance_type="euclidean", weighted=False):

    predictions = []

    # Make a prediction for each validation example
    for x in X_val:
        prediction = predict_one(X_train, y_train, x, k, distance_type, weighted)
        predictions.append(prediction)

    return np.array(predictions)


# Test k-NN with a small example
if __name__ == "__main__":

    X_train = np.array([
        [1, 1],
        [2, 2],
        [3, 3],
        [8, 8],
        [9, 9]
    ])

    y_train = np.array([0, 0, 0, 1, 1])

    X_val = np.array([
        [2, 3],
        [8, 9]
    ])

    predictions = predict(X_train, y_train, X_val, k=3)

    print("Predictions:", predictions)
