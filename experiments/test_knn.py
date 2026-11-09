import numpy as np
from sklearn.decomposition import PCA
from models.knn import predict
from evaluation import evaluate_model

# Evaluate k-NN on both datasets using the best settings
for dataset in ["image", "sentiment"]:

    # Load the training and validation data
    data = np.load(f"data/processed/{dataset}_split.npz")

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]

    if dataset == "image":

        # Standardize using the training data
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
        std[std == 0] = 1

        X_train = (X_train - mean) / std
        X_val = (X_val - mean) / std

        # Reduce the image features using PCA
        pca = PCA(n_components=150, random_state=42)

        X_train = pca.fit_transform(X_train)
        X_val = pca.transform(X_val)

        k = 1
        weighted = False

    else:

        # Normalize each sentiment example using its vector length
        train_norms = np.linalg.norm(X_train, axis=1, keepdims=True)
        val_norms = np.linalg.norm(X_val, axis=1, keepdims=True)

        train_norms[train_norms == 0] = 1
        val_norms[val_norms == 0] = 1

        X_train = X_train / train_norms
        X_val = X_val / val_norms

        k = 41
        weighted = True

    print("\nDATASET:", dataset)

    # Make predictions using the best k-NN settings
    predictions = predict(
        X_train,
        y_train,
        X_val,
        k=k,
        distance_type="euclidean",
        weighted=weighted
    )

    # Evaluate the predictions
    accuracy, matrix = evaluate_model(y_val, predictions)

    # Save the predictions and confusion matrix
    np.save(f"results/{dataset}_knn_predictions.npy", predictions)
    np.save(f"results/{dataset}_knn_confusion_matrix.npy", matrix)

    print("Results saved!")
