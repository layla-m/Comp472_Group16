import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from models.knn import predict

# Load the training and test datasets
image_train = pd.read_csv("data/image_train.csv")
image_test = pd.read_csv("data/image_test.csv")

sentiment_train = pd.read_csv("data/sentiment_train.csv")
sentiment_test = pd.read_csv("data/sentiment_test.csv")

# Get the feature columns
image_features = [col for col in image_train.columns if col.startswith("feature_")]
sentiment_features = [col for col in sentiment_train.columns if col.startswith("feature_")]

# Separate features and labels
X_image_train = image_train[image_features].to_numpy(dtype=np.float32)
y_image_train = image_train["label"].to_numpy(dtype=np.int64)
X_image_test = image_test[image_features].to_numpy(dtype=np.float32)

X_sent_train = sentiment_train[sentiment_features].to_numpy(dtype=np.float32)
y_sent_train = sentiment_train["label"].to_numpy(dtype=np.int64)
X_sent_test = sentiment_test[sentiment_features].to_numpy(dtype=np.float32)

# Standardize the image data using training statistics
mean = np.mean(X_image_train, axis=0)
std = np.std(X_image_train, axis=0)
std[std == 0] = 1

X_image_train = (X_image_train - mean) / std
X_image_test = (X_image_test - mean) / std

# Apply PCA to the image data
pca = PCA(n_components=150, random_state=42)

X_image_train = pca.fit_transform(X_image_train)
X_image_test = pca.transform(X_image_test)

# Predict the image test labels
print("Predicting image labels...")

image_predictions = predict(
    X_image_train,
    y_image_train,
    X_image_test,
    k=1,
    distance_type="euclidean",
    weighted=False
)

# Normalize the sentiment data using L2 normalization
train_norms = np.linalg.norm(X_sent_train, axis=1, keepdims=True)
test_norms = np.linalg.norm(X_sent_test, axis=1, keepdims=True)

train_norms[train_norms == 0] = 1
test_norms[test_norms == 0] = 1

X_sent_train = X_sent_train / train_norms
X_sent_test = X_sent_test / test_norms

# Predict the sentiment test labels
print("Predicting sentiment labels...")

sentiment_predictions = predict(
    X_sent_train,
    y_sent_train,
    X_sent_test,
    k=41,
    distance_type="euclidean",
    weighted=True
)

# Create a table with the predictions and their IDs
image_results = pd.DataFrame({
    "id": image_test["id"],
    "label": image_predictions
})

sentiment_results = pd.DataFrame({
    "id": sentiment_test["id"],
    "label": sentiment_predictions
})

# Combine predictions from both datasets
all_results = pd.concat(
    [image_results, sentiment_results],
    ignore_index=True
)

# Load the Kaggle submission template
sample_submission = pd.read_csv("data/sample_submission.csv")

# Match predictions to the IDs in the submission template
submission = sample_submission[["id"]].merge(
    all_results,
    on="id",
    how="left",
    validate="one_to_one"
)

# Check that every ID has a prediction
if submission["label"].isnull().any():
    raise ValueError("Some test examples are missing predictions.")

# Save the Kaggle submission file
submission.to_csv("results/knn_submission.csv", index=False)

print("Kaggle submission saved successfully!")
print("Total predictions:", len(submission))
