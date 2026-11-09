import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

# Load training datasets
image_df = pd.read_csv("data/image_train.csv")
sentiment_df = pd.read_csv("data/sentiment_train.csv")

# Use the same random seed so we get the same split each time
SEED = 42

# Function to separate the features from the labels
def extract_data(df):

    # Get all the columns that contain features
    feature_columns = [
        col for col in df.columns
        if col.startswith("feature_")
    ]

    # Convert the features and labels into NumPy arrays
    # x contains the numerical features used to make predictions
    # y contains the correct class labels that the model is trying to predict
    X = df[feature_columns].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int64)

    return X, y


# Extract the features and labels for the image dataset
X_image, y_image = extract_data(image_df)

# Split the image dataset into 80% training and 20% validation
X_img_train, X_img_val, y_img_train, y_img_val = train_test_split(
    X_image,
    y_image,
    test_size=0.2,
    random_state=SEED,
    stratify=y_image
)

# Extract the features and labels for the sentiment dataset
X_sentiment, y_sentiment = extract_data(sentiment_df)

# Split the sentiment dataset into 80% training and 20% validation
X_sent_train, X_sent_val, y_sent_train, y_sent_val = train_test_split(
    X_sentiment,
    y_sentiment,
    test_size=0.2,
    random_state=SEED,
    stratify=y_sentiment
)

# Create a folder to store the processed datasets
Path("data/processed").mkdir(parents=True, exist_ok=True)

# Save the image training and validation data
np.savez_compressed(
    "data/processed/image_split.npz",
    X_train=X_img_train,
    X_val=X_img_val,
    y_train=y_img_train,
    y_val=y_img_val
)

# Save the sentiment training and validation data
np.savez_compressed(
    "data/processed/sentiment_split.npz",
    X_train=X_sent_train,
    X_val=X_sent_val,
    y_train=y_sent_train,
    y_val=y_sent_val
)

# Confirm that both datasets were saved
print("\nDatasets saved successfully!")
