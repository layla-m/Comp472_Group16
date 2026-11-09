import pandas as pd

# Load predictions from both models
knn = pd.read_csv("results/knn_submission.csv")
rf = pd.read_csv("results/rf_submission.csv")

# Use k-NN predictions for images
image_predictions = knn[knn["id"].str.startswith("img_")]

# Use Random Forest predictions for sentiment
sentiment_predictions = rf[rf["id"].str.startswith("text_")]

# Combine both sets of predictions
predictions = pd.concat(
    [image_predictions, sentiment_predictions],
    ignore_index=True
)

# Match the original Kaggle submission order
sample = pd.read_csv("data/sample_submission.csv")

submission = sample[["id"]].merge(
    predictions,
    on="id",
    how="left",
    validate="one_to_one"
)

# Check that every example has a prediction
assert len(submission) == 10000
assert submission["label"].notna().all()

submission["label"] = submission["label"].astype(int)

# Save the new submission
submission.to_csv("results/hybrid_submission.csv", index=False)

print("Hybrid submission saved successfully!")
print("Total predictions:", len(submission))
