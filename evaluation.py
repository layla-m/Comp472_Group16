from sklearn.metrics import accuracy_score, confusion_matrix

# Function to evaluate the predictions made by a model
def evaluate_model(y_true, y_pred):

    # Calculate how many predictions the model got correct
    accuracy = accuracy_score(y_true, y_pred)

    # Show which classes were predicted correctly or incorrectly
    matrix = confusion_matrix(y_true, y_pred)

    # Print the accuracy and confusion matrix
    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Confusion Matrix:")
    print(matrix)

    return accuracy, matrix


# POTENTIALLY NEEDS TO BE DELETED BEFORE COMMITING
# Test the evaluation function with some example labels
if __name__ == "__main__":
    y_true = [0, 1, 2, 0, 1]
    y_pred = [0, 1, 1, 0, 2]

    evaluate_model(y_true, y_pred)
