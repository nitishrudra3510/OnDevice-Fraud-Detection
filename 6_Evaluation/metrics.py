import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def evaluate_binary(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm.tolist(),
    }


if __name__ == "__main__":
    # Example
    y_true = np.array([1, 1, 0, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 0, 0, 1, 1, 1])
    print(evaluate_binary(y_true, y_pred))


