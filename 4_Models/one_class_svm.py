import numpy as np
from pathlib import Path
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report
import sys

# Ensure local imports work when running as a script
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
from model_utils import save_sklearn_model


ART = Path(__file__).resolve().parents[1] / "3_Feature_Engineering/.artifacts"


def main():
    X_train = np.load(ART / "keystroke_features_X_train.npy")
    y_train = np.load(ART / "keystroke_features_y_train.npy")
    X_test = np.load(ART / "keystroke_features_X_test.npy")
    y_test = np.load(ART / "keystroke_features_y_test.npy")

    normal_mask = y_train == 1
    ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
    ocsvm.fit(X_train[normal_mask])

    pred_train = ocsvm.predict(X_train)
    pred_test = ocsvm.predict(X_test)
    # Map OneClassSVM outputs {1, -1} to {1, 0}
    y_pred = (pred_test == 1).astype(int)
    print(classification_report(y_test, y_pred, digits=3))
    save_sklearn_model(ocsvm, "typing_agent_ocsvm")


if __name__ == "__main__":
    main()


