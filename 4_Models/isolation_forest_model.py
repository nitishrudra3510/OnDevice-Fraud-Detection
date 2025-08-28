import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
import sys

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
from model_utils import save_sklearn_model


ART = Path(__file__).resolve().parents[1] / "3_Feature_Engineering/.artifacts"


def main():
    X_train = np.load(ART / "touch_features_X_train.npy")
    y_train = np.load(ART / "touch_features_y_train.npy")
    X_test = np.load(ART / "touch_features_X_test.npy")
    y_test = np.load(ART / "touch_features_y_test.npy")

    normal_mask = y_train == 1
    iso = IsolationForest(n_estimators=200, contamination=0.1, random_state=42)
    iso.fit(X_train[normal_mask])

    pred_test = iso.predict(X_test)
    y_pred = (pred_test == 1).astype(int)
    print(classification_report(y_test, y_pred, digits=3))
    save_sklearn_model(iso, "gesture_agent_isoforest")


if __name__ == "__main__":
    main()


