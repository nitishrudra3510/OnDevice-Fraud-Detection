import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import json


ARTIFACTS = Path(__file__).resolve().parent / ".artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)
SCALERS_DIR = ARTIFACTS / "scalers"
SCALERS_DIR.mkdir(parents=True, exist_ok=True)


def preprocess_feature_csv(csv_path: Path, label_col: str = "label"):
    df = pd.read_csv(csv_path)
    features = df.drop(columns=[label_col, "user_id"], errors="ignore")
    labels = df[label_col] if label_col in df.columns else None
    X_train, X_test, y_train, y_test = train_test_split(
        features.values, labels.values if labels is not None else None, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def main():
    for name in [
        "keystroke_features.csv",
        "touch_features.csv",
        "app_usage_features.csv",
        "movement_features.csv",
    ]:
        X_train, X_test, y_train, y_test, scaler = preprocess_feature_csv(ARTIFACTS / name)
        base = name.replace(".csv", "")
        np.save(ARTIFACTS / f"{base}_X_train.npy", X_train)
        np.save(ARTIFACTS / f"{base}_X_test.npy", X_test)
        if y_train is not None:
            np.save(ARTIFACTS / f"{base}_y_train.npy", y_train)
            np.save(ARTIFACTS / f"{base}_y_test.npy", y_test)
        print(f"Preprocessed {name}")
        # Persist scaler
        joblib.dump(scaler, SCALERS_DIR / f"{base}_scaler.joblib")
        stats = {
            "mean": getattr(scaler, "mean_", []).tolist() if hasattr(scaler, "mean_") else [],
            "scale": getattr(scaler, "scale_", []).tolist() if hasattr(scaler, "scale_") else [],
            "var": getattr(scaler, "var_", []).tolist() if hasattr(scaler, "var_") else [],
            "n_features_in": int(getattr(scaler, "n_features_in_", 0)),
        }
        with open(SCALERS_DIR / f"{base}_stats.json", "w") as f:
            json.dump(stats, f)


if __name__ == "__main__":
    main()


