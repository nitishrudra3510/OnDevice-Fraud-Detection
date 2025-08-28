from pathlib import Path
import joblib
import numpy as np


MODELS_DIR = Path(__file__).resolve().parent / ".models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def save_sklearn_model(model, name: str):
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    return path


def load_sklearn_model(name: str):
    path = MODELS_DIR / f"{name}.joblib"
    return joblib.load(path)


def save_numpy_arrays(prefix: Path, **arrays):
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for key, value in arrays.items():
        np.save(prefix.parent / f"{prefix.name}_{key}.npy", value)
    return prefix


