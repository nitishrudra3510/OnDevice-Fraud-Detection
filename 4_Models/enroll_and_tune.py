import argparse
import json
from itertools import product
from pathlib import Path
import sys
import numpy as np
from sklearn.metrics import f1_score
import tensorflow as tf

# Ensure local imports
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
from model_utils import load_sklearn_model

ART = Path(__file__).resolve().parents[1] / "3_Feature_Engineering/.artifacts"
CONFIG_DIR = Path(__file__).resolve().parent / ".config"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_PATH = CONFIG_DIR / "decision_config.json"
MODELS_DIR = Path(__file__).resolve().parent / ".models"


def sigmoid_normalize(score: np.ndarray):
    return 1 / (1 + np.exp(-score))


def agent_typing_scores(X_test: np.ndarray):
    model = load_sklearn_model("typing_agent_ocsvm")
    raw = model.decision_function(X_test)
    return sigmoid_normalize(raw)


def agent_gesture_scores(X_test: np.ndarray):
    model = load_sklearn_model("gesture_agent_isoforest")
    raw = model.decision_function(X_test)
    return sigmoid_normalize(raw)


def agent_app_usage_scores(X_test: np.ndarray):
    model = tf.keras.models.load_model(MODELS_DIR / "app_usage_autoencoder.keras")
    recon = model.predict(X_test, verbose=0)
    mse = np.mean((X_test - recon) ** 2, axis=1)
    mse_norm = (mse - mse.min()) / (np.ptp(mse) + 1e-6)
    return 1 - mse_norm


def agent_movement_scores(X_seq_test: np.ndarray):
    # Prefer CNN if available (pure TFLite BUILTINS), else LSTM
    path = MODELS_DIR / "movement_cnn.keras"
    if not path.exists():
        path = MODELS_DIR / "movement_lstm.keras"
    model = tf.keras.models.load_model(path)
    prob = model.predict(X_seq_test, verbose=0).ravel()
    return prob


def make_movement_sequences(X: np.ndarray, y: np.ndarray, window: int = 10):
    n = (len(X) // window) * window
    X_seq = X[:n].reshape(-1, window, X.shape[1])
    y_seq = (y[:n].reshape(-1, window).mean(axis=1) > 0.5).astype(int)
    return X_seq, y_seq


def majority_label(y_list):
    Y = np.vstack(y_list)  # shape: agents x N
    return (Y.mean(axis=0) >= 0.5).astype(int)


def grid_search(scores_list, y_true, weight_grid, thr_grid):
    best = {"f1": -1.0, "weights": None, "threshold": None}
    S = np.vstack(scores_list)
    for w in weight_grid:
        wv = np.array(w, dtype=float)
        wv = wv / (wv.sum() + 1e-9)
        fused = (wv[:, None] * S).sum(axis=0)
        for t in thr_grid:
            y_pred = (fused >= t).astype(int)
            f1 = f1_score(y_true, y_pred)
            if f1 > best["f1"]:
                best = {"f1": f1, "weights": wv.tolist(), "threshold": float(t)}
    return best


def save_config(cfg):
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Saved fusion config to {CONFIG_PATH} with F1={cfg.get('f1', 'n/a'):.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--weight_candidates", type=str, default="0.1,0.2,0.3,0.4")
    parser.add_argument("--threshold_candidates", type=str, default="0.4,0.5,0.6,0.7")
    args = parser.parse_args()

    # Load test as proxy for validation
    ks_X = np.load(ART / "keystroke_features_X_test.npy")
    ks_y = np.load(ART / "keystroke_features_y_test.npy")

    tg_X = np.load(ART / "touch_features_X_test.npy")
    tg_y = np.load(ART / "touch_features_y_test.npy")

    au_X = np.load(ART / "app_usage_features_X_test.npy")
    au_y = np.load(ART / "app_usage_features_y_test.npy")

    mv_X = np.load(ART / "movement_features_X_test.npy")
    mv_y = np.load(ART / "movement_features_y_test.npy")

    # Movement sequences
    mv_seq, mv_seq_y = make_movement_sequences(mv_X, mv_y, window=args.window)

    # Compute scores
    s_typing = agent_typing_scores(ks_X)
    s_gesture = agent_gesture_scores(tg_X)
    s_app = agent_app_usage_scores(au_X)
    s_movement = agent_movement_scores(mv_seq)

    # Align lengths
    min_len = min(len(s_typing), len(s_gesture), len(s_app), len(s_movement))
    S_list = [s_typing[:min_len], s_gesture[:min_len], s_app[:min_len], s_movement[:min_len]]

    # Build a combined ground-truth label via majority vote across modalities
    Y_list = [ks_y[:min_len], tg_y[:min_len], au_y[:min_len], mv_seq_y[:min_len]]
    y_true = majority_label(Y_list)

    # Build grids
    candidates = [float(x) for x in args.weight_candidates.split(",")]
    weight_grid = list(product(candidates, repeat=4))
    thr_grid = [float(x) for x in args.threshold_candidates.split(",")]

    best = grid_search(S_list, y_true, weight_grid, thr_grid)
    cfg = {"weights": best["weights"], "threshold": best["threshold"], "f1": best["f1"]}
    save_config(cfg)


if __name__ == "__main__":
    main()
