import numpy as np
from pathlib import Path
import tensorflow as tf
import sys
import json
import argparse

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
from model_utils import load_sklearn_model


ART = Path(__file__).resolve().parents[1] / "3_Feature_Engineering/.artifacts"
MODELS_DIR = Path(__file__).resolve().parent / ".models"
CONFIG_DIR = Path(__file__).resolve().parent / ".config"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_PATH = CONFIG_DIR / "decision_config.json"


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
    # Lower MSE => more normal; convert to score in [0,1]
    mse_norm = (mse - mse.min()) / (np.ptp(mse) + 1e-6)
    return 1 - mse_norm


def agent_movement_scores(X_seq_test: np.ndarray):
    cnn_path = MODELS_DIR / "movement_cnn.keras"
    lstm_path = MODELS_DIR / "movement_lstm.keras"
    load_path = cnn_path if cnn_path.exists() else lstm_path
    model = tf.keras.models.load_model(load_path)
    prob = model.predict(X_seq_test, verbose=0).ravel()
    return prob


def fuse_scores(scores_list, weights=None, threshold=0.5):
    scores = np.vstack(scores_list)
    if weights is None:
        weights = np.ones(scores.shape[0]) / scores.shape[0]
    fused = np.average(scores, axis=0, weights=weights)
    decisions = (fused >= threshold).astype(int)
    return fused, decisions


def load_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    return {"weights": [0.25, 0.25, 0.25, 0.25], "threshold": 0.6}


def save_config(cfg):
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Saved fusion config to {CONFIG_PATH}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, default=None, help="Comma-separated weights for agents in order: typing,gesture,app,movement")
    p.add_argument("--threshold", type=float, default=None, help="Decision threshold in [0,1]")
    p.add_argument("--save_config", action="store_true", help="Persist provided weights/threshold to config")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config()
    if args.weights is not None:
        cfg["weights"] = [float(x) for x in args.weights.split(",")]
    if args.threshold is not None:
        cfg["threshold"] = float(args.threshold)
    if args.save_config:
        save_config(cfg)

    # Load preprocessed features
    ks_X = np.load(ART / "keystroke_features_X_test.npy")
    tg_X = np.load(ART / "touch_features_X_test.npy")
    au_X = np.load(ART / "app_usage_features_X_test.npy")
    mv_X = np.load(ART / "movement_features_X_test.npy")

    # Create sequences for movement to match LSTM input
    window = 10
    n = (len(mv_X) // window) * window
    mv_seq = mv_X[:n].reshape(-1, window, mv_X.shape[1])

    s_typing = agent_typing_scores(ks_X)
    s_gesture = agent_gesture_scores(tg_X)
    s_app = agent_app_usage_scores(au_X)
    s_movement = agent_movement_scores(mv_seq)

    # Align lengths by truncating to the shortest
    min_len = min(len(s_typing), len(s_gesture), len(s_app), len(s_movement))
    fused, decisions = fuse_scores([
        s_typing[:min_len], s_gesture[:min_len], s_app[:min_len], s_movement[:min_len]
    ], weights=np.array(cfg["weights"]) / (np.sum(cfg["weights"]) + 1e-9), threshold=cfg["threshold"])

    normal_pct = (decisions == 1).mean() * 100
    print(f"Decision Agent: {normal_pct:.1f}% samples classified as normal")


if __name__ == "__main__":
    main()


