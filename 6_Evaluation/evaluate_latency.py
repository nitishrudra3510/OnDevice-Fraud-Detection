import time
import numpy as np
from pathlib import Path
import tensorflow as tf
from statistics import mean


ART = Path(__file__).resolve().parents[1] / "3_Feature_Engineering/.artifacts"
MODELS_DIR = Path(__file__).resolve().parents[1] / "4_Models/.models"


def time_inference(model, inputs, runs: int = 50):
    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        _ = model.predict(inputs, verbose=0)
        timings.append((time.perf_counter() - start) * 1000)
    return {
        "p50_ms": np.percentile(timings, 50),
        "p90_ms": np.percentile(timings, 90),
        "p99_ms": np.percentile(timings, 99),
        "mean_ms": mean(timings),
    }


def main():
    ae = tf.keras.models.load_model(MODELS_DIR / "app_usage_autoencoder.keras")
    lstm = tf.keras.models.load_model(MODELS_DIR / "movement_lstm.keras")

    X_app = np.load(ART / "app_usage_features_X_test.npy")[:64]
    stats_ae = time_inference(ae, X_app)
    print("Autoencoder inference (batch 64):", stats_ae)

    X_mv = np.load(ART / "movement_features_X_test.npy")
    window = 10
    n = (len(X_mv) // window) * window
    X_mv_seq = X_mv[:n].reshape(-1, window, X_mv.shape[1])[:64]
    stats_lstm = time_inference(lstm, X_mv_seq)
    print("LSTM inference (batch 64):", stats_lstm)

    # Battery proxy: CPU time approximated by sum of inference mean times
    est_cpu_ms = stats_ae["mean_ms"] + stats_lstm["mean_ms"]
    print(f"Estimated per-cycle CPU time: {est_cpu_ms:.2f} ms")


if __name__ == "__main__":
    main()


