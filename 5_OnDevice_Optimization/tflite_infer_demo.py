import numpy as np
import tensorflow as tf
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "deploy_bundle"
MODELS = BUNDLE / "models"
ART = ROOT / "3_Feature_Engineering/.artifacts"


def run_autoencoder_tflite(model_name: str = "app_usage_autoencoder_fp32.tflite"):
    model_path = MODELS / model_name
    assert model_path.exists(), f"Missing TFLite model: {model_path}"
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    input_details = interpreter.get_input_details()
    X = np.load(ART / "app_usage_features_X_test.npy")[:8].astype(np.float32)
    # Resize input to match batch
    in_shape = [len(X), X.shape[1]]
    interpreter.resize_tensor_input(input_details[0]['index'], in_shape)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], X)
    interpreter.invoke()
    recon = interpreter.get_tensor(output_details[0]['index'])
    mse = np.mean((X - recon) ** 2, axis=1)
    print(f"Autoencoder TFLite '{model_name}' -> batch={len(X)}  MSE mean={mse.mean():.6f}")


def run_movement_cnn_tflite(model_name: str = "movement_cnn_fp32.tflite", window: int = 10):
    model_path = MODELS / model_name
    assert model_path.exists(), f"Missing TFLite model: {model_path}"
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    input_details = interpreter.get_input_details()
    X = np.load(ART / "movement_features_X_test.npy").astype(np.float32)
    n = (len(X) // window) * window
    X_seq = X[:n].reshape(-1, window, X.shape[1])[:8]
    # Resize input to [batch, window, features]
    in_shape = [X_seq.shape[0], X_seq.shape[1], X_seq.shape[2]]
    interpreter.resize_tensor_input(input_details[0]['index'], in_shape)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], X_seq)
    interpreter.invoke()
    prob = interpreter.get_tensor(output_details[0]['index']).ravel()
    print(f"Movement CNN TFLite '{model_name}' -> batch={len(X_seq)}  mean_prob={prob.mean():.3f}")


if __name__ == "__main__":
    run_autoencoder_tflite("app_usage_autoencoder_fp32.tflite")
    run_autoencoder_tflite("app_usage_autoencoder_dynamic.tflite")
    # CNN models are preferred for BUILTINS-only
    if (MODELS / "movement_cnn_fp32.tflite").exists():
        run_movement_cnn_tflite("movement_cnn_fp32.tflite")
    if (MODELS / "movement_cnn_dynamic.tflite").exists():
        run_movement_cnn_tflite("movement_cnn_dynamic.tflite")

