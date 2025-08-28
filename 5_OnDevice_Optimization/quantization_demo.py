from pathlib import Path
import numpy as np
import tensorflow as tf


ART = Path(__file__).resolve().parents[1] / "3_Feature_Engineering/.artifacts"
MODELS_DIR = Path(__file__).resolve().parents[1] / "4_Models/.models"
OUT_DIR = Path(__file__).resolve().parent / ".tflite"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def representative_dataset(file_path: Path, batch_size: int = 100):
    X = np.load(file_path)
    for i in range(0, min(len(X), batch_size), 1):
        yield [X[i:i+1].astype(np.float32)]


def post_training_int8(keras_path: Path, rep_data: Path, out_name: str):
    model = tf.keras.models.load_model(keras_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(rep_data)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_model = converter.convert()
    out_path = OUT_DIR / f"{out_name}.tflite"
    out_path.write_bytes(tflite_model)
    print(f"Saved INT8 model {out_path}")


def main():
    ae = MODELS_DIR / "app_usage_autoencoder.keras"
    if ae.exists():
        post_training_int8(ae, ART / "app_usage_features_X_train.npy", "app_usage_autoencoder_int8")

    lstm = MODELS_DIR / "movement_lstm.keras"
    if lstm.exists():
        # Note: For sequence models, representative data must have matching input shapes
        # Here we approximate with flattened features per step, assuming same input shape.
        post_training_int8(lstm, ART / "movement_features_X_train.npy", "movement_lstm_int8")


if __name__ == "__main__":
    main()


