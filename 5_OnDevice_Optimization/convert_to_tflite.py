from pathlib import Path
import tensorflow as tf


MODELS_DIR = Path(__file__).resolve().parents[1] / "4_Models/.models"
OUT_DIR = Path(__file__).resolve().parent / ".tflite"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def convert_keras_to_tflite(
    keras_path: Path,
    out_name: str,
    quantize_dynamic: bool = False,
    allow_select_tf_ops: bool = False,
):
    model = tf.keras.models.load_model(keras_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize_dynamic:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if allow_select_tf_ops:
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        # Required for some sequence models using TensorList
        converter._experimental_lower_tensor_list_ops = False
    tflite_model = converter.convert()
    out_path = OUT_DIR / f"{out_name}.tflite"
    out_path.write_bytes(tflite_model)
    print(f"Saved {out_path}")


def main():
    # Convert Autoencoder
    ae = MODELS_DIR / "app_usage_autoencoder.keras"
    if ae.exists():
        convert_keras_to_tflite(ae, "app_usage_autoencoder_fp32", quantize_dynamic=False)
        convert_keras_to_tflite(ae, "app_usage_autoencoder_dynamic", quantize_dynamic=True)

    # Convert LSTM
    lstm = MODELS_DIR / "movement_lstm.keras"
    if lstm.exists():
        convert_keras_to_tflite(lstm, "movement_lstm_fp32", quantize_dynamic=False, allow_select_tf_ops=True)
        convert_keras_to_tflite(lstm, "movement_lstm_dynamic", quantize_dynamic=True, allow_select_tf_ops=True)

    # Convert CNN (pure BUILTINS expected)
    cnn = MODELS_DIR / "movement_cnn.keras"
    if cnn.exists():
        convert_keras_to_tflite(cnn, "movement_cnn_fp32", quantize_dynamic=False, allow_select_tf_ops=False)
        convert_keras_to_tflite(cnn, "movement_cnn_dynamic", quantize_dynamic=True, allow_select_tf_ops=False)


if __name__ == "__main__":
    main()


