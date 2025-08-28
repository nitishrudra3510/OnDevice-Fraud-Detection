from pathlib import Path
import shutil
import json

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "4_Models/.models"
TFLITE_DIR = ROOT / "5_OnDevice_Optimization/.tflite"
CONFIG_FILE = ROOT / "4_Models/.config/decision_config.json"
SCALERS_DIR = ROOT / "3_Feature_Engineering/.artifacts/scalers"
OUT = ROOT / "deploy_bundle"


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def main():
    ensure_dir(OUT)
    # Copy TFLite models
    tflite_out = OUT / "models"
    ensure_dir(tflite_out)
    for f in TFLITE_DIR.glob("*.tflite"):
        shutil.copy2(f, tflite_out / f.name)

    # Copy Keras fallbacks (optional on-device with TF Mobile)
    keras_out = OUT / "keras"
    ensure_dir(keras_out)
    for f in MODELS_DIR.glob("*.keras"):
        shutil.copy2(f, keras_out / f.name)

    # Copy config
    if CONFIG_FILE.exists():
        shutil.copy2(CONFIG_FILE, OUT / CONFIG_FILE.name)
    else:
        (OUT / "decision_config.json").write_text(json.dumps({"weights":[0.25]*4,"threshold":0.6}, indent=2))

    # Copy scalers (feature normalization)
    scalers_out = OUT / "scalers"
    ensure_dir(scalers_out)
    if SCALERS_DIR.exists():
        for f in SCALERS_DIR.iterdir():
            shutil.copy2(f, scalers_out / f.name)

    # Minimal README
    (OUT / "README_DEPLOY.md").write_text(
        """
On-Device Multi-Agent Deploy Bundle

Contents
- models/: TensorFlow Lite models (autoencoder, movement_cnn, movement_lstm*)
- keras/: Keras models (optional fallback)
- scalers/: StandardScaler artifacts for feature normalization
- decision_config.json: Fusion weights and threshold

Notes
- Prefer movement_cnn_fp32.tflite or movement_cnn_dynamic.tflite for pure TFLite BUILTINS.
- LSTM TFLite models require Select TF ops (Flex delegate).
- Load TFLite models with TensorFlow Lite Interpreter in your mobile app.
""".strip()
    )

    print(f"Deployment bundle created at: {OUT}")


if __name__ == "__main__":
    main()


