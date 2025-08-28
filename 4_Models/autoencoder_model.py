import numpy as np
from pathlib import Path
import tensorflow as tf
from sklearn.metrics import classification_report


ART = Path(__file__).resolve().parents[1] / "3_Feature_Engineering/.artifacts"
MODELS_DIR = Path(__file__).resolve().parent / ".models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def build_autoencoder(input_dim: int):
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(32, activation="relu")(inputs)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    bottleneck = tf.keras.layers.Dense(8, activation="relu")(x)
    x = tf.keras.layers.Dense(16, activation="relu")(bottleneck)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(input_dim, activation=None)(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    return model


def main():
    X_train = np.load(ART / "app_usage_features_X_train.npy")
    y_train = np.load(ART / "app_usage_features_y_train.npy")
    X_test = np.load(ART / "app_usage_features_X_test.npy")
    y_test = np.load(ART / "app_usage_features_y_test.npy")

    normal_mask = y_train == 1
    X_train_normal = X_train[normal_mask]

    model = build_autoencoder(X_train.shape[1])
    model.fit(X_train_normal, X_train_normal, epochs=20, batch_size=64, validation_split=0.1, verbose=0)
    model.save(MODELS_DIR / "app_usage_autoencoder.keras")

    recon = model.predict(X_test, verbose=0)
    mse = np.mean(np.square(X_test - recon), axis=1)
    threshold = np.percentile(mse[y_test == 1], 95)
    y_pred = (mse <= threshold).astype(int)
    print(classification_report(y_test, y_pred, digits=3))


if __name__ == "__main__":
    main()


