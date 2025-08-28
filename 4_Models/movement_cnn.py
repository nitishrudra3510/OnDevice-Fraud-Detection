import numpy as np
from pathlib import Path
import tensorflow as tf
from sklearn.metrics import classification_report


ART = Path(__file__).resolve().parents[1] / "3_Feature_Engineering/.artifacts"
MODELS_DIR = Path(__file__).resolve().parent / ".models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def build_cnn(input_dim: int):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(None, input_dim)),  # time-major sequence
        tf.keras.layers.Conv1D(32, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.Conv1D(32, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def make_sequences(X: np.ndarray, y: np.ndarray, window: int = 10):
    n = (len(X) // window) * window
    X = X[:n]
    y = y[:n]
    X_seq = X.reshape(-1, window, X.shape[1])
    y_seq = (y.reshape(-1, window).mean(axis=1) > 0.5).astype(int)
    return X_seq, y_seq


def main():
    X_train = np.load(ART / "movement_features_X_train.npy")
    y_train = np.load(ART / "movement_features_y_train.npy")
    X_test = np.load(ART / "movement_features_X_test.npy")
    y_test = np.load(ART / "movement_features_y_test.npy")

    Xtr_seq, ytr_seq = make_sequences(X_train, y_train)
    Xte_seq, yte_seq = make_sequences(X_test, y_test)

    model = build_cnn(Xtr_seq.shape[-1])
    model.fit(Xtr_seq, ytr_seq, epochs=10, batch_size=64, validation_split=0.1, verbose=0)
    model.save(MODELS_DIR / "movement_cnn.keras")

    y_prob = model.predict(Xte_seq, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    print(classification_report(yte_seq, y_pred, digits=3))


if __name__ == "__main__":
    main()


