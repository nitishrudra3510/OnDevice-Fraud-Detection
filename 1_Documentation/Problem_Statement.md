# Problem Statement

Modern mobile devices are prime targets for account takeover, device theft, and fraud. Traditional server-side anomaly detection captures only a subset of signals and raises latency and privacy concerns. We need an on-device, privacy-preserving system that continuously models user behavior (typing rhythm, touch/gesture dynamics, app usage, and movement) to detect anomalies in near real time and trigger protective actions without sending raw behavioral data off the device.

# Objectives

- Build lightweight on-device agents that model behavioral modalities:
  - Typing (keystroke dynamics)
  - Touch/gesture (swipes, taps, pressure)
  - App usage (hours, frequency, session patterns)
  - Movement (GPS/accelerometer temporal patterns)
- Fuse agent outputs in a decision agent to classify activity as normal or suspicious.
- Optimize models for on-device deployment (quantization, pruning) with minimal accuracy loss.
- Provide an end-to-end pipeline: data, features, models, evaluation, and demo UI.

# Scope

- Data: Use public datasets where available; otherwise generate realistic synthetic datasets.
- Features: Implement robust feature extraction and preprocessing.
- Models: One-Class SVM, Isolation Forest, Autoencoder, LSTM.
- Optimization: Convert to TensorFlow Lite / PyTorch Mobile; demonstrate quantization.
- Evaluation: Standard metrics, confusion matrices, latency and power proxy measurements.
- Demo: Kivy-based UI stub and guidance for Android/Flutter integration.

# Out of Scope

- Live production telemetry collection and MDM integration.
- Biometric identity verification beyond behavior-based signals.

