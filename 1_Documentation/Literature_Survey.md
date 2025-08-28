# Literature Survey

This survey summarizes key research and real-world use cases for behavior-based anomaly and fraud detection, focusing on on-device feasibility.

## Behavioral Biometrics

- Keystroke Dynamics: Reviews and surveys show high efficacy for continuous authentication.
  - Teh et al., "Keystroke dynamics in password authentication: A survey," Computers & Security, 2013.
  - Monaco & Tappert, "The Partially Observable Hidden Markov Model and its application to keystroke dynamics," 2012.
- Touch & Gesture Dynamics:
  - Frank et al., "Touchalytics: On the applicability of touchscreen input as a behavioral biometric for continuous authentication," IEEE TIFS, 2013.
  - Zheng et al., "You are how you touch: User verification on smartphones via tapping behaviors," IEEE TIFS, 2014.
- App Usage Patterns:
  - Malhotra et al., "LSTM-based Anomaly Detection for User Behavior Modeling," arXiv.
- Movement/GPS Patterns:
  - Zheng, "TrajPattern: A survey of trajectory pattern mining," ACM TIST.

## Models for Anomaly Detection

- One-Class SVM: Effective for boundary learning when only normal data available.
- Isolation Forest: Ensemble isolation of anomalies; robust and efficient.
- Autoencoders: Reconstruction-based anomaly detection for tabular and sequence data.
- RNN/LSTM: Captures temporal dependencies in sequences (e.g., GPS/accelerometer).

## On-Device ML and Privacy

- TensorFlow Lite and PyTorch Mobile enable on-device inference with quantization/pruning.
- Differential privacy and federated learning can further enhance privacy (future work).

## Real-World Use Cases

- FinTech: Detect anomalous payment patterns, account takeover, and bot usage.
- Cybersecurity: Device compromise detection via behavior drift.
- Authentication: Continuous authentication and step-up verification triggers.

## Datasets (Public)

- Keystroke: CMU Keystroke Dynamics Benchmark; Aalto University typing behavior datasets.
- Touch: Touchalytics dataset; DeepLogger (gesture) datasets.
- Mobility: Geolife GPS Trajectories; CRAWDAD mobility traces.

References: Provide appropriate links in report and cite as per format.

