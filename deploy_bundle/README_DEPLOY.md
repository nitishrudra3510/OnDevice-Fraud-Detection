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