Samsung EnnovateX 2025 AI Challenge Submission

- Problem Statement: On-Device Multi-Agent System for Behavior-Based Anomaly & Fraud Detection
- Team name: <Your Team Name>
- Team members (Names): <Member 1>, <Member 2>, <Member 3>, <Member 4>
- Demo Video Link: <YouTube URL>

Project Artefacts

- Technical Documentation: see docs in `1_Documentation/` (Problem, Literature, System Design)
- Source Code: this repository (`src` equivalent under project folders)
- Models Used: scikit-learn (OC-SVM, Isolation Forest), TensorFlow (Autoencoder, CNN/LSTM)
- Models Published: TFLite and Keras artifacts included in `deploy_bundle/`
- Datasets Used: Synthetic generator in `2_Datasets/` with schema mirroring public datasets
- Datasets Published: Synthetic CSVs generated locally (see generator)

Attribution

- Built by the team for Samsung EnnovateX 2025. Template structure adapted from the submission guidance [ai-challenge-submission-template](https://github.com/ennovatex-io/ai-challenge-submission-template).

On-Device Multi-Agent System for Behavior-Based Anomaly & Fraud Detection

Overview

This project implements an on-device, privacy-preserving multi-agent system to detect anomalous or fraudulent user behavior using behavioral biometrics and usage patterns. Multiple lightweight agents score signals locally (typing rhythm, touch gestures, app usage, and movement), and a central decision agent fuses their outputs to produce an overall risk verdict and trigger actions.

Key Components

- Documentation: Problem statement, literature, and system design.
- Datasets: Public dataset pointers and synthetic data generator producing CSVs.
- Feature Engineering: Scripts to extract features and preprocess data.
- Models: One-Class SVM, Isolation Forest, Autoencoder, and LSTM agents + integration.
- On-Device Optimization: TensorFlow Lite conversion and quantization examples.
- Evaluation: Metrics, latency measurement, and confusion matrix visualization.
- Demo App: Kivy-based UI stub and mobile integration pointers.

Quickstart

1. Create and activate a virtual environment, then install requirements:

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

2. Generate synthetic datasets (CSV files saved into 2_Datasets/):

```
python 2_Datasets/Synthetic_Data_Generator.py --num_users 100 --num_records 2000
```

3. Extract features and preprocess (outputs train/test NumPy files under 3_Feature_Engineering/.artifacts/):

```
python 3_Feature_Engineering/feature_extraction.py
python 3_Feature_Engineering/preprocessing.py
```

4. Train models and run multi-agent decision demo:

```
python 4_Models/one_class_svm.py
python 4_Models/isolation_forest_model.py
python 4_Models/autoencoder_model.py
python 4_Models/movement_cnn.py   # preferred for pure TFLite BUILTINS
# optional (sequence baseline using Select TF ops):
python 4_Models/movement_lstm.py
python 4_Models/multi_agent_system.py
```

5. Optimize on-device (example with TensorFlow Lite):

```
python 5_OnDevice_Optimization/convert_to_tflite.py
python 5_OnDevice_Optimization/quantization_demo.py
```

6. Evaluate performance and latency:

```
python 6_Evaluation/metrics.py
python 6_Evaluation/evaluate_latency.py
```

7. Run demo UI (Kivy):

```
python 7_Demo_App/mobile_integration/kivy_demo.py
```

Notes

- Public dataset links and references are provided in 1_Documentation/.
- For mobile deployment, see 7_Demo_App/mobile_integration for pointers to Android (TFLite) and Flutter.
- The .docx and .pptx files are placeholders; author content and export from the provided markdown as needed.
- Streamlit deployment:
  - Local: `streamlit run streamlit_app.py`
  - Streamlit Cloud/Heroku: Repo includes `.streamlit/config.toml`, `Procfile`, `packages.txt`.
- Movement model guidance:
  - Default decision agent prefers `movement_cnn.keras` (1D CNN). It converts to TFLite with BUILTINS only (no Flex).
  - The LSTM (`movement_lstm.keras`) converts with Select TF ops and requires the Flex delegate; use only if allowed on target.
- Fusion tuning:
  - Auto-enroll and tune fusion config (grid-search F1) and save:
    - `python 4_Models/enroll_and_tune.py --weight_candidates 0.1,0.2,0.3,0.4 --threshold_candidates 0.4,0.5,0.6,0.7`
  - Run decision agent (auto-loads saved config):
    - `python 4_Models/multi_agent_system.py`
  - Override at runtime and persist:
    - `python 4_Models/multi_agent_system.py --weights 0.2,0.2,0.4,0.2 --threshold 0.6 --save_config`


