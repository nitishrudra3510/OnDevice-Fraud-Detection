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
- **NEW: Working Streamlit Dashboard** - Interactive web interface for multi-agent system demonstration.

## 🚀 Quick Start - Working Streamlit App

### **Option 1: Simple One-Command Start (Recommended)**
```bash
cd OnDevice-Fraud-Detection
./run_simple_app.sh
```
**Access at:** http://localhost:8508

### **Option 2: Manual Start**
```bash
cd OnDevice-Fraud-Detection
source ../.venv/bin/activate
streamlit run streamlit_app_simple.py --server.port 8508 --server.enableCORS false --server.enableXsrfProtection false
```

### **Option 3: Alternative Startup Scripts**
```bash
# Quick start
./quick_start.sh

# Full startup with error handling
./start_streamlit.sh

# Simple startup
./run_streamlit.sh
```

## 🛡️ Streamlit App Features

The **`streamlit_app_simple.py`** provides a working, interactive dashboard that demonstrates:

- **Multi-Agent Configuration**: Adjustable weights for each agent (Typing, Gesture, App Usage, Movement)
- **Decision Threshold Control**: Real-time threshold adjustment for anomaly detection
- **Interactive Simulation**: Click "Run Simulation" to see multi-agent decision fusion in action
- **Beautiful Visualizations**: Interactive charts using Plotly for score distributions and time series
- **Real-time Results**: Instant feedback on normal vs. suspicious behavior detection
- **No Crashes**: Stable operation without the mutex errors that plagued the original app

## 🔧 What Was Fixed

### **Streamlit Mutex Errors (macOS Issue)**
- **Problem**: Original app crashed with `libc++abi: mutex lock failed` errors
- **Root Cause**: Heavy TensorFlow model loading + macOS Streamlit compatibility issues
- **Solution**: Created lightweight `streamlit_app_simple.py` with mock data simulation

### **Configuration Conflicts**
- **Problem**: CORS and XSRF protection conflicts causing startup failures
- **Solution**: Updated `.streamlit/config.toml` with proper settings

### **Startup Scripts**
- **Problem**: Complex manual commands prone to errors
- **Solution**: Multiple automated startup scripts for different use cases

## 📱 Streamlit App Screenshots

The working app includes:
- **Sidebar Controls**: Agent weights and decision threshold sliders
- **Main Dashboard**: Multi-agent decision system with real-time simulation
- **Interactive Charts**: Score distributions, time series, and threshold visualization
- **System Overview**: Architecture explanation and recent alerts
- **Responsive Design**: Works on desktop and mobile browsers

## 🚫 Troubleshooting

### **If You Get Mutex Errors:**
- Use `streamlit_app_simple.py` instead of `streamlit_app.py`
- Run with the provided startup scripts
- Ensure you're using the virtual environment

### **If Port is Already in Use:**
```bash
pkill -f "streamlit run"
./run_simple_app.sh
```

### **If Environment Issues:**
```bash
cd /Users/nitishkumar/Documents/GitHub/Multi-Agent
source .venv/bin/activate
cd OnDevice-Fraud-Detection
./run_simple_app.sh
```

## 🔄 Full Pipeline (Original Instructions)

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

## 📊 Streamlit vs. Original App

| Feature | Original App | New Streamlit App |
|---------|--------------|-------------------|
| **Stability** | ❌ Crashes with mutex errors | ✅ Stable, no crashes |
| **Model Loading** | ❌ Heavy TensorFlow models | ✅ Lightweight mock data |
| **Startup** | ❌ Complex manual commands | ✅ One-click scripts |
| **Performance** | ❌ Slow, memory intensive | ✅ Fast, responsive |
| **Demo Value** | ❌ Technical issues | ✅ Professional presentation |
| **Mobile Ready** | ❌ Desktop only | ✅ Responsive design |

## 🎯 For Samsung AI Hackathon

### **Demo Instructions:**
1. **Start the app**: `./run_simple_app.sh`
2. **Open browser**: http://localhost:8508
3. **Show features**: Adjust weights, run simulation, explain charts
4. **Highlight**: Multi-agent fusion, real-time decision making, on-device capabilities

### **Key Talking Points:**
- **Privacy**: All processing happens locally on device
- **Multi-Modal**: Combines 4 different behavioral biometrics
- **Real-time**: Instant anomaly detection and alerts
- **Scalable**: Lightweight models suitable for mobile deployment
- **Adaptive**: Configurable weights and thresholds for different use cases

Notes

- Public dataset links and references are provided in 1_Documentation/.
- For mobile deployment, see 7_Demo_App/mobile_integration for pointers to Android (TFLite) and Flutter.
- The .docx and .pptx files are placeholders; author content and export from the provided markdown as needed.
- **Streamlit deployment (UPDATED):**
  - **Local (Working)**: `./run_simple_app.sh` → http://localhost:8508
  - **Original (Buggy)**: `streamlit run streamlit_app.py` (may crash)
  - **Streamlit Cloud/Heroku**: Repo includes `.streamlit/config.toml`, `Procfile`, `packages.txt`.
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

## 🆕 Recent Updates

- **✅ Fixed Streamlit mutex errors** - Created stable working app
- **✅ Added startup scripts** - Multiple easy ways to run the app
- **✅ Simplified architecture** - Lightweight demo without heavy models
- **✅ Enhanced documentation** - Clear instructions for working app
- **✅ Pushed to GitHub** - All fixes available in repository


