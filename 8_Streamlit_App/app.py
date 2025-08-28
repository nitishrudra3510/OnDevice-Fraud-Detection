import sys
from pathlib import Path
import json
import numpy as np
import streamlit as st
import tensorflow as tf

# Local imports
CURRENT_DIR = Path(__file__).resolve().parent
ROOT = CURRENT_DIR.parents[0]
MODELS_DIR = ROOT / "4_Models/.models"
ART = ROOT / "3_Feature_Engineering/.artifacts"

if str(ROOT / "4_Models") not in sys.path:
    sys.path.insert(0, str(ROOT / "4_Models"))
from model_utils import load_sklearn_model


st.set_page_config(page_title="On-Device Multi-Agent - Streamlit", layout="wide")
st.title("On-Device Multi-Agent System - Dashboard")
st.caption("Behavior-based anomaly & fraud detection - local demo")


@st.cache_resource
def load_agents():
    agents = {}
    # Sklearn agents
    agents["typing_ocsvm"] = load_sklearn_model("typing_agent_ocsvm")
    agents["gesture_isoforest"] = load_sklearn_model("gesture_agent_isoforest")
    # Keras agents
    agents["app_autoenc"] = tf.keras.models.load_model(MODELS_DIR / "app_usage_autoencoder.keras")
    # Prefer CNN movement, fallback LSTM
    movement_path = MODELS_DIR / "movement_cnn.keras"
    if not movement_path.exists():
        movement_path = MODELS_DIR / "movement_lstm.keras"
    agents["movement"] = tf.keras.models.load_model(movement_path)
    return agents


@st.cache_data
def load_features():
    ks_X = np.load(ART / "keystroke_features_X_test.npy")
    ks_y = np.load(ART / "keystroke_features_y_test.npy")
    tg_X = np.load(ART / "touch_features_X_test.npy")
    tg_y = np.load(ART / "touch_features_y_test.npy")
    au_X = np.load(ART / "app_usage_features_X_test.npy")
    au_y = np.load(ART / "app_usage_features_y_test.npy")
    mv_X = np.load(ART / "movement_features_X_test.npy")
    mv_y = np.load(ART / "movement_features_y_test.npy")
    return ks_X, ks_y, tg_X, tg_y, au_X, au_y, mv_X, mv_y


def sigmoid_normalize(score: np.ndarray):
    return 1.0 / (1.0 + np.exp(-score))


def compute_scores(agents, ks_X, tg_X, au_X, mv_X, window: int = 10):
    s_typing = sigmoid_normalize(agents["typing_ocsvm"].decision_function(ks_X))
    s_gesture = sigmoid_normalize(agents["gesture_isoforest"].decision_function(tg_X))
    recon = agents["app_autoenc"].predict(au_X, verbose=0)
    mse = np.mean((au_X - recon) ** 2, axis=1)
    mse_norm = (mse - mse.min()) / (np.ptp(mse) + 1e-6)
    s_app = 1 - mse_norm
    n = (len(mv_X) // window) * window
    mv_seq = mv_X[:n].reshape(-1, window, mv_X.shape[1])
    s_movement = agents["movement"].predict(mv_seq, verbose=0).ravel()
    # Trim to same length
    min_len = min(len(s_typing), len(s_gesture), len(s_app), len(s_movement))
    return s_typing[:min_len], s_gesture[:min_len], s_app[:min_len], s_movement[:min_len]


left, right = st.columns([3, 2])
with right:
    st.subheader("Fusion Settings")
    w_typ = st.slider("Typing weight", 0.0, 1.0, 0.25, 0.05)
    w_gst = st.slider("Gesture weight", 0.0, 1.0, 0.25, 0.05)
    w_app = st.slider("App usage weight", 0.0, 1.0, 0.25, 0.05)
    w_mov = st.slider("Movement weight", 0.0, 1.0, 0.25, 0.05)
    thr = st.slider("Decision threshold", 0.0, 1.0, 0.6, 0.01)
    num = st.number_input("Samples to evaluate", min_value=50, max_value=1000, value=200, step=50)

with left:
    st.subheader("Run Inference")
    if st.button("Compute Decisions"):
        with st.spinner("Loading models & features..."):
            agents = load_agents()
            ks_X, ks_y, tg_X, tg_y, au_X, au_y, mv_X, mv_y = load_features()
        s_typ, s_gst, s_app, s_mov = compute_scores(
            agents, ks_X[:num], tg_X[:num], au_X[:num], mv_X[:num]
        )
        weights = np.array([w_typ, w_gst, w_app, w_mov])
        weights = weights / (weights.sum() + 1e-9)
        fused = weights[0] * s_typ + weights[1] * s_gst + weights[2] * s_app + weights[3] * s_mov
        decisions = (fused >= thr).astype(int)

        st.success(f"Normal: {(decisions==1).mean()*100:.1f}% | Suspicious: {(decisions==0).mean()*100:.1f}%")
        st.line_chart({
            "typing": s_typ,
            "gesture": s_gst,
            "app": s_app,
            "movement": s_mov,
            "fused": fused,
        })
        st.caption("Scores closer to 1 indicate more normal behavior.")

st.info("Run with: streamlit run 8_Streamlit_App/app.py")


