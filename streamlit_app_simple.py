import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Multi-Agent Fraud Detection - Demo",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ On-Device Multi-Agent Fraud Detection System")
st.markdown("**Behavior-based anomaly detection using multiple AI agents**")

# Sidebar for configuration
st.sidebar.header("⚙️ Configuration")
st.sidebar.markdown("---")

# Agent weights
st.sidebar.subheader("Agent Weights")
w_typing = st.sidebar.slider("Typing Agent", 0.0, 1.0, 0.25, 0.05)
w_gesture = st.sidebar.slider("Gesture Agent", 0.0, 1.0, 0.25, 0.05)
w_app = st.sidebar.slider("App Usage Agent", 0.0, 1.0, 0.25, 0.05)
w_movement = st.sidebar.slider("Movement Agent", 0.0, 1.0, 0.25, 0.05)

# Decision threshold
threshold = st.sidebar.slider("Decision Threshold", 0.0, 1.0, 0.6, 0.01)

# Normalize weights
weights = np.array([w_typing, w_gesture, w_app, w_movement])
weights = weights / weights.sum()

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Normalized Weights:**")
st.sidebar.markdown(f"Typing: {weights[0]:.2f}")
st.sidebar.markdown(f"Gesture: {weights[1]:.2f}")
st.sidebar.markdown(f"App Usage: {weights[2]:.2f}")
st.sidebar.markdown(f"Movement: {weights[3]:.2f}")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Multi-Agent Decision System")
    
    if st.button("🚀 Run Simulation", type="primary"):
        # Generate mock data for demonstration
        np.random.seed(42)
        n_samples = 100
        
        # Simulate agent scores (0-1, where 1 is normal)
        typing_scores = np.random.beta(2, 1, n_samples)
        gesture_scores = np.random.beta(1.5, 1.5, n_samples)
        app_scores = np.random.beta(1, 2, n_samples)
        movement_scores = np.random.beta(2.5, 1, n_samples)
        
        # Fuse scores using weighted average
        fused_scores = (weights[0] * typing_scores + 
                       weights[1] * gesture_scores + 
                       weights[2] * app_scores + 
                       weights[3] * movement_scores)
        
        # Make decisions
        decisions = (fused_scores >= threshold).astype(int)
        
        # Display results
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.metric("Normal Behavior", f"{(decisions==1).sum()}", f"{(decisions==1).mean()*100:.1f}%")
        
        with col_b:
            st.metric("Suspicious Activity", f"{(decisions==0).sum()}", f"{(decisions==0).mean()*100:.1f}%")
        
        # Create interactive charts
        st.subheader("📈 Agent Performance Analysis")
        
        # Time series of scores
        df_scores = pd.DataFrame({
            'Sample': range(n_samples),
            'Typing': typing_scores,
            'Gesture': gesture_scores,
            'App Usage': app_scores,
            'Movement': movement_scores,
            'Fused': fused_scores
        })
        
        fig_scores = px.line(df_scores, x='Sample', y=['Typing', 'Gesture', 'App Usage', 'Movement', 'Fused'],
                            title="Agent Scores Over Time",
                            labels={'value': 'Score', 'variable': 'Agent'})
        fig_scores.update_layout(height=400)
        st.plotly_chart(fig_scores, use_container_width=True)
        
        # Distribution of scores
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(x=typing_scores, name='Typing', opacity=0.7))
        fig_dist.add_trace(go.Histogram(x=gesture_scores, name='Gesture', opacity=0.7))
        fig_dist.add_trace(go.Histogram(x=app_scores, name='App Usage', opacity=0.7))
        fig_dist.add_trace(go.Histogram(x=movement_scores, name='Movement', opacity=0.7))
        fig_dist.update_layout(title="Score Distributions", height=400, barmode='overlay')
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Threshold visualization
        fig_thresh = px.scatter(x=fused_scores, y=np.zeros_like(fused_scores),
                               color=decisions,
                               title=f"Decision Threshold: {threshold:.2f}",
                               labels={'x': 'Fused Score', 'y': '', 'color': 'Decision'})
        fig_thresh.add_vline(x=threshold, line_dash="dash", line_color="red", 
                            annotation_text=f"Threshold: {threshold:.2f}")
        fig_thresh.update_layout(height=300)
        st.plotly_chart(fig_thresh, use_container_width=True)

with col2:
    st.subheader("🔍 System Overview")
    
    st.info("""
    **Multi-Agent Architecture:**
    
    🎯 **Typing Agent**: Keystroke dynamics analysis
    👆 **Gesture Agent**: Touch/gesture pattern recognition  
    📱 **App Usage Agent**: Application behavior monitoring
    🚶 **Movement Agent**: Location/mobility pattern analysis
    
    **Decision Fusion:**
    - Weighted combination of agent scores
    - Threshold-based classification
    - Real-time anomaly detection
    """)
    
    st.subheader("📋 Recent Alerts")
    
    # Mock alerts
    alerts = [
        {"time": "10:45 AM", "type": "Suspicious", "agent": "Movement", "confidence": "87%"},
        {"time": "10:42 AM", "type": "Normal", "agent": "Typing", "confidence": "92%"},
        {"time": "10:38 AM", "type": "Suspicious", "agent": "App Usage", "confidence": "78%"},
        {"time": "10:35 AM", "type": "Normal", "agent": "Gesture", "confidence": "89%"}
    ]
    
    for alert in alerts:
        if alert["type"] == "Suspicious":
            st.error(f"🚨 {alert['time']} - {alert['agent']} ({alert['confidence']})")
        else:
            st.success(f"✅ {alert['time']} - {alert['agent']} ({alert['confidence']})")

# Footer
st.markdown("---")
st.caption("On-Device Multi-Agent System for Behavior-Based Anomaly & Fraud Detection")
st.caption("Built with Streamlit • TensorFlow Lite • Multi-Agent AI")
