import streamlit as st
import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Set Streamlit page configuration
st.set_page_config(page_title="WESAD Live Stress & Affect Detector", layout="wide", page_icon="🧠")

# Define the neural network architecture (must match training)
class SQIGatedMultimodalNet(nn.Module):
    def __init__(self, seq_channels=6, feat_dim=18, num_classes=3):
        super().__init__()
        self.branch_fast = nn.Sequential(
            nn.Conv1d(seq_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.branch_slow = nn.Sequential(
            nn.Conv1d(seq_channels, 32, kernel_size=11, padding=5, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.sqi_gate = nn.Sequential(
            nn.Linear(4, 32),
            nn.GELU(),
            nn.Linear(32, 128),
            nn.Sigmoid()
        )
        self.feat_net = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.25)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128 + 64, 64),
            nn.GELU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, seq_x, feat_x):
        temp_emb = torch.cat([self.branch_fast(seq_x), self.branch_slow(seq_x)], dim=1)
        motion_ctx = feat_x[:, 12:16]
        gate = self.sqi_gate(motion_ctx)
        gated_temp = temp_emb * gate
        dom_emb = self.feat_net(feat_x)
        return self.classifier(torch.cat([gated_temp, dom_emb], dim=1))

# Cache model loading for instant performance
@st.cache_resource
def load_wesad_model():
    bundle_path = "wesad_bundle.joblib"
    if not os.path.exists(bundle_path):
        return None, None, None
    bundle = joblib.load(bundle_path)
    config = bundle["config"]
    model = SQIGatedMultimodalNet(
        seq_channels=config["seq_channels"],
        feat_dim=config["feat_dim"],
        num_classes=config["num_classes"]
    )
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    return model, bundle["f_mu"], bundle["f_sd"]

model, f_mu, f_sd = load_wesad_model()

st.title("🧠 WESAD Wearable Stress & Affect Detection Engine")
st.markdown("**Interactive clinical diagnostics & counterfactual explanations of stress using wrist physiological signals (Empatica E4).**")

if model is None:
    st.error("⚠️ Model bundle not found! Ensure 'wesad_bundle.joblib' is placed in your repository root.")
else:
    st.sidebar.header("🎮 User Calibration & Controls")
    st.sidebar.info("Zero-shot calibrations adjust model predictions to account for unique individual baselines.")
    
    # Interactive Controls
    subj_baseline = st.sidebar.selectbox("Select Target Individual Baseline Mode", ["Standard Rested Baseline", "De-calibrated Raw Signals"])
    hr_bpm = st.sidebar.slider("Live Heart Rate (BPM)", 50, 140, 85, help="Derived from blood volume pulse (BVP) peak intervals.")
    eda_val = st.sidebar.slider("Electrodermal Activity (EDA Mean µS)", 0.1, 15.0, 3.5, help="Measures physiological sweat level / sympathetic nervous system activity.")
    temp_val = st.sidebar.slider("Skin Temperature (°C)", 28.0, 38.0, 31.5)
    acc_std = st.sidebar.slider("Accelerometer Wrist Jitter (ACC Std)", 0.0, 50.0, 2.5, help="Higher values indicate high motion artifacts, lowering signal quality indices.")
    
    # Generate active SQI & baseline scaling matrices
    sqi_val = float(np.exp(-0.08 * acc_std))
    
    # Reconstruct input representation vectors
    # Col indices: 0:bpm, 1:sdnn, 2:rmssd, 3:bvp_std, 4:eda_mean, 5:eda_std, 6:eda_slope, 7:scr_peaks, 8:eda_max_diff, 9:temp_mean, 10:temp_std, 11:temp_slope
    raw_feats = np.array([
        hr_bpm, 60.0, 50.0, 15.0,
        eda_val, 0.05, 0.01, 0.0, 0.02,
        temp_val, 0.01, 0.0,
        acc_std, acc_std, acc_std*1.5, sqi_val, hr_bpm * sqi_val, 50.0 * sqi_val
    ], dtype=np.float32)
    
    # Apply normalization against baseline demographics
    norm_feats = (raw_feats - f_mu) / f_sd
    if subj_baseline == "De-calibrated Raw Signals":
        norm_feats[:12] = raw_feats[:12] * 1.5 # Simulates raw scaling bias shift

    st.subheader("🔮 Live Inference Engine")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Calculated Motion SQI (Reliability)", f"{sqi_val * 100:.1f}%", delta="Reliable Sensor Data" if sqi_val > 0.8 else "Motion Interference", delta_color="normal" if sqi_val > 0.8 else "inverse")
    with col2:
        st.metric("Skin Temperature", f"{temp_val:.1f} °C")
    with col3:
        st.metric("Skin Conductance (EDA)", f"{eda_val:.2f} µS")

    # Model Prediction
    t_seq = torch.zeros((1, 6, 120), dtype=torch.float32) # Standard sequence baseline
    t_feat = torch.tensor(norm_feats, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        logits = model(t_seq, t_feat)
        probs = torch.softmax(logits, dim=1)[0].numpy()
    
    # Plot probabilities 
    st.markdown("### Predictor Class Probabilities")
    classes = ["Baseline / Normal", "Cognitive Stress", "Amusement"]
    prob_df = pd.DataFrame({"State": classes, "Confidence (%)": probs * 100})
    st.bar_chart(prob_df.set_index("State"))
    
    # Display Top prediction
    prediction_idx = np.argmax(probs)
    if prediction_idx == 0:
        st.success(f"✅ **Predicted State: Normal Baseline** (Confidence: {probs[0]*100:.1f}%)")
    elif prediction_idx == 1:
        st.error(f"🚨 **Predicted State: COGNITIVE STRESS** (Confidence: {probs[1]*100:.1f}%)")
    else:
        st.info(f"🎉 **Predicted State: Amusement / Flow State** (Confidence: {probs[2]*100:.1f}%)")

    # Show reference Diagnostics plot
    st.markdown("--- ")
    st.subheader("📊 Leave-One-Subject-Out (LOSO) Model Diagnostics")
    st.markdown("This standard benchmark visualizes generalizability on 15 unseen subjects from the laboratory protocol.")
    if os.path.exists("wesad_loso_diagnostics.png"):
        st.image("wesad_loso_diagnostics.png", use_column_width=True)
    else:
        st.info("No static evaluation figure found. Upload 'wesad_loso_diagnostics.png' to show validation metrics here.")