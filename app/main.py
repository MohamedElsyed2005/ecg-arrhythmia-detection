import streamlit as st
import numpy as np
import pandas as pd
import os
import joblib
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Add src/ to the Python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "src"))
sys.path.append(SRC_DIR)

from preprocess_funcs import preprocessing

# Load model
model_path = os.path.join(BASE_DIR, '..', 'models', 'xgboost_pipeline.pkl')
model = joblib.load(model_path)

# Load encoder 
encoder_path = os.path.join(BASE_DIR, '..', 'models', 'label_encoder.pkl')
label_encoder = joblib.load(encoder_path)

st.title("ECG Beat Type Classifier")
st.markdown("Enter the following values to classify the type of heartbeat")

feature_names = [
    "0_pre-RR", "0_post-RR", "0_pPeak", "0_tPeak", "0_rPeak", "0_sPeak", "0_qPeak",
    "0_qrs_interval", "0_pq_interval", "0_qt_interval", "0_st_interval",
    "0_qrs_morph0", "0_qrs_morph1", "0_qrs_morph2", "0_qrs_morph3", "0_qrs_morph4",
    "1_pre-RR", "1_post-RR", "1_pPeak", "1_tPeak", "1_rPeak", "1_sPeak", "1_qPeak",
    "1_qrs_interval", "1_pq_interval", "1_qt_interval", "1_st_interval",
    "1_qrs_morph0", "1_qrs_morph1", "1_qrs_morph2", "1_qrs_morph3", "1_qrs_morph4"
]

user_input = []
for name in feature_names:
    val = st.number_input(f"{name}", format="%.5f")
    user_input.append(val)

if st.button("Predict Beat Type"):
    input_array = np.array(user_input).reshape(1, -1)
    input_df = pd.DataFrame(input_array, columns=feature_names)

    # Prediction
    prediction = model.predict(input_df)
    predicted_label = label_encoder.inverse_transform(prediction)[0]
    st.success(f"Predicted Beat Type: {predicted_label}")

    # Probabilities
    proba = model.predict_proba(input_df)[0]
    class_labels = label_encoder.inverse_transform(np.arange(len(proba)))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    explode = [0.07] * len(proba)

    # Pie Chart with custom label positioning
    fig, ax = plt.subplots(figsize=(9, 6))
    wedges, texts = ax.pie(
        proba,
        labels=None,
        colors=colors,
        startangle=90,
        explode=explode,
        radius=1.0
    )

    for i, (w, p) in enumerate(zip(wedges, proba)):
        ang = (w.theta2 + w.theta1) / 2.
        angle_offset = 0
        if i == 0:  # F
            angle_offset = -0.15
        elif i == 4:  # VEB
            angle_offset = 0.15

        x = np.cos(np.deg2rad(ang)) * 1.35
        y = np.sin(np.deg2rad(ang)) * 1.35 + angle_offset

        ax.text(
            x, y, f"{p*100:.1f}%",
            ha='center', va='center',
            fontsize=12, weight='bold',
            bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.4')
        )

    ax.legend(
        wedges,
        class_labels,
        title="Classes",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=12
    )

    fig.suptitle("Prediction Probabilities", fontsize=16, y=1.05)
    ax.set_title("")
    ax.axis('equal')
    plt.tight_layout()
    st.subheader("Prediction Probabilities")
    st.pyplot(fig)

    # Feature Importance
    st.subheader("Feature Importance")

    try:
        importances = model.named_steps['classifier'].feature_importances_
        n_importances = len(importances)
        used_features = feature_names[:n_importances]

        importance_df = pd.DataFrame({
            'Feature': used_features,
            'Importance': importances
        }).sort_values(by="Importance", ascending=False).head(20)

        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.barplot(data=importance_df, x='Importance', y='Feature', palette="viridis", ax=ax2)
        ax2.set_title("Top 20 Important Features")
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"Could not compute feature importances: {e}")