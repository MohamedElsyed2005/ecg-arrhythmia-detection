import streamlit as st
import numpy as np
import pandas as pd
import os
import joblib
import sys

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

    prediction = model.predict(input_df)
    predicted_label = label_encoder.inverse_transform(prediction)[0]
    st.success(f"Predicted Beat Type: {predicted_label}")