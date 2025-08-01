# src/preprocess_funcs.py

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd

# ================== Transformation Functions ==================

def rr_ratio_func(x):
    if isinstance(x, pd.DataFrame):
        x = x.to_numpy()
    denom = np.where(x[:, [1]] == 0, np.nan, x[:, [1]])
    return x[:, [0]] / denom

def qt_corrected_func(x):
    if isinstance(x, pd.DataFrame):
        x = x.to_numpy()
    denom = np.sqrt(x[:, [1]])
    denom[denom == 0] = 1e-6 
    return np.divide(x[:, [0]], denom, out=np.zeros_like(x[:, [0]]), where=denom != 0)

def beat_consistency_func(x):
    if isinstance(x, pd.DataFrame):
        x = x.to_numpy()
    return abs(x[:, [0]] - x[:, [1]])

def rr_std_func(x):
    if isinstance(x, pd.DataFrame):
        x = x.to_numpy()
    return np.std(x, axis=1, keepdims=True)

def qt_diff_func(x):
    if isinstance(x, pd.DataFrame):
        x = x.to_numpy()
    return x[:, [0]] - x[:, [1]]

# ================== Safe Log Function ==================

def safe_log1p(x):
    x = np.where(x <= -1, 0, x)
    return np.log1p(x)

# ================== Feature Name Out Functions ==================

def get_rr_ratio_name(_): return ["rr_ratio"]
def get_qt_corrected_name(_): return ["qt_corrected"]
def get_beat_consistency_name(_): return ["beat_consistency"]
def get_rr_std_name(_): return ["rr_std"]
def get_qt_diff_name(_): return ["qt_diff"]
def get_log_trans_names(transformer, input_features): return input_features

# ================== Pipelines for Derived Features ==================

def rr_ratio_pipeline():
    return make_pipeline(
        FunctionTransformer(rr_ratio_func, feature_names_out=get_rr_ratio_name),
        StandardScaler()
    )

def qt_corrected_pipeline():
    return make_pipeline(
        FunctionTransformer(qt_corrected_func, feature_names_out=get_qt_corrected_name),
        FunctionTransformer(safe_log1p, inverse_func=np.expm1, check_inverse=False),
        StandardScaler()
    )

def beat_consistency_pipeline():
    return make_pipeline(
        FunctionTransformer(beat_consistency_func, feature_names_out=get_beat_consistency_name),
        FunctionTransformer(safe_log1p, inverse_func=np.expm1, check_inverse=False),
        StandardScaler()
    )

def rr_std_pipeline():
    return make_pipeline(
        FunctionTransformer(rr_std_func, feature_names_out=get_rr_std_name),
        StandardScaler()
    )

def qt_diff_pipeline():
    return make_pipeline(
        FunctionTransformer(qt_diff_func, feature_names_out=get_qt_diff_name),
        FunctionTransformer(safe_log1p, inverse_func=np.expm1, check_inverse=False),
        StandardScaler()
    )

# ================== Log + Scaling Pipeline ==================

log_pipeline = make_pipeline(
    FunctionTransformer(
        safe_log1p,
        inverse_func=np.expm1,
        feature_names_out=get_log_trans_names,
        check_inverse=False
    ),
    StandardScaler()
)

# ================== Default Scaling Pipeline ==================

default_pipeline = make_pipeline(StandardScaler())

# ================== Full Preprocessing Pipeline ==================

# Define the columns that need log scaling
outlier_columns_to_handle = [
    '0_pPeak', '0_tPeak', '0_rPeak', '0_sPeak', '0_qPeak',
    '1_pPeak', '1_tPeak', '1_rPeak', '1_sPeak', '1_qPeak'
]

preprocessing = ColumnTransformer([
    ("rr_ratio", rr_ratio_pipeline(), ["0_post-RR", "0_pre-RR"]),
    ("qt_corrected", qt_corrected_pipeline(), ["0_qt_interval", "0_post-RR"]),
    ("beat_consistency", beat_consistency_pipeline(), ["0_qrs_interval", "1_qrs_interval"]),
    ("rr_std", rr_std_pipeline(), ["0_pre-RR", "0_post-RR"]),
    ("qt_diff", qt_diff_pipeline(), ["0_qt_interval", "1_qt_interval"]),
    ("log", log_pipeline, outlier_columns_to_handle)
], remainder=default_pipeline, force_int_remainder_cols=False)