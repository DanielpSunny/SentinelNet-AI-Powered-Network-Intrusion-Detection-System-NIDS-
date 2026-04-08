import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

st.set_page_config(page_title="Network Anomaly Detector", page_icon="🔍")
st.title("🔍 Ensemble Network Anomaly Detection")
st.write("Upload a CSV file in KDD format to detect anomalies.")

@st.cache_resource
def load_model():
    with open("model_bundle.pkl", "rb") as f:
        return pickle.load(f)

bundle = load_model()

uploaded_file = st.file_uploader("Upload KDD-format CSV", type=["csv"])

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
    df_raw.columns = [c.strip("'") for c in df_raw.columns]
    for c in df_raw.select_dtypes(include=["object"]).columns:
        df_raw[c] = df_raw[c].str.strip("'")

    df = df_raw.copy()
    if "class" in df.columns:
        y_true = df["class"].map({"normal": 0, "anomaly": 1}).values
        df.drop(columns=["class"], inplace=True)
    else:
        y_true = None

    df.drop(columns=["id"], errors="ignore", inplace=True)

    # Feature engineering (same as training)
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))

    log_feats = ["src_bytes","dst_bytes","duration","num_compromised",
                 "num_root","count","srv_count","dst_host_count","dst_host_srv_count"]
    for f in [x for x in log_feats if x in df.columns]:
        df[f] = np.log1p(df[f].clip(lower=0))

    if {"src_bytes","dst_bytes"}.issubset(df.columns):
        df["bytes_ratio"] = df["src_bytes"] / (df["dst_bytes"] + 1e-6)
        df["total_bytes"] = df["src_bytes"] + df["dst_bytes"]
    if {"serror_rate","rerror_rate"}.issubset(df.columns):
        df["total_error"] = df["serror_rate"] + df["rerror_rate"]

    # Keep only features seen during training
    feature_cols = bundle["feature_cols"]
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_cols]

    X_scaled = bundle["scaler"].transform(df.values.astype(np.float32))
    X_pca    = bundle["pca"].transform(X_scaled)
    X_pca_db = (X_pca - X_pca.mean(0)) / (X_pca.std(0) + 1e-8)

    # Isolation Forest score
    scores_if = -bundle["iso"].decision_function(X_pca)

    # DBSCAN score (distance to nearest normal)
    dist, _ = bundle["nn_db"].kneighbors(X_pca_db)
    scores_db = dist.mean(axis=1)

    # One-Class SVM score
    scores_svm = -bundle["svm_model"].decision_function(X_pca)

    # Normalise
    mm = MinMaxScaler()
    scores_if_n  = mm.fit_transform(scores_if .reshape(-1,1)).ravel()
    scores_db_n  = mm.fit_transform(scores_db .reshape(-1,1)).ravel()
    scores_svm_n = mm.fit_transform(scores_svm.reshape(-1,1)).ravel()

    # Weighted ensemble score
    w_if, w_db, w_svm, w_total = bundle["w_if"], bundle["w_db"], bundle["w_svm"], bundle["w_total"]
    scores_weighted = (w_if*scores_if_n + w_db*scores_db_n + w_svm*scores_svm_n) / w_total
    y_pred = (scores_weighted >= bundle["best_thresh_wt"]).astype(int)

    df_raw["anomaly_score"] = np.round(scores_weighted, 4)
    df_raw["prediction"]    = np.where(y_pred == 1, "🚨 Anomaly", "✅ Normal")

    st.success(f"Scanned {len(df_raw):,} records")
    col1, col2 = st.columns(2)
    col1.metric("🚨 Anomalies", int(y_pred.sum()))
    col2.metric("✅ Normal",    int((y_pred == 0).sum()))

    st.dataframe(df_raw[["prediction","anomaly_score"]].head(100))

    csv_out = df_raw.to_csv(index=False).encode()
    st.download_button("⬇️ Download Results", csv_out, "results.csv", "text/csv")