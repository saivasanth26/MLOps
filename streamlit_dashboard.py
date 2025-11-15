# streamlit_inspector.py
"""
Streamlit Inspector for model monitoring (local-first).
Reads JSON-lines from ./monitoring_logs/predictions.log (created by your Flask app).
Features:
 - Summary metrics (count, avg entropy, avg confidence, OOD rate)
 - Class distribution bar chart
 - Entropy & confidence histograms
 - Top-K high-entropy samples for manual labeling (radio buttons)
 - Save labels locally to labeled_samples.csv and export download
 - Toggle refresh / choose how many recent records to load

Run:
    pip install streamlit pandas numpy matplotlib
    streamlit run streamlit_inspector.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
from io import BytesIO

# ---------------------------
# Config & paths
# ---------------------------
LOCAL_LOG_FILE = os.environ.get("LOCAL_LOG_FILE", "./monitoring_logs/predictions.log")
LABELED_OUTPUT = os.environ.get("LABELED_OUTPUT", "./monitoring_logs/labeled_samples.csv")

st.set_page_config(page_title="Model Monitoring Inspector", layout="wide")

# ---------------------------
# Helpers
# ---------------------------
@st.cache_data(ttl=10)
def load_local_jsonlines(path: str, max_records: int = 5000) -> pd.DataFrame:
    """Load up to max_records most recent JSON-lines from path (file contains one JSON per line)."""
    if not os.path.exists(path):
        return pd.DataFrame()
    # Read lines (we expect file size to be reasonably small for local dev)
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    # Keep last max_records lines
    lines = lines[-max_records:]
    rows = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            rec = json.loads(ln)
            rows.append(rec)
        except Exception:
            # skip malformed lines
            continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # normalize common columns
    if "timestamp" in df.columns:
        # try to parse timestamp strings into pandas.Timestamp
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    # fill missing monitoring fields with NaNs
    for col in ["entropy", "max_confidence", "is_ood", "predicted", "probability_vector", "processed_text", "input_text", "model", "video_id"]:
        if col not in df.columns:
            df[col] = np.nan
    return df

def compute_summary(df: pd.DataFrame):
    n = len(df)
    avg_entropy = float(df["entropy"].dropna().mean()) if "entropy" in df.columns and df["entropy"].notna().any() else None
    avg_conf = float(df["max_confidence"].dropna().mean()) if "max_confidence" in df.columns and df["max_confidence"].notna().any() else None
    ood_rate = float(df["is_ood"].sum() / n) if ("is_ood" in df.columns and n>0) else 0.0
    return {"count": n, "avg_entropy": avg_entropy, "avg_confidence": avg_conf, "ood_rate": ood_rate}

def prepare_class_dist(df: pd.DataFrame):
    if "predicted" not in df.columns:
        return pd.Series(dtype=int)
    counts = df["predicted"].value_counts().sort_index()
    return counts

def save_labeled_samples(df_labels: pd.DataFrame, out_path: str):
    # Append to CSV if exists, otherwise write with header
    if df_labels.empty:
        return
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if os.path.exists(out_path):
        df_labels.to_csv(out_path, mode="a", header=False, index=False)
    else:
        df_labels.to_csv(out_path, index=False)

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

# ---------------------------
# UI
# ---------------------------
st.title("🕵️ Model Monitoring — Inspector (Local)")
st.markdown(
    "This inspector reads local monitoring logs created by your Flask model server "
    "(`./monitoring_logs/predictions.log`) and helps you spot suspicious samples and label them."
)

# Controls
col1, col2, col3 = st.columns([1,1,1])
with col1:
    max_records = st.number_input("Load up to N recent records", min_value=50, max_value=20000, value=2000, step=50)
with col2:
    top_k = st.number_input("Top-K high-entropy samples to inspect", min_value=5, max_value=500, value=50, step=5)
with col3:
    refresh = st.button("Refresh data")

# Load data
df = load_local_jsonlines(LOCAL_LOG_FILE, max_records=max_records)
if df.empty:
    st.warning(f"No monitoring logs found at `{LOCAL_LOG_FILE}`. Run your Flask app and generate predictions first.")
    st.stop()

# Summary metrics
summary = compute_summary(df)
st.subheader("Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total records loaded", summary["count"])
col2.metric("Average entropy", f"{summary['avg_entropy']:.3f}" if summary["avg_entropy"] is not None else "N/A")
col3.metric("Average confidence", f"{summary['avg_confidence']:.3f}" if summary["avg_confidence"] is not None else "N/A")
col4.metric("OOD rate (fraction)", f"{summary['ood_rate']:.3f}")

# Class distribution and histograms
st.subheader("Distributions")
dist_col1, dist_col2 = st.columns([1,2])
with dist_col1:
    st.write("Prediction class distribution")
    class_dist = prepare_class_dist(df)
    if class_dist.empty:
        st.info("No 'predicted' values available in logs.")
    else:
        st.bar_chart(class_dist)

with dist_col2:
    fig, axes = plt.subplots(1,2, figsize=(10,3))
    if "entropy" in df.columns and df["entropy"].notna().any():
        axes[0].hist(df["entropy"].dropna(), bins=30)
        axes[0].set_title("Entropy distribution")
        axes[0].set_xlabel("Entropy")
    else:
        axes[0].text(0.5,0.5,"No entropy data", ha="center")
    if "max_confidence" in df.columns and df["max_confidence"].notna().any():
        axes[1].hist(df["max_confidence"].dropna(), bins=30)
        axes[1].set_title("Max confidence distribution")
        axes[1].set_xlabel("Max confidence")
    else:
        axes[1].text(0.5,0.5,"No confidence data", ha="center")
    st.pyplot(fig)

# Time selector and filtering (fixed for Timestamp -> datetime slider)
st.subheader("Filter & sample")
time_col1, time_col2 = st.columns([2,1])
with time_col1:
    # Only allow slider when timestamp column exists and has at least one non-null value
    if "timestamp" in df.columns and df["timestamp"].notna().any():
        min_ts_pd = df["timestamp"].min()
        max_ts_pd = df["timestamp"].max()
        # Convert to native python datetime objects for Streamlit slider
        try:
            min_ts = min_ts_pd.to_pydatetime()
            max_ts = max_ts_pd.to_pydatetime()
        except Exception:
            # fallback: if values are already datetime or conversion fails, check types
            import datetime as _dt
            if isinstance(min_ts_pd, _dt.datetime) and isinstance(max_ts_pd, _dt.datetime):
                min_ts = min_ts_pd
                max_ts = max_ts_pd
            else:
                st.info("Timestamps present but not in a recognized format; using all records.")
                min_ts = None
                max_ts = None

        if min_ts is not None and max_ts is not None:
            # st.slider handles datetime objects; provide readable format
            sel_range = st.slider(
                "Select time range",
                min_value=min_ts,
                max_value=max_ts,
                value=(min_ts, max_ts),
                format="YYYY-MM-DD HH:mm:ss"
            )
            # sel_range is a tuple of python datetimes; convert back to pandas timestamps for filtering
            sel_start = pd.to_datetime(sel_range[0])
            sel_end = pd.to_datetime(sel_range[1])
            df = df[(df["timestamp"] >= sel_start) & (df["timestamp"] <= sel_end)]
    else:
        st.info("No timestamp data present; using all records.")

with time_col2:
    ood_only = st.checkbox("Show OOD-only samples", value=False)
    if ood_only and "is_ood" in df.columns:
        df = df[df["is_ood"]==True]

st.markdown("---")

# Top high-entropy samples for inspection
st.subheader(f"Top {top_k} high-entropy samples (for manual inspection & labeling)")
if "entropy" not in df.columns or not df["entropy"].notna().any():
    st.info("No entropy values in logs. Your Flask app must log 'entropy' to enable sample selection.")
else:
    df_sorted = df.sort_values(by="entropy", ascending=False).reset_index(drop=True)
    top_df = df_sorted.head(top_k).copy()
    # Prepare labeling state: keep label selections in session_state
    if "labels" not in st.session_state:
        st.session_state["labels"] = {}
    if "notes" not in st.session_state:
        st.session_state["notes"] = {}

    # Display each sample with radio buttons for labels
    for idx, row in top_df.iterrows():
        key = f"sample_{idx}"
        with st.expander(f"[{idx+1}] Pred: {row.get('predicted')}  Entropy: {row.get('entropy'):.3f}  Conf: {row.get('max_confidence') if pd.notna(row.get('max_confidence')) else 'N/A'}"):
            st.markdown(f"**Raw text:** {row.get('input_text')}")
            st.markdown(f"**Processed text:** {row.get('processed_text')}")
            st.markdown(f"**Prob vector:** {row.get('probability_vector')}")
            # radio for label
            current = st.session_state["labels"].get(key, "unlabeled")
            choices = ["unlabeled", "1 (Positive)", "0 (Neutral)", "-1 (Negative)"]
            index_choice = choices.index(current) if current in choices else 0
            choice = st.radio("Assign true label (use this for evaluation):", options=choices, index=index_choice, key=f"lab_{key}")
            st.session_state["labels"][key] = choice
            note = st.text_input("Notes (optional)", value=st.session_state["notes"].get(key, ""), key=f"note_{key}")
            st.session_state["notes"][key] = note
            # show timestamp and model
            if "timestamp" in row and pd.notna(row["timestamp"]):
                st.caption(f"Logged at: {row['timestamp']}")
            st.caption(f"Model: {row.get('model')}  VideoID: {row.get('video_id')}")

    # Save labeled selections button
    if st.button("Save selected labels to CSV"):
        # Build DataFrame from selections
        rows_to_save = []
        for idx, row in top_df.iterrows():
            key = f"sample_{idx}"
            sel = st.session_state["labels"].get(key, "unlabeled")
            if sel != "unlabeled":
                # map label string to numeric label
                label_map = {"1 (Positive)": 1, "0 (Neutral)": 0, "-1 (Negative)": -1}
                rows_to_save.append({
                    "timestamp": row.get("timestamp"),
                    "input_text": row.get("input_text"),
                    "processed_text": row.get("processed_text"),
                    "predicted": row.get("predicted"),
                    "true_label": label_map.get(sel, sel),
                    "entropy": row.get("entropy"),
                    "max_confidence": row.get("max_confidence"),
                    "prob_vector": json.dumps(row.get("probability_vector")),
                    "model": row.get("model"),
                    "video_id": row.get("video_id"),
                    "notes": st.session_state["notes"].get(key, "")
                })
        if rows_to_save:
            df_labels = pd.DataFrame(rows_to_save)
            save_labeled_samples(df_labels, LABELED_OUTPUT)
            st.success(f"Saved {len(df_labels)} labeled rows to `{LABELED_OUTPUT}`")
        else:
            st.info("No labeled rows selected to save.")

    # Allow downloading the labeled rows as CSV (in-memory)
    if st.button("Export current selected labels as CSV (download)"):
        rows_to_export = []
        for idx, row in top_df.iterrows():
            key = f"sample_{idx}"
            sel = st.session_state["labels"].get(key, "unlabeled")
            if sel != "unlabeled":
                label_map = {"1 (Positive)": 1, "0 (Neutral)": 0, "-1 (Negative)": -1}
                rows_to_export.append({
                    "timestamp": row.get("timestamp"),
                    "input_text": row.get("input_text"),
                    "processed_text": row.get("processed_text"),
                    "predicted": row.get("predicted"),
                    "true_label": label_map.get(sel, sel),
                    "entropy": row.get("entropy"),
                    "max_confidence": row.get("max_confidence"),
                    "prob_vector": json.dumps(row.get("probability_vector")),
                    "model": row.get("model"),
                    "video_id": row.get("video_id"),
                    "notes": st.session_state["notes"].get(key, "")
                })
        if rows_to_export:
            df_export = pd.DataFrame(rows_to_export)
            csv_bytes = df_to_csv_bytes(df_export)
            st.download_button("Download labeled CSV", data=csv_bytes, file_name="labeled_samples_export.csv", mime="text/csv")
        else:
            st.info("No labeled rows selected to export.")

st.markdown("---")
# Show raw sample of logs and option to download whole slice
st.subheader("Raw log sample (most recent records)")
show_n = st.number_input("Show first N rows of loaded dataframe", min_value=5, max_value=1000, value=50, step=5)
if not df.empty:
    st.dataframe(df.head(show_n))
    # Download entire loaded dataframe
    csv_all = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download loaded records CSV", data=csv_all, file_name="monitoring_loaded.csv", mime="text/csv")

# Show path to labeled file
st.info(f"Labeled samples saved at: `{LABELED_OUTPUT}` (appends if file exists)")

st.caption(f"Last refreshed: {datetime.utcnow().isoformat()}  | Local log: `{LOCAL_LOG_FILE}`")
