# streamlit_inspector.py
"""
Streamlit Inspector (S3-only)
Reads JSON-lines from S3 bucket (date-partitioned or json files).
Requires:
  - boto3 installed
  - MONITORING_BUCKET environment variable set to your bucket name
Optional env vars:
  - MONITORING_PREFIX (default "monitoring/logs")
  - AWS_REGION
  - S3_DAYS (how many recent days to scan, default 1)
  - S3_MAX_FILES (max files to fetch, default 500)
  - S3_MAX_WORKERS (concurrent downloads, default 8)
  - LABELED_OUTPUT (default ./monitoring_logs/labeled_samples.csv)
Run:
  pip install streamlit pandas numpy boto3 matplotlib
  export MONITORING_BUCKET=mlflow-vassu-monitoring
  streamlit run streamlit_inspector.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt
from io import BytesIO
import concurrent.futures
from botocore.exceptions import ClientError

# boto3 is required (S3-only)
try:
    import boto3
except Exception as e:
    boto3 = None

# ---------------------------
# Config & paths (from env)
# ---------------------------
MONITORING_BUCKET = os.environ.get("MONITORING_BUCKET", "").strip()
MONITORING_PREFIX = os.environ.get("MONITORING_PREFIX", "monitoring/logs")
AWS_REGION = os.environ.get("AWS_REGION", None)

S3_DAYS_DEFAULT = int(os.environ.get("S3_DAYS", "1"))
S3_MAX_FILES_DEFAULT = int(os.environ.get("S3_MAX_FILES", "500"))
S3_MAX_WORKERS = int(os.environ.get("S3_MAX_WORKERS", "8"))

LABELED_OUTPUT = os.environ.get("LABELED_OUTPUT", "./monitoring_logs/labeled_samples.csv")

st.set_page_config(page_title="Model Monitoring Inspector (S3-only)", layout="wide")

# ---------------------------
# Validate environment
# ---------------------------
if boto3 is None:
    st.error("boto3 is not installed. Install boto3 (pip install boto3) and restart the app.")
    st.stop()

if not MONITORING_BUCKET:
    st.error("MONITORING_BUCKET is not set. Set environment variable MONITORING_BUCKET to your S3 bucket (e.g. mlflow-vassu-monitoring).")
    st.stop()

# ---------------------------
# S3 helpers
# ---------------------------
def _get_s3_client():
    if AWS_REGION:
        return boto3.client("s3", region_name=AWS_REGION)
    return boto3.client("s3")

def list_keys_for_prefix(client, bucket, prefix):
    keys = []
    paginator = client.get_paginator("list_objects_v2")
    try:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for c in page.get("Contents", []):
                keys.append(c["Key"])
    except ClientError:
        return []
    return keys

def download_text_from_s3(client, bucket, key):
    resp = client.get_object(Bucket=bucket, Key=key)
    return resp["Body"].read().decode("utf-8", errors="ignore")

def parse_json_or_jsonlines(text):
    if not text:
        return []
    s = text.strip()
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return [p for p in parsed if isinstance(p, dict)]
        if isinstance(parsed, dict):
            return [parsed]
    except Exception:
        pass
    rows = []
    for line in s.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            j = json.loads(line)
            if isinstance(j, dict):
                rows.append(j)
        except Exception:
            continue
    return rows

# ---------------------------
# Cached S3 loader
# ---------------------------
@st.cache_data(ttl=30)
def load_s3_recent_logs_cached(days:int=S3_DAYS_DEFAULT, max_files:int=S3_MAX_FILES_DEFAULT) -> pd.DataFrame:
    client = _get_s3_client()
    keys = []
    today = date.today()
    for d in range(days):
        dt = today - timedelta(days=d)
        prefix = f"{MONITORING_PREFIX}/{dt.year:04d}/{dt.month:02d}/{dt.day:02d}/"
        keys.extend(list_keys_for_prefix(client, MONITORING_BUCKET, prefix))
    # If nothing found using date-partitioned prefix, try the prefix root
    if not keys:
        keys = list_keys_for_prefix(client, MONITORING_BUCKET, MONITORING_PREFIX + "/")
    # order newest-first and limit
    keys = sorted(keys, reverse=True)[:max_files]
    if not keys:
        return pd.DataFrame()
    rows = []
    def fetch_parse(key):
        try:
            txt = download_text_from_s3(client, MONITORING_BUCKET, key)
            parsed = parse_json_or_jsonlines(txt)
            for r in parsed:
                if isinstance(r, dict):
                    r["_s3_key"] = key
            return parsed
        except Exception:
            return []
    with concurrent.futures.ThreadPoolExecutor(max_workers=S3_MAX_WORKERS) as ex:
        futures = [ex.submit(fetch_parse, k) for k in keys]
        for f in concurrent.futures.as_completed(futures):
            parsed = f.result()
            if parsed:
                rows.extend(parsed)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for col in ["entropy", "max_confidence", "is_ood", "predicted", "probability_vector", "processed_text", "input_text", "model", "video_id"]:
        if col not in df.columns:
            df[col] = np.nan
    return df

# ---------------------------
# Utilities
# ---------------------------
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
st.title("🕵️ Model Monitoring — Inspector (S3-only)")
st.markdown(
    "This inspector loads monitoring logs directly from S3 (no local fallback). "
    "Make sure MONITORING_BUCKET is set and your environment has AWS credentials."
)

# Controls
col1, col2, col3 = st.columns([1,1,1])
with col1:
    days_to_scan = st.number_input("S3: days to scan (most recent)", min_value=1, max_value=30, value=S3_DAYS_DEFAULT, step=1)
with col2:
    max_files = st.number_input("S3: max files to fetch", min_value=10, max_value=5000, value=S3_MAX_FILES_DEFAULT, step=10)
with col3:
    top_k = st.number_input("Top-K high-entropy samples to inspect", min_value=5, max_value=500, value=50, step=5)

refresh = st.button("Refresh (clear cache & reload)")
if refresh:
    # clear the cache for S3 loader then reload
    load_s3_recent_logs_cached.clear()
    st.experimental_rerun()

# Load data from S3 (no fallback)
try:
    df = load_s3_recent_logs_cached(days=int(days_to_scan), max_files=int(max_files))
except Exception as e:
    st.error(f"Failed to load logs from S3: {e}")
    st.stop()

if df.empty:
    st.warning("No monitoring rows found in S3 for the given prefix/days. Check MONITORING_BUCKET/MONITORING_PREFIX and that objects contain JSON/JSONL.")
    st.stop()

# Summary metrics
summary = compute_summary(df)
st.subheader("Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total records loaded", summary["count"])
col2.metric("Average entropy", f"{summary['avg_entropy']:.3f}" if summary["avg_entropy"] is not None else "N/A")
col3.metric("Average confidence", f"{summary['avg_confidence']:.3f}" if summary["avg_confidence"] is not None else "N/A")
col4.metric("OOD rate (fraction)", f"{summary['ood_rate']:.3f}")

# Distributions
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

# Filter & sampling
st.subheader("Filter & sample")
time_col1, time_col2 = st.columns([2,1])
with time_col1:
    if "timestamp" in df.columns and df["timestamp"].notna().any():
        min_ts_pd = df["timestamp"].min()
        max_ts_pd = df["timestamp"].max()
        try:
            min_ts = min_ts_pd.to_pydatetime()
            max_ts = max_ts_pd.to_pydatetime()
        except Exception:
            import datetime as _dt
            if isinstance(min_ts_pd, _dt.datetime) and isinstance(max_ts_pd, _dt.datetime):
                min_ts = min_ts_pd
                max_ts = max_ts_pd
            else:
                st.info("Timestamps present but not in a recognized format; using all records.")
                min_ts = None
                max_ts = None
        if min_ts is not None and max_ts is not None:
            sel_range = st.slider(
                "Select time range",
                min_value=min_ts,
                max_value=max_ts,
                value=(min_ts, max_ts),
                format="YYYY-MM-DD HH:mm:ss"
            )
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

# Top high-entropy samples
st.subheader(f"Top {top_k} high-entropy samples (for manual inspection & labeling)")
if "entropy" not in df.columns or not df["entropy"].notna().any():
    st.info("No entropy values in logs. Your Flask app must log 'entropy' to enable sample selection.")
else:
    df_sorted = df.sort_values(by="entropy", ascending=False).reset_index(drop=True)
    top_df = df_sorted.head(top_k).copy()
    if "labels" not in st.session_state:
        st.session_state["labels"] = {}
    if "notes" not in st.session_state:
        st.session_state["notes"] = {}
    for idx, row in top_df.iterrows():
        key = f"sample_{idx}"
        with st.expander(f"[{idx+1}] Pred: {row.get('predicted')}  Entropy: {row.get('entropy'):.3f}  Conf: {row.get('max_confidence') if pd.notna(row.get('max_confidence')) else 'N/A'}"):
            st.markdown(f"**Raw text:** {row.get('input_text')}")
            st.markdown(f"**Processed text:** {row.get('processed_text')}")
            st.markdown(f"**Prob vector:** {row.get('probability_vector')}")
            current = st.session_state["labels"].get(key, "unlabeled")
            choices = ["unlabeled", "1 (Positive)", "0 (Neutral)", "-1 (Negative)"]
            index_choice = choices.index(current) if current in choices else 0
            choice = st.radio("Assign true label (use this for evaluation):", options=choices, index=index_choice, key=f"lab_{key}")
            st.session_state["labels"][key] = choice
            note = st.text_input("Notes (optional)", value=st.session_state["notes"].get(key, ""), key=f"note_{key}")
            st.session_state["notes"][key] = note
            if "timestamp" in row and pd.notna(row["timestamp"]):
                st.caption(f"Logged at: {row['timestamp']}")
            meta_caption = f"Model: {row.get('model')}  VideoID: {row.get('video_id')}"
            if "_s3_key" in row and pd.notna(row["_s3_key"]):
                meta_caption += f"  S3: {row.get('_s3_key')}"
            st.caption(meta_caption)

    if st.button("Save selected labels to CSV"):
        rows_to_save = []
        for idx, row in top_df.iterrows():
            key = f"sample_{idx}"
            sel = st.session_state["labels"].get(key, "unlabeled")
            if sel != "unlabeled":
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
                    "s3_key": row.get("_s3_key") if "_s3_key" in row else None,
                    "notes": st.session_state["notes"].get(key, "")
                })
        if rows_to_save:
            df_labels = pd.DataFrame(rows_to_save)
            save_labeled_samples(df_labels, LABELED_OUTPUT)
            st.success(f"Saved {len(df_labels)} labeled rows to `{LABELED_OUTPUT}`")
        else:
            st.info("No labeled rows selected to save.")

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
                    "s3_key": row.get("_s3_key") if "_s3_key" in row else None,
                    "notes": st.session_state["notes"].get(key, "")
                })
        if rows_to_export:
            df_export = pd.DataFrame(rows_to_export)
            csv_bytes = df_to_csv_bytes(df_export)
            st.download_button("Download labeled CSV", data=csv_bytes, file_name="labeled_samples_export.csv", mime="text/csv")
        else:
            st.info("No labeled rows selected to export.")

st.markdown("---")
st.subheader("Raw log sample (most recent records)")
show_n = st.number_input("Show first N rows of loaded dataframe", min_value=5, max_value=1000, value=50, step=5)
if not df.empty:
    st.dataframe(df.head(show_n))
    csv_all = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download loaded records CSV", data=csv_all, file_name="monitoring_loaded.csv", mime="text/csv")

st.info(f"Labeled samples saved at: `{LABELED_OUTPUT}` (appends if file exists)")
st.caption(f"Last refreshed: {datetime.utcnow().isoformat()}  | Source: S3 bucket `{MONITORING_BUCKET}` prefix `{MONITORING_PREFIX}`")
