# app_instrumented.py
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot

from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mlflow
import numpy as np
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient
import matplotlib.dates as mdates
import pickle
import os
import json
import uuid
import datetime
import time
import logging

# Monitoring deps
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Optional boto3 for S3 logging (only used if MONITORING_S3 env var set)
try:
    import boto3
except Exception:
    boto3 = None

app = Flask(__name__)
CORS(app, origins=["chrome-extension://fbnbnolhlfkdgedhibaegminabjfbhke"])
logger = app.logger
logger.setLevel(logging.INFO)

# ---------------------------
# Prometheus metrics (labels: model)
# ---------------------------
MODEL_NAME = os.environ.get("MODEL_NAME", "lgbm_local_v1")

REQUESTS = Counter('inference_requests_total', 'Total inference requests', ['model'])
ERRORS = Counter('inference_errors_total', 'Inference errors', ['model'])
LATENCY = Histogram('inference_latency_seconds', 'Inference latency seconds', ['model'])
OOD_FLAGS = Counter('inference_ood_total', 'Inputs flagged OOD', ['model'])
PRED_CONF_SUM = Gauge('prediction_confidence_sum', 'Sum of max confidences', ['model'])
PRED_CONF_COUNT = Gauge('prediction_confidence_count', 'Count of confidences', ['model'])
PRED_CLASS = Counter('prediction_count', 'Predicted class counts', ['model', 'label'])

# ---------------------------
# Monitoring / logging config
# ---------------------------
LOCAL_LOG_DIR = os.environ.get("LOCAL_LOG_DIR", "./monitoring_logs")
LOCAL_LOG_FILE = os.path.join(LOCAL_LOG_DIR, "predictions.log")
MONITORING_S3 = os.environ.get("MONITORING_S3", "false").lower() in ("1", "true", "yes")
S3_BUCKET = os.environ.get("MONITORING_BUCKET", "")
S3_PREFIX = os.environ.get("MONITORING_PREFIX", "monitoring/ingest/")
AWS_REGION = os.environ.get("AWS_REGION", None)

# OOD thresholds (tune later)
OOD_CONFIDENCE_THRESHOLD = float(os.environ.get("OOD_CONFIDENCE_THRESHOLD", 0.5))
OOD_ENTROPY_THRESHOLD = float(os.environ.get("OOD_ENTROPY_THRESHOLD", 1.0))

# Initialize S3 client if needed
s3_client = None
if MONITORING_S3:
    if boto3 is None:
        logger.warning("MONITORING_S3 enabled but boto3 not installed. Install boto3 or disable MONITORING_S3.")
        MONITORING_S3 = False
    else:
        if AWS_REGION:
            s3_client = boto3.client("s3", region_name=AWS_REGION)
        else:
            s3_client = boto3.client("s3")

# Ensure local log dir exists
os.makedirs(LOCAL_LOG_DIR, exist_ok=True)

# entropy helper (scipy fallback optional)
try:
    from scipy.stats import entropy as scipy_entropy
    def entropy_fn(p): 
        return float(scipy_entropy(p))
except Exception:
    def entropy_fn(p):
        p = np.clip(np.array(p, dtype=float), 1e-12, 1.0)
        return float(-np.sum(p * np.log(p)))

# ---------------------------
# Your existing preprocess and model loading
# ---------------------------
def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        comment = comment.lower()
        comment = comment.strip()
        comment = re.sub(r'\n', ' ', comment)
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])
        return comment
    except Exception as e:
        logger.exception("Error in preprocessing comment")
        return comment

def load_model(model_path, vectorizer_path):
    """Load the trained model and vectorizer from disk."""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except Exception as e:
        logger.exception("Failed to load model/vectorizer")
        raise

# Update these paths if needed
MODEL_PICKLE_PATH = os.environ.get("MODEL_PICKLE_PATH", "./flask_app/lgbm_model.pkl")
VECT_PICKLE_PATH = os.environ.get("VECT_PICKLE_PATH", "./flask_app/tfidf_vectorizer.pkl")
model, vectorizer = load_model(MODEL_PICKLE_PATH, VECT_PICKLE_PATH)

# ---------------------------
# Logging helpers (local + optional S3)
# ---------------------------
def append_local_log(record: dict):
    """Append a JSON line to the local monitoring log file."""
    try:
        with open(LOCAL_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error("Failed to write local monitoring log: %s", e)

def upload_to_s3(record: dict):
    """Upload single JSON record to S3 (best for low QPS)."""
    if not MONITORING_S3 or s3_client is None or not S3_BUCKET:
        return
    key = S3_PREFIX + f"{datetime.datetime.utcnow().isoformat()}-{uuid.uuid4().hex[:8]}.json"
    try:
        s3_client.put_object(Bucket=S3_BUCKET, Key=key, Body=json.dumps(record).encode("utf-8"))
    except Exception as e:
        logger.error("Failed to upload monitoring record to S3: %s", e)

def log_prediction_event(record: dict):
    """Write monitoring event locally and optionally to S3."""
    append_local_log(record)
    if MONITORING_S3:
        upload_to_s3(record)

# ---------------------------
# Prometheus / metrics endpoint
# ---------------------------
@app.route('/metrics')
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

# ---------------------------
# Shared prediction helper
# ---------------------------
def process_and_predict(comments, timestamps=None, extra_meta=None):
    """
    - comments: list[str]
    - timestamps: list[str] or None
    - extra_meta: dict (e.g., video_id)
    Returns results list of dicts (includes monitoring fields).
    """
    extra_meta = extra_meta or {}
    if timestamps is None:
        timestamps = [datetime.datetime.utcnow().isoformat()] * len(comments)

    REQUESTS.labels(MODEL_NAME).inc(len(comments))
    start_all = time.time()

    # Preprocess & vectorize
    preprocessed = [preprocess_comment(c) for c in comments]
    transformed = vectorizer.transform(preprocessed)
    dense = transformed.toarray()

    # Predict
    start_pred = time.time()
    try:
        preds = model.predict(dense)
    except Exception as e:
        ERRORS.labels(MODEL_NAME).inc()
        logger.exception("Prediction failed")
        raise
    pred_time = time.time() - start_pred
    LATENCY.labels(MODEL_NAME).observe(pred_time)

    # Try to get probabilities
    probs = None
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(dense)
    except Exception as e:
        logger.warning("predict_proba unavailable or failed: %s", e)
        probs = None

    results = []
    for i, comment in enumerate(comments):
        ts = timestamps[i] if i < len(timestamps) else datetime.datetime.utcnow().isoformat()
        raw_pred = preds[i]
        # normalize pred to int/string
        try:
            pred_out = int(raw_pred)
        except Exception:
            pred_out = str(raw_pred)

        ent = None
        max_conf = None
        probs_list = None
        try:
            if probs is not None:
                p = np.array(probs[i], dtype=float)
                probs_list = p.tolist()
                ent = entropy_fn(p)
                max_conf = float(p.max())
                # update gauges
                PRED_CONF_SUM.labels(MODEL_NAME).inc(max_conf)
                PRED_CONF_COUNT.labels(MODEL_NAME).inc(1)
        except Exception as e:
            logger.warning("Failed to compute probs/entropy for sample %d: %s", i, e)

        # determine OOD via simple rule
        is_ood = False
        if max_conf is not None and max_conf < OOD_CONFIDENCE_THRESHOLD:
            is_ood = True
        if ent is not None and ent > OOD_ENTROPY_THRESHOLD:
            is_ood = True
        if is_ood:
            OOD_FLAGS.labels(MODEL_NAME).inc()

        # increment class counter (label as string)
        try:
            PRED_CLASS.labels(MODEL_NAME, str(pred_out)).inc()
        except Exception:
            pass

        record = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "input_text": comment,
            "processed_text": preprocessed[i],
            "predicted": pred_out,
            "probability_vector": probs_list,
            "max_confidence": max_conf,
            "entropy": ent,
            "is_ood": bool(is_ood),
            "model": MODEL_NAME,
        }
        record.update(extra_meta or {})

        # write logs (local and optional S3)
        try:
            log_prediction_event(record)
        except Exception as e:
            logger.error("Failed to log prediction event: %s", e)

        # response-friendly version
        resp = {
            "comment": comment,
            "sentiment": str(pred_out),
            "timestamp": ts,
            # keep monitoring fields if you want to return them (comment out if not)
            "monitoring": {
                "entropy": ent,
                "max_confidence": max_conf,
                "is_ood": bool(is_ood)
            }
        }
        results.append(resp)

    total_time = time.time() - start_all
    logger.info("Processed %d comments in %.3fs (predict %.3fs)", len(comments), total_time, pred_time)
    return results

# ---------------------------
# Routes (existing functionality preserved + monitoring)
# ---------------------------

@app.route('/')
def home():
    return "Welcome to our flask api"

@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():
    data = request.json
    comments_data = data.get('comments')
    video_id = data.get('video_id') if data else None

    if not comments_data:
        return jsonify({"error": "No comments provided"}), 400
    try:
        comments = [item['text'] for item in comments_data]
        timestamps = [item.get('timestamp') for item in comments_data]
        extra_meta = {"video_id": video_id} if video_id else {}
        results = process_and_predict(comments, timestamps=timestamps, extra_meta=extra_meta)
        # keep response structure same as before (comment, sentiment, timestamp)
        simple_resp = [{"comment": r["comment"], "sentiment": r["sentiment"], "timestamp": r["timestamp"]} for r in results]
        return jsonify(simple_resp)
    except Exception as e:
        logger.exception("predict_with_timestamps error")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    comments = data.get('comments')
    video_id = data.get('video_id') if data else None

    logger.debug("predict called; comments type: %s", type(comments))
    if not comments:
        return jsonify({"error": "No comments provided"}), 400
    try:
        extra_meta = {"video_id": video_id} if video_id else {}
        results = process_and_predict(comments, extra_meta=extra_meta)
        # Return simplified output (comment + sentiment)
        simple_resp = [{"comment": r["comment"], "sentiment": r["sentiment"]} for r in results]
        return jsonify(simple_resp)
    except Exception as e:
        logger.exception("predict error")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

# --------------- keep chart/wordcloud/trend endpoints unchanged ---------------
@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    try:
        data = request.get_json()
        sentiment_counts = data.get('sentiment_counts')
        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400

        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            int(sentiment_counts.get('1', 0)),
            int(sentiment_counts.get('0', 0)),
            int(sentiment_counts.get('-1', 0))
        ]
        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")

        colors = ['#36A2EB', '#C9CBCF', '#FF6384']

        plt.figure(figsize=(6, 6))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, textprops={'color': 'w'})
        plt.axis('equal')

        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_chart: {e}")
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500

@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        data = request.get_json()
        comments = data.get('comments')
        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        text = ' '.join(preprocessed_comments)

        wordcloud = WordCloud(
            width=800, height=400, background_color='black',
            colormap='Blues', stopwords=set(stopwords.words('english')), collocations=False
        ).generate(text)

        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_wordcloud: {e}")
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500

@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    try:
        data = request.get_json()
        sentiment_data = data.get('sentiment_data')
        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df['sentiment'] = df['sentiment'].astype(int)

        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}
        monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)
        monthly_totals = monthly_counts.sum(axis=1)
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0
        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        plt.figure(figsize=(12, 6))
        colors = {-1: 'red', 0: 'gray', 1: 'green'}
        for sentiment_value in [-1, 0, 1]:
            plt.plot(monthly_percentages.index, monthly_percentages[sentiment_value],
                     marker='o', linestyle='-', label=sentiment_labels[sentiment_value],
                     color=colors[sentiment_value])
        plt.title('Monthly Sentiment Percentage Over Time')
        plt.xlabel('Month')
        plt.ylabel('Percentage of Comments (%)')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
        plt.legend()
        plt.tight_layout()

        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG')
        img_io.seek(0)
        plt.close()
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_trend_graph: {e}")
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500

# ---------------------------
# Run
# ---------------------------
if __name__ == '__main__':
    # For local testing run with debug=True. For production use gunicorn.
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5001)), debug=True)
