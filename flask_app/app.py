# flask_app_monitoring.py
import os
import io
import json
import time
import pickle
import math
import boto3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from datetime import datetime
from uuid import uuid4
from botocore.exceptions import ClientError

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# ----------------------------
# Config from env
# ----------------------------
MONITORING_S3 = os.environ.get("MONITORING_S3", "false").lower() in ("1", "true", "yes")
MONITORING_BUCKET = os.environ.get("MONITORING_BUCKET", "mlflow-vassu-monitoring")
MONITORING_PREFIX = os.environ.get("MONITORING_PREFIX", "monitoring/logs")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
LOCAL_MONITORING_FILE = os.environ.get("LOCAL_MONITORING_FILE", "./monitoring_logs/predictions.log")
OOD_ENTROPY_THRESHOLD = float(os.environ.get("OOD_ENTROPY_THRESHOLD", "1.0"))
CORS_ORIGIN = os.environ.get("CORS_ORIGIN", "chrome-extension://fbnbnolhlfkdgedhibaegminabjfbhke")

# create local directory for fallback logs
os.makedirs(os.path.dirname(LOCAL_MONITORING_FILE) or ".", exist_ok=True)

# ----------------------------
# Flask app
# ----------------------------
app = Flask(__name__)
CORS(app, origins=[CORS_ORIGIN])

# ----------------------------
# S3 client wrapper with retries
# ----------------------------
_s3_client = None
def s3_client():
    global _s3_client
    if _s3_client is None:
        if AWS_REGION:
            _s3_client = boto3.client("s3", region_name=AWS_REGION)
        else:
            _s3_client = boto3.client("s3")
    return _s3_client

def upload_json_to_s3(obj: dict, bucket=MONITORING_BUCKET, prefix=MONITORING_PREFIX, max_retries=3):
    """Upload JSON object to s3 with retries. Returns key on success, None on failure."""
    if not MONITORING_S3:
        return None
    client = s3_client()
    ts_path = datetime.utcnow().strftime("%Y/%m/%d")
    key = f"{prefix}/{ts_path}/{datetime.utcnow().strftime('%H%M%S')}_{uuid4().hex}.json"
    body = json.dumps(obj, default=str).encode("utf-8")
    attempt = 0
    backoff = 0.5
    while attempt < max_retries:
        try:
            client.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
            return key
        except ClientError as e:
            attempt += 1
            time.sleep(backoff)
            backoff *= 2
    return None

def append_local_jsonline(obj: dict, path=LOCAL_MONITORING_FILE):
    """Append JSON-lines locally (safe fallback)."""
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, default=str) + "\n")
    except Exception as e:
        # last resort: print
        print("Failed to append local monitoring file:", e)

def safe_log_event(event: dict):
    """Try S3, fallback to local file. Never raise."""
    try:
        if MONITORING_S3:
            key = upload_json_to_s3(event)
            if key:
                # good
                return
        # fallback
        append_local_jsonline(event)
    except Exception as e:
        print("Monitoring logging failed (ignored):", e)
        try:
            append_local_jsonline(event)
        except Exception:
            print("Also failed to write local fallback (ignored).")

# ----------------------------
# Helper functions: preprocessing, entropy, safe predict_proba
# ----------------------------
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure nltk data exists in your environment - if not, download separately in CI / runtime
# import nltk
# nltk.download('stopwords'); nltk.download('wordnet')  # run during setup if required

lemmatizer = WordNetLemmatizer()
DEFAULT_STOPWORDS = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}

def preprocess_comment(comment: str):
    try:
        c = comment.lower().strip()
        c = re.sub(r'\n', ' ', c)
        c = re.sub(r'[^A-Za-z0-9\s!?.,]', '', c)
        c = ' '.join([w for w in c.split() if w not in DEFAULT_STOPWORDS])
        c = ' '.join([lemmatizer.lemmatize(w) for w in c.split()])
        return c
    except Exception:
        return comment

def safe_predict_proba(model, X):
    """Return probability vectors if supported, else None."""
    try:
        proba = model.predict_proba(X)
        # ensure list of lists of floats
        return [list(map(float, p)) for p in proba]
    except Exception:
        return None

def compute_entropy_from_probs(probs):
    """Shannon entropy in nats (use natural log). Input: list of probabilities."""
    try:
        p = np.array(probs, dtype=float)
        # numeric stability
        p = np.clip(p, 1e-12, 1.0)
        ent = -np.sum(p * np.log(p))
        return float(ent)
    except Exception:
        return None

# ----------------------------
# Load model & vectorizer (your existing local load)
# ----------------------------
def load_model_and_vectorizer(model_path="./flask_app/lgbm_model.pkl", vec_path="./flask_app/tfidf_vectorizer.pkl"):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vec_path, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

# initialize
try:
    model, vectorizer = load_model_and_vectorizer()
    print("Model and vectorizer loaded")
except Exception as e:
    print("Failed to load model/vectorizer:", e)
    model, vectorizer = None, None

# ----------------------------
# Endpoints
# ----------------------------
@app.route("/")
def home():
    return "Flask model + monitoring API (monitoring -> S3/local)."

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    comments = data.get("comments")
    video_id = data.get("video_id", None)

    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    # Preprocess -> vectorize -> predict
    try:
        preprocessed = [preprocess_comment(c) for c in comments]
        transformed = vectorizer.transform(preprocessed)
        # convert to dense only if required by model; try to call predict directly
        try:
            preds = model.predict(transformed)
        except Exception:
            # fallback: some models require dense
            preds = model.predict(transformed.toarray())
        # try to get probabilities
        prob_vectors = safe_predict_proba(model, transformed) or safe_predict_proba(model, transformed.toarray())
        max_confidences = [max(p) if p is not None else None for p in (prob_vectors or [None]*len(preds))]
        entropies = [compute_entropy_from_probs(p) if p is not None else None for p in (prob_vectors or [None]*len(preds))]

        # ensure preds are JSON-serializable ints or strings like earlier
        preds_out = [str(int(p)) if (p is not None and (isinstance(p, (int, np.integer)) or (isinstance(p, float) and p.is_integer()))) else str(p) for p in preds]

        # prepare response
        response = [{"comment": c, "sentiment": pred} for c, pred in zip(comments, preds_out)]

        # log each event (non-blocking in sense we don't let failures break)
        for i, c in enumerate(comments):
            ev = {
                "timestamp": datetime.utcnow().isoformat(),
                "input_text": c,
                "processed_text": preprocessed[i],
                "predicted": int(preds[i]) if preds is not None else None,
                "probability_vector": prob_vectors[i] if (prob_vectors is not None and len(prob_vectors)>i) else None,
                "max_confidence": float(max_confidences[i]) if max_confidences[i] is not None else None,
                "entropy": float(entropies[i]) if entropies[i] is not None else None,
                "is_ood": bool(entropies[i] is not None and entropies[i] > OOD_ENTROPY_THRESHOLD),
                "model": "lgbm_local_v1",
                "video_id": video_id
            }
            # best-effort send
            try:
                safe_log_event(ev)
            except Exception as e:
                print("Logging event failed (ignored):", e)

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route("/predict_with_timestamps", methods=["POST"])
def predict_with_timestamps():
    payload = request.json
    comments_data = payload.get("comments")
    if not comments_data:
        return jsonify({"error": "No comments provided"}), 400
    try:
        comments = [item.get("text") for item in comments_data]
        timestamps = [item.get("timestamp") for item in comments_data]
        preprocessed = [preprocess_comment(c) for c in comments]
        transformed = vectorizer.transform(preprocessed)
        try:
            preds = model.predict(transformed)
        except Exception:
            preds = model.predict(transformed.toarray())
        prob_vectors = safe_predict_proba(model, transformed) or safe_predict_proba(model, transformed.toarray())
        max_confidences = [max(p) if p is not None else None for p in (prob_vectors or [None]*len(preds))]
        entropies = [compute_entropy_from_probs(p) if p is not None else None for p in (prob_vectors or [None]*len(preds))]

        preds_out = [str(int(p)) if (p is not None and (isinstance(p, (int, np.integer)) or (isinstance(p, float) and p.is_integer()))) else str(p) for p in preds]

        # log events
        for i, c in enumerate(comments):
            ev = {
                "timestamp": timestamps[i] if timestamps and i < len(timestamps) else datetime.utcnow().isoformat(),
                "input_text": c,
                "processed_text": preprocessed[i],
                "predicted": int(preds[i]) if preds is not None else None,
                "probability_vector": prob_vectors[i] if (prob_vectors is not None and len(prob_vectors)>i) else None,
                "max_confidence": float(max_confidences[i]) if max_confidences[i] is not None else None,
                "entropy": float(entropies[i]) if entropies[i] is not None else None,
                "is_ood": bool(entropies[i] is not None and entropies[i] > OOD_ENTROPY_THRESHOLD),
                "model": "lgbm_local_v1",
                "video_id": payload.get("video_id")
            }
            try:
                safe_log_event(ev)
            except Exception as e:
                print("Logging event failed (ignored):", e)

        response = [{"comment": c, "sentiment": pred, "timestamp": timestamps[i] if timestamps and i < len(timestamps) else None} for i, (c, pred) in enumerate(zip(comments, preds_out))]
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": f"Prediction with timestamp failed: {str(e)}"}), 500

# The rest of your endpoints (generate_chart, generate_wordcloud, generate_trend_graph)
# remain functionally same as before but we keep them here for completeness.
# Copy paste your existing implementations below (unchanged), they do not affect monitoring.

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
        plt.figure(figsize=(6, 6))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.axis('equal')
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500

@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        data = request.get_json()
        comments = data.get('comments')
        if not comments:
            return jsonify({"error": "No comments provided"}), 400
        preprocessed_comments = [preprocess_comment(c) for c in comments]
        text = ' '.join(preprocessed_comments)
        wordcloud = WordCloud(width=800, height=400, background_color='black', stopwords=set(stopwords.words('english')), collocations=False).generate(text)
        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
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
        for sentiment_value in [-1, 0, 1]:
            plt.plot(monthly_percentages.index, monthly_percentages[sentiment_value], marker='o', linestyle='-', label=sentiment_labels[sentiment_value])
        plt.title('Monthly Sentiment Percentage Over Time')
        plt.xlabel('Month')
        plt.ylabel('Percentage of Comments (%)')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG')
        img_io.seek(0)
        plt.close()
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500

# ----------------------------
# Run app
# ----------------------------
if __name__ == "__main__":
    # use port 5001 as before
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5001)), debug=False)
