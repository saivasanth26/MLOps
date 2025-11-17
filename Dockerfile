# Dockerfile (minimal changes for S3 logging & nltk)
FROM python:3.11-slim

WORKDIR /app

# Install OS-level dependencies (required by lightgbm and build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 build-essential ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python deps
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /app/requirements.txt

# Download nltk data at build time to avoid runtime downloads
# (stopwords and wordnet are used by your app)
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"


# Copy application code
COPY . /app

# Expose the same port as before
EXPOSE 5001

# Default command (same as before)
CMD ["python3", "flask_app/app.py"]
