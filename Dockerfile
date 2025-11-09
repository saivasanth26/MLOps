# Use a newer slim base to get modern wheel compatibility
FROM python:3.11-slim-bullseye

# install system deps required by some Python packages (lightgbm, wordcloud, matplotlib, dvc, etc.)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    build-essential gcc g++ git curl ca-certificates \
    libgomp1 libfreetype6-dev libpng-dev libjpeg-dev pkg-config \
    libssl-dev libffi-dev libxml2-dev libxslt1-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy whole repo (requirements at project root)
COPY . /app

# upgrade pip & install python deps
RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r /app/requirements.txt

# ensure your app binds to 0.0.0.0 and the expected port (5001)
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5001

# make sure we run the correct entrypoint (adjust path if needed)
# If your entry is flask_app/app.py:
CMD ["python3", "flask_app/app.py"]
