FROM python:3.11-slim

WORKDIR /app

# Install OS-level dependencies (required by lightgbm)
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

EXPOSE 5001
CMD ["python3", "flask_app/app.py"]
