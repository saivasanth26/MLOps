# Use a recent slim image (avoid old buster repo problems)
FROM python:3.11-slim

WORKDIR /app

# Copy only requirements first to use build cache
COPY requirements.txt /app/requirements.txt

# Upgrade pip and install python deps
RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r /app/requirements.txt

# Copy entire repo (includes flask_app/, model pkls, etc)
COPY . /app

# Expose the port your app listens on (informational)
EXPOSE 5001

# Run exactly like you run locally
CMD ["python3", "flask_app/app.py"]
