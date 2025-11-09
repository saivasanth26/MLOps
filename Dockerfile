# Dockerfile (replace existing)
FROM python:3.8.5-slim-buster

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install OS packages required for many scientific libs and compilation
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential gcc libgomp1 libfreetype6-dev libpng-dev libjpeg-dev pkg-config curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Copy all files first (you said you rely on repo layout)
COPY . /app

# Upgrade pip and install python deps.
# We strip '-e .' lines (editable local install) from requirements during build,
# so your repo doesn't need pyproject/setup.py for the container build.
RUN python -m pip install --upgrade pip setuptools wheel \
 && grep -v "^-e[[:space:]]*\\." /app/requirements.txt > /tmp/requirements-stripped.txt || cp /app/requirements.txt /tmp/requirements-stripped.txt \
 && cat /tmp/requirements-stripped.txt \
 && pip install --no-cache-dir -r /tmp/requirements-stripped.txt

# Expose the port your Flask app uses
EXPOSE 5001

# Ensure the container runs your flask app (adjust if you use gunicorn)
CMD ["python3", "flask_app/app.py"]
