FROM python:3.11-slim-bullseye

WORKDIR /app

# system deps required for many scientific libs + building wheels
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential gcc libgomp1 libfreetype6-dev libpng-dev libjpeg-dev pkg-config curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# copy app
COPY . /app

# upgrade pip/tools first (helps with newer wheels)
RUN python -m pip install --upgrade pip setuptools wheel

# install requirements
RUN pip install --no-cache-dir -r /app/requirements.txt

CMD ["python3", "flask_app/app.py"]
