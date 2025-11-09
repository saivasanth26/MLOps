FROM python:3.8.5-slim-buster

# set working dir
WORKDIR /app

# copy only requirements first to leverage Docker layer cache
COPY requirements.txt /app/requirements.txt

# optional: install system deps if your pip packages need compilation
# RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# upgrade pip + install python deps (no cache to reduce image size)
RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r /app/requirements.txt

# copy the rest of the application
COPY . /app

# expose port your flask app uses (change if you use another port)
EXPOSE 5001

# start the app
CMD ["python3", "app.py"]

