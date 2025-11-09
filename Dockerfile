FROM python:3.8.5-slim-buster
WORKDIR /app

# copy entire app first (so file:///app points to a real package)
COPY . /app

# then install requirements (which can include file:///app)
RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r /app/requirements.txt

#EXPOSE 5001
CMD ["python3", "app.py"]

