FROM python:3.8.5-slim-buster

WORKDIR /app

RUN pip install -r requirements.txt

COPY . /app

CMD [ "python3","app.py" ]
