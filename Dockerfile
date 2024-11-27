FROM python:3.12.7-bookworm

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY taxis/ taxis/
COPY trajectory_clustering/ .

