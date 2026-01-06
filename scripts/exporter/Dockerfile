FROM python:3.11-slim

WORKDIR /app

COPY ../dependency_prometheus_exporter.py .

RUN pip install --no-cache-dir prometheus-client

EXPOSE 9105

ENTRYPOINT ["python", "dependency_prometheus_exporter.py"]
