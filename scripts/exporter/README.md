# Dependency Prometheus Exporter

This folder packages `../dependency_prometheus_exporter.py` so you can run it as a sidecar.

## Build

```bash
cd /path/to/v8
docker build -t eleanor/dependency-exporter -f scripts/exporter/Dockerfile scripts
```

## Run

```bash
docker run -d --name dependency-exporter \
  -e DEP_EXPORT_ADMIN_ENDPOINT=http://eleanor-api:8000/admin/dependencies \
  -e DEP_EXPORT_LISTEN_ADDR=0.0.0.0 \
  -e DEP_EXPORT_LISTEN_PORT=9105 \
  -p 9105:9105 \
  eleanor/dependency-exporter
```

Adjust `DEP_EXPORT_*` env vars if you need a different admin host, listen port, or scrape interval.

## Prometheus scrape config

Add a job that points at the exporter, e.g.

```yaml
scrape_configs:
  - job_name: eleanor_dependencies
    metrics_path: /metrics
    static_configs:
      - targets: ["dependency-exporter:9105"]
```
