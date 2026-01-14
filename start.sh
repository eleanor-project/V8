#!/bin/bash

echo "-----------------------------------------"
echo "Starting ELEANOR V8 Appliance (Docker Compose)"
echo "-----------------------------------------"

docker compose up --build -d

echo "-----------------------------------------"
echo "ELEANOR V8 Appliance Running:"
echo " - UI:  http://localhost:8000/ui/"
echo " - API: http://localhost:8000"
echo " - WS:  ws://localhost:8000/ws/deliberate"
echo " - OPA: http://localhost:8181"
echo " - Weaviate: http://localhost:8080"
echo " - PGVector: postgresql://postgres:postgres@localhost:5432/eleanor"
echo " - Grafana: http://localhost:3000"
echo " - Prometheus: http://localhost:9090"
echo "-----------------------------------------"
echo "To view logs: docker compose logs -f"
