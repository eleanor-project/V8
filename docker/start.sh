#!/bin/bash

echo "-----------------------------------------"
echo "Starting ELEANOR V8 Appliance"
echo "-----------------------------------------"

docker compose up --build

echo "-----------------------------------------"
echo "ELEANOR V8 Appliance Running:"
echo " - API: http://localhost:8000"
echo " - WS:  ws://localhost:8000/ws/deliberate"
echo " - OPA: http://localhost:8181"
echo " - Weaviate: http://localhost:8080"
echo " - PGVector: postgresql://postgres:postgres@localhost:5432/eleanor"
echo "-----------------------------------------"
