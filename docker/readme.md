# ELEANOR V8 Appliance — Docker Deployment

This directory contains everything needed to run the entire
ELEANOR V8 governance engine as a fully integrated local appliance.

## Components

### 1. ELEANOR API
- FastAPI (REST)
- WebSocket streaming
- CLI integration
- Unified Engine Runtime
- Audit logging

### 2. OPA (Open Policy Agent)
Houses the V8 constitutional policies and enforces allow/deny rules.

### 3. Vector DBs for Precedent Retrieval
Two options (controlled via PRECEDENT_BACKEND):

#### Weaviate (default)
- Flexible
- Fast for prototyping
- No schema migrations

#### PGVector
- Stable for enterprise deployments
- Uses PostgreSQL with pgvector extension
- Great for large precedent stores

### 4. Networking
All services run inside a shared docker network.

---

## Running the Appliance

```bash
cd docker
./start.sh
