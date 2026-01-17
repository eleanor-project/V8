# FastAPI Integration

The REST API uses FastAPI under `api/rest/`. The main app is in
`api/rest/main.py`.

## Run locally

```bash
uvicorn api.rest.main:app --reload
```

## Deliberation endpoint

`POST /deliberate` accepts `DeliberationRequest` defined in `api/schemas.py`.

See `api/rest/routes/deliberation.py` for request handling.
