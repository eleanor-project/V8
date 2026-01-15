## ELEANOR V8 — React SPA Console

Lightweight Vite + React console for running deliberations, streaming events, replaying traces, managing critic bindings, and viewing metrics.

### Prerequisites
- Node 18+ and npm/pnpm/yarn

### Install & Run (dev)
```bash
cd ui
npm install
npm run dev   # Vite dev server (default http://localhost:5173)
```

### Build
```bash
npm run build   # outputs to ui/dist
```

### Serve with API
- FastAPI serves `/ui`:
  - If `ui/dist` exists, it serves the built assets.
  - Otherwise it serves the raw `ui/` directory (useful after `npm run dev` with `npm run build`).

### Notes
- The console calls the same REST/WebSocket/admin endpoints already provided by the API.
- The Grafana iframe assumes Grafana on `http://localhost:3000` unless configured via `VITE_GRAFANA_URL`.
- The Governance and Human Review tabs require roles configured via `REVIEWER_ROLE` on the API.
- Admin write actions require `ELEANOR_ENABLE_ADMIN_WRITE=true` on the API.
- The header role check probes `/admin/write-enabled` and `/review/pending` to confirm admin/reviewer access.

### Runtime Configuration
Set Vite env vars to point the console at your API stack:

- `VITE_API_BASE_URL` (optional): Base URL for REST calls (default: same origin).
- `VITE_WS_BASE_URL` (optional): Base URL for WebSocket streaming (default: same origin).
- `VITE_GRAFANA_URL` (optional): Full Grafana dashboard URL to embed.

The UI stores an auth token in `localStorage` under `eleanor_token` and sends it as a Bearer token
for REST calls and as a `?token=` query param for WebSocket connections.
