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
- The Grafana iframe assumes Grafana on `http://localhost:3000`; adjust in `src/App.jsx` if needed.
