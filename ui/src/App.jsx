import React, { useEffect, useMemo, useState } from "react";

const apiBase = "";
const wsUrl = () => {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${window.location.host}/ws/deliberate`;
};

const fetchJson = async (path, opts = {}) => {
  const res = await fetch(apiBase + path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  return await res.json();
};

const pretty = (obj) => JSON.stringify(obj, null, 2);

const DeliberatePanel = () => {
  const [input, setInput] = useState("Should I approve this loan application?");
  const [context, setContext] = useState('{"user_id":"demo","category":"finance"}');
  const [result, setResult] = useState(null);
  const [streamEvents, setStreamEvents] = useState([]);
  const [loading, setLoading] = useState(false);
  const [streaming, setStreaming] = useState(false);

  const run = async () => {
    setLoading(true);
    setResult(null);
    try {
      const body = { input, context: context ? JSON.parse(context) : {} };
      const data = await fetchJson("/deliberate", {
        method: "POST",
        body: JSON.stringify(body),
      });
      setResult(data);
    } catch (err) {
      setResult({ error: err.message });
    } finally {
      setLoading(false);
    }
  };

  const stream = async () => {
    setStreaming(true);
    setStreamEvents([]);
    try {
      const ws = new WebSocket(wsUrl());
      ws.onopen = () =>
        ws.send(JSON.stringify({ input, context: context ? JSON.parse(context) : {} }));
      ws.onmessage = (ev) => {
        const msg = JSON.parse(ev.data);
        setStreamEvents((prev) => [...prev, msg]);
        if (msg.event === "final") {
          setStreaming(false);
          ws.close();
        }
      };
      ws.onerror = (e) => {
        setStreamEvents((prev) => [...prev, { event: "error", data: e.message || "ws error" }]);
        setStreaming(false);
      };
    } catch (err) {
      setStreamEvents((prev) => [...prev, { event: "error", data: err.message }]);
      setStreaming(false);
    }
  };

  return (
    <div className="grid two">
      <div className="panel">
        <h3>Deliberate</h3>
        <label className="small">Input</label>
        <textarea value={input} onChange={(e) => setInput(e.target.value)} />
        <label className="small">Context (JSON)</label>
        <textarea value={context} onChange={(e) => setContext(e.target.value)} />
        <div className="row" style={{ marginTop: 8 }}>
          <button onClick={run} disabled={loading || streaming}>
            Run REST
          </button>
          <button onClick={stream} disabled={loading || streaming}>
            Stream
          </button>
          {loading && <span className="small">Running…</span>}
          {streaming && <span className="small">Streaming…</span>}
        </div>
        <h4>Result</h4>
        <pre>{result ? pretty(result) : "—"}</pre>
      </div>
      <div className="panel">
        <h3>Stream Events</h3>
        <div className="scroll">
          {streamEvents.map((evt, idx) => (
            <pre key={idx}>{pretty(evt)}</pre>
          ))}
          {!streamEvents.length && <div className="small">No events yet.</div>}
        </div>
      </div>
    </div>
  );
};

const SimpleConsole = () => {
  const [prompt, setPrompt] = useState("Ask Eleanor something ethically complicated...");
  const [events, setEvents] = useState([]);
  const [status, setStatus] = useState("idle");

  const run = () => {
    setEvents([]);
    setStatus("connecting");

    try {
      const ws = new WebSocket(wsUrl());
      ws.onopen = () => {
        setStatus("streaming");
        ws.send(JSON.stringify({ input: prompt, context: {} }));
      };
      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        setEvents((prev) => [...prev, msg]);
        if (msg.event === "final") {
          setStatus("done");
          ws.close();
        }
      };
      ws.onerror = (e) => {
        setEvents((prev) => [...prev, { event: "error", data: e.message || "ws error" }]);
        setStatus("error");
      };
      ws.onclose = () => {
        setStatus((prev) => (prev === "streaming" ? "closed" : prev));
      };
    } catch (err) {
      setEvents((prev) => [...prev, { event: "error", data: err.message }]);
      setStatus("error");
    }
  };

  return (
    <div className="grid two">
      <div className="panel">
        <h3>One-Click Stream</h3>
        <textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} />
        <div className="row" style={{ marginTop: 8 }}>
          <button onClick={run} disabled={status === "streaming" || status === "connecting"}>
            Stream
          </button>
          <span className="small">
            {status === "idle" && "Idle"}
            {status === "connecting" && "Connecting…"}
            {status === "streaming" && "Streaming…"}
            {status === "done" && "Done"}
            {status === "closed" && "Closed"}
            {status === "error" && "Error"}
          </span>
        </div>
        <p className="small" style={{ marginTop: 6 }}>
          Uses the same `/ws/deliberate` endpoint as the console above, with an empty context payload.
        </p>
      </div>
      <div className="panel">
        <h3>Events</h3>
        <div className="scroll">
          {events.map((evt, idx) => (
            <pre key={idx}>{pretty(evt)}</pre>
          ))}
          {!events.length && <div className="small">No events yet.</div>}
        </div>
      </div>
    </div>
  );
};

const TracePanel = () => {
  const [traceId, setTraceId] = useState("");
  const [trace, setTrace] = useState(null);
  const [rerun, setRerun] = useState(null);
  const [audit, setAudit] = useState(null);
  const [params, setParams] = useState({ critic: "", severity: "", searchTrace: "" });

  const loadTrace = async () => {
    setTrace(null);
    try {
      const data = await fetchJson(`/trace/${traceId}`);
      setTrace(data);
    } catch (err) {
      setTrace({ error: err.message });
    }
  };

  const replay = async () => {
    setRerun(null);
    try {
      const data = await fetchJson(`/replay/${traceId}?rerun=true`);
      setRerun(data);
    } catch (err) {
      setRerun({ error: err.message });
    }
  };

  const searchAudit = async () => {
    setAudit(null);
    const qs = new URLSearchParams();
    if (params.critic) qs.append("critic", params.critic);
    if (params.severity) qs.append("severity", params.severity);
    if (params.searchTrace) qs.append("trace_id", params.searchTrace);
    qs.append("limit", "50");
    try {
      const data = await fetchJson(`/audit/search?${qs.toString()}`);
      setAudit(data);
    } catch (err) {
      setAudit({ error: err.message });
    }
  };

  return (
    <div className="grid two">
      <div className="panel">
        <h3>Trace Lookup</h3>
        <input placeholder="trace id" value={traceId} onChange={(e) => setTraceId(e.target.value)} />
        <div className="row" style={{ marginTop: 8 }}>
          <button onClick={loadTrace} disabled={!traceId}>
            Load
          </button>
          <button onClick={replay} disabled={!traceId}>
            Replay with current engine
          </button>
        </div>
        <h4>Trace</h4>
        <pre>{trace ? pretty(trace) : "—"}</pre>
        <h4>Replay</h4>
        <pre>{rerun ? pretty(rerun) : "—"}</pre>
      </div>
      <div className="panel">
        <h3>Audit Search</h3>
        <div className="row">
          <input
            placeholder="critic (e.g. rights)"
            value={params.critic}
            onChange={(e) => setParams({ ...params, critic: e.target.value })}
          />
          <input
            placeholder="severity (e.g. S2)"
            value={params.severity}
            onChange={(e) => setParams({ ...params, severity: e.target.value })}
          />
        </div>
        <input
          style={{ marginTop: 8 }}
          placeholder="trace_id (optional)"
          value={params.searchTrace}
          onChange={(e) => setParams({ ...params, searchTrace: e.target.value })}
        />
        <div style={{ marginTop: 8 }}>
          <button onClick={searchAudit}>Search</button>
        </div>
        <h4>Results</h4>
        <pre>{audit ? pretty(audit) : "—"}</pre>
      </div>
    </div>
  );
};

const AdminPanel = () => {
  const [bindings, setBindings] = useState({});
  const [routerHealth, setRouterHealth] = useState(null);
  const [critics, setCritics] = useState([]);
  const [adapters, setAdapters] = useState([]);
  const [newBinding, setNewBinding] = useState({ critic: "", adapter: "" });

  const loadBindings = async () => {
    try {
      const data = await fetchJson("/admin/critics/bindings");
      setBindings(data.bindings || {});
      setCritics(data.critics || []);
      setAdapters(data.available_adapters || []);
      setNewBinding((prev) => ({
        critic: data.critics?.[0] || prev.critic || "",
        adapter: data.available_adapters?.[0] || prev.adapter || "",
      }));
    } catch (err) {
      setBindings({ error: err.message });
    }
  };

  const loadRouterHealth = async () => {
    try {
      const data = await fetchJson("/admin/router/health");
      setRouterHealth(data);
    } catch (err) {
      setRouterHealth({ error: err.message });
    }
  };

  const saveBinding = async () => {
    try {
      await fetchJson("/admin/critics/bindings", {
        method: "POST",
        body: JSON.stringify(newBinding),
      });
      await loadBindings();
    } catch (err) {
      alert(err.message);
    }
  };

  useEffect(() => {
    loadBindings();
    loadRouterHealth();
  }, []);

  return (
    <div className="grid two">
      <div className="panel">
        <h3>Critic Bindings</h3>
        <div className="row">
          <select
            value={newBinding.critic}
            onChange={(e) => setNewBinding({ ...newBinding, critic: e.target.value })}
          >
            {critics.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
          <select
            value={newBinding.adapter}
            onChange={(e) => setNewBinding({ ...newBinding, adapter: e.target.value })}
          >
            {adapters.map((a) => (
              <option key={a} value={a}>
                {a}
              </option>
            ))}
          </select>
          <button onClick={saveBinding} disabled={!newBinding.critic || !newBinding.adapter}>
            Save
          </button>
        </div>
        <p className="small" style={{ marginTop: 4 }}>
          Select a critic and bind it to a registered adapter (LLM backend). Current adapters are
          discovered from the router.
        </p>
        <h4>Current</h4>
        <pre>{bindings ? pretty(bindings) : "—"}</pre>
      </div>
      <div className="panel">
        <h3>Router Health</h3>
        <div className="row" style={{ marginBottom: 8 }}>
          <button onClick={loadRouterHealth}>Refresh</button>
        </div>
        <pre>{routerHealth ? pretty(routerHealth) : "—"}</pre>
      </div>
    </div>
  );
};

const MetricsPanel = () => {
  const grafanaUrl = useMemo(() => {
    const host = window.location.hostname || "localhost";
    const port = 3000;
    return `http://${host}:${port}/d/eleanor-obsv/eleanor-v8-observability?orgId=1&refresh=10s`;
  }, []);
  return (
    <div className="panel">
      <h3>Grafana</h3>
      <p className="small">Requires docker compose stack (Grafana at :3000) or adjust URL.</p>
      <iframe src={grafanaUrl} title="Grafana Dashboard"></iframe>
    </div>
  );
};

const App = () => {
  const [tab, setTab] = useState("deliberate");
  const tabs = [
    { id: "deliberate", label: "Deliberate" },
    { id: "simple", label: "Simple Stream" },
    { id: "traces", label: "Traces & Audit" },
    { id: "admin", label: "Admin" },
    { id: "metrics", label: "Metrics" },
  ];

  return (
    <div>
      <header>
        <div className="brand">ELEANOR V8 — Ops Console</div>
        <div className="pill">API Console</div>
      </header>
      <div className="container">
        <div className="tabs">
          {tabs.map((t) => (
            <div
              key={t.id}
              className={`tab ${tab === t.id ? "active" : ""}`}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </div>
          ))}
        </div>
        {tab === "deliberate" && <DeliberatePanel />}
        {tab === "simple" && <SimpleConsole />}
        {tab === "traces" && <TracePanel />}
        {tab === "admin" && <AdminPanel />}
        {tab === "metrics" && <MetricsPanel />}
      </div>
    </div>
  );
};

export default App;
