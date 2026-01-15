import React, { useCallback, useEffect, useMemo, useState } from "react";

const apiBase = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");
const wsBase = (import.meta.env.VITE_WS_BASE_URL || "").replace(/\/$/, "");
const tokenStorageKey = "eleanor_token";

const getStoredToken = () => {
  try {
    return localStorage.getItem(tokenStorageKey) || "";
  } catch {
    return "";
  }
};

const setStoredToken = (value) => {
  try {
    if (value) {
      localStorage.setItem(tokenStorageKey, value);
    } else {
      localStorage.removeItem(tokenStorageKey);
    }
  } catch {
    // Ignore storage errors (e.g., private mode)
  }
};

const apiUrl = (path) => (apiBase ? `${apiBase}${path}` : path);

const wsUrl = () => {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  const base = wsBase || `${proto}://${window.location.host}`;
  const url = base.endsWith("/ws/deliberate") ? base : `${base}/ws/deliberate`;
  const token = getStoredToken();
  if (!token) return url;
  const joiner = url.includes("?") ? "&" : "?";
  return `${url}${joiner}token=${encodeURIComponent(token)}`;
};

const buildHeaders = (extraHeaders = {}) => {
  const token = getStoredToken();
  return {
    "Content-Type": "application/json",
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
    ...extraHeaders,
  };
};

const fetchJson = async (path, opts = {}) => {
  const res = await fetch(apiUrl(path), {
    ...opts,
    headers: buildHeaders(opts.headers),
  });
  if (!res.ok) {
    const text = await res.text();
    const detail = text || res.statusText;
    let message = detail;
    if (res.status === 401) {
      message = `Authentication required. ${detail}`;
    } else if (res.status === 403) {
      message = `Forbidden. ${detail}`;
    }
    const err = new Error(message);
    err.status = res.status;
    throw err;
  }
  return await res.json();
};

const probeRole = async (path) => {
  const res = await fetch(apiUrl(path), { headers: buildHeaders() });
  if (res.ok) return "ok";
  if (res.status === 401) return "unauth";
  if (res.status === 403) return "forbidden";
  return "error";
};

const handleAuthError = (err, setAuthError) => {
  if (err && (err.status === 401 || err.status === 403)) {
    setAuthError(err.message);
    return true;
  }
  return false;
};

const AuthBanner = ({ authError, hint }) => {
  if (!authError) return null;
  return (
    <div
      style={{
        marginBottom: 12,
        padding: 10,
        borderRadius: 6,
        border: "1px solid #f87171",
        backgroundColor: "rgba(248, 113, 113, 0.12)",
        color: "#fecaca",
        fontSize: 13,
      }}
    >
      <div style={{ fontWeight: 600 }}>{authError}</div>
      {hint && <div style={{ marginTop: 4 }}>{hint}</div>}
    </div>
  );
};

const pretty = (obj) => JSON.stringify(obj, null, 2);
const truncate = (text, len = 240) => {
  if (!text) return "—";
  return text.length > len ? `${text.slice(0, len)}…` : text;
};

const DeliberatePanel = () => {
  const [input, setInput] = useState("Should I approve this loan application?");
  const [context, setContext] = useState('{"user_id":"demo","category":"finance"}');
  const [result, setResult] = useState(null);
  const [streamEvents, setStreamEvents] = useState([]);
  const [loading, setLoading] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [authError, setAuthError] = useState("");

  const run = async () => {
    setAuthError("");
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
      if (handleAuthError(err, setAuthError)) return;
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
        <AuthBanner
          authError={authError}
          hint="Requires an authenticated JWT. WebSocket uses the ?token= query param."
        />
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
  const [showRaw, setShowRaw] = useState(false);

  const finalData = useMemo(() => {
    const last = [...events].reverse().find((e) => e.event === "final");
    return last?.data || null;
  }, [events]);

  const modelInfo = useMemo(() => {
    const routerEvt = [...events].reverse().find((e) => e.event === "router_selected");
    return routerEvt?.data?.model_info || finalData?.model_info || null;
  }, [events, finalData]);

  const governanceDecisions = useMemo(() => {
    return events.filter((e) => ["governance", "governance_decision"].includes(e.event));
  }, [events]);

  const timeline = useMemo(
    () => events.map((e, idx) => `${idx + 1}. ${e.event || "event"}`).join(" • "),
    [events],
  );

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
        <h3>Result</h3>
        <div className="row" style={{ justifyContent: "space-between", alignItems: "center" }}>
          <div className="small">Timeline: {events.length ? timeline : "Waiting…"}</div>
          <button onClick={() => setShowRaw((v) => !v)} style={{ padding: "6px 10px" }}>
            {showRaw ? "Hide JSON" : "View JSON"}
          </button>
        </div>

        {!showRaw && (
          <div style={{ marginTop: 10, display: "grid", gap: 10 }}>
            {governanceDecisions.length > 0 && (
              <div style={{ marginBottom: 16, padding: 12, backgroundColor: "#f8f9fa", borderRadius: 4 }}>
                <h4 style={{ marginTop: 0, marginBottom: 8 }}>Streaming Governance Decisions</h4>
                {governanceDecisions.map((gd, idx) => {
                  const decision = gd.data?.decision || gd.data || gd.decision;
                  if (!decision) return null;
                  if (typeof decision !== "object") return null;
                  const signal = decision.signal || decision.assessment || decision.decision || decision.outcome;
                  const confidence = decision.confidence ?? decision.score ?? 0;
                  const stage = decision.stage || decision.phase || "unknown";
                  return (
                    <div
                      key={idx}
                      style={{
                        marginBottom: 8,
                        padding: 8,
                        border: "1px solid #ddd",
                        borderRadius: 4,
                        backgroundColor: "white",
                      }}
                    >
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                        <div>
                          <div style={{ fontWeight: 600 }}>
                            {signal?.replace("_", " ").toUpperCase() || "UNKNOWN"}
                          </div>
                          <div className="small" style={{ color: "#666" }}>
                            Stage: {stage} • Confidence: {(confidence * 100).toFixed(1)}%
                          </div>
                        </div>
                        <div
                          style={{
                            padding: "4px 8px",
                            borderRadius: 4,
                            fontSize: 11,
                            fontWeight: 600,
                            backgroundColor:
                              signal?.includes("ALLOW") || signal?.includes("allow")
                                ? "#d4edda"
                                : signal?.includes("DENY") || signal?.includes("deny")
                                ? "#f8d7da"
                                : "#fff3cd",
                            color:
                              signal?.includes("ALLOW") || signal?.includes("allow")
                                ? "#155724"
                                : signal?.includes("DENY") || signal?.includes("deny")
                                ? "#721c24"
                                : "#856404",
                          }}
                        >
                          {signal || "PENDING"}
                        </div>
                      </div>
                      {decision.rationale && (
                        <div className="small" style={{ marginTop: 4, color: "#666" }}>
                          {decision.rationale}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
            <div className="metric">
              <div className="label">Assessment</div>
              <div className="value">{finalData?.final_decision || "—"}</div>
            </div>
            <div className="metric">
              <div className="label">Model</div>
              <div className="value">
                {modelInfo?.model_name || "—"}
                {modelInfo?.model_version ? ` (${modelInfo.model_version})` : ""}
              </div>
            </div>
            <div className="metric">
              <div className="label">Output</div>
              <div className="value small">{truncate(finalData?.model_output)}</div>
            </div>
            <div className="metric">
              <div className="label">Aggregation</div>
              <div className="value small">
                {finalData?.aggregation?.decision || finalData?.aggregation?.final_output
                  ? `${finalData?.aggregation?.decision || ""} ${truncate(finalData?.aggregation?.final_output, 140)}`
                  : "—"}
              </div>
            </div>
            <div>
              <div className="label">Critics</div>
              {finalData?.critic_outputs ? (
                <div className="critics-grid">
                  {Object.entries(finalData.critic_outputs).map(([name, data]) => (
                    <div key={name} className="critic-card">
                      <div className="critic-name">{name}</div>
                      <div className="critic-detail small">
                        severity: {data.severity ?? data.score ?? "—"}
                      </div>
                      <div className="critic-detail small">{truncate(data.justification || data.concern, 120)}</div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="small">No critic outputs yet.</div>
              )}
            </div>
          </div>
        )}

        {showRaw && (
          <div className="scroll" style={{ marginTop: 10 }}>
            {events.map((evt, idx) => (
              <pre key={idx}>{pretty(evt)}</pre>
            ))}
            {!events.length && <div className="small">No events yet.</div>}
          </div>
        )}
      </div>
    </div>
  );
};

const TracePanel = () => {
  const [traceId, setTraceId] = useState("");
  const [trace, setTrace] = useState(null);
  const [rerun, setRerun] = useState(null);
  const [audit, setAudit] = useState(null);
  const [params, setParams] = useState({ query: "", decision: "" });
  const [authError, setAuthError] = useState("");

  const loadTrace = async () => {
    setAuthError("");
    setTrace(null);
    try {
      const data = await fetchJson(`/audit/trace/${traceId}`);
      setTrace(data);
    } catch (err) {
      if (handleAuthError(err, setAuthError)) return;
      setTrace({ error: err.message });
    }
  };

  const replay = async () => {
    setAuthError("");
    setRerun(null);
    try {
      const data = await fetchJson(`/audit/replay/${traceId}`, { method: "POST" });
      setRerun(data);
    } catch (err) {
      if (handleAuthError(err, setAuthError)) return;
      setRerun({ error: err.message });
    }
  };

  const searchAudit = async () => {
    setAuthError("");
    setAudit(null);
    try {
      const payload = {
        query: params.query || undefined,
        decision: params.decision || undefined,
        limit: 50,
      };
      const data = await fetchJson("/audit/search", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      setAudit(data);
    } catch (err) {
      if (handleAuthError(err, setAuthError)) return;
      setAudit({ error: err.message });
    }
  };

  return (
    <div className="grid two">
      <div className="panel">
        <h3>Trace Lookup</h3>
        <AuthBanner authError={authError} hint="Audit trace access requires a valid JWT." />
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
            placeholder="search text (input or trace id)"
            value={params.query}
            onChange={(e) => setParams({ ...params, query: e.target.value })}
          />
          <input
            placeholder="decision (allow/deny/escalate)"
            value={params.decision}
            onChange={(e) => setParams({ ...params, decision: e.target.value })}
          />
        </div>
        <div style={{ marginTop: 8 }}>
          <button onClick={searchAudit}>Search</button>
        </div>
        <h4>Results</h4>
        <pre>{audit ? pretty(audit) : "—"}</pre>
      </div>
    </div>
  );
};

const AuditLedgerPanel = () => {
  const [filters, setFilters] = useState({
    query: "",
    event: "",
    actorId: "",
    traceId: "",
    startTime: "",
    endTime: "",
  });
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [authError, setAuthError] = useState("");

  const update = (key, value) => setFilters((prev) => ({ ...prev, [key]: value }));

  const runQuery = async () => {
    setAuthError("");
    setLoading(true);
    setError(null);
    try {
      const payload = {
        search_text: filters.query || undefined,
        event_types: filters.event ? [filters.event] : undefined,
        user: filters.actorId || undefined,
        request_id: filters.traceId || undefined,
        start_time: filters.startTime || undefined,
        end_time: filters.endTime || undefined,
        limit: 200,
        offset: 0,
      };
      const data = await fetchJson("/audit/query", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      setResults(data.events || []);
    } catch (err) {
      if (handleAuthError(err, setAuthError)) return;
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const exportData = async (format = "jsonl") => {
    setAuthError("");
    try {
      const payload = {
        search_text: filters.query || undefined,
        event_types: filters.event ? [filters.event] : undefined,
        user: filters.actorId || undefined,
        request_id: filters.traceId || undefined,
        start_time: filters.startTime || undefined,
        end_time: filters.endTime || undefined,
        limit: 10000,
        offset: 0,
      };
      const res = await fetch(`${apiUrl("/audit/export")}?format=${format}`, {
        method: "POST",
        headers: buildHeaders(),
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const detail = await res.text();
        let message = detail;
        if (res.status === 401) {
          message = `Authentication required. ${detail}`;
        } else if (res.status === 403) {
          message = `Forbidden. ${detail}`;
        }
        const err = new Error(message);
        err.status = res.status;
        throw err;
      }
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `audit_export.${format === "csv" ? "csv" : "jsonl"}`;
      link.click();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      if (handleAuthError(err, setAuthError)) return;
      alert(`Export failed: ${err.message}`);
    }
  };

  return (
    <div className="panel">
      <h3>Audit Ledger Query</h3>
      <AuthBanner authError={authError} hint="Audit ledger queries require authentication." />
      <p className="small">
        Query immutable audit ledger entries with optional filters. Times should be ISO8601 (e.g.
        2025-05-01T00:00:00Z).
      </p>
      <div className="grid two" style={{ gap: 8 }}>
        <input
          placeholder="Free-text search"
          value={filters.query}
          onChange={(e) => update("query", e.target.value)}
        />
        <input
          placeholder="Event (e.g. secret_access_audit)"
          value={filters.event}
          onChange={(e) => update("event", e.target.value)}
        />
        <input
          placeholder="Actor ID"
          value={filters.actorId}
          onChange={(e) => update("actorId", e.target.value)}
        />
        <input
          placeholder="Trace ID"
          value={filters.traceId}
          onChange={(e) => update("traceId", e.target.value)}
        />
        <input
          placeholder="Start time (ISO)"
          value={filters.startTime}
          onChange={(e) => update("startTime", e.target.value)}
        />
        <input
          placeholder="End time (ISO)"
          value={filters.endTime}
          onChange={(e) => update("endTime", e.target.value)}
        />
      </div>
      <div className="row" style={{ marginTop: 8, gap: 8 }}>
        <button onClick={runQuery} disabled={loading}>
          {loading ? "Searching…" : "Search Ledger"}
        </button>
        <button onClick={() => exportData("csv")} disabled={loading}>
          Export CSV
        </button>
        <button onClick={() => exportData("jsonl")} disabled={loading}>
          Export JSONL
        </button>
      </div>
      {error && (
        <div className="small" style={{ color: "red", marginTop: 8 }}>
          {error}
        </div>
      )}
      <div style={{ marginTop: 12 }}>
        <div className="row" style={{ justifyContent: "space-between", marginBottom: 4 }}>
          <h4 style={{ margin: 0 }}>Results</h4>
          <div className="small">{results.length} record(s)</div>
        </div>
        {results.length === 0 ? (
          <div className="small">No records found.</div>
        ) : (
          <div className="scroll" style={{ maxHeight: 360 }}>
            <table className="table">
              <thead>
                <tr>
                  <th style={{ width: "18%" }}>Timestamp</th>
                  <th style={{ width: "18%" }}>Event</th>
                  <th style={{ width: "18%" }}>Trace</th>
                  <th style={{ width: "18%" }}>Actor</th>
                  <th>Payload</th>
                </tr>
              </thead>
              <tbody>
                {results.map((row, idx) => (
                  <tr key={`${row.event_id || row.record_hash || idx}-${idx}`}>
                    <td className="small">{row.timestamp || "—"}</td>
                    <td className="small">{row.event || row.event_type || row.event_id || "—"}</td>
                    <td className="small">{row.trace_id || "—"}</td>
                    <td className="small">{row.actor_id || "—"}</td>
                    <td className="small">{truncate(JSON.stringify(row.payload || {}), 120)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};

const GovernancePanel = () => {
  const [authError, setAuthError] = useState("");
  const [previewInput, setPreviewInput] = useState(
    pretty({
      critics: { rights: { severity: 0.5 } },
      aggregator: { decision: "allow" },
      precedent: { alignment_score: 0.8 },
      uncertainty: { overall_uncertainty: 0.2 },
    }),
  );
  const [previewResult, setPreviewResult] = useState(null);
  const [previewError, setPreviewError] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [explainTrace, setExplainTrace] = useState("");
  const [explainDetail, setExplainDetail] = useState("summary");
  const [reviewCase, setReviewCase] = useState("");
  const [reviewMetrics, setReviewMetrics] = useState(null);
  const [quarantine, setQuarantine] = useState(null);

  const runPreview = async () => {
    setAuthError("");
    setPreviewError(null);
    setPreviewResult(null);
    try {
      const payload = JSON.parse(previewInput);
      const data = await fetchJson("/governance/preview", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      setPreviewResult(data);
    } catch (err) {
      if (handleAuthError(err, setAuthError)) return;
      setPreviewError(err.message);
    }
  };

  const loadExplanation = async () => {
    setAuthError("");
    setExplanation(null);
    if (!explainTrace) {
      setExplanation({ error: "Trace ID required" });
      return;
    }
    try {
      const data = await fetchJson(`/explanation/${explainTrace}?detail_level=${explainDetail}`);
      setExplanation(data);
    } catch (err) {
      if (handleAuthError(err, setAuthError)) return;
      setExplanation({ error: err.message });
    }
  };

  const loadReviewMetrics = async () => {
    setAuthError("");
    setReviewMetrics(null);
    if (!reviewCase) {
      setReviewMetrics({ error: "Case ID required" });
      return;
    }
    try {
      const data = await fetchJson(`/governance/review/metrics/${reviewCase}`);
      setReviewMetrics(data);
    } catch (err) {
      if (handleAuthError(err, setAuthError)) return;
      setReviewMetrics({ error: err.message });
    }
  };

  const loadQuarantine = async () => {
    setAuthError("");
    setQuarantine(null);
    try {
      const data = await fetchJson("/governance/review/quarantine");
      setQuarantine(data);
    } catch (err) {
      if (handleAuthError(err, setAuthError)) return;
      setQuarantine({ error: err.message });
    }
  };

  return (
    <div className="grid two">
      <div className="panel">
        <h3>Governance Preview (OPA)</h3>
        <AuthBanner
          authError={authError}
          hint="Preview requires authentication. Review metrics/quarantine require the reviewer role."
        />
        <p className="small">
          Run OPA governance evaluation on a mock evidence bundle without executing the full pipeline.
        </p>
        <label className="small">Preview Payload (JSON)</label>
        <textarea value={previewInput} onChange={(e) => setPreviewInput(e.target.value)} />
        <div className="row" style={{ marginTop: 8 }}>
          <button onClick={runPreview}>Run Preview</button>
        </div>
        {previewError && (
          <div className="small" style={{ color: "red", marginTop: 8 }}>
            {previewError}
          </div>
        )}
        <h4>Result</h4>
        <pre>{previewResult ? pretty(previewResult) : "—"}</pre>
      </div>
      <div className="panel">
        <h3>Explainable Governance</h3>
        <AuthBanner
          authError={authError}
          hint="Explainable governance requires authentication and feature flag enablement."
        />
        <div className="row" style={{ marginBottom: 8 }}>
          <input
            placeholder="trace id"
            value={explainTrace}
            onChange={(e) => setExplainTrace(e.target.value)}
            style={{ flex: 1 }}
          />
          <select value={explainDetail} onChange={(e) => setExplainDetail(e.target.value)}>
            <option value="summary">summary</option>
            <option value="detailed">detailed</option>
            <option value="interactive">interactive</option>
          </select>
          <button onClick={loadExplanation} disabled={!explainTrace}>
            Load
          </button>
        </div>
        <pre>{explanation ? pretty(explanation) : "—"}</pre>
        <h4>Review Metrics</h4>
        <div className="row" style={{ marginBottom: 8 }}>
          <input
            placeholder="case id"
            value={reviewCase}
            onChange={(e) => setReviewCase(e.target.value)}
            style={{ flex: 1 }}
          />
          <button onClick={loadReviewMetrics} disabled={!reviewCase}>
            Fetch Metrics
          </button>
        </div>
        <pre>{reviewMetrics ? pretty(reviewMetrics) : "—"}</pre>
        <h4>Quarantine Queue</h4>
        <div className="row" style={{ marginBottom: 8 }}>
          <button onClick={loadQuarantine}>Refresh</button>
        </div>
        <pre>{quarantine ? pretty(quarantine) : "—"}</pre>
      </div>
    </div>
  );
};

const HumanReviewPanel = () => {
  const [authError, setAuthError] = useState("");
  const buildReviewTemplate = () => ({
    review_id: (typeof crypto !== "undefined" && crypto.randomUUID && crypto.randomUUID()) || `review_${Date.now()}`,
    case_id: "",
    reviewer_role: "reviewer",
    timestamp: new Date().toISOString(),
    coverage_issues: [],
    severity_assessment: { original: 0.0, adjusted: null, justification: "" },
    dissent_evaluation: { present: false, preserved: false, notes: "" },
    uncertainty_adequate: true,
    outcome: "affirmed",
    reviewer_justification: "",
  });

  const [reviewId, setReviewId] = useState("");
  const [caseId, setCaseId] = useState("");
  const [lane, setLane] = useState("precedent_candidate");
  const [reviewJson, setReviewJson] = useState(pretty(buildReviewTemplate()));
  const [reviewResult, setReviewResult] = useState(null);
  const [pending, setPending] = useState(null);
  const [stats, setStats] = useState(null);
  const [caseReviews, setCaseReviews] = useState(null);
  const [laneContents, setLaneContents] = useState(null);
  const [validation, setValidation] = useState(null);

  const submitReview = async () => {
    setAuthError("");
    setReviewResult(null);
    try {
      const payload = JSON.parse(reviewJson);
      const data = await fetchJson("/review/submit", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      setReviewResult(data);
    } catch (err) {
      if (handleAuthError(err, setAuthError)) return;
      setReviewResult({ error: err.message });
    }
  };

  const loadPending = async () => {
    setAuthError("");
    setPending(null);
    try {
      const data = await fetchJson("/review/pending");
      setPending(data);
    } catch (err) {
      if (handleAuthError(err, setAuthError)) return;
      setPending({ error: err.message });
    }
  };

  const loadStats = async () => {
    setAuthError("");
    setStats(null);
    const qs = caseId ? `?case_id=${encodeURIComponent(caseId)}` : "";
    try {
      const data = await fetchJson(`/review/stats${qs}`);
      setStats(data);
    } catch (err) {
      if (handleAuthError(err, setAuthError)) return;
      setStats({ error: err.message });
    }
  };

  const loadCaseReviews = async () => {
    setAuthError("");
    setCaseReviews(null);
    if (!caseId) {
      setCaseReviews({ error: "Case ID required" });
      return;
    }
    try {
      const data = await fetchJson(`/review/case/${caseId}`);
      setCaseReviews(data);
    } catch (err) {
      if (handleAuthError(err, setAuthError)) return;
      setCaseReviews({ error: err.message });
    }
  };

  const loadReview = async () => {
    setAuthError("");
    setReviewResult(null);
    if (!reviewId) {
      setReviewResult({ error: "Review ID required" });
      return;
    }
    try {
      const data = await fetchJson(`/review/get/${reviewId}`);
      setReviewResult(data);
    } catch (err) {
      if (handleAuthError(err, setAuthError)) return;
      setReviewResult({ error: err.message });
    }
  };

  const loadLane = async () => {
    setAuthError("");
    setLaneContents(null);
    try {
      const data = await fetchJson(`/review/lane/${lane}`);
      setLaneContents(data);
    } catch (err) {
      if (handleAuthError(err, setAuthError)) return;
      setLaneContents({ error: err.message });
    }
  };

  const validateCase = async () => {
    setAuthError("");
    setValidation(null);
    if (!caseId) {
      setValidation({ error: "Case ID required" });
      return;
    }
    try {
      const data = await fetchJson(`/review/validate/${caseId}`);
      setValidation(data);
    } catch (err) {
      if (handleAuthError(err, setAuthError)) return;
      setValidation({ error: err.message });
    }
  };

  return (
    <div className="grid two">
      <div className="panel">
        <h3>Submit Human Review</h3>
        <AuthBanner authError={authError} hint="Reviewer role required for all review operations." />
        <p className="small">
          Reviews require the reviewer role. Provide a full HumanReviewRecord payload.
        </p>
        <textarea value={reviewJson} onChange={(e) => setReviewJson(e.target.value)} />
        <div className="row" style={{ marginTop: 8 }}>
          <button onClick={submitReview}>Submit Review</button>
          <button onClick={() => setReviewJson(pretty(buildReviewTemplate()))}>Reset Template</button>
        </div>
        <h4>Result</h4>
        <pre>{reviewResult ? pretty(reviewResult) : "—"}</pre>
      </div>
      <div className="panel">
        <h3>Review Operations</h3>
        <AuthBanner authError={authError} hint="Reviewer role required for all review operations." />
        <div className="row" style={{ marginBottom: 8 }}>
          <button onClick={loadPending}>Load Pending</button>
          <button onClick={loadStats}>Load Stats</button>
        </div>
        <div className="row" style={{ marginBottom: 8 }}>
          <input
            placeholder="case id"
            value={caseId}
            onChange={(e) => setCaseId(e.target.value)}
            style={{ flex: 1 }}
          />
          <button onClick={loadCaseReviews} disabled={!caseId}>
            Case Reviews
          </button>
          <button onClick={validateCase} disabled={!caseId}>
            Validate Case
          </button>
        </div>
        <div className="row" style={{ marginBottom: 8 }}>
          <input
            placeholder="review id"
            value={reviewId}
            onChange={(e) => setReviewId(e.target.value)}
            style={{ flex: 1 }}
          />
          <button onClick={loadReview} disabled={!reviewId}>
            Load Review
          </button>
        </div>
        <div className="row" style={{ marginBottom: 8 }}>
          <select value={lane} onChange={(e) => setLane(e.target.value)}>
            <option value="precedent_candidate">precedent_candidate</option>
            <option value="training_calibration">training_calibration</option>
            <option value="policy_insight">policy_insight</option>
            <option value="quarantine">quarantine</option>
          </select>
          <button onClick={loadLane}>Load Lane</button>
        </div>
        <h4>Pending</h4>
        <pre>{pending ? pretty(pending) : "—"}</pre>
        <h4>Stats</h4>
        <pre>{stats ? pretty(stats) : "—"}</pre>
        <h4>Case Reviews</h4>
        <pre>{caseReviews ? pretty(caseReviews) : "—"}</pre>
        <h4>Lane Contents</h4>
        <pre>{laneContents ? pretty(laneContents) : "—"}</pre>
        <h4>Validation</h4>
        <pre>{validation ? pretty(validation) : "—"}</pre>
      </div>
    </div>
  );
};

const FeatureFlagsPanel = ({ adminWriteEnabled, onAuthError }) => {
  const [flags, setFlags] = useState({
    explainable_governance: false,
    semantic_cache: false,
    intelligent_model_selection: false,
    anomaly_detection: false,
    streaming_governance: false,
    adaptive_critic_weighting: false,
    temporal_precedent_evolution: false,
    reflection: true,
    drift_check: true,
    precedent_analysis: true,
  });
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);

  const loadFlags = async () => {
    setLoading(true);
    try {
      const data = await fetchJson("/admin/feature-flags");
      setFlags(data);
    } catch (err) {
      if (onAuthError && handleAuthError(err, onAuthError)) {
        setLoading(false);
        return;
      }
      alert(`Failed to load feature flags: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const saveFlags = async () => {
    setSaving(true);
    try {
      const data = await fetchJson("/admin/feature-flags", {
        method: "POST",
        body: JSON.stringify(flags),
      });
      setFlags(data);
      alert("Feature flags updated successfully!");
    } catch (err) {
      if (onAuthError && handleAuthError(err, onAuthError)) {
        setSaving(false);
        return;
      }
      alert(`Failed to save feature flags: ${err.message}`);
    } finally {
      setSaving(false);
    }
  };

  const toggleFlag = (key) => {
    setFlags((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  useEffect(() => {
    loadFlags();
  }, []);

  const featureDescriptions = {
    explainable_governance: "Explainable Governance: Provides detailed explanations of governance decisions with causal reasoning",
    semantic_cache: "Semantic Cache: Cache based on semantic similarity (3-5x better hit rates)",
    intelligent_model_selection: "Intelligent Model Selection: Automatically select optimal models for cost/latency/quality",
    anomaly_detection: "Anomaly Detection: Proactively detect unusual system behavior using ML",
    streaming_governance: "Streaming Governance: Real-time governance decisions via WebSocket",
    adaptive_critic_weighting: "Adaptive Critic Weighting: Learn optimal critic weights from historical decisions",
    temporal_precedent_evolution: "Temporal Precedent Evolution: Track precedent lifecycle and detect drift",
    reflection: "Reflection: Enable reflection and uncertainty analysis",
    drift_check: "Drift Check: Enable drift detection",
    precedent_analysis: "Precedent Analysis: Enable precedent-based reasoning",
  };

  return (
    <div className="panel">
      <div className="row" style={{ justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
        <h3 style={{ margin: 0 }}>Feature Flags</h3>
        <div className="row" style={{ gap: 8 }}>
          <button onClick={loadFlags} disabled={loading || saving} style={{ padding: "6px 12px" }}>
            {loading ? "Loading..." : "Refresh"}
          </button>
          <button
            onClick={saveFlags}
            disabled={loading || saving || adminWriteEnabled === false}
            style={{ padding: "6px 12px" }}
            title={adminWriteEnabled === false ? "Admin writes are disabled" : ""}
          >
            {saving ? "Saving..." : "Save Changes"}
          </button>
        </div>
      </div>
      <p className="small" style={{ marginBottom: 16 }}>
        Enable or disable optional features. Changes take effect immediately but require server restart for persistence.
      </p>
      {adminWriteEnabled === false && (
        <div className="small" style={{ color: "#f87171", marginBottom: 12 }}>
          Admin writes are disabled. Set `ELEANOR_ENABLE_ADMIN_WRITE=true` to enable updates.
        </div>
      )}
      <div style={{ display: "grid", gap: 12 }}>
        {Object.entries(flags).map(([key, value]) => (
          <div key={key} style={{ display: "flex", alignItems: "flex-start", gap: 12, padding: 12, border: "1px solid #ddd", borderRadius: 4 }}>
            <label style={{ display: "flex", alignItems: "center", gap: 8, cursor: "pointer", flex: 1 }}>
              <input
                type="checkbox"
                checked={value}
                onChange={() => toggleFlag(key)}
                disabled={loading || saving}
                style={{ width: 18, height: 18, cursor: "pointer" }}
              />
              <div>
                <div style={{ fontWeight: 500, marginBottom: 4 }}>
                  {key.split("_").map((word) => word.charAt(0).toUpperCase() + word.slice(1)).join(" ")}
                </div>
                <div className="small" style={{ color: "#666" }}>
                  {featureDescriptions[key] || "Feature description not available"}
                </div>
              </div>
            </label>
            <div
              style={{
                padding: "4px 8px",
                borderRadius: 4,
                fontSize: 12,
                fontWeight: 500,
                backgroundColor: value ? "#d4edda" : "#f8d7da",
                color: value ? "#155724" : "#721c24",
              }}
            >
              {value ? "ON" : "OFF"}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

const TemporalEvolutionPanel = ({ onAuthError }) => {
  const [analytics, setAnalytics] = useState(null);
  const [caseId, setCaseId] = useState("");
  const [driftInfo, setDriftInfo] = useState(null);
  const [recommendations, setRecommendations] = useState(null);
  const [loading, setLoading] = useState(false);

  const loadAnalytics = async (cid = null) => {
    setLoading(true);
    try {
      const url = cid ? `/admin/precedent-evolution/analytics?case_id=${cid}` : "/admin/precedent-evolution/analytics";
      const data = await fetchJson(url);
      setAnalytics(data);
    } catch (err) {
      if (onAuthError && handleAuthError(err, onAuthError)) {
        setLoading(false);
        return;
      }
      setAnalytics({ error: err.message });
    } finally {
      setLoading(false);
    }
  };

  const loadDrift = async () => {
    if (!caseId) return;
    setLoading(true);
    try {
      const data = await fetchJson(`/admin/precedent-evolution/${caseId}/drift`);
      setDriftInfo(data);
    } catch (err) {
      if (onAuthError && handleAuthError(err, onAuthError)) {
        setLoading(false);
        return;
      }
      setDriftInfo({ error: err.message });
    } finally {
      setLoading(false);
    }
  };

  const loadRecommendations = async () => {
    setLoading(true);
    try {
      const data = await fetchJson("/admin/precedent-evolution/recommendations?min_versions=3");
      setRecommendations(data);
    } catch (err) {
      if (onAuthError && handleAuthError(err, onAuthError)) {
        setLoading(false);
        return;
      }
      setRecommendations({ error: err.message });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadAnalytics();
  }, []);

  return (
    <div className="grid two">
      <div className="panel">
        <h3>Temporal Evolution Analytics</h3>
        <div className="row" style={{ marginBottom: 8 }}>
          <button onClick={() => loadAnalytics()} disabled={loading}>
            Refresh
          </button>
          <input
            placeholder="Case ID (optional)"
            value={caseId}
            onChange={(e) => setCaseId(e.target.value)}
            style={{ flex: 1, padding: "6px 8px" }}
          />
          <button onClick={() => loadAnalytics(caseId)} disabled={loading || !caseId}>
            Load Case
          </button>
        </div>
        {analytics && (
          <div>
            {analytics.error ? (
              <div className="small" style={{ color: "red" }}>Error: {analytics.error}</div>
            ) : (
              <div>
                {analytics.case_id ? (
                  <div>
                    <div className="metric">
                      <div className="label">Case ID</div>
                      <div className="value">{analytics.case_id}</div>
                    </div>
                    <div className="metric">
                      <div className="label">Lifecycle State</div>
                      <div className="value">{analytics.lifecycle_state}</div>
                    </div>
                    <div className="metric">
                      <div className="label">Version Count</div>
                      <div className="value">{analytics.version_count}</div>
                    </div>
                    {analytics.drift_metrics && (
                      <div className="metric">
                        <div className="label">Drift Score</div>
                        <div className="value">{analytics.drift_metrics.drift_score?.toFixed(3) || "N/A"}</div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div>
                    <div className="metric">
                      <div className="label">Total Precedents</div>
                      <div className="value">{analytics.total_precedents || 0}</div>
                    </div>
                    <div className="metric">
                      <div className="label">Average Versions</div>
                      <div className="value">{analytics.average_versions_per_precedent?.toFixed(2) || 0}</div>
                    </div>
                    {analytics.lifecycle_distribution && (
                      <div>
                        <h4>Lifecycle Distribution</h4>
                        <pre>{pretty(analytics.lifecycle_distribution)}</pre>
                      </div>
                    )}
                    {analytics.drift_distribution && (
                      <div>
                        <h4>Drift Distribution</h4>
                        <pre>{pretty(analytics.drift_distribution)}</pre>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        )}
        <h4>Drift Detection</h4>
        <div className="row" style={{ marginBottom: 8 }}>
          <input
            placeholder="Enter case ID"
            value={caseId}
            onChange={(e) => setCaseId(e.target.value)}
            style={{ flex: 1, padding: "6px 8px" }}
          />
          <button onClick={loadDrift} disabled={loading || !caseId}>
            Detect Drift
          </button>
        </div>
        {driftInfo && (
          <pre style={{ fontSize: 12, maxHeight: 200, overflow: "auto" }}>
            {pretty(driftInfo)}
          </pre>
        )}
        <h4>Deprecation Recommendations</h4>
        <button onClick={loadRecommendations} disabled={loading} style={{ marginBottom: 8 }}>
          Get Recommendations
        </button>
        {recommendations && (
          <div style={{ maxHeight: 300, overflow: "auto" }}>
            {recommendations.error ? (
              <div className="small" style={{ color: "red" }}>Error: {recommendations.error}</div>
            ) : (
              <div>
                <div className="small">Found {recommendations.count || 0} recommendations</div>
                {recommendations.recommendations?.map((rec, idx) => (
                  <div key={idx} style={{ marginTop: 8, padding: 8, border: "1px solid #ddd", borderRadius: 4 }}>
                    <div style={{ fontWeight: 500 }}>Case: {rec.case_id}</div>
                    <div className="small">Reason: {rec.reason}</div>
                    <div className="small">Drift Score: {rec.drift_metrics?.drift_score?.toFixed(3)}</div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
      <div className="panel">
        <h3>Evolution Details</h3>
        {analytics && analytics.latest_version && (
          <div>
            <h4>Latest Version</h4>
            <pre style={{ fontSize: 12, maxHeight: 400, overflow: "auto" }}>
              {pretty(analytics.latest_version)}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
};

const AdaptiveWeightingPanel = ({ adminWriteEnabled, onAuthError }) => {
  const [performance, setPerformance] = useState(null);
  const [weights, setWeights] = useState(null);
  const [loading, setLoading] = useState(false);
  const [updating, setUpdating] = useState(false);

  const loadPerformance = async () => {
    setLoading(true);
    try {
      const data = await fetchJson("/admin/adaptive-weighting/performance");
      setPerformance(data);
    } catch (err) {
      if (onAuthError && handleAuthError(err, onAuthError)) {
        setLoading(false);
        return;
      }
      setPerformance({ error: err.message });
    } finally {
      setLoading(false);
    }
  };

  const loadWeights = async () => {
    setLoading(true);
    try {
      const data = await fetchJson("/admin/adaptive-weighting/weights");
      setWeights(data);
    } catch (err) {
      if (onAuthError && handleAuthError(err, onAuthError)) {
        setLoading(false);
        return;
      }
      setWeights({ error: err.message });
    } finally {
      setLoading(false);
    }
  };

  const updateWeights = async () => {
    setUpdating(true);
    try {
      const data = await fetchJson("/admin/adaptive-weighting/update", { method: "POST" });
      alert(`Updated ${data.updates_count || 0} weights`);
      await loadWeights();
      await loadPerformance();
    } catch (err) {
      if (onAuthError && handleAuthError(err, onAuthError)) {
        setUpdating(false);
        return;
      }
      alert(`Failed to update weights: ${err.message}`);
    } finally {
      setUpdating(false);
    }
  };

  const resetWeights = async () => {
    if (!confirm("Reset all weights to uniform (1.0)? This cannot be undone.")) return;
    setUpdating(true);
    try {
      await fetchJson("/admin/adaptive-weighting/reset", { method: "POST" });
      alert("Weights reset successfully");
      await loadWeights();
      await loadPerformance();
    } catch (err) {
      if (onAuthError && handleAuthError(err, onAuthError)) {
        setUpdating(false);
        return;
      }
      alert(`Failed to reset weights: ${err.message}`);
    } finally {
      setUpdating(false);
    }
  };

  useEffect(() => {
    loadWeights();
    loadPerformance();
  }, []);

  return (
    <div className="grid two">
      <div className="panel">
        <h3>Critic Weights</h3>
        <div className="row" style={{ marginBottom: 8 }}>
          <button onClick={loadWeights} disabled={loading}>
            Refresh
          </button>
          <button
            onClick={updateWeights}
            disabled={updating || loading || adminWriteEnabled === false}
            title={adminWriteEnabled === false ? "Admin writes are disabled" : ""}
          >
            {updating ? "Updating..." : "Update Weights"}
          </button>
          <button
            onClick={resetWeights}
            disabled={updating || loading || adminWriteEnabled === false}
            style={{ backgroundColor: "#dc3545" }}
            title={adminWriteEnabled === false ? "Admin writes are disabled" : ""}
          >
            Reset
          </button>
        </div>
        {adminWriteEnabled === false && (
          <div className="small" style={{ color: "#f87171", marginBottom: 8 }}>
            Admin writes are disabled. Set `ELEANOR_ENABLE_ADMIN_WRITE=true` to enable updates.
          </div>
        )}
        {weights && (
          <div>
            {weights.error ? (
              <div className="small" style={{ color: "red" }}>Error: {weights.error}</div>
            ) : (
              <div>
                {weights.weights && Object.entries(weights.weights).map(([name, weight]) => (
                  <div key={name} style={{ marginBottom: 8, padding: 8, border: "1px solid #ddd", borderRadius: 4 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                      <span style={{ fontWeight: 500 }}>{name}</span>
                      <span style={{ fontSize: 18, fontWeight: 600 }}>{weight.toFixed(3)}</span>
                    </div>
                    <div
                      style={{
                        marginTop: 4,
                        height: 8,
                        backgroundColor: "#e0e0e0",
                        borderRadius: 4,
                        overflow: "hidden",
                      }}
                    >
                      <div
                        style={{
                          height: "100%",
                          width: `${(weight / 2.0) * 100}%`,
                          backgroundColor: weight > 1.2 ? "#28a745" : weight < 0.8 ? "#dc3545" : "#ffc107",
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
      <div className="panel">
        <h3>Performance Metrics</h3>
        <button onClick={loadPerformance} disabled={loading} style={{ marginBottom: 8 }}>
          Refresh
        </button>
        {performance && (
          <div>
            {performance.error ? (
              <div className="small" style={{ color: "red" }}>Error: {performance.error}</div>
            ) : (
              <div>
                {performance.summary && (
                  <div style={{ marginBottom: 16 }}>
                    <div className="metric">
                      <div className="label">Total Feedback Samples</div>
                      <div className="value">{performance.summary.total_feedback_samples || 0}</div>
                    </div>
                    <div className="metric">
                      <div className="label">Weight Updates</div>
                      <div className="value">{performance.summary.total_weight_updates || 0}</div>
                    </div>
                    <div className="metric">
                      <div className="label">Critics Tracked</div>
                      <div className="value">{performance.summary.critics_tracked || 0}</div>
                    </div>
                  </div>
                )}
                {performance.critic_metrics && (
                  <div>
                    <h4>Critic Performance</h4>
                    <div style={{ maxHeight: 400, overflow: "auto" }}>
                      {Object.entries(performance.critic_metrics).map(([name, metrics]) => (
                        <div
                          key={name}
                          style={{ marginBottom: 12, padding: 12, border: "1px solid #ddd", borderRadius: 4 }}
                        >
                          <div style={{ fontWeight: 600, marginBottom: 8 }}>{name}</div>
                          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, fontSize: 12 }}>
                            <div>
                              <div className="small">Weight</div>
                              <div>{metrics.weight?.toFixed(3)}</div>
                            </div>
                            <div>
                              <div className="small">Accuracy</div>
                              <div>{(metrics.accuracy * 100).toFixed(1)}%</div>
                            </div>
                            <div>
                              <div className="small">F1 Score</div>
                              <div>{metrics.f1_score?.toFixed(3)}</div>
                            </div>
                            <div>
                              <div className="small">Evaluations</div>
                              <div>{metrics.total_evaluations || 0}</div>
                            </div>
                            <div>
                              <div className="small">False Positives</div>
                              <div>{metrics.false_positives || 0}</div>
                            </div>
                            <div>
                              <div className="small">False Negatives</div>
                              <div>{metrics.false_negatives || 0}</div>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

const AdminPanel = () => {
  const [bindings, setBindings] = useState({});
  const [routerHealth, setRouterHealth] = useState(null);
  const [critics, setCritics] = useState([]);
  const [adapters, setAdapters] = useState([]);
  const [adminWriteState, setAdminWriteState] = useState({ enabled: null, error: null });
  const [authError, setAuthError] = useState("");
  const [newBinding, setNewBinding] = useState({ critic: "", adapter: "" });
  const [adapterForm, setAdapterForm] = useState({
    name: "ollama-phi3",
    type: "ollama",
    model: "phi3",
    device: "cpu",
    apiKey: "",
  });
  const [savingAdapter, setSavingAdapter] = useState(false);
  const [advancedTab, setAdvancedTab] = useState("evolution");

  const loadBindings = async () => {
    setAuthError("");
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
      if (handleAuthError(err, setAuthError)) return;
      setBindings({ error: err.message });
    }
  };

  const loadRouterHealth = async () => {
    setAuthError("");
    try {
      const data = await fetchJson("/admin/router/health");
      setRouterHealth(data);
    } catch (err) {
      if (handleAuthError(err, setAuthError)) return;
      setRouterHealth({ error: err.message });
    }
  };

  const loadAdminWriteEnabled = async () => {
    setAuthError("");
    try {
      const data = await fetchJson("/admin/write-enabled");
      setAdminWriteState({ enabled: !!data.admin_write_enabled, error: null });
    } catch (err) {
      if (handleAuthError(err, setAuthError)) return;
      setAdminWriteState({ enabled: null, error: err.message });
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
      if (handleAuthError(err, setAuthError)) return;
      alert(err.message);
    }
  };

  const registerAdapter = async () => {
    setSavingAdapter(true);
    try {
      const payload = { ...adapterForm, api_key: adapterForm.apiKey };
      delete payload.apiKey;
      await fetchJson("/admin/router/adapters", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      await loadBindings();
    } catch (err) {
      if (handleAuthError(err, setAuthError)) return;
      alert(err.message);
    } finally {
      setSavingAdapter(false);
    }
  };

  useEffect(() => {
    loadBindings();
    loadRouterHealth();
    loadAdminWriteEnabled();
  }, []);

  const writesDisabled = adminWriteState.enabled === false;
  const writeStatusLabel =
    adminWriteState.enabled === true
      ? "enabled"
      : adminWriteState.enabled === false
      ? "disabled"
      : "unknown";

  return (
    <div>
      <div className="panel" style={{ marginBottom: 16 }}>
        <AuthBanner
          authError={authError}
          hint="Admin role required for admin endpoints. Writes also require ELEANOR_ENABLE_ADMIN_WRITE=true."
        />
        <div className="row" style={{ justifyContent: "space-between" }}>
          <div>
            <h3 style={{ margin: 0 }}>Admin Write Access</h3>
            <div className="small">
              Admin writes control mutating endpoints (bindings, adapters, flags, weights).
            </div>
          </div>
          <div className="row">
            <button onClick={loadAdminWriteEnabled} style={{ padding: "6px 12px" }}>
              Refresh
            </button>
            <div className={`pill`} style={{ background: writesDisabled ? "#fef2f2" : "#ecfeff", color: writesDisabled ? "#991b1b" : "#0e7490" }}>
              {writeStatusLabel}
            </div>
          </div>
        </div>
        {adminWriteState.error && (
          <div className="small" style={{ color: "#f87171", marginTop: 6 }}>
            {adminWriteState.error}
          </div>
        )}
        {writesDisabled && (
          <div className="small" style={{ color: "#f87171", marginTop: 6 }}>
            Writes are disabled. Set `ELEANOR_ENABLE_ADMIN_WRITE=true` to allow updates.
          </div>
        )}
      </div>
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
          <button
            onClick={saveBinding}
            disabled={!newBinding.critic || !newBinding.adapter || writesDisabled}
            title={writesDisabled ? "Admin writes are disabled" : ""}
          >
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
        <h3>Add Router Adapter</h3>
        <div className="row">
          <input
            placeholder="name (e.g. ollama-phi3)"
            value={adapterForm.name}
            onChange={(e) => setAdapterForm({ ...adapterForm, name: e.target.value })}
          />
          <select
            value={adapterForm.type}
            onChange={(e) => setAdapterForm({ ...adapterForm, type: e.target.value })}
          >
            <option value="ollama">Ollama (local)</option>
            <option value="hf">HuggingFace (local)</option>
            <option value="openai">OpenAI</option>
            <option value="claude">Claude</option>
            <option value="grok">Grok</option>
          </select>
        </div>
        <div className="row" style={{ marginTop: 8 }}>
          <input
            placeholder="model (phi3, llama3, gpt-4o-mini, etc.)"
            value={adapterForm.model}
            onChange={(e) => setAdapterForm({ ...adapterForm, model: e.target.value })}
          />
          <input
            placeholder="device (cpu / cuda)"
            value={adapterForm.device}
            onChange={(e) => setAdapterForm({ ...adapterForm, device: e.target.value })}
            disabled={adapterForm.type !== "hf"}
          />
        </div>
        <input
          style={{ marginTop: 8 }}
          placeholder="API key (only for OpenAI/Claude/Grok)"
          value={adapterForm.apiKey}
          onChange={(e) => setAdapterForm({ ...adapterForm, apiKey: e.target.value })}
          disabled={!["openai", "claude", "grok"].includes(adapterForm.type)}
        />
        <div className="row" style={{ marginTop: 8 }}>
          <button
            onClick={registerAdapter}
            disabled={savingAdapter || !adapterForm.name || writesDisabled}
            title={writesDisabled ? "Admin writes are disabled" : ""}
          >
            {savingAdapter ? "Saving…" : "Register Adapter"}
          </button>
          <div className="small">Adapters available: {adapters.join(", ") || "none"}</div>
        </div>
      </div>
      <div className="panel">
        <h3>Router Health</h3>
        <div className="row" style={{ marginBottom: 8 }}>
          <button onClick={loadRouterHealth}>Refresh</button>
        </div>
        <pre>{routerHealth ? pretty(routerHealth) : "—"}</pre>
      </div>
      </div>
      <div style={{ marginTop: 20 }}>
        <FeatureFlagsPanel
          adminWriteEnabled={adminWriteState.enabled}
          onAuthError={setAuthError}
        />
      </div>
      <div style={{ marginTop: 20 }}>
        <div className="row" style={{ marginBottom: 12, gap: 8 }}>
          <button
            onClick={() => setAdvancedTab("evolution")}
            style={{
              padding: "8px 16px",
              backgroundColor: advancedTab === "evolution" ? "#007bff" : "#6c757d",
              color: "white",
              border: "none",
              borderRadius: 4,
              cursor: "pointer",
            }}
          >
            Temporal Evolution
          </button>
          <button
            onClick={() => setAdvancedTab("weighting")}
            style={{
              padding: "8px 16px",
              backgroundColor: advancedTab === "weighting" ? "#007bff" : "#6c757d",
              color: "white",
              border: "none",
              borderRadius: 4,
              cursor: "pointer",
            }}
          >
            Adaptive Weighting
          </button>
        </div>
        {advancedTab === "evolution" && <TemporalEvolutionPanel onAuthError={setAuthError} />}
        {advancedTab === "weighting" && (
          <AdaptiveWeightingPanel
            adminWriteEnabled={adminWriteState.enabled}
            onAuthError={setAuthError}
          />
        )}
      </div>
    </div>
  );
};

const MetricsPanel = () => {
  const grafanaUrl = useMemo(() => {
    const envUrl = import.meta.env.VITE_GRAFANA_URL;
    if (envUrl) return envUrl;
    const host = window.location.hostname || "localhost";
    const port = 3000;
    return `http://${host}:${port}/d/eleanor-obsv/eleanor-v8-observability?orgId=1&refresh=10s`;
  }, []);

  const [dependencyMetrics, setDependencyMetrics] = useState(null);
  const [authError, setAuthError] = useState("");
  const loadDependencyMetrics = useCallback(async () => {
    setAuthError("");
    try {
      const data = await fetchJson("/admin/dependencies");
      setDependencyMetrics(data);
    } catch (err) {
      if (handleAuthError(err, setAuthError)) return;
      setDependencyMetrics({ error: err.message || "Unable to load metrics" });
    }
  }, []);

  useEffect(() => {
    loadDependencyMetrics();
  }, [loadDependencyMetrics]);

  const formattedLastChecked =
    dependencyMetrics?.last_checked &&
    (() => {
      try {
        return new Date(dependencyMetrics.last_checked).toLocaleString();
      } catch {
        return dependencyMetrics.last_checked;
      }
    })();

  const failureEntries = dependencyMetrics?.failures || {};
  const hasFailures = dependencyMetrics?.has_failures;
  const failureGrid =
    hasFailures && Object.keys(failureEntries).length > 0 ? (
      <div className="dependency-grid">
        {Object.entries(failureEntries).map(([name, info]) => (
          <div key={name} className="dependency-card">
            <div className="dependency-name">{name}</div>
            <div className="dependency-count">{info.count}</div>
            <div className="small">{info.count === 1 ? "failure" : "failures"}</div>
            <div className="small">{info.last_failure ?? "—"}</div>
            {info.last_error && (
              <div className="small">Error: {truncate(info.last_error, 120)}</div>
            )}
          </div>
        ))}
      </div>
    ) : dependencyMetrics?.tracked_dependencies ? (
      <div className="small">No active failures for tracked dependencies.</div>
    ) : (
      <div className="small">No dependencies have reported yet.</div>
    );

  const rawStatus = dependencyMetrics?.status ?? "unknown";
  const statusLabel = rawStatus.replace("_", " ");
  const friendlyStatus = statusLabel
    ? statusLabel.charAt(0).toUpperCase() + statusLabel.slice(1)
    : "Unknown";
  const alertMessage = dependencyMetrics?.total_failures
    ? `Trigger a Grafana alert whenever /admin/dependencies returns more than 0 total failures (current ${dependencyMetrics.total_failures}).`
    : "All dependency checks are healthy. Maintain this visibility by alerting on /admin/dependencies total_failures > 0.";

  const dependencyContent = dependencyMetrics ? (
    dependencyMetrics.error ? (
      <div className="small">Error: {dependencyMetrics.error}</div>
    ) : (
      <>
        <div className="dependency-summary">
          <div className="summary-card">
            <div className="label">Status</div>
            <div className={`status-pill status-${rawStatus}`}>
              {friendlyStatus}
            </div>
          </div>
          <div className="summary-card">
            <div className="label">Total Failures</div>
            <div className="value">{dependencyMetrics.total_failures ?? 0}</div>
          </div>
          <div className="summary-card">
            <div className="label">Tracked dependencies</div>
            <div className="value">{dependencyMetrics.tracked_dependencies ?? 0}</div>
          </div>
          <div className="summary-card">
            <div className="label">Last checked</div>
            <div className="value">{formattedLastChecked || "—"}</div>
          </div>
        </div>
        <div className="dependency-alerts">
          <div
            className={`dependency-alert-card ${hasFailures ? "warning" : "positive"}`}
            role="status"
          >
            <div className="label">Alert suggestion</div>
            <div className="value">{alertMessage}</div>
          </div>
        </div>
        {failureGrid}
      </>
    )
  ) : (
    <div className="small">Loading dependency metrics…</div>
  );

  return (
    <div className="panel">
      <AuthBanner authError={authError} hint="Admin role required to view dependency metrics." />
      <h3>Grafana</h3>
      <p className="small">Requires docker compose stack (Grafana at :3000) or adjust URL.</p>
      <iframe src={grafanaUrl} title="Grafana Dashboard"></iframe>
      <div className="dependency-section">
        <div className="row" style={{ justifyContent: "space-between", marginBottom: 4 }}>
          <h4 style={{ margin: 0 }}>Dependency Dashboard</h4>
          <button onClick={loadDependencyMetrics}>Refresh</button>
        </div>
        {dependencyContent}
      </div>
    </div>
  );
};

const App = () => {
  const [tab, setTab] = useState("simple");
  const [authToken, setAuthToken] = useState(() => getStoredToken());
  const [roleStatus, setRoleStatus] = useState({ admin: "unknown", reviewer: "unknown" });
  const [roleCheckError, setRoleCheckError] = useState("");
  const [checkingRoles, setCheckingRoles] = useState(false);
  useEffect(() => {
    setStoredToken(authToken.trim());
  }, [authToken]);
  const authStatus = authToken ? "token set" : "no token";
  const checkRoles = async () => {
    setCheckingRoles(true);
    setRoleCheckError("");
    try {
      const [adminStatus, reviewerStatus] = await Promise.all([
        probeRole("/admin/write-enabled"),
        probeRole("/review/pending"),
      ]);
      setRoleStatus({ admin: adminStatus, reviewer: reviewerStatus });
    } catch (err) {
      setRoleCheckError(err.message || "Role check failed");
    } finally {
      setCheckingRoles(false);
    }
  };
  const roleBadgeStyle = (status) => {
    if (status === "ok") return { background: "rgba(16,185,129,0.2)", color: "#34d399" };
    if (status === "forbidden") return { background: "rgba(248,113,113,0.2)", color: "#f87171" };
    if (status === "unauth") return { background: "rgba(251,191,36,0.2)", color: "#fbbf24" };
    return { background: "rgba(148,163,184,0.2)", color: "#94a3b8" };
  };
  const roleLabel = (status) => {
    if (status === "ok") return "ok";
    if (status === "forbidden") return "missing role";
    if (status === "unauth") return "login required";
    return "unknown";
  };
  const tabs = [
    { id: "simple", label: "Simple Stream" },
    { id: "deliberate", label: "Deliberate" },
    { id: "traces", label: "Traces & Audit" },
    { id: "audit", label: "Audit Ledger" },
    { id: "governance", label: "Governance" },
    { id: "review", label: "Human Review" },
    { id: "admin", label: "Admin" },
    { id: "metrics", label: "Metrics" },
  ];

  return (
    <div>
      <header>
        <div className="row" style={{ gap: 12 }}>
          <div className="brand">ELEANOR V8 — Ops Console</div>
          <div className="pill">API Console</div>
        </div>
        <div className="row" style={{ gap: 8 }}>
          <input
            type="password"
            placeholder="Auth token (JWT)"
            value={authToken}
            onChange={(e) => setAuthToken(e.target.value)}
            style={{ width: 260 }}
          />
          <button onClick={() => setAuthToken("")} style={{ padding: "8px 12px" }}>
            Clear
          </button>
          <span className="small">{authStatus}</span>
          <button onClick={checkRoles} disabled={checkingRoles} style={{ padding: "8px 12px" }}>
            {checkingRoles ? "Checking…" : "Check Roles"}
          </button>
          <span
            className="pill"
            style={roleBadgeStyle(roleStatus.admin)}
            title="Admin endpoints require ADMIN_ROLE"
          >
            admin: {roleLabel(roleStatus.admin)}
          </span>
          <span
            className="pill"
            style={roleBadgeStyle(roleStatus.reviewer)}
            title="Human review endpoints require REVIEWER_ROLE"
          >
            reviewer: {roleLabel(roleStatus.reviewer)}
          </span>
        </div>
        {roleCheckError && (
          <div className="small" style={{ color: "#f87171", marginTop: 6 }}>
            {roleCheckError}
          </div>
        )}
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
        {tab === "audit" && <AuditLedgerPanel />}
        {tab === "governance" && <GovernancePanel />}
        {tab === "review" && <HumanReviewPanel />}
        {tab === "admin" && <AdminPanel />}
        {tab === "metrics" && <MetricsPanel />}
      </div>
    </div>
  );
};

export default App;
