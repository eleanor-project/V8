import React, { useCallback, useEffect, useMemo, useState } from "react";

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
    return events.filter((e) => e.event === "governance_decision");
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
                  const decision = gd.decision || gd.data?.decision;
                  if (!decision) return null;
                  const signal = decision.signal || decision.signal;
                  const confidence = decision.confidence || 0;
                  const stage = decision.stage || "unknown";
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

  const update = (key, value) => setFilters((prev) => ({ ...prev, [key]: value }));

  const runQuery = async () => {
    setLoading(true);
    setError(null);
    try {
      const qs = new URLSearchParams();
      if (filters.query) qs.append("query", filters.query);
      if (filters.event) qs.append("event", filters.event);
      if (filters.actorId) qs.append("actor_id", filters.actorId);
      if (filters.traceId) qs.append("trace_id", filters.traceId);
      if (filters.startTime) qs.append("start_time", filters.startTime);
      if (filters.endTime) qs.append("end_time", filters.endTime);
      qs.append("limit", "200");
      const data = await fetchJson(`/audit/query?${qs.toString()}`);
      setResults(data.results || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const exportData = async (format = "jsonl") => {
    try {
      const qs = new URLSearchParams();
      if (filters.query) qs.append("query", filters.query);
      if (filters.event) qs.append("event", filters.event);
      if (filters.actorId) qs.append("actor_id", filters.actorId);
      if (filters.traceId) qs.append("trace_id", filters.traceId);
      if (filters.startTime) qs.append("start_time", filters.startTime);
      if (filters.endTime) qs.append("end_time", filters.endTime);
      qs.append("format", format);
      const res = await fetch(`/audit/export?${qs.toString()}`);
      if (!res.ok) {
        throw new Error(await res.text());
      }
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `audit_export.${format === "csv" ? "csv" : "jsonl"}`;
      link.click();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      alert(`Export failed: ${err.message}`);
    }
  };

  return (
    <div className="panel">
      <h3>Audit Ledger Query</h3>
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
                    <td className="small">{row.event || "—"}</td>
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

const FeatureFlagsPanel = () => {
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
          <button onClick={saveFlags} disabled={loading || saving} style={{ padding: "6px 12px" }}>
            {saving ? "Saving..." : "Save Changes"}
          </button>
        </div>
      </div>
      <p className="small" style={{ marginBottom: 16 }}>
        Enable or disable optional features. Changes take effect immediately but require server restart for persistence.
      </p>
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

const TemporalEvolutionPanel = () => {
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

const AdaptiveWeightingPanel = () => {
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
          <button onClick={updateWeights} disabled={updating || loading}>
            {updating ? "Updating..." : "Update Weights"}
          </button>
          <button onClick={resetWeights} disabled={updating || loading} style={{ backgroundColor: "#dc3545" }}>
            Reset
          </button>
        </div>
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

  const registerAdapter = async () => {
    setSavingAdapter(true);
    try {
      await fetchJson("/admin/router/adapters", {
        method: "POST",
        body: JSON.stringify(adapterForm),
      });
      await loadBindings();
    } catch (err) {
      alert(err.message);
    } finally {
      setSavingAdapter(false);
    }
  };

  useEffect(() => {
    loadBindings();
    loadRouterHealth();
  }, []);

  return (
    <div>
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
          <button onClick={registerAdapter} disabled={savingAdapter || !adapterForm.name}>
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
        <FeatureFlagsPanel />
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
        {advancedTab === "evolution" && <TemporalEvolutionPanel />}
        {advancedTab === "weighting" && <AdaptiveWeightingPanel />}
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

  const [dependencyMetrics, setDependencyMetrics] = useState(null);
  const loadDependencyMetrics = useCallback(async () => {
    try {
      const data = await fetchJson("/admin/dependencies");
      setDependencyMetrics(data);
    } catch (err) {
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
  const tabs = [
    { id: "simple", label: "Simple Stream" },
    { id: "deliberate", label: "Deliberate" },
    { id: "traces", label: "Traces & Audit" },
    { id: "audit", label: "Audit Ledger" },
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
        {tab === "audit" && <AuditLedgerPanel />}
        {tab === "admin" && <AdminPanel />}
        {tab === "metrics" && <MetricsPanel />}
      </div>
    </div>
  );
};

export default App;
