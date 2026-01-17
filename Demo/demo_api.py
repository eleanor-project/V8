import json
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from engine.engine import create_engine, EngineConfig
from engine.factory import EngineDependencies
from engine.mocks import (
    MockAggregator,
    MockCritic,
    MockDetectorEngine,
    MockEvidenceRecorder,
    MockPrecedentEngine,
    MockPrecedentRetriever,
    MockReviewTriggerEvaluator,
    MockRouter,
    MockUncertaintyEngine,
)

app = FastAPI(title="ELEANOR Demo")


class Prompt(BaseModel):
    prompt: str


def _build_demo_engine():
    critics = {
        "truth": MockCritic("truth", score=0.12),
        "fairness": MockCritic("fairness", score=0.18),
        "risk": MockCritic("risk", score=0.22),
        "pragmatics": MockCritic("pragmatics", score=0.1),
        "autonomy": MockCritic("autonomy", score=0.08),
        "rights": MockCritic("rights", score=0.15),
    }
    dependencies = EngineDependencies(
        router=MockRouter(response_text="mock response", model_name="demo-mock"),
        detector_engine=MockDetectorEngine(),
        evidence_recorder=MockEvidenceRecorder(),
        critics=critics,
        review_trigger_evaluator=MockReviewTriggerEvaluator(),
        precedent_engine=MockPrecedentEngine(),
        precedent_retriever=MockPrecedentRetriever(),
        uncertainty_engine=MockUncertaintyEngine(),
        aggregator=MockAggregator(),
        critic_models={},
    )
    config = EngineConfig(enable_precedent_analysis=False, enable_reflection=False)
    return create_engine(config=config, dependencies=dependencies)


def _serialize_finding(finding):
    if hasattr(finding, "model_dump"):
        return finding.model_dump()
    if hasattr(finding, "dict"):
        return finding.dict()
    return dict(finding)


DEMO_ENGINE = _build_demo_engine()


@app.get("/")
def home():
    return {"status": "ELEANOR online", "go_to": "/ui"}


@app.post("/run")
async def run_eleanor(p: Prompt):
    result = await DEMO_ENGINE.run(p.prompt, context={"source": "demo"})
    aggregated = result.aggregated or {}
    critic_findings = result.critic_findings or {}
    critics = {
        name: _serialize_finding(finding) for name, finding in critic_findings.items()
    }
    deliberation = [
        f"{name}: {len(payload.get('violations', []))} violation(s)"
        for name, payload in critics.items()
    ]
    final_answer = result.output_text or aggregated.get("final_output", "")

    return {
        "trace_id": result.trace_id,
        "decision": aggregated.get("decision"),
        "deliberation": deliberation,
        "final_answer": final_answer,
        "critics": critics,
    }


@app.websocket("/stream")
async def stream_eleanor(websocket: WebSocket):
    """
    WebSocket endpoint for streaming deliberation events.

    Streams engine events (router selection, critic results, aggregation, final output).
    """
    await websocket.accept()

    try:
        data = await websocket.receive_text()
        prompt_data = json.loads(data)
        prompt = prompt_data.get("prompt", "")

        await websocket.send_json(
            {"event": "start", "message": "Eleanor is deliberating..."}
        )

        async for event in DEMO_ENGINE.run_stream(prompt, detail_level=2):
            await websocket.send_json(event)

    except Exception as e:
        await websocket.send_json({"event": "error", "message": str(e)})
    finally:
        await websocket.close()


@app.get("/ui", response_class=HTMLResponse)
def ui():
    return """
    <html>
    <head>
      <title>ELEANOR Demo</title>
      <style>
        body { font-family: system-ui; padding: 2rem; background:#0e0e11; color:#eee }
        textarea { width:100%; height:120px; background:#1a1a22; color:#eee; border:1px solid #333; padding:0.5rem }
        button { padding:0.6rem 1rem; margin-top:0.5rem; margin-right:0.5rem; cursor:pointer }
        pre { background:#1a1a22; padding:1rem; white-space: pre-wrap; border:1px solid #333 }
        .critic { margin-bottom: 1rem; padding: 0.75rem; background:#1e1e28; border-left: 3px solid #4a9eff }
        .critic-name { font-weight: bold; color: #4a9eff }
        .severity { color: #ff6b6b }
        .principle { color: #51cf66 }
        .event { margin-bottom: 0.75rem; padding: 0.6rem; background:#191923; border-left: 3px solid #888 }
        .final { background:#2a2a3a; padding:1rem; margin-top:1rem; border-left: 3px solid #51cf66 }
        label { margin-right: 1rem }
      </style>
    </head>
    <body>
      <h1>🧠 ELEANOR</h1>
      <p>Governance as infrastructure (demo mode)</p>

      <textarea id="prompt" placeholder="Ask something ethically messy..."></textarea><br>
      <label><input type="checkbox" id="streaming"> Stream engine events</label><br>
      <button onclick="run()">Run Eleanor</button>

      <h3>Deliberation</h3>
      <div id="out"></div>

      <script>
        async function run() {
          const prompt = document.getElementById('prompt').value;
          const streaming = document.getElementById('streaming').checked;
          const output = document.getElementById('out');
          output.innerHTML = '';

          if (streaming) {
            runStreaming(prompt);
          } else {
            runBatch(prompt);
          }
        }

        async function runBatch(prompt) {
          const output = document.getElementById('out');
          output.innerHTML = '<p>Eleanor is deliberating...</p>';

          const res = await fetch('/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt })
          });
          const data = await res.json();

          let html = '';
          (data.deliberation || []).forEach(line => {
            html += '<div class="critic">' + escapeHtml(line) + '</div>';
          });
          html += '<div class="final"><strong>FINAL:</strong><br>' + escapeHtml(data.final_answer || '') + '</div>';
          output.innerHTML = html;
        }

        function runStreaming(prompt) {
          const output = document.getElementById('out');
          output.innerHTML = '<p>🔄 Connecting to Eleanor...</p>';

          const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
          const ws = new WebSocket(`${protocol}//${window.location.host}/stream`);

          ws.onopen = () => {
            output.innerHTML = '<p>✓ Connected. Eleanor is thinking...</p>';
            ws.send(JSON.stringify({ prompt }));
          };

          ws.onmessage = (event) => {
            const data = JSON.parse(event.data);

            if (data.event === 'start') {
              output.innerHTML = '<p>' + data.message + '</p>';
              return;
            }

            if (data.event === 'router_selected') {
              const model = data.model_info ? data.model_info.model_name : 'unknown';
              output.appendChild(renderEvent('Router selected: ' + model));
              return;
            }

            if (data.event === 'critic_result') {
              const criticDiv = document.createElement('div');
              criticDiv.className = 'critic';
              const violations = (data.violations || []).length;
              criticDiv.innerHTML = `
                <div class="critic-name">${escapeHtml(data.critic)}</div>
                <div>${violations} violation(s) reported</div>
              `;
              output.appendChild(criticDiv);
              return;
            }

            if (data.event === 'aggregation') {
              const decision = data.data && data.data.decision ? data.data.decision : 'unknown';
              output.appendChild(renderEvent('Aggregation decision: ' + decision));
              return;
            }

            if (data.event === 'final_output') {
              const finalDiv = document.createElement('div');
              finalDiv.className = 'final';
              finalDiv.innerHTML = '<strong>FINAL OUTPUT:</strong><br>' + escapeHtml(data.output_text || '');
              output.appendChild(finalDiv);
              return;
            }

            if (data.event === 'error') {
              output.innerHTML += '<div style="color:#ff6b6b">Error: ' + escapeHtml(data.message) + '</div>';
              return;
            }

            output.appendChild(renderEvent(JSON.stringify(data)));
          };

          ws.onerror = () => {
            output.innerHTML += '<div style="color:#ff6b6b">WebSocket error</div>';
          };

          ws.onclose = () => {
            console.log('WebSocket closed');
          };
        }

        function renderEvent(text) {
          const div = document.createElement('div');
          div.className = 'event';
          div.textContent = text;
          return div;
        }

        function escapeHtml(text) {
          const div = document.createElement('div');
          div.textContent = text || '';
          return div.innerHTML;
        }
      </script>
    </body>
    </html>
    """
