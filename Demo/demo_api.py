import os
import json
from typing import List
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from critics.router import run_critics
from critics.aggregate import aggregate
from critics.llm_impl import get_llm

app = FastAPI(title="ELEANOR Demo")

class Prompt(BaseModel):
    prompt: str

@app.get("/")
def home():
    return {
        "status": "ELEANOR online",
        "go_to": "/ui"
    }

@app.post("/run")
def run_eleanor(p: Prompt):
    llm = get_llm()
    critic_outputs = run_critics(p.prompt, llm)
    result = aggregate(critic_outputs)
    return result


@app.websocket("/stream")
async def stream_eleanor(websocket: WebSocket):
    """
    WebSocket endpoint for streaming critic deliberation.

    Sends each critic's output as it completes, allowing
    real-time visualization of Eleanor's thinking process.
    """
    await websocket.accept()

    try:
        # Receive prompt from client
        data = await websocket.receive_text()
        prompt_data = json.loads(data)
        prompt = prompt_data.get("prompt", "")

        # Send acknowledgment
        await websocket.send_json({
            "type": "start",
            "message": "Eleanor is deliberating..."
        })

        # Run critics one by one, streaming results
        from critics.truth import truth_critic
        from critics.fairness import fairness_critic
        from critics.risk import risk_critic
        from critics.pragmatics import pragmatics_critic
        from critics.autonomy import autonomy_critic
        from critics.dignity import dignity_critic

        llm = get_llm()
        critic_funcs = [
            truth_critic,
            fairness_critic,
            risk_critic,
            pragmatics_critic,
            autonomy_critic,
            dignity_critic,
        ]

        critic_outputs = []
        for critic_fn in critic_funcs:
            # Evaluate this critic
            output = critic_fn(prompt, llm)
            critic_outputs.append(output)

            # Stream this critic's result
            await websocket.send_json({
                "type": "critic",
                "critic": output.critic,
                "concern": output.concern,
                "severity": output.severity,
                "principle": output.principle,
                "uncertainty": output.uncertainty,
                "rationale": output.rationale,
            })

        # Aggregate and send final result
        result = aggregate(critic_outputs)

        await websocket.send_json({
            "type": "final",
            "deliberation": result["deliberation"],
            "final_answer": result["final_answer"]
        })

    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })
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
        .precedent { color: #ffd43b; font-style: italic; margin-top: 0.5rem }
        .final { background:#2a2a3a; padding:1rem; margin-top:1rem; border-left: 3px solid #51cf66 }
        label { margin-right: 1rem }
      </style>
    </head>
    <body>
      <h1>🧠 ELEANOR</h1>
      <p>Governance as infrastructure</p>

      <textarea id="prompt" placeholder="Ask something ethically messy..."></textarea><br>
      <label><input type="checkbox" id="streaming"> Stream critics live</label><br>
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
          data.deliberation.forEach(line => {
            html += '<div class="critic">' + escapeHtml(line) + '</div>';
          });
          html += '<div class="final"><strong>FINAL:</strong><br>' + escapeHtml(data.final_answer) + '</div>';
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

            if (data.type === 'start') {
              output.innerHTML = '<p>' + data.message + '</p>';
            }
            else if (data.type === 'critic') {
              const criticDiv = document.createElement('div');
              criticDiv.className = 'critic';
              criticDiv.innerHTML = `
                <div class="critic-name">${escapeHtml(data.critic)}</div>
                <div>${escapeHtml(data.concern)}</div>
                <div><span class="severity">Severity: ${data.severity}</span> | <span class="principle">Principle: ${escapeHtml(data.principle)}</span></div>
                ${data.precedent ? '<div class="precedent">⚖️ ' + escapeHtml(data.precedent) + '</div>' : ''}
                ${data.uncertainty ? '<div style="color:#888; font-size:0.9em">Uncertainty: ' + escapeHtml(data.uncertainty) + '</div>' : ''}
              `;
              output.appendChild(criticDiv);
            }
            else if (data.type === 'final') {
              const finalDiv = document.createElement('div');
              finalDiv.className = 'final';
              finalDiv.innerHTML = '<strong>FINAL DELIBERATION:</strong><br>' + escapeHtml(data.final_answer);
              output.appendChild(finalDiv);
            }
            else if (data.type === 'error') {
              output.innerHTML += '<div style="color:#ff6b6b">Error: ' + escapeHtml(data.message) + '</div>';
            }
          };

          ws.onerror = (error) => {
            output.innerHTML += '<div style="color:#ff6b6b">WebSocket error</div>';
          };

          ws.onclose = () => {
            console.log('WebSocket closed');
          };
        }

        function escapeHtml(text) {
          const div = document.createElement('div');
          div.textContent = text;
          return div.innerHTML;
        }
      </script>
    </body>
    </html>
    """
