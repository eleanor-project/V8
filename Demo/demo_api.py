from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from engine.critic_runner import run_critics
from engine.critic_aggregator import aggregate

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
    critic_outputs = run_critics(p.prompt)
    result = aggregate(critic_outputs)
    return result

@app.get("/ui", response_class=HTMLResponse)
def ui():
    return """
    <html>
    <head>
      <title>ELEANOR Demo</title>
      <style>
        body { font-family: system-ui; padding: 2rem; background:#0e0e11; color:#eee }
        textarea { width:100%; height:120px }
        button { padding:0.6rem; margin-top:0.5rem }
        pre { background:#1a1a22; padding:1rem }
      </style>
    </head>
    <body>
      <h1>🧠 ELEANOR</h1>
      <p>Governance as infrastructure</p>

      <textarea id="prompt" placeholder="Ask something ethically messy..."></textarea><br>
      <button onclick="run()">Run Eleanor</button>

      <h3>Deliberation</h3>
      <pre id="out"></pre>

      <script>
        async function run() {
          const res = await fetch('/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: document.getElementById('prompt').value })
          });
          const data = await res.json();
          document.getElementById('out').textContent =
            data.deliberation.join("\\n") + "\\n\\nFINAL:\\n" + data.final_answer;
        }
      </script>
    </body>
    </html>
    """
