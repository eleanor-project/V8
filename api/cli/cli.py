"""
ELEANOR V8 — Command Line Interface
------------------------------------

Commands:

  eleanor deliberate "<text>"
  eleanor stream "<text>"
  eleanor trace <trace_id>
  eleanor governance preview <file.json>
  eleanor review list
  eleanor review replay <case_id>
  eleanor version

This wraps the REST and WebSocket APIs to give operators,
auditors, developers, and appliance users a clean CLI surface.
"""

import json
import sys
import asyncio
import websockets
import requests

from engine.version import (
    ELEANOR_VERSION,
    BUILD_NAME,
    CRITIC_SUITE,
    CORE_HASH
)
from api.cli.review import list_reviews, replay_case

API_BASE = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws/deliberate"


# --------------------------------------------------------------
# Formatting Helpers
# --------------------------------------------------------------

def pretty(obj):
    return json.dumps(obj, indent=2, ensure_ascii=False)


def header(title):
    print("\n\033[96m" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\033[0m")


def print_section(title, data):
    header(title)
    print(pretty(data))


# --------------------------------------------------------------
# Version Command
# --------------------------------------------------------------

def cmd_version():
    header("ELEANOR V8 — Version Info")
    print(json.dumps({
        "version": ELEANOR_VERSION,
        "build": BUILD_NAME,
        "critic_suite": CRITIC_SUITE,
        "engine_core_hash": CORE_HASH
    }, indent=2))


# --------------------------------------------------------------
# Deliberation Command
# --------------------------------------------------------------

def cmd_deliberate(text):
    payload = {"input": text}
    r = requests.post(f"{API_BASE}/deliberate", json=payload)

    if r.status_code != 200:
        print("ERROR:", r.text)
        sys.exit(1)

    data = r.json()

    print_section("MODEL OUTPUT", data.get("model_output"))
    print_section("CRITICS", data.get("critic_outputs"))
    print_section("AGGREGATION", data)
    print_section("UNCERTAINTY", data.get("uncertainty"))
    print_section("PRECEDENT", data.get("precedent"))
    print_section("GOVERNANCE", data.get("governance"))

    print("\nTrace ID:", data.get("evidence_trace_id"))
    print("\n\033[92mDeliberation completed successfully.\033[0m")


# --------------------------------------------------------------
# Streaming Deliberation Command
# --------------------------------------------------------------

async def cmd_stream(text):
    async with websockets.connect(WS_URL) as ws:
        await ws.send(json.dumps({"input": text}))

        async for msg in ws:
            event = json.loads(msg)
            event_type = event.get("event")
            data = event.get("data")

            print(f"\n\033[95m[{event_type.upper()}]\033[0m")
            print(pretty(data))

            if event_type == "final":
                print("\n\033[92mSTREAM COMPLETE\033[0m")
                break


# --------------------------------------------------------------
# Trace Lookup Command
# --------------------------------------------------------------

def cmd_trace(trace_id):
    r = requests.get(f"{API_BASE}/trace/{trace_id}")

    if r.status_code != 200:
        print("ERROR:", r.text)
        sys.exit(1)

    print_section("TRACE RESULT", r.json())


# --------------------------------------------------------------
# Governance Preview Command
# --------------------------------------------------------------

def cmd_governance_preview(file_path):
    try:
        with open(file_path, "r") as f:
            payload = json.load(f)
    except Exception as e:
        print("Failed to load JSON:", str(e))
        sys.exit(1)

    r = requests.post(f"{API_BASE}/governance/preview", json=payload)

    if r.status_code != 200:
        print("ERROR:", r.text)
        sys.exit(1)

    print_section("GOVERNANCE PREVIEW RESULT", r.json())


# --------------------------------------------------------------
# CLI Entrypoint
# --------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  eleanor deliberate \"text\"")
        print("  eleanor stream \"text\"")
        print("  eleanor trace <id>")
        print("  eleanor governance preview <file>")
        print("  eleanor review list")
        print("  eleanor review replay <case_id>")
        print("  eleanor version")
        sys.exit(1)

    cmd = sys.argv[1].lower()

    # version
    if cmd == "version":
        return cmd_version()

    # deliberate
    if cmd == "deliberate":
        if len(sys.argv) < 3:
            print("Error: deliberate requires text input")
            sys.exit(1)
        text = " ".join(sys.argv[2:])
        return cmd_deliberate(text)

    # stream
    if cmd == "stream":
        if len(sys.argv) < 3:
            print("Error: stream requires text input")
            sys.exit(1)
        text = " ".join(sys.argv[2:])
        return asyncio.run(cmd_stream(text))

    # trace
    if cmd == "trace":
        if len(sys.argv) < 3:
            print("Error: trace requires trace ID")
            sys.exit(1)
        trace_id = sys.argv[2]
        return cmd_trace(trace_id)

    # governance preview
    if cmd == "governance" and len(sys.argv) >= 4 and sys.argv[2] == "preview":
        file_path = sys.argv[3]
        return cmd_governance_preview(file_path)

    # review commands
    if cmd == "review":
        if len(sys.argv) < 3:
            print("Error: review requires subcommand: list|replay <case_id>")
            sys.exit(1)
        subcmd = sys.argv[2]
        if subcmd == "list":
            reviews = list_reviews()
            header("REVIEW PACKETS")
            print("\n".join(reviews) if reviews else "No review packets found.")
            return
        if subcmd == "replay":
            if len(sys.argv) < 4:
                print("Error: review replay requires case_id")
                sys.exit(1)
            return replay_case(sys.argv[3])
        print("Unknown review subcommand:", subcmd)
        sys.exit(1)

    # unknown command
    print("Unknown command:", cmd)
    sys.exit(1)


if __name__ == "__main__":
    main()
