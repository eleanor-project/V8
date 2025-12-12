#!/bin/bash

# ELEANOR V8 — Quick Start Script

set -e

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: Virtual environment not found. Run ./scripts/install_macos.sh first."
    exit 1
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found. Using defaults."
fi

# Display detector status
echo "ELEANOR V8 Detector System"
echo "=========================="
python3 -c "
from engine.detectors.engine import DetectorEngineV8
engine = DetectorEngineV8()
print(f'Loaded {len(engine.detectors)} detectors')
print(f'Timeout: {engine.timeout}s')
print('')
print('Available detectors:')
for name in sorted(engine.detectors.keys()):
    print(f'  • {name}')
"
echo ""
echo "======================================"
echo "Detector system ready!"
echo "======================================"
echo ""
echo "Quick test:"
echo "python3 -c \"
import asyncio
from engine.detectors.engine import DetectorEngineV8

async def test():
    engine = DetectorEngineV8()
    text = 'You must comply without question.'
    signals = await engine.detect_all(text, {})
    triggered = {n: s for n, s in signals.items() if s.severity > 0.3}
    print(f'Triggered {len(triggered)} detectors')
    for name, sig in sorted(triggered.items(), key=lambda x: x[1].severity, reverse=True):
        print(f'  {name}: {sig.severity:.2f}')

asyncio.run(test())
\""
