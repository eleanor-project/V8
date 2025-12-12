#!/bin/bash

# ELEANOR V8 — macOS Installation Script
# Automates installation on macOS systems

set -e  # Exit on error

echo "======================================"
echo "ELEANOR V8 Installation"
echo "======================================"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.9"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)"; then
    echo "Error: Python 3.9+ required. Found: $python_version"
    echo "Install with: brew install python@3.11"
    exit 1
fi
echo "✓ Python $python_version detected"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists. Skipping."
else
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel
echo "✓ pip upgraded"
echo ""

# Install dependencies
echo "Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
elif [ -f "setup.py" ]; then
    pip install -e .
elif [ -f "pyproject.toml" ]; then
    pip install -e .
else
    echo "Warning: No requirements.txt, setup.py, or pyproject.toml found"
    echo "Installing common dependencies..."
    pip install pydantic fastapi uvicorn pytest pytest-asyncio
fi
echo "✓ Core dependencies installed"
echo ""

# Install dev dependencies (optional)
read -p "Install development dependencies? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -f "requirements-dev.txt" ]; then
        pip install -r requirements-dev.txt
    else
        pip install pytest pytest-asyncio pytest-cov black ruff mypy
    fi
    echo "✓ Development dependencies installed"
fi
echo ""

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cat > .env << EOF
# ELEANOR V8 Configuration

# API Configuration
ELEANOR_HOST=127.0.0.1
ELEANOR_PORT=8000

# LLM Configuration (add your API keys)
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...

# Database (optional)
# POSTGRES_URL=postgresql://user:pass@localhost:5432/eleanor
# WEAVIATE_URL=http://localhost:8080

# OPA (optional)
# OPA_URL=http://localhost:8181

# CORS
CORS_ORIGINS=http://localhost:3000,http://localhost:8000

# Logging
LOG_LEVEL=INFO
EOF
    echo "✓ .env file created (edit with your API keys)"
else
    echo ".env file already exists. Skipping."
fi
echo ""

# Test detector loading
echo "Testing detector system..."
python3 -c "
from engine.detectors.engine import DetectorEngineV8
engine = DetectorEngineV8()
print(f'✓ Loaded {len(engine.detectors)} detectors successfully')
" 2>&1
echo ""

# Run tests
echo "Running tests..."
if command -v pytest &> /dev/null; then
    pytest tests/test_detectors_comprehensive.py --tb=short -v 2>&1 | tail -20
    test_result=${PIPESTATUS[0]}
else
    echo "pytest not found, skipping tests"
    test_result=0
fi
echo ""

if [ $test_result -eq 0 ]; then
    echo "======================================"
    echo "✓ Installation Complete!"
    echo "======================================"
    echo ""
    echo "Next steps:"
    echo "1. Add your API keys to .env file (if using LLMs)"
    echo "2. Activate environment: source venv/bin/activate"
    echo "3. Test detectors: python3 -c 'from engine.detectors.engine import DetectorEngineV8; import asyncio; asyncio.run(DetectorEngineV8().detect_all(\"test\", {}))'"
    echo ""
    echo "For more information, see INSTALL.md"
else
    echo "======================================"
    echo "⚠ Installation complete but tests had issues"
    echo "======================================"
    echo ""
    echo "This may be due to missing dependencies or test setup."
    echo "The detector system should still work for basic operations."
    echo ""
    echo "To run manually:"
    echo "1. source venv/bin/activate"
    echo "2. python3 -c 'from engine.detectors.engine import DetectorEngineV8'"
fi
