# orchestrator/config.py
import os

class Settings:
    def __init__(self):
        # URL where Ollama is running
        # For local use: http://localhost:11434
        # In Docker/Logic Core: http://ollama:11434
        self.OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        
        # Critic model names
        self.RIGHTS_MODEL_NAME = os.getenv("RIGHTS_MODEL_NAME", "eleanor-rights")
        self.FAIRNESS_MODEL_NAME = os.getenv("FAIRNESS_MODEL_NAME", "eleanor-fairness")
        self.RISK_MODEL_NAME = os.getenv("RISK_MODEL_NAME", "eleanor-risk")
        self.TRUTH_MODEL_NAME = os.getenv("TRUTH_MODEL_NAME", "eleanor-truth")
        self.PRAGMATICS_MODEL_NAME = os.getenv("PRAGMATICS_MODEL_NAME", "eleanor-pragmatics")

settings = Settings()
