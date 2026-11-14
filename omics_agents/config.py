"""Central configuration for the multi-omics agent demo."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# --- backend selection --------------------------------------------------------
# Set LLM_PROVIDER=ollama (in .env or shell) to use a free local model via Ollama.
# Set LLM_PROVIDER=anthropic (default) to use the Anthropic API.
PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")

# Anthropic settings
# Switch to "claude-sonnet-5-5" (cheaper) or "claude-haiku-5-5" (cheapest) if you like.
MODEL = "claude-opus-5-5"
# Effort controls how much the model thinks. "low" keeps the demo fast and cheap.
EFFORT = "low"
MAX_TOKENS = 16000
MAX_STEPS = 12  # safety cap on agent tool-use iterations

# USD per 1M tokens (input, output) - used only to print a cost estimate.
PRICES = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
}

# Ollama settings (used when PROVIDER="ollama")
# Pull the model first: ollama pull qwen2.5:7b
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:2b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
# The "ollama" path is a generic OpenAI-compatible client, so OLLAMA_BASE_URL can also point at a
# hosted endpoint (e.g. Gemini, see omics_agents/colab.py); its key goes in LLM_API_KEY.
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")

DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
MEMORY_FILE = ROOT / "memory" / "findings.json"
KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"

DATA_URL = (
    "https://raw.githubusercontent.com/mixOmicsTeam/mixOmics/master/data/breast.TCGA.rda"
)

for d in (DATA_DIR, OUTPUT_DIR, MEMORY_FILE.parent):
    d.mkdir(parents=True, exist_ok=True)
