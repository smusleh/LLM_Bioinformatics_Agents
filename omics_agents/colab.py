"""One-call setup for Google Colab, so the notebooks run with a free LLM and no local install.

Used by the first code cell of every notebook (it does nothing outside Colab):

    from omics_agents import colab
    colab.setup("ollama")     # or "gemini" / "anthropic"

Backends
--------
- "ollama"    : free, no account. Installs Ollama on the Colab VM and pulls a local model.
                Use a GPU runtime (Runtime -> Change runtime type -> T4 GPU), otherwise it is slow.
                First run takes ~3-5 min (install + ~4.7 GB model download).
- "gemini"    : free API key from https://aistudio.google.com/apikey, stored as the Colab
                secret GEMINI_API_KEY (key icon in the left sidebar). Fast, no download, and
                follows tool-use instructions better than a 7B local model.
- "anthropic" : Claude; needs the Colab secret ANTHROPIC_API_KEY and API credits.

This module must not import omics_agents.config: config reads the environment variables set
here when it is first imported (in the next notebook cell).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
import urllib.request

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
# Gemini model ids change over time; check https://ai.google.dev/gemini-api/docs/models
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"


def in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def _secret(name: str) -> str:
    from google.colab import userdata
    try:
        value = userdata.get(name)
    except Exception as e:  # SecretNotFoundError / NotebookAccessError
        raise RuntimeError(
            f"Add a Colab secret named {name} (key icon in the left sidebar) and enable "
            f"'Notebook access' for it, then re-run this cell."
        ) from e
    if not value:
        raise RuntimeError(f"Colab secret {name} is empty.")
    return value


def _ollama_up(url: str = "http://localhost:11434") -> bool:
    try:
        urllib.request.urlopen(url, timeout=2)
        return True
    except Exception:
        return False


def _setup_ollama(model: str) -> None:
    if not shutil.which("ollama"):
        print("Installing Ollama (one-off per Colab session) ...")
        subprocess.run("apt-get -qq install -y zstd > /dev/null", shell=True, check=False)
        subprocess.run("curl -fsSL https://ollama.com/install.sh | sh > /dev/null",
                       shell=True, check=True)
    if not _ollama_up():
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)
        for _ in range(30):
            if _ollama_up():
                break
            time.sleep(1)
        else:
            raise RuntimeError("Ollama server did not start.")
    print(f"Pulling {model} (skipped if already downloaded) ...")
    subprocess.run(["ollama", "pull", model], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not shutil.which("nvidia-smi"):
        print("⚠️  No GPU detected: switch to a T4 GPU runtime or the agents will be very slow.")
    os.environ.update(LLM_PROVIDER="ollama", OLLAMA_MODEL=model,
                      OLLAMA_BASE_URL="http://localhost:11434/v1", LLM_API_KEY="ollama")


def setup(backend: str = "ollama", model: str | None = None) -> None:
    """Configure the LLM backend for this Colab session (no-op outside Colab)."""
    if not in_colab():
        print("Not running in Colab: using your local .env settings.")
        return
    backend = backend.lower()
    if backend == "ollama":
        _setup_ollama(model or DEFAULT_OLLAMA_MODEL)
    elif backend == "gemini":
        # Gemini speaks the OpenAI API, so it reuses the same client path as Ollama.
        os.environ.update(LLM_PROVIDER="ollama", OLLAMA_MODEL=model or DEFAULT_GEMINI_MODEL,
                          OLLAMA_BASE_URL=GEMINI_BASE_URL,
                          LLM_API_KEY=_secret("GEMINI_API_KEY"))
    elif backend == "anthropic":
        os.environ.update(LLM_PROVIDER="anthropic", ANTHROPIC_API_KEY=_secret("ANTHROPIC_API_KEY"))
    else:
        raise ValueError(f"Unknown backend {backend!r}: use 'ollama', 'gemini' or 'anthropic'.")
    shown = "see config.MODEL" if backend == "anthropic" else os.environ["OLLAMA_MODEL"]
    print(f"✅ Colab ready: backend={backend}, model={shown}")
