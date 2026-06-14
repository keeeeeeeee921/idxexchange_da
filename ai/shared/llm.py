"""
Shared LLM client
=================
One provider-agnostic chat entry point used by every ai/ module (M1 report,
M2 assistant). Switch backends with LLM_PROVIDER=anthropic|ollama (auto-detect
otherwise). Stdlib-only HTTP (urllib + json) — no SDK install required.

    from ai.shared import llm
    text = llm.complete("...", system="...", force_json=True)
    provider, model = llm.resolve_provider(), llm.resolve_model(llm.resolve_provider())
"""

import os
import json
import urllib.request
import urllib.error

# Reuse the tiny .env loader from the FRED connector (single source of truth).
try:
    from connectors.fred_connector import load_dotenv
except Exception:  # pragma: no cover
    def load_dotenv(path=None):
        return

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_DEFAULT_MODEL = "claude-sonnet-4-6"
OLLAMA_DEFAULT_HOST = "http://localhost:11434"
OLLAMA_DEFAULT_MODEL = "qwen2.5:3b"


# --------------------------------------------------------------------------- #
# Provider / model / key resolution
# --------------------------------------------------------------------------- #
def get_anthropic_key():
    load_dotenv()
    k = os.environ.get("ANTHROPIC_API_KEY")
    return None if (not k or k.startswith("your_")) else k


def resolve_provider(provider=None):
    """Active provider: explicit arg > LLM_PROVIDER env > auto-detect.
    Returns one of: 'anthropic', 'ollama', 'stub'."""
    load_dotenv()
    p = (provider or os.environ.get("LLM_PROVIDER") or "auto").lower()
    if p == "auto":
        p = "anthropic" if get_anthropic_key() else "ollama"
    return p


def resolve_model(provider, model=None):
    if model:
        return model
    if provider == "anthropic":
        return os.environ.get("ANTHROPIC_MODEL", ANTHROPIC_DEFAULT_MODEL)
    if provider == "ollama":
        return os.environ.get("OLLAMA_MODEL", OLLAMA_DEFAULT_MODEL)
    return "?"


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
def _anthropic(prompt, system, model, key, temperature, timeout):
    body = json.dumps({
        "model": model,
        "max_tokens": 1024,
        "temperature": temperature,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")
    req = urllib.request.Request(
        ANTHROPIC_URL, data=body, method="POST",
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    parts = [c.get("text", "") for c in payload.get("content", []) if c.get("type") == "text"]
    return "".join(parts).strip()


def _ollama(prompt, system, force_json, model, temperature, timeout):
    host = os.environ.get("OLLAMA_HOST", OLLAMA_DEFAULT_HOST).rstrip("/")
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if force_json:
        payload["format"] = "json"
    req = urllib.request.Request(
        host + "/api/chat", data=json.dumps(payload).encode("utf-8"),
        method="POST", headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("message", {}).get("content", "").strip()


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def complete(prompt, system="", force_json=False, provider=None, model=None,
             temperature=0.2, timeout=180):
    """Single-turn completion. Returns the model's text reply.

    Raises RuntimeError if no usable provider is configured, or urllib errors
    on transport/API failure — callers decide how to degrade.
    """
    provider = resolve_provider(provider)
    model = resolve_model(provider, model)
    if provider == "anthropic":
        key = get_anthropic_key()
        if not key:
            raise RuntimeError("anthropic provider selected but ANTHROPIC_API_KEY is not set")
        return _anthropic(prompt, system, model, key, temperature, timeout)
    if provider == "ollama":
        return _ollama(prompt, system, force_json, model, temperature, timeout)
    raise RuntimeError(f"no usable LLM provider (resolved: {provider!r})")
