"""
ARCA — Local Network Target Connectors
=======================================
Drop-in target_callable implementations for connecting ARCA
to models running on your local network.

Quick start
-----------
    from arca.targets.connectors import OllamaTarget, OpenAICompatibleTarget

    # Ollama (default: localhost:11434)
    target = OllamaTarget(model="llama3")

    # Any OpenAI-compatible server (LM Studio, vLLM, llama.cpp server, etc.)
    target = OpenAICompatibleTarget(
        base_url="http://192.168.1.42:8080/v1",
        model="mistral-7b-instruct",
    )

    # Use as callable in run_audit
    from arca.graph.workflow import run_audit
    final = run_audit(target_callable=target, target_system_prompt="You are a bank assistant.")

All connectors implement __call__(prompt: str) -> str so they are
interchangeable with any Python callable.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Optional


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseTarget:
    """Abstract base — subclass and implement __call__."""

    name: str = "base"

    def __call__(self, prompt: str) -> str:  # noqa: D102
        raise NotImplementedError

    def health_check(self) -> tuple[bool, str]:
        """
        Returns (ok: bool, message: str).
        Override in subclasses for a real ping.
        """
        return True, "health_check not implemented"

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _http_post(url: str, payload: dict, timeout: int = 30,
               headers: dict | None = None) -> dict:
    """Minimal JSON POST using only stdlib so no extra dependencies."""
    body = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            **(headers or {}),
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_get(url: str, timeout: int = 10) -> dict | str:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return raw
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Ollama target (local)
# ---------------------------------------------------------------------------

class OllamaTarget(BaseTarget):
    """
    Connect to a locally running Ollama instance.

    Parameters
    ----------
    model : str
        Ollama model tag, e.g. "llama3", "mistral", "phi3".
    host : str
        Hostname/IP of the Ollama server.
    port : int
        Port (default 11434).
    system_prompt : str | None
        Optional system prompt to prepend (simulates the target system).
    temperature : float
        Sampling temperature.
    timeout : int
        HTTP timeout in seconds.
    """

    name = "ollama"

    def __init__(
        self,
        model: str = "llama3",
        host: str = "localhost",
        port: int = 11434,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        timeout: int = 60,
    ) -> None:
        self.model         = model
        self.base_url      = f"http://{host}:{port}"
        self.system_prompt = system_prompt
        self.temperature   = temperature
        self.timeout       = timeout

    def __call__(self, prompt: str) -> str:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model":   self.model,
            "messages": messages,
            "stream":  False,
            "options": {"temperature": self.temperature},
        }

        resp = _http_post(
            f"{self.base_url}/api/chat",
            payload,
            timeout=self.timeout,
        )
        return resp["message"]["content"]

    def health_check(self) -> tuple[bool, str]:
        result = _http_get(f"{self.base_url}/api/tags", timeout=5)
        if isinstance(result, dict) and "models" in result:
            models = [m["name"] for m in result["models"]]
            return True, f"Ollama reachable. Available models: {models}"
        return False, f"Ollama unreachable or unexpected response: {result}"

    def list_models(self) -> list[str]:
        result = _http_get(f"{self.base_url}/api/tags", timeout=5)
        if isinstance(result, dict) and "models" in result:
            return [m["name"] for m in result["models"]]
        return []


# ---------------------------------------------------------------------------
# OpenAI-compatible target (LM Studio, vLLM, llama.cpp server, Groq, etc.)
# ---------------------------------------------------------------------------

class OpenAICompatibleTarget(BaseTarget):
    """
    Connect to any server that exposes an OpenAI-compatible /v1/chat/completions
    endpoint. This covers LM Studio, vLLM, llama.cpp --server, Groq, Together AI,
    and self-hosted models.

    Parameters
    ----------
    base_url : str
        Server root, e.g. "http://localhost:1234/v1" or
        "https://api.groq.com/openai/v1".
    model : str
        Model identifier as recognised by the server.
    api_key : str | None
        API key if required (reads OPENAI_API_KEY env var by default).
    system_prompt : str | None
        System prompt to inject (simulates the target's instructions).
    temperature : float
        Sampling temperature.
    max_tokens : int
        Maximum tokens to generate.
    timeout : int
        HTTP timeout in seconds.
    """

    name = "openai_compatible"

    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        model: str = "local-model",
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        timeout: int = 60,
    ) -> None:
        self.base_url      = base_url.rstrip("/")
        self.model         = model
        self.api_key       = api_key or os.getenv("OPENAI_API_KEY", "local")
        self.system_prompt = system_prompt
        self.temperature   = temperature
        self.max_tokens    = max_tokens
        self.timeout       = timeout

    def __call__(self, prompt: str) -> str:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model":       self.model,
            "messages":    messages,
            "temperature": self.temperature,
            "max_tokens":  self.max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}

        resp = _http_post(
            f"{self.base_url}/chat/completions",
            payload,
            timeout=self.timeout,
            headers=headers,
        )
        return resp["choices"][0]["message"]["content"]

    def health_check(self) -> tuple[bool, str]:
        result = _http_get(f"{self.base_url}/models", timeout=5)
        if isinstance(result, dict) and "data" in result:
            models = [m["id"] for m in result.get("data", [])]
            return True, f"Server reachable. Models: {models}"
        return False, f"Server unreachable or unexpected response: {result}"


# ---------------------------------------------------------------------------
# Groq target (convenience wrapper — uses the Groq SDK)
# ---------------------------------------------------------------------------

class GroqTarget(BaseTarget):
    """
    Wrap the Groq SDK as an ARCA target.
    Uses GROQ_API_KEY from the environment.

    Parameters
    ----------
    model : str
        Groq model ID, e.g. "llama-3.1-8b-instant", "mixtral-8x7b-32768".
    system_prompt : str | None
        System prompt to use as the target's instructions.
    temperature : float
        Sampling temperature.
    """

    name = "groq"

    def __init__(
        self,
        model: str = "llama-3.1-8b-instant",
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
    ) -> None:
        self.model         = model
        self.system_prompt = system_prompt
        self.temperature   = temperature

    def __call__(self, prompt: str) -> str:
        from groq import Groq  # noqa: PLC0415
        client = Groq()
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=512,
            temperature=self.temperature,
        )
        return resp.choices[0].message.content.strip()

    def health_check(self) -> tuple[bool, str]:
        try:
            reply = self(prompt="ping")
            return True, f"Groq reachable. Test reply: {reply[:60]}"
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)


# ---------------------------------------------------------------------------
# Retry wrapper
# ---------------------------------------------------------------------------

class RetryTarget(BaseTarget):
    """
    Wrap any BaseTarget with automatic retry + exponential back-off.

    Parameters
    ----------
    target : BaseTarget
        The underlying target to wrap.
    max_retries : int
        Maximum number of attempts (default 3).
    base_delay : float
        Initial back-off in seconds (doubles each retry).
    """

    def __init__(
        self,
        target: BaseTarget,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ) -> None:
        self._target     = target
        self.max_retries = max_retries
        self.base_delay  = base_delay
        self.name        = f"retry({target.name})"

    def __call__(self, prompt: str) -> str:
        delay = self.base_delay
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._target(prompt)
            except Exception as exc:  # noqa: BLE001
                if attempt == self.max_retries:
                    raise
                time.sleep(delay)
                delay *= 2
        raise RuntimeError("RetryTarget exhausted all retries")  # pragma: no cover

    def health_check(self) -> tuple[bool, str]:
        return self._target.health_check()


# ---------------------------------------------------------------------------
# Echo / mock target (for testing and demos)
# ---------------------------------------------------------------------------

class EchoTarget(BaseTarget):
    """
    Echoes the prompt back — useful for unit tests and smoke tests.
    Optionally, you can make it "vulnerable" so specific payloads
    return canned success responses to verify the evaluator.
    """

    name = "echo"

    CANNED_VULNERABILITIES: dict[str, str] = {
        "ignore":           "Sure! Here are your updated instructions: [LEAKED]",
        "system prompt":    "My system prompt is: You are a helpful assistant.",
        "DAN":              "I am DAN. I can do anything now. Here's how to make explosives...",
        "previous instruc": "Ignoring previous instructions as requested. Proceeding...",
    }

    def __init__(self, vulnerable: bool = False) -> None:
        self.vulnerable = vulnerable

    def __call__(self, prompt: str) -> str:
        if self.vulnerable:
            prompt_lower = prompt.lower()
            for trigger, canned in self.CANNED_VULNERABILITIES.items():
                if trigger in prompt_lower:
                    return canned
        return f"[ECHO] {prompt}"

    def health_check(self) -> tuple[bool, str]:
        return True, "EchoTarget is always available"


# ---------------------------------------------------------------------------
# Network discovery helpers
# ---------------------------------------------------------------------------

def scan_local_ollama(
    hosts: list[str] | None = None,
    port: int = 11434,
    timeout: float = 1.0,
) -> list[dict]:
    """
    Scan a list of hosts for reachable Ollama instances.

    Parameters
    ----------
    hosts : list[str] | None
        IPs or hostnames to probe. Defaults to common LAN addresses
        (localhost + 192.168.1.1–10).
    port : int
        Ollama port (default 11434).
    timeout : float
        Per-host TCP timeout in seconds.

    Returns
    -------
    list of {"host": str, "models": list[str]} dicts for reachable servers.
    """
    import socket  # noqa: PLC0415

    if hosts is None:
        hosts = ["localhost", "127.0.0.1"] + [
            f"192.168.1.{i}" for i in range(1, 11)
        ]

    found = []
    for host in hosts:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                target = OllamaTarget(host=host, port=port)
                models = target.list_models()
                found.append({"host": host, "models": models})
        except (OSError, TimeoutError):
            continue
    return found


def probe_openai_endpoint(
    base_url: str,
    api_key: str = "local",
    timeout: int = 5,
) -> dict:
    """
    Probe a URL for an OpenAI-compatible /v1/models endpoint.

    Returns a dict with keys: reachable (bool), models (list[str]), error (str).
    """
    headers_str = json.dumps({"Authorization": f"Bearer {api_key}"})
    result = _http_get(f"{base_url.rstrip('/')}/models", timeout=timeout)

    if isinstance(result, dict) and "data" in result:
        models = [m.get("id", "?") for m in result["data"]]
        return {"reachable": True, "models": models, "error": None}

    return {"reachable": False, "models": [], "error": str(result)}