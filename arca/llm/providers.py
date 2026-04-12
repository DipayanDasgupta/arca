"""
arca.llm.providers
==================
Unified LLM provider layer for ARCA.

Supports: Ollama (local), Groq (fast free tier), Anthropic (Claude), OpenAI
All providers implement the same interface: .complete(prompt) -> str

Priority order (auto-detect):
  1. Ollama  (if running locally)
  2. Groq    (if GROQ_API_KEY is set — free, fast)
  3. Anthropic (if ANTHROPIC_API_KEY is set)
  4. OpenAI  (if OPENAI_API_KEY is set)
  5. Rule-based fallback (always works, no API needed)

Setup:
  Local:  ollama pull llama3.2:3b && ollama serve
  Groq:   export GROQ_API_KEY=gsk_...  (free at console.groq.com)
  Claude: export ANTHROPIC_API_KEY=sk-ant-...
"""

import os
from abc import ABC, abstractmethod
from typing import Optional


# ── Base interface ────────────────────────────────────────────────────────────

class LLMProvider(ABC):
    """Abstract base for all LLM providers."""

    @abstractmethod
    def complete(self, prompt: str, max_tokens: int = 512) -> str:
        ...

    @abstractmethod
    def is_available(self) -> bool:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


# ── Ollama (local) ────────────────────────────────────────────────────────────

class OllamaProvider(LLMProvider):
    """
    Local LLM via Ollama.
    Install: curl -fsSL https://ollama.com/install.sh | sh
    Models:  ollama pull llama3.2:3b  (recommended for 6GB VRAM)
             ollama pull gemma2:2b     (even lighter)
             ollama pull phi3:mini     (fastest)
    """

    def __init__(self, model: str = "llama3.2:3b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._available: Optional[bool] = None

    @property
    def name(self) -> str:
        return f"ollama:{self.model}"

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import urllib.request
            req = urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=2)
            self._available = req.status == 200
        except Exception:
            self._available = False
        return self._available

    def complete(self, prompt: str, max_tokens: int = 512) -> str:
        import json
        import urllib.request
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.2},
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read())
        return data.get("response", "").strip()


# ── Groq (fixed - forces correct model) ───────────────────────────────────────

class GroqProvider(LLMProvider):
    """
    Groq inference — free tier, very fast (100+ tokens/sec).
    Get key at: https://console.groq.com
    export GROQ_API_KEY=gsk_...
    
    Best models for ARCA:
      llama-3.1-8b-instant   — fast, good reasoning
      llama-3.3-70b-versatile — slow but very capable
    """

    def __init__(self, model: str = "llama-3.1-8b-instant", api_key: Optional[str] = None):
        # Force correct model - ignore any wrong "llama3" passed from orchestrator
        self.model = "llama-3.1-8b-instant" if model in ("llama3", "llama-3") else model
        self.api_key = (api_key or os.environ.get("GROQ_API_KEY", "")).strip()
        self._client = None

    @property
    def name(self) -> str:
        return f"groq:{self.model}"

    def is_available(self) -> bool:
        # Strict check: must start with 'gsk_' and not be empty
        return bool(self.api_key and self.api_key.startswith("gsk_"))

    def _get_client(self):
        if self._client is None:
            try:
                from groq import Groq
                self._client = Groq(api_key=self.api_key)
            except ImportError:
                raise ImportError("Install groq: pip install groq")
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Groq client: {e}")
        return self._client

    def complete(self, prompt: str, max_tokens: int = 512) -> str:
        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.2,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            # Let the caller know it failed so fallback can trigger cleanly
            raise RuntimeError(f"Groq API call failed: {e}")


# ── Anthropic / Claude ────────────────────────────────────────────────────────

class AnthropicProvider(LLMProvider):
    """
    Anthropic Claude API.
    export ANTHROPIC_API_KEY=sk-ant-...
    Best model for ARCA: claude-haiku-4-5-20251001 (fast + cheap)
    """

    def __init__(self, model: str = "claude-haiku-4-5-20251001", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._client = None

    @property
    def name(self) -> str:
        return f"anthropic:{self.model}"

    def is_available(self) -> bool:
        return bool(self.api_key)

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("Install anthropic: pip install anthropic")
        return self._client

    def complete(self, prompt: str, max_tokens: int = 512) -> str:
        client = self._get_client()
        message = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()


# ── OpenAI ────────────────────────────────────────────────────────────────────

class OpenAIProvider(LLMProvider):
    """
    OpenAI API.
    export OPENAI_API_KEY=sk-...
    """

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._client = None

    @property
    def name(self) -> str:
        return f"openai:{self.model}"

    def is_available(self) -> bool:
        return bool(self.api_key)

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("Install openai: pip install openai")
        return self._client

    def complete(self, prompt: str, max_tokens: int = 512) -> str:
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()


# ── Rule-based fallback (always works) ───────────────────────────────────────

class RuleBasedProvider(LLMProvider):
    """Always-available fallback. No API, no LLM, pure logic."""

    @property
    def name(self) -> str:
        return "rule-based"

    def is_available(self) -> bool:
        return True

    def complete(self, prompt: str, max_tokens: int = 512) -> str:
        # Extract context clues from the prompt to generate intelligent responses
        p = prompt.lower()

        if "analyst" in p or "describe" in p or "situation" in p:
            return self._analyst_response(prompt)
        elif "critic" in p or "evaluate" in p or "mistakes" in p:
            return self._critic_response(prompt)
        elif "reflect" in p or "learn" in p or "pattern" in p:
            return "Agent should scan more hosts before attempting exploits. Prioritize high-value and critical targets."
        elif "plan" in p or "suggest" in p or "next" in p:
            return self._plan_response(prompt)
        elif "remediat" in p or "fix" in p or "patch" in p:
            return self._remediation_response(prompt)
        return "Analysis complete. Recommend reviewing the attack path for optimization opportunities."

    def _analyst_response(self, prompt: str) -> str:
        # Extract numbers from prompt for smarter response
        import re
        nums = re.findall(r'\d+', prompt)
        comp = int(nums[0]) if nums else 1
        total = int(nums[1]) if len(nums) > 1 else 8
        return (
            f"The agent has compromised {comp} of {total} hosts. "
            f"Current penetration depth is {comp/max(total,1)*100:.0f}%. "
            f"The attack is progressing via lateral movement through the network."
        )

    def _critic_response(self, prompt: str) -> str:
        return (
            "The agent's exploit efficiency could be improved. "
            "It appears to be attempting exploits on undiscovered hosts, wasting actions. "
            "Recommend: scan-first strategy before exploit attempts."
        )

    def _plan_response(self, prompt: str) -> str:
        return (
            "1. SCAN: Enumerate all reachable hosts in current subnet\n"
            "2. EXPLOIT: Target hosts with highest exploit_prob vulnerabilities\n"
            "3. PIVOT: Move to compromised host with most network connections\n"
            "4. EXFILTRATE: Extract data from critical/high-value hosts"
        )

    def _remediation_response(self, prompt: str) -> str:
        return (
            "Immediate actions: patch critical CVEs (CVSS ≥ 9.0) within 24h, "
            "enable host-based firewall on all endpoints, "
            "rotate credentials for all compromised accounts, "
            "segment IoT devices onto isolated VLAN."
        )


# ── Auto-detect factory ───────────────────────────────────────────────────────

def auto_detect_provider(
    preferred: str = "auto",
    model: Optional[str] = None,
) -> LLMProvider:
    """
    Automatically detect and return the best available LLM provider.

    Args:
        preferred: "auto" | "ollama" | "groq" | "anthropic" | "openai" | "rule"
        model: Override the default model for the chosen provider.

    Returns:
        The first available provider in priority order.
    """
    candidates: list[LLMProvider] = []

    if preferred == "ollama" or preferred == "auto":
        candidates.append(OllamaProvider(model=model or "llama3.2:3b"))

    if preferred == "groq" or preferred == "auto":
        candidates.append(GroqProvider(model=model or "llama-3.1-8b-instant"))

    if preferred == "anthropic" or preferred == "auto":
        candidates.append(AnthropicProvider(model=model or "claude-haiku-4-5-20251001"))

    if preferred == "openai" or preferred == "auto":
        candidates.append(OpenAIProvider(model=model or "gpt-4o-mini"))

    # Add rule-based as final fallback
    candidates.append(RuleBasedProvider())

    for provider in candidates:
        if provider.is_available():
            return provider

    return RuleBasedProvider()  # Should never reach here


# ── Provider info ─────────────────────────────────────────────────────────────

def list_providers() -> list[dict]:
    """Return status of all providers."""
    return [
        {
            "name": "ollama",
            "model": "llama3.2:3b",
            "available": OllamaProvider().is_available(),
            "setup": "curl -fsSL https://ollama.com/install.sh | sh && ollama pull llama3.2:3b",
            "cost": "Free (local)",
            "speed": "Medium",
        },
        {
            "name": "groq",
            "model": "llama-3.1-8b-instant",
            "available": GroqProvider().is_available(),   # Uses improved check
            "setup": "export GROQ_API_KEY=gsk_... (free at console.groq.com)",
            "cost": "Free tier",
            "speed": "Very fast",
        },
        {
            "name": "anthropic",
            "model": "claude-haiku-4-5-20251001",
            "available": bool(os.environ.get("ANTHROPIC_API_KEY")),
            "setup": "export ANTHROPIC_API_KEY=sk-ant-...",
            "cost": "Paid",
            "speed": "Fast",
        },
        {
            "name": "openai",
            "model": "gpt-4o-mini",
            "available": bool(os.environ.get("OPENAI_API_KEY")),
            "setup": "export OPENAI_API_KEY=sk-...",
            "cost": "Paid",
            "speed": "Fast",
        },
        {
            "name": "rule-based",
            "model": "N/A",
            "available": True,
            "setup": "Always available, no setup needed",
            "cost": "Free",
            "speed": "Instant",
        },
    ]