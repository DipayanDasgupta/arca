"""
arca/llm/local_llm.py
======================
Zero-API local LLM via llama-cpp-python + GGUF models.
GPU offload supported (n_gpu_layers=-1).

First run can auto-download the model if auto_download=True.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# ── Try llama-cpp-python ──────────────────────────────────────────────────────
try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    Llama = None
    LLAMA_AVAILABLE = False

# ── Model registry ────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "llama-3.2-3b": {
        "filename": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "url": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "chat_template": "llama3",
    },
    "phi-3-mini": {
        "filename": "Phi-3-mini-4k-instruct-q4.gguf",
        "url": "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf",
        "chat_template": "phi3",
    },
    "gemma-2-2b": {
        "filename": "gemma-2-2b-it-Q4_K_M.gguf",
        "url": "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf",
        "chat_template": "gemma",
    },
}

DEFAULT_MODEL_KEY = "llama-3.2-3b"
DEFAULT_MODEL_DIR = Path.home() / ".arca" / "models"


class LocalLLM:
    """
    Wrapper around llama-cpp-python for fully local inference.

    Usage:
        llm = LocalLLM()                    # uses llama-3.2-3b by default
        response = llm.chat(system=..., user=...)
    """

    def __init__(
        self,
        model_key: str = DEFAULT_MODEL_KEY,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        n_gpu_layers: int = -1,      # -1 = full GPU offload
        n_ctx: int = 4096,
        n_batch: int = 512,
        verbose: bool = False,
        auto_download: bool = True,   # Changed default to True for better UX
    ):
        self.model_key = model_key
        self.model_dir = Path(model_dir)
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self.verbose = verbose
        self.auto_download = auto_download

        self._llm: Optional[Llama] = None
        self._loaded = False

        meta = MODEL_REGISTRY.get(model_key, MODEL_REGISTRY[DEFAULT_MODEL_KEY])
        self._filename = meta["filename"]
        self._url = meta["url"]
        self._template = meta.get("chat_template", "llama3")

        self.model_path = self.model_dir / self._filename

    # ── Model management ──────────────────────────────────────────────────────

    def _ensure_model(self) -> bool:
        """Ensure model exists, download if allowed."""
        self.model_dir.mkdir(parents=True, exist_ok=True)

        if self.model_path.exists():
            return True

        if not self.auto_download:
            print(f"\n[ARCA LocalLLM] Model not found: {self._filename}")
            print(f"   Download manually:\n   wget -O {self.model_path} '{self._url}'\n")
            return False

        print(f"[ARCA LocalLLM] Downloading {self._filename} (~2GB)...")
        try:
            import urllib.request
            urllib.request.urlretrieve(self._url, self.model_path)
            print(f"[ARCA LocalLLM] ✓ Downloaded → {self.model_path}")
            return True
        except Exception as e:
            print(f"[ARCA LocalLLM] Download failed: {e}")
            return False

    def load(self) -> bool:
        if not LLAMA_AVAILABLE:
            print("[ARCA LocalLLM] llama-cpp-python not installed.")
            return False

        if not self._ensure_model():
            return False

        try:
            self._llm = Llama(
                model_path=str(self.model_path),
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=self.n_ctx,
                n_batch=self.n_batch,
                verbose=self.verbose,
            )
            self._loaded = True
            print(f"[ARCA LocalLLM] ✓ Loaded {self._filename} (gpu_layers={self.n_gpu_layers})")
            return True
        except Exception as e:
            print(f"[ARCA LocalLLM] Load failed: {e}")
            return False

    @property
    def available(self) -> bool:
        return LLAMA_AVAILABLE and self.model_path.exists()

    # ── Inference ─────────────────────────────────────────────────────────────

    def complete(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
        """Raw completion."""
        if not self._loaded:
            if not self.load():
                return "LocalLLM fallback: Prioritize critical hosts and scan first."
        try:
            out = self._llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["</s>", "[INST]", "Human:", "User:", "<|eot_id|>"],
                echo=False,
            )
            return out["choices"][0]["text"].strip()
        except Exception as e:
            print(f"[ARCA LocalLLM] Inference error: {e}")
            return "LocalLLM fallback: Focus on high-value targets."

    def chat(self, system: str, user: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
        """Chat-formatted completion."""
        if self._template == "llama3":
            prompt = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
                f"{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
                f"{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            )
        elif self._template == "phi3":
            prompt = f"<|system|>\n{system}<|end|>\n<|user|>\n{user}<|end|>\n<|assistant|>\n"
        elif self._template == "gemma":
            prompt = f"<start_of_turn>user\n{system}\n\n{user}<end_of_turn>\n<start_of_turn>model\n"
        else:
            prompt = f"System: {system}\nUser: {user}\nAssistant:"

        return self.complete(prompt, max_tokens=max_tokens, temperature=temperature)


# ── Singleton helper (optional but convenient) ────────────────────────────────
_instance: Optional[LocalLLM] = None


def get_local_llm(model_key: str = DEFAULT_MODEL_KEY, **kwargs) -> LocalLLM:
    global _instance
    if _instance is None or _instance.model_key != model_key:
        _instance = LocalLLM(model_key=model_key, **kwargs)
    return _instance