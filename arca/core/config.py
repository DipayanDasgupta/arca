"""
arca/core/config.py  (v3.1)
============================
Central configuration for ARCA.
Key change v3.1: ent_coef raised to 0.05 (was 0.01) to prevent the GNN
policy collapsing to a single action and getting stuck at -75 reward.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import yaml


@dataclass
class EnvConfig:
    """Network simulation environment settings."""
    preset: str = "small_office"
    num_hosts: int = 10
    num_subnets: int = 3
    vulnerability_density: float = 0.4
    use_cpp_backend: bool = True
    max_steps: int = 200
    reward_goal: float = 100.0
    reward_step: float = -0.5
    reward_discovery: float = 5.0
    reward_exploit: float = 20.0


@dataclass
class RLConfig:
    """PPO / CleanRL training settings."""
    algorithm: Literal["PPO", "A2C", "DQN"] = "PPO"
    policy: str = "MlpPolicy"
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    # v3.1 FIX: raised from 0.01 → 0.05 to prevent GNN policy collapse
    ent_coef: float = 0.05
    total_timesteps: int = 100_000
    eval_freq: int = 10_000
    n_eval_episodes: int = 5
    device: str = "auto"
    tensorboard_log: Optional[str] = "arca_outputs/tensorboard"

    # ── v3: GNN settings ──────────────────────────────────────────────────────
    use_gnn: bool = True
    gnn_hidden_dim: int = 128
    use_gat: bool = False   # GATv2 instead of GCN (slower but more expressive)

    # ── v3: Online reflection ─────────────────────────────────────────────────
    online_reflection_interval: int = 5_000   # LLM critique every N global steps


@dataclass
class LLMConfig:
    """LLM settings — local-first by default."""
    use_local_llm: bool = True
    local_model_key: str = "llama-3.2-3b"
    local_model_dir: str = str(Path.home() / ".arca" / "models")
    local_n_gpu_layers: int = -1   # -1 = all on GPU; 0 = CPU
    auto_download_model: bool = False

    # Legacy remote provider (fallback when use_local_llm=False or model missing)
    provider: Literal["ollama", "openai", "anthropic", "groq"] = "ollama"
    model: str = "llama3"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.2
    max_tokens: int = 512
    reflection_interval: int = 10
    critic_enabled: bool = True
    reflection_enabled: bool = True

    # v3: enabled flag (set False to skip all LLM calls)
    enabled: bool = True


@dataclass
class MemoryConfig:
    """Persistent episodic memory settings (v3.1)."""
    enabled: bool = True
    memory_dir: str = str(Path.home() / ".arca" / "memory")
    max_episodes: int = 1000
    # Min hosts compromised before an episode is worth recording
    min_compromised_to_record: int = 1
    # Seed reward mods from past memory at training start
    seed_reward_mods_from_memory: bool = True


@dataclass
class APIConfig:
    """FastAPI server settings."""
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    log_level: str = "info"
    cors_origins: list[str] = field(default_factory=lambda: ["*"])


@dataclass
class VizConfig:
    """Visualization settings."""
    enabled: bool = True
    backend: Literal["plotly", "matplotlib"] = "plotly"
    dashboard_port: int = 8050
    live_update_interval: int = 2
    save_figures: bool = True
    output_dir: str = "arca_outputs/figures"


@dataclass
class ARCAConfig:
    """Master configuration for ARCA."""
    env:       EnvConfig    = field(default_factory=EnvConfig)
    rl:        RLConfig     = field(default_factory=RLConfig)
    llm:       LLMConfig    = field(default_factory=LLMConfig)
    memory:    MemoryConfig = field(default_factory=MemoryConfig)
    api:       APIConfig    = field(default_factory=APIConfig)
    viz:       VizConfig    = field(default_factory=VizConfig)
    model_dir: str = "arca_outputs/models"
    log_dir:   str = "arca_outputs/logs"
    seed:      int = 42
    verbose:   int = 1

    @classmethod
    def default(cls) -> "ARCAConfig":
        return cls()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ARCAConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        cfg = cls()
        _apply_dict(cfg, data or {})
        return cfg

    def to_yaml(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(_to_dict(self), f, default_flow_style=False)

    def ensure_dirs(self) -> None:
        for d in [self.model_dir, self.log_dir, self.viz.output_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)
        if self.rl.tensorboard_log:
            Path(self.rl.tensorboard_log).mkdir(parents=True, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_dict(obj):
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _to_dict(getattr(obj, k)) for k in obj.__dataclass_fields__}
    return obj


def _apply_dict(cfg, data: dict) -> None:
    for k, v in data.items():
        if not hasattr(cfg, k):
            continue
        attr = getattr(cfg, k)
        if hasattr(attr, "__dataclass_fields__") and isinstance(v, dict):
            _apply_dict(attr, v)
        else:
            setattr(cfg, k, v)