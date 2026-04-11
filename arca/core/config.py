"""
arca.core.config
~~~~~~~~~~~~~~~~
Central configuration for ARCA. All subsystems read from ARCAConfig.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import yaml


@dataclass
class EnvConfig:
    """Network simulation environment settings."""
    preset: str = "small_office"          # "small_office" | "enterprise" | "custom"
    num_hosts: int = 10
    num_subnets: int = 3
    vulnerability_density: float = 0.4    # fraction of hosts with vulns
    use_cpp_backend: bool = True           # use C++ sim engine if available
    max_steps: int = 200
    reward_goal: float = 100.0
    reward_step: float = -0.5
    reward_discovery: float = 5.0
    reward_exploit: float = 20.0


@dataclass
class RLConfig:
    """PPO / SB3 training settings."""
    algorithm: Literal["PPO", "A2C", "DQN"] = "PPO"
    policy: str = "MlpPolicy"
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    total_timesteps: int = 100_000
    eval_freq: int = 10_000
    n_eval_episodes: int = 5
    device: str = "auto"                  # "auto" | "cpu" | "cuda" | "mps"
    tensorboard_log: Optional[str] = None


@dataclass
class LLMConfig:
    """LangGraph / LLM critic and reflection settings."""
    enabled: bool = True
    provider: Literal["ollama", "openai", "anthropic"] = "ollama"
    model: str = "llama3"                 # ollama model name or API model id
    base_url: str = "http://localhost:11434"
    temperature: float = 0.2
    max_tokens: int = 512
    reflection_interval: int = 10        # run reflection every N episodes
    critic_enabled: bool = True
    reflection_enabled: bool = True


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
    """Visualization suite settings."""
    enabled: bool = True
    backend: Literal["plotly", "matplotlib"] = "plotly"
    dashboard_port: int = 8050
    live_update_interval: int = 2        # seconds
    save_figures: bool = True
    output_dir: str = "arca_outputs/figures"


@dataclass
class ARCAConfig:
    """Master configuration container for ARCA."""
    env: EnvConfig = field(default_factory=EnvConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    api: APIConfig = field(default_factory=APIConfig)
    viz: VizConfig = field(default_factory=VizConfig)
    model_dir: str = "arca_outputs/models"
    log_dir: str = "arca_outputs/logs"
    seed: int = 42
    verbose: int = 1

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _to_dict(obj) -> dict:
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _to_dict(getattr(obj, k)) for k in obj.__dataclass_fields__}
    return obj


def _apply_dict(cfg, data: dict) -> None:
    for k, v in data.items():
        if hasattr(cfg, k):
            attr = getattr(cfg, k)
            if hasattr(attr, "__dataclass_fields__") and isinstance(v, dict):
                _apply_dict(attr, v)
            else:
                setattr(cfg, k, v)