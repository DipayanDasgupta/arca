"""
arca/core/config.py  (v3.3)
============================
Changes vs v3.2:
  - LLMConfig: ethical_mode (bool) — controls prompt framing, not safety
  - OfflineRLConfig: replay-buffer fine-tuning settings
  - ReportConfig: automated markdown report settings
  - CurriculumConfig: promote_reward_threshold added (reward-based promotion)
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
    ent_coef: float = 0.05
    total_timesteps: int = 100_000
    eval_freq: int = 10_000
    n_eval_episodes: int = 5
    device: str = "auto"
    tensorboard_log: Optional[str] = "arca_outputs/tensorboard"

    # GNN settings
    use_gnn: bool = True
    gnn_hidden_dim: int = 128
    use_gat: bool = False

    # Online reflection
    online_reflection_interval: int = 5_000


@dataclass
class LLMConfig:
    """LLM settings — local-first by default."""
    use_local_llm: bool = True
    local_model_key: str = "llama-3.2-3b"
    local_model_dir: str = str(Path.home() / ".arca" / "models")
    local_n_gpu_layers: int = -1
    auto_download_model: bool = False

    # Remote provider fallback
    provider: Literal["ollama", "openai", "anthropic", "groq"] = "ollama"
    model: str = "llama3"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.2
    max_tokens: int = 512
    reflection_interval: int = 10
    critic_enabled: bool = True
    reflection_enabled: bool = True
    enabled: bool = True

    # v3.3: Ethical mode — when True, prompts are framed as "authorized
    # security simulation" but never refuse. When False, prompts are more
    # direct for advanced red-team research contexts.
    # NOTE: This does NOT enable any real-world attack capability — ARCA
    # is always a sandboxed simulation.
    ethical_mode: bool = True


@dataclass
class MemoryConfig:
    """Persistent episodic memory settings."""
    enabled: bool = True
    memory_dir: str = str(Path.home() / ".arca" / "memory")
    max_episodes: int = 1000
    min_compromised_to_record: int = 1
    seed_reward_mods_from_memory: bool = True


@dataclass
class OfflineRLConfig:
    """v3.3: Offline RL / behavioral cloning from replay buffer."""
    enabled: bool = True
    # Fraction of top episodes to keep in replay buffer
    top_episode_fraction: float = 0.20
    # Fine-tune every N training steps (0 = only at end of training)
    finetune_every_n_steps: int = 0
    # BC learning rate (usually lower than online LR)
    bc_learning_rate: float = 1e-4
    # BC epochs per fine-tune call
    bc_epochs: int = 3
    # BC batch size
    bc_batch_size: int = 32
    # Minimum episodes before BC is triggered
    min_episodes_for_bc: int = 50
    # Path to save the replay buffer
    replay_buffer_path: str = str(Path.home() / ".arca" / "memory" / "replay_buffer.pkl")


@dataclass
class ReportConfig:
    """v3.3: Automated markdown report generation."""
    enabled: bool = True
    output_dir: str = "arca_outputs/reports"
    include_attack_paths: bool = True
    include_reward_curves: bool = True
    include_llm_lessons: bool = True
    max_attack_paths_shown: int = 10
    max_lessons_shown: int = 5


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
    """Master configuration for ARCA v3.3."""
    env:        EnvConfig     = field(default_factory=EnvConfig)
    rl:         RLConfig      = field(default_factory=RLConfig)
    llm:        LLMConfig     = field(default_factory=LLMConfig)
    memory:     MemoryConfig  = field(default_factory=MemoryConfig)
    offline_rl: OfflineRLConfig = field(default_factory=OfflineRLConfig)
    report:     ReportConfig  = field(default_factory=ReportConfig)
    api:        APIConfig     = field(default_factory=APIConfig)
    viz:        VizConfig     = field(default_factory=VizConfig)
    model_dir:  str           = "arca_outputs/models"
    log_dir:    str           = "arca_outputs/logs"
    seed:       int           = 42
    verbose:    int           = 1

    @classmethod
    def default(cls) -> "ARCAConfig":
        return cls()

    @classmethod
    def from_yaml(cls, path) -> "ARCAConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        cfg = cls()
        _apply_dict(cfg, data or {})
        return cfg

    def to_yaml(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(_to_dict(self), f, default_flow_style=False)

    def ensure_dirs(self) -> None:
        for d in [
            self.model_dir, self.log_dir, self.viz.output_dir,
            self.report.output_dir, self.memory.memory_dir,
        ]:
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