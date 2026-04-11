"""
arca.core.agent
~~~~~~~~~~~~~~~
ARCAAgent — the main interface. Combines RL policy (SB3/PPO) with
optional LangGraph multi-agent orchestration for explainable decisions.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from arca.core.config import ARCAConfig
from arca.sim.environment import NetworkEnv, EpisodeInfo


class ARCAAgent:
    """
    High-level agent interface.

    Usage::

        env = NetworkEnv.from_preset("small_office")
        agent = ARCAAgent(env=env)
        agent.train(timesteps=50_000)
        result = agent.run_episode(render=True)
    """

    def __init__(
        self,
        env: Optional[NetworkEnv] = None,
        cfg: Optional[ARCAConfig] = None,
        model_path: Optional[str] = None,
    ):
        self.cfg = cfg or ARCAConfig()
        self.env = env or NetworkEnv(cfg=self.cfg)
        self.cfg.ensure_dirs()

        self._model = None
        self._trainer = None
        self._langgraph = None

        if model_path:
            self.load(model_path)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        timesteps: Optional[int] = None,
        callback=None,
        progress_bar: bool = True,
    ) -> "ARCAAgent":
        from arca.core.trainer import ARCATrainer

        self._trainer = ARCATrainer(cfg=self.cfg, env=self.env)
        self._model = self._trainer.train(
            timesteps=timesteps or self.cfg.rl.total_timesteps,
            callback=callback,
            progress_bar=progress_bar,
        )
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def run_episode(
        self,
        render: bool = False,
        use_langgraph: bool = False,
        deterministic: bool = True,
    ) -> EpisodeInfo:
        if self._model is None:
            raise RuntimeError("Agent not trained. Call agent.train() first or load a model.")

        obs, info = self.env.reset()
        done = False
        episode_info = None

        while not done:
            action, _ = self._model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = self.env.step(int(action))
            done = terminated or truncated

            if render:
                print(self.env.render())

            if use_langgraph and self._langgraph:
                state = self.env.get_state_dict()
                self._langgraph.step(state)

            episode_info = info.get("episode_info") or self.env.episode_info

        return episode_info or self.env.episode_info

    def predict(self, obs, deterministic: bool = True):
        if self._model is None:
            raise RuntimeError("No model loaded.")
        return self._model.predict(obs, deterministic=deterministic)

    # ------------------------------------------------------------------
    # LangGraph
    # ------------------------------------------------------------------

    def enable_langgraph(self) -> "ARCAAgent":
        from arca.agents.langgraph_orchestrator import ARCAOrchestrator
        self._langgraph = ARCAOrchestrator(cfg=self.cfg)
        return self

    def reflect(self, state: dict) -> dict:
        if self._langgraph is None:
            self.enable_langgraph()
        return self._langgraph.reflect(state)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> str:
        if self._model is None:
            raise RuntimeError("No model to save.")
        save_path = path or str(Path(self.cfg.model_dir) / "arca_model")
        self._model.save(save_path)
        return save_path

    def load(self, path: str) -> "ARCAAgent":
        try:
            from stable_baselines3 import PPO, A2C, DQN
            algo_map = {"PPO": PPO, "A2C": A2C, "DQN": DQN}
            AlgoCls = algo_map.get(self.cfg.rl.algorithm, PPO)
            self._model = AlgoCls.load(path, env=self.env)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {path}: {e}")
        return self

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        trained = self._model is not None
        return (
            f"ARCAAgent(algo={self.cfg.rl.algorithm}, "
            f"env={self.cfg.env.preset}, "
            f"trained={trained})"
        )