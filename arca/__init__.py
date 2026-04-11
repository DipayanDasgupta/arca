"""
ARCA — Autonomous Reinforcement Cyber Agent
============================================
A fully local RL-powered autonomous pentesting agent with:
  • Custom network simulation environment (Gymnasium-compatible)
  • PPO-based reinforcement learning (Stable-Baselines3)
  • LangGraph multi-agent orchestration with LLM critic & reflection
  • C++ accelerated simulation via pybind11 (optional)
  • FastAPI REST interface
  • Rich visualization suite (Plotly + NetworkX)
  • Full CLI via Typer

Quickstart
----------
    from arca import ARCAAgent, NetworkEnv, ARCAConfig

    env = NetworkEnv.from_preset("small_office")
    agent = ARCAAgent(env=env)
    agent.train(timesteps=50_000)
    result = agent.run_episode()
    print(result.summary())
"""

"""ARCA — Autonomous Reinforcement Cyber Agent"""

from arca.__version__ import __version__
from arca.core.config import ARCAConfig
from arca.sim.environment import NetworkEnv
from arca.core.agent import ARCAAgent
from arca.core.trainer import ARCATrainer
from arca.viz.visualizer import ARCAVisualizer

__all__ = [
    "__version__",
    "ARCAConfig",
    "NetworkEnv",
    "ARCAAgent",
    "ARCATrainer",
    "ARCAVisualizer",
]