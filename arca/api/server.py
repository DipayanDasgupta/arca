"""
arca.api.server
===============
FastAPI REST interface for ARCA.

Start with:  arca serve
             uvicorn arca.api.server:app --reload --port 8000

Endpoints:
  GET  /             — health + version
  GET  /status       — current agent/env status
  POST /train        — start a training run
  POST /audit        — run an audit episode
  POST /reflect      — run LangGraph reflection
  GET  /presets      — list available network presets
"""

from __future__ import annotations

import time
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from arca.__version__ import __version__

# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ARCA — Autonomous Reinforcement Cyber Agent",
    description=(
        "Fully local RL-powered pentesting simulation with LangGraph orchestration. "
        "All computation runs on your machine — no data leaves locally."
    ),
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory state (single-agent server) ────────────────────────────────────

_state: dict[str, Any] = {
    "agent": None,
    "env": None,
    "cfg": None,
    "training_active": False,
    "last_trained_at": None,
    "total_timesteps_trained": 0,
}


# ── Request / Response schemas ───────────────────────────────────────────────

class TrainRequest(BaseModel):
    preset: str = Field("small_office", description="Network preset name")
    timesteps: int = Field(50_000, ge=1_000, le=5_000_000, description="Training timesteps")
    algorithm: str = Field("PPO", description="RL algorithm: PPO | A2C | DQN")
    learning_rate: float = Field(3e-4, description="Learning rate")


class AuditRequest(BaseModel):
    preset: str = Field("small_office", description="Network preset")
    timesteps: int = Field(20_000, ge=500, description="Quick-train timesteps if no agent loaded")
    use_existing: bool = Field(True, description="Use already-trained agent if available")
    langgraph: bool = Field(False, description="Enable LangGraph LLM reflection")


class ReflectRequest(BaseModel):
    state: Optional[dict] = Field(None, description="Network state dict (auto-generates if None)")


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {
        "status": "ok",
        "service": "ARCA API",
        "version": __version__,
        "docs": "/docs",
        "agent_ready": _state["agent"] is not None,
    }


@app.get("/status", tags=["Health"])
def status():
    return {
        "agent_loaded": _state["agent"] is not None,
        "training_active": _state["training_active"],
        "last_trained_at": _state["last_trained_at"],
        "total_timesteps_trained": _state["total_timesteps_trained"],
        "preset": _state["cfg"].env.preset if _state["cfg"] else None,
    }


@app.get("/presets", tags=["Info"])
def list_presets():
    from arca.sim.environment import PRESETS
    return {
        "presets": {
            name: {
                "num_hosts": cfg.num_hosts,
                "num_subnets": cfg.num_subnets,
                "vulnerability_density": cfg.vulnerability_density,
                "max_steps": cfg.max_steps,
            }
            for name, cfg in PRESETS.items()
        }
    }


@app.post("/train", tags=["Training"])
def train(req: TrainRequest):
    """Train a new RL agent. Blocks until training is complete."""
    if _state["training_active"]:
        raise HTTPException(status_code=409, detail="Training already in progress.")

    try:
        from arca.core.config import ARCAConfig
        from arca.core.agent import ARCAAgent
        from arca.sim.environment import NetworkEnv

        cfg = ARCAConfig.default()
        cfg.env.preset = req.preset
        cfg.rl.algorithm = req.algorithm
        cfg.rl.learning_rate = req.learning_rate
        cfg.verbose = 0
        cfg.ensure_dirs()

        env = NetworkEnv.from_preset(req.preset, cfg=cfg)
        agent = ARCAAgent(env=env, cfg=cfg)

        _state["training_active"] = True
        start = time.time()
        agent.train(timesteps=req.timesteps, progress_bar=False)
        elapsed = round(time.time() - start, 2)

        _state["agent"] = agent
        _state["env"] = env
        _state["cfg"] = cfg
        _state["training_active"] = False
        _state["last_trained_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        _state["total_timesteps_trained"] += req.timesteps

        return {
            "status": "success",
            "preset": req.preset,
            "algorithm": req.algorithm,
            "timesteps": req.timesteps,
            "elapsed_seconds": elapsed,
            "model_ready": True,
        }

    except Exception as e:
        _state["training_active"] = False
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/audit", tags=["Audit"])
def audit(req: AuditRequest):
    """Run a security audit episode and return a structured report."""
    try:
        from arca.core.config import ARCAConfig
        from arca.core.agent import ARCAAgent
        from arca.sim.environment import NetworkEnv

        agent = _state["agent"]
        env = _state["env"]
        cfg = _state["cfg"]

        if not req.use_existing or agent is None:
            cfg = ARCAConfig.default()
            cfg.env.preset = req.preset
            cfg.rl.n_steps = 64
            cfg.rl.batch_size = 32
            cfg.verbose = 0
            cfg.ensure_dirs()
            env = NetworkEnv.from_preset(req.preset, cfg=cfg)
            agent = ARCAAgent(env=env, cfg=cfg)
            agent.train(timesteps=req.timesteps, progress_bar=False)
            _state["agent"] = agent
            _state["env"] = env
            _state["cfg"] = cfg

        info = agent.run_episode()

        report = {
            "preset": cfg.env.preset,
            "total_reward": round(info.total_reward, 2),
            "steps": info.steps,
            "hosts_compromised": info.hosts_compromised,
            "hosts_discovered": info.hosts_discovered,
            "total_hosts": cfg.env.num_hosts,
            "goal_reached": info.goal_reached,
            "attack_path": info.attack_path,
            "summary": info.summary(),
        }

        if req.langgraph:
            agent.enable_langgraph()
            state = env.get_state_dict()
            reflection = agent.reflect(state)
            report["llm_analysis"] = reflection.get("reflection", "N/A")
            report["llm_plan"] = reflection.get("plan", "N/A")

        return report

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reflect", tags=["LangGraph"])
def reflect(req: ReflectRequest):
    """Run a LangGraph reflection cycle on a network state."""
    try:
        from arca.core.agent import ARCAAgent
        from arca.sim.environment import NetworkEnv
        from arca.core.config import ARCAConfig

        agent = _state["agent"]
        env = _state["env"]

        if agent is None:
            cfg = ARCAConfig.default()
            env = NetworkEnv.from_preset("small_office", cfg=cfg)
            env.reset()
            agent = ARCAAgent(env=env, cfg=cfg)

        state = req.state or env.get_state_dict()
        agent.enable_langgraph()
        result = agent.reflect(state)
        return {"status": "ok", "reflection": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))