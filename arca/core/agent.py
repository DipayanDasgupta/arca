"""
arca/core/agent.py  (v3.1)
===========================
ARCAAgent routes between:
  - CleanRLPPO + GNNPolicy   (use_gnn=True,  default)
  - Stable-Baselines3 PPO    (use_gnn=False, legacy)

v3.1 additions:
  - Records eval episodes to persistent EpisodeBuffer
  - Saves LLM reflection text back into the memory record
  - _make_reflection_callback uses memory buffer for enriched prompts
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from arca.core.config import ARCAConfig
from arca.sim.environment import NetworkEnv, EpisodeInfo


class ARCAAgent:
    """
    High-level agent interface — unchanged public API, upgraded internals.

    Usage::

        env   = NetworkEnv.from_preset("small_office")
        agent = ARCAAgent(env=env)
        agent.train(timesteps=100_000)
        result = agent.run_episode(render=True)
        print(result.summary())
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

        self._model    = None   # SB3 model (use_gnn=False)
        self._trainer  = None   # CleanRLPPO (use_gnn=True)
        self._langgraph = None

        if model_path:
            self.load(model_path)

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        timesteps: Optional[int] = None,
        callback=None,
        progress_bar: bool = True,
    ) -> "ARCAAgent":
        ts = timesteps or self.cfg.rl.total_timesteps

        if self.cfg.rl.use_gnn:
            self._train_gnn(ts)
        else:
            self._train_sb3(ts, callback, progress_bar)

        return self

    def _train_gnn(self, timesteps: int) -> None:
        """CleanRL + GNNPolicy training."""
        from arca.core.cleanrl_ppo import CleanRLPPO

        reflection_cb = self._make_reflection_callback()
        self._trainer = CleanRLPPO(
            env                  = self.env,
            cfg                  = self.cfg,
            reflection_callback  = reflection_cb,
        )
        self._trainer.learn(total_timesteps=timesteps)
        self._model = self._trainer   # unified predict interface

    def _train_sb3(self, timesteps: int, callback, progress_bar: bool) -> None:
        """Legacy Stable-Baselines3 training."""
        from arca.core.trainer import ARCATrainer
        trainer = ARCATrainer(cfg=self.cfg, env=self.env)
        self._model = trainer.train(
            timesteps    = timesteps,
            callback     = callback,
            progress_bar = progress_bar,
        )

    def _make_reflection_callback(self):
        """Return a callable that runs LLM critique and returns a string."""
        if not self.cfg.llm.enabled:
            return None

        if self.cfg.llm.use_local_llm:
            from arca.llm.local_llm import LocalLLM
            llm = LocalLLM(
                model_key     = self.cfg.llm.local_model_key,
                model_dir     = self.cfg.llm.local_model_dir,
                n_gpu_layers  = self.cfg.llm.local_n_gpu_layers,
                auto_download = self.cfg.llm.auto_download_model,
            )
            if not llm.available:
                print("[ARCA Agent] Local LLM not available — reflection disabled.")
                return None

            def local_reflect(state: dict) -> str:
                ep   = state.get("episode_info", {})
                mean = state.get("mean_reward", 0.0)
                mem  = state.get("memory_summary", "  No past episodes.")
                prompt_system = (
                    "You are an RL training coach for a cybersecurity agent. "
                    "Analyse the episode statistics and give a 2-sentence critique "
                    "focused on which actions the agent should prioritise or avoid."
                )
                prompt_user = (
                    f"Step: {state.get('global_step', 0)}\n"
                    f"Mean reward (last 10 ep): {mean:.2f}\n"
                    f"Hosts compromised: {ep.get('hosts_compromised', '?')}\n"
                    f"Attack path: {ep.get('attack_path', [])}\n"
                    f"Lifelong memory summary:\n{mem}\n\n"
                    "In 2 sentences: what should the agent do differently?"
                )
                return llm.chat(
                    system     = prompt_system,
                    user       = prompt_user,
                    max_tokens = 200,
                )

            return local_reflect

        # Fallback: orchestrator-based reflection (Ollama / Groq / rule-based)
        def orchestrator_reflect(state: dict) -> str:
            try:
                self.enable_langgraph()
                result = self._langgraph.step(state)
                return result.get("reflection", "")
            except Exception:
                return ""

        return orchestrator_reflect

    # ── Inference ─────────────────────────────────────────────────────────────

    def run_episode(
        self,
        render: bool = False,
        use_langgraph: bool = False,
        deterministic: bool = True,
    ) -> EpisodeInfo:
        if self._model is None:
            raise RuntimeError(
                "Agent not trained. Call agent.train() first or load a model."
            )

        obs, info = self.env.reset()
        done = False

        while not done:
            action, _ = self._model.predict(obs, deterministic=deterministic)
            if hasattr(action, "item"):
                action = action.item()
            else:
                action = int(action)

            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            if render:
                print(self.env.render())

            if use_langgraph and self._langgraph:
                self._langgraph.step(self.env.get_state_dict())

        ep_info = self.env.episode_info

        # ── Record eval episode to persistent memory ───────────────────────
        self._record_eval_episode(ep_info)

        return ep_info

    def _record_eval_episode(self, ep_info: EpisodeInfo) -> None:
        """Write a completed eval episode into the CleanRL trainer's memory buffer."""
        # Only available when using GNN trainer (has memory_buffer attribute)
        if not (self.cfg.rl.use_gnn and self._trainer is not None):
            return
        buf = getattr(self._trainer, "memory_buffer", None)
        if buf is None:
            return
        if ep_info.hosts_compromised < self.cfg.memory.min_compromised_to_record:
            return
        try:
            buf.record(
                preset            = self.cfg.env.preset,
                total_reward      = ep_info.total_reward,
                hosts_compromised = ep_info.hosts_compromised,
                hosts_total       = self.cfg.env.num_hosts,
                steps             = ep_info.steps,
                goal_reached      = ep_info.goal_reached,
                attack_path       = list(ep_info.attack_path),
                reward_modifiers  = dict(
                    getattr(self._trainer, "_reward_modifiers", {})
                ),
                reflection        = "",
                severity_score    = 0.0,
            )
        except Exception as e:
            if self.cfg.verbose:
                print(f"[ARCA Agent] Memory record failed: {e}")

    def predict(self, obs, deterministic: bool = True):
        if self._model is None:
            raise RuntimeError("No model loaded.")
        return self._model.predict(obs, deterministic=deterministic)

    # ── LangGraph ─────────────────────────────────────────────────────────────

    def enable_langgraph(self) -> "ARCAAgent":
        from arca.agents.langgraph_orchestrator import ARCAOrchestrator
        self._langgraph = ARCAOrchestrator(cfg=self.cfg)
        return self

    def reflect(self, state: dict) -> dict:
        if self._langgraph is None:
            self.enable_langgraph()
        result = self._langgraph.reflect(state)

        # ── Write LLM reflection back into the last memory record ──────────
        buf = getattr(getattr(self, "_trainer", None), "memory_buffer", None)
        if buf and len(buf) > 0 and result.get("reflection"):
            try:
                last = buf._records[-1]
                last.reflection    = result["reflection"][:300]
                last.severity_score = result.get("severity_score", 0.0)
                buf._save()
            except Exception:
                pass

        return result

    # ── Memory convenience ────────────────────────────────────────────────────

    @property
    def memory_buffer(self):
        """Convenience accessor for the CleanRL trainer's memory buffer."""
        return getattr(getattr(self, "_trainer", None), "memory_buffer", None)

    def memory_stats(self) -> dict:
        """Return persistent memory statistics."""
        buf = self.memory_buffer
        return buf.get_stats() if buf else {}

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Optional[str] = None) -> str:
        if self._model is None:
            raise RuntimeError("No model to save.")

        save_path = path or str(Path(self.cfg.model_dir) / "arca_model")

        if self.cfg.rl.use_gnn and hasattr(self._model, "save"):
            final = self._model.save(save_path)
        else:
            self._model.save(save_path)
            final = save_path

        if self.cfg.verbose:
            print(f"[ARCA] Model saved → {final}")
        return final

    def load(self, path: str) -> "ARCAAgent":
        if self.cfg.rl.use_gnn:
            from arca.core.cleanrl_ppo import CleanRLPPO
            self._trainer = CleanRLPPO(env=self.env, cfg=self.cfg)
            self._trainer.load(path)
            self._model = self._trainer
        else:
            try:
                from stable_baselines3 import PPO, A2C, DQN
                algo_map = {"PPO": PPO, "A2C": A2C, "DQN": DQN}
                Cls = algo_map.get(self.cfg.rl.algorithm, PPO)
                self._model = Cls.load(path, env=self.env)
            except Exception as e:
                raise RuntimeError(f"Failed to load model: {e}")
        return self

    # ── Repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        backend = "GNN+CleanRL" if self.cfg.rl.use_gnn else f"SB3/{self.cfg.rl.algorithm}"
        trained = self._model is not None
        return (
            f"ARCAAgent(backend={backend}, preset={self.cfg.env.preset}, "
            f"trained={trained})"
        )