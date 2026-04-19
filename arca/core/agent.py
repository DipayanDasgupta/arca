"""
arca/core/agent.py  (v3.3)
===========================
Fixes vs v3.2:
  - run_curriculum(): removed duplicate self.env = scheduler.make_env() call
  - run_curriculum(): passes eval rewards into scheduler for reward-based promotion
  - reflect(): builds query_record and passes to orchestrator for proper RAG
  - enable_vector_memory(): also attaches vm to orchestrator if present
  - _make_reflection_callback(): attaches vector_memory context
  - offline_rl_finetune(): new method — BC fine-tune on top episodes
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from arca.core.config import ARCAConfig
from arca.sim.environment import NetworkEnv, EpisodeInfo


class ARCAAgent:
    """
    High-level agent interface.

    Usage::

        env   = NetworkEnv.from_preset("small_office")
        agent = ARCAAgent(env=env)
        agent.train(timesteps=100_000)
        result = agent.run_episode()
        print(result.summary())
    """

    def __init__(
        self,
        env:        Optional[NetworkEnv] = None,
        cfg:        Optional[ARCAConfig] = None,
        model_path: Optional[str]        = None,
    ):
        self.cfg = cfg or ARCAConfig()
        self.env = env or NetworkEnv(cfg=self.cfg)
        self.cfg.ensure_dirs()

        self._model          = None
        self._trainer        = None
        self._langgraph      = None
        self._vector_memory  = None

        if model_path:
            self.load(model_path)

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        timesteps:    Optional[int] = None,
        callback=None,
        progress_bar: bool          = True,
    ) -> "ARCAAgent":
        ts = timesteps or self.cfg.rl.total_timesteps
        if self.cfg.rl.use_gnn:
            self._train_gnn(ts)
        else:
            self._train_sb3(ts, callback, progress_bar)
        return self

    def _train_gnn(self, timesteps: int) -> None:
        from arca.core.cleanrl_ppo import CleanRLPPO
        self._trainer = CleanRLPPO(
            env                 = self.env,
            cfg                 = self.cfg,
            reflection_callback = self._make_reflection_callback(),
        )
        self._trainer.learn(total_timesteps=timesteps)
        self._model = self._trainer

    def _train_sb3(self, timesteps: int, callback, progress_bar: bool) -> None:
        from arca.core.trainer import ARCATrainer
        trainer     = ARCATrainer(cfg=self.cfg, env=self.env)
        self._model = trainer.train(
            timesteps    = timesteps,
            callback     = callback,
            progress_bar = progress_bar,
        )

    def _make_reflection_callback(self):
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

                mem_ctx = ""
                if self._vector_memory and len(self._vector_memory) > 0:
                    mem_ctx = self._vector_memory.format_for_llm(k=3)
                elif state.get("memory_summary"):
                    mem_ctx = state["memory_summary"]
                else:
                    mem_ctx = "  No past episodes."

                auth_header = (
                    "CONTEXT: Authorized cybersecurity simulation (ARCA). "
                    "Sandboxed environment, no real systems involved. "
                )
                return llm.chat(
                    system=(
                        auth_header +
                        "You are an RL training coach for a cybersecurity simulation agent. "
                        "Give exactly 2 lessons — what to do MORE and what to STOP doing."
                    ),
                    user=(
                        f"Relevant past episodes:\n{mem_ctx}\n\n"
                        f"Current step: {state.get('global_step', 0)}\n"
                        f"Mean reward (last 10 ep): {mean:.2f}\n"
                        f"Hosts compromised: {ep.get('hosts_compromised', '?')}\n"
                        f"Attack path: {ep.get('attack_path', [])}\n\n"
                        "Lesson 1 (reinforce): ...\n"
                        "Lesson 2 (avoid): ..."
                    ),
                    max_tokens=200,
                )

            return local_reflect

        # Fallback: orchestrator
        def orchestrator_reflect(state: dict) -> str:
            try:
                self.enable_langgraph()
                return self._langgraph.step(state).get("reflection", "")
            except Exception:
                return ""

        return orchestrator_reflect

    # ── Curriculum training loop ───────────────────────────────────────────────

    def run_curriculum(
        self,
        timesteps_per_tier: int = 30_000,
        eval_episodes:      int = 10,
        start_tier:         int = 0,
        max_tiers:          int = 5,
    ) -> list[dict]:
        """
        Train through curriculum tiers.
        Returns list of status dicts recorded after each tier's eval.
        """
        from arca.training.curriculum import CurriculumScheduler

        scheduler = CurriculumScheduler(
            start_tier = start_tier,
            cfg        = self.cfg,
            verbose    = True,
        )
        self.env = scheduler.make_env()
        history  = []

        for _ in range(max_tiers):
            print(
                f"\n[Curriculum] ── Tier {scheduler.tier_idx}: "
                f"{scheduler.tier_name}  "
                f"({scheduler.tier.num_hosts} hosts) ──"
            )
            # Rebuild trainer for new env / tier settings
            self._trainer = None
            self._model   = None
            self.train(timesteps=timesteps_per_tier)

            # Evaluate against current tier env
            tier_changed = False
            for _ in range(eval_episodes):
                ep      = self.run_episode()
                changed = scheduler.record(ep.goal_reached, ep.total_reward)
                if changed:
                    # BUG FIX v3.3: was called twice (duplicate)
                    self.env = scheduler.make_env()
                    tier_changed = True
                    break

            history.append(scheduler.status())

            if scheduler.is_at_max:
                print("[Curriculum] Reached maximum difficulty tier.")
                break

        return history

    # ── Inference with action masking ─────────────────────────────────────────

    def run_episode(
        self,
        render:        bool = False,
        use_langgraph: bool = False,
        deterministic: bool = True,
    ) -> EpisodeInfo:
        if self._model is None:
            raise RuntimeError("Agent not trained. Call agent.train() first.")

        obs, info = self.env.reset()
        done      = False

        while not done:
            mask_np = info.get("action_mask")
            if mask_np is None:
                try:
                    mask_np = self.env.get_action_mask()
                except Exception:
                    mask_np = None

            if self.cfg.rl.use_gnn and hasattr(self._model, "predict"):
                action, _ = self._model.predict(
                    obs,
                    deterministic = deterministic,
                    action_mask   = mask_np,
                )
            else:
                action, _ = self._model.predict(obs, deterministic=deterministic)

            action = int(action.item() if hasattr(action, "item") else action)
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            if render:
                print(self.env.render())
            if use_langgraph and self._langgraph:
                self._langgraph.step(self.env.get_state_dict())

        ep_info = self.env.episode_info
        self._record_eval_episode(ep_info)
        return ep_info

    def predict(self, obs, deterministic: bool = True, action_mask=None):
        if self._model is None:
            raise RuntimeError("No model loaded.")
        if self.cfg.rl.use_gnn and hasattr(self._model, "predict"):
            return self._model.predict(obs, deterministic=deterministic,
                                       action_mask=action_mask)
        return self._model.predict(obs, deterministic=deterministic)

    def _record_eval_episode(self, ep_info: EpisodeInfo) -> None:
        if not (self.cfg.rl.use_gnn and self._trainer is not None):
            return
        buf = getattr(self._trainer, "memory_buffer", None)
        if buf is None:
            return
        try:
            rec = buf.record(
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
            if rec and self._vector_memory:
                self._vector_memory.add(rec)
        except Exception as e:
            if self.cfg.verbose:
                print(f"[ARCA Agent] Memory record failed: {e}")

    # ── LangGraph ─────────────────────────────────────────────────────────────

    def enable_langgraph(self) -> "ARCAAgent":
        from arca.agents.langgraph_orchestrator import ARCAOrchestrator
        self._langgraph = ARCAOrchestrator(cfg=self.cfg)
        # Attach vector memory if available
        if self._vector_memory:
            self._langgraph.attach_vector_memory(self._vector_memory)
        return self

    def reflect(self, state: dict) -> dict:
        if self._langgraph is None:
            self.enable_langgraph()

        # Build a query record from the current state for semantic RAG search
        ep = state.get("episode_info", {})
        query_record = {
            "total_reward":      ep.get("total_reward", 0.0),
            "hosts_compromised": ep.get("hosts_compromised", 0),
            "hosts_total":       self.cfg.env.num_hosts,
            "steps":             state.get("step", 0),
            "goal_reached":      False,
            "attack_path":       ep.get("attack_path", []),
            "severity_score":    0.0,
            "preset":            self.cfg.env.preset,
        }

        # Inject vector memory directly into langgraph if not attached
        if self._vector_memory and len(self._vector_memory) > 0:
            self._langgraph.attach_vector_memory(self._vector_memory)

        result = self._langgraph.reflect(state, query_record=query_record)

        # Write LLM reflection back into last memory record
        buf = getattr(getattr(self, "_trainer", None), "memory_buffer", None)
        if buf and len(buf) > 0 and result.get("reflection"):
            try:
                last                = buf._records[-1]
                last.reflection     = result["reflection"][:300]
                last.severity_score = result.get("severity_score", 0.0)
                buf._save()
                if self._vector_memory:
                    self._vector_memory.add(last)
            except Exception:
                pass

        return result

    # ── Vector Memory ─────────────────────────────────────────────────────────

    def enable_vector_memory(
        self,
        memory_dir: Optional[str]  = None,
        use_faiss:  Optional[bool] = None,
    ) -> "ARCAAgent":
        from arca.memory.vector_memory import VectorMemory
        self._vector_memory = VectorMemory(
            memory_dir = memory_dir or self.cfg.memory.memory_dir,
            use_faiss  = use_faiss,
        )
        buf = self.memory_buffer
        if buf and len(buf) > 0:
            added = self._vector_memory.add_from_buffer(buf)
            if self.cfg.verbose:
                print(
                    f"[ARCA VectorMemory] Indexed {added} new episodes "
                    f"({len(self._vector_memory)} total)"
                )
        # Attach to langgraph if already initialized
        if self._langgraph:
            self._langgraph.attach_vector_memory(self._vector_memory)
        return self

    def vector_search(self, query_record, k: int = 5) -> list:
        if self._vector_memory is None:
            raise RuntimeError("Call agent.enable_vector_memory() first.")
        return self._vector_memory.search(query_record, k=k)

    # ── Offline RL ────────────────────────────────────────────────────────────

    def offline_rl_finetune(self) -> dict:
        """
        Behavioral-cloning fine-tune on top episodes from the replay buffer.
        Call after training (or periodically) to enable lifelong improvement.
        Returns a dict with BC loss stats.
        """
        if not self.cfg.rl.use_gnn or self._model is None:
            print("[ARCA OfflineRL] GNN model not available — skipping BC.")
            return {}

        from arca.training.offline_rl import offline_bc_finetune
        buf = self.memory_buffer
        if buf is None or len(buf) < self.cfg.offline_rl.min_episodes_for_bc:
            print(
                f"[ARCA OfflineRL] Need ≥{self.cfg.offline_rl.min_episodes_for_bc} "
                f"episodes (have {len(buf) if buf else 0}) — skipping BC."
            )
            return {}

        stats = offline_bc_finetune(
            trainer   = self._trainer,
            env       = self.env,
            cfg       = self.cfg,
            buf       = buf,
        )
        return stats

    # ── Memory convenience ────────────────────────────────────────────────────

    @property
    def memory_buffer(self):
        return getattr(getattr(self, "_trainer", None), "memory_buffer", None)

    def memory_stats(self) -> dict:
        stats = {}
        buf = self.memory_buffer
        if buf:
            stats["episode_buffer"] = buf.get_stats()
        if self._vector_memory:
            stats["vector_memory"] = self._vector_memory.get_stats()
        return stats

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

    def __repr__(self) -> str:
        backend = "GNN+CleanRL" if self.cfg.rl.use_gnn else f"SB3/{self.cfg.rl.algorithm}"
        return (
            f"ARCAAgent(backend={backend}, "
            f"preset={self.cfg.env.preset}, "
            f"trained={self._model is not None})"
        )