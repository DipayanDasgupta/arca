"""
arca.core.trainer
~~~~~~~~~~~~~~~~~
ARCATrainer wraps Stable-Baselines3 training with rich logging,
eval callbacks, and TensorBoard support.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from arca.core.config import ARCAConfig
from arca.sim.environment import NetworkEnv


class ARCATrainer:
    def __init__(self, cfg: ARCAConfig, env: Optional[NetworkEnv] = None):
        self.cfg = cfg
        self.env = env or NetworkEnv(cfg=cfg)
        self._model = None

    def train(
        self,
        timesteps: int,
        callback=None,
        progress_bar: bool = True,
    ):
        try:
            from stable_baselines3 import PPO, A2C, DQN
            from stable_baselines3.common.callbacks import (
                EvalCallback, CheckpointCallback, CallbackList
            )
            from stable_baselines3.common.monitor import Monitor
        except ImportError as e:
            raise ImportError(f"stable-baselines3 not installed: {e}")

        rl = self.cfg.rl
        algo_map = {"PPO": PPO, "A2C": A2C, "DQN": DQN}
        AlgoCls = algo_map.get(rl.algorithm, PPO)

        monitored_env = Monitor(self.env)
        eval_env = Monitor(NetworkEnv(cfg=self.cfg))

        tb_log = self.cfg.rl.tensorboard_log
        if tb_log:
            Path(tb_log).mkdir(parents=True, exist_ok=True)

        kwargs = dict(
            policy=rl.policy,
            env=monitored_env,
            learning_rate=rl.learning_rate,
            gamma=rl.gamma,
            verbose=self.cfg.verbose,
            seed=self.cfg.seed,
            device=rl.device,
            tensorboard_log=tb_log,
        )
        if rl.algorithm == "PPO":
            kwargs.update(
                n_steps=rl.n_steps,
                batch_size=rl.batch_size,
                n_epochs=rl.n_epochs,
                gae_lambda=rl.gae_lambda,
                clip_range=rl.clip_range,
                ent_coef=rl.ent_coef,
            )

        self._model = AlgoCls(**kwargs)

        callbacks = []
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=self.cfg.model_dir,
            log_path=self.cfg.log_dir,
            eval_freq=rl.eval_freq,
            n_eval_episodes=rl.n_eval_episodes,
            deterministic=True,
            verbose=0,
        )
        ckpt_cb = CheckpointCallback(
            save_freq=rl.eval_freq,
            save_path=self.cfg.model_dir,
            name_prefix="arca_ckpt",
            verbose=0,
        )
        callbacks.extend([eval_cb, ckpt_cb])
        if callback:
            callbacks.append(callback)

        cb_list = CallbackList(callbacks)

        start = time.time()
        self._model.learn(
            total_timesteps=timesteps,
            callback=cb_list,
            progress_bar=progress_bar,
        )
        elapsed = time.time() - start
        if self.cfg.verbose:
            print(f"\n[ARCA] Training complete in {elapsed:.1f}s ({timesteps} steps)")

        return self._model

    @property
    def model(self):
        return self._model