"""
arca/core/cleanrl_ppo.py  (v3.5 — AMP safe + fp16 masking)
============================================================
Changes vs v3.4:
  - AMP autocast context kept for speed on CUDA
  - No _NEG_INF constant here (fill logic moved entirely to gnn_policy._apply_mask)
  - GradScaler + unscale_ + step pattern unchanged
  - All other logic (action masking, GAE, memory, reflection) unchanged
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from torch_geometric.data import Data, Batch
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TB_AVAILABLE = True
except ImportError:
    TB_AVAILABLE = False

from arca.core.gnn_policy import GNNPolicy
from arca.core.config import ARCAConfig

try:
    from arca.memory.episode_buffer import EpisodeBuffer
    _BUFFER_AVAILABLE = True
except ImportError:
    EpisodeBuffer = None
    _BUFFER_AVAILABLE = False


# ── Rollout Buffer ─────────────────────────────────────────────────────────────

@dataclass
class RolloutBuffer:
    obs:       list
    actions:   torch.Tensor
    log_probs: torch.Tensor
    rewards:   torch.Tensor
    dones:     torch.Tensor
    values:    torch.Tensor
    masks:     torch.Tensor   # [n_steps, n_actions]  bool
    infos:     list = field(default_factory=list)

    @classmethod
    def empty(
        cls,
        n_steps:   int,
        n_actions: int,
        device:    torch.device,
    ) -> "RolloutBuffer":
        return cls(
            obs       = [None] * n_steps,
            actions   = torch.zeros(n_steps,            dtype=torch.long,  device=device),
            log_probs = torch.zeros(n_steps,                               device=device),
            rewards   = torch.zeros(n_steps,                               device=device),
            dones     = torch.zeros(n_steps,                               device=device),
            values    = torch.zeros(n_steps,                               device=device),
            masks     = torch.ones(n_steps, n_actions,  dtype=torch.bool,  device=device),
            infos     = [],
        )


# ── Trainer ────────────────────────────────────────────────────────────────────

class CleanRLPPO:
    """
    CleanRL-style PPO with masked GNN policy + AMP mixed precision.
    The fp16-safe masking is handled inside GNNPolicy._apply_mask (v3.5).
    """

    def __init__(
        self,
        env,
        cfg: ARCAConfig,
        reflection_callback: Optional[Callable[[dict], str]] = None,
    ):
        self.env                 = env
        self.cfg                 = cfg
        self.rl                  = cfg.rl
        self.reflection_callback = reflection_callback

        # ── Device ────────────────────────────────────────────────────────────
        if self.rl.device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(self.rl.device)

        print(f"[ARCA CleanRL-PPO] Device: {self.device}")

        # ── Mixed Precision (AMP) — CUDA only ─────────────────────────────────
        # AMP gives ~25-40% speedup on CUDA with negligible accuracy loss.
        # On CPU/MPS we disable it (autocast would slow things down there).
        # The fp16 overflow in masked_fill is fixed in GNNPolicy._apply_mask (v3.5).
        self._use_amp = (self.device.type == "cuda")
        if self._use_amp:
            self._scaler = torch.amp.GradScaler("cuda")
        else:
            self._scaler = None

        # ── Policy ────────────────────────────────────────────────────────────
        self._n_actions = env.action_space.n
        self.policy = GNNPolicy(
            feature_dim = env._HOST_FEATURES,
            hidden_dim  = self.rl.gnn_hidden_dim,
            num_actions = self._n_actions,
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr  = self.rl.learning_rate,
            eps = 1e-5,
        )

        # ── TensorBoard ───────────────────────────────────────────────────────
        self.writer: Optional[SummaryWriter] = None
        if self.rl.tensorboard_log and TB_AVAILABLE:
            self.writer = SummaryWriter(self.rl.tensorboard_log)

        # ── State ─────────────────────────────────────────────────────────────
        self.global_step       = 0
        self.episode_rewards:  list[float] = []
        self.episode_lengths:  list[int]   = []
        self._reward_modifiers: dict       = {}

        # ── Persistent memory ─────────────────────────────────────────────────
        self.memory_buffer: Optional[EpisodeBuffer] = None
        if _BUFFER_AVAILABLE and cfg.memory.enabled:
            try:
                self.memory_buffer = EpisodeBuffer(
                    memory_dir   = cfg.memory.memory_dir,
                    max_episodes = cfg.memory.max_episodes,
                )
            except Exception as e:
                print(f"[ARCA CleanRL-PPO] Memory buffer init failed: {e}")

    # ── Obs + mask helpers ────────────────────────────────────────────────────

    def _to_pyg(self, obs, env=None) -> "Data":
        target = env or self.env
        if PYG_AVAILABLE and isinstance(obs, Data):
            return obs.to(self.device)
        n        = target.env_cfg.num_hosts
        feat_dim = target._HOST_FEATURES
        node_x   = torch.tensor(
            obs.reshape(n, feat_dim), dtype=torch.float32, device=self.device
        )
        graph = target.graph
        if graph.number_of_edges() > 0:
            edges      = list(graph.edges())
            edge_index = torch.tensor(edges, dtype=torch.long, device=self.device).T
            if edge_index.dim() == 1:
                edge_index = edge_index.unsqueeze(0)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
        return Data(x=node_x, edge_index=edge_index)

    def _mask_from_info(self, info: dict) -> torch.Tensor:
        raw = info.get("action_mask")
        if raw is None:
            return torch.ones(self._n_actions, dtype=torch.bool, device=self.device)
        if isinstance(raw, torch.Tensor):
            return raw.bool().to(self.device)
        return torch.tensor(raw, dtype=torch.bool, device=self.device)

    # ── Rollout collection ────────────────────────────────────────────────────

    def _collect_rollout(self, n_steps: int):
        buf = RolloutBuffer.empty(n_steps, self._n_actions, self.device)

        obs, info = self.env.reset()
        mask      = self._mask_from_info(info)
        ep_reward = 0.0
        ep_length = 0

        for step in range(n_steps):
            self.global_step += 1

            pyg = self._to_pyg(obs)
            buf.obs[step]   = pyg
            buf.masks[step] = mask

            # AMP inference — faster on CUDA, no-op on CPU/MPS.
            # GNNPolicy._apply_mask (v3.5) handles fp16 overflow safely.
            with torch.no_grad():
                with torch.amp.autocast(
                    device_type=self.device.type, enabled=self._use_amp
                ):
                    action, log_prob, _, value = self.policy.get_action_and_value(
                        pyg, action_mask=mask.unsqueeze(0)
                    )

            buf.actions[step]   = action
            buf.log_probs[step] = log_prob
            buf.values[step]    = value

            obs, reward, terminated, truncated, info = self.env.step(action.item())
            reward = float(self._apply_modifiers(reward, info))
            mask   = self._mask_from_info(info)

            buf.rewards[step] = reward
            buf.dones[step]   = float(terminated or truncated)
            buf.infos.append(info)

            ep_reward += reward
            ep_length += 1

            if terminated or truncated:
                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_length)

                if self.writer:
                    self.writer.add_scalar(
                        "charts/ep_reward", ep_reward, self.global_step
                    )
                    self.writer.add_scalar(
                        "charts/ep_length", ep_length, self.global_step
                    )

                if len(self.episode_rewards) % 5 == 0:
                    mean_r = np.mean(self.episode_rewards[-10:])
                    print(
                        f"  step={self.global_step:>8,}  "
                        f"ep_reward={ep_reward:>8.2f}  "
                        f"mean10={mean_r:>8.2f}"
                    )

                # ── Save to persistent memory ──────────────────────────────
                ep_info = info.get("episode_info")
                if (
                    self.memory_buffer
                    and ep_info is not None
                    and ep_info.hosts_compromised
                    >= self.cfg.memory.min_compromised_to_record
                ):
                    try:
                        preset = getattr(
                            getattr(self.env, "env_cfg", None), "preset",
                            getattr(self.cfg.env, "preset", "unknown"),
                        )
                        self.memory_buffer.record(
                            preset            = preset,
                            total_reward      = ep_reward,
                            hosts_compromised = ep_info.hosts_compromised,
                            hosts_total       = getattr(
                                getattr(self.env, "env_cfg", self.cfg.env),
                                "num_hosts", 10,
                            ),
                            steps             = ep_length,
                            goal_reached      = ep_info.goal_reached,
                            attack_path       = list(ep_info.attack_path),
                            reward_modifiers  = dict(self._reward_modifiers),
                            reflection        = "",
                            severity_score    = 0.0,
                        )
                    except Exception as e:
                        print(f"  [Memory] Save failed: {e}")

                ep_reward = 0.0
                ep_length = 0
                obs, info = self.env.reset()
                mask      = self._mask_from_info(info)

            if (
                self.rl.online_reflection_interval > 0
                and self.global_step % self.rl.online_reflection_interval == 0
            ):
                self._run_reflection()

        return buf, obs, mask

    # ── GAE ────────────────────────────────────────────────────────────────────

    def _compute_gae(
        self,
        buf:        RolloutBuffer,
        last_obs,
        last_mask:  torch.Tensor,
        gamma:      float = 0.99,
        gae_lambda: float = 0.95,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n          = len(buf.rewards)
        advantages = torch.zeros(n, device=self.device)

        last_pyg = self._to_pyg(last_obs)
        with torch.no_grad():
            with torch.amp.autocast(
                device_type=self.device.type, enabled=self._use_amp
            ):
                last_val = self.policy.get_value(
                    last_pyg, action_mask=last_mask.unsqueeze(0)
                )

        last_gae = 0.0
        for t in reversed(range(n)):
            next_non_terminal = 1.0 - buf.dones[t]
            next_val          = (
                last_val if t == n - 1
                else buf.values[t + 1]
            )
            delta    = (
                buf.rewards[t]
                + gamma * next_val * next_non_terminal
                - buf.values[t]
            )
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        return advantages, advantages + buf.values

    # ── PPO update (AMP) ──────────────────────────────────────────────────────

    def _update(
        self,
        buf:        RolloutBuffer,
        advantages: torch.Tensor,
        returns:    torch.Tensor,
    ) -> dict:
        n       = len(buf.rewards)
        indices = np.arange(n)

        pg_losses, v_losses, ent_losses, kl_approxs = [], [], [], []

        for _ in range(self.rl.n_epochs):
            np.random.shuffle(indices)

            for start in range(0, n, self.rl.batch_size):
                mb_idx = indices[start : start + self.rl.batch_size]
                if len(mb_idx) == 0:
                    continue

                mb_obs  = Batch.from_data_list(
                    [buf.obs[i] for i in mb_idx]
                ).to(self.device)

                mb_act   = buf.actions[mb_idx]
                mb_lp    = buf.log_probs[mb_idx]
                mb_adv   = advantages[mb_idx]
                mb_ret   = returns[mb_idx]
                mb_masks = buf.masks[mb_idx]    # [B, n_actions]

                if mb_adv.std() > 1e-8:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # ── Forward pass under AMP autocast ───────────────────────
                # GNNPolicy._apply_mask (v3.5) uses dtype-safe fill values,
                # so the fp16 overflow from v3.4 is fixed.
                with torch.amp.autocast(
                    device_type=self.device.type, enabled=self._use_amp
                ):
                    _, new_lp, entropy, new_val = self.policy.get_action_and_value(
                        mb_obs, mb_act, action_mask=mb_masks
                    )

                    ratio    = (new_lp - mb_lp).exp()
                    pg1      = -mb_adv * ratio
                    pg2      = -mb_adv * ratio.clamp(
                        1 - self.rl.clip_range, 1 + self.rl.clip_range
                    )
                    pg_loss  = torch.max(pg1, pg2).mean()
                    v_loss   = 0.5 * ((new_val - mb_ret) ** 2).mean()
                    ent_loss = entropy.mean()
                    loss     = pg_loss + 0.5 * v_loss - self.rl.ent_coef * ent_loss

                # ── Backward pass (scaler-aware on CUDA) ──────────────────
                self.optimizer.zero_grad()
                if self._use_amp and self._scaler is not None:
                    self._scaler.scale(loss).backward()
                    self._scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                    self._scaler.step(self.optimizer)
                    self._scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                    self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - (new_lp - mb_lp)).mean()

                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                ent_losses.append(ent_loss.item())
                kl_approxs.append(approx_kl.item())

        return {
            "pg_loss":   float(np.mean(pg_losses)),
            "v_loss":    float(np.mean(v_losses)),
            "entropy":   float(np.mean(ent_losses)),
            "approx_kl": float(np.mean(kl_approxs)),
        }

    # ── Reward modifiers ──────────────────────────────────────────────────────

    def _apply_modifiers(self, reward: float, info: dict) -> float:
        if not self._reward_modifiers:
            return reward
        ar = info.get("action_result", {})
        if self._reward_modifiers.get("boost_critical"):
            ch = ar.get("compromised_host")
            if ch is not None:
                host = self.env.hosts.get(ch)
                if host and getattr(host, "is_critical", False):
                    reward *= self._reward_modifiers.get("critical_mult", 1.5)
        if self._reward_modifiers.get("penalize_failed_exploit"):
            if not ar.get("success", True) and ar.get("type") == "EXPLOIT":
                reward -= self._reward_modifiers.get("fail_delta", 0.3)
        if self._reward_modifiers.get("penalize_redundant_scan"):
            if ar.get("type") == "SCAN" and not ar.get("discovered_hosts"):
                reward -= self._reward_modifiers.get("redundant_scan_delta", 0.3)
        return reward

    def _run_reflection(self) -> None:
        if self.reflection_callback is None:
            if self.memory_buffer:
                mods = self.memory_buffer.infer_reward_mods()
                if mods:
                    self._reward_modifiers.update(mods)
                    print(
                        f"  [Memory Reflection @ step {self.global_step}] "
                        f"Mods: {mods}"
                    )
            return
        try:
            state = self.env.get_state_dict()
            state["global_step"]    = self.global_step
            state["recent_rewards"] = self.episode_rewards[-10:]
            state["mean_reward"]    = (
                float(np.mean(self.episode_rewards[-10:]))
                if self.episode_rewards else 0.0
            )
            if self.memory_buffer:
                state["memory_summary"] = self.memory_buffer.format_for_llm(n=3)
            critique = self.reflection_callback(state)
            if critique:
                self._parse_critique(critique)
                print(
                    f"  [Reflection @ step {self.global_step}] "
                    f"Mods: {self._reward_modifiers}"
                )
        except Exception as e:
            print(f"  [Reflection] Failed: {e}")

    def _parse_critique(self, critique: str) -> None:
        c    = critique.lower()
        mods: dict = {}
        if any(w in c for w in ["critical", "high-value", "crown"]):
            mods["boost_critical"] = True
            mods["critical_mult"]  = 1.5
        if any(w in c for w in ["scan spam", "too many scans", "redundant scan"]):
            mods["penalize_redundant_scan"] = True
            mods["redundant_scan_delta"]    = 0.2
        if any(w in c for w in ["failed exploit", "avoid failed"]):
            mods["penalize_failed_exploit"] = True
            mods["fail_delta"]              = 0.2
        self._reward_modifiers.update(mods)

    # ── Main training loop ────────────────────────────────────────────────────

    def learn(self, total_timesteps: int) -> "CleanRLPPO":
        amp_str = "ENABLED (fp16-safe via _apply_mask)" if self._use_amp else "disabled"
        print(f"\n[ARCA CleanRL-PPO] Starting training")
        print(f"  Total steps  : {total_timesteps:,}")
        print(f"  Device       : {self.device}")
        print(f"  GNN hidden   : {self.rl.gnn_hidden_dim}")
        print(f"  Rollout steps: {self.rl.n_steps}")
        print(f"  Batch size   : {self.rl.batch_size}")
        print(f"  PPO epochs   : {self.rl.n_epochs}")
        print(f"  Entropy coef : {self.rl.ent_coef}")
        print(f"  Action masking: ENABLED (dynamic size + AMP-safe)")
        print(f"  Mixed prec.  : {amp_str}\n")

        # Seed reward mods from memory
        if (
            self.memory_buffer
            and self.cfg.memory.seed_reward_mods_from_memory
            and not self._reward_modifiers
        ):
            mods = self.memory_buffer.infer_reward_mods()
            if mods:
                self._reward_modifiers.update(mods)
                print(f"[ARCA] Seeded reward mods from memory: {mods}")
            stats = self.memory_buffer.get_stats()
            if stats:
                print(
                    f"[ARCA] Memory: {stats['total_episodes']} past episodes, "
                    f"max_reward={stats['max_reward']:.1f}, "
                    f"goal_rate={stats['goal_rate']*100:.0f}%"
                )

        t0   = time.time()
        done = 0

        while done < total_timesteps:
            n_collect                = min(self.rl.n_steps, total_timesteps - done)
            buf, last_obs, last_mask = self._collect_rollout(n_collect)
            adv, ret                 = self._compute_gae(
                buf, last_obs, last_mask, self.rl.gamma, self.rl.gae_lambda
            )
            metrics = self._update(buf, adv, ret)
            done   += n_collect

            if self.writer:
                for k, v in metrics.items():
                    self.writer.add_scalar(f"losses/{k}", v, self.global_step)
                sps = int(done / (time.time() - t0 + 1e-8))
                self.writer.add_scalar("charts/SPS", sps, self.global_step)

        elapsed = time.time() - t0
        mean_r  = (
            float(np.mean(self.episode_rewards[-20:]))
            if self.episode_rewards else 0.0
        )
        sps = int(total_timesteps / (elapsed + 1e-8))
        print(
            f"\n[ARCA CleanRL-PPO] Done in {elapsed:.1f}s  |  "
            f"mean(last20ep)={mean_r:.2f}  |  {sps:,} SPS"
        )

        if self.memory_buffer:
            stats = self.memory_buffer.get_stats()
            if stats:
                print(
                    f"[ARCA Memory] Stored: {stats['total_episodes']} episodes  |  "
                    f"goal_rate={stats['goal_rate']*100:.0f}%"
                )

        if self.writer:
            self.writer.close()
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(
        self,
        obs,
        deterministic: bool = True,
        action_mask:   "np.ndarray | None" = None,
    ):
        pyg = self._to_pyg(obs)
        if action_mask is not None:
            mask = torch.tensor(
                action_mask, dtype=torch.bool, device=self.device
            ).unsqueeze(0)
        else:
            try:
                mask_np = self.env.get_action_mask()
                mask    = torch.tensor(
                    mask_np, dtype=torch.bool, device=self.device
                ).unsqueeze(0)
            except Exception:
                mask = None

        with torch.no_grad():
            with torch.amp.autocast(
                device_type=self.device.type, enabled=self._use_amp
            ):
                action = self.policy.get_action(
                    pyg, deterministic=deterministic, action_mask=mask
                )
        return action.cpu().numpy(), None

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> str:
        full = path if path.endswith(".pt") else path + ".pt"
        torch.save(
            {
                "policy":           self.policy.state_dict(),
                "optimizer":        self.optimizer.state_dict(),
                "global_step":      self.global_step,
                "rewards":          self.episode_rewards,
                "reward_modifiers": self._reward_modifiers,
                "n_actions":        self._n_actions,
            },
            full,
        )
        return full

    def load(self, path: str) -> "CleanRLPPO":
        full = path if path.endswith(".pt") else path + ".pt"
        ckpt = torch.load(full, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.global_step       = ckpt.get("global_step", 0)
        self.episode_rewards   = ckpt.get("rewards", [])
        self._reward_modifiers = ckpt.get("reward_modifiers", {})
        return self