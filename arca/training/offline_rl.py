"""
arca/training/offline_rl.py  (v3.3 — NEW)
==========================================
Offline RL via Behavioral Cloning (BC) on the top-K episodes.

Why BC and not CQL / IQL?
  BC is the simplest offline RL method. For ARCA's GNN policy it is:
    - Computationally cheap (no Q-network, no double networks)
    - Directly stable: supervised cross-entropy on best actions
    - Compatible with the existing GNNPolicy without code changes

How it works:
  1. Filter the episode buffer: keep top `top_fraction` by total_reward.
  2. Re-simulate each stored attack path in the environment to reconstruct
     (state, action) pairs (we use the stored attack_path strings as indices).
  3. Run supervised learning: minimise cross-entropy between policy logits
     and the expert action at each step.
  4. Fine-tune for `bc_epochs` epochs, then return loss stats.

Limitations / future work:
  - We re-simulate the attack path, so if the environment is stochastic the
    replayed trajectory may differ from the original. This is acceptable for
    BC since we're training on the expert's *intent*, not exact outcomes.
  - For conservative Q-learning, replace the BC loss with a CQL penalty.

Usage (called from ARCAAgent.offline_rl_finetune())::

    from arca.training.offline_rl import offline_bc_finetune
    stats = offline_bc_finetune(trainer, env, cfg, buf)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from torch_geometric.data import Batch
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False


@dataclass
class BCStats:
    n_episodes_used:  int
    n_steps_used:     int
    epochs:           int
    initial_loss:     float
    final_loss:       float
    elapsed_s:        float

    def summary(self) -> str:
        return (
            f"[OfflineRL/BC] episodes={self.n_episodes_used}  "
            f"steps={self.n_steps_used}  epochs={self.epochs}  "
            f"loss {self.initial_loss:.4f} → {self.final_loss:.4f}  "
            f"({self.elapsed_s:.1f}s)"
        )


def offline_bc_finetune(
    trainer,
    env,
    cfg,
    buf,
) -> dict:
    """
    Behavioral cloning fine-tune on top episodes.

    Parameters
    ----------
    trainer : CleanRLPPO
        The online trainer object (holds policy + optimizer).
    env : NetworkEnv
        The environment (used to reconstruct states from attack paths).
    cfg : ARCAConfig
        Configuration (reads cfg.offline_rl.*).
    buf : EpisodeBuffer
        Persistent episode buffer with stored records.

    Returns
    -------
    dict  with keys matching BCStats fields, plus "skipped" (bool).
    """
    orl  = cfg.offline_rl
    t0   = time.time()

    if not PYG_AVAILABLE:
        print("[OfflineRL] torch-geometric not available — skipping BC.")
        return {"skipped": True}

    # ── 1. Select top episodes ─────────────────────────────────────────────────
    all_recs  = buf._records
    n_keep    = max(1, int(len(all_recs) * orl.top_episode_fraction))
    top_recs  = sorted(all_recs, key=lambda r: r.total_reward, reverse=True)[:n_keep]

    print(
        f"[OfflineRL/BC] Selected {len(top_recs)}/{len(all_recs)} episodes "
        f"(top {orl.top_episode_fraction*100:.0f}%  "
        f"min_reward={top_recs[-1].total_reward:.0f})"
    )

    # ── 2. Reconstruct (obs, action) pairs from attack paths ───────────────────
    # Each attack_path entry looks like "0→2(CVE:CVE-2017-0144)"
    # We parse src→dst and try to find the corresponding action id.
    all_obs:     list = []
    all_actions: list[int] = []

    for rec in top_recs:
        if not rec.attack_path:
            continue
        try:
            obs, info = env.reset()
            for path_step in rec.attack_path:
                action_id = _path_to_action(path_step, env)
                if action_id is None:
                    continue
                pyg = trainer._to_pyg(obs, env=env)
                all_obs.append(pyg)
                all_actions.append(action_id)
                obs, _, term, trunc, info = env.step(action_id)
                if term or trunc:
                    break
        except Exception as e:
            continue  # skip malformed records

    n_steps = len(all_actions)
    if n_steps == 0:
        print("[OfflineRL/BC] No valid (obs, action) pairs found — skipping.")
        return {"skipped": True, "n_episodes_used": len(top_recs), "n_steps": 0}

    print(f"[OfflineRL/BC] Reconstructed {n_steps} (obs, action) pairs.")

    # ── 3. BC training loop ────────────────────────────────────────────────────
    policy    = trainer.policy
    bc_opt    = optim.Adam(policy.parameters(), lr=orl.bc_learning_rate)
    criterion = nn.CrossEntropyLoss()

    actions_tensor = torch.tensor(all_actions, dtype=torch.long, device=trainer.device)

    initial_loss = final_loss = 0.0

    for epoch in range(orl.bc_epochs):
        indices = np.random.permutation(n_steps)
        epoch_losses = []

        for start in range(0, n_steps, orl.bc_batch_size):
            mb_idx = indices[start : start + orl.bc_batch_size]
            if len(mb_idx) == 0:
                continue

            mb_obs     = Batch.from_data_list([all_obs[i] for i in mb_idx]).to(trainer.device)
            mb_actions = actions_tensor[mb_idx]

            logits, _ = policy(mb_obs)          # [B, n_actions]
            loss       = criterion(logits, mb_actions)

            bc_opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            bc_opt.step()

            epoch_losses.append(loss.item())

        epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        if epoch == 0:
            initial_loss = epoch_loss
        final_loss = epoch_loss

        print(f"  [BC] Epoch {epoch+1}/{orl.bc_epochs}  loss={epoch_loss:.4f}")

    elapsed = time.time() - t0
    stats   = BCStats(
        n_episodes_used = len(top_recs),
        n_steps_used    = n_steps,
        epochs          = orl.bc_epochs,
        initial_loss    = initial_loss,
        final_loss      = final_loss,
        elapsed_s       = elapsed,
    )
    print(stats.summary())
    return {
        "skipped":         False,
        "n_episodes_used": stats.n_episodes_used,
        "n_steps_used":    stats.n_steps_used,
        "epochs":          stats.epochs,
        "initial_loss":    stats.initial_loss,
        "final_loss":      stats.final_loss,
        "elapsed_s":       stats.elapsed_s,
    }


def _path_to_action(path_step: str, env) -> Optional[int]:
    """
    Parse a path string like "0→2(CVE:CVE-2017-0144)" and return the
    corresponding action integer for env.action_space.

    We match the EXPLOIT action with the first vulnerability on the
    destination host. SCAN and PIVOT are handled heuristically.
    Returns None if no valid action can be inferred.
    """
    try:
        # Parse src → dst
        arrow  = path_step.find("→")
        if arrow < 0:
            return None
        src_s  = path_step[:arrow].strip()
        rest   = path_step[arrow+1:]
        paren  = rest.find("(")
        dst_s  = rest[:paren].strip() if paren >= 0 else rest.strip()

        src = int(src_s)
        dst = int(dst_s)

        n = env.env_cfg.num_hosts
        e = env._NUM_EXPLOITS

        # Try EXPLOIT with first available exploit slot
        from arca.sim.action import ActionType
        for exploit_idx in range(e):
            action_id = (ActionType.EXPLOIT * n * e) + (dst * e) + exploit_idx
            if 0 <= action_id < env.action_space.n:
                # Verify the action is (approximately) valid
                mask = env.get_action_mask()
                if mask[action_id]:
                    return action_id

        # Fallback: SCAN the destination
        scan_id = (ActionType.SCAN * n * e) + (dst * e) + 0
        if 0 <= scan_id < env.action_space.n:
            return scan_id

        return None
    except Exception:
        return None