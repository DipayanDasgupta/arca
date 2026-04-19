"""
arca/training/curriculum.py  (v3.3)
=====================================
Key fixes vs v3.2:
  1. DifficultyTier gains promote_reward_threshold — promotion fires if
     EITHER goal_rate OR mean_reward exceeds the threshold (OR logic).
     This unblocks the agent when it has good rewards but 0% goal rate.
  2. Micro tier window reduced from 15 → 8 so promotion evaluates sooner.
  3. promote_reward_threshold set to ~800 for micro (well below the
     1100–1979 rewards we actually see in training logs).
  4. record() prints a concise progress line every 5 episodes.
  5. _apply_tier() also sets cfg.rl.ent_coef per tier (harder → less entropy).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Optional

from arca.core.config import ARCAConfig, EnvConfig


@dataclass
class DifficultyTier:
    name:                    str
    num_hosts:               int
    num_subnets:             int
    vulnerability_density:   float
    max_steps:               int
    firewall_subnets:        int
    reward_exploit:          float = 20.0
    reward_discovery:        float = 5.0
    reward_step:             float = -0.5
    # ── Promotion criteria (OR logic: either condition triggers advance) ──────
    promote_threshold:       float = 0.60   # goal-rate fraction
    promote_reward_threshold: float = 800.0  # mean reward over window
    # ── Demotion ──────────────────────────────────────────────────────────────
    demote_threshold:        float = 0.15
    demote_reward_threshold: float = 50.0   # demote if mean reward is this low
    # ── Rolling window size ───────────────────────────────────────────────────
    window:                  int   = 20
    # ── Entropy coefficient override (None = keep cfg default) ───────────────
    ent_coef:                Optional[float] = None


TIERS: list[DifficultyTier] = [
    DifficultyTier(
        name="micro",
        num_hosts=4, num_subnets=2,
        vulnerability_density=0.9, max_steps=80,
        firewall_subnets=0,
        promote_threshold=0.60,
        promote_reward_threshold=700.0,   # micro runs often hit 1000–2100
        demote_threshold=0.0,
        demote_reward_threshold=0.0,
        window=8,                          # small window → faster evaluation
        ent_coef=0.07,
    ),
    DifficultyTier(
        name="small_office",
        num_hosts=8, num_subnets=2,
        vulnerability_density=0.5, max_steps=150,
        firewall_subnets=0,
        promote_threshold=0.55,
        promote_reward_threshold=600.0,
        demote_threshold=0.10,
        demote_reward_threshold=80.0,
        window=15,
        ent_coef=0.05,
    ),
    DifficultyTier(
        name="medium",
        num_hosts=12, num_subnets=3,
        vulnerability_density=0.45, max_steps=200,
        firewall_subnets=1,
        promote_threshold=0.50,
        promote_reward_threshold=500.0,
        demote_threshold=0.10,
        demote_reward_threshold=60.0,
        window=20,
        ent_coef=0.04,
    ),
    DifficultyTier(
        name="hard",
        num_hosts=18, num_subnets=4,
        vulnerability_density=0.35, max_steps=280,
        firewall_subnets=2,
        promote_threshold=0.45,
        promote_reward_threshold=400.0,
        demote_threshold=0.08,
        demote_reward_threshold=50.0,
        window=25,
        ent_coef=0.03,
    ),
    DifficultyTier(
        name="enterprise",
        num_hosts=25, num_subnets=5,
        vulnerability_density=0.30, max_steps=350,
        firewall_subnets=3,
        promote_threshold=1.01,          # never auto-promotes from hardest
        promote_reward_threshold=9999.0,
        demote_threshold=0.05,
        demote_reward_threshold=40.0,
        window=30,
        ent_coef=0.02,
    ),
]


class CurriculumScheduler:
    """
    Tracks agent performance and advances/retreats through difficulty tiers.

    Promotion logic (OR):
      - goal_rate  >= tier.promote_threshold
      - mean_reward >= tier.promote_reward_threshold

    Demotion logic (OR):
      - goal_rate  <= tier.demote_threshold  (and tier > 0)
      - mean_reward <= tier.demote_reward_threshold  (and tier > 0)
    """

    def __init__(
        self,
        start_tier: int              = 0,
        cfg:        Optional[ARCAConfig] = None,
        verbose:    bool             = True,
    ):
        self.tier_idx   = max(0, min(start_tier, len(TIERS) - 1))
        self.cfg        = cfg or ARCAConfig()
        self.verbose    = verbose

        self._history:    deque[bool]  = deque()
        self._rewards:    deque[float] = deque()
        self._ep_count:   int          = 0
        self._promotions: int          = 0
        self._demotions:  int          = 0

        # History of (tier_name, ep_count, goal_rate, mean_reward) for report
        self.tier_history: list[dict]  = []

        self._apply_tier()

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def tier(self) -> DifficultyTier:
        return TIERS[self.tier_idx]

    @property
    def tier_name(self) -> str:
        return self.tier.name

    @property
    def is_at_max(self) -> bool:
        return self.tier_idx == len(TIERS) - 1

    @property
    def is_at_min(self) -> bool:
        return self.tier_idx == 0

    # ── Core API ──────────────────────────────────────────────────────────────

    def record(self, goal_reached: bool, total_reward: float) -> bool:
        """
        Record one episode outcome.
        Returns True if the tier changed (caller should rebuild env).
        """
        t = self.tier
        self._ep_count += 1
        self._history.append(goal_reached)
        self._rewards.append(total_reward)

        # Keep window
        while len(self._history) > t.window:
            self._history.popleft()
            self._rewards.popleft()

        # Need at least half the window before evaluating
        min_samples = max(4, t.window // 2)
        if len(self._history) < min_samples:
            return False

        goal_rate   = sum(self._history) / len(self._history)
        mean_reward = sum(self._rewards)  / len(self._rewards)

        # Verbose progress every 5 episodes
        if self.verbose and self._ep_count % 5 == 0:
            print(
                f"  [Curriculum/{t.name}] ep={self._ep_count}  "
                f"goal_rate={goal_rate*100:.0f}%  "
                f"mean_reward={mean_reward:.0f}  "
                f"(promote_r≥{t.promote_reward_threshold:.0f}  "
                f"promote_g≥{t.promote_threshold*100:.0f}%)"
            )

        # ── Promotion (OR logic) ───────────────────────────────────────────────
        if not self.is_at_max and (
            goal_rate   >= t.promote_threshold or
            mean_reward >= t.promote_reward_threshold
        ):
            self.tier_history.append(self._snapshot(goal_rate, mean_reward))
            self.tier_idx += 1
            self._promotions += 1
            self._clear_window()
            self._apply_tier()
            if self.verbose:
                reason = (
                    f"goal_rate={goal_rate*100:.0f}%"
                    if goal_rate >= t.promote_threshold
                    else f"mean_reward={mean_reward:.0f}"
                )
                print(
                    f"\n[Curriculum] ↑ PROMOTED → [{self.tier.name}]  "
                    f"(reason: {reason})\n"
                )
            return True

        # ── Demotion (OR logic, never below tier 0) ────────────────────────────
        if not self.is_at_min and (
            (goal_rate   <= t.demote_threshold  and t.demote_threshold  > 0) or
            (mean_reward <= t.demote_reward_threshold and t.demote_reward_threshold > 0)
        ):
            self.tier_history.append(self._snapshot(goal_rate, mean_reward))
            self.tier_idx -= 1
            self._demotions += 1
            self._clear_window()
            self._apply_tier()
            if self.verbose:
                print(
                    f"\n[Curriculum] ↓ DEMOTED → [{self.tier.name}]  "
                    f"(goal_rate={goal_rate*100:.0f}%  "
                    f"mean_reward={mean_reward:.0f})\n"
                )
            return True

        return False

    def make_env(self):
        """Build a fresh NetworkEnv for the current tier."""
        from arca.sim.environment import NetworkEnv
        return NetworkEnv(cfg=self.cfg)

    def status(self) -> dict:
        goal_rate   = sum(self._history) / len(self._history) if self._history else 0.0
        mean_reward = sum(self._rewards) / len(self._rewards) if self._rewards else 0.0
        return {
            "tier_idx":    self.tier_idx,
            "tier_name":   self.tier.name,
            "episodes":    self._ep_count,
            "goal_rate":   round(goal_rate, 3),
            "mean_reward": round(mean_reward, 1),
            "promotions":  self._promotions,
            "demotions":   self._demotions,
            "num_hosts":   self.tier.num_hosts,
        }

    def __repr__(self) -> str:
        s = self.status()
        return (
            f"CurriculumScheduler("
            f"tier={s['tier_name']}, "
            f"ep={s['episodes']}, "
            f"goal={s['goal_rate']*100:.0f}%, "
            f"reward={s['mean_reward']:.0f})"
        )

    # ── Internals ─────────────────────────────────────────────────────────────

    def _apply_tier(self) -> None:
        t = self.tier
        self.cfg.env.preset                = t.name
        self.cfg.env.num_hosts             = t.num_hosts
        self.cfg.env.num_subnets           = t.num_subnets
        self.cfg.env.vulnerability_density = t.vulnerability_density
        self.cfg.env.max_steps             = t.max_steps
        self.cfg.env.reward_exploit        = t.reward_exploit
        self.cfg.env.reward_discovery      = t.reward_discovery
        self.cfg.env.reward_step           = t.reward_step
        # Per-tier entropy coefficient
        if t.ent_coef is not None:
            self.cfg.rl.ent_coef = t.ent_coef

    def _clear_window(self) -> None:
        self._history.clear()
        self._rewards.clear()

    def _snapshot(self, goal_rate: float, mean_reward: float) -> dict:
        return {
            "tier_name":   self.tier.name,
            "tier_idx":    self.tier_idx,
            "episodes":    self._ep_count,
            "goal_rate":   round(goal_rate, 3),
            "mean_reward": round(mean_reward, 1),
            "promotions":  self._promotions,
            "demotions":   self._demotions,
        }