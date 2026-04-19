"""
arca/training/curriculum.py  (v3.6)
=====================================
Key fix vs v3.5:
  Promotion now requires BOTH conditions (AND logic, not OR):
    1. goal_rate  >= tier.promote_threshold  (≥ 30% by default)
    2. mean_reward >= tier.promote_reward_threshold  (positive reward)

  Previously OR logic meant the agent promoted on high raw reward even
  with 0% goal rate — leading to a policy that never actually completes
  the objective advancing into harder tiers, causing negative eval rewards.

  Demotion logic unchanged (OR — either bad metric triggers demotion).
  promote_threshold raised to 0.30 across all tiers so 0% goal rate
  is never sufficient for promotion.
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
    # ── Promotion criteria (AND logic: BOTH conditions must be true) ──────────
    promote_threshold:       float = 0.30   # goal-rate fraction (≥ 30%)
    promote_reward_threshold: float = 200.0  # mean reward must be positive
    # ── Demotion ──────────────────────────────────────────────────────────────
    demote_threshold:        float = 0.05
    demote_reward_threshold: float = 30.0
    # ── Rolling window ────────────────────────────────────────────────────────
    window:                  int   = 20
    # ── Entropy coefficient override ─────────────────────────────────────────
    ent_coef:                Optional[float] = None


TIERS: list[DifficultyTier] = [
    DifficultyTier(
        name="micro",
        num_hosts=4, num_subnets=2,
        vulnerability_density=0.9, max_steps=80,
        firewall_subnets=0,
        promote_threshold=0.30,          # ← was 0.60 (OR), now 0.30 (AND)
        promote_reward_threshold=300.0,
        demote_threshold=0.0,
        demote_reward_threshold=0.0,
        window=10,
        ent_coef=0.07,
    ),
    DifficultyTier(
        name="small_office",
        num_hosts=8, num_subnets=2,
        vulnerability_density=0.5, max_steps=150,
        firewall_subnets=0,
        promote_threshold=0.30,
        promote_reward_threshold=250.0,
        demote_threshold=0.05,
        demote_reward_threshold=50.0,
        window=15,
        ent_coef=0.05,
    ),
    DifficultyTier(
        name="medium",
        num_hosts=12, num_subnets=3,
        vulnerability_density=0.45, max_steps=200,
        firewall_subnets=1,
        promote_threshold=0.30,
        promote_reward_threshold=200.0,
        demote_threshold=0.05,
        demote_reward_threshold=40.0,
        window=20,
        ent_coef=0.04,
    ),
    DifficultyTier(
        name="hard",
        num_hosts=18, num_subnets=4,
        vulnerability_density=0.35, max_steps=280,
        firewall_subnets=2,
        promote_threshold=0.30,
        promote_reward_threshold=150.0,
        demote_threshold=0.05,
        demote_reward_threshold=30.0,
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
        demote_threshold=0.03,
        demote_reward_threshold=20.0,
        window=30,
        ent_coef=0.02,
    ),
]


class CurriculumScheduler:
    """
    Tracks agent performance and advances/retreats through difficulty tiers.

    Promotion logic (AND — BOTH must be true):
      - goal_rate   >= tier.promote_threshold      (≥ 30%)
      - mean_reward >= tier.promote_reward_threshold

    Demotion logic (OR — either triggers retreat):
      - goal_rate  <= tier.demote_threshold
      - mean_reward <= tier.demote_reward_threshold
    """

    def __init__(
        self,
        start_tier: int               = 0,
        cfg:        Optional[ARCAConfig] = None,
        verbose:    bool              = True,
    ):
        self.tier_idx   = max(0, min(start_tier, len(TIERS) - 1))
        self.cfg        = cfg or ARCAConfig()
        self.verbose    = verbose

        self._history:    deque[bool]  = deque()
        self._rewards:    deque[float] = deque()
        self._ep_count:   int          = 0
        self._promotions: int          = 0
        self._demotions:  int          = 0

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

        while len(self._history) > t.window:
            self._history.popleft()
            self._rewards.popleft()

        min_samples = max(4, t.window // 2)
        if len(self._history) < min_samples:
            return False

        goal_rate   = sum(self._history) / len(self._history)
        mean_reward = sum(self._rewards)  / len(self._rewards)

        if self.verbose and self._ep_count % 5 == 0:
            print(
                f"  [Curriculum/{t.name}] ep={self._ep_count}  "
                f"goal_rate={goal_rate*100:.0f}%  "
                f"mean_reward={mean_reward:.0f}  "
                f"(need goal≥{t.promote_threshold*100:.0f}%  "
                f"AND reward≥{t.promote_reward_threshold:.0f})"
            )

        # ── Promotion: BOTH conditions required ───────────────────────────────
        if not self.is_at_max and (
            goal_rate   >= t.promote_threshold and
            mean_reward >= t.promote_reward_threshold
        ):
            self.tier_history.append(self._snapshot(goal_rate, mean_reward))
            self.tier_idx += 1
            self._promotions += 1
            self._clear_window()
            self._apply_tier()
            if self.verbose:
                print(
                    f"\n[Curriculum] ↑ PROMOTED → [{self.tier.name}]  "
                    f"(goal={goal_rate*100:.0f}%  reward={mean_reward:.0f})\n"
                )
            return True

        # ── Demotion: OR logic, never below tier 0 ────────────────────────────
        if not self.is_at_min and (
            (goal_rate   <= t.demote_threshold      and t.demote_threshold      > 0) or
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
                    f"(goal={goal_rate*100:.0f}%  reward={mean_reward:.0f})\n"
                )
            return True

        return False

    def make_env(self):
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