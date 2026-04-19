"""
arca/memory/episode_buffer.py
==============================
Persistent episodic memory for ARCA v3.

Stores successful attack patterns, network topologies, and LLM reflections
across training runs.  Survives process restarts via JSON on disk.

Used by:
  - CleanRLPPO: seeds reward shaping from past episodes
  - ARCAOrchestrator: provides episode history to reflector node
  - ARCAAgent: persists lessons across calls to .train()

Design:
  - Ring buffer capped at max_episodes (default 1000)
  - Indexed by attack_path fingerprint for near-dedup
  - Serialised to ~/.arca/memory/episode_buffer.json
"""
from __future__ import annotations

import json
import hashlib
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class EpisodeRecord:
    """One stored episode."""
    episode_id:        str
    timestamp:         float
    preset:            str
    total_reward:      float
    hosts_compromised: int
    hosts_total:       int
    steps:             int
    goal_reached:      bool
    attack_path:       list[str]
    reward_modifiers:  dict        # LLM-derived shaping active during this episode
    reflection:        str         # LLM reflection text (lesson)
    severity_score:    float
    efficiency:        float = 0.0  # compromised/steps * 100

    def __post_init__(self):
        if self.steps > 0:
            self.efficiency = self.hosts_compromised / self.steps * 100

    @classmethod
    def from_dict(cls, d: dict) -> "EpisodeRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class EpisodeBuffer:
    """
    Persistent ring buffer of episode records.

    Usage::

        buf = EpisodeBuffer()

        buf.record(
            preset="small_office",
            total_reward=180.0,
            hosts_compromised=4,
            hosts_total=8,
            steps=95,
            goal_reached=True,
            attack_path=["0→2(EternalBlue)", "2→7(Log4Shell)"],
            reward_modifiers={"boost_critical": True},
            reflection="Prioritise critical hosts early.",
            severity_score=7.5,
        )

        stats = buf.get_stats()
        best  = buf.get_best_episodes(n=5)
    """

    def __init__(
        self,
        memory_dir: str | Path = Path.home() / ".arca" / "memory",
        max_episodes: int = 1000,
    ):
        self.memory_dir   = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.buffer_path  = self.memory_dir / "episode_buffer.json"
        self.max_episodes = max_episodes
        self._records: list[EpisodeRecord] = []
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self.buffer_path.exists():
            try:
                raw = json.loads(self.buffer_path.read_text())
                self._records = [EpisodeRecord.from_dict(r) for r in raw]
                print(
                    f"[ARCA Memory] Loaded {len(self._records)} past episodes "
                    f"from {self.buffer_path}"
                )
            except Exception as e:
                print(f"[ARCA Memory] Could not load buffer: {e}. Starting fresh.")
                self._records = []

    def _save(self) -> None:
        try:
            self.buffer_path.write_text(
                json.dumps([asdict(r) for r in self._records], indent=2)
            )
        except Exception as e:
            print(f"[ARCA Memory] Save failed: {e}")

    # ── Core API ──────────────────────────────────────────────────────────────
    def record(
        self,
        preset: str,
        total_reward: float,
        hosts_compromised: int,
        hosts_total: int,
        steps: int,
        goal_reached: bool,
        attack_path: list[str],
        reward_modifiers: dict,
        reflection: str,
        severity_score: float,
    ) -> None:
        ep_id = hashlib.sha256(
            f"{preset}:{attack_path}".encode()
        ).hexdigest()[:8]
        record = EpisodeRecord(
            episode_id=ep_id,
            timestamp=time.time(),
            preset=preset,
            total_reward=total_reward,
            hosts_compromised=hosts_compromised,
            hosts_total=hosts_total,
            steps=steps,
            goal_reached=goal_reached,
            attack_path=attack_path,
            reward_modifiers=reward_modifiers,
            reflection=reflection,
            severity_score=severity_score,
        )
        self._records.append(record)
        if len(self._records) > self.max_episodes:
            self._records.pop(0)
        self._save()

    def get_best_episodes(
        self, n: int = 10, preset: Optional[str] = None
    ) -> list[EpisodeRecord]:
        recs = self._records
        if preset:
            recs = [r for r in recs if r.preset == preset]
        return sorted(recs, key=lambda r: r.total_reward, reverse=True)[:n]

    def get_recent(self, n: int = 20) -> list[EpisodeRecord]:
        return self._records[-n:]

    def get_stats(self) -> dict:
        if not self._records:
            return {}
        rewards   = [r.total_reward for r in self._records]
        comps     = [r.hosts_compromised for r in self._records]
        goal_rate = sum(1 for r in self._records if r.goal_reached) / len(self._records)
        return {
            "total_episodes":   len(self._records),
            "mean_reward":      sum(rewards) / len(rewards),
            "max_reward":       max(rewards),
            "mean_compromised": sum(comps) / len(comps),
            "goal_rate":        goal_rate,
            "best_attack_paths": [r.attack_path for r in self.get_best_episodes(3)],
        }

    def infer_reward_mods(self) -> dict:
        """
        Analyse recent episodes and return reward modifier hints.
        Used by CleanRLPPO._run_reflection() when no LLM is available.
        """
        recent = self.get_recent(50)
        if not recent:
            return {}
        mods: dict = {}
        best = self.get_best_episodes(10)
        crit_ep_paths = [r.attack_path for r in best if r.total_reward > 100]
        if crit_ep_paths:
            mods["boost_critical"] = True
            mods["critical_mult"]  = 1.5
        avg_steps = sum(r.steps for r in recent) / len(recent)
        if avg_steps > 100:
            mods["penalize_redundant_scan"] = True
            mods["redundant_scan_delta"]    = 0.3
        return mods

    def format_for_llm(self, n: int = 5) -> str:
        """Return a compact text summary for LLM prompts."""
        recent = self.get_recent(n)
        if not recent:
            return "  No previous episodes."
        lines = []
        for i, r in enumerate(recent):
            lines.append(
                f"  Ep{i+1}: reward={r.total_reward:.1f} "
                f"comp={r.hosts_compromised}/{r.hosts_total} "
                f"steps={r.steps} goal={'✓' if r.goal_reached else '✗'}"
            )
            if r.reflection:
                lines.append(f"    Lesson: {r.reflection[:80]}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return f"EpisodeBuffer(episodes={len(self)}, path={self.buffer_path})"