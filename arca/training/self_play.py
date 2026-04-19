"""
arca/training/self_play.py  (v3.3 — NEW)
==========================================
Red-Blue Self-Play Evaluation.

The BlueTeamDefender is a rule-based defender that:
  1. Observes which hosts were compromised in the previous episode
  2. Patches (removes) 1–2 top-probability vulnerabilities on those hosts
  3. Adds firewall rules to compromised hosts' neighbours
  4. Re-runs the red agent and measures whether it can still succeed

This gives a realistic measure of the red agent's robustness against
an adaptive defender, which is far more useful than just measuring
reward in a static environment.

Usage::

    from arca.training.self_play import SelfPlayEvaluator

    evaluator = SelfPlayEvaluator(agent=agent, env=env, n_rounds=5)
    report    = evaluator.run()
    print(report.summary())
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from arca.sim.environment import NetworkEnv, EpisodeInfo
from arca.sim.host import Host, HostStatus


@dataclass
class SelfPlayResult:
    round_idx:           int
    red_reward_baseline: float
    red_reward_vs_blue:  float
    red_goal_baseline:   bool
    red_goal_vs_blue:    bool
    blue_patches_applied: int
    blue_firewalls_added: int
    red_hosts_compromised_vs_blue: int
    total_hosts: int

    @property
    def red_win(self) -> bool:
        return self.red_goal_vs_blue

    @property
    def reward_delta(self) -> float:
        return self.red_reward_vs_blue - self.red_reward_baseline


@dataclass
class SelfPlayReport:
    rounds:          list[SelfPlayResult]
    n_rounds:        int
    red_win_rate:    float
    red_win_rate_baseline: float
    mean_reward_baseline: float
    mean_reward_vs_blue:  float
    mean_compromised_vs_blue: float
    total_hosts:     int

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  ARCA Red-Blue Self-Play Evaluation",
            "=" * 60,
            f"  Rounds          : {self.n_rounds}",
            f"  Red win (no def): {self.red_win_rate_baseline*100:.0f}%",
            f"  Red win (vs def): {self.red_win_rate*100:.0f}%  "
            f"{'🔴 agent robust' if self.red_win_rate > 0.3 else '🟢 defender effective'}",
            f"  Mean reward baseline: {self.mean_reward_baseline:.1f}",
            f"  Mean reward vs blue: {self.mean_reward_vs_blue:.1f}  "
            f"(delta={self.mean_reward_vs_blue - self.mean_reward_baseline:+.1f})",
            f"  Mean compromised vs blue: "
            f"{self.mean_compromised_vs_blue:.1f}/{self.total_hosts}",
            "",
            "  Per-round breakdown:",
        ]
        for r in self.rounds:
            lines.append(
                f"    Round {r.round_idx+1}: "
                f"baseline_r={r.red_reward_baseline:.0f}  "
                f"blue_r={r.red_reward_vs_blue:.0f}  "
                f"patches={r.blue_patches_applied}  "
                f"fw={r.blue_firewalls_added}  "
                f"red_win={'✓' if r.red_win else '✗'}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)


class BlueTeamDefender:
    """
    Rule-based blue team that hardens the network based on prior attack knowledge.

    Strategy:
      1. Remove the top-probability vulnerability from each compromised host.
      2. Add firewall=True to direct neighbours of compromised hosts.
      3. Optionally shuffle remaining vulnerability probabilities down by 10%.
    """

    def __init__(
        self,
        patch_top_n_vulns: int   = 1,
        add_firewall_to_neighbours: bool = True,
        reduce_vuln_probs: bool  = True,
        rng: Optional[random.Random] = None,
    ):
        self.patch_top_n_vulns          = patch_top_n_vulns
        self.add_firewall_to_neighbours = add_firewall_to_neighbours
        self.reduce_vuln_probs          = reduce_vuln_probs
        self._rng = rng or random.Random(42)

    def harden(
        self,
        env: NetworkEnv,
        prev_episode: EpisodeInfo,
    ) -> tuple[int, int]:
        """
        Apply defensive changes to the env's hosts based on what was compromised.

        Returns (patches_applied, firewalls_added).
        """
        patches   = 0
        firewalls = 0

        compromised_ids = [
            hid for hid, h in env.hosts.items()
            if h.status == HostStatus.COMPROMISED
        ]

        # Also parse attack path for host IDs
        for path_entry in prev_episode.attack_path:
            # entries look like "0→2(CVE:...)"
            try:
                parts = path_entry.split("→")
                dst   = int(parts[1].split("(")[0])
                if dst not in compromised_ids:
                    compromised_ids.append(dst)
            except Exception:
                pass

        for hid in compromised_ids:
            host = env.hosts.get(hid)
            if host is None:
                continue

            # Patch top-N vulnerabilities (sorted by exploit_prob descending)
            vuln_list = sorted(
                [v for v in host.vulnerabilities if isinstance(v, dict)],
                key=lambda v: v.get("exploit_prob", 0),
                reverse=True,
            )
            to_remove = vuln_list[: self.patch_top_n_vulns]
            for vuln in to_remove:
                host.vulnerabilities.remove(vuln)
                patches += 1

            # Reduce probabilities of remaining vulns
            if self.reduce_vuln_probs:
                for v in host.vulnerabilities:
                    if isinstance(v, dict):
                        v["exploit_prob"] = max(
                            0.05,
                            v.get("exploit_prob", 0.5) * 0.85,
                        )

        # Add firewall to neighbours of compromised hosts
        if self.add_firewall_to_neighbours:
            for hid in compromised_ids:
                for neighbour in list(env.graph.successors(hid)):
                    n_host = env.hosts.get(neighbour)
                    if n_host and not n_host.firewall:
                        n_host.firewall = True
                        firewalls += 1

        return patches, firewalls


class SelfPlayEvaluator:
    """
    Runs N rounds of red-blue self-play evaluation.

    Each round:
      1. Red agent runs a baseline episode (no defender).
      2. Blue defender hardens the env based on what was compromised.
      3. Red agent runs again on the hardened network.
      4. Compare results.
    """

    def __init__(
        self,
        agent,
        env:        NetworkEnv,
        n_rounds:   int             = 5,
        defender:   Optional[BlueTeamDefender] = None,
        verbose:    bool            = True,
    ):
        self.agent   = agent
        self.env     = env
        self.n_rounds = n_rounds
        self.defender = defender or BlueTeamDefender()
        self.verbose  = verbose

    def run(self) -> SelfPlayReport:
        results: list[SelfPlayResult] = []

        if self.verbose:
            print(f"\n[SelfPlay] Running {self.n_rounds} red-blue rounds...")
            print(f"  Blue defender: patch_top={self.defender.patch_top_n_vulns}  "
                  f"firewall_neighbours={self.defender.add_firewall_to_neighbours}")

        for i in range(self.n_rounds):
            # ── Baseline (undefended) ─────────────────────────────────────────
            ep_baseline = self.agent.run_episode(deterministic=True)

            if self.verbose:
                print(
                    f"  Round {i+1}/{self.n_rounds} | "
                    f"baseline: r={ep_baseline.total_reward:.0f} "
                    f"comp={ep_baseline.hosts_compromised} "
                    f"goal={'✓' if ep_baseline.goal_reached else '✗'}"
                )

            # ── Blue team hardens the network ─────────────────────────────────
            # Reset the env first so hosts are in post-baseline state
            self.env.reset()
            # Simulate the baseline attack path to set compromised flags
            self._replay_compromises(ep_baseline)

            patches, firewalls = self.defender.harden(self.env, ep_baseline)

            # ── Reset env back to start but with hardened vulnerabilities ─────
            # We need to carry forward the harden changes to the NEXT episode.
            # We do this by saving the hardened hosts, resetting, then restoring vulns.
            hardened_hosts = self._snapshot_vulns_and_firewalls()
            obs, info = self.env.reset()
            self._restore_vulns_and_firewalls(hardened_hosts)

            # ── Red agent vs hardened env ─────────────────────────────────────
            ep_vs_blue = self._run_episode_on_current_env(obs, info)

            if self.verbose:
                print(
                    f"             | vs blue:   r={ep_vs_blue.total_reward:.0f} "
                    f"comp={ep_vs_blue.hosts_compromised} "
                    f"goal={'✓' if ep_vs_blue.goal_reached else '✗'} "
                    f"(patches={patches} fw_added={firewalls})"
                )

            results.append(SelfPlayResult(
                round_idx            = i,
                red_reward_baseline  = ep_baseline.total_reward,
                red_reward_vs_blue   = ep_vs_blue.total_reward,
                red_goal_baseline    = ep_baseline.goal_reached,
                red_goal_vs_blue     = ep_vs_blue.goal_reached,
                blue_patches_applied = patches,
                blue_firewalls_added = firewalls,
                red_hosts_compromised_vs_blue = ep_vs_blue.hosts_compromised,
                total_hosts          = len(self.env.hosts),
            ))

        # Compile report
        total = len(self.env.hosts)
        report = SelfPlayReport(
            rounds               = results,
            n_rounds             = self.n_rounds,
            red_win_rate         = sum(r.red_win for r in results) / len(results),
            red_win_rate_baseline= sum(r.red_goal_baseline for r in results) / len(results),
            mean_reward_baseline = np.mean([r.red_reward_baseline for r in results]),
            mean_reward_vs_blue  = np.mean([r.red_reward_vs_blue  for r in results]),
            mean_compromised_vs_blue = np.mean([r.red_hosts_compromised_vs_blue for r in results]),
            total_hosts          = total,
        )

        if self.verbose:
            print(report.summary())

        return report

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _replay_compromises(self, ep: EpisodeInfo) -> None:
        """Mark hosts as COMPROMISED based on the episode's attack path."""
        for path_entry in ep.attack_path:
            try:
                dst = int(path_entry.split("→")[1].split("(")[0])
                if dst in self.env.hosts:
                    self.env.hosts[dst].status = HostStatus.COMPROMISED
                    self.env.hosts[dst].discovered = True
            except Exception:
                pass

    def _snapshot_vulns_and_firewalls(self) -> dict:
        """Save current vuln lists and firewall flags for all hosts."""
        return {
            hid: {
                "vulnerabilities": [dict(v) if isinstance(v, dict) else v
                                    for v in h.vulnerabilities],
                "firewall": h.firewall,
            }
            for hid, h in self.env.hosts.items()
        }

    def _restore_vulns_and_firewalls(self, snapshot: dict) -> None:
        """Restore saved vuln lists and firewall flags after env.reset()."""
        for hid, saved in snapshot.items():
            h = self.env.hosts.get(hid)
            if h:
                h.vulnerabilities = saved["vulnerabilities"]
                h.firewall        = saved["firewall"]

    def _run_episode_on_current_env(
        self,
        obs,
        info: dict,
    ) -> EpisodeInfo:
        """Run one episode starting from an already-reset env."""
        from arca.core.cleanrl_ppo import CleanRLPPO
        import torch

        model = self.agent._model
        done  = False

        while not done:
            mask_np = info.get("action_mask")
            if mask_np is None:
                try:
                    mask_np = self.env.get_action_mask()
                except Exception:
                    mask_np = None

            if hasattr(model, "predict"):
                action, _ = model.predict(obs, deterministic=True, action_mask=mask_np)
            else:
                action, _ = model.predict(obs, deterministic=True)

            action = int(action.item() if hasattr(action, "item") else action)
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

        return self.env.episode_info