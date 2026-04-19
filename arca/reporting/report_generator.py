"""
arca/reporting/report_generator.py  (v3.3 — NEW)
==================================================
Generates a rich markdown report at the end of every ARCA run.

Saved to: arca_outputs/reports/report_YYYYMMDD_HHMMSS.md

Sections:
  1. Executive Summary
  2. Curriculum Progression Table
  3. Training Performance (reward stats per tier)
  4. Most Common Successful Attack Paths
  5. Top LLM Lessons Learned (from reflection memory)
  6. Self-Play Red-Blue Results (if available)
  7. Offline RL Stats (if available)
  8. Recommendations

Usage::

    from arca.reporting.report_generator import ARCAReportGenerator

    gen = ARCAReportGenerator(cfg=cfg)
    gen.add_curriculum_history(history)
    gen.add_episode_buffer(buf)
    gen.add_self_play_report(sp_report)
    gen.add_offline_rl_stats(bc_stats)
    gen.add_reflection_result(reflection)
    path = gen.save()
    print(f"Report saved → {path}")
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Any


class ARCAReportGenerator:
    """
    Collects data during a run and generates a markdown report.
    """

    def __init__(self, cfg=None):
        self.cfg = cfg
        self._sections: list[str] = []
        self._ts = time.strftime("%Y%m%d_%H%M%S")

        # Data buckets
        self._curriculum_history:  list[dict] = []
        self._episode_buffer       = None
        self._self_play_report     = None
        self._offline_rl_stats:    dict = {}
        self._reflection_result:   dict = {}
        self._extra_notes: list[str]   = []

    # ── Data collectors ───────────────────────────────────────────────────────

    def add_curriculum_history(self, history: list[dict]) -> "ARCAReportGenerator":
        self._curriculum_history = history or []
        return self

    def add_episode_buffer(self, buf) -> "ARCAReportGenerator":
        self._episode_buffer = buf
        return self

    def add_self_play_report(self, report) -> "ARCAReportGenerator":
        self._self_play_report = report
        return self

    def add_offline_rl_stats(self, stats: dict) -> "ARCAReportGenerator":
        self._offline_rl_stats = stats or {}
        return self

    def add_reflection_result(self, result: dict) -> "ARCAReportGenerator":
        self._reflection_result = result or {}
        return self

    def add_note(self, note: str) -> "ARCAReportGenerator":
        self._extra_notes.append(note)
        return self

    # ── Report generation ─────────────────────────────────────────────────────

    def generate(self) -> str:
        """Build and return the full markdown string."""
        parts = []

        # Header
        preset = "unknown"
        if self.cfg:
            preset = getattr(getattr(self.cfg, "env", None), "preset", "unknown")

        parts.append(f"# ARCA v3.3 — Security Simulation Report")
        parts.append(f"")
        parts.append(f"| Field       | Value |")
        parts.append(f"|-------------|-------|")
        parts.append(f"| Generated   | {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} |")
        parts.append(f"| Preset      | `{preset}` |")
        parts.append(f"| Run ID      | `{self._ts}` |")
        parts.append(f"")

        # 1. Executive Summary
        parts.append("---")
        parts.append("## 1. Executive Summary")
        parts.append("")
        parts.append(self._executive_summary())
        parts.append("")

        # 2. Curriculum Progression
        if self._curriculum_history:
            parts.append("---")
            parts.append("## 2. Curriculum Progression")
            parts.append("")
            parts.append(self._curriculum_table())
            parts.append("")

        # 3. Training Performance
        if self._episode_buffer and len(self._episode_buffer) > 0:
            parts.append("---")
            parts.append("## 3. Training Performance")
            parts.append("")
            parts.append(self._training_performance())
            parts.append("")

        # 4. Attack Paths
        if self._episode_buffer and len(self._episode_buffer) > 0:
            parts.append("---")
            parts.append("## 4. Most Successful Attack Paths")
            parts.append("")
            parts.append(self._attack_paths_section())
            parts.append("")

        # 5. LLM Lessons
        if self._reflection_result or (self._episode_buffer and len(self._episode_buffer) > 0):
            parts.append("---")
            parts.append("## 5. Agent Lessons Learned")
            parts.append("")
            parts.append(self._lessons_section())
            parts.append("")

        # 6. Self-Play
        if self._self_play_report is not None:
            parts.append("---")
            parts.append("## 6. Red-Blue Self-Play Results")
            parts.append("")
            parts.append(self._self_play_section())
            parts.append("")

        # 7. Offline RL
        if self._offline_rl_stats and not self._offline_rl_stats.get("skipped"):
            parts.append("---")
            parts.append("## 7. Offline RL / Behavioral Cloning")
            parts.append("")
            parts.append(self._offline_rl_section())
            parts.append("")

        # 8. Recommendations
        parts.append("---")
        parts.append("## 8. Defensive Recommendations")
        parts.append("")
        parts.append(self._recommendations())
        parts.append("")

        # Extra notes
        if self._extra_notes:
            parts.append("---")
            parts.append("## Notes")
            parts.append("")
            for note in self._extra_notes:
                parts.append(f"- {note}")
            parts.append("")

        parts.append("---")
        parts.append("*Generated by ARCA — Autonomous Reinforcement Cyber Agent v3.3*")
        parts.append("")

        return "\n".join(parts)

    def save(self, output_dir: Optional[str] = None) -> str:
        """Save the report and return the file path."""
        if output_dir is None and self.cfg:
            output_dir = getattr(getattr(self.cfg, "report", None), "output_dir",
                                 "arca_outputs/reports")
        output_dir = output_dir or "arca_outputs/reports"
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        filename = f"report_{self._ts}.md"
        path     = Path(output_dir) / filename
        path.write_text(self.generate(), encoding="utf-8")
        print(f"[ARCA Report] Saved → {path}")
        return str(path)

    # ── Section builders ──────────────────────────────────────────────────────

    def _executive_summary(self) -> str:
        lines = []

        if self._episode_buffer and len(self._episode_buffer) > 0:
            stats = self._episode_buffer.get_stats()
            lines.append(
                f"The ARCA agent completed **{stats.get('total_episodes', 0)} training episodes** "
                f"achieving a maximum reward of **{stats.get('max_reward', 0):.0f}** and a "
                f"goal completion rate of **{stats.get('goal_rate', 0)*100:.0f}%**."
            )
            lines.append("")
            lines.append(
                f"- Mean reward: **{stats.get('mean_reward', 0):.0f}**"
            )
            lines.append(
                f"- Mean hosts compromised: **{stats.get('mean_compromised', 0):.1f}**"
            )
        else:
            lines.append("Training data not available for summary.")

        if self._curriculum_history:
            n_tiers = len(set(h["tier_name"] for h in self._curriculum_history))
            lines.append(
                f"- Curriculum tiers reached: **{n_tiers}** "
                f"({' → '.join(h['tier_name'] for h in self._curriculum_history)})"
            )

        if self._self_play_report is not None:
            sp = self._self_play_report
            lines.append(
                f"- Self-play red win rate vs defender: "
                f"**{sp.red_win_rate*100:.0f}%** "
                f"(baseline: {sp.red_win_rate_baseline*100:.0f}%)"
            )

        if self._reflection_result.get("severity_score"):
            sev = self._reflection_result["severity_score"]
            label = ("🔴 CRITICAL" if sev >= 8 else "🟠 HIGH" if sev >= 6
                     else "🟡 MEDIUM" if sev >= 4 else "🟢 LOW")
            lines.append(f"- Final severity score: **{sev:.1f}/10** {label}")

        return "\n".join(lines)

    def _curriculum_table(self) -> str:
        lines = [
            "| Tier | Episodes | Goal Rate | Mean Reward | ↑ Promoted | ↓ Demoted |",
            "|------|----------|-----------|-------------|-----------|---------|",
        ]
        for h in self._curriculum_history:
            lines.append(
                f"| {h.get('tier_name','?'):<12} "
                f"| {h.get('episodes',0):<8} "
                f"| {h.get('goal_rate',0)*100:.0f}%{'':<7} "
                f"| {h.get('mean_reward',0):.0f}{'':<10} "
                f"| {h.get('promotions',0):<9} "
                f"| {h.get('demotions',0):<7} |"
            )
        return "\n".join(lines)

    def _training_performance(self) -> str:
        stats = self._episode_buffer.get_stats()
        best  = self._episode_buffer.get_best_episodes(n=3)
        lines = [
            f"**Total episodes stored:** {stats.get('total_episodes', 0)}  ",
            f"**Max reward:** {stats.get('max_reward', 0):.0f}  ",
            f"**Mean reward:** {stats.get('mean_reward', 0):.0f}  ",
            f"**Goal rate:** {stats.get('goal_rate', 0)*100:.0f}%  ",
            f"**Mean hosts compromised:** {stats.get('mean_compromised', 0):.1f}  ",
            "",
            "### Top 3 Episodes by Reward",
            "",
            "| # | Reward | Comp | Steps | Goal | Preset |",
            "|---|--------|------|-------|------|--------|",
        ]
        for i, ep in enumerate(best):
            lines.append(
                f"| {i+1} | {ep.total_reward:.0f} | "
                f"{ep.hosts_compromised}/{ep.hosts_total} | "
                f"{ep.steps} | "
                f"{'✓' if ep.goal_reached else '✗'} | "
                f"{ep.preset} |"
            )
        return "\n".join(lines)

    def _attack_paths_section(self) -> str:
        max_show = 10
        if self.cfg:
            max_show = getattr(getattr(self.cfg, "report", None),
                               "max_attack_paths_shown", 10)

        best = self._episode_buffer.get_best_episodes(n=max_show)
        lines = []
        path_counts: dict[str, int] = {}

        for ep in best:
            for step in ep.attack_path:
                path_counts[step] = path_counts.get(step, 0) + 1

        if not path_counts:
            return "_No attack paths recorded._"

        # Most common individual steps
        top_steps = sorted(path_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        lines.append("### Most Common Attack Steps")
        lines.append("")
        lines.append("| Step | Count |")
        lines.append("|------|-------|")
        for step, count in top_steps:
            lines.append(f"| `{step}` | {count} |")

        lines.append("")
        lines.append("### Top Episode Attack Chains")
        lines.append("")
        for i, ep in enumerate(best[:5]):
            if ep.attack_path:
                chain = " → ".join(ep.attack_path[:6])
                if len(ep.attack_path) > 6:
                    chain += " → …"
                lines.append(f"{i+1}. `{chain}` (reward={ep.total_reward:.0f})")

        return "\n".join(lines)

    def _lessons_section(self) -> str:
        lines = []
        max_lessons = 5
        if self.cfg:
            max_lessons = getattr(getattr(self.cfg, "report", None),
                                  "max_lessons_shown", 5)

        # From latest reflection
        refl = self._reflection_result.get("reflection", "")
        if refl and len(refl) > 10:
            lines.append("### Latest LLM Reflection")
            lines.append("")
            for line in refl.strip().split("\n"):
                lines.append(f"> {line}")
            lines.append("")

        # From episode buffer
        if self._episode_buffer and len(self._episode_buffer) > 0:
            best = self._episode_buffer.get_best_episodes(n=max_lessons)
            reflections = [(ep.reflection, ep.total_reward)
                           for ep in best if ep.reflection and len(ep.reflection) > 10]
            if reflections:
                lines.append("### Historical Lessons from Top Episodes")
                lines.append("")
                for i, (lesson, reward) in enumerate(reflections[:max_lessons]):
                    lines.append(f"**{i+1}.** (reward={reward:.0f}) {lesson[:200]}")
                    lines.append("")

        if not lines:
            return "_No lessons recorded yet. Enable LLM reflection to generate insights._"

        return "\n".join(lines)

    def _self_play_section(self) -> str:
        sp = self._self_play_report
        lines = [
            f"The red agent was evaluated against an adaptive rule-based defender "
            f"over **{sp.n_rounds} rounds**.",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Red win rate (baseline, no defender) | {sp.red_win_rate_baseline*100:.0f}% |",
            f"| Red win rate (vs blue team defender) | {sp.red_win_rate*100:.0f}% |",
            f"| Mean reward baseline | {sp.mean_reward_baseline:.0f} |",
            f"| Mean reward vs blue  | {sp.mean_reward_vs_blue:.0f} "
            f"({sp.mean_reward_vs_blue - sp.mean_reward_baseline:+.0f}) |",
            f"| Mean hosts compromised vs blue | {sp.mean_compromised_vs_blue:.1f}/{sp.total_hosts} |",
            "",
        ]

        if sp.red_win_rate >= 0.6:
            lines.append(
                "⚠️ **The agent remains highly effective even against an adaptive defender.** "
                "This indicates the vulnerabilities are fundamental and require urgent patching."
            )
        elif sp.red_win_rate <= 0.2:
            lines.append(
                "✅ **The blue team defender successfully neutralised most attacks.** "
                "Basic patching strategies are effective for this network configuration."
            )
        else:
            lines.append(
                "⚡ **Mixed results.** The defender reduces red team effectiveness but "
                "does not fully prevent compromise. Deeper hardening is recommended."
            )

        return "\n".join(lines)

    def _offline_rl_section(self) -> str:
        s = self._offline_rl_stats
        lines = [
            f"Behavioral cloning was applied to the top "
            f"**{s.get('n_episodes_used', 0)} episodes**.",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Episodes used | {s.get('n_episodes_used', 0)} |",
            f"| Steps reconstructed | {s.get('n_steps_used', 0)} |",
            f"| BC epochs | {s.get('epochs', 0)} |",
            f"| Initial loss | {s.get('initial_loss', 0):.4f} |",
            f"| Final loss   | {s.get('final_loss', 0):.4f} |",
            f"| Time elapsed | {s.get('elapsed_s', 0):.1f}s |",
        ]
        improvement = s.get("initial_loss", 0) - s.get("final_loss", 0)
        if improvement > 0:
            lines.append(f"")
            lines.append(
                f"✅ Policy improved: loss reduced by **{improvement:.4f}** "
                f"({improvement/max(s.get('initial_loss',1),1e-9)*100:.1f}%)."
            )
        return "\n".join(lines)

    def _recommendations(self) -> str:
        lines = []

        if self._reflection_result.get("remediation"):
            lines.append("### From LLM Analysis")
            lines.append("")
            for line in self._reflection_result["remediation"].strip().split("\n"):
                lines.append(f"- {line.lstrip('- ')}")
            lines.append("")

        lines.append("### General Hardening Checklist")
        lines.append("")
        checklist = [
            "[ ] Patch all CVE-scored vulnerabilities with CVSS ≥ 7.0 within 30 days",
            "[ ] Enable host-based firewall on all non-DMZ hosts",
            "[ ] Implement network segmentation between subnets",
            "[ ] Deploy SIEM and configure alerts for lateral movement",
            "[ ] Enable MFA on all administrative accounts",
            "[ ] Rotate all service account credentials",
            "[ ] Disable Telnet, FTP, and other plaintext protocols",
            "[ ] Schedule quarterly red-team simulations using ARCA",
        ]
        for item in checklist:
            lines.append(f"- {item}")

        return "\n".join(lines)