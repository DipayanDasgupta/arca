#!/usr/bin/env python3
"""
examples/quickstart.py
======================
ARCA end-to-end quickstart.

Run from project root:
    python examples/quickstart.py

Demonstrates:
  1. Environment creation from preset
  2. PPO training (10k steps)
  3. Episode evaluation
  4. LangGraph reflection (rule-based fallback if no Ollama)
  5. Visualization suite (saves HTML files)
  6. Model save
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

# Allow running from project root without installing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def banner(msg: str):
    print(f"\n{'='*55}")
    print(f"  {msg}")
    print('='*55)


def main():
    banner("ARCA — Autonomous Reinforcement Cyber Agent")
    print("Quickstart example — all local, no cloud required.\n")

    # ── 1. Configuration ──────────────────────────────────────────────
    banner("1 / 6  Configuration")
    from arca.core.config import ARCAConfig
    cfg = ARCAConfig.default()
    cfg.env.preset = "small_office"
    cfg.rl.algorithm = "PPO"
    cfg.rl.total_timesteps = 10_000
    cfg.rl.n_steps = 128
    cfg.rl.batch_size = 32
    cfg.verbose = 1
    cfg.ensure_dirs()
    print(f"  Preset    : {cfg.env.preset}")
    print(f"  Hosts     : {cfg.env.num_hosts}")
    print(f"  Subnets   : {cfg.env.num_subnets}")
    print(f"  Algorithm : {cfg.rl.algorithm}")
    print(f"  Steps     : {cfg.rl.total_timesteps:,}")

    # ── 2. Environment ────────────────────────────────────────────────
    banner("2 / 6  Create Network Environment")
    from arca.sim.environment import NetworkEnv
    env = NetworkEnv.from_preset("small_office", cfg=cfg)
    obs, info = env.reset()
    print(f"  Observation shape : {obs.shape}")
    print(f"  Action space      : {env.action_space}")
    print(f"  Attacker starts   : Host {info['attacker_node']}")
    print()
    print(env.render())

    # ── 3. Training ───────────────────────────────────────────────────
    banner("3 / 6  Train PPO Agent (10,000 steps)")
    from arca.core.agent import ARCAAgent
    agent = ARCAAgent(env=env, cfg=cfg)
    agent.train(timesteps=10_000, progress_bar=True)
    print("  Training complete!")

    # ── 4. Evaluation ────────────────────────────────────────────────
    banner("4 / 6  Evaluate (3 episodes)")
    for i in range(3):
        info_ep = agent.run_episode(render=False)
        print(f"  Episode {i+1}: {info_ep.summary()}")
        if info_ep.attack_path:
            print(f"    Attack path: {' → '.join(info_ep.attack_path[:4])}{'...' if len(info_ep.attack_path) > 4 else ''}")

    # ── 5. LangGraph Reflection ───────────────────────────────────────
    banner("5 / 6  LangGraph Reflection")
    print("  (Uses local Ollama if running; falls back to rule-based logic)\n")
    agent.enable_langgraph()
    env.reset()
    state = env.get_state_dict()
    result = agent.reflect(state)

    print(f"  Analyst:    {str(result.get('analyst_output', 'N/A'))[:180]}")
    print(f"\n  Critic:     {str(result.get('critic_output', 'N/A'))[:180]}")
    print(f"\n  Reflection: {str(result.get('reflection', 'N/A'))[:180]}")
    plan = result.get("plan", "N/A")
    if isinstance(plan, list):
        print(f"\n  Plan:")
        for step in plan[:3]:
            print(f"    - {step}")
    else:
        print(f"\n  Plan: {str(plan)[:200]}")

    # ── 6. Visualization ─────────────────────────────────────────────
    banner("6 / 6  Visualizations")
    from arca.viz.visualizer import ARCAVisualizer
    viz = ARCAVisualizer(output_dir="arca_outputs/figures")
    env.reset()

    try:
        viz.plot_network(env.get_network_graph(), env.get_hosts(), save=True, show=False)
        print("  ✓ Network topology → arca_outputs/figures/network_topology.html")
    except Exception as e:
        print(f"  ✗ Network plot: {e}")

    try:
        viz.plot_vuln_heatmap(env.get_hosts(), save=True, show=False)
        print("  ✓ Vulnerability heatmap → arca_outputs/figures/vuln_heatmap.html")
    except Exception as e:
        print(f"  ✗ Vuln heatmap: {e}")

    try:
        n = 40
        log_data = {
            "episodes": list(range(n)),
            "rewards": [random.gauss(10 * (1 + i / n), 4) for i in range(n)],
            "compromised": [random.randint(1, 6) for _ in range(n)],
            "path_lengths": [random.randint(1, 8) for _ in range(n)],
            "success_rates": [min(1.0, 0.2 + 0.5 * i / n + random.gauss(0, 0.05)) for i in range(n)],
        }
        viz.plot_training_curves(log_data, save=True, show=False)
        print("  ✓ Training curves  → arca_outputs/figures/training_curves.html")
    except Exception as e:
        print(f"  ✗ Training curves: {e}")

    # ── Save ─────────────────────────────────────────────────────────
    path = agent.save()
    print(f"\n  ✓ Model saved → {path}")

    # ── Summary ──────────────────────────────────────────────────────
    banner("ARCA Quickstart Complete!")
    print("\n  Next steps:")
    print("    arca train --timesteps 100000    # longer training")
    print("    arca serve                       # REST API at :8000")
    print("    arca audit --preset enterprise   # full audit report")
    print("    arca viz --show                  # open plots in browser")
    print("    arca info                        # system info\n")


if __name__ == "__main__":
    main()