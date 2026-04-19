#!/usr/bin/env python3
"""
quickstart_v3.py  (ARCA v3.1)
==============================
End-to-end demo:
  GNN + CleanRL-PPO  →  Persistent Memory  →  LangGraph Reflection

Run:
    python quickstart_v3.py
    python quickstart_v3.py --local-llm --timesteps 50000
    python quickstart_v3.py --preset enterprise --timesteps 100000
"""
from __future__ import annotations

# ── Silence noisy PyG / torch_scatter warnings before any imports ─────────────
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric")
warnings.filterwarnings("ignore", category=UserWarning, module="torch_scatter")
warnings.filterwarnings("ignore", category=UserWarning, module="torch_sparse")
warnings.filterwarnings("ignore", message=".*CUDA.*")
warnings.filterwarnings("ignore", message=".*TypedStorage.*")

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


# ─────────────────────────────────────────────────────────────────────────────

def banner(msg: str) -> None:
    print(f"\n{'=' * 64}")
    print(f"  {msg}")
    print('=' * 64)


def parse_args():
    p = argparse.ArgumentParser(description="ARCA v3.1 quickstart")
    p.add_argument(
        "--preset", default="small_office",
        choices=["small_office", "enterprise", "dmz", "iot_network"],
    )
    p.add_argument("--timesteps",  type=int,  default=20_000)
    p.add_argument("--hidden-dim", type=int,  default=128)
    p.add_argument("--local-llm",  action="store_true",
                   help="Enable local LLM (llama-cpp-python)")
    p.add_argument("--no-gnn",     action="store_true",
                   help="Fall back to SB3 legacy mode")
    p.add_argument("--n-eval",     type=int,  default=5)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── 1. Configuration ──────────────────────────────────────────────────────
    banner("1 / 6  Configuration")

    from arca.core.config import ARCAConfig
    cfg = ARCAConfig.default()

    cfg.env.preset            = args.preset
    cfg.rl.use_gnn            = not args.no_gnn
    cfg.rl.gnn_hidden_dim     = args.hidden_dim
    cfg.rl.n_steps            = 256
    cfg.rl.batch_size         = 64
    cfg.rl.n_epochs           = 6
    # ent_coef is already 0.05 in config — keeps GNN exploring
    cfg.rl.online_reflection_interval = 5_000 if args.local_llm else 0
    cfg.llm.enabled           = True          # always for orchestrator
    cfg.llm.use_local_llm     = args.local_llm
    cfg.memory.enabled        = True
    cfg.memory.seed_reward_mods_from_memory = True
    cfg.verbose               = 1
    cfg.ensure_dirs()

    mode = "GNN + CleanRL-PPO + EpisodeBuffer" if cfg.rl.use_gnn else "SB3 PPO (legacy)"
    print(f"  Preset        : {cfg.env.preset}")
    print(f"  Training mode : {mode}")
    print(f"  GNN hidden    : {cfg.rl.gnn_hidden_dim}  (LayerNorm · dual-pool)")
    print(f"  Entropy coef  : {cfg.rl.ent_coef}  (boosted for exploration)")
    print(f"  Timesteps     : {args.timesteps:,}")
    print(f"  Local LLM     : {'✓ enabled' if args.local_llm else 'rule-based fallback'}")
    print(f"  Memory dir    : {cfg.memory.memory_dir}")

    # ── 2. Environment ────────────────────────────────────────────────────────
    banner("2 / 6  Environment")

    from arca.sim.environment import NetworkEnv
    env = NetworkEnv.from_preset(cfg.env.preset, cfg=cfg)
    obs, info = env.reset()

    print(f"  Hosts      : {cfg.env.num_hosts}  |  Subnets: {cfg.env.num_subnets}")
    print(f"  Obs shape  : {obs.shape}  |  Actions: {env.action_space.n}")
    print(f"  Attacker @ : Host {info['attacker_node']}")

    try:
        pyg = env.get_pyg_data()
        print(
            f"  PyG graph  : {pyg.num_nodes} nodes, "
            f"{pyg.num_edges} edges, {pyg.x.shape[1]}d features"
        )
    except Exception as e:
        print(f"  PyG graph  : {e}")

    print()
    print(env.render())

    # ── 3. Training ───────────────────────────────────────────────────────────
    banner("3 / 6  Training")

    from arca.core.agent import ARCAAgent
    agent = ARCAAgent(env=env, cfg=cfg)
    agent.train(timesteps=args.timesteps)
    print("  ✓ Training complete")

    # Print memory stats if anything was stored during training
    mstats = agent.memory_stats()
    if mstats:
        print(
            f"  Memory  : {mstats['total_episodes']} training episodes stored  |  "
            f"max_reward={mstats['max_reward']:.1f}  |  "
            f"goal_rate={mstats['goal_rate']*100:.0f}%"
        )

    # ── 4. Evaluation ─────────────────────────────────────────────────────────
    banner("4 / 6  Evaluation")

    import numpy as np
    results = []
    for i in range(args.n_eval):
        ep = agent.run_episode()
        results.append(ep)
        goal = "✅" if ep.goal_reached else "❌"
        print(
            f"  Ep {i+1}: {goal}  reward={ep.total_reward:>8.2f}  "
            f"compromised={ep.hosts_compromised}/{cfg.env.num_hosts}  "
            f"steps={ep.steps}"
        )
        if ep.attack_path:
            path_str = " → ".join(ep.attack_path[:4])
            suffix   = "…" if len(ep.attack_path) > 4 else ""
            print(f"         path: {path_str}{suffix}")

    print(f"\n  Mean reward     : {np.mean([r.total_reward for r in results]):.2f}")
    print(
        f"  Mean compromised: "
        f"{np.mean([r.hosts_compromised for r in results]):.1f}/{cfg.env.num_hosts}"
    )
    print(
        f"  Goal rate       : "
        f"{sum(r.goal_reached for r in results)}/{args.n_eval}"
    )

    # Print updated memory stats (now includes eval episodes)
    mstats = agent.memory_stats()
    if mstats:
        print(
            f"\n  Memory (post-eval): {mstats['total_episodes']} episodes  |  "
            f"goal_rate={mstats['goal_rate']*100:.0f}%"
        )
        best_paths = mstats.get("best_attack_paths", [])
        if best_paths:
            print(f"  Best path so far : {best_paths[0][:3]}")

    # ── 5. LangGraph Reflection ───────────────────────────────────────────────
    banner("5 / 6  LangGraph Reflection")

    agent.enable_langgraph()
    env.reset()
    state = env.get_state_dict()

    best_ep = max(results, key=lambda r: r.hosts_compromised)
    state["episode_info"] = {
        "total_reward":      best_ep.total_reward,
        "hosts_compromised": best_ep.hosts_compromised,
        "hosts_discovered":  best_ep.hosts_discovered,
        "attack_path":       best_ep.attack_path,
    }

    report = agent.reflect(state)   # also writes reflection back to memory

    print(f"\n  Provider      : {agent._langgraph.get_provider_name()}")
    print(f"  Severity      : {report.get('severity_score', 0):.1f} / 10.0")
    print(f"\n  [Analyst]     {str(report.get('analyst_output', ''))[:240]}")
    print(f"\n  [Reflection]  {str(report.get('reflection', ''))[:240]}")
    print(f"\n  [Plan]")
    for line in str(report.get("plan", "")).split("\n")[:6]:
        print(f"    {line}")
    print(f"\n  [Remediation]")
    for line in str(report.get("remediation", "")).split("\n")[:5]:
        print(f"    {line}")

    # ── 6. Save & Summary ─────────────────────────────────────────────────────
    banner("6 / 6  Save & Summary")

    save_path = agent.save()
    print(f"  ✓ Model saved → {save_path}")
    print(f"  ✓ Memory stored → {cfg.memory.memory_dir}/episode_buffer.json")
    print(
        f"\n  Run 'cat {cfg.memory.memory_dir}/episode_buffer.json | python -m json.tool | head -60'"
        f"\n  to inspect what the agent has learned.\n"
    )

    banner("ARCA v3.1 Complete!")
    print(f"""
  Preset       : {cfg.env.preset}
  Backend      : {mode}
  Severity     : {report.get('severity_score', 0):.1f} / 10.0
  Model        : {save_path}
  Memory       : {cfg.memory.memory_dir}/episode_buffer.json

  Next steps:
    python quickstart_v3.py --timesteps 100000          # deeper training
    python quickstart_v3.py --local-llm                 # enable local LLM
    python quickstart_v3.py --preset enterprise          # larger network
    tensorboard --logdir arca_outputs/tensorboard        # live metrics
    arca audit --preset enterprise                       # full report
""")


if __name__ == "__main__":
    main()