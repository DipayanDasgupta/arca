#!/usr/bin/env python3
"""
quickstart_v3.py  (ARCA v3.3)
================================
Integrates all v3.3 features:
  Phase 1 (fixes):
    ✓ LLM refusals eliminated via authorized-assessment framing
    ✓ Curriculum promotion works via reward-based OR goal-rate threshold
    ✓ VectorMemory RAG injected into every reflection node

  Phase 2 (new features):
    ✓ Adaptive Ethical Guardrails  (--ethical / --no-ethical)
    ✓ Self-Play Red-Blue           (--self-play)
    ✓ Offline RL Replay (BC)       (--offline-rl)
    ✓ Automated Markdown Report    (always generated at end)

Run:
    python quickstart_v3.py --curriculum --local-llm
    python quickstart_v3.py --curriculum --local-llm --self-play --offline-rl
    python quickstart_v3.py --preset enterprise --no-curriculum
    python quickstart_v3.py --local-llm --no-ethical  # direct red-team mode
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric")
warnings.filterwarnings("ignore", category=UserWarning, module="torch_scatter")
warnings.filterwarnings("ignore", message=".*CUDA.*")
warnings.filterwarnings("ignore", message=".*TypedStorage.*")
warnings.filterwarnings("ignore", message=".*duplicate.*begin_of_text.*")

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def banner(msg: str) -> None:
    print(f"\n{'=' * 66}")
    print(f"  {msg}")
    print('=' * 66)


def parse_args():
    p = argparse.ArgumentParser(description="ARCA v3.3 quickstart")
    p.add_argument("--preset",     default="small_office",
                   choices=["small_office", "enterprise", "dmz", "iot_network"])
    p.add_argument("--timesteps",  type=int, default=30_000)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--local-llm",  action="store_true")
    p.add_argument("--curriculum", action="store_true",
                   help="Run multi-tier curriculum training (recommended)")
    p.add_argument("--no-curriculum", action="store_true",
                   help="Skip curriculum, train directly on --preset")
    p.add_argument("--n-eval",     type=int, default=5,
                   help="Evaluation episodes after training")
    # Phase 2 flags
    p.add_argument("--self-play",  action="store_true",
                   help="Run red-blue self-play evaluation after training")
    p.add_argument("--offline-rl", action="store_true",
                   help="Run behavioral cloning on top episodes after training")
    p.add_argument("--ethical",    action="store_true", default=True,
                   help="Use ethical-mode framing in LLM prompts (default: on)")
    p.add_argument("--no-ethical", action="store_true",
                   help="Use direct red-team framing (more technical, same safety)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    use_curriculum = args.curriculum and not args.no_curriculum
    ethical_mode   = not args.no_ethical

    # ── Config ────────────────────────────────────────────────────────────────
    banner("1 / 7  Configuration  (ARCA v3.3)")
    from arca.core.config import ARCAConfig
    cfg = ARCAConfig.default()
    cfg.env.preset                        = args.preset
    cfg.rl.use_gnn                        = True
    cfg.rl.gnn_hidden_dim                 = args.hidden_dim
    cfg.rl.n_steps                        = 256
    cfg.rl.batch_size                     = 64
    cfg.rl.n_epochs                       = 6
    cfg.rl.online_reflection_interval     = 5_000 if args.local_llm else 0
    cfg.llm.enabled                       = True
    cfg.llm.use_local_llm                 = args.local_llm
    cfg.llm.ethical_mode                  = ethical_mode
    cfg.memory.enabled                    = True
    cfg.memory.seed_reward_mods_from_memory = True
    cfg.offline_rl.enabled                = args.offline_rl
    cfg.verbose                           = 1
    cfg.ensure_dirs()

    print(f"  Preset        : {cfg.env.preset}")
    print(f"  Action masking: ENABLED")
    print(f"  Curriculum    : {'ENABLED' if use_curriculum else 'disabled'}")
    print(f"  Local LLM     : {'✓ enabled' if args.local_llm else 'rule-based'}")
    print(f"  Ethical mode  : {'on (authorized-assessment framing)' if ethical_mode else 'off (direct red-team)'}")
    print(f"  Self-Play     : {'✓ enabled' if args.self_play else 'disabled'}")
    print(f"  Offline RL    : {'✓ enabled' if args.offline_rl else 'disabled'}")
    print(f"  Memory dir    : {cfg.memory.memory_dir}")

    # ── Environment ───────────────────────────────────────────────────────────
    banner("2 / 7  Environment")
    from arca.sim.environment import NetworkEnv
    env = NetworkEnv.from_preset(cfg.env.preset, cfg=cfg)
    obs, info = env.reset()

    mask    = info["action_mask"]
    n_valid = int(mask.sum())
    print(f"  Hosts      : {cfg.env.num_hosts}  |  Actions: {env.action_space.n}")
    print(f"  Valid actions at start: {n_valid}/{env.action_space.n}  "
          f"({n_valid/env.action_space.n*100:.1f}% reachable)")
    print(env.render())

    # ── Training ──────────────────────────────────────────────────────────────
    from arca.core.agent import ARCAAgent
    agent = ARCAAgent(env=env, cfg=cfg)

    curriculum_history = []

    if use_curriculum:
        banner("3 / 7  Curriculum Training  (reward-based + goal-rate promotion)")
        print("  Starting from tier 0 (micro), auto-advancing…\n")
        curriculum_history = agent.run_curriculum(
            timesteps_per_tier = args.timesteps,
            eval_episodes      = 10,
            start_tier         = 0,
            max_tiers          = 5,
        )
        print("\n  Curriculum history:")
        for h in curriculum_history:
            print(
                f"    {h['tier_name']:<15} ep={h['episodes']:<5} "
                f"goal={h['goal_rate']*100:.0f}%  "
                f"reward={h.get('mean_reward',0):.0f}  "
                f"↑{h['promotions']} ↓{h['demotions']}"
            )
    else:
        banner("3 / 7  Training")
        agent.train(timesteps=args.timesteps)
        print("  ✓ Training complete")

    # ── Offline RL ────────────────────────────────────────────────────────────
    offline_rl_stats = {}
    if args.offline_rl:
        banner("3b / 7  Offline RL — Behavioral Cloning")
        offline_rl_stats = agent.offline_rl_finetune()

    # ── Evaluation ────────────────────────────────────────────────────────────
    banner("4 / 7  Masked Evaluation")
    import numpy as np
    results = []
    for i in range(args.n_eval):
        ep   = agent.run_episode(deterministic=True)
        results.append(ep)
        goal = "✅" if ep.goal_reached else "❌"
        print(
            f"  Ep {i+1}: {goal}  reward={ep.total_reward:>8.2f}  "
            f"comp={ep.hosts_compromised}/{cfg.env.num_hosts}  "
            f"steps={ep.steps}"
        )
        if ep.attack_path:
            p = " → ".join(ep.attack_path[:4])
            print(f"         {p}{'…' if len(ep.attack_path) > 4 else ''}")

    print(f"\n  Mean reward     : {np.mean([r.total_reward for r in results]):.2f}")
    print(f"  Mean compromised: {np.mean([r.hosts_compromised for r in results]):.1f}")
    print(f"  Goal rate       : {sum(r.goal_reached for r in results)}/{args.n_eval}")

    mstats = agent.memory_stats()
    if "episode_buffer" in mstats and mstats["episode_buffer"]:
        s = mstats["episode_buffer"]
        print(f"\n  EpisodeBuffer   : {s.get('total_episodes',0)} episodes  "
              f"max_reward={s.get('max_reward',0):.1f}  "
              f"goal_rate={s.get('goal_rate',0)*100:.0f}%")

    # ── Vector Memory ─────────────────────────────────────────────────────────
    banner("5 / 7  Vector Memory + RAG")
    agent.enable_vector_memory()
    vm = agent._vector_memory
    print(f"  Backend  : {vm.backend}")
    print(f"  Indexed  : {len(vm)} episodes")

    if len(vm) > 0 and results:
        best_ep = max(results, key=lambda r: r.total_reward)
        query = {
            "total_reward":      best_ep.total_reward,
            "hosts_compromised": best_ep.hosts_compromised,
            "hosts_total":       cfg.env.num_hosts,
            "steps":             best_ep.steps,
            "goal_reached":      best_ep.goal_reached,
            "attack_path":       best_ep.attack_path,
            "severity_score":    0.0,
            "preset":            cfg.env.preset,
        }
        hits = vm.search(query, k=min(3, len(vm)))
        print(f"\n  Top-{len(hits)} semantically similar past episodes:")
        for rec, dist in hits:
            def _g(r, k, d=0):
                if hasattr(r, k): return getattr(r, k)
                return r.get(k, d) if isinstance(r, dict) else d
            print(
                f"    dist={dist:.3f}  reward={_g(rec,'total_reward',0):.1f}  "
                f"comp={_g(rec,'hosts_compromised',0)}/{_g(rec,'hosts_total',1)}  "
                f"goal={'✓' if _g(rec,'goal_reached') else '✗'}"
            )

    # ── Self-Play Evaluation ──────────────────────────────────────────────────
    sp_report = None
    if args.self_play:
        banner("5b / 7  Self-Play Red-Blue Evaluation")
        from arca.training.self_play import SelfPlayEvaluator, BlueTeamDefender
        defender = BlueTeamDefender(
            patch_top_n_vulns          = 1,
            add_firewall_to_neighbours = True,
            reduce_vuln_probs          = True,
        )
        evaluator = SelfPlayEvaluator(
            agent     = agent,
            env       = env,
            n_rounds  = 5,
            defender  = defender,
            verbose   = True,
        )
        sp_report = evaluator.run()

    # ── LangGraph Reflection ──────────────────────────────────────────────────
    banner("6 / 7  LangGraph Reflection  (RAG-enriched)")
    agent.enable_langgraph()
    state = env.get_state_dict()
    if results:
        best = max(results, key=lambda r: r.hosts_compromised)
        state["episode_info"] = {
            "total_reward":      best.total_reward,
            "hosts_compromised": best.hosts_compromised,
            "hosts_discovered":  best.hosts_discovered,
            "attack_path":       best.attack_path,
        }

    report = agent.reflect(state)

    print(f"  Provider      : {agent._langgraph.get_provider_name()}")
    print(f"  Severity      : {report.get('severity_score', 0):.1f} / 10.0")
    print(f"\n  [Analyst]     {str(report.get('analyst_output',''))[:220]}")
    print(f"\n  [Reflection]  {str(report.get('reflection',''))[:220]}")
    print(f"\n  [Plan]")
    for line in str(report.get("plan", "")).split("\n")[:5]:
        print(f"    {line}")
    print(f"\n  [Remediation]")
    for line in str(report.get("remediation", "")).split("\n")[:4]:
        print(f"    {line}")

    # ── Save + Report ─────────────────────────────────────────────────────────
    banner("7 / 7  Save + Automated Report")
    save_path = agent.save()
    print(f"  Model   → {save_path}")
    print(f"  Memory  → {cfg.memory.memory_dir}/episode_buffer.json")
    print(f"  Vectors → {cfg.memory.memory_dir}/vector_cache.pkl")

    # Generate markdown report
    from arca.reporting.report_generator import ARCAReportGenerator
    gen = ARCAReportGenerator(cfg=cfg)
    gen.add_curriculum_history(curriculum_history)
    if agent.memory_buffer:
        gen.add_episode_buffer(agent.memory_buffer)
    if sp_report:
        gen.add_self_play_report(sp_report)
    if offline_rl_stats:
        gen.add_offline_rl_stats(offline_rl_stats)
    gen.add_reflection_result(report)
    gen.add_note(f"Local LLM: {args.local_llm}  Ethical mode: {ethical_mode}")
    gen.add_note(f"Provider: {agent._langgraph.get_provider_name()}")
    report_path = gen.save()
    print(f"  Report  → {report_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    banner("ARCA v3.3 Complete!")
    print(f"""
  ✓ Action masking   — invalid actions blocked
  ✓ Curriculum       — reward-based promotion (no more tier-0 stuck)
  ✓ Vector Memory    — {len(vm)} episodes indexed, RAG injected into LLM
  ✓ LLM refusals     — eliminated via authorized-assessment framing
  ✓ Ethical guardrails — {'on' if ethical_mode else 'off (direct mode)'}
  ✓ Self-Play        — {'ran, red win rate vs blue: ' + f'{sp_report.red_win_rate*100:.0f}%' if sp_report else 'skipped (add --self-play)'}
  ✓ Offline RL       — {'BC applied, loss: ' + f"{offline_rl_stats.get('initial_loss',0):.3f}→{offline_rl_stats.get('final_loss',0):.3f}" if offline_rl_stats and not offline_rl_stats.get('skipped') else 'skipped (add --offline-rl)'}
  ✓ Automated report — {report_path}

  Next commands:
    python quickstart_v3.py --curriculum --local-llm --self-play --offline-rl
    python quickstart_v3.py --preset enterprise --no-curriculum
    python quickstart_v3.py --local-llm --no-ethical   # direct red-team mode
    tensorboard --logdir arca_outputs/tensorboard
""")


if __name__ == "__main__":
    main()