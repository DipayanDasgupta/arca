#!/usr/bin/env python3
"""
examples/test_my_network.py
============================
Complete end-to-end example: test ARCA on YOUR OWN custom network.

Run:
    python examples/test_my_network.py

This script:
  1. Generates a home network YAML template
  2. Builds a custom network environment from it
  3. Trains a PPO agent against it
  4. Runs a full security audit
  5. Gets LLM analysis (auto-detects: Ollama → Groq → rule-based)
  6. Prints remediation recommendations
  7. Generates HTML visualizations
"""

from __future__ import annotations
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def banner(msg: str, char: str = "="):
    width = 65
    print(f"\n{char * width}")
    print(f"  {msg}")
    print(f"{char * width}")


def section(msg: str):
    print(f"\n  ▶ {msg}")


# ── 1. Show available CVEs ─────────────────────────────────────────────────────

banner("ARCA — Custom Network Security Tester")
print("\n  Welcome! This tool trains an RL agent to find attack paths")
print("  in YOUR network topology and generates remediation advice.\n")

from arca.sim.custom_network import CustomNetworkBuilder, CVE_LIBRARY

print("  Available CVEs you can use in your YAML network definition:")
print(f"  {'Name':<22} {'CVE':<18} {'Prob':>5}  {'OS'}")
print(f"  {'-'*65}")
for name, info in CVE_LIBRARY.items():
    print(f"  {name:<22} {info['cve']:<18} {info['exploit_prob']:>5.0%}  {', '.join(info['os'])}")


# ── 2. Generate template and build network ────────────────────────────────────

banner("STEP 1 / 6  Build Custom Network")

YAML_PATH = "my_home_network.yaml"

# Generate template if it doesn't exist
if not Path(YAML_PATH).exists():
    section(f"Generating template → {YAML_PATH}")
    CustomNetworkBuilder.generate_template(YAML_PATH, preset="home")
    print(f"\n  📝 Template created at: {YAML_PATH}")
    print("  Edit it to match your real network, then re-run this script.")
    print("  For now, continuing with the default template...\n")
else:
    section(f"Loading existing network from {YAML_PATH}")

# Build the environment
env = CustomNetworkBuilder.from_yaml(YAML_PATH)
obs, info = env.reset()

print(f"\n  Network loaded: {env.spec.name}")
print(f"  Description:    {env.spec.description}")
print(f"  Hosts:          {len(env.spec.hosts)}")
print(f"  Subnets:        {len({h.subnet for h in env.spec.hosts})}")
print(f"  Connections:    {len(env.spec.connections)} links")
print(f"  Obs shape:      {obs.shape}")
print(f"  Attacker entry: {env.spec.attacker_entry}")
print()

# Show host table
print("  Host Map:")
print(f"  {'ID':<4} {'Name':<22} {'IP':<16} {'OS':<14} {'Vulns':<35} {'Critical'}")
print(f"  {'-'*100}")
for h in env.spec.hosts:
    print(f"  [{h.id}]  {h.name:<22} {h.ip:<16} {h.os:<14} "
          f"{', '.join(h.vulns[:3]):<35} {'⭐' if h.is_critical else ''}")

print()
print(env.render())


# ── 3. Check LLM provider ─────────────────────────────────────────────────────

banner("STEP 2 / 6  LLM Provider Status")

from arca.llm.providers import list_providers, auto_detect_provider

providers = list_providers()
print()
for p in providers:
    status = "✅ available" if p["available"] else "❌ not set up"
    print(f"  {p['name']:<12} {status:<20} {p['cost']:<15} {p.get('speed','')}")

active = auto_detect_provider()
print(f"\n  Active provider: [{active.name}]")

if active.name == "rule-based":
    print("\n  💡 For real LLM analysis, set up one of:")
    print("     Groq (free):  export GROQ_API_KEY=gsk_...   (console.groq.com)")
    print("     Ollama:       curl -fsSL https://ollama.com/install.sh | sh")
    print("                   ollama pull llama3.2:3b && ollama serve")


# ── 4. Train PPO agent ────────────────────────────────────────────────────────

banner("STEP 3 / 6  Train RL Agent")

from arca.core.config import ARCAConfig
from arca.core.agent import ARCAAgent

cfg = ARCAConfig.default()
cfg.rl.n_steps = 128
cfg.rl.batch_size = 64
cfg.rl.total_timesteps = 15_000
cfg.verbose = 0
cfg.ensure_dirs()

section(f"Training PPO for {cfg.rl.total_timesteps:,} steps...")
agent = ARCAAgent(env=env, cfg=cfg)
agent.train(timesteps=cfg.rl.total_timesteps, progress_bar=True)
print("  ✓ Training complete!")


# ── 5. Run audit episodes ─────────────────────────────────────────────────────

banner("STEP 4 / 6  Run Security Audit (5 Episodes)")

results = []
for i in range(5):
    ep = agent.run_episode(render=False)
    results.append(ep)
    goal = "✅" if ep.goal_reached else "❌"
    print(f"  Episode {i+1}: {goal} reward={ep.total_reward:>8.1f}  "
          f"compromised={ep.hosts_compromised}/{len(env.spec.hosts)}  "
          f"steps={ep.steps}")
    if ep.attack_path:
        print(f"    Path: {' → '.join(ep.attack_path[:5])}{'...' if len(ep.attack_path) > 5 else ''}")

# Summary stats
avg_reward = sum(r.total_reward for r in results) / len(results)
avg_comp = sum(r.hosts_compromised for r in results) / len(results)
success_rate = sum(1 for r in results if r.goal_reached) / len(results)

print(f"\n  Summary: avg_reward={avg_reward:.1f}  "
      f"avg_compromised={avg_comp:.1f}/{len(env.spec.hosts)}  "
      f"goal_rate={success_rate*100:.0f}%")


# ── 6. Full LLM analysis ──────────────────────────────────────────────────────

banner("STEP 5 / 6  LangGraph Security Analysis")

# Use the best episode for analysis
best_ep = max(results, key=lambda r: r.hosts_compromised)

# Re-run a fresh episode to get live state
obs, info = env.reset()
agent.enable_langgraph()

# Override the provider in the orchestrator
from arca.agents.langgraph_orchestrator import ARCAOrchestrator
agent._langgraph = ARCAOrchestrator(cfg=cfg, provider="auto")

section(f"Running analysis via {agent._langgraph.get_provider_name()}...")
state = env.get_state_dict()

# Enrich state with episode info from best run
state["episode_info"] = {
    "total_reward": best_ep.total_reward,
    "hosts_compromised": best_ep.hosts_compromised,
    "hosts_discovered": best_ep.hosts_discovered,
    "attack_path": best_ep.attack_path,
}

report = agent.reflect(state)

print(f"\n  Severity Score: {report.get('severity_score', 0):.1f} / 10.0")

print("\n  ┌─ ANALYST ────────────────────────────────────────────")
print(f"  │ {report.get('analyst_output', 'N/A')[:300]}")

print("\n  ┌─ ATTACKER INSIGHTS ──────────────────────────────────")
print(f"  │ {report.get('attacker_output', 'N/A')[:300]}")

print("\n  ┌─ CRITIC ─────────────────────────────────────────────")
print(f"  │ {report.get('critic_output', 'N/A')[:300]}")

print("\n  ┌─ REFLECTION ─────────────────────────────────────────")
print(f"  │ {report.get('reflection', 'N/A')[:300]}")

print("\n  ┌─ ACTION PLAN ────────────────────────────────────────")
print(f"  │ {report.get('plan', 'N/A')[:400]}")

print("\n  ┌─ REMEDIATION RECOMMENDATIONS ────────────────────────")
print(f"  │ {report.get('remediation', 'N/A')[:500]}")


# ── 7. Visualizations ─────────────────────────────────────────────────────────

banner("STEP 6 / 6  Generate Visualizations")

from arca.viz.visualizer import ARCAVisualizer
import random

viz = ARCAVisualizer(output_dir="arca_outputs/my_network")
obs, _ = env.reset()

section("Network topology...")
try:
    viz.plot_network(env.get_network_graph(), env.get_hosts(),
                     title=f"ARCA: {env.spec.name}", save=True, show=False)
    print("  ✓ arca_outputs/my_network/network_topology.html")
except Exception as e:
    print(f"  ✗ {e}")

section("Vulnerability heatmap...")
try:
    viz.plot_vuln_heatmap(env.get_hosts(), save=True, show=False)
    print("  ✓ arca_outputs/my_network/vuln_heatmap.html")
except Exception as e:
    print(f"  ✗ {e}")

section("Training performance curves...")
try:
    n = len(results)
    log_data = {
        "episodes": list(range(n)),
        "rewards": [r.total_reward for r in results],
        "compromised": [r.hosts_compromised for r in results],
        "path_lengths": [len(r.attack_path) for r in results],
        "success_rates": [1.0 if r.goal_reached else 0.0 for r in results],
    }
    viz.plot_training_curves(log_data, save=True, show=False)
    print("  ✓ arca_outputs/my_network/training_curves.html")
except Exception as e:
    print(f"  ✗ {e}")


# ── Final summary ──────────────────────────────────────────────────────────────

banner("ARCA AUDIT COMPLETE", char="★")

sev = report.get("severity_score", 0)
sev_label = "CRITICAL" if sev >= 8 else "HIGH" if sev >= 6 else "MEDIUM" if sev >= 4 else "LOW"

print(f"""
  Network:        {env.spec.name}
  Severity:       {sev:.1f}/10.0  [{sev_label}]
  LLM Provider:   {agent._langgraph.get_provider_name()}
  Hosts tested:   {len(env.spec.hosts)}
  Success rate:   {success_rate*100:.0f}% (goal reached in {sum(1 for r in results if r.goal_reached)}/5 episodes)
  Avg compromised:{avg_comp:.1f}/{len(env.spec.hosts)} hosts per episode

  Visualizations: arca_outputs/my_network/

  Next steps:
    1. Edit {YAML_PATH} to match your real network topology
    2. Add more hosts, vulnerabilities, and connections
    3. Re-run with more training: arca train --network {YAML_PATH} --timesteps 100000
    4. Enable real LLM: export GROQ_API_KEY=gsk_... (free at console.groq.com)
    5. Check remediation report above and patch your network!
""")