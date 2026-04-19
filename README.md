<div align="center">

<img src="https://github.com/DipayanDasgupta/arca/raw/main/logo.png" alt="ARCA Logo" width="320">

# ARCA — Autonomous Reinforcement Cyber Agent

**A fully local, pip-installable RL-powered cyber pentesting simulation framework featuring GNN policies, curriculum learning, mixed-precision PPO, persistent memory with RAG, local LLM reflection, self-play evaluation, offline RL, and an interactive Dash visualizer.**

[![PyPI version](https://img.shields.io/pypi/v/arca-agent.svg)](https://pypi.org/project/arca-agent/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-brightgreen)](https://developer.nvidia.com/cuda-toolkit)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-purple)](https://langchain-ai.github.io/langgraph)

</div>

---

## What is ARCA?

**ARCA** trains autonomous agents to discover and exploit vulnerabilities in synthetic computer networks using **Graph Neural Networks + Reinforcement Learning**. It provides a complete local red-team simulation pipeline — from training and curriculum progression, to LLM-guided reflection, interactive visualization, and automated security reports.

Everything runs **100% locally**. No external APIs, no data leaving your machine.

---

## What's New in v3.6 (vs. the old SB3-based version)

| Area | Before (v0.2.x) | Now (v3.6) |
|---|---|---|
| **RL Engine** | Stable-Baselines3 (PPO/A2C/DQN) | Custom CleanRL-style PPO, purpose-built for graph observations |
| **Policy** | Flat MLP via SB3 | 3-layer GCN/GATv2 with dual graph pooling — understands network topology natively |
| **Curriculum** | None | 5-tier progressive training: `micro → small_office → medium → hard → enterprise` |
| **Action Masking** | None | Dynamic AMP-safe masking — prevents invalid actions across all curriculum tiers |
| **Training Speed** | Standard | Mixed Precision (AMP) on CUDA + `--fast` mode for rapid iteration |
| **Memory** | None | `EpisodeBuffer` + `VectorMemory` (FAISS/numpy RAG) with lifelong reward shaping |
| **LLM Reflection** | Basic LangGraph over Ollama/Groq | Local Llama-3.2-3B (fully offline, GGUF, GPU offload) with RAG-enriched prompts and ethical framing |
| **Self-Play** | None | `BlueTeamDefender` patches vulnerabilities and adds firewalls between rounds |
| **Offline RL** | None | Behavioral Cloning fine-tune on top-K episodes from the replay buffer |
| **Visualizer** | Static Plotly/Matplotlib files | Interactive Dash dashboard: real-time topology, attack replay, clickable host details, live metrics |
| **Reporting** | None | Automated markdown reports: executive summary, attack chains, lessons, remediation checklist |
| **C++ Backend** | pybind11 BFS/Floyd-Warshall | Preserved as optional accelerator with graceful pure-Python fallback |
| **API** | Basic FastAPI stub | Full REST server: `/train`, `/audit`, `/reflect`, `/presets` |

---

## Installation

### From PyPI *(Recommended)*

```bash
pip install arca-agent[all]
```

### From Source *(Development)*

```bash
git clone https://github.com/DipayanDasgupta/arca.git
cd arca

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -e ".[all]"       # All extras: GPU, SB3, C++, Dash, Groq
pip install -e ".[dev]"       # Dev tools: pytest, black, ruff, mypy
```

> **GPU note:** CUDA 11.8+ recommended for AMP mixed-precision training (~30–40% speedup). CPU fallback works seamlessly on all platforms.

> **C++ note:** If `g++` or `clang` is available, the BFS/Floyd-Warshall extension builds automatically. Otherwise ARCA uses the pure-Python fallback silently.

---

## Quick Start

### Recommended first run

```bash
# Fast curriculum run with local LLM + interactive visualizer
python quickstart_v3.py --curriculum --local-llm --fast --viz
```

The Dash dashboard launches automatically at `http://127.0.0.1:8051`.

### Other useful invocations

```bash
# Full run: curriculum + self-play + offline RL (no viz)
python quickstart_v3.py --curriculum --local-llm --self-play --offline-rl

# Skip curriculum, train directly on enterprise preset
python quickstart_v3.py --preset enterprise --no-curriculum

# Direct red-team framing (advanced — skips ethical guardrails in prompts)
python quickstart_v3.py --local-llm --no-ethical

# Profile with TensorBoard
tensorboard --logdir arca_outputs/tensorboard
```

### Python API

```python
from arca import ARCAAgent, NetworkEnv, ARCAConfig

# Load a preset environment
env = NetworkEnv.from_preset("small_office")

# Train with the GNN + CleanRL-PPO pipeline (default)
agent = ARCAAgent(env=env)
agent.train(timesteps=50_000)

# Run a masked, deterministic evaluation episode
result = agent.run_episode()
print(result.summary())

# Curriculum training — auto-advances through 5 difficulty tiers
history = agent.run_curriculum(timesteps_per_tier=30_000)

# Enable semantic memory (FAISS RAG over past episodes)
agent.enable_vector_memory()

# LangGraph reflection with RAG-enriched LLM prompts
agent.enable_langgraph()
report = agent.reflect(env.get_state_dict())
print(report["plan"])
print(report["remediation"])
```

### CLI

```bash
arca train --timesteps 50000 --preset small_office
arca audit --preset small_office --langgraph
arca viz --output ./figures
arca serve                         # FastAPI REST server on :8000
arca scan --subnet 192.168.1       # Discover local Ollama endpoints
arca info                          # Version, CUDA, SB3 status
```

---

## Network Presets

| Preset | Hosts | Subnets | Vuln Density | Max Steps |
|---|---|---|---|---|
| `small_office` | 8 | 2 | ~50% | 150 |
| `enterprise` | 25 | 5 | ~35% | 300 |
| `dmz` | 15 | 3 | ~45% | 200 |
| `iot_network` | 20 | 4 | ~60% | 250 |

Custom topologies can be defined in YAML via `CustomNetworkBuilder`:

```python
from arca.sim.custom_network import CustomNetworkBuilder

# Generate a YAML template, edit it, then load
CustomNetworkBuilder.generate_template("my_network.yaml", preset="home")
env = CustomNetworkBuilder.from_yaml("my_network.yaml")
```

The YAML CVE library includes: `EternalBlue`, `Log4Shell`, `BlueKeep`, `ZeroLogon`, `PrintNightmare`, `PwnKit`, `Spring4Shell`, `IoTDefaultCreds`, `Shellshock`, `Heartbleed`, `RouterDefaultPwd`, `SQLInjection`, and more.

---

## Curriculum Learning

ARCA trains through five progressive difficulty tiers. The scheduler advances the agent when performance exceeds a reward **or** goal-rate threshold (OR logic), and demotes it if performance drops.

| Tier | Hosts | Firewalled Subnets | Promote (reward ≥) | Window |
|---|---|---|---|---|
| `micro` | 4 | 0 | 700 | 8 eps |
| `small_office` | 8 | 0 | 600 | 15 eps |
| `medium` | 12 | 1 | 500 | 20 eps |
| `hard` | 18 | 2 | 400 | 25 eps |
| `enterprise` | 25 | 3 | — (max tier) | 30 eps |

```python
history = agent.run_curriculum(
    timesteps_per_tier=30_000,
    eval_episodes=10,
    start_tier=0,
    max_tiers=5,
)
```

---

## Actions

| Action | Description |
|---|---|
| `SCAN` | Discover reachable hosts and enumerate their services and CVEs |
| `EXPLOIT` | Attempt to compromise a discovered host using a specific CVE |
| `PIVOT` | Relocate the attacker's foothold to a compromised host |
| `EXFILTRATE` | Extract data value from a compromised host |

Action masking ensures the policy only selects actions that are valid given the current network state, host discovery status, and exploit availability. The mask is computed dynamically each step and is safe to use under mixed-precision (fp16) training.

---

## Core Components

### `arca.core` — Training Engine

- **`GNNPolicy`** — 3-layer GCN (or GATv2) with dual mean+max global pooling, orthogonal init, and masked actor-critic head. AMP-safe: fill values for masked logits are computed from `torch.finfo(logits.dtype).min / 2` to avoid fp16 overflow.
- **`CleanRLPPO`** — Custom PPO trainer with rollout buffer, GAE, gradient clipping, AMP autocast, TensorBoard logging, and reflection callbacks.
- **`ARCATrainer`** — Wraps Stable-Baselines3 for the non-GNN path (PPO/A2C/DQN) with `EvalCallback` and `CheckpointCallback`.
- **`ARCAAgent`** — High-level interface for training, evaluation, curriculum, save/load, and reflection.
- **`ARCAConfig`** — Dataclass-based config covering env, RL, LLM, memory, offline RL, reporting, API, and visualization.

### `arca.sim` — Network Simulation

- **`NetworkEnv`** — Gymnasium-compatible environment with presets, procedural generation, action masking, PyG data export, and LLM reward shaping hooks.
- **`CustomNetworkEnv`** — User-defined YAML topologies with a rich CVE library and subnet-level firewall simulation.
- **`NetworkGenerator`** — Procedural OS/service/CVE assignment with realistic subnet topology.
- **`Host`, `Action`, `ActionResult`** — Core simulation primitives.

### `arca.training` — Advanced Training Utilities

- **`CurriculumScheduler`** — OR-logic promotion (goal rate **or** mean reward), demotion protection, per-tier entropy coefficient tuning.
- **`SelfPlayEvaluator`** — Red vs. blue rounds: the `BlueTeamDefender` patches top CVEs, adds firewalls to neighbours, and reduces exploit probabilities between rounds.
- **`offline_bc_finetune`** — Behavioural cloning on the top-K episodes from the replay buffer, reconstructing `(obs, action)` pairs by replaying stored attack paths.

### `arca.memory` — Persistent Episodic Memory

- **`EpisodeBuffer`** — JSON-persisted ring buffer (up to 1000 episodes) with reward/goal-rate stats, attack-path fingerprinting, and `infer_reward_mods()` for heuristic shaping.
- **`VectorMemory`** — 13-dimensional episode embeddings (reward, compromise ratio, efficiency, OS fingerprint, severity, path length) indexed by FAISS or pure numpy. Semantic search feeds the top-k most similar past episodes into every LLM prompt as RAG context.

### `arca.agents` — LangGraph Orchestrator

Six-node multi-agent graph: `analyst → attacker → critic → reflector → planner → remediator`.

- **Local-first LLM priority:** LocalLLM (llama-cpp-python GGUF) → Ollama → Groq → rule-based fallback.
- **Ethical framing:** All system prompts include an authorized-assessment header that prevents small models from refusing to respond.
- **Refusal detection:** Responses matching known refusal phrases automatically fall back to rule-based output.
- **RAG injection:** VectorMemory semantic search results are prepended to every node's user prompt.

### `arca.llm` — LLM Providers

- **`LocalLLM`** — llama-cpp-python wrapper supporting Llama-3.2-3B-Q4, Phi-3-mini, and Gemma-2-2B. GPU offload via `n_gpu_layers=-1`. C-level stderr noise suppressed via OS-level fd redirect.
- **`auto_detect_provider()`** — Priority chain: Ollama → Groq → Anthropic → OpenAI → rule-based.

### `arca.viz` — Visualization

- **`ARCAVisualizer`** — Static Plotly network topology, vulnerability heatmap, attack path overlay, and training curves (HTML output).
- **Interactive Dash Dashboard** — Real-time network graph with status coloring, clickable host details, live reward/compromise metrics, curriculum progress, and attack replay controls. Launch with `--viz`.

### `arca.reporting` — Automated Reports

`ARCAReportGenerator` assembles a markdown report with: executive summary, curriculum progression table, top episode attack chains, LLM lessons learned, self-play results, offline RL stats, and a defensive remediation checklist.

### `arca.graph` — LLM Red-Team Workflow

Separate from the RL simulation — adversarially red-teams an LLM target using a LangGraph attacker/evaluator/defender/reporter loop. Tests 8 attack vectors: `direct_prompt_injection`, `role_play_hijack`, `system_prompt_extraction`, `pii_extraction`, `jailbreak_prefix`, `indirect_rag_injection`, `delimiter_confusion`, `context_overflow`.

### `arca.api` — REST Server

FastAPI server (`arca serve`) exposing `/train`, `/audit`, `/reflect`, `/presets`, and `/status`. CORS-enabled, fully self-contained.

### `arca.cpp_ext` — C++ Acceleration *(Optional)*

pybind11 module exposing `compute_reachability` (BFS), `floyd_warshall`, and `batch_exploit`. Built automatically if a C++ compiler is present; otherwise all three fall back to pure Python transparently.

---

## Project Structure

```
arca/
├── arca/
│   ├── agents/
│   │   └── langgraph_orchestrator.py   # 6-node LangGraph multi-agent graph
│   ├── api/
│   │   └── server.py                   # FastAPI REST interface
│   ├── cli/
│   │   └── main.py                     # Typer CLI (train/audit/viz/serve/scan/info)
│   ├── core/
│   │   ├── agent.py                    # High-level ARCAAgent
│   │   ├── cleanrl_ppo.py              # CleanRL PPO trainer (AMP, masking, memory)
│   │   ├── config.py                   # Dataclass config (env/rl/llm/memory/report/…)
│   │   ├── gnn_policy.py               # GNNEncoder + masked actor-critic (AMP-safe)
│   │   └── trainer.py                  # SB3 trainer (non-GNN path)
│   ├── cpp_ext/
│   │   ├── __init__.py                 # Graceful fallback loader
│   │   └── sim_engine.cpp              # BFS, Floyd-Warshall, batch_exploit
│   ├── graph/
│   │   └── workflow.py                 # LLM red-team LangGraph workflow
│   ├── llm/
│   │   ├── local_llm.py                # llama-cpp-python wrapper + stderr suppression
│   │   └── providers.py                # Ollama / Groq / Anthropic / OpenAI / rule-based
│   ├── memory/
│   │   ├── episode_buffer.py           # JSON-persisted ring buffer
│   │   └── vector_memory.py            # FAISS/numpy semantic episode search
│   ├── reporting/
│   │   └── report_generator.py         # Automated markdown reports
│   ├── sim/
│   │   ├── action.py
│   │   ├── custom_network.py           # YAML topology builder + CVE library
│   │   ├── environment.py              # NetworkEnv (Gymnasium, masking, PyG)
│   │   ├── host.py
│   │   └── network_generator.py        # Procedural network generation
│   ├── targets/
│   │   └── connectors.py               # OllamaTarget, OpenAICompatibleTarget, EchoTarget
│   ├── training/
│   │   ├── curriculum.py               # CurriculumScheduler (5 tiers, OR-logic)
│   │   ├── offline_rl.py               # Behavioral cloning from replay buffer
│   │   └── self_play.py                # Red-blue self-play + BlueTeamDefender
│   ├── utils/
│   │   └── __init__.py
│   └── viz/
│       └── visualizer.py               # Plotly + Dash visualizations
├── examples/
│   ├── quickstart.py
│   └── test_my_network.py
├── tests/
│   ├── conftest.py
│   ├── test_arca.py
│   └── test_comprehensive.py
├── quickstart_v3.py                    # Main entry point with all flags
├── pyproject.toml
├── setup.py
└── README.md
```

---

## Configuration

ARCA is fully configuration-driven. The main dataclass is `ARCAConfig`:

```python
from arca.core.config import ARCAConfig

cfg = ARCAConfig.default()

# Environment
cfg.env.preset = "small_office"
cfg.env.num_hosts = 8

# RL
cfg.rl.use_gnn = True
cfg.rl.gnn_hidden_dim = 128
cfg.rl.learning_rate = 3e-4
cfg.rl.ent_coef = 0.05

# LLM
cfg.llm.use_local_llm = True
cfg.llm.local_model_key = "llama-3.2-3b"
cfg.llm.ethical_mode = True

# Memory
cfg.memory.enabled = True
cfg.memory.seed_reward_mods_from_memory = True

# Save/load YAML
cfg.to_yaml("my_config.yaml")
cfg2 = ARCAConfig.from_yaml("my_config.yaml")
```

---

## LLM Provider Setup

ARCA auto-detects the best available provider (local-first):

```bash
# Option 1: Fully local (no internet required after download)
# Model downloads automatically to ~/.arca/models/ on first use
python quickstart_v3.py --local-llm

# Option 2: Ollama (local server)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:3b
ollama serve
# ARCA detects Ollama automatically

# Option 3: Groq (free API, very fast)
export GROQ_API_KEY=gsk_...   # console.groq.com
# ARCA detects GROQ_API_KEY automatically

# Option 4: Always-available rule-based fallback (no setup needed)
# Used automatically when no LLM is configured
```

---

## Running Tests

```bash
pytest tests/ -v                          # All tests
pytest tests/ -v -k "not slow"            # Skip real network tests
pytest tests/test_comprehensive.py -v     # LangGraph workflow tests only
```

---

## Disclaimer

ARCA is an **educational and research simulation tool only**.

- All attacks and simulations occur in a fully sandboxed, in-memory graph.
- It does not perform real network scanning, exploitation, or generate real network traffic.
- Use only on networks you are authorized to test.

---

## Author

**Dipayan Dasgupta** — IIT Madras, Civil Engineering
[GitHub](https://github.com/DipayanDasgupta) · [LinkedIn](https://linkedin.com/in/dipayandasgupta)