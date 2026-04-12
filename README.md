<div align="center">

<img src="https://github.com/DipayanDasgupta/arca/raw/main/logo.png" alt="ARCA Logo" width="320">

# ARCA — Autonomous Reinforcement Cyber Agent

**A fully local, pip-installable RL-powered cyber pentesting simulation framework with Gymnasium environment, Stable-Baselines3 training, optional C++ acceleration, custom network support, and LangGraph-powered red-teaming.**

[![PyPI version](https://img.shields.io/pypi/v/arca-agent.svg)](https://pypi.org/project/arca-agent/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![RL](https://img.shields.io/badge/RL-PPO%20%7C%20A2C%20%7C%20DQN-orange)](https://stable-baselines3.readthedocs.io)
[![LangGraph](https://img.shields.io/badge/Red--Team-LangGraph-purple)](https://langchain-ai.github.io/langgraph)

</div>

---

## What is ARCA?

**ARCA** is a local simulation framework that trains reinforcement learning agents to autonomously discover and exploit vulnerabilities in synthetic computer networks.

It provides:

- A **Gymnasium-compatible** network simulation environment with realistic hosts, subnets, services, and CVEs
- **Reinforcement Learning** support via Stable-Baselines3 (PPO, A2C, DQN) with training, evaluation, and checkpointing
- **Custom Network Builder** — define your own network topologies using YAML
- **Optional C++ acceleration** via pybind11 for performance-critical operations, with a pure-Python fallback
- **LangGraph-based red-teaming** for LLM prompt injection and jailbreak testing, separate from the RL pentesting simulation
- **Rich visualization** tools using Plotly and Matplotlib
- **CLI interface** via Typer
- **Configuration-driven** design for easy customization

Everything runs **100% locally** — no external cloud services, no data exfiltration.

---

## Installation

### From PyPI *(Recommended)*

```bash
pip install arca-agent
```

> If a C++ compiler (`g++` / `clang`) is available, the high-performance C++ extensions will be compiled automatically. Otherwise, ARCA gracefully falls back to pure Python.

### From Source *(Development)*

```bash
git clone https://github.com/DipayanDasgupta/arca.git
cd arca

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -e .                  # Base installation
pip install -e ".[cpp]"           # With C++ extensions
pip install -e ".[dev]"           # With dev dependencies
pip install -e ".[all]"           # All extras
```

---

## Quickstart

### Python API

```python
from arca import ARCAAgent, NetworkEnv, ARCAConfig

# Load a preset environment
env = NetworkEnv.from_preset("small_office")

# Create agent and train
agent = ARCAAgent(env=env)
agent.train(timesteps=50_000)

# Run a trained episode
result = agent.run_episode(render=True)
print(result.summary())

# Optional: Enable LangGraph reflection / red-teaming
agent.enable_langgraph()
report = agent.reflect(env.get_state_dict())
print(report)
```

### CLI

```bash
arca train --timesteps 50000 --preset small_office   # Train on a preset network
arca audit --preset small_office                     # Run a single episode
arca viz --output ./figures                          # Generate visualizations
arca info                                            # Show system and version info
```

---

## Network Presets

| Preset | Hosts | Subnets | Vuln Density | Max Steps |
|---|---|---|---|---|
| `small_office` | 8 | 2 | ~50% | 150 |
| `enterprise` | 25 | 5 | ~35% | 300 |
| `dmz` | 15 | 3 | ~45% | 200 |
| `iot_network` | 20 | 4 | ~60% | 250 |

You can also define fully custom topologies using YAML via `CustomNetworkBuilder`.

---

## Actions

| Action | Description |
|---|---|
| `SCAN` | Discover reachable hosts and their services/vulnerabilities |
| `EXPLOIT` | Attempt to compromise a discovered host using a CVE |
| `PIVOT` | Move the attacker's control to a compromised host |
| `EXFILTRATE` | Extract data value from a compromised host |

---

## Core Components

### 1. Simulation — `arca.sim`

- `NetworkEnv` — main Gymnasium environment (presets + custom)
- `CustomNetworkEnv` — user-defined topologies from YAML
- `Host`, `Action`, `ActionResult` — core simulation objects
- `NetworkGenerator` — procedural network creation
- Rich CVE library with realistic exploit probabilities

### 2. Reinforcement Learning — `arca.core`

- `ARCAAgent` — high-level interface for training and inference
- `ARCATrainer` — wraps Stable-Baselines3 with `EvalCallback`, `CheckpointCallback`, and TensorBoard support
- `ARCAConfig` — centralized dataclass-based configuration (env, rl, llm, viz, api)

### 3. LangGraph Red-Teaming — `arca.graph`

- Dedicated LangGraph workflow for prompt injection and jailbreak red-teaming against LLMs
- Nodes: `attacker_node`, `evaluator_node`, `defender_node`, `reporter_node`
- Supports `EchoTarget`, `OllamaTarget`, OpenAI-compatible targets, and a Retry wrapper
- Produces structured attack records and mitigation recommendations

### 4. C++ Acceleration — `arca.cpp_ext`

- Optional `sim_engine.cpp` built with pybind11
- Functions: `compute_reachability`, `floyd_warshall`, `batch_exploit`
- Graceful fallback to pure Python if compilation fails

### 5. Visualization — `arca.viz`

- `ARCAVisualizer` class
- Network graphs, vulnerability heatmaps, training curves, attack path overlays

### 6. CLI — `arca.cli`

- Entry point defined in `pyproject.toml`
- Commands: `train`, `audit`, `viz`, `info`

---

## Project Structure

```
arca/
├── arca/
│   ├── __init__.py
│   ├── __version__.py                  # 0.2.6
│   ├── core/
│   │   ├── config.py
│   │   ├── agent.py
│   │   └── trainer.py
│   ├── sim/
│   │   ├── environment.py
│   │   ├── host.py
│   │   ├── action.py
│   │   ├── custom_network.py
│   │   └── network_generator.py
│   ├── graph/                          # LangGraph red-teaming workflow
│   │   └── workflow.py
│   ├── targets/                        # LLM connectors (Echo, Ollama, OpenAI-compatible)
│   │   └── connectors.py
│   ├── cpp_ext/
│   │   ├── __init__.py
│   │   └── sim_engine.cpp              # Optional C++ backend
│   ├── viz/
│   │   └── visualizer.py
│   └── cli/
│       └── main.py                     # Typer CLI
├── tests/
│   └── test_comprehensive.py
├── examples/
│   └── quickstart.py
├── pyproject.toml
├── setup.py
└── README.md
```

---

## Disclaimer

ARCA is an **educational and research simulation tool only**.

- All attacks and simulations occur in a fully sandboxed, in-memory graph
- It does not perform real network scanning, exploitation, or generate real network traffic
- Use only on networks you are authorized to test

---

## Author

**Dipayan Dasgupta** — IIT Madras, Civil Engineering  
[GitHub](https://github.com/DipayanDasgupta) · [LinkedIn](https://linkedin.com/in/dipayandasgupta)