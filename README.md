# ARCA — Autonomous Reinforcement Cyber Agent

> **A fully local, pip-installable RL-powered cyber pentesting simulation framework with LangGraph orchestration and optional C++ acceleration.**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![RL](https://img.shields.io/badge/RL-PPO%20%7C%20A2C%20%7C%20DQN-orange)](https://stable-baselines3.readthedocs.io)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-purple)](https://langchain-ai.github.io/langgraph)

---

## What is ARCA?

ARCA trains a reinforcement learning agent to autonomously discover and exploit vulnerabilities in simulated computer networks. It combines:

- **Gymnasium-compatible environment** — realistic hosts, subnets, CVEs, and network topology
- **PPO/A2C/DQN via Stable-Baselines3** — policy training with eval callbacks and checkpointing
- **LangGraph multi-agent orchestration** — Analyst → Attacker → Critic → Reflection pipeline with LLM-powered explanations
- **C++ acceleration via pybind11** — BFS reachability, batch exploit simulation, Floyd-Warshall (with pure-Python fallback)
- **FastAPI REST interface** — `/train`, `/audit`, `/reflect`, `/visualize` endpoints
- **Rich visualization suite** — Plotly network graphs, training curves, attack path overlays, vulnerability heatmaps
- **Full CLI** via Typer: `arca train`, `arca serve`, `arca audit`, `arca viz`

Everything runs **100% locally** — no cloud, no data leaves your machine.

---

## Installation

```bash
git clone https://github.com/dipayandasgupta/arca.git
cd arca

# Create virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# Install (pure Python — always works)
pip install -e .

# Install with C++ acceleration (requires g++/clang and pybind11)
pip install -e ".[cpp]"

# Install dev dependencies
pip install -e ".[dev]"
```

---

## Quickstart

```python
from arca import ARCAAgent, NetworkEnv, ARCAConfig

# Create environment
env = NetworkEnv.from_preset("small_office")

# Create and train agent
agent = ARCAAgent(env=env)
agent.train(timesteps=50_000)

# Run one episode
result = agent.run_episode(render=True)
print(result.summary())

# LangGraph reflection
agent.enable_langgraph()
report = agent.reflect(env.get_state_dict())
print(report["reflection"])
```

Or via CLI:

```bash
arca train --timesteps 50000 --preset small_office
arca serve                      # starts FastAPI at http://localhost:8000
arca audit --preset enterprise  # one-shot audit report
arca viz --output ./figures     # generate all plots
```

---

## Network Presets

| Preset        | Hosts | Subnets | Vuln Density | Max Steps |
|---------------|-------|---------|--------------|-----------|
| `small_office`  | 8     | 2       | 50%          | 150       |
| `enterprise`    | 25    | 5       | 35%          | 300       |
| `dmz`           | 15    | 3       | 45%          | 200       |
| `iot_network`   | 20    | 4       | 60%          | 250       |

---

## Actions

| Action      | Description                                          |
|-------------|------------------------------------------------------|
| `SCAN`      | Discover a reachable host and its services/vulns     |
| `EXPLOIT`   | Attempt to compromise a discovered host via a CVE    |
| `PIVOT`     | Move attacker's position to a compromised host       |
| `EXFILTRATE`| Extract data value from a compromised host           |

---

## LangGraph Architecture

```
START → analyst_node → attacker_node → critic_node → reflect_node → END
                              ↑___________________________|
                                   (reflection loop)
```

Each node uses a local LLM (via Ollama, default: `llama3`) for natural-language analysis. Falls back to rule-based logic if Ollama is not running.

---

## C++ Acceleration

The optional `_cpp_sim` module (built via pybind11) provides:

- `compute_reachability(adj, n)` — BFS all-pairs reachability (~10x faster than NetworkX for dense graphs)
- `floyd_warshall(weights, n)` — All-pairs shortest path
- `batch_exploit(hosts, actions, seed)` — Vectorised exploit simulation

Falls back to pure Python automatically if not compiled.

---

## API Endpoints

Once you run `arca serve`:

| Endpoint              | Method | Description                        |
|-----------------------|--------|------------------------------------|
| `/`                   | GET    | Health check + status              |
| `/train`              | POST   | Start a training run               |
| `/audit`              | POST   | Run an audit episode + get report  |
| `/reflect`            | POST   | Run LangGraph reflection on state  |
| `/status`             | GET    | Current training / agent status    |
| `/docs`               | GET    | Auto-generated Swagger UI          |

---

## Project Structure

```
arca/
├── arca/
│   ├── __init__.py          # public API
│   ├── __version__.py
│   ├── cli.py               # Typer CLI
│   ├── core/
│   │   ├── config.py        # ARCAConfig dataclass
│   │   ├── agent.py         # ARCAAgent (PPO wrapper + LangGraph)
│   │   └── trainer.py       # SB3 training harness
│   ├── sim/
│   │   ├── environment.py   # Gymnasium NetworkEnv
│   │   ├── host.py          # Host dataclass
│   │   ├── action.py        # Action / ActionResult types
│   │   └── network_generator.py
│   ├── agents/
│   │   └── langgraph_orchestrator.py
│   ├── cpp_ext/
│   │   ├── __init__.py      # Python fallback + CPP_AVAILABLE flag
│   │   └── sim_engine.cpp   # pybind11 C++ module
│   ├── viz/
│   │   └── visualizer.py    # Plotly + NetworkX charts
│   └── api/
│       └── server.py        # FastAPI app
├── tests/
│   └── test_arca.py
├── examples/
│   └── quickstart.py
├── pyproject.toml
├── setup.py
└── README.md
```

---

## Disclaimer

ARCA is a **simulation and education tool only**. All attack actions run inside a sandboxed in-memory graph. It does **not** perform any real network scanning, exploitation, or traffic generation. For authorised security testing only.

---

## Author

**Dipayan Dasgupta** — IIT Madras, Civil Engineering  
[GitHub](https://github.com/dipayandasgupta) · [LinkedIn](https://linkedin.com/in/dipayandasgupta)