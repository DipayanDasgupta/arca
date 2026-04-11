"""
tests/test_arca.py
==================
Comprehensive test suite for ARCA.

Run with:
    pytest tests/ -v
    pytest tests/ -v --tb=short -x   # stop on first failure
"""

from __future__ import annotations

import numpy as np
import pytest


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

class TestARCAConfig:
    def test_default_config(self):
        from arca.core.config import ARCAConfig
        cfg = ARCAConfig.default()
        assert cfg.env.num_hosts == 10
        assert cfg.rl.algorithm == "PPO"
        assert cfg.verbose in (0, 1)

    def test_config_ensure_dirs(self, tmp_path):
        from arca.core.config import ARCAConfig
        cfg = ARCAConfig.default()
        cfg.model_dir = str(tmp_path / "models")
        cfg.log_dir = str(tmp_path / "logs")
        cfg.viz.output_dir = str(tmp_path / "figures")
        cfg.ensure_dirs()
        assert (tmp_path / "models").exists()
        assert (tmp_path / "logs").exists()
        assert (tmp_path / "figures").exists()

    def test_config_yaml_roundtrip(self, tmp_path):
        from arca.core.config import ARCAConfig
        cfg = ARCAConfig.default()
        cfg.env.num_hosts = 15
        cfg.rl.learning_rate = 1e-4
        yaml_path = tmp_path / "config.yaml"
        cfg.to_yaml(yaml_path)
        cfg2 = ARCAConfig.from_yaml(yaml_path)
        assert cfg2.env.num_hosts == 15
        assert abs(cfg2.rl.learning_rate - 1e-4) < 1e-10


# ──────────────────────────────────────────────────────────────────────────────
# HOST / ACTION
# ──────────────────────────────────────────────────────────────────────────────

class TestHostAndAction:
    def test_host_creation(self):
        from arca.sim.host import Host, HostStatus
        h = Host(id=0, subnet=0, os="Linux", ip="10.0.1.1")
        assert h.status == HostStatus.UNKNOWN
        assert not h.discovered
        d = h.to_dict()
        assert d["id"] == 0
        assert d["os"] == "Linux"

    def test_host_status_transitions(self):
        from arca.sim.host import Host, HostStatus
        h = Host(id=1, subnet=0, os="Windows", ip="10.0.1.2")
        h.discovered = True
        h.status = HostStatus.COMPROMISED
        assert h.status == HostStatus.COMPROMISED

    def test_action_creation(self):
        from arca.sim.action import Action, ActionType, ActionResult
        act = Action(action_type=ActionType.SCAN, target_host=3, exploit_id=0, source_host=0)
        assert act.action_type == ActionType.SCAN
        d = act.to_dict()
        assert d["type"] == "SCAN"

    def test_action_result(self):
        from arca.sim.action import ActionResult
        r = ActionResult(success=True, message="Scan OK", discovered_hosts=[2])
        assert r.success
        assert 2 in r.discovered_hosts
        d = r.to_dict()
        assert d["success"] is True


# ──────────────────────────────────────────────────────────────────────────────
# NETWORK GENERATOR
# ──────────────────────────────────────────────────────────────────────────────

class TestNetworkGenerator:
    def test_generator_creates_graph(self):
        import random
        from arca.sim.network_generator import NetworkGenerator
        from arca.core.config import EnvConfig
        cfg = EnvConfig(num_hosts=8, num_subnets=2)
        gen = NetworkGenerator(cfg, rng=random.Random(42))
        graph, hosts = gen.generate()
        assert len(hosts) == 8
        assert graph.number_of_nodes() == 8

    def test_attacker_node_in_subnet0(self):
        import random
        from arca.sim.network_generator import NetworkGenerator
        from arca.core.config import EnvConfig
        cfg = EnvConfig(num_hosts=8, num_subnets=2)
        gen = NetworkGenerator(cfg, rng=random.Random(0))
        _, hosts = gen.generate()
        assert hosts[gen.attacker_node].subnet == 0

    def test_vulnerability_assignment(self):
        import random
        from arca.sim.network_generator import NetworkGenerator
        from arca.core.config import EnvConfig
        cfg = EnvConfig(num_hosts=20, num_subnets=3, vulnerability_density=1.0)
        gen = NetworkGenerator(cfg, rng=random.Random(7))
        _, hosts = gen.generate()
        any_vulns = any(len(h.vulnerabilities) > 0 for h in hosts.values())
        assert any_vulns


# ──────────────────────────────────────────────────────────────────────────────
# NETWORK ENVIRONMENT
# ──────────────────────────────────────────────────────────────────────────────

class TestNetworkEnv:
    def test_env_reset(self):
        from arca.sim.environment import NetworkEnv
        env = NetworkEnv.from_preset("small_office")
        obs, info = env.reset()
        assert obs.shape == env.observation_space.shape
        assert "attacker_node" in info

    def test_env_step_returns_valid_obs(self):
        from arca.sim.environment import NetworkEnv
        env = NetworkEnv.from_preset("small_office")
        obs, _ = env.reset()
        action = env.action_space.sample()
        obs2, reward, terminated, truncated, info = env.step(action)
        assert obs2.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_env_render_returns_string(self):
        from arca.sim.environment import NetworkEnv
        env = NetworkEnv.from_preset("small_office")
        env.reset()
        rendered = env.render()
        assert isinstance(rendered, str)
        assert "ARCA" in rendered or "Host" in rendered

    def test_env_observation_in_bounds(self):
        from arca.sim.environment import NetworkEnv
        env = NetworkEnv.from_preset("small_office")
        obs, _ = env.reset()
        assert np.all(obs >= env.observation_space.low)
        assert np.all(obs <= env.observation_space.high)

    def test_env_presets(self):
        from arca.sim.environment import NetworkEnv
        for preset in ["small_office", "enterprise", "dmz", "iot_network"]:
            env = NetworkEnv.from_preset(preset)
            obs, info = env.reset()
            assert obs is not None, f"Preset {preset} failed to reset"

    def test_env_episode_terminates(self):
        from arca.sim.environment import NetworkEnv
        env = NetworkEnv.from_preset("small_office")
        obs, _ = env.reset()
        done = False
        max_iters = 500
        for _ in range(max_iters):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                done = True
                break
        assert done, "Episode should terminate within max_steps"

    def test_env_get_state_dict(self):
        from arca.sim.environment import NetworkEnv
        env = NetworkEnv.from_preset("small_office")
        env.reset()
        state = env.get_state_dict()
        assert "hosts" in state
        assert "attacker_node" in state
        assert "episode_info" in state

    def test_env_action_space_discrete(self):
        from arca.sim.environment import NetworkEnv
        from gymnasium.spaces import Discrete
        env = NetworkEnv.from_preset("small_office")
        assert isinstance(env.action_space, Discrete)

    def test_env_from_custom_config(self):
        from arca.sim.environment import NetworkEnv
        from arca.core.config import ARCAConfig, EnvConfig
        cfg = ARCAConfig.default()
        cfg.env = EnvConfig(num_hosts=6, num_subnets=2, max_steps=50)
        env = NetworkEnv(cfg=cfg)
        obs, _ = env.reset()
        assert obs.shape[0] == 6 * env._HOST_FEATURES


# ──────────────────────────────────────────────────────────────────────────────
# C++ EXTENSION (both CPP and fallback must work)
# ──────────────────────────────────────────────────────────────────────────────

class TestCppExtension:
    def test_import_does_not_crash(self):
        from arca.cpp_ext import CPP_AVAILABLE, compute_reachability, floyd_warshall, batch_exploit
        assert isinstance(CPP_AVAILABLE, bool)

    def test_compute_reachability_2node(self):
        from arca.cpp_ext import compute_reachability
        adj = [[1], [0]]  # node0 → node1, node1 → node0
        reach = compute_reachability(adj, 2)
        assert reach[0][1] is True
        assert reach[1][0] is True

    def test_compute_reachability_disconnected(self):
        from arca.cpp_ext import compute_reachability
        adj = [[], []]  # no edges
        reach = compute_reachability(adj, 2)
        assert reach[0][0] is True   # self
        assert reach[0][1] is False  # disconnected

    def test_floyd_warshall_simple(self):
        from arca.cpp_ext import floyd_warshall
        import math
        INF = math.inf
        w = [
            [0,   1,   INF],
            [INF, 0,   2  ],
            [INF, INF, 0  ],
        ]
        dist = floyd_warshall(w, 3)
        assert dist[0][2] == 3.0  # 0→1→2

    def test_batch_exploit_success_rate(self):
        from arca.cpp_ext import batch_exploit
        hosts = [{"exploit_prob": 1.0}] * 5
        actions = [(i, 0) for i in range(5)]
        results = batch_exploit(hosts, actions, seed=42)
        assert len(results) == 5
        for r in results:
            if isinstance(r, dict):
                assert r["success"] is True
            else:
                assert r.success is True

    def test_batch_exploit_zero_prob(self):
        from arca.cpp_ext import batch_exploit
        hosts = [{"exploit_prob": 0.0}] * 3
        actions = [(0, 0), (1, 0), (2, 0)]
        results = batch_exploit(hosts, actions, seed=1)
        for r in results:
            success = r["success"] if isinstance(r, dict) else r.success
            assert success is False


# ──────────────────────────────────────────────────────────────────────────────
# AGENT (no training — just instantiation and env interaction)
# ──────────────────────────────────────────────────────────────────────────────

class TestARCAAgent:
    def test_agent_instantiation(self):
        from arca.core.agent import ARCAAgent
        from arca.sim.environment import NetworkEnv
        env = NetworkEnv.from_preset("small_office")
        agent = ARCAAgent(env=env)
        assert agent is not None
        assert agent.env is env

    def test_agent_repr(self):
        from arca.core.agent import ARCAAgent
        agent = ARCAAgent()
        r = repr(agent)
        assert "ARCAAgent" in r

    def test_agent_reflect_without_llm(self):
        from arca.core.agent import ARCAAgent
        from arca.sim.environment import NetworkEnv
        env = NetworkEnv.from_preset("small_office")
        env.reset()
        agent = ARCAAgent(env=env)
        state = env.get_state_dict()
        # Should not raise even without Ollama running
        result = agent.reflect(state)
        assert isinstance(result, dict)

    def test_agent_train_and_run_episode(self):
        """Quick 1000-step training + 1 episode — validates full pipeline."""
        from arca.core.agent import ARCAAgent
        from arca.sim.environment import NetworkEnv
        from arca.core.config import ARCAConfig
        cfg = ARCAConfig.default()
        cfg.rl.n_steps = 64
        cfg.rl.batch_size = 32
        cfg.verbose = 0
        env = NetworkEnv.from_preset("small_office", cfg=cfg)
        agent = ARCAAgent(env=env, cfg=cfg)
        agent.train(timesteps=1000, progress_bar=False)
        info = agent.run_episode()
        assert info is not None
        assert info.steps > 0

    def test_agent_save_and_load(self, tmp_path):
        from arca.core.agent import ARCAAgent
        from arca.sim.environment import NetworkEnv
        from arca.core.config import ARCAConfig
        cfg = ARCAConfig.default()
        cfg.model_dir = str(tmp_path / "models")
        cfg.rl.n_steps = 64
        cfg.rl.batch_size = 32
        cfg.verbose = 0
        env = NetworkEnv.from_preset("small_office", cfg=cfg)
        agent = ARCAAgent(env=env, cfg=cfg)
        agent.train(timesteps=500, progress_bar=False)
        save_path = str(tmp_path / "model")
        agent.save(save_path)
        assert (tmp_path / "model.zip").exists()

        agent2 = ARCAAgent(env=env, cfg=cfg)
        agent2.load(save_path)
        info = agent2.run_episode()
        assert info.steps > 0


# ──────────────────────────────────────────────────────────────────────────────
# VISUALIZER
# ──────────────────────────────────────────────────────────────────────────────

class TestARCAVisualizer:
    def test_visualizer_instantiation(self, tmp_path):
        from arca.viz.visualizer import ARCAVisualizer
        viz = ARCAVisualizer(output_dir=str(tmp_path))
        assert viz.output_dir.exists()

    def test_plot_network_saves_html(self, tmp_path):
        from arca.viz.visualizer import ARCAVisualizer
        from arca.sim.environment import NetworkEnv
        env = NetworkEnv.from_preset("small_office")
        env.reset()
        viz = ARCAVisualizer(output_dir=str(tmp_path))
        viz.plot_network(env.get_network_graph(), env.get_hosts(), save=True, show=False)
        outputs = list(tmp_path.iterdir())
        assert len(outputs) >= 1

    def test_plot_vuln_heatmap_saves(self, tmp_path):
        from arca.viz.visualizer import ARCAVisualizer
        from arca.sim.environment import NetworkEnv
        env = NetworkEnv.from_preset("small_office")
        env.reset()
        viz = ARCAVisualizer(output_dir=str(tmp_path))
        viz.plot_vuln_heatmap(env.get_hosts(), save=True, show=False)

    def test_plot_training_curves_saves(self, tmp_path):
        from arca.viz.visualizer import ARCAVisualizer
        import random
        n = 20
        log_data = {
            "episodes": list(range(n)),
            "rewards": [random.gauss(5, 2) for _ in range(n)],
            "compromised": [random.randint(1, 4) for _ in range(n)],
            "path_lengths": [random.randint(1, 6) for _ in range(n)],
            "success_rates": [random.uniform(0.2, 0.8) for _ in range(n)],
        }
        viz = ARCAVisualizer(output_dir=str(tmp_path))
        viz.plot_training_curves(log_data, save=True, show=False)


# ──────────────────────────────────────────────────────────────────────────────
# EPISODE INFO
# ──────────────────────────────────────────────────────────────────────────────

class TestEpisodeInfo:
    def test_summary_string(self):
        from arca.sim.environment import EpisodeInfo
        info = EpisodeInfo(
            total_reward=42.5,
            steps=100,
            hosts_compromised=3,
            hosts_discovered=5,
            goal_reached=True,
        )
        s = info.summary()
        assert "42.5" in s
        assert "100" in s


# ──────────────────────────────────────────────────────────────────────────────
# INTEGRATION: Full pipeline smoke test
# ──────────────────────────────────────────────────────────────────────────────

class TestIntegration:
    def test_full_pipeline_small(self, tmp_path):
        """Smoke test: config → env → agent → train → eval → reflect → viz."""
        from arca.core.config import ARCAConfig
        from arca.core.agent import ARCAAgent
        from arca.sim.environment import NetworkEnv
        from arca.viz.visualizer import ARCAVisualizer
        import random

        cfg = ARCAConfig.default()
        cfg.env.preset = "small_office"
        cfg.rl.n_steps = 64
        cfg.rl.batch_size = 32
        cfg.verbose = 0
        cfg.model_dir = str(tmp_path / "models")
        cfg.log_dir = str(tmp_path / "logs")
        cfg.viz.output_dir = str(tmp_path / "figures")
        cfg.ensure_dirs()

        env = NetworkEnv.from_preset("small_office", cfg=cfg)
        agent = ARCAAgent(env=env, cfg=cfg)
        agent.train(timesteps=500, progress_bar=False)

        info = agent.run_episode()
        assert info.steps > 0

        state = env.get_state_dict()
        reflection = agent.reflect(state)
        assert isinstance(reflection, dict)

        viz = ARCAVisualizer(output_dir=str(tmp_path / "figures"))
        env.reset()
        viz.plot_network(env.get_network_graph(), env.get_hosts(), save=True, show=False)

        print(f"\n[Integration] Steps={info.steps}, Compromised={info.hosts_compromised}, Goal={info.goal_reached}")