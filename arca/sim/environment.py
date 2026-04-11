"""
arca.sim.environment
~~~~~~~~~~~~~~~~~~~~
Gymnasium-compatible network penetration testing environment.

Nodes  = hosts (id, subnet, os, vulnerabilities, compromised)
Edges  = reachability between hosts
Actions = (action_type, target_host, exploit_id)
Observation = flat feature vector of known network state
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Optional

import gymnasium as gym
import networkx as nx
import numpy as np
from gymnasium import spaces

from arca.core.config import ARCAConfig, EnvConfig
from arca.sim.network_generator import NetworkGenerator
from arca.sim.host import Host, HostStatus
from arca.sim.action import Action, ActionType, ActionResult

# Try to import C++ backend
try:
    from arca._cpp_sim import SimEngine as _CppSimEngine  # type: ignore
    _CPP_AVAILABLE = True
except ImportError:
    _CppSimEngine = None
    _CPP_AVAILABLE = False


PRESETS = {
    "small_office": EnvConfig(num_hosts=8, num_subnets=2, vulnerability_density=0.5, max_steps=150),
    "enterprise": EnvConfig(num_hosts=25, num_subnets=5, vulnerability_density=0.35, max_steps=300),
    "dmz": EnvConfig(num_hosts=15, num_subnets=3, vulnerability_density=0.45, max_steps=200),
    "iot_network": EnvConfig(num_hosts=20, num_subnets=4, vulnerability_density=0.6, max_steps=250),
}


@dataclass
class EpisodeInfo:
    total_reward: float = 0.0
    steps: int = 0
    hosts_compromised: int = 0
    hosts_discovered: int = 0
    goal_reached: bool = False
    attack_path: list[str] = field(default_factory=list)
    action_log: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"EpisodeInfo(reward={self.total_reward:.2f}, steps={self.steps}, "
            f"compromised={self.hosts_compromised}/{self.hosts_discovered} discovered, "
            f"goal={'✓' if self.goal_reached else '✗'})"
        )


class NetworkEnv(gym.Env):
    """
    Cyber pentesting simulation environment.

    Observation space: flat vector encoding known host states.
    Action space: Discrete — (action_type × num_hosts × num_exploits).
    """

    metadata = {"render_modes": ["human", "rgb_array", "ansi"]}

    # Feature dims per host: [discovered, compromised, os_enc(4), subnet_id, vuln_count, service_count]
    _HOST_FEATURES = 9
    _NUM_EXPLOITS = 5

    def __init__(self, cfg: Optional[ARCAConfig] = None, env_cfg: Optional[EnvConfig] = None):
        super().__init__()
        self.cfg = cfg or ARCAConfig()
        self.env_cfg = env_cfg or self.cfg.env
        self._rng = random.Random(self.cfg.seed if cfg else 42)
        self._np_rng = np.random.default_rng(self.cfg.seed if cfg else 42)

        # Use C++ backend if available and configured
        self._use_cpp = self.env_cfg.use_cpp_backend and _CPP_AVAILABLE

        self._generator = NetworkGenerator(self.env_cfg, self._rng)

        # Will be populated in reset()
        self.graph: nx.DiGraph = nx.DiGraph()
        self.hosts: dict[int, Host] = {}
        self._attacker_node: int = 0
        self._episode_info: EpisodeInfo = EpisodeInfo()
        self._step_count: int = 0

        n = self.env_cfg.num_hosts
        e = self._NUM_EXPLOITS
        num_action_types = len(ActionType)

        # Action: flat index = action_type * (n * e) + host * e + exploit
        self.action_space = spaces.Discrete(num_action_types * n * e)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(n * self._HOST_FEATURES,),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_preset(cls, preset: str, cfg: Optional[ARCAConfig] = None) -> "NetworkEnv":
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset '{preset}'. Available: {list(PRESETS)}")
        base_cfg = cfg or ARCAConfig()
        base_cfg.env = PRESETS[preset]
        return cls(cfg=base_cfg)

    @classmethod
    def from_config(cls, cfg: ARCAConfig) -> "NetworkEnv":
        return cls(cfg=cfg)

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = random.Random(seed)
            self._np_rng = np.random.default_rng(seed)

        self.graph, self.hosts = self._generator.generate()
        self._attacker_node = self._generator.attacker_node
        self._step_count = 0
        self._episode_info = EpisodeInfo()

        # Mark attacker start as discovered & compromised
        self.hosts[self._attacker_node].status = HostStatus.COMPROMISED
        self.hosts[self._attacker_node].discovered = True
        self._episode_info.hosts_compromised = 1
        self._episode_info.hosts_discovered = 1

        obs = self._get_obs()
        info = {"attacker_node": self._attacker_node, "num_hosts": len(self.hosts)}
        return obs, info

    def step(self, action: int):
        n = self.env_cfg.num_hosts
        e = self._NUM_EXPLOITS
        action_type_idx = action // (n * e)
        remainder = action % (n * e)
        target_host_idx = remainder // e
        exploit_idx = remainder % e

        action_type = ActionType(action_type_idx % len(ActionType))
        target_host = target_host_idx % n

        act = Action(
            action_type=action_type,
            target_host=target_host,
            exploit_id=exploit_idx,
            source_host=self._attacker_node,
        )

        result = self._execute_action(act)
        reward = self._compute_reward(result)
        self._step_count += 1
        self._episode_info.total_reward += reward
        self._episode_info.steps = self._step_count
        self._episode_info.action_log.append({
            "step": self._step_count,
            "action": act.to_dict(),
            "result": result.to_dict(),
            "reward": reward,
        })

        terminated = self._check_goal()
        truncated = self._step_count >= self.env_cfg.max_steps
        self._episode_info.goal_reached = terminated

        obs = self._get_obs()
        info = {
            "action_result": result.to_dict(),
            "episode_info": self._episode_info if (terminated or truncated) else None,
        }
        return obs, reward, terminated, truncated, info

    def render(self, mode="ansi"):
        lines = ["=" * 50, "ARCA Network State", "=" * 50]
        for hid, host in self.hosts.items():
            lines.append(
                f"  Host {hid:02d} [{host.subnet}] {host.os:<10} "
                f"{'🔴 PWNED' if host.status == HostStatus.COMPROMISED else '🟡 SEEN' if host.discovered else '⬛ UNKNOWN'}"
                f"  vulns={len(host.vulnerabilities)}"
            )
        lines.append(f"Step {self._step_count}/{self.env_cfg.max_steps}  "
                     f"Reward={self._episode_info.total_reward:.1f}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal mechanics
    # ------------------------------------------------------------------

    def _execute_action(self, act: Action) -> ActionResult:
        target = self.hosts.get(act.target_host)
        if target is None:
            return ActionResult(success=False, message="Invalid target")

        if act.action_type == ActionType.SCAN:
            return self._do_scan(act, target)
        elif act.action_type == ActionType.EXPLOIT:
            return self._do_exploit(act, target)
        elif act.action_type == ActionType.PIVOT:
            return self._do_pivot(act, target)
        elif act.action_type == ActionType.EXFILTRATE:
            return self._do_exfiltrate(act, target)
        else:
            return ActionResult(success=False, message="Unknown action")

    def _do_scan(self, act: Action, target: Host) -> ActionResult:
        # Check reachability
        if not self._is_reachable(act.source_host, act.target_host):
            return ActionResult(success=False, message="Host unreachable")
        was_new = not target.discovered
        target.discovered = True
        if was_new:
            self._episode_info.hosts_discovered += 1
        return ActionResult(
            success=True,
            discovered_hosts=[act.target_host] if was_new else [],
            message=f"Scanned host {act.target_host}: {target.os}, {len(target.vulnerabilities)} vulns",
        )

    def _do_exploit(self, act: Action, target: Host) -> ActionResult:
        if not target.discovered:
            return ActionResult(success=False, message="Host not discovered yet")
        if not self._is_reachable(act.source_host, act.target_host):
            return ActionResult(success=False, message="Host unreachable")
        if target.status == HostStatus.COMPROMISED:
            return ActionResult(success=False, message="Already compromised")
        # Check if exploit matches a vulnerability
        if act.exploit_id < len(target.vulnerabilities):
            vuln = target.vulnerabilities[act.exploit_id]
            success_prob = vuln.get("exploit_prob", 0.6)
            if self._np_rng.random() < success_prob:
                target.status = HostStatus.COMPROMISED
                self._attacker_node = act.target_host  # pivot point
                self._episode_info.hosts_compromised += 1
                self._episode_info.attack_path.append(
                    f"{act.source_host}→{act.target_host}(CVE:{vuln.get('cve','?')})"
                )
                return ActionResult(
                    success=True,
                    compromised_host=act.target_host,
                    message=f"Exploited {target.os} via {vuln.get('name','unknown')}",
                )
        return ActionResult(success=False, message="Exploit failed or no matching vuln")

    def _do_pivot(self, act: Action, target: Host) -> ActionResult:
        if target.status != HostStatus.COMPROMISED:
            return ActionResult(success=False, message="Cannot pivot to non-compromised host")
        self._attacker_node = act.target_host
        return ActionResult(success=True, message=f"Pivoted to host {act.target_host}")

    def _do_exfiltrate(self, act: Action, target: Host) -> ActionResult:
        if target.status != HostStatus.COMPROMISED:
            return ActionResult(success=False, message="Cannot exfiltrate from non-compromised host")
        data_value = target.data_value
        return ActionResult(success=True, data_exfiltrated=data_value,
                            message=f"Exfiltrated {data_value:.1f} units from host {act.target_host}")

    def _is_reachable(self, src: int, dst: int) -> bool:
        if src == dst:
            return True
        try:
            return nx.has_path(self.graph, src, dst)
        except nx.NetworkXError:
            return False

    def _compute_reward(self, result: ActionResult) -> float:
        if not result.success:
            return self.env_cfg.reward_step
        r = 0.0
        if result.discovered_hosts:
            r += self.env_cfg.reward_discovery * len(result.discovered_hosts)
        if result.compromised_host is not None:
            r += self.env_cfg.reward_exploit
        if result.data_exfiltrated > 0:
            r += result.data_exfiltrated * 2.0
        r += self.env_cfg.reward_step
        return r

    def _check_goal(self) -> bool:
        n_compromised = sum(1 for h in self.hosts.values() if h.status == HostStatus.COMPROMISED)
        return n_compromised >= max(3, len(self.hosts) // 2)

    def _get_obs(self) -> np.ndarray:
        n = self.env_cfg.num_hosts
        obs = np.zeros(n * self._HOST_FEATURES, dtype=np.float32)
        os_map = {"Windows": 0, "Linux": 1, "macOS": 2, "IoT": 3}
        for i in range(n):
            host = self.hosts.get(i)
            base = i * self._HOST_FEATURES
            if host is None:
                continue
            obs[base + 0] = float(host.discovered)
            obs[base + 1] = float(host.status == HostStatus.COMPROMISED)
            os_idx = os_map.get(host.os, 0)
            obs[base + 2 + os_idx] = 1.0
            obs[base + 6] = host.subnet / max(self.env_cfg.num_subnets, 1)
            obs[base + 7] = len(host.vulnerabilities) / 10.0
            obs[base + 8] = len(host.services) / 10.0
        return obs

    # ------------------------------------------------------------------
    # Utility / introspection
    # ------------------------------------------------------------------

    @property
    def episode_info(self) -> EpisodeInfo:
        return self._episode_info

    def get_network_graph(self) -> nx.DiGraph:
        return self.graph

    def get_hosts(self) -> dict[int, Host]:
        return self.hosts

    def get_state_dict(self) -> dict:
        return {
            "step": self._step_count,
            "attacker_node": self._attacker_node,
            "hosts": {hid: h.to_dict() for hid, h in self.hosts.items()},
            "episode_info": {
                "total_reward": self._episode_info.total_reward,
                "hosts_compromised": self._episode_info.hosts_compromised,
                "hosts_discovered": self._episode_info.hosts_discovered,
                "attack_path": self._episode_info.attack_path,
            },
        }