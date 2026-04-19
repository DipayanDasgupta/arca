"""
arca/sim/environment.py  (v3.2 — Action Masking + Curriculum hooks)
=====================================================================
New in v3.2:
  - get_action_mask()  → np.bool_ array of shape [n_actions]
    Marks invalid actions so the policy never wastes steps on them.
  - CurriculumEnvConfig  (used by CurriculumScheduler in curriculum.py)
  - Fully backward compatible with v3.1
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

# PyG optional
try:
    from torch_geometric.data import Data as PyGData
    import torch
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False

# C++ optional
try:
    from arca._cpp_sim import SimEngine as _CppSimEngine  # type: ignore
    _CPP_AVAILABLE = True
except ImportError:
    _CppSimEngine = None
    _CPP_AVAILABLE = False

PRESETS = {
    "small_office": EnvConfig(
        num_hosts=8,  num_subnets=2,
        vulnerability_density=0.5, max_steps=150,
    ),
    "enterprise": EnvConfig(
        num_hosts=25, num_subnets=5,
        vulnerability_density=0.35, max_steps=300,
    ),
    "dmz": EnvConfig(
        num_hosts=15, num_subnets=3,
        vulnerability_density=0.45, max_steps=200,
    ),
    "iot_network": EnvConfig(
        num_hosts=20, num_subnets=4,
        vulnerability_density=0.6,  max_steps=250,
    ),
}


@dataclass
class EpisodeInfo:
    total_reward:      float = 0.0
    steps:             int   = 0
    hosts_compromised: int   = 0
    hosts_discovered:  int   = 0
    goal_reached:      bool  = False
    attack_path:       list[str] = field(default_factory=list)
    action_log:        list[dict] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"EpisodeInfo(reward={self.total_reward:.2f}, steps={self.steps}, "
            f"compromised={self.hosts_compromised}/{self.hosts_discovered} disc, "
            f"goal={'✓' if self.goal_reached else '✗'})"
        )


class NetworkEnv(gym.Env):
    """
    Cyber pentesting simulation (Gymnasium).

    Key additions vs v3.1:
      - get_action_mask() → boolean numpy array [n_actions]
      - mask applied in step() to return observation with mask info
    """

    metadata      = {"render_modes": ["human", "rgb_array", "ansi"]}
    _HOST_FEATURES = 9
    _NUM_EXPLOITS  = 5

    def __init__(
        self,
        cfg:     Optional[ARCAConfig] = None,
        env_cfg: Optional[EnvConfig]  = None,
    ):
        super().__init__()
        self.cfg      = cfg     or ARCAConfig()
        self.env_cfg  = env_cfg or self.cfg.env
        self._rng     = random.Random(self.cfg.seed if cfg else 42)
        self._np_rng  = np.random.default_rng(self.cfg.seed if cfg else 42)

        self._use_cpp   = self.env_cfg.use_cpp_backend and _CPP_AVAILABLE
        self._generator = NetworkGenerator(self.env_cfg, self._rng)

        self.graph:  nx.DiGraph      = nx.DiGraph()
        self.hosts:  dict[int, Host] = {}
        self._attacker_node: int     = 0
        self._episode_info: EpisodeInfo = EpisodeInfo()
        self._step_count:   int      = 0

        n = self.env_cfg.num_hosts
        e = self._NUM_EXPLOITS
        self.action_space      = spaces.Discrete(len(ActionType) * n * e)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(n * self._HOST_FEATURES,),
            dtype=np.float32,
        )

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_preset(
        cls, preset: str, cfg: Optional[ARCAConfig] = None
    ) -> "NetworkEnv":
        if preset not in PRESETS:
            raise ValueError(
                f"Unknown preset '{preset}'. Options: {list(PRESETS)}"
            )
        base_cfg     = cfg or ARCAConfig()
        base_cfg.env = PRESETS[preset]
        return cls(cfg=base_cfg)

    @classmethod
    def from_config(cls, cfg: ARCAConfig) -> "NetworkEnv":
        return cls(cfg=cfg)

    # ── Gym ───────────────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng    = random.Random(seed)
            self._np_rng = np.random.default_rng(seed)

        self.graph, self.hosts = self._generator.generate()
        self._attacker_node    = self._generator.attacker_node
        self._step_count       = 0
        self._episode_info     = EpisodeInfo()

        self.hosts[self._attacker_node].status     = HostStatus.COMPROMISED
        self.hosts[self._attacker_node].discovered  = True
        self._episode_info.hosts_compromised = 1
        self._episode_info.hosts_discovered  = 1

        return self._get_obs(), {
            "attacker_node": self._attacker_node,
            "num_hosts":     len(self.hosts),
            "action_mask":   self.get_action_mask(),
        }

    def step(self, action: int):
        n = self.env_cfg.num_hosts
        e = self._NUM_EXPLOITS

        action_type_idx = action // (n * e)
        remainder       = action % (n * e)
        target_host_idx = remainder // e
        exploit_idx     = remainder % e

        action_type = ActionType(action_type_idx % len(ActionType))
        target_host = target_host_idx % n

        act = Action(
            action_type = action_type,
            target_host = target_host,
            exploit_id  = exploit_idx,
            source_host = self._attacker_node,
        )

        result = self._execute_action(act)
        reward = self._compute_reward(result)
        self._step_count += 1
        self._episode_info.total_reward += reward
        self._episode_info.steps         = self._step_count
        self._episode_info.action_log.append({
            "step":   self._step_count,
            "action": act.to_dict(),
            "result": result.to_dict(),
            "reward": reward,
        })

        terminated = self._check_goal()
        truncated  = self._step_count >= self.env_cfg.max_steps
        self._episode_info.goal_reached = terminated

        return (
            self._get_obs(),
            reward,
            terminated,
            truncated,
            {
                "action_result": result.to_dict(),
                "action_mask":   self.get_action_mask(),
                "episode_info":  (
                    self._episode_info
                    if (terminated or truncated) else None
                ),
            },
        )

    def render(self, mode="ansi"):
        lines = ["=" * 50, "ARCA Network State", "=" * 50]
        for hid, host in self.hosts.items():
            lines.append(
                f"  Host {hid:02d} [{host.subnet}] {host.os:<10} "
                f"{'💀 PWNED' if host.status == HostStatus.COMPROMISED else '🔍 SEEN' if host.discovered else '❓ ?'}"
                f"  vulns={len(host.vulnerabilities)}"
            )
        lines.append(
            f"Step {self._step_count}/{self.env_cfg.max_steps}  "
            f"Reward={self._episode_info.total_reward:.1f}"
        )
        return "\n".join(lines)

    # ── Action Masking (v3.2) ─────────────────────────────────────────────────

    def get_action_mask(self) -> np.ndarray:
        """
        Returns a boolean array of shape [n_actions] where True = valid.

        Rules:
          SCAN       — valid if host is reachable AND not already discovered
          EXPLOIT    — valid if host is discovered, reachable, not compromised,
                       AND has at least (exploit_idx+1) vulnerabilities
          PIVOT      — valid if host is already compromised (free move)
          EXFILTRATE — valid if host is compromised
        """
        n    = self.env_cfg.num_hosts
        e    = self._NUM_EXPLOITS
        n_at = len(ActionType)
        mask = np.zeros(n_at * n * e, dtype=np.bool_)

        for at in ActionType:
            for target in range(n):
                h = self.hosts.get(target)
                if h is None:
                    continue
                reachable = self._is_reachable(self._attacker_node, target)

                for exploit_idx in range(e):
                    action_id = at * (n * e) + target * e + exploit_idx
                    valid = False

                    if at == ActionType.SCAN:
                        # Only worth scanning undiscovered reachable hosts
                        valid = reachable and not h.discovered

                    elif at == ActionType.EXPLOIT:
                        # Must be discovered, reachable, not yet owned,
                        # and exploit_idx must index a real vulnerability
                        valid = (
                            h.discovered
                            and reachable
                            and h.status != HostStatus.COMPROMISED
                            and exploit_idx < len(h.vulnerabilities)
                        )

                    elif at == ActionType.PIVOT:
                        # Can pivot to any compromised host (exploit_idx irrelevant)
                        valid = (
                            h.status == HostStatus.COMPROMISED
                            and target != self._attacker_node
                            and exploit_idx == 0   # only first exploit slot
                        )

                    elif at == ActionType.EXFILTRATE:
                        # Exfil from any compromised host with exploit_idx==0
                        valid = (
                            h.status == HostStatus.COMPROMISED
                            and exploit_idx == 0
                        )

                    mask[action_id] = valid

        # Safety: always allow at least one action (random scan) to avoid
        # all-False masks that break Categorical
        if not mask.any():
            for target in range(n):
                h = self.hosts.get(target)
                if h and self._is_reachable(self._attacker_node, target):
                    action_id = ActionType.SCAN * (n * e) + target * e + 0
                    mask[action_id] = True
                    break

        return mask

    # ── v3 PyG Data ───────────────────────────────────────────────────────────

    def get_pyg_data(self) -> "PyGData":
        if not PYG_AVAILABLE:
            raise ImportError(
                "torch-geometric required: pip install torch-geometric"
            )
        n      = self.env_cfg.num_hosts
        os_map = {"Windows": 0, "Linux": 1, "macOS": 2, "IoT": 3}
        feat   = np.zeros((n, self._HOST_FEATURES), dtype=np.float32)

        for i in range(n):
            host = self.hosts.get(i)
            if host is None:
                continue
            feat[i, 0] = float(host.discovered)
            feat[i, 1] = float(host.status == HostStatus.COMPROMISED)
            feat[i, 2 + os_map.get(host.os, 0)] = 1.0
            feat[i, 6] = host.subnet / max(self.env_cfg.num_subnets, 1)
            feat[i, 7] = len(host.vulnerabilities) / 10.0
            feat[i, 8] = len(host.services) / 10.0

        x = torch.tensor(feat, dtype=torch.float32)
        if self.graph.number_of_edges() > 0:
            edges      = list(self.graph.edges())
            edge_index = torch.tensor(edges, dtype=torch.long).T
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        return PyGData(x=x, edge_index=edge_index)

    # ── LLM reward shaping ────────────────────────────────────────────────────

    def modify_reward_weights_from_critique(self, critique: str) -> None:
        c = critique.lower()
        if any(w in c for w in ["critical", "high-value", "crown"]):
            self.env_cfg.reward_exploit  *= 1.2
        if any(w in c for w in ["scan", "discover", "recon"]):
            self.env_cfg.reward_discovery *= 1.1
        if "penalty" in c or "step penalty" in c:
            self.env_cfg.reward_step = min(-0.1, self.env_cfg.reward_step * 0.8)

    # ── Action mechanics ──────────────────────────────────────────────────────

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
        return ActionResult(success=False, message="Unknown action")

    def _do_scan(self, act, target) -> ActionResult:
        if not self._is_reachable(act.source_host, act.target_host):
            return ActionResult(success=False, message="Host unreachable")
        was_new           = not target.discovered
        target.discovered = True
        if was_new:
            self._episode_info.hosts_discovered += 1
        return ActionResult(
            success          = True,
            discovered_hosts = [act.target_host] if was_new else [],
            message          = (
                f"Scanned host {act.target_host}: "
                f"{target.os}, {len(target.vulnerabilities)} vulns"
            ),
        )

    def _do_exploit(self, act, target) -> ActionResult:
        if not target.discovered:
            return ActionResult(success=False, message="Host not discovered yet")
        if not self._is_reachable(act.source_host, act.target_host):
            return ActionResult(success=False, message="Host unreachable")
        if target.status == HostStatus.COMPROMISED:
            return ActionResult(success=False, message="Already compromised")
        if act.exploit_id < len(target.vulnerabilities):
            vuln = target.vulnerabilities[act.exploit_id]
            prob = vuln.get("exploit_prob", 0.6)
            if self._np_rng.random() < prob:
                target.status = HostStatus.COMPROMISED
                self._attacker_node = act.target_host
                self._episode_info.hosts_compromised += 1
                self._episode_info.attack_path.append(
                    f"{act.source_host}→{act.target_host}"
                    f"(CVE:{vuln.get('cve','?')})"
                )
                return ActionResult(
                    success          = True,
                    compromised_host = act.target_host,
                    message          = (
                        f"Exploited {target.os} via {vuln.get('name','?')}"
                    ),
                )
        return ActionResult(success=False, message="Exploit failed")

    def _do_pivot(self, act, target) -> ActionResult:
        if target.status != HostStatus.COMPROMISED:
            return ActionResult(success=False, message="Cannot pivot")
        self._attacker_node = act.target_host
        return ActionResult(
            success=True, message=f"Pivoted to host {act.target_host}"
        )

    def _do_exfiltrate(self, act, target) -> ActionResult:
        if target.status != HostStatus.COMPROMISED:
            return ActionResult(success=False, message="Cannot exfiltrate")
        return ActionResult(
            success          = True,
            data_exfiltrated = target.data_value,
            message          = f"Exfiltrated {target.data_value:.1f} units",
        )

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
            host  = self.hosts.get(result.compromised_host)
            bonus = 1.5 if (host and host.is_critical) else 1.0
            r    += self.env_cfg.reward_exploit * bonus
        if result.data_exfiltrated > 0:
            r += result.data_exfiltrated * 2.0
        r += self.env_cfg.reward_step
        return r

    def _check_goal(self) -> bool:
        n = sum(
            1 for h in self.hosts.values()
            if h.status == HostStatus.COMPROMISED
        )
        return n >= max(3, len(self.hosts) // 2)

    def _get_obs(self) -> np.ndarray:
        n      = self.env_cfg.num_hosts
        obs    = np.zeros(n * self._HOST_FEATURES, dtype=np.float32)
        os_map = {"Windows": 0, "Linux": 1, "macOS": 2, "IoT": 3}
        for i in range(n):
            host = self.hosts.get(i)
            if host is None:
                continue
            base = i * self._HOST_FEATURES
            obs[base + 0] = float(host.discovered)
            obs[base + 1] = float(host.status == HostStatus.COMPROMISED)
            obs[base + 2 + os_map.get(host.os, 0)] = 1.0
            obs[base + 6] = host.subnet / max(self.env_cfg.num_subnets, 1)
            obs[base + 7] = len(host.vulnerabilities) / 10.0
            obs[base + 8] = len(host.services) / 10.0
        return obs

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def episode_info(self) -> EpisodeInfo:
        return self._episode_info

    def get_network_graph(self) -> nx.DiGraph:
        return self.graph

    def get_hosts(self) -> dict[int, Host]:
        return self.hosts

    def get_state_dict(self) -> dict:
        return {
            "step":          self._step_count,
            "attacker_node": self._attacker_node,
            "hosts":         {hid: h.to_dict() for hid, h in self.hosts.items()},
            "episode_info": {
                "total_reward":      self._episode_info.total_reward,
                "hosts_compromised": self._episode_info.hosts_compromised,
                "hosts_discovered":  self._episode_info.hosts_discovered,
                "attack_path":       self._episode_info.attack_path,
            },
        }