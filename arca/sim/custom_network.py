"""
arca.sim.custom_network
=======================
Lets users define and load their own real-world-inspired network topologies.

Usage (Python):
    from arca.sim.custom_network import CustomNetworkBuilder
    env = CustomNetworkBuilder.from_yaml("my_network.yaml")

Usage (CLI):
    arca train --network my_network.yaml

YAML Format:
    See examples/my_home_network.yaml for a full template.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import networkx as nx
import yaml

from arca.sim.host import Host, HostStatus
from arca.core.config import ARCAConfig, EnvConfig


# ── Well-known CVE library (expanded) ────────────────────────────────────────

CVE_LIBRARY = {
    # Windows vulns
    "EternalBlue":      {"cve": "CVE-2017-0144", "exploit_prob": 0.78, "os": ["Windows"], "service": "SMB",  "severity": "CRITICAL"},
    "BlueKeep":         {"cve": "CVE-2019-0708", "exploit_prob": 0.72, "os": ["Windows"], "service": "RDP",  "severity": "CRITICAL"},
    "ProxyLogon":       {"cve": "CVE-2021-26855","exploit_prob": 0.71, "os": ["Windows"], "service": "HTTP", "severity": "CRITICAL"},
    "PrintNightmare":   {"cve": "CVE-2021-34527","exploit_prob": 0.65, "os": ["Windows"], "service": "SMB",  "severity": "HIGH"},
    "ZeroLogon":        {"cve": "CVE-2020-1472", "exploit_prob": 0.85, "os": ["Windows"], "service": "SMB",  "severity": "CRITICAL"},
    "MS17-010":         {"cve": "CVE-2017-0145", "exploit_prob": 0.75, "os": ["Windows"], "service": "SMB",  "severity": "CRITICAL"},
    # Linux vulns
    "Log4Shell":        {"cve": "CVE-2021-44228","exploit_prob": 0.82, "os": ["Linux"],   "service": "HTTP", "severity": "CRITICAL"},
    "Shellshock":       {"cve": "CVE-2014-6271", "exploit_prob": 0.66, "os": ["Linux"],   "service": "HTTP", "severity": "HIGH"},
    "Heartbleed":       {"cve": "CVE-2014-0160", "exploit_prob": 0.61, "os": ["Linux"],   "service": "HTTPS","severity": "HIGH"},
    "DirtyCOW":         {"cve": "CVE-2016-5195", "exploit_prob": 0.57, "os": ["Linux"],   "service": "SSH",  "severity": "HIGH"},
    "PwnKit":           {"cve": "CVE-2021-4034", "exploit_prob": 0.80, "os": ["Linux"],   "service": "SSH",  "severity": "HIGH"},
    "Spring4Shell":     {"cve": "CVE-2022-22965","exploit_prob": 0.68, "os": ["Linux"],   "service": "HTTP", "severity": "CRITICAL"},
    # IoT / embedded
    "IoTDefaultCreds":  {"cve": "CVE-2020-8958", "exploit_prob": 0.91, "os": ["IoT"],     "service": "Telnet","severity": "CRITICAL"},
    "RouteRCE":         {"cve": "CVE-2021-20090","exploit_prob": 0.74, "os": ["IoT"],     "service": "HTTP", "severity": "HIGH"},
    # macOS
    "TCC_Bypass":       {"cve": "CVE-2023-41990","exploit_prob": 0.52, "os": ["macOS"],   "service": "HTTP", "severity": "HIGH"},
    # Web / app layer
    "SQLInjection":     {"cve": "CWE-89",        "exploit_prob": 0.70, "os": ["Linux","Windows"], "service": "HTTP", "severity": "HIGH"},
    "RCE_WebApp":       {"cve": "CWE-78",        "exploit_prob": 0.63, "os": ["Linux","Windows"], "service": "HTTP", "severity": "CRITICAL"},
    # Router / network
    "Cisco_CVE":        {"cve": "CVE-2023-20198","exploit_prob": 0.77, "os": ["Router"],  "service": "HTTP", "severity": "CRITICAL"},
    "RouterDefaultPwd": {"cve": "CWE-798",       "exploit_prob": 0.88, "os": ["Router"],  "service": "HTTP", "severity": "HIGH"},
}

DEFAULT_SERVICES = {
    "Windows": ["SMB", "RDP", "WinRM", "IIS", "MSSQL", "HTTP"],
    "Linux":   ["SSH", "HTTP", "HTTPS", "FTP", "PostgreSQL", "MySQL"],
    "macOS":   ["SSH", "HTTP", "AFP", "VNC"],
    "IoT":     ["Telnet", "HTTP", "MQTT", "SNMP"],
    "Router":  ["HTTP", "HTTPS", "SSH", "Telnet", "SNMP"],
    "Android": ["ADB", "HTTP"],
    "Windows Server": ["SMB", "RDP", "IIS", "MSSQL", "AD-DS"],
}


# ── Schema ────────────────────────────────────────────────────────────────────

@dataclass
class HostSpec:
    """Raw spec from YAML before conversion to Host."""
    id: int
    name: str
    os: str
    subnet: int
    ip: str
    services: list[str] = field(default_factory=list)
    vulns: list[str] = field(default_factory=list)        # CVE names from CVE_LIBRARY
    is_critical: bool = False
    firewall: bool = False
    data_value: float = 5.0
    notes: str = ""

    @classmethod
    def from_dict(cls, d: dict, host_id: int) -> "HostSpec":
        return cls(
            id=host_id,
            name=d.get("name", f"host_{host_id}"),
            os=d.get("os", "Linux"),
            subnet=d.get("subnet", 0),
            ip=d.get("ip", f"192.168.{d.get('subnet',0)}.{host_id+1}"),
            services=d.get("services", DEFAULT_SERVICES.get(d.get("os", "Linux"), ["SSH"])),
            vulns=d.get("vulns", []),
            is_critical=d.get("is_critical", False),
            firewall=d.get("firewall", False),
            data_value=float(d.get("data_value", 5.0)),
            notes=d.get("notes", ""),
        )


@dataclass
class NetworkSpec:
    """Full network topology spec from YAML."""
    name: str
    description: str
    attacker_entry: str          # IP or host name of entry point
    hosts: list[HostSpec]
    connections: list[tuple[int, int]]   # (host_id_a, host_id_b) bidirectional
    subnet_names: dict[int, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "NetworkSpec":
        raw_hosts = d.get("hosts", [])
        hosts = [HostSpec.from_dict(h, i) for i, h in enumerate(raw_hosts)]

        # Parse connections: support both [a, b] and {from: a, to: b}
        raw_conns = d.get("connections", [])
        connections = []
        for c in raw_conns:
            if isinstance(c, list) and len(c) == 2:
                connections.append((int(c[0]), int(c[1])))
            elif isinstance(c, dict):
                connections.append((int(c["from"]), int(c["to"])))

        return cls(
            name=d.get("name", "custom_network"),
            description=d.get("description", ""),
            attacker_entry=d.get("attacker_entry", hosts[0].ip if hosts else ""),
            hosts=hosts,
            connections=connections,
            subnet_names=d.get("subnet_names", {}),
        )


# ── Builder ───────────────────────────────────────────────────────────────────

class CustomNetworkBuilder:
    """
    Build a NetworkEnv from a user-defined YAML/JSON topology.

    Example:
        env = CustomNetworkBuilder.from_yaml("my_home_network.yaml")
        agent = ARCAAgent(env=env)
        agent.train(timesteps=30_000)
    """

    @classmethod
    def from_yaml(cls, path: str | Path, cfg: Optional[ARCAConfig] = None) -> "CustomNetworkEnv":
        with open(path) as f:
            data = yaml.safe_load(f)
        spec = NetworkSpec.from_dict(data)
        return cls._build(spec, cfg)

    @classmethod
    def from_json(cls, path: str | Path, cfg: Optional[ARCAConfig] = None) -> "CustomNetworkEnv":
        with open(path) as f:
            data = json.load(f)
        spec = NetworkSpec.from_dict(data)
        return cls._build(spec, cfg)

    @classmethod
    def from_dict(cls, data: dict, cfg: Optional[ARCAConfig] = None) -> "CustomNetworkEnv":
        spec = NetworkSpec.from_dict(data)
        return cls._build(spec, cfg)

    @classmethod
    def _build(cls, spec: NetworkSpec, cfg: Optional[ARCAConfig]) -> "CustomNetworkEnv":
        cfg = cfg or ARCAConfig()

        # Override env config to match the custom network
        cfg.env = EnvConfig(
            num_hosts=len(spec.hosts),
            num_subnets=len({h.subnet for h in spec.hosts}),
            vulnerability_density=1.0,   # user explicitly set vulns
            max_steps=max(150, len(spec.hosts) * 20),
        )

        return CustomNetworkEnv(spec=spec, cfg=cfg)

    @staticmethod
    def generate_template(
        path: str | Path,
        preset: str = "home",
        overwrite: bool = False,
    ) -> None:
        """
        Generate a YAML template for the user to fill in.

        Presets: "home" | "small_office" | "datacenter"
        """
        templates = {
            "home": HOME_NETWORK_TEMPLATE,
            "small_office": SMALL_OFFICE_TEMPLATE,
            "datacenter": DATACENTER_TEMPLATE,
        }
        content = templates.get(preset, HOME_NETWORK_TEMPLATE)
        p = Path(path)
        if p.exists() and not overwrite:
            raise FileExistsError(f"{path} already exists. Pass overwrite=True to replace.")
        p.write_text(content)
        print(f"[ARCA] Template written → {path}")
        print(f"[ARCA] Edit the file, then run: CustomNetworkBuilder.from_yaml('{path}')")


# ── CustomNetworkEnv ──────────────────────────────────────────────────────────

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from arca.sim.action import Action, ActionType, ActionResult
from arca.core.config import ARCAConfig, EnvConfig
from arca.sim.environment import EpisodeInfo


class CustomNetworkEnv(gym.Env):
    """
    Gymnasium environment backed by a user-defined network topology.
    Identical interface to NetworkEnv — works with all ARCA tools.
    """

    metadata = {"render_modes": ["ansi", "human"]}
    _HOST_FEATURES = 9
    _NUM_EXPLOITS = 5

    def __init__(self, spec: NetworkSpec, cfg: Optional[ARCAConfig] = None):
        super().__init__()
        self.spec = spec
        self.cfg = cfg or ARCAConfig()
        self.env_cfg = self.cfg.env
        self._rng = random.Random(self.cfg.seed)
        self._np_rng = np.random.default_rng(self.cfg.seed)

        # Build static graph + hosts from spec
        self._base_graph, self._base_hosts = self._build_from_spec()
        self._find_attacker_node()

        # Runtime state
        self.graph: nx.DiGraph = nx.DiGraph()
        self.hosts: dict[int, Host] = {}
        self._attacker_node: int = 0
        self._episode_info = EpisodeInfo()
        self._step_count = 0

        n = len(spec.hosts)
        e = self._NUM_EXPLOITS
        self.action_space = spaces.Discrete(len(ActionType) * n * e)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(n * self._HOST_FEATURES,),
            dtype=np.float32,
        )

    def _build_from_spec(self) -> tuple[nx.DiGraph, dict[int, Host]]:
        g = nx.DiGraph()
        hosts: dict[int, Host] = {}

        for spec_host in self.spec.hosts:
            # Resolve vulns from CVE_LIBRARY
            resolved_vulns = []
            for vname in spec_host.vulns:
                if vname in CVE_LIBRARY:
                    resolved_vulns.append({**CVE_LIBRARY[vname], "name": vname})
                else:
                    # Unknown vuln — add with moderate prob
                    resolved_vulns.append({
                        "name": vname, "cve": "UNKNOWN",
                        "exploit_prob": 0.5, "os": [spec_host.os],
                        "service": "HTTP", "severity": "MEDIUM",
                    })

            h = Host(
                id=spec_host.id,
                subnet=spec_host.subnet,
                os=spec_host.os,
                ip=spec_host.ip,
                services=spec_host.services,
                vulnerabilities=resolved_vulns,
                is_critical=spec_host.is_critical,
                firewall=spec_host.firewall,
                data_value=spec_host.data_value,
            )
            hosts[spec_host.id] = h
            g.add_node(spec_host.id, name=spec_host.name, ip=spec_host.ip, os=spec_host.os)

        # Add edges from spec connections (bidirectional)
        for a, b in self.spec.connections:
            g.add_edge(a, b)
            g.add_edge(b, a)

        return g, hosts

    def _find_attacker_node(self):
        # Find the host matching attacker_entry IP
        for spec_host in self.spec.hosts:
            if spec_host.ip == self.spec.attacker_entry:
                self._default_attacker = spec_host.id
                return
        self._default_attacker = 0

    # ── Gym interface ─────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Deep copy hosts (reset status)
        self.hosts = {}
        for hid, h in self._base_hosts.items():
            self.hosts[hid] = Host(
                id=h.id, subnet=h.subnet, os=h.os, ip=h.ip,
                services=list(h.services),
                vulnerabilities=list(h.vulnerabilities),
                is_critical=h.is_critical, firewall=h.firewall,
                data_value=h.data_value,
            )
        self.graph = self._base_graph.copy()
        self._attacker_node = self._default_attacker
        self._step_count = 0
        self._episode_info = EpisodeInfo()

        self.hosts[self._attacker_node].status = HostStatus.COMPROMISED
        self.hosts[self._attacker_node].discovered = True
        self._episode_info.hosts_compromised = 1
        self._episode_info.hosts_discovered = 1

        return self._get_obs(), {"attacker_node": self._attacker_node, "num_hosts": len(self.hosts)}

    def step(self, action: int):
        n = len(self.spec.hosts)
        e = self._NUM_EXPLOITS
        action_type = ActionType(action // (n * e) % len(ActionType))
        remainder = action % (n * e)
        target_host = (remainder // e) % n
        exploit_idx = remainder % e

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

        return self._get_obs(), reward, terminated, truncated, {
            "action_result": result.to_dict(),
            "episode_info": self._episode_info if (terminated or truncated) else None,
        }

    def render(self, mode="ansi"):
        lines = ["=" * 60, f"ARCA Custom Network: {self.spec.name}", "=" * 60]
        for hid, host in self.hosts.items():
            name = self.spec.hosts[hid].name
            lines.append(
                f"  [{hid:02d}] {name:<20} {host.ip:<16} {host.os:<12} "
                f"{'🔴 PWNED' if host.status == HostStatus.COMPROMISED else '🟡 SEEN' if host.discovered else '⬛ ?'}"
                f"  vulns={len(host.vulnerabilities)}"
            )
        lines.append(f"\nStep {self._step_count}/{self.env_cfg.max_steps}  "
                     f"Reward={self._episode_info.total_reward:.1f}  "
                     f"Compromised={self._episode_info.hosts_compromised}/{len(self.hosts)}")
        return "\n".join(lines)

    # ── Mechanics (reuse sim logic) ───────────────────────────────────────────

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
        return ActionResult(success=False, message="Unknown")

    def _do_scan(self, act: Action, target: Host) -> ActionResult:
        if not nx.has_path(self.graph, act.source_host, act.target_host):
            return ActionResult(success=False, message="Unreachable")
        was_new = not target.discovered
        target.discovered = True
        if was_new:
            self._episode_info.hosts_discovered += 1
        return ActionResult(success=True, discovered_hosts=[act.target_host] if was_new else [],
                            message=f"Scanned {target.ip}: {target.os}, {len(target.vulnerabilities)} vulns")

    def _do_exploit(self, act: Action, target: Host) -> ActionResult:
        if not target.discovered or target.status == HostStatus.COMPROMISED:
            return ActionResult(success=False, message="Not discoverable/already owned")
        if not nx.has_path(self.graph, act.source_host, act.target_host):
            return ActionResult(success=False, message="Unreachable")
        if act.exploit_id < len(target.vulnerabilities):
            vuln = target.vulnerabilities[act.exploit_id]
            prob = vuln.get("exploit_prob", 0.5)
            # Firewall reduces probability
            if target.firewall:
                prob *= 0.5
            if self._np_rng.random() < prob:
                target.status = HostStatus.COMPROMISED
                self._attacker_node = act.target_host
                self._episode_info.hosts_compromised += 1
                cve = vuln.get("cve", "?")
                self._episode_info.attack_path.append(
                    f"{act.source_host}→{act.target_host}({vuln.get('name','?')}:{cve})"
                )
                return ActionResult(success=True, compromised_host=act.target_host,
                                    message=f"Exploited {target.ip} via {vuln.get('name')} [{cve}]")
        return ActionResult(success=False, message="Exploit failed")

    def _do_pivot(self, act: Action, target: Host) -> ActionResult:
        if target.status != HostStatus.COMPROMISED:
            return ActionResult(success=False, message="Not compromised")
        self._attacker_node = act.target_host
        return ActionResult(success=True, message=f"Pivoted to {target.ip}")

    def _do_exfiltrate(self, act: Action, target: Host) -> ActionResult:
        if target.status != HostStatus.COMPROMISED:
            return ActionResult(success=False, message="Not compromised")
        return ActionResult(success=True, data_exfiltrated=target.data_value,
                            message=f"Exfiltrated {target.data_value:.1f} units from {target.ip}")

    def _compute_reward(self, result: ActionResult) -> float:
        if not result.success:
            return self.env_cfg.reward_step
        r = 0.0
        if result.discovered_hosts:
            r += self.env_cfg.reward_discovery * len(result.discovered_hosts)
        if result.compromised_host is not None:
            h = self.hosts[result.compromised_host]
            bonus = 2.0 if h.is_critical else 1.0
            r += self.env_cfg.reward_exploit * bonus
        if result.data_exfiltrated > 0:
            r += result.data_exfiltrated * 2.0
        r += self.env_cfg.reward_step
        return r

    def _check_goal(self) -> bool:
        n = sum(1 for h in self.hosts.values() if h.status == HostStatus.COMPROMISED)
        return n >= max(3, len(self.hosts) // 2)

    def _get_obs(self) -> np.ndarray:
        n = len(self.spec.hosts)
        obs = np.zeros(n * self._HOST_FEATURES, dtype=np.float32)
        os_map = {"Windows": 0, "Linux": 1, "macOS": 2, "IoT": 3, "Router": 3, "Android": 2, "Windows Server": 0}
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
            "step": self._step_count,
            "attacker_node": self._attacker_node,
            "network_name": self.spec.name,
            "hosts": {
                hid: {
                    **h.to_dict(),
                    "name": self.spec.hosts[hid].name,
                    "notes": self.spec.hosts[hid].notes,
                }
                for hid, h in self.hosts.items()
            },
            "episode_info": {
                "total_reward": self._episode_info.total_reward,
                "hosts_compromised": self._episode_info.hosts_compromised,
                "hosts_discovered": self._episode_info.hosts_discovered,
                "attack_path": self._episode_info.attack_path,
            },
        }

    def print_cve_library(self):
        """Helper: show all available CVEs for YAML authoring."""
        print("\nAvailable CVEs for your YAML topology:")
        print(f"{'Name':<20} {'CVE':<18} {'Prob':>5}  {'Severity':<10} {'OS'}")
        print("-" * 75)
        for name, info in CVE_LIBRARY.items():
            print(f"{name:<20} {info['cve']:<18} {info['exploit_prob']:>5.0%}  {info['severity']:<10} {', '.join(info['os'])}")


# ── YAML Templates ────────────────────────────────────────────────────────────

HOME_NETWORK_TEMPLATE = """\
# =====================================================
# ARCA Custom Network — Home Network Template
# =====================================================
# Edit this file to describe your real network layout.
# Run: from arca.sim.custom_network import CustomNetworkBuilder
#      env = CustomNetworkBuilder.from_yaml("this_file.yaml")
# =====================================================

name: "My Home Network"
description: "Typical home network with router, laptops, IoT devices"
attacker_entry: "192.168.1.1"   # The host ARCA starts from (your entry point / "attacker foothold")

# Available OS types: Windows, Linux, macOS, IoT, Router, Android, "Windows Server"
# Available CVEs: EternalBlue, BlueKeep, Log4Shell, Shellshock, Heartbleed, IoTDefaultCreds,
#                 PwnKit, Spring4Shell, PrintNightmare, ZeroLogon, RouterDefaultPwd, SQLInjection

hosts:
  - name: "Home Router"
    os: Router
    subnet: 0
    ip: "192.168.1.1"
    services: [HTTP, HTTPS, SSH, Telnet]
    vulns: [RouterDefaultPwd]
    firewall: false
    data_value: 3.0
    notes: "ISP-provided router, default credentials likely unchanged"

  - name: "Dad's Laptop"
    os: Windows
    subnet: 1
    ip: "192.168.1.10"
    services: [SMB, RDP, HTTP]
    vulns: [EternalBlue, BlueKeep]
    is_critical: false
    data_value: 8.0
    notes: "Runs Windows 10, rarely updated"

  - name: "Mom's MacBook"
    os: macOS
    subnet: 1
    ip: "192.168.1.11"
    services: [SSH, HTTP, AFP]
    vulns: [TCC_Bypass]
    data_value: 6.0

  - name: "Smart TV"
    os: IoT
    subnet: 2
    ip: "192.168.1.30"
    services: [HTTP, MQTT]
    vulns: [IoTDefaultCreds]
    data_value: 2.0

  - name: "NAS Server"
    os: Linux
    subnet: 1
    ip: "192.168.1.20"
    services: [SSH, HTTP, FTP, NFS]
    vulns: [Log4Shell, DirtyCOW]
    is_critical: true
    data_value: 15.0
    notes: "Family photos and important documents — HIGH VALUE TARGET"

  - name: "Smart Camera"
    os: IoT
    subnet: 2
    ip: "192.168.1.31"
    services: [HTTP, Telnet, RTSP]
    vulns: [IoTDefaultCreds, RouteRCE]
    data_value: 4.0

# Define which hosts can reach each other.
# Format: [host_id_a, host_id_b]  (bidirectional)
# host IDs are 0-indexed in the order listed above
connections:
  - [0, 1]   # Router <-> Dad's Laptop
  - [0, 2]   # Router <-> Mom's MacBook
  - [0, 3]   # Router <-> Smart TV
  - [0, 4]   # Router <-> NAS
  - [0, 5]   # Router <-> Smart Camera
  - [1, 4]   # Dad's Laptop <-> NAS (file sharing)
  - [2, 4]   # Mom's MacBook <-> NAS

subnet_names:
  0: "DMZ / Router"
  1: "Trusted LAN"
  2: "IoT VLAN"
"""

SMALL_OFFICE_TEMPLATE = """\
name: "Small Office Network"
description: "10-person startup office with shared server and cloud access"
attacker_entry: "10.0.0.1"

hosts:
  - name: "Edge Router"
    os: Router
    subnet: 0
    ip: "10.0.0.1"
    services: [HTTP, HTTPS, SSH]
    vulns: [RouterDefaultPwd, Cisco_CVE]
    firewall: true
    data_value: 5.0

  - name: "Web Server"
    os: Linux
    subnet: 0
    ip: "10.0.0.10"
    services: [HTTP, HTTPS, SSH]
    vulns: [Log4Shell, Spring4Shell, SQLInjection]
    data_value: 10.0
    notes: "Public-facing web server — likely initial attack surface"

  - name: "File Server"
    os: "Windows Server"
    subnet: 1
    ip: "10.0.1.5"
    services: [SMB, RDP, IIS]
    vulns: [EternalBlue, PrintNightmare, ZeroLogon]
    is_critical: true
    data_value: 20.0

  - name: "Dev Laptop"
    os: macOS
    subnet: 1
    ip: "10.0.1.20"
    services: [SSH, HTTP]
    vulns: [TCC_Bypass]
    data_value: 12.0

  - name: "HR Laptop"
    os: Windows
    subnet: 1
    ip: "10.0.1.21"
    services: [SMB, RDP]
    vulns: [BlueKeep, EternalBlue]
    is_critical: true
    data_value: 18.0
    notes: "Contains sensitive employee records"

  - name: "Printer"
    os: IoT
    subnet: 2
    ip: "10.0.2.1"
    services: [HTTP, SNMP]
    vulns: [IoTDefaultCreds]
    data_value: 2.0

connections:
  - [0, 1]   # Router <-> Web Server
  - [0, 2]   # Router <-> File Server
  - [1, 2]   # Web Server <-> File Server
  - [2, 3]   # File Server <-> Dev Laptop
  - [2, 4]   # File Server <-> HR Laptop
  - [0, 5]   # Router <-> Printer
  - [2, 5]   # File Server <-> Printer

subnet_names:
  0: "DMZ"
  1: "Internal LAN"
  2: "Peripherals"
"""

DATACENTER_TEMPLATE = """\
name: "Mini Datacenter"
description: "Small datacenter with web tier, app tier, DB tier"
attacker_entry: "172.16.0.1"

hosts:
  - name: "Firewall"
    os: Router
    subnet: 0
    ip: "172.16.0.1"
    services: [SSH, HTTP]
    vulns: [RouterDefaultPwd]
    firewall: true
    data_value: 5.0

  - name: "Load Balancer"
    os: Linux
    subnet: 0
    ip: "172.16.0.2"
    services: [HTTP, HTTPS]
    vulns: [Shellshock]
    data_value: 8.0

  - name: "Web App 1"
    os: Linux
    subnet: 1
    ip: "172.16.1.10"
    services: [HTTP, SSH]
    vulns: [Log4Shell, RCE_WebApp]
    data_value: 10.0

  - name: "Web App 2"
    os: Linux
    subnet: 1
    ip: "172.16.1.11"
    services: [HTTP, SSH]
    vulns: [Spring4Shell]
    data_value: 10.0

  - name: "App Server"
    os: Linux
    subnet: 2
    ip: "172.16.2.5"
    services: [SSH, HTTP, PostgreSQL]
    vulns: [PwnKit, DirtyCOW]
    is_critical: true
    data_value: 15.0

  - name: "Primary DB"
    os: Linux
    subnet: 3
    ip: "172.16.3.1"
    services: [PostgreSQL, SSH]
    vulns: [Heartbleed, SQLInjection]
    is_critical: true
    data_value: 25.0
    notes: "Crown jewel — production database"

  - name: "Backup DB"
    os: Linux
    subnet: 3
    ip: "172.16.3.2"
    services: [PostgreSQL, SSH]
    vulns: [Heartbleed]
    is_critical: true
    data_value: 20.0

connections:
  - [0, 1]   # Firewall <-> LB
  - [1, 2]   # LB <-> Web1
  - [1, 3]   # LB <-> Web2
  - [2, 4]   # Web1 <-> App
  - [3, 4]   # Web2 <-> App
  - [4, 5]   # App <-> Primary DB
  - [5, 6]   # Primary DB <-> Backup DB

subnet_names:
  0: "Edge"
  1: "Web Tier"
  2: "App Tier"
  3: "DB Tier"
"""