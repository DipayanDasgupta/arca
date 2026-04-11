"""arca.sim.network_generator — Procedural network topology generator."""

from __future__ import annotations

import random
from typing import Tuple

import networkx as nx

from arca.sim.host import Host, HostStatus
from arca.core.config import EnvConfig

# Vulnerability DB (simplified)
VULN_DB = [
    {"name": "EternalBlue", "cve": "CVE-2017-0144", "exploit_prob": 0.75, "os": "Windows"},
    {"name": "Log4Shell", "cve": "CVE-2021-44228", "exploit_prob": 0.8, "os": "Linux"},
    {"name": "ProxyLogon", "cve": "CVE-2021-26855", "exploit_prob": 0.7, "os": "Windows"},
    {"name": "Shellshock", "cve": "CVE-2014-6271", "exploit_prob": 0.65, "os": "Linux"},
    {"name": "Heartbleed", "cve": "CVE-2014-0160", "exploit_prob": 0.6, "os": "Linux"},
    {"name": "BlueKeep", "cve": "CVE-2019-0708", "exploit_prob": 0.7, "os": "Windows"},
    {"name": "PrintNightmare", "cve": "CVE-2021-34527", "exploit_prob": 0.65, "os": "Windows"},
    {"name": "Dirty COW", "cve": "CVE-2016-5195", "exploit_prob": 0.55, "os": "Linux"},
    {"name": "IoT Default Creds", "cve": "CVE-2020-8958", "exploit_prob": 0.9, "os": "IoT"},
    {"name": "macOS TCC Bypass", "cve": "CVE-2023-41990", "exploit_prob": 0.5, "os": "macOS"},
]

OS_BY_SUBNET = {
    0: ["Windows", "Linux"],  # DMZ
    1: ["Windows", "Linux", "macOS"],  # Corp
    2: ["Linux", "IoT"],  # OT/IoT
    3: ["Windows"],  # AD
    4: ["Linux"],  # Servers
}

SERVICES = {
    "Windows": ["SMB", "RDP", "WinRM", "IIS", "MSSQL"],
    "Linux": ["SSH", "HTTP", "HTTPS", "FTP", "NFS", "PostgreSQL"],
    "macOS": ["SSH", "HTTP", "AFP"],
    "IoT": ["Telnet", "HTTP", "MQTT"],
}


class NetworkGenerator:
    def __init__(self, cfg: EnvConfig, rng: random.Random):
        self.cfg = cfg
        self.rng = rng
        self.attacker_node: int = 0

    def generate(self) -> Tuple[nx.DiGraph, dict[int, Host]]:
        n = self.cfg.num_hosts
        s = self.cfg.num_subnets
        g = nx.DiGraph()
        hosts: dict[int, Host] = {}

        # Create hosts
        for i in range(n):
            subnet = i % s
            os_choices = OS_BY_SUBNET.get(subnet % 5, ["Linux"])
            os = self.rng.choice(os_choices)
            svc_pool = SERVICES.get(os, ["SSH"])
            services = self.rng.sample(svc_pool, k=min(self.rng.randint(1, 3), len(svc_pool)))

            # Assign vulnerabilities
            vulns = []
            if self.rng.random() < self.cfg.vulnerability_density:
                os_vulns = [v for v in VULN_DB if v["os"] == os or v["os"] == "Linux"]
                n_vulns = self.rng.randint(1, min(3, len(os_vulns)))
                vulns = self.rng.sample(os_vulns, k=n_vulns)

            ip = f"10.{subnet}.{self.rng.randint(1, 254)}.{i+1}"
            hosts[i] = Host(
                id=i,
                subnet=subnet,
                os=os,
                ip=ip,
                services=services,
                vulnerabilities=vulns,
                data_value=self.rng.uniform(1.0, 15.0),
                is_critical=(i == n - 1),  # last host is the crown jewel
                firewall=(subnet == 0),
            )
            g.add_node(i, **hosts[i].to_dict())

        # Create edges (reachability)
        # Within-subnet: full mesh
        for a in range(n):
            for b in range(n):
                if a != b and hosts[a].subnet == hosts[b].subnet:
                    g.add_edge(a, b)

        # Cross-subnet: limited edges (gateway links)
        for i in range(n):
            for j in range(n):
                if hosts[i].subnet != hosts[j].subnet:
                    if abs(hosts[i].subnet - hosts[j].subnet) == 1:
                        if self.rng.random() < 0.3:
                            g.add_edge(i, j)

        # Attacker starts at a random host in subnet 0 (internet-facing)
        subnet0_hosts = [i for i, h in hosts.items() if h.subnet == 0]
        self.attacker_node = self.rng.choice(subnet0_hosts) if subnet0_hosts else 0

        return g, hosts