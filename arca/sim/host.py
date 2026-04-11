"""arca.sim.host — Host node in the network simulation."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class HostStatus(IntEnum):
    UNKNOWN = 0
    DISCOVERED = 1
    COMPROMISED = 2


@dataclass
class Host:
    id: int
    subnet: int
    os: str
    ip: str
    services: list[str] = field(default_factory=list)
    vulnerabilities: list[dict] = field(default_factory=list)
    status: HostStatus = HostStatus.UNKNOWN
    discovered: bool = False
    data_value: float = 0.0
    is_critical: bool = False
    firewall: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "subnet": self.subnet,
            "os": self.os,
            "ip": self.ip,
            "services": self.services,
            "vulnerabilities": [v.get("name", "?") for v in self.vulnerabilities],
            "status": self.status.name,
            "discovered": self.discovered,
            "data_value": round(self.data_value, 2),
            "is_critical": self.is_critical,
        }