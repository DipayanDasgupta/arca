"""arca.sim.action — Action and ActionResult types."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class ActionType(IntEnum):
    SCAN = 0
    EXPLOIT = 1
    PIVOT = 2
    EXFILTRATE = 3


@dataclass
class Action:
    action_type: ActionType
    target_host: int
    exploit_id: int
    source_host: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.action_type.name,
            "source": self.source_host,
            "target": self.target_host,
            "exploit_id": self.exploit_id,
        }

    def __str__(self) -> str:
        return f"{self.action_type.name}({self.source_host}→{self.target_host}, exploit={self.exploit_id})"


@dataclass
class ActionResult:
    success: bool
    message: str = ""
    discovered_hosts: list[int] = field(default_factory=list)
    compromised_host: int | None = None
    data_exfiltrated: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "message": self.message,
            "discovered_hosts": self.discovered_hosts,
            "compromised_host": self.compromised_host,
            "data_exfiltrated": round(self.data_exfiltrated, 2),
        }