"""
arca.agents.langgraph_orchestrator
===================================
Improved LangGraph multi-agent system with:
  - Proper multi-provider LLM (Ollama / Groq / Anthropic / OpenAI / rule-based)
  - Rich, context-aware prompts using full host details
  - 5-node graph: Analyst → Attacker → Critic → Reflector → Planner
  - Remediation node: generates defender's patch recommendations
  - Persistent memory across episodes
  - Structured output with severity scoring
"""

from __future__ import annotations

import json
from typing import Any, TypedDict, Optional

from arca.core.config import ARCAConfig


# ── Graph state ───────────────────────────────────────────────────────────────

class ARCAGraphState(TypedDict):
    network_state: dict
    analyst_output: str
    attacker_output: str
    critic_output: str
    reflection: str
    plan: str
    remediation: str
    severity_score: float       # 0.0–10.0 CVSS-like overall risk
    episode_history: list[dict]


# ── Orchestrator ──────────────────────────────────────────────────────────────

class ARCAOrchestrator:
    """
    LangGraph multi-agent orchestrator.

    Nodes:
      analyst    — describes current network state in plain English
      attacker   — suggests optimal next exploit moves
      critic     — evaluates agent efficiency, finds weaknesses
      reflector  — extracts lessons for future episodes
      planner    — generates structured action plan
      remediator — provides defender patch recommendations (bonus node)

    LLM providers auto-detected in order:
      Ollama → Groq → Anthropic → OpenAI → Rule-based fallback
    """

    def __init__(
        self,
        cfg: Optional[ARCAConfig] = None,
        provider: Optional[str] = None,    # "auto" | "ollama" | "groq" | "anthropic" | "openai" | "rule"
        model: Optional[str] = None,
    ):
        self.cfg = cfg or ARCAConfig()
        self._memory: list[dict] = []
        self._graph = None

        # Init LLM provider
        from arca.llm.providers import auto_detect_provider
        self._provider = auto_detect_provider(
            preferred=provider or self.cfg.llm.provider or "auto",
            model=model or self.cfg.llm.model or None,
        )
        self._llm_available = not self._provider.name.startswith("rule")
        print(f"[ARCA] LLM provider: {self._provider.name} "
              f"({'✓ connected' if self._llm_available else '⚠ rule-based fallback'})")

        self._build_graph()

    # ── Graph construction ────────────────────────────────────────────────────

    def _build_graph(self) -> None:
        try:
            from langgraph.graph import StateGraph, END

            graph = StateGraph(ARCAGraphState)
            graph.add_node("analyst",    self._analyst_node)
            graph.add_node("attacker",   self._attacker_node)
            graph.add_node("critic",     self._critic_node)
            graph.add_node("reflector",  self._reflector_node)
            graph.add_node("planner",    self._planner_node)
            graph.add_node("remediator", self._remediator_node)

            graph.set_entry_point("analyst")
            graph.add_edge("analyst",   "attacker")
            graph.add_edge("attacker",  "critic")
            graph.add_edge("critic",    "reflector")
            graph.add_edge("reflector", "planner")
            graph.add_edge("planner",   "remediator")
            graph.add_edge("remediator", END)

            self._graph = graph.compile()
        except Exception as e:
            print(f"[ARCA] LangGraph build failed ({e}). Sequential fallback active.")
            self._graph = None

    # ── Node implementations ──────────────────────────────────────────────────

    def _analyst_node(self, state: ARCAGraphState) -> ARCAGraphState:
        ns = state["network_state"]
        hosts = ns.get("hosts", {})
        ep = ns.get("episode_info", {})

        compromised = [
            f"Host {hid} ({h.get('name', h.get('ip', hid))}, {h.get('os')})"
            for hid, h in hosts.items() if h.get("status") == "COMPROMISED"
        ]
        discovered = [
            f"Host {hid} ({h.get('name', h.get('ip', hid))}, {h.get('os')}, vulns: {h.get('vulnerabilities', [])})"
            for hid, h in hosts.items() if h.get("discovered") and h.get("status") != "COMPROMISED"
        ]
        undiscovered_count = sum(1 for h in hosts.values() if not h.get("discovered"))

        prompt = f"""You are a senior cybersecurity analyst reviewing a penetration test in progress.

NETWORK: {ns.get('network_name', 'Unknown')}
STEP: {ns.get('step', 0)}
ATTACKER POSITION: Host {ns.get('attacker_node', 0)} — {hosts.get(str(ns.get('attacker_node', 0)), {}).get('ip', 'unknown')}

COMPROMISED HOSTS ({len(compromised)}):
{chr(10).join(f'  ✓ {c}' for c in compromised) or '  None yet'}

DISCOVERED (not yet compromised) ({len(discovered)}):
{chr(10).join(f'  ~ {d}' for d in discovered[:5]) or '  None yet'}

UNDISCOVERED: {undiscovered_count} hosts
TOTAL REWARD: {ep.get('total_reward', 0):.1f}
ATTACK PATH: {' → '.join(ep.get('attack_path', [])) or 'None yet'}

In 3 sentences: describe the current attack situation, what the attacker controls, and what the most valuable remaining targets are."""

        output = self._llm_call(prompt) or self._rule_analyst(ns)
        state["analyst_output"] = output

        # Compute severity score (0-10)
        total = max(len(hosts), 1)
        comp = len(compromised)
        crit_comp = sum(1 for h in hosts.values()
                        if h.get("status") == "COMPROMISED" and h.get("is_critical"))
        state["severity_score"] = min(10.0, round(
            (comp / total) * 5.0 + crit_comp * 2.5 + (1 if len(ep.get("attack_path", [])) > 0 else 0), 1
        ))
        return state

    def _attacker_node(self, state: ARCAGraphState) -> ARCAGraphState:
        ns = state["network_state"]
        hosts = ns.get("hosts", {})

        # Build rich host table
        host_lines = []
        for hid, h in hosts.items():
            vulns = h.get("vulnerabilities", [])
            vuln_str = ", ".join(f"{v if isinstance(v, str) else v.get('name','?')}({v.get('exploit_prob',0.5)*100:.0f}%)" if isinstance(v, dict) else v for v in vulns[:3])
            host_lines.append(
                f"  [{hid}] {h.get('ip','?'):<16} {h.get('os','?'):<10} "
                f"{'COMPROMISED' if h.get('status')=='COMPROMISED' else 'DISCOVERED' if h.get('discovered') else 'UNKNOWN':<12} "
                f"critical={h.get('is_critical',False)} firewall={h.get('firewall',False)} "
                f"vulns=[{vuln_str}]"
            )

        prompt = f"""You are a penetration tester AI planning the next move.

{state.get('analyst_output', '')}

DETAILED HOST STATUS:
{chr(10).join(host_lines)}

CURRENT POSITION: Host {ns.get('attacker_node', 0)}

As the attacker, identify:
1. The single best next exploit (highest probability, most impactful)
2. Best pivot path to reach critical/high-value hosts
3. Any quick wins (firewall=false + high exploit_prob)

Be specific: name the target host ID, the CVE/vuln, and why it's the best move."""

        output = self._llm_call(prompt) or self._rule_attacker(ns)
        state["attacker_output"] = output
        return state

    def _critic_node(self, state: ARCAGraphState) -> ARCAGraphState:
        ns = state["network_state"]
        ep = ns.get("episode_info", {})
        hosts = ns.get("hosts", {})

        comp = ep.get("hosts_compromised", 0)
        disc = ep.get("hosts_discovered", 0)
        total = len(hosts)
        steps = ns.get("step", 1)
        reward = ep.get("total_reward", 0)

        efficiency = comp / max(steps, 1) * 100
        discovery_ratio = disc / max(total, 1)
        exploit_ratio = comp / max(disc, 1) if disc > 0 else 0

        prompt = f"""You are a red team exercise evaluator critiquing an RL agent's performance.

PERFORMANCE METRICS:
  Steps taken: {steps}
  Hosts discovered: {disc}/{total} ({discovery_ratio*100:.0f}%)
  Hosts compromised: {comp}/{total} ({comp/total*100:.0f}%)
  Exploit success rate: {exploit_ratio*100:.0f}% (of discovered)
  Actions per compromise: {steps/max(comp,1):.1f}
  Total reward: {reward:.1f}
  Efficiency score: {efficiency:.2f} compromises/100 steps

ANALYST ASSESSMENT: {state.get('analyst_output', '')[:200]}
ATTACKER PLAN: {state.get('attacker_output', '')[:200]}

Critique in 3 points:
1. What is the agent doing wrong / inefficiently?
2. What opportunities is it missing?
3. What is the risk level for the defender? (Low/Medium/High/Critical)"""

        output = self._llm_call(prompt) or self._rule_critic(ns)
        state["critic_output"] = output
        return state

    def _reflector_node(self, state: ARCAGraphState) -> ARCAGraphState:
        history = state.get("episode_history", [])
        recent = history[-3:] if history else []
        hist_str = "\n".join(
            f"  Episode {i+1}: comp={h.get('compromised',0)}, steps={h.get('steps',0)}, reward={h.get('reward',0):.1f}"
            for i, h in enumerate(recent)
        ) if recent else "  No previous episodes"

        prompt = f"""You are an RL training coach analyzing an agent's learning progress.

CURRENT EPISODE CRITIQUE:
{state.get('critic_output', '')[:300]}

RECENT EPISODE HISTORY:
{hist_str}

SEVERITY SCORE: {state.get('severity_score', 0)}/10

Provide 2 key lessons:
1. What behavioral pattern should the agent reinforce?
2. What behavior should it avoid in future episodes?

Be specific and actionable (max 2 sentences each)."""

        output = self._llm_call(prompt) or "1. Scan before exploit. 2. Prioritize critical hosts."
        state["reflection"] = output
        return state

    def _planner_node(self, state: ARCAGraphState) -> ARCAGraphState:
        ns = state["network_state"]
        hosts = ns.get("hosts", {})

        # Find best targets
        undiscovered = [hid for hid, h in hosts.items() if not h.get("discovered")]
        exploitable = [
            (hid, h) for hid, h in hosts.items()
            if h.get("discovered") and h.get("status") != "COMPROMISED"
        ]
        # Sort exploitable by highest exploit prob
        if exploitable:
            def best_prob(item):
                vulns = item[1].get("vulnerabilities", [])
                return max((v.get("exploit_prob", 0) if isinstance(v, dict) else 0.5
                            for v in vulns), default=0)
            exploitable.sort(key=best_prob, reverse=True)

        critical_unowned = [
            hid for hid, h in hosts.items()
            if h.get("is_critical") and h.get("status") != "COMPROMISED"
        ]

        prompt = f"""You are a penetration testing planner creating a concrete action sequence.

REFLECTION: {state.get('reflection', '')[:200]}
ATTACKER INSIGHTS: {state.get('attacker_output', '')[:200]}

AVAILABLE TARGETS:
  Undiscovered hosts: {undiscovered[:5]}
  Exploitable (discovered): {[h for h, _ in exploitable[:5]]}
  Critical targets not yet owned: {critical_unowned}

Generate a prioritized action plan with exactly 5 steps.
Format each step as:
  STEP N: [ACTION] on Host [ID] — [reason and expected outcome]

Actions: SCAN | EXPLOIT | PIVOT | EXFILTRATE"""

        output = self._llm_call(prompt) or self._rule_plan(ns)
        state["plan"] = output
        return state

    def _remediator_node(self, state: ARCAGraphState) -> ARCAGraphState:
        ns = state["network_state"]
        hosts = ns.get("hosts", {})
        ep = ns.get("episode_info", {})

        # Gather all exploited vulnerabilities
        attack_path = ep.get("attack_path", [])
        compromised_details = [
            f"  - {h.get('ip','?')} ({h.get('os','?')}): vulns={h.get('vulnerabilities',[])}"
            for hid, h in hosts.items() if h.get("status") == "COMPROMISED"
        ]

        prompt = f"""You are a defensive security engineer (blue team) analyzing a completed penetration test.

ATTACK PATH: {' → '.join(attack_path) or 'No successful exploits'}
SEVERITY SCORE: {state.get('severity_score', 0)}/10
COMPROMISED HOSTS:
{chr(10).join(compromised_details) or '  None'}

Generate a prioritized remediation report with:
1. CRITICAL (fix within 24h): Patch or mitigate most dangerous vulnerabilities
2. HIGH (fix within 1 week): Network segmentation and access controls
3. MEDIUM (fix within 1 month): Monitoring, logging, credential hygiene
4. QUICK WINS: Immediate low-effort improvements (firewall rules, default creds)

Be specific — name CVEs, suggest exact patches or config changes."""

        output = self._llm_call(prompt) or self._rule_remediation(ns)
        state["remediation"] = output
        return state

    # ── Public interface ──────────────────────────────────────────────────────

    def step(self, network_state: dict) -> dict:
        """Run full analysis graph on current network state."""
        initial_state: ARCAGraphState = {
            "network_state": network_state,
            "analyst_output": "",
            "attacker_output": "",
            "critic_output": "",
            "reflection": "",
            "plan": "",
            "remediation": "",
            "severity_score": 0.0,
            "episode_history": self._memory[-5:],
        }

        if self._graph:
            result = self._graph.invoke(initial_state)
        else:
            # Sequential fallback
            result = initial_state.copy()
            for node in [
                self._analyst_node, self._attacker_node, self._critic_node,
                self._reflector_node, self._planner_node, self._remediator_node,
            ]:
                result = node(result)

        # Store in memory
        ep = network_state.get("episode_info", {})
        self._memory.append({
            "step": network_state.get("step"),
            "compromised": ep.get("hosts_compromised", 0),
            "steps": network_state.get("step", 0),
            "reward": ep.get("total_reward", 0),
            "severity": result.get("severity_score", 0),
            "reflection": result.get("reflection", ""),
        })

        return result

    def reflect(self, state: dict) -> dict:
        return self.step(state)

    def get_memory(self) -> list[dict]:
        return self._memory

    def get_provider_name(self) -> str:
        return self._provider.name

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _llm_call(self, prompt: str) -> Optional[str]:
        try:
            result = self._provider.complete(prompt, max_tokens=self.cfg.llm.max_tokens)
            if result and len(result) > 10:
                return result
        except Exception as e:
            print(f"[ARCA] LLM call failed: {e}. Using rule-based.")
        return None

    def _rule_analyst(self, ns: dict) -> str:
        hosts = ns.get("hosts", {})
        ep = ns.get("episode_info", {})
        comp = ep.get("hosts_compromised", 0)
        disc = ep.get("hosts_discovered", 0)
        total = len(hosts)
        critical_owned = sum(1 for h in hosts.values()
                             if h.get("status") == "COMPROMISED" and h.get("is_critical"))
        return (
            f"Agent controls {comp}/{total} hosts ({disc} discovered) "
            f"from position Host {ns.get('attacker_node', 0)}. "
            f"Critical hosts owned: {critical_owned}. "
            f"Progress: {comp/max(total,1)*100:.0f}% network penetration."
        )

    def _rule_attacker(self, ns: dict) -> str:
        hosts = ns.get("hosts", {})
        best_target = None
        best_prob = 0
        for hid, h in hosts.items():
            if h.get("discovered") and h.get("status") != "COMPROMISED":
                for v in h.get("vulnerabilities", []):
                    p = v.get("exploit_prob", 0) if isinstance(v, dict) else 0.5
                    if p > best_prob and not h.get("firewall", False):
                        best_prob = p
                        best_target = (hid, h, v)
        if best_target:
            hid, h, v = best_target
            return f"Best target: Host {hid} ({h.get('ip')}) via {v.get('name','?')} ({best_prob*100:.0f}% success rate)."
        return "All reachable hosts discovered. Focus on pivoting to new subnets."

    def _rule_critic(self, ns: dict) -> str:
        ep = ns.get("episode_info", {})
        comp = ep.get("hosts_compromised", 0)
        disc = ep.get("hosts_discovered", 0)
        total = len(ns.get("hosts", {}))
        if disc == 0:
            return "1. No hosts discovered. 2. Agent not scanning. 3. Risk: Low (no penetration)."
        ratio = comp / max(disc, 1)
        if ratio < 0.3:
            return (
                "1. Low exploit success — agent discovering but not compromising. "
                "2. Missing high-probability vulns. "
                "3. Risk: Medium."
            )
        return (
            f"1. Reasonable progress ({comp}/{total} hosts). "
            "2. Should target critical hosts next. "
            "3. Risk: High — attacker has significant foothold."
        )

    def _rule_plan(self, ns: dict) -> str:
        hosts = ns.get("hosts", {})
        undiscovered = [hid for hid, h in hosts.items() if not h.get("discovered")]
        exploitable = [hid for hid, h in hosts.items()
                       if h.get("discovered") and h.get("status") != "COMPROMISED"]
        critical = [hid for hid, h in hosts.items()
                    if h.get("is_critical") and h.get("status") != "COMPROMISED"]
        steps = []
        if undiscovered:
            steps.append(f"STEP 1: SCAN Host {undiscovered[0]} — discover new hosts in network")
        if exploitable:
            steps.append(f"STEP 2: EXPLOIT Host {exploitable[0]} — compromise discovered host")
        if critical:
            steps.append(f"STEP 3: EXPLOIT Host {critical[0]} (CRITICAL) — high-value target")
        if exploitable:
            steps.append(f"STEP 4: PIVOT to compromised host — expand attack surface")
        steps.append("STEP 5: EXFILTRATE from highest data_value compromised host")
        return "\n".join(steps) if steps else "STEP 1: SCAN all reachable hosts"

    def _rule_remediation(self, ns: dict) -> str:
        hosts = ns.get("hosts", {})
        compromised_vulns = []
        for h in hosts.values():
            if h.get("status") == "COMPROMISED":
                for v in h.get("vulnerabilities", []):
                    if isinstance(v, dict):
                        compromised_vulns.append(v.get("name", "unknown"))

        vuln_str = ", ".join(set(compromised_vulns[:5])) if compromised_vulns else "None identified"
        return (
            f"CRITICAL (24h): Patch {vuln_str}. Apply emergency security updates.\n"
            "HIGH (1 week): Enable host firewall on all endpoints. Segment IoT to separate VLAN.\n"
            "MEDIUM (1 month): Deploy SIEM, enable audit logging, enforce MFA.\n"
            "QUICK WINS: Change all default credentials, disable Telnet/unused services."
        )