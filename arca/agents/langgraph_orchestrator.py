"""
arca.agents.langgraph_orchestrator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
LangGraph-based multi-agent system for explainable RL decision-making.

Graph topology:
  ┌──────────────┐
  │  State Input │
  └──────┬───────┘
         ▼
  ┌──────────────┐    ┌────────────────┐
  │  Analyst     │───▶│  Critic        │
  │  (describes  │    │  (evaluates    │
  │   state)     │    │   RL action)   │
  └──────────────┘    └───────┬────────┘
                              ▼
                     ┌────────────────┐
                     │  Reflector     │
                     │  (learns from  │
                     │   mistakes)    │
                     └───────┬────────┘
                             ▼
                     ┌────────────────┐
                     │  Planner       │
                     │  (suggests     │
                     │   next steps)  │
                     └────────────────┘
"""

from __future__ import annotations

import json
from typing import Any, TypedDict, Optional

from arca.core.config import ARCAConfig


class ARCAGraphState(TypedDict):
    network_state: dict
    analyst_output: str
    critic_output: str
    reflection: str
    plan: str
    episode_history: list[dict]


class ARCAOrchestrator:
    """
    LangGraph-powered orchestrator with LLM nodes.
    Falls back to rule-based heuristics if LLM is unavailable.
    """

    def __init__(self, cfg: Optional[ARCAConfig] = None):
        self.cfg = cfg or ARCAConfig()
        self._llm = None
        self._graph = None
        self._memory: list[dict] = []
        self._llm_available = False

        if self.cfg.llm.enabled:
            self._init_llm()
            if self._llm_available:
                self._build_graph()

    def _init_llm(self) -> None:
        try:
            from langchain_community.llms import Ollama  # type: ignore
            self._llm = Ollama(
                model=self.cfg.llm.model,
                base_url=self.cfg.llm.base_url,
                temperature=self.cfg.llm.temperature,
            )
            # Quick connectivity test
            self._llm.invoke("ping")
            self._llm_available = True
            print(f"[ARCA] LLM connected: {self.cfg.llm.model} @ {self.cfg.llm.base_url}")
        except Exception as e:
            print(f"[ARCA] LLM unavailable ({e}). Using rule-based fallback.")
            self._llm_available = False

    def _build_graph(self) -> None:
        try:
            from langgraph.graph import StateGraph, END  # type: ignore

            graph = StateGraph(ARCAGraphState)
            graph.add_node("analyst", self._analyst_node)
            graph.add_node("critic", self._critic_node)
            graph.add_node("reflector", self._reflector_node)
            graph.add_node("planner", self._planner_node)

            graph.set_entry_point("analyst")
            graph.add_edge("analyst", "critic")
            graph.add_edge("critic", "reflector")
            graph.add_edge("reflector", "planner")
            graph.add_edge("planner", END)

            self._graph = graph.compile()
        except Exception as e:
            print(f"[ARCA] LangGraph build failed: {e}. Using sequential fallback.")
            self._graph = None

    # ------------------------------------------------------------------
    # Node implementations
    # ------------------------------------------------------------------

    def _analyst_node(self, state: ARCAGraphState) -> ARCAGraphState:
        ns = state["network_state"]
        prompt = f"""You are a cybersecurity analyst examining a network penetration test.

Current network state:
- Step: {ns.get('step', 0)}
- Attacker position: Host {ns.get('attacker_node', 0)}
- Hosts discovered: {sum(1 for h in ns.get('hosts', {}).values() if h.get('discovered'))}
- Hosts compromised: {sum(1 for h in ns.get('hosts', {}).values() if h.get('status') == 'COMPROMISED')}
- Total hosts: {len(ns.get('hosts', {}))}
- Attack path so far: {ns.get('episode_info', {}).get('attack_path', [])}

Briefly describe the current attack situation in 2-3 sentences."""

        output = self._invoke_llm(prompt) or self._rule_based_analyst(ns)
        state["analyst_output"] = output
        return state

    def _critic_node(self, state: ARCAGraphState) -> ARCAGraphState:
        ns = state["network_state"]
        analyst = state.get("analyst_output", "")
        prompt = f"""You are a cybersecurity critic reviewing an RL agent's decisions.

Situation: {analyst}

Recent reward: {ns.get('episode_info', {}).get('total_reward', 0):.2f}

Critically evaluate:
1. Is the agent making good use of discovered hosts?
2. Is it being efficient in its attack path?
3. What mistakes is it making?

Be concise (2-3 sentences)."""

        output = self._invoke_llm(prompt) or self._rule_based_critic(ns)
        state["critic_output"] = output
        return state

    def _reflector_node(self, state: ARCAGraphState) -> ARCAGraphState:
        critic = state.get("critic_output", "")
        history = state.get("episode_history", [])
        prompt = f"""You are a reflection module for an RL agent learning pentesting.

Critic feedback: {critic}
Past reflections: {history[-3:] if history else 'None'}

What should the agent learn from this? What patterns should it avoid or repeat?
Be specific and actionable (1-2 sentences)."""

        output = self._invoke_llm(prompt) or "Agent should prioritize scanning before exploiting."
        state["reflection"] = output
        return state

    def _planner_node(self, state: ARCAGraphState) -> ARCAGraphState:
        ns = state["network_state"]
        reflection = state.get("reflection", "")
        prompt = f"""You are a penetration testing planner.

Reflection: {reflection}
Current position: Host {ns.get('attacker_node', 0)}
Compromised hosts: {[hid for hid, h in ns.get('hosts', {}).items() if h.get('status') == 'COMPROMISED']}

Suggest the next 3 high-priority actions for the RL agent (SCAN/EXPLOIT/PIVOT/EXFILTRATE).
Format: action: target_host reason"""

        output = self._invoke_llm(prompt) or self._rule_based_plan(ns)
        state["plan"] = output
        return state

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def step(self, network_state: dict) -> dict:
        initial_state: ARCAGraphState = {
            "network_state": network_state,
            "analyst_output": "",
            "critic_output": "",
            "reflection": "",
            "plan": "",
            "episode_history": self._memory[-5:],
        }

        if self._graph:
            result = self._graph.invoke(initial_state)
        else:
            # Sequential fallback
            result = self._analyst_node(initial_state)
            result = self._critic_node(result)
            result = self._reflector_node(result)
            result = self._planner_node(result)

        self._memory.append({
            "step": network_state.get("step"),
            "reflection": result.get("reflection"),
            "plan": result.get("plan"),
        })

        return result

    def reflect(self, state: dict) -> dict:
        return self.step(state)

    def get_memory(self) -> list[dict]:
        return self._memory

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _invoke_llm(self, prompt: str) -> Optional[str]:
        if not self._llm_available or self._llm is None:
            return None
        try:
            return self._llm.invoke(prompt)
        except Exception:
            return None

    def _rule_based_analyst(self, ns: dict) -> str:
        comp = sum(1 for h in ns.get("hosts", {}).values() if h.get("status") == "COMPROMISED")
        disc = sum(1 for h in ns.get("hosts", {}).values() if h.get("discovered"))
        total = len(ns.get("hosts", {}))
        return (
            f"Agent has compromised {comp}/{total} hosts and discovered {disc}/{total}. "
            f"Currently operating from host {ns.get('attacker_node', 0)}. "
            f"Progress: {comp/max(total,1)*100:.0f}% of objective achieved."
        )

    def _rule_based_critic(self, ns: dict) -> str:
        comp = sum(1 for h in ns.get("hosts", {}).values() if h.get("status") == "COMPROMISED")
        disc = sum(1 for h in ns.get("hosts", {}).values() if h.get("discovered"))
        if disc == 0:
            return "Agent should scan more aggressively to discover hosts before exploiting."
        ratio = comp / max(disc, 1)
        if ratio < 0.3:
            return "Low exploit ratio. Agent discovers hosts but fails to compromise them efficiently."
        return "Agent is making reasonable progress. Consider pivoting to reach deeper subnets."

    def _rule_based_plan(self, ns: dict) -> str:
        hosts = ns.get("hosts", {})
        undiscovered = [hid for hid, h in hosts.items() if not h.get("discovered")]
        exploitable = [
            hid for hid, h in hosts.items()
            if h.get("discovered") and h.get("status") != "COMPROMISED"
        ]
        plan_parts = []
        if undiscovered:
            plan_parts.append(f"SCAN: target host {undiscovered[0]} — discover new hosts")
        if exploitable:
            plan_parts.append(f"EXPLOIT: target host {exploitable[0]} — compromise vulnerable host")
        plan_parts.append("PIVOT: move to highest-value compromised host for better reach")
        return "\n".join(plan_parts)