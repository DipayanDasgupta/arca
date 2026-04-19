"""
arca/agents/langgraph_orchestrator.py  (v3.1 — FIXED)
=======================================================
Critical fix: self._llm was both an instance attribute (LocalLLM object)
AND a method name, causing `TypeError: 'LocalLLM' object is not callable`.

Renamed:
  self._local_llm_instance  — stores the LocalLLM object
  self._llm_call            — stores the bound callable (function pointer)
  self._call_llm()          — the internal dispatch helper method

Improvements:
  - Integrates EpisodeBuffer for persistent memory across runs
  - LLM prompts enriched with past episode statistics
  - LocalLLM → Ollama → Groq → rule-based priority chain
  - get_provider_name() returns a clean string
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict, Optional

from arca.core.config import ARCAConfig


class ARCAGraphState(TypedDict):
    network_state:    dict
    analyst_output:   str
    attacker_output:  str
    critic_output:    str
    reflection:       str
    plan:             str
    remediation:      str
    severity_score:   float       # 0.0–10.0
    episode_history:  list[dict]


class ARCAOrchestrator:
    """
    LangGraph multi-agent orchestrator (v3.1).

    LLM provider priority (local-first):
      1. LocalLLM  (llama-cpp-python, fully local, GPU offload)
      2. Ollama    (local server)
      3. Groq      (free API key)
      4. Rule-based fallback (always works, zero dependencies)
    """

    def __init__(
        self,
        cfg: Optional[ARCAConfig] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.cfg = cfg or ARCAConfig()
        self._memory: list[dict] = []
        self._graph = None

        # These are the correctly-named attributes (no name collision with methods)
        self._local_llm_instance = None   # LocalLLM object, if loaded
        self._llm_call = None             # Callable: (system, user) -> str
        self._provider_name = "rule-based"

        self._resolve_llm_provider(provider, model)
        self._build_graph()

        # Load persistent episode memory if available
        self._episode_buffer = None
        self._try_load_episode_buffer()

    # ── LLM Provider Resolution ───────────────────────────────────────────────

    def _resolve_llm_provider(
        self,
        provider_override: Optional[str],
        model_override: Optional[str],
    ) -> None:
        """Resolve LLM provider and store as self._llm_call callable."""
        # ── AFTER (fixed) ─────────────────────────────────────────────────────────────
        cfg = self.cfg.llm
        # provider_override lets CLI force a specific backend; otherwise we use the flag.
        preferred = provider_override or "auto"

# ── 1. LocalLLM (default, fully offline) ──────────────────────────────────────
# FIX: use_local_llm=True must always win, regardless of cfg.provider default.
#      The old guard checked `preferred not in ("ollama",...)` but cfg.provider
#      defaults to "ollama", so preferred was always "ollama" and LocalLLM
#      was silently skipped.
        if getattr(cfg, "use_local_llm", False) and preferred not in ("groq", "rule"):
            try:
                from arca.llm.local_llm import LocalLLM
                llm = LocalLLM(
                    model_key=model_override or getattr(cfg, "local_model_key", "llama-3.2-3b"),
                    model_dir=getattr(cfg, "local_model_dir", str(Path.home() / ".arca" / "models")),
                    n_gpu_layers=getattr(cfg, "local_n_gpu_layers", -1),
                    auto_download=getattr(cfg, "auto_download_model", False),
                )
                if llm.available:
                    self._local_llm_instance = llm   # NOTE: attribute, NOT method
                    self._provider_name = f"LocalLLM ({llm._filename})"
                    print(f"[ARCA Orchestrator] ✓ Provider: {self._provider_name}")

                    # Store the bound method as a plain callable
                    def _local_call(system: str, user: str, **kw) -> str:
                        return llm.chat(
                            system=system,
                            user=user,
                            max_tokens=getattr(cfg, "max_tokens", 512),
                            temperature=getattr(cfg, "temperature", 0.2),
                        )

                    self._llm_call = _local_call
                    return
                else:
                    print("[ARCA Orchestrator] LocalLLM not available (model file missing).")
            except Exception as e:
                print(f"[ARCA Orchestrator] LocalLLM init failed: {e}")

        # ── 2. Ollama (local server) ───────────────────────────────────────────
        if preferred in ("auto", "ollama"):
            try:
                import urllib.request
                urllib.request.urlopen(
                    getattr(cfg, "base_url", "http://localhost:11434") + "/api/tags",
                    timeout=1.5,
                )
                import ollama as _ollama
                self._provider_name = "Ollama"
                print("[ARCA Orchestrator] ✓ Provider: Ollama")
                _model = getattr(cfg, "model", "llama3.2")

                def _ollama_call(system: str, user: str, **kw) -> str:
                    resp = _ollama.chat(
                        model=_model,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        options={"temperature": getattr(cfg, "temperature", 0.2)},
                    )
                    return resp["message"]["content"]

                self._llm_call = _ollama_call
                return
            except Exception:
                pass

        # ── 3. Groq (remote, free API key) ────────────────────────────────────
        import os
        if os.getenv("GROQ_API_KEY") and preferred in ("auto", "groq"):
            try:
                from groq import Groq
                client = Groq()
                self._provider_name = "Groq"
                print("[ARCA Orchestrator] ✓ Provider: Groq")

                def _groq_call(system: str, user: str, **kw) -> str:
                    resp = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        max_tokens=getattr(cfg, "max_tokens", 512),
                        temperature=getattr(cfg, "temperature", 0.2),
                    )
                    return resp.choices[0].message.content

                self._llm_call = _groq_call
                return
            except Exception:
                pass

        # ── 4. Rule-based fallback ─────────────────────────────────────────────
        self._provider_name = "rule-based"
        self._llm_call = None
        print("[ARCA Orchestrator] Provider: rule-based fallback (no LLM available)")

    def _call_llm(self, system: str, user: str) -> Optional[str]:
        """
        Safe internal dispatch to whichever LLM provider is active.
        Returns None on failure or if no LLM is configured.
        """
        if self._llm_call is None:
            return None
        try:
            result = self._llm_call(system=system, user=user)
            if isinstance(result, str) and len(result.strip()) > 5:
                return result.strip()
            return None
        except Exception as e:
            print(f"[ARCA Orchestrator] LLM call failed: {e}")
            return None

    # ── Episode Memory ────────────────────────────────────────────────────────

    def _try_load_episode_buffer(self) -> None:
        try:
            from arca.memory.episode_buffer import EpisodeBuffer
            self._episode_buffer = EpisodeBuffer()
        except Exception:
            self._episode_buffer = None

    def _format_memory_for_prompt(self) -> str:
        """Return a compact past-episode summary for LLM context."""
        if self._episode_buffer and len(self._episode_buffer) > 0:
            return self._episode_buffer.format_for_llm(n=3)
        if self._memory:
            recent = self._memory[-3:]
            lines = [
                f"  Ep{i+1}: comp={m.get('compromised',0)} "
                f"steps={m.get('steps',0)} r={m.get('reward',0):.1f}"
                for i, m in enumerate(recent)
            ]
            return "\n".join(lines)
        return "  No prior episodes."

    # ── LangGraph Construction ─────────────────────────────────────────────────

    def _build_graph(self) -> None:
        try:
            from langgraph.graph import StateGraph, END

            g = StateGraph(ARCAGraphState)
            g.add_node("analyst",    self._analyst_node)
            g.add_node("attacker",   self._attacker_node)
            g.add_node("critic",     self._critic_node)
            g.add_node("reflector",  self._reflector_node)
            g.add_node("planner",    self._planner_node)
            g.add_node("remediator", self._remediator_node)

            g.set_entry_point("analyst")
            for src, dst in [
                ("analyst",    "attacker"),
                ("attacker",   "critic"),
                ("critic",     "reflector"),
                ("reflector",  "planner"),
                ("planner",    "remediator"),
                ("remediator", END),
            ]:
                g.add_edge(src, dst)

            self._graph = g.compile()
        except Exception as e:
            print(f"[ARCA Orchestrator] LangGraph compile failed: {e}. Using sequential mode.")
            self._graph = None

    # ── Node Implementations ──────────────────────────────────────────────────

    def _analyst_node(self, state: ARCAGraphState) -> ARCAGraphState:
        ns = state["network_state"]
        hosts = ns.get("hosts", {})
        ep = ns.get("episode_info", {})

        compromised = [
            f"H{hid}({h.get('os','?')})"
            for hid, h in hosts.items()
            if h.get("status") == "COMPROMISED"
        ]
        discovered = [
            f"H{hid}({h.get('os','?')})"
            for hid, h in hosts.items()
            if h.get("discovered") and h.get("status") != "COMPROMISED"
        ]
        crit_not_owned = [
            hid for hid, h in hosts.items()
            if h.get("is_critical") and h.get("status") != "COMPROMISED"
        ]

        system = (
            "You are a senior cybersecurity red-team analyst. "
            "Be concise and focus on actionable insights."
        )
        user = (
            f"Network: {ns.get('network_name', 'Unknown')} | Step: {ns.get('step', 0)}\n"
            f"Compromised ({len(compromised)}): {compromised}\n"
            f"Discovered not owned ({len(discovered)}): {discovered[:4]}\n"
            f"Critical unowned: {crit_not_owned}\n"
            f"Reward: {ep.get('total_reward', 0):.1f} | Path: {ep.get('attack_path', [])}\n"
            f"Past sessions:\n{self._format_memory_for_prompt()}\n"
            "In 2 sentences: describe the situation and the highest-priority target."
        )

        output = self._call_llm(system, user) or self._rule_analyst(ns)
        state["analyst_output"] = output

        # Severity score (0–10)
        total = max(len(hosts), 1)
        n_comp = len(compromised)
        n_crit = sum(1 for h in hosts.values()
                     if h.get("status") == "COMPROMISED" and h.get("is_critical"))
        state["severity_score"] = min(10.0, round(
            (n_comp / total) * 5.0 + n_crit * 2.5 +
            (1.0 if ep.get("attack_path") else 0.0), 1
        ))
        return state

    def _attacker_node(self, state: ARCAGraphState) -> ARCAGraphState:
        ns = state["network_state"]
        hosts = ns.get("hosts", {})

        host_lines = []
        for hid, h in hosts.items():
            vulns = ", ".join(
                f"{v.get('name','?')}({v.get('exploit_prob',0.5)*100:.0f}%)"
                if isinstance(v, dict) else str(v)
                for v in h.get("vulnerabilities", [])[:3]
            )
            status = "COMP" if h.get("status") == "COMPROMISED" else (
                "DISC" if h.get("discovered") else "?"
            )
            host_lines.append(
                f"  [{hid}] {h.get('os','?'):<10} {status:<5} "
                f"crit={h.get('is_critical',False)} fw={h.get('firewall',False)} "
                f"vulns=[{vulns}]"
            )

        system = (
            "You are an elite penetration tester AI. "
            "Recommend precise, stealthy actions — avoid noisy scans of already-known hosts."
        )
        user = (
            f"{state.get('analyst_output', '')}\n"
            f"Attacker position: Host {ns.get('attacker_node', 0)}\n"
            "HOST TABLE:\n" + "\n".join(host_lines) + "\n\n"
            "Recommend ONE action: target host ID, action type "
            "(SCAN/EXPLOIT/PIVOT/EXFILTRATE), vulnerability name, and why. "
            "Prefer high-probability exploits on critical/high-value hosts."
        )

        output = self._call_llm(system, user) or self._rule_attacker(ns)
        state["attacker_output"] = output
        return state

    def _critic_node(self, state: ARCAGraphState) -> ARCAGraphState:
        ns = state["network_state"]
        ep = ns.get("episode_info", {})
        hosts = ns.get("hosts", {})
        total = max(len(hosts), 1)
        comp = ep.get("hosts_compromised", 0)
        disc = ep.get("hosts_discovered", 0)
        steps = max(ns.get("step", 1), 1)
        reward = ep.get("total_reward", 0)

        system = "You are a strict red-team performance evaluator."
        user = (
            f"Compromised: {comp}/{total} | Discovered: {disc}/{total} | Steps: {steps}\n"
            f"Efficiency: {comp / steps * 100:.1f} compromises/100steps | Reward: {reward:.1f}\n"
            f"Analyst: {state.get('analyst_output', '')[:180]}\n"
            f"Attacker plan: {state.get('attacker_output', '')[:180]}\n"
            "3 bullet critique:\n"
            "• Main inefficiency\n• Best missed opportunity\n• Defender risk level (Low/Medium/High/Critical)"
        )

        output = self._call_llm(system, user) or self._rule_critic(ns)
        state["critic_output"] = output
        return state

    def _reflector_node(self, state: ARCAGraphState) -> ARCAGraphState:
        history = state.get("episode_history", [])[-3:]
        hist_str = "\n".join(
            f"  Ep{i+1}: comp={h.get('compromised',0)} "
            f"steps={h.get('steps',0)} r={h.get('reward',0):.1f}"
            for i, h in enumerate(history)
        ) or "  No prior episodes in this session."

        system = "You are an RL training coach for a cybersecurity agent."
        user = (
            f"Performance critique:\n{state.get('critic_output','')[:280]}\n\n"
            f"Session history:\n{hist_str}\n\n"
            f"All-time memory:\n{self._format_memory_for_prompt()}\n\n"
            "Provide exactly 2 lessons (1 sentence each):\n"
            "Lesson 1: What behaviour to reinforce (e.g., 'prioritise X').\n"
            "Lesson 2: What behaviour to avoid (e.g., 'stop doing Y')."
        )

        output = self._call_llm(system, user) or (
            "Lesson 1: Prioritise exploiting critical hosts early for high rewards. "
            "Lesson 2: Avoid scanning already-discovered hosts — exploit them instead."
        )
        state["reflection"] = output
        return state

    def _planner_node(self, state: ARCAGraphState) -> ARCAGraphState:
        ns = state["network_state"]
        hosts = ns.get("hosts", {})

        undiscovered = [hid for hid, h in hosts.items() if not h.get("discovered")]
        exploitable = sorted(
            [(hid, h) for hid, h in hosts.items()
             if h.get("discovered") and h.get("status") != "COMPROMISED"],
            key=lambda x: max(
                (v.get("exploit_prob", 0) if isinstance(v, dict) else 0.5
                 for v in x[1].get("vulnerabilities", [])),
                default=0.0
            ),
            reverse=True,
        )
        critical_unowned = [
            hid for hid, h in hosts.items()
            if h.get("is_critical") and h.get("status") != "COMPROMISED"
        ]

        system = "You are a precision penetration testing planner."
        user = (
            f"Reflection: {state.get('reflection','')[:200]}\n"
            f"Attacker insight: {state.get('attacker_output','')[:200]}\n\n"
            f"Undiscovered: {undiscovered[:4]}\n"
            f"Exploitable (sorted by prob): {[h for h, _ in exploitable[:4]]}\n"
            f"Critical unowned: {critical_unowned}\n\n"
            "Generate EXACTLY 5 steps:\nSTEP N: [ACTION] on Host [ID] — [brief reason]"
        )

        output = self._call_llm(system, user) or self._rule_plan(ns)
        state["plan"] = output
        return state

    def _remediator_node(self, state: ARCAGraphState) -> ARCAGraphState:
        ns = state["network_state"]
        hosts = ns.get("hosts", {})
        ep = ns.get("episode_info", {})

        exploited_vulns = sorted({
            v.get("name", "?")
            for h in hosts.values()
            if h.get("status") == "COMPROMISED"
            for v in h.get("vulnerabilities", [])
            if isinstance(v, dict)
        })

        system = "You are a senior defensive security engineer (blue team)."
        user = (
            f"Attack path: {ep.get('attack_path', [])}\n"
            f"Severity score: {state.get('severity_score', 0):.1f}/10\n"
            f"Exploited vulnerabilities: {exploited_vulns[:6]}\n\n"
            "Prioritised remediation report:\n"
            "CRITICAL (fix within 24h): ...\n"
            "HIGH (fix within 1 week): ...\n"
            "MEDIUM (fix within 1 month): ...\n"
            "QUICK WINS (immediate, low-effort): ..."
        )

        output = self._call_llm(system, user) or self._rule_remediation(ns)
        state["remediation"] = output
        return state

    # ── Rule-based Fallbacks ──────────────────────────────────────────────────

    def _rule_analyst(self, ns: dict) -> str:
        hosts = ns.get("hosts", {})
        ep = ns.get("episode_info", {})
        comp = ep.get("hosts_compromised", 0)
        total = max(len(hosts), 1)
        crit_owned = sum(1 for h in hosts.values()
                         if h.get("status") == "COMPROMISED" and h.get("is_critical"))
        return (
            f"Agent controls {comp}/{total} hosts "
            f"({crit_owned} critical) from position H{ns.get('attacker_node', 0)}. "
            f"Progress: {comp / total * 100:.0f}% network penetration."
        )

    def _rule_attacker(self, ns: dict) -> str:
        hosts = ns.get("hosts", {})
        best_target, best_prob = None, 0.0
        for hid, h in hosts.items():
            if h.get("discovered") and h.get("status") != "COMPROMISED":
                for v in h.get("vulnerabilities", []):
                    p = v.get("exploit_prob", 0) if isinstance(v, dict) else 0.5
                    if p > best_prob and not h.get("firewall", False):
                        best_prob = p
                        best_target = (hid, h, v)
        if best_target:
            hid, h, v = best_target
            return (
                f"EXPLOIT Host {hid} ({h.get('ip','?')}, {h.get('os','?')}) "
                f"via {v.get('name','?')} — {best_prob*100:.0f}% success probability."
            )
        undiscovered = [hid for hid, h in hosts.items() if not h.get("discovered")]
        if undiscovered:
            return f"SCAN Host {undiscovered[0]} — no exploitable discovered hosts yet."
        return "All reachable hosts discovered. PIVOT to expand attack surface."

    def _rule_critic(self, ns: dict) -> str:
        ep = ns.get("episode_info", {})
        comp = ep.get("hosts_compromised", 0)
        disc = ep.get("hosts_discovered", 0)
        if disc == 0:
            return (
                "• No hosts discovered — agent is not scanning.\n"
                "• Reconnaissance phase entirely missed.\n"
                "• Risk: Low (no actual penetration)."
            )
        ratio = comp / max(disc, 1)
        if ratio < 0.3:
            return (
                "• Low exploit-to-discovery ratio — discovering but not compromising.\n"
                "• High-probability vulnerabilities likely being skipped.\n"
                "• Risk: Medium — partial foothold."
            )
        return (
            f"• Good progress: {comp} hosts compromised.\n"
            "• Should target critical/high-value hosts next.\n"
            "• Risk: High — significant attacker foothold."
        )

    def _rule_plan(self, ns: dict) -> str:
        hosts = ns.get("hosts", {})
        steps = []
        for hid, h in hosts.items():
            if not h.get("discovered"):
                steps.append(f"STEP 1: SCAN Host {hid} — discover new attack surface")
                break
        for hid, h in sorted(
            [(k, v) for k, v in hosts.items()
             if v.get("discovered") and v.get("status") != "COMPROMISED"],
            key=lambda x: max(
                (vv.get("exploit_prob", 0) if isinstance(vv, dict) else 0.5
                 for vv in x[1].get("vulnerabilities", [])), default=0
            ),
            reverse=True,
        )[:1]:
            steps.append(f"STEP 2: EXPLOIT Host {hid} — highest-probability target")
        for hid, h in hosts.items():
            if h.get("is_critical") and h.get("status") != "COMPROMISED":
                steps.append(f"STEP 3: EXPLOIT Host {hid} (CRITICAL) — crown jewel")
                break
        steps.append("STEP 4: PIVOT to furthest compromised host to expand reach")
        steps.append("STEP 5: EXFILTRATE from highest data_value compromised host")
        return "\n".join(steps) or "STEP 1: SCAN all reachable hosts"

    def _rule_remediation(self, ns: dict) -> str:
        hosts = ns.get("hosts", {})
        vulns = {
            v.get("name", "?")
            for h in hosts.values()
            if h.get("status") == "COMPROMISED"
            for v in h.get("vulnerabilities", [])
            if isinstance(v, dict)
        }
        vstr = ", ".join(list(vulns)[:4]) or "identified vulnerabilities"
        return (
            f"CRITICAL (24h): Emergency patch {vstr}. "
            "Isolate compromised hosts immediately.\n"
            "HIGH (1 week): Network segmentation — IoT to separate VLAN. "
            "Enable host-based firewalls on all endpoints.\n"
            "MEDIUM (1 month): Deploy SIEM with alert rules. "
            "Enable MFA on all privileged accounts. Audit logging.\n"
            "QUICK WINS: Change all default credentials immediately. "
            "Disable Telnet and unused services. Update firmware."
        )

    # ── Public Interface ──────────────────────────────────────────────────────

    def step(self, network_state: dict) -> dict:
        """Run full analysis graph on current network state."""
        initial: ARCAGraphState = {
            "network_state":   network_state,
            "analyst_output":  "",
            "attacker_output": "",
            "critic_output":   "",
            "reflection":      "",
            "plan":            "",
            "remediation":     "",
            "severity_score":  0.0,
            "episode_history": self._memory[-5:],
        }

        if self._graph:
            result = self._graph.invoke(initial)
        else:
            # Sequential fallback when LangGraph compile failed
            result = dict(initial)
            for node_fn in [
                self._analyst_node,
                self._attacker_node,
                self._critic_node,
                self._reflector_node,
                self._planner_node,
                self._remediator_node,
            ]:
                result = node_fn(result)

        # Store in session memory
        ep = network_state.get("episode_info", {})
        self._memory.append({
            "step":        network_state.get("step", 0),
            "compromised": ep.get("hosts_compromised", 0),
            "steps":       network_state.get("step", 0),
            "reward":      ep.get("total_reward", 0.0),
            "severity":    result.get("severity_score", 0.0),
            "reflection":  result.get("reflection", ""),
        })

        return result

    def reflect(self, state: dict) -> dict:
        return self.step(state)

    def get_memory(self) -> list[dict]:
        return self._memory

    def get_provider_name(self) -> str:
        return self._provider_name