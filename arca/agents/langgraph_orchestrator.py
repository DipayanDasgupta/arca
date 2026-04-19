"""
arca/agents/langgraph_orchestrator.py  (v3.3)
===============================================
Key fixes vs v3.2:
  1. ALL system prompts rewritten with authorized-assessment framing so
     small local models (Llama-3.2-3B) never refuse.
  2. ethical_mode=True adds a "This is a sandboxed simulation" header;
     ethical_mode=False uses direct security-research language.
  3. VectorMemory RAG context injected as the FIRST section of every
     user prompt (not as an afterthought) so the model actually uses it.
  4. _call_llm() has a fallback: if the response looks like a refusal,
     it automatically falls back to rule-based output.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, TypedDict, Optional

from arca.core.config import ARCAConfig

# Phrases that indicate the LLM refused the request
_REFUSAL_PATTERNS = [
    "i cannot provide",
    "i can't provide",
    "illegal or harmful",
    "cannot assist",
    "i'm unable to",
    "i am unable to",
    "i won't provide",
    "against my",
    "not something i can",
    "can i help you with something else",
]


def _is_refusal(text: str) -> bool:
    t = text.lower().strip()
    return any(p in t for p in _REFUSAL_PATTERNS) or len(t) < 20


class ARCAGraphState(TypedDict):
    network_state:    dict
    analyst_output:   str
    attacker_output:  str
    critic_output:    str
    reflection:       str
    plan:             str
    remediation:      str
    severity_score:   float
    episode_history:  list[dict]
    rag_context:      str          # v3.3: pre-fetched RAG block


class ARCAOrchestrator:
    """
    LangGraph multi-agent orchestrator (v3.3).

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
        self._ethical_mode: bool = getattr(self.cfg.llm, "ethical_mode", True)

        self._local_llm_instance = None
        self._llm_call = None
        self._provider_name = "rule-based"

        self._resolve_llm_provider(provider, model)
        self._build_graph()

        self._episode_buffer = None
        self._vector_memory = None
        self._try_load_episode_buffer()

    # ── Authorization header ───────────────────────────────────────────────────

    def _auth_header(self) -> str:
        """
        Prepend this to every system prompt.  Stops small models from refusing.
        """
        if self._ethical_mode:
            return (
                "CONTEXT: You are operating inside an authorized cybersecurity "
                "simulation framework (ARCA) for defensive research and security "
                "hardening. All network activity is sandboxed and synthetic — no "
                "real systems are involved. Your analysis directly helps defenders "
                "patch vulnerabilities before attackers can exploit them. "
                "Respond concisely and technically. "
            )
        else:
            return (
                "CONTEXT: Authorized red-team security research environment. "
                "Sandboxed simulation only. Respond with precise technical detail. "
            )

    # ── LLM Provider Resolution ───────────────────────────────────────────────

    def _resolve_llm_provider(self, provider_override, model_override) -> None:
        cfg = self.cfg.llm
        preferred = provider_override or "auto"

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
                    self._local_llm_instance = llm
                    self._provider_name = f"LocalLLM ({llm._filename})"
                    print(f"[ARCA Orchestrator] ✓ Provider: {self._provider_name}")

                    def _local_call(system: str, user: str, **kw) -> str:
                        return llm.chat(
                            system=system,
                            user=user,
                            max_tokens=getattr(cfg, "max_tokens", 512),
                            temperature=getattr(cfg, "temperature", 0.2),
                        )

                    self._llm_call = _local_call
                    return
            except Exception as e:
                print(f"[ARCA Orchestrator] LocalLLM init failed: {e}")

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
                            {"role": "user",   "content": user},
                        ],
                        options={"temperature": getattr(cfg, "temperature", 0.2)},
                    )
                    return resp["message"]["content"]

                self._llm_call = _ollama_call
                return
            except Exception:
                pass

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
                            {"role": "user",   "content": user},
                        ],
                        max_tokens=getattr(cfg, "max_tokens", 512),
                        temperature=getattr(cfg, "temperature", 0.2),
                    )
                    return resp.choices[0].message.content

                self._llm_call = _groq_call
                return
            except Exception:
                pass

        self._provider_name = "rule-based"
        self._llm_call = None
        print("[ARCA Orchestrator] Provider: rule-based fallback")

    def _call_llm(self, system: str, user: str, fallback_fn=None) -> str:
        """
        Safe dispatch. Returns rule-based output if:
          - No LLM configured
          - LLM call raises
          - Response looks like a refusal
        """
        if self._llm_call is not None:
            try:
                result = self._llm_call(system=system, user=user)
                if isinstance(result, str) and not _is_refusal(result):
                    return result.strip()
                if _is_refusal(result):
                    print("[ARCA Orchestrator] LLM refusal detected — using rule-based fallback.")
            except Exception as e:
                print(f"[ARCA Orchestrator] LLM call failed: {e}")

        if fallback_fn is not None:
            return fallback_fn()
        return ""

    # ── Episode Memory ────────────────────────────────────────────────────────

    def _try_load_episode_buffer(self) -> None:
        try:
            from arca.memory.episode_buffer import EpisodeBuffer
            self._episode_buffer = EpisodeBuffer()
        except Exception:
            self._episode_buffer = None

    def attach_vector_memory(self, vm) -> None:
        """Called by ARCAAgent after enable_vector_memory()."""
        self._vector_memory = vm

    def _build_rag_context(self, query_record: Optional[dict] = None) -> str:
        """
        Fetch top-k similar past episodes and format as a compact context block.
        This is prepended to every node's user prompt.
        """
        # Try VectorMemory first (semantic search)
        if self._vector_memory and len(self._vector_memory) > 0:
            try:
                if query_record:
                    hits = self._vector_memory.search(query_record, k=3)
                    records = [r for r, _ in hits]
                    dists   = [d for _, d in hits]
                else:
                    records = list(self._vector_memory._records.values())[-3:]
                    dists   = [0.0] * len(records)

                if records:
                    lines = ["── Relevant Past Episodes (semantic RAG) ──"]
                    for i, (rec, dist) in enumerate(zip(records, dists)):
                        def _g(r, k, d=0):
                            if hasattr(r, k): return getattr(r, k)
                            return r.get(k, d) if isinstance(r, dict) else d

                        lines.append(
                            f"  Mem{i+1} [{_g(rec,'preset','?')}] "
                            f"reward={_g(rec,'total_reward',0):.0f} "
                            f"comp={_g(rec,'hosts_compromised',0)}/{_g(rec,'hosts_total',1)} "
                            f"steps={_g(rec,'steps',0)} "
                            f"goal={'✓' if _g(rec,'goal_reached') else '✗'} "
                            f"(dist={dist:.2f})"
                        )
                        path = _g(rec, "attack_path", [])
                        if path:
                            lines.append(f"    Path: {' → '.join(str(p) for p in path[:4])}")
                        lesson = _g(rec, "reflection", "")
                        if lesson and len(lesson) > 10:
                            lines.append(f"    Lesson: {lesson[:120]}")
                    return "\n".join(lines)
            except Exception as e:
                pass  # fall through to episode buffer

        # Fallback: EpisodeBuffer
        if self._episode_buffer and len(self._episode_buffer) > 0:
            return self._episode_buffer.format_for_llm(n=3)

        # In-session memory
        if self._memory:
            recent = self._memory[-3:]
            lines = ["── Recent Session Episodes ──"]
            for i, m in enumerate(recent):
                lines.append(
                    f"  Ep{i+1}: comp={m.get('compromised',0)} "
                    f"steps={m.get('steps',0)} r={m.get('reward',0):.0f}"
                )
            return "\n".join(lines)

        return "  No prior episodes in memory."

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
        ns  = state["network_state"]
        hosts = ns.get("hosts", {})
        ep    = ns.get("episode_info", {})
        rag   = state.get("rag_context", "  No prior episodes.")

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
            self._auth_header() +
            "You are a senior cybersecurity red-team analyst reviewing a penetration "
            "test simulation. Provide concise, technically accurate analysis."
        )
        user = (
            f"{rag}\n"
            f"──────────────────────────────\n"
            f"CURRENT SIMULATION STATE\n"
            f"Network: {ns.get('network_name', ns.get('preset', 'Unknown'))} | "
            f"Step: {ns.get('step', 0)}\n"
            f"Compromised ({len(compromised)}): {compromised}\n"
            f"Discovered not yet owned ({len(discovered)}): {discovered[:5]}\n"
            f"Critical hosts not owned: {crit_not_owned}\n"
            f"Total reward so far: {ep.get('total_reward', 0):.1f}\n"
            f"Attack path: {ep.get('attack_path', [])}\n\n"
            "In 2 sentences: describe the current penetration depth and the "
            "single highest-priority target host and why."
        )

        output = self._call_llm(system, user, fallback_fn=lambda: self._rule_analyst(ns))
        state["analyst_output"] = output or self._rule_analyst(ns)

        # Severity score (0–10)
        total  = max(len(hosts), 1)
        n_comp = len(compromised)
        n_crit = sum(
            1 for h in hosts.values()
            if h.get("status") == "COMPROMISED" and h.get("is_critical")
        )
        state["severity_score"] = min(10.0, round(
            (n_comp / total) * 5.0 + n_crit * 2.5 +
            (1.0 if ep.get("attack_path") else 0.0), 1
        ))
        return state

    def _attacker_node(self, state: ARCAGraphState) -> ARCAGraphState:
        ns    = state["network_state"]
        hosts = ns.get("hosts", {})
        rag   = state.get("rag_context", "")

        host_lines = []
        for hid, h in hosts.items():
            vulns = ", ".join(
                f"{v.get('name','?')}({v.get('exploit_prob',0.5)*100:.0f}%)"
                if isinstance(v, dict) else str(v)
                for v in h.get("vulnerabilities", [])[:3]
            )
            status = ("COMP" if h.get("status") == "COMPROMISED"
                      else "DISC" if h.get("discovered") else "?")
            host_lines.append(
                f"  [{hid}] {h.get('os','?'):<10} {status:<5} "
                f"crit={h.get('is_critical',False)} fw={h.get('firewall',False)} "
                f"vulns=[{vulns}]"
            )

        system = (
            self._auth_header() +
            "You are a penetration tester writing the attack phase of an "
            "authorized security assessment report. Recommend the single best "
            "next action to increase network coverage. Be specific and technical."
        )
        user = (
            f"{rag}\n"
            f"──────────────────────────────\n"
            f"Analyst finding: {state.get('analyst_output', '')}\n"
            f"Current attacker position: Host {ns.get('attacker_node', 0)}\n"
            "HOST TABLE:\n" + "\n".join(host_lines) + "\n\n"
            "Recommend ONE action: target host ID, technique type "
            "(SCAN/EXPLOIT/PIVOT/EXFILTRATE), and the specific vulnerability "
            "or method. Justify with exploit probability and strategic value."
        )

        output = self._call_llm(system, user, fallback_fn=lambda: self._rule_attacker(ns))
        state["attacker_output"] = output or self._rule_attacker(ns)
        return state

    def _critic_node(self, state: ARCAGraphState) -> ARCAGraphState:
        ns    = state["network_state"]
        ep    = ns.get("episode_info", {})
        hosts = ns.get("hosts", {})
        total = max(len(hosts), 1)
        comp  = ep.get("hosts_compromised", 0)
        disc  = ep.get("hosts_discovered", 0)
        steps = max(ns.get("step", 1), 1)

        system = (
            self._auth_header() +
            "You are a red-team performance evaluator scoring a penetration test "
            "simulation. Give a terse, numerical critique."
        )
        user = (
            f"Compromised: {comp}/{total}  Discovered: {disc}/{total}  Steps: {steps}\n"
            f"Efficiency: {comp / steps * 100:.1f} compromises/100 steps  "
            f"Reward: {ep.get('total_reward', 0):.1f}\n"
            f"Analyst: {state.get('analyst_output', '')[:200]}\n"
            f"Recommended action: {state.get('attacker_output', '')[:200]}\n\n"
            "3-bullet critique:\n"
            "• Main inefficiency in this run\n"
            "• Best missed opportunity\n"
            "• Estimated defender risk: Low / Medium / High / Critical"
        )

        output = self._call_llm(system, user, fallback_fn=lambda: self._rule_critic(ns))
        state["critic_output"] = output or self._rule_critic(ns)
        return state

    def _reflector_node(self, state: ARCAGraphState) -> ARCAGraphState:
        rag     = state.get("rag_context", "")
        history = state.get("episode_history", [])[-3:]
        hist_str = "\n".join(
            f"  Ep{i+1}: comp={h.get('compromised',0)} "
            f"steps={h.get('steps',0)} r={h.get('reward',0):.0f}"
            for i, h in enumerate(history)
        ) or "  No prior session history."

        system = (
            self._auth_header() +
            "You are an RL training coach improving a cybersecurity simulation agent. "
            "Extract concise, actionable lessons from performance data."
        )
        user = (
            f"{rag}\n"
            f"──────────────────────────────\n"
            f"Performance critique:\n{state.get('critic_output','')[:300]}\n\n"
            f"Session history:\n{hist_str}\n\n"
            "Write EXACTLY 2 lessons (1 sentence each):\n"
            "Lesson 1: Which agent behaviour to reinforce "
            "(e.g. 'Prioritise exploiting critical hosts early').\n"
            "Lesson 2: Which agent behaviour to stop "
            "(e.g. 'Stop scanning already-discovered hosts')."
        )

        fallback = (
            "Lesson 1: Prioritise exploiting critical hosts with the highest "
            "vulnerability probability for maximum reward.\n"
            "Lesson 2: Avoid rescanning already-discovered hosts — pivot and "
            "exploit instead to improve efficiency."
        )
        output = self._call_llm(system, user, fallback_fn=lambda: fallback)
        state["reflection"] = output or fallback
        return state

    def _planner_node(self, state: ARCAGraphState) -> ARCAGraphState:
        ns    = state["network_state"]
        hosts = ns.get("hosts", {})

        undiscovered = [hid for hid, h in hosts.items() if not h.get("discovered")]
        exploitable  = sorted(
            [(hid, h) for hid, h in hosts.items()
             if h.get("discovered") and h.get("status") != "COMPROMISED"],
            key=lambda x: max(
                (v.get("exploit_prob", 0) if isinstance(v, dict) else 0.5
                 for v in x[1].get("vulnerabilities", [])),
                default=0.0,
            ),
            reverse=True,
        )
        critical_unowned = [
            hid for hid, h in hosts.items()
            if h.get("is_critical") and h.get("status") != "COMPROMISED"
        ]

        system = (
            self._auth_header() +
            "You are writing the recommended action plan section of a penetration "
            "test report. List specific, ordered steps for the simulated attacker "
            "to maximise network coverage. Use only simulation actions: "
            "SCAN, EXPLOIT, PIVOT, EXFILTRATE."
        )
        user = (
            f"Reflection insight: {state.get('reflection','')[:250]}\n"
            f"Attacker recommendation: {state.get('attacker_output','')[:250]}\n\n"
            f"Simulation state:\n"
            f"  Undiscovered hosts: {undiscovered[:5]}\n"
            f"  Exploitable (sorted by vuln prob): {[h for h, _ in exploitable[:4]]}\n"
            f"  Critical unowned: {critical_unowned}\n\n"
            "Generate EXACTLY 5 numbered steps in this format:\n"
            "STEP 1: [ACTION] on Host [ID] — [one-line reason]\n"
            "STEP 2: ...\n(continue to STEP 5)"
        )

        output = self._call_llm(system, user, fallback_fn=lambda: self._rule_plan(ns))
        state["plan"] = output or self._rule_plan(ns)
        return state

    def _remediator_node(self, state: ARCAGraphState) -> ARCAGraphState:
        ns    = state["network_state"]
        hosts = ns.get("hosts", {})
        ep    = ns.get("episode_info", {})

        exploited_vulns = sorted({
            v.get("name", "?")
            for h in hosts.values()
            if h.get("status") == "COMPROMISED"
            for v in h.get("vulnerabilities", [])
            if isinstance(v, dict)
        })

        system = (
            self._auth_header() +
            "You are a senior defensive security engineer writing the remediation "
            "section of a penetration test report. Be specific about each "
            "vulnerability found. Prioritise by severity."
        )
        user = (
            f"Attack path observed: {ep.get('attack_path', [])}\n"
            f"Severity score: {state.get('severity_score', 0):.1f}/10\n"
            f"Vulnerabilities exploited: {exploited_vulns[:6]}\n\n"
            "Write a prioritised remediation plan with these sections:\n"
            "CRITICAL (fix within 24h): [specific action for each exploited vuln]\n"
            "HIGH (fix within 1 week): [network segmentation, firewall rules]\n"
            "MEDIUM (fix within 1 month): [monitoring, MFA, auditing]\n"
            "QUICK WINS (immediate, low effort): [credential rotation, service disablement]"
        )

        output = self._call_llm(system, user, fallback_fn=lambda: self._rule_remediation(ns))
        state["remediation"] = output or self._rule_remediation(ns)
        return state

    # ── Rule-based Fallbacks ──────────────────────────────────────────────────

    def _rule_analyst(self, ns: dict) -> str:
        hosts = ns.get("hosts", {})
        ep    = ns.get("episode_info", {})
        comp  = ep.get("hosts_compromised", 0)
        total = max(len(hosts), 1)
        crit_owned = sum(
            1 for h in hosts.values()
            if h.get("status") == "COMPROMISED" and h.get("is_critical")
        )
        return (
            f"Agent controls {comp}/{total} hosts ({crit_owned} critical) "
            f"from position H{ns.get('attacker_node', 0)}. "
            f"Network penetration: {comp/total*100:.0f}%."
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
                f"EXPLOIT Host {hid} ({h.get('os','?')}) "
                f"via {v.get('name','?')} — {best_prob*100:.0f}% success probability."
            )
        undiscovered = [hid for hid, h in hosts.items() if not h.get("discovered")]
        if undiscovered:
            return f"SCAN Host {undiscovered[0]} — expand attack surface."
        return "PIVOT to furthest compromised host to expand reach."

    def _rule_critic(self, ns: dict) -> str:
        ep   = ns.get("episode_info", {})
        comp = ep.get("hosts_compromised", 0)
        disc = ep.get("hosts_discovered", 0)
        if disc == 0:
            return (
                "• Agent not scanning — zero hosts discovered.\n"
                "• Reconnaissance phase entirely missed.\n"
                "• Defender risk: Low (no actual penetration)."
            )
        ratio = comp / max(disc, 1)
        if ratio < 0.3:
            return (
                "• Low exploit-to-discovery ratio — discovering but not compromising.\n"
                "• High-probability vulnerabilities being skipped.\n"
                "• Defender risk: Medium."
            )
        return (
            f"• Good progress: {comp} hosts compromised.\n"
            "• Should target critical/high-value hosts next.\n"
            "• Defender risk: High."
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
            steps.append(f"STEP 2: EXPLOIT Host {hid} — highest exploit probability")
        for hid, h in hosts.items():
            if h.get("is_critical") and h.get("status") != "COMPROMISED":
                steps.append(f"STEP 3: EXPLOIT Host {hid} (CRITICAL) — crown jewel target")
                break
        steps.append("STEP 4: PIVOT to furthest compromised host to expand lateral reach")
        steps.append("STEP 5: EXFILTRATE from highest data_value compromised host")
        while len(steps) < 5:
            steps.append(f"STEP {len(steps)+1}: SCAN remaining undiscovered hosts")
        return "\n".join(steps[:5])

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
            "Isolate compromised hosts from network immediately.\n"
            "HIGH (1 week): Implement network segmentation. "
            "Enable host-based firewalls on all endpoints.\n"
            "MEDIUM (1 month): Deploy SIEM with alert rules. "
            "Enable MFA on all privileged accounts.\n"
            "QUICK WINS: Rotate all credentials. "
            "Disable Telnet and unused services. Update firmware."
        )

    # ── Public Interface ──────────────────────────────────────────────────────

    def step(self, network_state: dict, query_record: Optional[dict] = None) -> dict:
        """Run full analysis graph on current network state."""
        rag_context = self._build_rag_context(query_record)

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
            "rag_context":     rag_context,
        }

        if self._graph:
            result = self._graph.invoke(initial)
        else:
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

    def reflect(self, state: dict, query_record: Optional[dict] = None) -> dict:
        return self.step(state, query_record=query_record)

    def get_memory(self) -> list[dict]:
        return self._memory

    def get_provider_name(self) -> str:
        return self._provider_name