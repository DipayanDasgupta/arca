"""
arca/viz/visualizer.py  (v3.6 — Interactive Dashboard)
========================================================
A Plotly + Dash reactive dashboard in the style of Mesa/Solara.

Features:
  • Live network topology graph (hosts, subnets, attack path highlighted)
  • Real-time metrics panel (reward, compromised, steps, goal progress)
  • Curriculum progress indicator (tier + promotion status)
  • Attack path timeline with CVE labels and reward per step
  • Reward curve (live plot across episodes)
  • Host detail panel (click a node → OS, vulns, services, status)
  • Play / Pause / Step controls for episode replay
  • Static export helpers (HTML, PNG) for reports

Usage (interactive dashboard):
    from arca.viz.visualizer import ARCAVisualizer
    viz = ARCAVisualizer(env, agent)
    viz.show()                     # launches Dash server on localhost:8051
    viz.show(port=8052, debug=True)

Usage (static export, no server needed):
    viz = ARCAVisualizer(env)
    viz.plot_network(env.get_network_graph(), env.get_hosts(), save=True)
    viz.plot_training_curves(log_data, save=True)

Integration from quickstart_v3.py:
    viz = ARCAVisualizer(agent.env, agent, output_dir="arca_outputs/figures")
    viz.record_episode(ep)            # call after each episode
    viz.show()                        # or viz.save_all()
"""
from __future__ import annotations

import json
import time
import threading
from pathlib import Path
from typing import Optional, Any

# ── Optional heavy imports (graceful fallback if not installed) ───────────────
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import networkx as nx
    NX_AVAILABLE = True
except ImportError:
    NX_AVAILABLE = False

try:
    from dash import Dash, dcc, html, Input, Output, State, callback_context
    import dash_bootstrap_components as dbc
    DASH_AVAILABLE = True
except ImportError:
    try:
        from dash import Dash, dcc, html, Input, Output, State, callback_context
        DASH_AVAILABLE = True
        dbc = None
    except ImportError:
        DASH_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

# ── ARCA palette ──────────────────────────────────────────────────────────────
_BG       = "#0a0e1a"
_SURFACE  = "#111827"
_CARD     = "#1a2233"
_BORDER   = "#1e2d47"
_ACCENT   = "#00d4ff"
_RED      = "#ff4d6d"
_AMBER    = "#fbbf24"
_GREEN    = "#22c55e"
_PURPLE   = "#a855f7"
_TEXT     = "#e2e8f0"
_MUTED    = "#64748b"

_STATUS_COLOR = {
    "UNKNOWN":     _MUTED,
    "DISCOVERED":  _AMBER,
    "COMPROMISED": _RED,
}
_OS_ICON = {
    "Windows": "🪟",
    "Linux":   "🐧",
    "macOS":   "🍎",
    "IoT":     "📡",
    "Router":  "📶",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

class EpisodeSnapshot:
    """Lightweight snapshot of one completed episode for replay."""
    def __init__(self, ep_info, env_state: dict):
        self.total_reward      = ep_info.total_reward
        self.steps             = ep_info.steps
        self.hosts_compromised = ep_info.hosts_compromised
        self.hosts_discovered  = ep_info.hosts_discovered
        self.goal_reached      = ep_info.goal_reached
        self.attack_path       = list(ep_info.attack_path)
        self.env_state         = env_state      # deep-copied state dict


# ─────────────────────────────────────────────────────────────────────────────
# ARCAVisualizer
# ─────────────────────────────────────────────────────────────────────────────

class ARCAVisualizer:
    """
    Reactive dashboard for ARCA.

    Parameters
    ----------
    env  : NetworkEnv  (optional)
    agent : ARCAAgent  (optional, for live callbacks)
    output_dir : str   where static exports land
    """

    def __init__(
        self,
        env        = None,
        agent      = None,
        output_dir: str = "arca_outputs/figures",
    ):
        self.env        = env
        self.agent      = agent
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ── Live state ────────────────────────────────────────────────────────
        self._episode_rewards:    list[float] = []
        self._episode_goals:      list[bool]  = []
        self._episode_comp:       list[int]   = []
        self._episode_snapshots:  list[EpisodeSnapshot] = []
        self._curriculum_history: list[dict]  = []
        self._current_tier:       str         = "micro"
        self._replay_step:        int         = 0
        self._replay_playing:     bool        = False

        # ── Dash app (created lazily) ─────────────────────────────────────────
        self._app: Optional["Dash"] = None

    # ── Public data ingestion ─────────────────────────────────────────────────

    def record_episode(self, ep_info, env_state: Optional[dict] = None) -> None:
        """Call after each episode to feed the live dashboard."""
        self._episode_rewards.append(ep_info.total_reward)
        self._episode_goals.append(ep_info.goal_reached)
        self._episode_comp.append(ep_info.hosts_compromised)
        if env_state is not None:
            snap = EpisodeSnapshot(ep_info, env_state)
            self._episode_snapshots.append(snap)

    def record_curriculum(self, history: list[dict], current_tier: str = "") -> None:
        self._curriculum_history = history
        self._current_tier = current_tier

    # ── Figure builders ───────────────────────────────────────────────────────

    def _fig_network(
        self,
        graph=None,
        hosts=None,
        attack_path: Optional[list[str]] = None,
        title: str = "Network Topology",
    ) -> "go.Figure":
        """Build an interactive Plotly network graph."""
        if not PLOTLY_AVAILABLE or not NX_AVAILABLE:
            return go.Figure()

        g     = graph or (self.env.get_network_graph() if self.env else None)
        hosts = hosts or (self.env.get_hosts() if self.env else {})
        if g is None or not hosts:
            return go.Figure(layout=go.Layout(
                title="No network data", paper_bgcolor=_BG,
                font=dict(color=_TEXT),
            ))

        pos = nx.spring_layout(g, seed=42, k=2.0)

        # Parse attack path edges
        attack_edges: set[tuple] = set()
        if attack_path:
            for step in attack_path:
                try:
                    arrow = step.find("→")
                    if arrow > 0:
                        src = int(step[:arrow].strip())
                        dst = int(step[arrow+1:].split("(")[0].strip())
                        attack_edges.add((src, dst))
                except Exception:
                    pass

        # Edges: normal vs attack-path
        def add_edge_trace(edges, color, width, dash, name):
            ex, ey = [], []
            for u, v in edges:
                if u in pos and v in pos:
                    x0, y0 = pos[u]; x1, y1 = pos[v]
                    ex += [x0, x1, None]; ey += [y0, y1, None]
            return go.Scatter(
                x=ex, y=ey, mode="lines", name=name, hoverinfo="none",
                line=dict(color=color, width=width, dash=dash),
            )

        normal_edges = [
            (u, v) for u, v in g.edges() if (u, v) not in attack_edges
        ]
        traces = [
            add_edge_trace(normal_edges, _BORDER, 1.2, "solid", "Edge"),
            add_edge_trace(list(attack_edges), _RED, 3, "dot", "Attack path"),
        ]

        # Nodes
        node_x, node_y = [], []
        node_colors, node_sizes, node_text, node_hover = [], [], [], []

        for nid, h in hosts.items():
            if nid not in pos:
                continue
            x, y = pos[nid]
            node_x.append(x); node_y.append(y)

            status = h.status.name if hasattr(h.status, "name") else str(h.status)
            color  = _STATUS_COLOR.get(status, _MUTED)
            size   = 22 if status == "COMPROMISED" else 16 if status == "DISCOVERED" else 12

            # Extra size for critical hosts
            if getattr(h, "is_critical", False):
                size += 8
                color = _PURPLE if status != "COMPROMISED" else _RED

            node_colors.append(color)
            node_sizes.append(size)
            node_text.append(f"{_OS_ICON.get(h.os, '?')}{nid}")

            vulns = ", ".join(
                v.get("cve", "?") if isinstance(v, dict) else str(v)
                for v in h.vulnerabilities[:3]
            ) or "none"
            fw_str = " 🔥fw" if getattr(h, "firewall", False) else ""
            crit_str = " ⭐CRIT" if getattr(h, "is_critical", False) else ""
            node_hover.append(
                f"<b>Host {nid}</b>{crit_str}{fw_str}<br>"
                f"OS: {h.os} | Subnet: {h.subnet}<br>"
                f"Status: <b>{status}</b><br>"
                f"IP: {getattr(h, 'ip', '?')}<br>"
                f"Vulns: {vulns}<br>"
                f"Services: {', '.join(getattr(h, 'services', [])[:3])}"
            )

        traces.append(go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="top center",
            textfont=dict(color=_TEXT, size=9),
            hovertext=node_hover,
            hoverinfo="text",
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color=_SURFACE),
                symbol="circle",
            ),
            name="Hosts",
        ))

        fig = go.Figure(data=traces)
        fig.update_layout(
            title=dict(text=title, font=dict(color=_ACCENT, size=14)),
            paper_bgcolor=_BG,
            plot_bgcolor=_BG,
            showlegend=False,
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            font=dict(color=_TEXT),
            margin=dict(l=10, r=10, t=40, b=10),
            annotations=[dict(
                text=(
                    f"<span style='color:{_MUTED}'>●</span> Unknown&nbsp;&nbsp;"
                    f"<span style='color:{_AMBER}'>●</span> Discovered&nbsp;&nbsp;"
                    f"<span style='color:{_RED}'>●</span> Compromised&nbsp;&nbsp;"
                    f"<span style='color:{_PURPLE}'>●</span> Critical"
                ),
                x=0.5, y=-0.04, xref="paper", yref="paper",
                showarrow=False,
                font=dict(color=_MUTED, size=10),
            )],
        )
        return fig

    def _fig_reward_curve(self) -> "go.Figure":
        """Episode reward history with smoothed trend."""
        if not PLOTLY_AVAILABLE:
            return go.Figure()
        rewards = self._episode_rewards
        if not rewards:
            rewards = [0]

        import numpy as np
        eps = list(range(1, len(rewards) + 1))

        # Smooth
        w = min(10, len(rewards))
        kernel = np.ones(w) / w
        smooth = np.convolve(rewards, kernel, mode="valid").tolist()
        smooth_eps = eps[w - 1:]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=eps, y=rewards,
            mode="lines", name="Episode reward",
            line=dict(color=_ACCENT, width=1),
            opacity=0.4,
        ))
        fig.add_trace(go.Scatter(
            x=smooth_eps, y=smooth,
            mode="lines", name=f"Mean{w}",
            line=dict(color=_GREEN, width=2),
        ))
        # Goal markers
        goal_eps = [e for e, g in zip(eps, self._episode_goals) if g]
        goal_r   = [rewards[e - 1] for e in goal_eps]
        if goal_eps:
            fig.add_trace(go.Scatter(
                x=goal_eps, y=goal_r,
                mode="markers", name="Goal ✓",
                marker=dict(color=_GREEN, size=8, symbol="star"),
            ))

        fig.update_layout(
            title=dict(text="Episode Rewards", font=dict(color=_ACCENT, size=13)),
            paper_bgcolor=_BG, plot_bgcolor=_SURFACE,
            font=dict(color=_TEXT),
            xaxis=dict(title="Episode", gridcolor=_BORDER, color=_MUTED),
            yaxis=dict(title="Reward",  gridcolor=_BORDER, color=_MUTED),
            legend=dict(bgcolor=_CARD, bordercolor=_BORDER),
            margin=dict(l=50, r=10, t=40, b=40),
        )
        return fig

    def _fig_attack_timeline(
        self,
        attack_path: Optional[list[str]] = None,
        title: str = "Attack Path Timeline",
    ) -> "go.Figure":
        if not PLOTLY_AVAILABLE:
            return go.Figure()
        path = attack_path or []
        if not path:
            path = ["(no attack path recorded)"]

        labels, colors, tips = [], [], []
        for i, step in enumerate(path):
            cve = step.split("CVE:")[-1].rstrip(")") if "CVE:" in step else "?"
            labels.append(f"Step {i+1}: {step[:40]}")
            colors.append(_RED if i % 2 == 0 else _PURPLE)
            tips.append(f"<b>{step}</b><br>CVE: {cve}")

        fig = go.Figure(go.Bar(
            x=list(range(1, len(path) + 1)),
            y=[1] * len(path),
            text=labels,
            textposition="inside",
            hovertext=tips,
            hoverinfo="text",
            marker=dict(color=colors, line=dict(color=_BG, width=1)),
        ))
        fig.update_layout(
            title=dict(text=title, font=dict(color=_ACCENT, size=13)),
            paper_bgcolor=_BG, plot_bgcolor=_SURFACE,
            font=dict(color=_TEXT),
            xaxis=dict(title="Step", gridcolor=_BORDER),
            yaxis=dict(showticklabels=False),
            margin=dict(l=10, r=10, t=40, b=40),
            showlegend=False,
        )
        return fig

    def _fig_curriculum_progress(self) -> "go.Figure":
        if not PLOTLY_AVAILABLE:
            return go.Figure()

        tiers = ["micro", "small_office", "medium", "hard", "enterprise"]
        reached = set()
        for h in self._curriculum_history:
            reached.add(h.get("tier_name", ""))
        reached.add(self._current_tier)

        colors = []
        for t in tiers:
            if t == self._current_tier:
                colors.append(_ACCENT)
            elif t in reached:
                colors.append(_GREEN)
            else:
                colors.append(_BORDER)

        fig = go.Figure(go.Bar(
            x=tiers,
            y=[1] * len(tiers),
            marker=dict(color=colors),
            text=[
                f"✓ {t}" if t in reached and t != self._current_tier
                else f"► {t}" if t == self._current_tier
                else t
                for t in tiers
            ],
            textposition="inside",
            hoverinfo="skip",
        ))
        fig.update_layout(
            title=dict(text="Curriculum Progress", font=dict(color=_ACCENT, size=13)),
            paper_bgcolor=_BG, plot_bgcolor=_SURFACE,
            font=dict(color=_TEXT),
            xaxis=dict(showgrid=False),
            yaxis=dict(showticklabels=False, showgrid=False),
            margin=dict(l=10, r=10, t=40, b=10),
            showlegend=False,
        )
        return fig

    def _fig_metrics_gauge(self, label, value, max_val, color=_ACCENT) -> "go.Figure":
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            title=dict(text=label, font=dict(color=_TEXT, size=12)),
            number=dict(font=dict(color=color, size=22)),
            gauge=dict(
                axis=dict(range=[0, max_val], tickcolor=_MUTED),
                bar=dict(color=color),
                bgcolor=_SURFACE,
                bordercolor=_BORDER,
                steps=[dict(range=[0, max_val], color=_CARD)],
            ),
        ))
        fig.update_layout(
            paper_bgcolor=_BG,
            font=dict(color=_TEXT),
            height=160,
            margin=dict(l=20, r=20, t=40, b=10),
        )
        return fig

    # ── Static export helpers (no server needed) ──────────────────────────────

    def plot_network(
        self,
        graph=None,
        hosts: Optional[dict] = None,
        title: str = "ARCA Network Topology",
        attack_path: Optional[list[str]] = None,
        save: bool = True,
        show: bool = False,
    ):
        if not PLOTLY_AVAILABLE:
            return self._mpl_network(graph, hosts, title, save)
        fig = self._fig_network(graph, hosts, attack_path, title)
        if save:
            path = self.output_dir / "network_topology.html"
            fig.write_html(str(path))
            print(f"[ARCA Viz] Network graph → {path}")
        if show:
            fig.show()
        return fig

    def plot_training_curves(
        self,
        log_data: dict,
        save: bool = True,
        show: bool = False,
    ):
        if not PLOTLY_AVAILABLE:
            return None

        import numpy as np
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Episode Reward", "Hosts Compromised",
                "Attack Path Length", "Goal Rate (sliding 20)",
            ],
            vertical_spacing=0.18, horizontal_spacing=0.12,
        )

        eps      = log_data.get("episodes", list(range(len(log_data.get("rewards", [])))))
        rewards  = log_data.get("rewards", [])
        comp     = log_data.get("compromised", [])
        plen     = log_data.get("path_lengths", [])
        success  = log_data.get("success_rates", [])

        def smooth(d, w=8):
            if len(d) < w:
                return d
            return np.convolve(d, np.ones(w) / w, mode="valid").tolist()

        for data, row, col, color in [
            (rewards, 1, 1, _ACCENT),
            (comp,    1, 2, _AMBER),
            (plen,    2, 1, _PURPLE),
            (success, 2, 2, _GREEN),
        ]:
            if not data:
                continue
            fig.add_trace(go.Scatter(
                x=eps[:len(data)], y=data,
                mode="lines", line=dict(color=color, width=1),
                opacity=0.4, showlegend=False,
            ), row=row, col=col)
            s = smooth(data)
            if s:
                offset = len(data) - len(s)
                fig.add_trace(go.Scatter(
                    x=eps[offset:len(data)], y=s,
                    mode="lines", line=dict(color=color, width=2),
                    showlegend=False,
                ), row=row, col=col)

        fig.update_layout(
            title=dict(text="ARCA Training Metrics", font=dict(color=_ACCENT, size=15)),
            paper_bgcolor=_BG, plot_bgcolor=_SURFACE,
            font=dict(color=_TEXT),
            height=560,
        )
        fig.update_xaxes(gridcolor=_BORDER, zerolinecolor=_BORDER)
        fig.update_yaxes(gridcolor=_BORDER, zerolinecolor=_BORDER)

        if save:
            path = self.output_dir / "training_curves.html"
            fig.write_html(str(path))
            print(f"[ARCA Viz] Training curves → {path}")
        if show:
            fig.show()
        return fig

    def plot_vuln_heatmap(
        self,
        hosts: Optional[dict] = None,
        save: bool = True,
        show: bool = False,
    ):
        if not PLOTLY_AVAILABLE:
            return None
        hosts = hosts or (self.env.get_hosts() if self.env else {})
        if not hosts:
            return None

        import numpy as np
        host_ids = sorted(hosts.keys())
        os_list  = sorted({h.os for h in hosts.values()})
        matrix   = np.zeros((len(os_list), len(host_ids)))

        for j, hid in enumerate(host_ids):
            h = hosts[hid]
            i = os_list.index(h.os)
            matrix[i][j] = len(h.vulnerabilities)

        fig = go.Figure(go.Heatmap(
            z=matrix,
            x=[f"H{i}" for i in host_ids],
            y=os_list,
            colorscale=[[0, _CARD], [0.5, _AMBER], [1, _RED]],
            showscale=True,
            text=matrix.astype(int),
            texttemplate="%{text}",
            hoverongaps=False,
        ))
        fig.update_layout(
            title=dict(text="Vulnerability Density by Host & OS",
                       font=dict(color=_ACCENT)),
            paper_bgcolor=_BG, plot_bgcolor=_SURFACE,
            font=dict(color=_TEXT),
            height=320,
            margin=dict(l=80, r=10, t=50, b=40),
        )
        if save:
            path = self.output_dir / "vuln_heatmap.html"
            fig.write_html(str(path))
            print(f"[ARCA Viz] Heatmap → {path}")
        if show:
            fig.show()
        return fig

    def save_all(self, attack_path: Optional[list[str]] = None) -> None:
        """Export all static plots to output_dir."""
        if self.env:
            self.plot_network(attack_path=attack_path, save=True)
            self.plot_vuln_heatmap(save=True)
        if self._episode_rewards:
            n = len(self._episode_rewards)
            log_data = {
                "episodes":     list(range(1, n + 1)),
                "rewards":      self._episode_rewards,
                "compromised":  self._episode_comp,
                "path_lengths": [0] * n,
                "success_rates":[float(g) for g in self._episode_goals],
            }
            self.plot_training_curves(log_data, save=True)
        if self._curriculum_history:
            fig = self._fig_curriculum_progress()
            if PLOTLY_AVAILABLE:
                fig.write_html(str(self.output_dir / "curriculum_progress.html"))
                print(f"[ARCA Viz] Curriculum → {self.output_dir}/curriculum_progress.html")
        print(f"\n[ARCA Viz] All static figures saved to {self.output_dir}/")

    # ── Dash interactive dashboard ────────────────────────────────────────────

    def _build_dash_app(self) -> "Dash":
        if not DASH_AVAILABLE:
            raise ImportError(
                "Dash not installed. Run: pip install dash dash-bootstrap-components"
            )

        app = Dash(
            __name__,
            title="ARCA Dashboard",
            external_stylesheets=(
                [dbc.themes.CYBORG] if dbc else []
            ),
            suppress_callback_exceptions=True,
        )

        # ── CSS ───────────────────────────────────────────────────────────────
        _card_style = {
            "background": _CARD, "border": f"1px solid {_BORDER}",
            "borderRadius": "10px", "padding": "12px",
        }
        _header_style = {
            "background": _SURFACE,
            "borderBottom": f"2px solid {_ACCENT}",
            "padding": "12px 24px",
            "display": "flex", "alignItems": "center",
            "justifyContent": "space-between",
        }

        # ── Layout ────────────────────────────────────────────────────────────
        app.layout = html.Div(
            style={"background": _BG, "minHeight": "100vh", "fontFamily": "'Courier New', monospace"},
            children=[
                # Header
                html.Div(style=_header_style, children=[
                    html.Div([
                        html.Span("⚡ ARCA", style={
                            "color": _ACCENT, "fontWeight": "bold",
                            "fontSize": "22px", "letterSpacing": "3px",
                        }),
                        html.Span(" v3.6 · Live Dashboard", style={
                            "color": _MUTED, "fontSize": "13px",
                            "marginLeft": "10px",
                        }),
                    ]),
                    html.Div([
                        html.Span("◉ LIVE", style={
                            "color": _GREEN, "fontSize": "12px",
                            "border": f"1px solid {_GREEN}",
                            "borderRadius": "4px", "padding": "3px 8px",
                        })
                    ]),
                ]),

                # Interval for live updates
                dcc.Interval(id="interval", interval=2_000, n_intervals=0),

                html.Div(style={"padding": "16px"}, children=[

                    # ── Top row: gauges ───────────────────────────────────────
                    html.Div(style={"display": "flex", "gap": "12px",
                                    "marginBottom": "12px"}, children=[
                        html.Div(style={**_card_style, "flex": 1},
                                 children=[dcc.Graph(id="gauge-reward",   config={"displayModeBar": False})]),
                        html.Div(style={**_card_style, "flex": 1},
                                 children=[dcc.Graph(id="gauge-comp",     config={"displayModeBar": False})]),
                        html.Div(style={**_card_style, "flex": 1},
                                 children=[dcc.Graph(id="gauge-goalrate", config={"displayModeBar": False})]),
                        html.Div(style={**_card_style, "flex": 1},
                                 children=[dcc.Graph(id="gauge-episodes", config={"displayModeBar": False})]),
                    ]),

                    # ── Middle row: network + reward curve ───────────────────
                    html.Div(style={"display": "flex", "gap": "12px",
                                    "marginBottom": "12px"}, children=[
                        html.Div(
                            style={**_card_style, "flex": "0 0 55%"},
                            children=[
                                html.P("Click a host node for details",
                                       style={"color": _MUTED, "fontSize": "11px",
                                              "margin": "0 0 6px 0"}),
                                dcc.Graph(
                                    id="network-graph",
                                    config={"displayModeBar": True},
                                    style={"height": "400px"},
                                ),
                                html.Div(id="host-detail", style={
                                    "color": _TEXT, "fontSize": "12px",
                                    "marginTop": "8px", "background": _SURFACE,
                                    "borderRadius": "6px", "padding": "8px",
                                }),
                            ],
                        ),
                        html.Div(style={**_card_style, "flex": 1},
                                 children=[
                                     dcc.Graph(
                                         id="reward-curve",
                                         style={"height": "260px"},
                                         config={"displayModeBar": False},
                                     ),
                                     html.Div(style={"marginTop": "10px"},
                                              children=[
                                                  dcc.Graph(
                                                      id="curriculum-bar",
                                                      style={"height": "130px"},
                                                      config={"displayModeBar": False},
                                                  )
                                              ]),
                                 ]),
                    ]),

                    # ── Bottom row: attack timeline + replay controls ─────────
                    html.Div(style={"display": "flex", "gap": "12px"}, children=[
                        html.Div(style={**_card_style, "flex": "0 0 65%"},
                                 children=[
                                     dcc.Graph(
                                         id="attack-timeline",
                                         style={"height": "180px"},
                                         config={"displayModeBar": False},
                                     ),
                                 ]),
                        html.Div(style={**_card_style, "flex": 1},
                                 children=[
                                     html.P("Episode Replay",
                                            style={"color": _ACCENT, "fontWeight": "bold",
                                                   "fontSize": "13px", "margin": "0 0 10px 0"}),
                                     html.Div(style={"display": "flex", "gap": "8px",
                                                     "marginBottom": "12px"}, children=[
                                         html.Button("⏮ Reset",  id="btn-reset",
                                                     style=_btn_style(_BORDER, _TEXT)),
                                         html.Button("⏪ Prev",   id="btn-prev",
                                                     style=_btn_style(_BORDER, _AMBER)),
                                         html.Button("▶ Play",    id="btn-play",
                                                     style=_btn_style(_ACCENT, _BG)),
                                         html.Button("⏩ Next",   id="btn-next",
                                                     style=_btn_style(_BORDER, _AMBER)),
                                     ]),
                                     html.Div(id="replay-status",
                                              style={"color": _MUTED, "fontSize": "12px"}),
                                     dcc.Store(id="store-replay-step", data=0),
                                 ]),
                    ]),
                ]),  # end padding div
            ],
        )

        # ── Callbacks ─────────────────────────────────────────────────────────

        @app.callback(
            Output("gauge-reward",   "figure"),
            Output("gauge-comp",     "figure"),
            Output("gauge-goalrate", "figure"),
            Output("gauge-episodes", "figure"),
            Output("reward-curve",   "figure"),
            Output("curriculum-bar", "figure"),
            Output("attack-timeline","figure"),
            Input("interval",        "n_intervals"),
        )
        def update_live(_):
            rewards = self._episode_rewards
            comp    = self._episode_comp
            goals   = self._episode_goals

            mean_r  = sum(rewards[-10:]) / max(len(rewards[-10:]), 1) if rewards else 0
            mean_c  = sum(comp[-10:])    / max(len(comp[-10:]),    1) if comp    else 0
            gr      = sum(goals[-20:])   / max(len(goals[-20:]),   1) if goals   else 0

            n_hosts = (
                len(self.env.hosts) if self.env and self.env.hosts
                else 25
            )

            g_reward   = self._fig_metrics_gauge(
                "Mean Reward (10ep)", round(mean_r, 1), 500, _ACCENT
            )
            g_comp     = self._fig_metrics_gauge(
                "Mean Compromised", round(mean_c, 1), n_hosts, _RED
            )
            g_goalrate = self._fig_metrics_gauge(
                "Goal Rate (20ep)", round(gr * 100, 1), 100, _GREEN
            )
            g_eps      = self._fig_metrics_gauge(
                "Total Episodes", len(rewards), max(len(rewards) + 5, 50), _PURPLE
            )

            r_curve    = self._fig_reward_curve()
            curric     = self._fig_curriculum_progress()

            # Attack path from latest snapshot
            latest_path: list[str] = []
            if self._episode_snapshots:
                latest_path = self._episode_snapshots[-1].attack_path
            a_timeline = self._fig_attack_timeline(latest_path)

            return g_reward, g_comp, g_goalrate, g_eps, r_curve, curric, a_timeline

        @app.callback(
            Output("network-graph", "figure"),
            Output("host-detail",   "children"),
            Input("interval",       "n_intervals"),
            Input("network-graph",  "clickData"),
        )
        def update_network(_, click):
            graph = hosts = None
            path  = []
            if self.env:
                try:
                    graph = self.env.get_network_graph()
                    hosts = self.env.get_hosts()
                except Exception:
                    pass
            if self._episode_snapshots:
                path = self._episode_snapshots[-1].attack_path

            fig    = self._fig_network(graph, hosts, path)
            detail = "Click a host node to see details."

            if click and hosts:
                try:
                    pt   = click["points"][0]
                    text = pt.get("text", "")
                    nid  = int("".join(filter(str.isdigit, text.split("›")[-1].strip())))
                    h    = hosts.get(nid)
                    if h:
                        status = h.status.name if hasattr(h.status, "name") else str(h.status)
                        vulns  = [
                            (v.get("cve","?") if isinstance(v, dict) else str(v))
                            for v in h.vulnerabilities
                        ]
                        detail = html.Div([
                            html.B(f"Host {nid} — {_OS_ICON.get(h.os,'?')} {h.os}"),
                            html.Br(),
                            html.Span(f"IP: {getattr(h,'ip','?')} | Subnet: {h.subnet}"),
                            html.Br(),
                            html.Span(f"Status: ", style={"color": _MUTED}),
                            html.Span(status, style={"color": _STATUS_COLOR.get(status, _TEXT)}),
                            html.Br(),
                            html.Span(f"Vulns: {', '.join(vulns[:4]) or 'none'}"),
                            html.Br(),
                            html.Span(f"Services: {', '.join(getattr(h,'services',[])[:4])}"),
                            html.Br(),
                            html.Span("⭐ CRITICAL  " if getattr(h,"is_critical",False) else "",
                                      style={"color": _PURPLE}),
                            html.Span("🔥 Firewall" if getattr(h,"firewall",False) else "",
                                      style={"color": _AMBER}),
                        ])
                except Exception:
                    pass

            return fig, detail

        @app.callback(
            Output("store-replay-step", "data"),
            Output("replay-status",     "children"),
            Input("btn-reset",  "n_clicks"),
            Input("btn-prev",   "n_clicks"),
            Input("btn-next",   "n_clicks"),
            Input("btn-play",   "n_clicks"),
            State("store-replay-step", "data"),
            prevent_initial_call=True,
        )
        def replay_control(r_reset, r_prev, r_next, r_play, step):
            ctx = callback_context
            if not ctx.triggered:
                return step, ""
            btn = ctx.triggered[0]["prop_id"].split(".")[0]
            snaps = self._episode_snapshots
            n     = len(snaps)
            if not snaps:
                return 0, "No episodes recorded yet."

            if btn == "btn-reset": step = 0
            elif btn == "btn-prev": step = max(0, step - 1)
            elif btn == "btn-next": step = min(n - 1, step + 1)
            elif btn == "btn-play": step = (step + 1) % n

            ep = snaps[step]
            status = (
                f"Episode {step + 1}/{n}  |  "
                f"Reward: {ep.total_reward:.0f}  |  "
                f"Comp: {ep.hosts_compromised}  |  "
                f"Goal: {'✓' if ep.goal_reached else '✗'}"
            )
            return step, status

        return app

    def show(self, port: int = 8051, debug: bool = False, open_browser: bool = True) -> None:
        """
        Launch the interactive Dash dashboard.

        Parameters
        ----------
        port : int
            Port for the local server (default 8051).
        debug : bool
            Enable Dash debug mode.
        open_browser : bool
            Open a browser tab automatically.
        """
        if not DASH_AVAILABLE:
            print(
                "[ARCA Viz] Dash not available. Install with:\n"
                "  pip install dash dash-bootstrap-components\n"
                "Falling back to static export."
            )
            self.save_all()
            return

        if self._app is None:
            self._app = self._build_dash_app()

        url = f"http://127.0.0.1:{port}"
        print(f"\n[ARCA Viz] Dashboard running → {url}")
        print("  Press Ctrl+C to stop.\n")

        if open_browser:
            import webbrowser, threading
            threading.Timer(1.5, lambda: webbrowser.open(url)).start()

        self._app.run(debug=debug, port=port, use_reloader=False)

    # ── matplotlib fallback ───────────────────────────────────────────────────

    def _mpl_network(self, graph, hosts, title, save):
        if not MPL_AVAILABLE or not NX_AVAILABLE:
            print("[ARCA Viz] No viz backend available. Install plotly or matplotlib.")
            return None
        color_map = {
            "UNKNOWN": "#475569", "DISCOVERED": "#f59e0b", "COMPROMISED": "#ef4444"
        }
        if graph is None or not hosts:
            return None
        colors = [
            color_map.get(
                hosts[n].status.name if hasattr(hosts[n].status, "name") else "UNKNOWN",
                "#475569",
            )
            for n in graph.nodes() if n in hosts
        ]
        fig, ax = plt.subplots(figsize=(11, 7), facecolor="#0a0e1a")
        ax.set_facecolor("#111827")
        pos = nx.spring_layout(graph, seed=42)
        nx.draw_networkx(graph, pos=pos, ax=ax, node_color=colors,
                         edge_color="#1e2d47", font_color="#e2e8f0",
                         node_size=600, font_size=9)
        ax.set_title(title, color="#00d4ff", fontsize=13)
        if save:
            path = self.output_dir / "network_topology.png"
            plt.savefig(str(path), dpi=150, bbox_inches="tight",
                        facecolor="#0a0e1a")
            print(f"[ARCA Viz] Network graph → {path}")
        plt.close()
        return fig

    # ── Convenience: attack path figure ──────────────────────────────────────

    def plot_attack_path(
        self,
        attack_path: list[str],
        hosts: Optional[dict] = None,
        save: bool = True,
        show: bool = False,
    ):
        fig = self._fig_attack_timeline(attack_path, "Attack Path")
        if save:
            path = self.output_dir / "attack_path.html"
            fig.write_html(str(path)) if PLOTLY_AVAILABLE else None
        if show and PLOTLY_AVAILABLE:
            fig.show()
        return fig


# ── Tiny helper ───────────────────────────────────────────────────────────────

def _btn_style(bg: str, fg: str) -> dict:
    return {
        "background": bg, "color": fg,
        "border": f"1px solid {_BORDER}",
        "borderRadius": "6px", "padding": "6px 12px",
        "cursor": "pointer", "fontSize": "12px",
        "fontFamily": "'Courier New', monospace",
    }