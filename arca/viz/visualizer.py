"""
arca.viz.visualizer
~~~~~~~~~~~~~~~~~~~
Rich visualization suite for ARCA using Plotly and NetworkX.

Plots:
  - Network topology graph (with host status coloring)
  - Attack path overlay
  - Training reward curve
  - Exploit success heatmap
  - Host vulnerability radar
  - Episode statistics
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Any

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
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False


class ARCAVisualizer:
    """Static visualization methods for ARCA network states and training metrics."""

    def __init__(self, output_dir: str = "arca_outputs/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Network Graph
    # ------------------------------------------------------------------

    def plot_network(
        self,
        graph,
        hosts: dict,
        title: str = "ARCA Network Topology",
        save: bool = True,
        show: bool = False,
    ):
        if not PLOTLY_AVAILABLE or not NX_AVAILABLE:
            return self._mpl_network(graph, hosts, title, save)

        pos = nx.spring_layout(graph, seed=42)

        # Edge traces
        edge_x, edge_y = [], []
        for u, v in graph.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.8, color="#334155"),
            hoverinfo="none",
            mode="lines",
        )

        # Node traces by status
        color_map = {
            "UNKNOWN": "#1e293b",
            "DISCOVERED": "#f59e0b",
            "COMPROMISED": "#ef4444",
        }
        icon_map = {
            "UNKNOWN": "❓",
            "DISCOVERED": "🔍",
            "COMPROMISED": "💀",
        }

        node_x, node_y, node_colors, node_text, node_hover = [], [], [], [], []
        for node, host in hosts.items():
            if node not in pos:
                continue
            x, y = pos[node]
            status = host.status.name if hasattr(host.status, "name") else str(host.status)
            node_x.append(x)
            node_y.append(y)
            node_colors.append(color_map.get(status, "#64748b"))
            node_text.append(str(node))
            node_hover.append(
                f"Host {node}<br>"
                f"OS: {host.os}<br>"
                f"IP: {host.ip}<br>"
                f"Status: {status}<br>"
                f"Subnet: {host.subnet}<br>"
                f"Vulns: {len(host.vulnerabilities)}<br>"
                f"Services: {', '.join(host.services)}<br>"
                f"Critical: {'⭐' if host.is_critical else 'No'}"
            )

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="top center",
            hovertext=node_hover,
            hoverinfo="text",
            marker=dict(
                size=20,
                color=node_colors,
                line=dict(width=2, color="#94a3b8"),
                symbol="circle",
            ),
        )

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=dict(text=title, font=dict(color="#f1f5f9", size=16)),
                paper_bgcolor="#0f172a",
                plot_bgcolor="#0f172a",
                showlegend=False,
                hovermode="closest",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                font=dict(color="#f1f5f9"),
                margin=dict(l=20, r=20, t=50, b=20),
                annotations=[
                    dict(text="⬛ Unknown  🟡 Discovered  🔴 Compromised",
                         x=0.5, y=-0.05, xref="paper", yref="paper",
                         showarrow=False, font=dict(color="#94a3b8", size=11))
                ],
            ),
        )

        if save:
            path = self.output_dir / "network_topology.html"
            fig.write_html(str(path))
            print(f"[ARCA] Network graph saved → {path}")
        if show:
            fig.show()
        return fig

    # ------------------------------------------------------------------
    # Training Curves
    # ------------------------------------------------------------------

    def plot_training_curves(
        self,
        log_data: dict,
        save: bool = True,
        show: bool = False,
    ):
        if not PLOTLY_AVAILABLE:
            return None

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Episode Reward", "Hosts Compromised / Episode",
                "Attack Path Length", "Exploit Success Rate"
            ],
            vertical_spacing=0.15,
        )

        episodes = log_data.get("episodes", list(range(len(log_data.get("rewards", [])))))
        rewards = log_data.get("rewards", [])
        compromised = log_data.get("compromised", [])
        path_lengths = log_data.get("path_lengths", [])
        success_rates = log_data.get("success_rates", [])

        # Smooth helper
        def smooth(data, w=5):
            import numpy as np
            if len(data) < w:
                return data
            kernel = np.ones(w) / w
            return np.convolve(data, kernel, mode="valid").tolist()

        color = "#22d3ee"
        smooth_color = "#f472b6"

        if rewards:
            fig.add_trace(go.Scatter(x=episodes[:len(rewards)], y=rewards,
                                     mode="lines", name="Reward", line=dict(color=color, width=1),
                                     opacity=0.5), row=1, col=1)
            s = smooth(rewards)
            fig.add_trace(go.Scatter(x=episodes[len(rewards)-len(s):len(rewards)], y=s,
                                     mode="lines", name="Smoothed", line=dict(color=smooth_color, width=2)),
                          row=1, col=1)

        if compromised:
            fig.add_trace(go.Scatter(x=episodes[:len(compromised)], y=compromised,
                                     mode="lines+markers", name="Compromised",
                                     line=dict(color="#f59e0b", width=1.5),
                                     marker=dict(size=3)), row=1, col=2)

        if path_lengths:
            fig.add_trace(go.Scatter(x=episodes[:len(path_lengths)], y=path_lengths,
                                     mode="lines", name="Path Length",
                                     line=dict(color="#a78bfa", width=1.5)), row=2, col=1)

        if success_rates:
            fig.add_trace(go.Scatter(x=episodes[:len(success_rates)], y=success_rates,
                                     mode="lines", name="Success Rate",
                                     fill="tozeroy",
                                     line=dict(color="#34d399", width=1.5)), row=2, col=2)

        fig.update_layout(
            title=dict(text="ARCA Training Metrics", font=dict(color="#f1f5f9", size=16)),
            paper_bgcolor="#0f172a",
            plot_bgcolor="#1e293b",
            font=dict(color="#f1f5f9"),
            showlegend=False,
            height=600,
        )
        fig.update_xaxes(gridcolor="#334155", zerolinecolor="#334155")
        fig.update_yaxes(gridcolor="#334155", zerolinecolor="#334155")

        if save:
            path = self.output_dir / "training_curves.html"
            fig.write_html(str(path))
            print(f"[ARCA] Training curves saved → {path}")
        if show:
            fig.show()
        return fig

    # ------------------------------------------------------------------
    # Attack Path
    # ------------------------------------------------------------------

    def plot_attack_path(
        self,
        attack_path: list[str],
        hosts: dict,
        save: bool = True,
        show: bool = False,
    ):
        if not PLOTLY_AVAILABLE or not attack_path:
            return None

        steps = list(range(len(attack_path)))
        labels = [p.split("(")[0] for p in attack_path]
        cvss = [p.split("CVE:")[1].rstrip(")") if "CVE:" in p else "?" for p in attack_path]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=steps, y=[1] * len(steps),
            mode="markers+text+lines",
            text=labels,
            textposition="top center",
            marker=dict(size=20, color="#ef4444",
                        symbol="arrow-right", line=dict(width=2, color="#fca5a5")),
            line=dict(color="#ef4444", width=2, dash="dot"),
            hovertext=[f"Step {i}: {p}<br>CVE: {c}" for i, (p, c) in enumerate(zip(attack_path, cvss))],
            hoverinfo="text",
        ))

        fig.update_layout(
            title=dict(text="ARCA Attack Path", font=dict(color="#f1f5f9", size=16)),
            paper_bgcolor="#0f172a",
            plot_bgcolor="#1e293b",
            font=dict(color="#f1f5f9"),
            xaxis=dict(title="Attack Step", gridcolor="#334155"),
            yaxis=dict(showticklabels=False, range=[0.5, 1.5]),
            height=300,
        )

        if save:
            path = self.output_dir / "attack_path.html"
            fig.write_html(str(path))
        if show:
            fig.show()
        return fig

    # ------------------------------------------------------------------
    # Vulnerability heatmap
    # ------------------------------------------------------------------

    def plot_vuln_heatmap(
        self,
        hosts: dict,
        save: bool = True,
        show: bool = False,
    ):
        if not PLOTLY_AVAILABLE:
            return None

        import numpy as np
        host_ids = sorted(hosts.keys())
        os_list = sorted({h.os for h in hosts.values()})
        matrix = np.zeros((len(os_list), len(host_ids)))

        for j, hid in enumerate(host_ids):
            host = hosts[hid]
            i = os_list.index(host.os)
            matrix[i][j] = len(host.vulnerabilities)

        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=[f"H{i}" for i in host_ids],
            y=os_list,
            colorscale="Reds",
            showscale=True,
            text=matrix.astype(int),
            texttemplate="%{text}",
        ))
        fig.update_layout(
            title=dict(text="Vulnerability Density by Host & OS", font=dict(color="#f1f5f9")),
            paper_bgcolor="#0f172a",
            plot_bgcolor="#1e293b",
            font=dict(color="#f1f5f9"),
            height=350,
        )
        if save:
            path = self.output_dir / "vuln_heatmap.html"
            fig.write_html(str(path))
        if show:
            fig.show()
        return fig

    # ------------------------------------------------------------------
    # MPL fallback
    # ------------------------------------------------------------------

    def _mpl_network(self, graph, hosts, title, save):
        if not MPL_AVAILABLE or not NX_AVAILABLE:
            print("[ARCA] No visualization backend available.")
            return None

        color_map = {"UNKNOWN": "gray", "DISCOVERED": "orange", "COMPROMISED": "red"}
        colors = [color_map.get(hosts[n].status.name if hasattr(hosts[n].status, "name") else "UNKNOWN", "gray")
                  for n in graph.nodes() if n in hosts]

        fig, ax = plt.subplots(figsize=(10, 7), facecolor="#0f172a")
        ax.set_facecolor("#1e293b")
        pos = nx.spring_layout(graph, seed=42)
        nx.draw_networkx(graph, pos=pos, ax=ax, node_color=colors,
                         edge_color="#334155", font_color="white", node_size=500)
        ax.set_title(title, color="white")
        if save:
            path = self.output_dir / "network_topology.png"
            plt.savefig(str(path), dpi=150, bbox_inches="tight", facecolor="#0f172a")
            print(f"[ARCA] Network graph saved → {path}")
        plt.close()
        return fig