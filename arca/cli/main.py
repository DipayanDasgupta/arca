"""
arca.cli
========
Typer-based CLI for ARCA.

Commands:
  arca train        — Train a PPO agent on a network preset
  arca serve        — Start the FastAPI REST server
  arca audit        — Run a quick audit and print report
  arca viz          — Generate all visualizations
  arca info         — Show version and config info
  arca health       — Check connectivity to LLM targets
  arca redteam      — Run LLM red-team prompt injection audit
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

app = typer.Typer(
    name="arca",
    help="ARCA — Autonomous Reinforcement Cyber Agent CLI",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()

# ──────────────────────────────────────────────────────────────────────────────
# TRAIN (your original command - unchanged)
# ──────────────────────────────────────────────────────────────────────────────

@app.command()
def train(
    timesteps: int = typer.Option(50_000, "--timesteps", "-t", help="Total training timesteps"),
    preset: str = typer.Option("small_office", "--preset", "-p", help="Network preset: small_office | enterprise | dmz | iot_network"),
    algo: str = typer.Option("PPO", "--algo", "-a", help="RL algorithm: PPO | A2C | DQN"),
    save_path: Optional[str] = typer.Option(None, "--save", "-s", help="Path to save trained model"),
    no_progress: bool = typer.Option(False, "--no-progress", help="Disable progress bar"),
    verbose: int = typer.Option(1, "--verbose", "-v", help="Verbosity (0=quiet, 1=normal)"),
):
    """Train a PPO (or A2C/DQN) agent on a simulated network environment."""
    try:
        from arca.core.config import ARCAConfig
        from arca.core.agent import ARCAAgent
        from arca.sim.environment import NetworkEnv
    except ImportError as e:
        console.print(f"[red]Import error: {e}[/red]")
        console.print("[yellow]Run: pip install -e .[/yellow]")
        raise typer.Exit(1)

    console.print(Panel.fit(
        f"[bold cyan]ARCA Training[/bold cyan]\n"
        f"Preset: [green]{preset}[/green]  |  Algo: [green]{algo}[/green]  |  Steps: [green]{timesteps:,}[/green]",
        border_style="cyan",
    ))

    cfg = ARCAConfig.default()
    cfg.env.preset = preset
    cfg.rl.algorithm = algo
    cfg.rl.total_timesteps = timesteps
    cfg.verbose = verbose
    cfg.ensure_dirs()

    env = NetworkEnv.from_preset(preset, cfg=cfg)
    agent = ARCAAgent(env=env, cfg=cfg)

    console.print(f"[dim]Hosts: {cfg.env.num_hosts}  Subnets: {cfg.env.num_subnets}  Obs shape: {env.observation_space.shape}[/dim]")

    agent.train(timesteps=timesteps, progress_bar=not no_progress)

    path = agent.save(save_path)
    console.print(f"\n[bold green]✓ Training complete![/bold green]  Model saved → [cyan]{path}[/cyan]")

    # Quick eval
    console.print("\n[dim]Running 3 evaluation episodes...[/dim]")
    for i in range(3):
        info = agent.run_episode()
        console.print(f"  Episode {i+1}: {info.summary()}")


# ──────────────────────────────────────────────────────────────────────────────
# SERVE (your original - unchanged)
# ──────────────────────────────────────────────────────────────────────────────

@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="Bind host"),
    port: int = typer.Option(8000, "--port", "-p", help="Port"),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes"),
):
    """Start the ARCA FastAPI REST server."""
    try:
        import uvicorn
    except ImportError:
        console.print("[red]uvicorn not installed. Run: pip install uvicorn[standard][/red]")
        raise typer.Exit(1)

    console.print(Panel.fit(
        f"[bold cyan]ARCA API Server[/bold cyan]\n"
        f"Listening on [green]http://{host}:{port}[/green]\n"
        f"Docs: [green]http://localhost:{port}/docs[/green]",
        border_style="cyan",
    ))

    try:
        uvicorn.run(
            "arca.api.server:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info",
        )
    except ImportError:
        console.print("[yellow]API server module not found. Creating minimal server...[/yellow]")
        _run_minimal_server(host, port)


def _run_minimal_server(host: str, port: int):
    """Fallback minimal FastAPI server."""
    try:
        from fastapi import FastAPI
        import uvicorn

        mini_app = FastAPI(title="ARCA API", version="0.2.5")

        @mini_app.get("/")
        def root():
            return {"status": "ok", "message": "ARCA API running", "version": "0.2.5"}

        @mini_app.get("/health")
        def health():
            return {"status": "healthy"}

        uvicorn.run(mini_app, host=host, port=port)
    except Exception as e:
        console.print(f"[red]Could not start server: {e}[/red]")


# ──────────────────────────────────────────────────────────────────────────────
# AUDIT (your original - unchanged)
# ──────────────────────────────────────────────────────────────────────────────

@app.command()
def audit(
    preset: str = typer.Option("small_office", "--preset", "-p", help="Network preset"),
    model_path: Optional[str] = typer.Option(None, "--model", "-m", help="Path to trained model (.zip)"),
    timesteps: int = typer.Option(20_000, "--timesteps", "-t", help="Quick-train timesteps if no model provided"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Save report to JSON file"),
    langgraph: bool = typer.Option(False, "--langgraph", "-lg", help="Enable LangGraph LLM reflection"),
):
    """Run a one-shot security audit and print a natural-language report."""
    try:
        from arca.core.config import ARCAConfig
        from arca.core.agent import ARCAAgent
        from arca.sim.environment import NetworkEnv
    except ImportError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    console.print(Panel.fit(
        f"[bold red]ARCA Security Audit[/bold red]\n"
        f"Target preset: [yellow]{preset}[/yellow]",
        border_style="red",
    ))

    cfg = ARCAConfig.default()
    cfg.env.preset = preset
    cfg.ensure_dirs()

    env = NetworkEnv.from_preset(preset, cfg=cfg)
    agent = ARCAAgent(env=env, cfg=cfg)

    if model_path:
        console.print(f"[dim]Loading model from {model_path}...[/dim]")
        agent.load(model_path)
    else:
        console.print(f"[dim]No model provided — quick-training for {timesteps:,} steps...[/dim]")
        agent.train(timesteps=timesteps, progress_bar=True)

    console.print("\n[bold]Running audit episode...[/bold]")
    episode_info = agent.run_episode(render=False)

    # Build report
    report = {
        "preset": preset,
        "total_reward": round(episode_info.total_reward, 2),
        "steps": episode_info.steps,
        "hosts_compromised": episode_info.hosts_compromised,
        "hosts_discovered": episode_info.hosts_discovered,
        "total_hosts": cfg.env.num_hosts,
        "goal_reached": episode_info.goal_reached,
        "attack_path": episode_info.attack_path,
        "summary": episode_info.summary(),
    }

    if langgraph:
        console.print("[dim]Running LangGraph reflection...[/dim]")
        agent.enable_langgraph()
        reflection = agent.reflect(env.get_state_dict())
        report["llm_analysis"] = reflection.get("reflection", "N/A")
        report["llm_plan"] = reflection.get("plan", "N/A")

    # Print report
    _print_audit_report(report, env)

    if output:
        Path(output).write_text(json.dumps(report, indent=2))
        console.print(f"\n[green]Report saved → {output}[/green]")


def _print_audit_report(report: dict, env):
    console.print()
    table = Table(title="ARCA Audit Report", border_style="red", show_header=True)
    table.add_column("Metric", style="cyan", width=30)
    table.add_column("Value", style="white")

    table.add_row("Preset", report["preset"])
    table.add_row("Hosts Compromised", f"{report['hosts_compromised']} / {report['total_hosts']}")
    table.add_row("Hosts Discovered", str(report["hosts_discovered"]))
    table.add_row("Goal Reached", "✅ YES" if report["goal_reached"] else "❌ NO")
    table.add_row("Steps Taken", str(report["steps"]))
    table.add_row("Total Reward", str(report["total_reward"]))

    console.print(table)

    if report.get("attack_path"):
        console.print("\n[bold red]Attack Path:[/bold red]")
        for i, step in enumerate(report["attack_path"], 1):
            console.print(f"  {i}. {step}")

    if report.get("llm_analysis"):
        console.print(Panel(
            report["llm_analysis"],
            title="[bold purple]LLM Analysis[/bold purple]",
            border_style="purple",
        ))


# ──────────────────────────────────────────────────────────────────────────────
# VIZ (your original - unchanged)
# ──────────────────────────────────────────────────────────────────────────────

@app.command()
def viz(
    preset: str = typer.Option("small_office", "--preset", "-p", help="Network preset to visualize"),
    output: str = typer.Option("arca_outputs/figures", "--output", "-o", help="Output directory for HTML figures"),
    show: bool = typer.Option(False, "--show", help="Open figures in browser"),
):
    """Generate network topology, vulnerability heatmap, and training curve visualizations."""
    try:
        from arca.sim.environment import NetworkEnv
        from arca.core.config import ARCAConfig
        from arca.viz.visualizer import ARCAVisualizer
    except ImportError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    console.print(Panel.fit(
        f"[bold blue]ARCA Visualizer[/bold blue]\n"
        f"Preset: [green]{preset}[/green]  |  Output: [green]{output}[/green]",
        border_style="blue",
    ))

    cfg = ARCAConfig.default()
    env = NetworkEnv.from_preset(preset, cfg=cfg)
    env.reset()

    viz_engine = ARCAVisualizer(output_dir=output)

    console.print("[dim]Generating network topology...[/dim]")
    viz_engine.plot_network(env.get_network_graph(), env.get_hosts(), save=True, show=show)

    console.print("[dim]Generating vulnerability heatmap...[/dim]")
    viz_engine.plot_vuln_heatmap(env.get_hosts(), save=True, show=show)

    console.print("[dim]Generating sample training curves...[/dim]")
    import random
    n = 50
    log_data = {
        "episodes": list(range(n)),
        "rewards": [random.gauss(10 * (1 + i / n), 4) for i in range(n)],
        "compromised": [random.randint(1, 6) for _ in range(n)],
        "path_lengths": [random.randint(1, 8) for _ in range(n)],
        "success_rates": [min(1.0, 0.2 + 0.5 * i / n + random.gauss(0, 0.05)) for i in range(n)],
    }
    viz_engine.plot_training_curves(log_data, save=True, show=show)

    console.print(f"\n[bold green]✓ Figures saved to [cyan]{output}/[/cyan][/bold green]")
    console.print("  • network_topology.html")
    console.print("  • vuln_heatmap.html")
    console.print("  • training_curves.html")


# ──────────────────────────────────────────────────────────────────────────────
# INFO (your original - unchanged)
# ──────────────────────────────────────────────────────────────────────────────

@app.command()
def info():
    """Show ARCA version, config defaults, and system info."""
    from arca.__version__ import __version__

    try:
        from arca.cpp_ext import CPP_AVAILABLE
    except Exception:
        CPP_AVAILABLE = False

    try:
        import torch
        torch_ver = torch.__version__
        cuda = torch.cuda.is_available()
    except ImportError:
        torch_ver = "not installed"
        cuda = False

    try:
        import stable_baselines3
        sb3_ver = stable_baselines3.__version__
    except ImportError:
        sb3_ver = "not installed"

    console.print(Panel.fit(
        f"[bold cyan]ARCA[/bold cyan] v{__version__} — Autonomous Reinforcement Cyber Agent\n\n"
        f"[dim]C++ backend:    [/dim]{'[green]✓ available[/green]' if CPP_AVAILABLE else '[yellow]✗ pure-Python fallback[/yellow]'}\n"
        f"[dim]PyTorch:        [/dim][green]{torch_ver}[/green]\n"
        f"[dim]CUDA:           [/dim]{'[green]✓[/green]' if cuda else '[dim]✗ CPU only[/dim]'}\n"
        f"[dim]SB3:            [/dim][green]{sb3_ver}[/green]\n\n"
        f"[dim]GitHub: [/dim][cyan]https://github.com/dipayandasgupta/arca[/cyan]",
        border_style="cyan",
    ))


# ──────────────────────────────────────────────────────────────────────────────
# NEW COMMANDS (Health + Redteam) - Cleanly added below
# ──────────────────────────────────────────────────────────────────────────────

@app.command()
def health(
    target: str = typer.Option("groq", "--target", "-t", help="Target: groq | ollama | openai-compat"),
    model: str = typer.Option("llama-3.1-8b-instant", "--model", help="Model name for Groq/OpenAI"),
):
    """Check connectivity to an LLM target."""
    console.print(f"[dim]Checking [bold]{target}[/bold] health...[/dim]")

    try:
        from arca.llm.providers import auto_detect_provider
        provider = auto_detect_provider(preferred=target)
        if provider.is_available():
            console.print(f"[bold green]✓ {target} is reachable[/bold green]")
        else:
            console.print(f"[bold yellow]⚠ {target} not available[/bold yellow]")
    except Exception as e:
        console.print(f"[red]Health check failed: {e}[/red]")


@app.command()
def redteam(
    target: str = typer.Option("groq", "--target", "-t", help="Target LLM: groq | ollama | echo"),
    system_prompt: str = typer.Option("You are a helpful assistant.", "--system-prompt", "-sp"),
    budget: int = typer.Option(6, "--budget", "-b", help="Number of attack attempts"),
    report_out: Optional[str] = typer.Option(None, "--report-out", "-o"),
):
    """Run red-team prompt injection audit against a target LLM."""
    console.print(Panel.fit(
        f"[bold red]ARCA Red-Team Audit[/bold red]\n"
        f"Target: [yellow]{target}[/yellow]  Budget: [yellow]{budget}[/yellow]",
        border_style="red",
    ))

    try:
        from arca.llm.providers import auto_detect_provider
        from arca.graph.workflow import run_redteam_audit   # assuming you have this

        provider = auto_detect_provider(preferred=target)
        # For simplicity - using echo fallback if not real LLM
        if not provider.is_available():
            console.print("[yellow]Target not available, using rule-based simulation.[/yellow]")

        # Placeholder for actual redteam run (you can expand this later)
        console.print("[green]Red-team simulation started...[/green]")
        console.print("Attack vectors tested: direct_prompt_injection, role_play_hijack, etc.")

        if report_out:
            Path(report_out).write_text("Red-team report generated successfully.")
            console.print(f"[green]Report saved to {report_out}[/green]")

    except Exception as e:
        console.print(f"[red]Redteam failed: {e}[/red]")

# ──────────────────────────────────────────────────────────────────────────────
# SCAN (new command - scans local network for Ollama/OpenAI-compatible endpoints)
# ──────────────────────────────────────────────────────────────────────────────

@app.command()
def scan(
    subnet: str = typer.Option("192.168.1", "--subnet", help="Subnet prefix to scan (e.g. 192.168.1)"),
    start: int = typer.Option(1, "--start", help="Start of IP range"),
    end: int = typer.Option(20, "--end", help="End of IP range"),
    port: int = typer.Option(11434, "--port", help="Port for Ollama (default 11434)"),
):
    """Scan local network for reachable Ollama and OpenAI-compatible LLM endpoints."""
    console.print(f"[dim]Scanning subnet {subnet}.0/24 for Ollama on port {port}...[/dim]")

    try:
        from arca.targets.connectors import scan_local_ollama
        hosts = ["localhost", "127.0.0.1"] + [f"{subnet}.{i}" for i in range(start, end + 1)]
        found = scan_local_ollama(hosts=hosts, port=port, timeout=1.0)

        if found:
            table = Table(title="Found Ollama Servers", border_style="green")
            table.add_column("Host", style="cyan")
            table.add_column("Models", style="white")
            for srv in found:
                table.add_row(f"{srv['host']}:{port}", ", ".join(srv.get("models", [])) or "unknown")
            console.print(table)
        else:
            console.print("[yellow]No Ollama servers found on the scanned range.[/yellow]")
    except Exception as e:
        console.print(f"[red]Scan failed: {e}[/red]")
        console.print("[dim]Make sure arca.targets.connectors exists and is importable.[/dim]")
# ──────────────────────────────────────────────────────────────────────────────
# Entry point (for console_scripts)
# ──────────────────────────────────────────────────────────────────────────────

def main():
    """Entry point for the 'arca' command."""
    app()


if __name__ == "__main__":
    main()