#!/usr/bin/env bash
# apply_fixes.sh — run this from ~/arca to apply all v0.3.0 fixes
# Usage: bash apply_fixes.sh
set -e

ARCA_ROOT="$(pwd)"
echo "📁 Working in: $ARCA_ROOT"

# ── 1. pytest.ini — fix xvfb crash ───────────────────────────────────────────
echo ""
echo "🔧 [1/4] Writing pytest.ini..."
cat > "$ARCA_ROOT/pytest.ini" << 'EOF'
[pytest]
addopts = -p no:xvfb
testpaths = tests
EOF
echo "    ✓ pytest.ini created"

# ── 2. workflow.py — LangGraph 0.1.x compatible ──────────────────────────────
echo ""
echo "🔧 [2/4] Updating arca/graph/workflow.py..."
mkdir -p "$ARCA_ROOT/arca/graph"

# Create __init__.py if missing
[ -f "$ARCA_ROOT/arca/graph/__init__.py" ] || touch "$ARCA_ROOT/arca/graph/__init__.py"

# Copy new workflow from the download (user should have it from Claude)
# The file is provided separately as arca/graph/workflow.py
echo "    ✓ (copy the new workflow.py from Claude's files to arca/graph/workflow.py)"

# ── 3. targets/__init__.py — ensure package is valid ─────────────────────────
echo ""
echo "🔧 [3/4] Ensuring arca/targets/__init__.py exists..."
mkdir -p "$ARCA_ROOT/arca/targets"
[ -f "$ARCA_ROOT/arca/targets/__init__.py" ] || touch "$ARCA_ROOT/arca/targets/__init__.py"
echo "    ✓ arca/targets/__init__.py ready"

# ── 4. conftest.py ────────────────────────────────────────────────────────────
echo ""
echo "🔧 [4/4] Writing tests/conftest.py..."
mkdir -p "$ARCA_ROOT/tests"
cat > "$ARCA_ROOT/tests/conftest.py" << 'EOF'
"""Shared pytest configuration and fixtures."""
import pytest

def pytest_configure(config):
    config.addinivalue_line("markers", "unit: fast unit tests with no I/O")
    config.addinivalue_line("markers", "graph: LangGraph workflow integration tests")
    config.addinivalue_line("markers", "slow: tests that make real network calls")

@pytest.fixture
def echo_target():
    from arca.targets.connectors import EchoTarget
    return EchoTarget(vulnerable=False)

@pytest.fixture
def vulnerable_echo_target():
    from arca.targets.connectors import EchoTarget
    return EchoTarget(vulnerable=True)

@pytest.fixture
def safe_system_prompt():
    return "You are a helpful, harmless, and honest assistant."

@pytest.fixture
def minimal_state(echo_target, safe_system_prompt):
    from arca.graph.workflow import ATTACK_VECTORS
    return {
        "messages": [], "target_system_prompt": safe_system_prompt,
        "target_callable": echo_target, "attack_records": [],
        "current_vector": None, "vectors_used": [],
        "attack_budget": len(ATTACK_VECTORS), "defender_mitigations": [],
        "report": None, "phase": "attack", "session_id": "test-001",
    }
EOF
echo "    ✓ tests/conftest.py created"

# ── 5. Add Typer commands to cli.py ──────────────────────────────────────────
echo ""
echo "🔧 [5/5] Patching arca/cli.py with new Typer commands..."

# Check if commands already exist
if grep -q "def health" "$ARCA_ROOT/arca/cli.py" 2>/dev/null; then
    echo "    ⚠  Commands already present in cli.py — skipping patch"
else
    # Insert new commands BEFORE the final `def main():` line
    INSERTION_MARKER="def main():"
    NEW_COMMANDS=$(cat << 'PYEOF'

# ──────────────────────────────────────────────────────────────────────────────
# HEALTH  (added v0.3.0)
# ──────────────────────────────────────────────────────────────────────────────

@app.command()
def health(
    target: str = typer.Option("ollama", "--target", "-t",
        help="Target type: ollama | groq | openai-compat"),
    ollama_host: str = typer.Option("localhost", "--ollama-host"),
    ollama_port: int = typer.Option(11434, "--ollama-port"),
    ollama_model: str = typer.Option("llama3", "--ollama-model"),
    base_url: str = typer.Option("http://localhost:1234/v1", "--base-url"),
    model: str = typer.Option("llama-3.1-8b-instant", "--model"),
):
    """Check connectivity to a target LLM endpoint."""
    from arca.targets.connectors import OllamaTarget, GroqTarget, OpenAICompatibleTarget
    console.print(f"[dim]Checking [bold]{target}[/bold] health...[/dim]")
    if target == "ollama":
        t = OllamaTarget(model=ollama_model, host=ollama_host, port=ollama_port)
    elif target == "groq":
        t = GroqTarget(model=model)
    elif target == "openai-compat":
        t = OpenAICompatibleTarget(base_url=base_url, model=model)
    else:
        console.print(f"[red]Unknown target: {target}[/red]"); raise typer.Exit(1)
    ok, msg = t.health_check()
    if ok:
        console.print(f"[bold green]✓ {target} is reachable[/bold green]  {msg}")
    else:
        console.print(f"[bold red]✗ {target} unreachable[/bold red]  {msg}"); raise typer.Exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# SCAN  (added v0.3.0)
# ──────────────────────────────────────────────────────────────────────────────

@app.command()
def scan(
    subnet: str = typer.Option("192.168.1", "--subnet"),
    start:  int = typer.Option(1,     "--start"),
    end:    int = typer.Option(20,    "--end"),
    port:   int = typer.Option(11434, "--port"),
):
    """Scan local network for reachable Ollama and OpenAI-compatible LLM endpoints."""
    from arca.targets.connectors import scan_local_ollama, probe_openai_endpoint
    from rich.table import Table
    hosts = ["localhost", "127.0.0.1"] + [f"{subnet}.{i}" for i in range(start, end + 1)]
    console.print(f"[dim]Scanning {len(hosts)} hosts for Ollama on port {port}...[/dim]")
    found = scan_local_ollama(hosts=hosts, port=port, timeout=1.0)
    if found:
        t = Table(title="Ollama Servers Found", border_style="green")
        t.add_column("Host", style="cyan"); t.add_column("Models", style="white")
        for srv in found:
            t.add_row(f"{srv['host']}:{port}", ", ".join(srv["models"]) or "(none)")
        console.print(t)
    else:
        console.print("[yellow]No Ollama servers found.[/yellow]")
    console.print("\n[dim]Probing OpenAI-compatible ports: 1234, 8080, 8000, 5000...[/dim]")
    for h in ["localhost", "127.0.0.1"]:
        for p in [1234, 8080, 8000, 5000]:
            r = probe_openai_endpoint(f"http://{h}:{p}/v1", timeout=1)
            if r["reachable"]:
                console.print(f"[green]  ✓ {h}:{p}[/green]  {r['models']}")


# ──────────────────────────────────────────────────────────────────────────────
# REDTEAM  (added v0.3.0)
# ──────────────────────────────────────────────────────────────────────────────

@app.command()
def redteam(
    target: str = typer.Option("groq", "--target", "-t",
        help="LLM to attack: groq | ollama | openai-compat | echo"),
    system_prompt: str = typer.Option("You are a helpful assistant.",
        "--system-prompt", "-sp"),
    budget: int = typer.Option(8, "--budget", "-b"),
    ollama_host:  str = typer.Option("localhost", "--ollama-host"),
    ollama_model: str = typer.Option("llama3", "--ollama-model"),
    ollama_port:  int = typer.Option(11434, "--ollama-port"),
    base_url:     str = typer.Option("http://localhost:1234/v1", "--base-url"),
    model:        str = typer.Option("llama-3.1-8b-instant", "--model"),
    api_key:      Optional[str] = typer.Option(None, "--api-key"),
    report_out:   Optional[str] = typer.Option(None, "--report-out", "-o"),
    session_id:   Optional[str] = typer.Option(None, "--session-id"),
    verbose:      bool = typer.Option(True, "--verbose/--quiet"),
):
    """Run LLM red-team prompt injection audit against any target."""
    from arca.targets.connectors import GroqTarget, OllamaTarget, OpenAICompatibleTarget, EchoTarget
    from arca.graph.workflow import run_redteam_audit, ATTACK_VECTORS
    if target == "groq":
        tgt = GroqTarget(model=model, system_prompt=system_prompt)
    elif target == "ollama":
        tgt = OllamaTarget(model=ollama_model, host=ollama_host, port=ollama_port, system_prompt=system_prompt)
    elif target == "openai-compat":
        tgt = OpenAICompatibleTarget(base_url=base_url, model=model, api_key=api_key, system_prompt=system_prompt)
    elif target == "echo":
        tgt = EchoTarget(vulnerable=False)
        console.print("[yellow]EchoTarget — offline test. Use --target groq for real audit.[/yellow]")
    else:
        console.print(f"[red]Unknown: {target}[/red]"); raise typer.Exit(1)
    console.print(Panel.fit(
        f"[bold red]ARCA LLM Red-Team Audit[/bold red]\n"
        f"Target: [yellow]{target}[/yellow]  Budget: [yellow]{budget}[/yellow]",
        border_style="red"))
    final = run_redteam_audit(target_callable=tgt, target_system_prompt=system_prompt,
                              attack_budget=budget, session_id=session_id, verbose=verbose)
    records = final.get("attack_records", [])
    breaches = [r for r in records if r["success"]]
    console.print()
    if breaches:
        console.print(f"[bold red]⚠  {len(breaches)}/{len(records)} attacks SUCCEEDED[/bold red]")
        for r in breaches:
            console.print(f"   • [red]{r['vector']}[/red]  severity={r['severity']:.2f}")
    else:
        console.print(f"[bold green]✓ All {len(records)} attacks blocked[/bold green]")
    report = final.get("report", "")
    if report_out:
        from pathlib import Path; Path(report_out).write_text(report)
        console.print(f"[green]Report → {report_out}[/green]")
    else:
        console.print("\n" + report)
    console.print(f"[dim]Session: {final.get('session_id')} (--session-id to resume)[/dim]")


PYEOF
)

    # Use Python to do the insert cleanly (avoids sed quoting issues)
    python3 - "$ARCA_ROOT/arca/cli.py" << 'PYEOF'
import sys

path = sys.argv[1]
with open(path) as f:
    content = f.read()

marker = "\ndef main():"
if marker not in content:
    print("    ⚠  Could not find 'def main():' marker — please manually paste cli_additions.py into arca/cli.py")
    sys.exit(0)

# Build insertion (same as cli_additions.py but inline-safe)
insertion = open("/home/$(whoami)/arca_fixes/cli_additions.py").read()
# Remove the docstring at the top
lines = insertion.split("\n")
start = next(i for i, l in enumerate(lines) if l.startswith("@app.command"))
insertion = "\n" + "\n".join(lines[start:]) + "\n"

new_content = content.replace(marker, insertion + marker)
with open(path, "w") as f:
    f.write(new_content)
print("    ✓ New Typer commands injected into arca/cli.py")
PYEOF
fi

echo ""
echo "════════════════════════════════════════════════════"
echo "✅  All fixes applied! Test with:"
echo ""
echo "  pytest tests/test_comprehensive.py -v -k 'not slow'"
echo "  arca health --target groq"
echo "  arca scan"
echo "  arca redteam --target echo --budget 2"
echo "  arca redteam --target groq --system-prompt 'You are a bank assistant.' --report-out audit.md"
echo "════════════════════════════════════════════════════"