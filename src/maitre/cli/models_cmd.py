"""CLI sub-commands: maitre models ..."""

from __future__ import annotations

from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

from maitre.cli._client import get, post

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command("list")
def list_models(
    provider: Annotated[Optional[str], typer.Option(help="Filter by provider")] = None,
    tag: Annotated[Optional[str], typer.Option(help="Filter by tag")] = None,
    loaded: Annotated[bool, typer.Option("--loaded", help="Show only loaded models")] = False,
) -> None:
    """List registered models."""
    params = []
    if provider:
        params.append(f"provider={provider}")
    if tag:
        params.append(f"tag={tag}")
    qs = f"?{'&'.join(params)}" if params else ""
    models = get(f"/api/v1/models/{qs}")

    if loaded:
        models = [m for m in models if m["status"] == "loaded"]

    table = Table(title="Models")
    table.add_column("Name", style="bold")
    table.add_column("Provider")
    table.add_column("VRAM (MB)", justify="right")
    table.add_column("Tags")
    table.add_column("Status")

    status_colors = {"loaded": "green", "loading": "yellow", "unloaded": "dim", "error": "red"}
    for m in models:
        color = status_colors.get(m["status"], "white")
        table.add_row(
            m["name"],
            m["provider"],
            str(m["vram_mb"]),
            ", ".join(m["tags"]),
            f"[{color}]{m['status']}[/{color}]",
        )
    console.print(table)


@app.command("load")
def load_model(name: str) -> None:
    """Load a model (evicts LRU models if needed)."""
    console.print(f"Loading [bold]{name}[/bold] …")
    result = post(f"/api/v1/models/{name}/load")
    console.print(f"[green]Loaded[/green] {name} (status={result.get('status', 'loaded')})")


@app.command("unload")
def unload_model(name: str) -> None:
    """Unload a model."""
    post(f"/api/v1/models/{name}/unload")
    console.print(f"[yellow]Unloaded[/yellow] {name}")


@app.command("info")
def model_info(name: str) -> None:
    """Show detailed info for a model."""
    data = get(f"/api/v1/models/{name}/status")
    spec = data["spec"]
    console.print(f"[bold]{spec['name']}[/bold]")
    console.print(f"  Provider:  {spec['provider']}")
    console.print(f"  VRAM:      {spec['vram_mb']} MB")
    console.print(f"  Tags:      {', '.join(spec.get('tags', []))}")
    console.print(f"  Params:    {spec.get('params', {})}")
    console.print(f"  Loaded:    {data['loaded']}")
    if data.get("info"):
        info = data["info"]
        console.print(f"  Loaded at: {info.get('loaded_at', 'N/A')}")
        console.print(f"  Last used: {info.get('last_accessed', 'N/A')}")
