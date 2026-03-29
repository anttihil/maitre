"""CLI sub-commands: maitre system ..."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from maitre.cli._client import get
from maitre.config import user_config_path

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command("vram")
def vram() -> None:
    """Show GPU VRAM status and model budget."""
    data = get("/api/v1/system/vram/budget")
    device = data["device"]

    table = Table(title=f"VRAM — {device['device_name']} (GPU {device['device_index']})")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Total", f"{device['total_mb']} MB")
    table.add_row("Used (device)", f"{device['used_mb']} MB")
    table.add_row("Free (device)", f"{device['free_mb']} MB")
    table.add_row("Reserved", f"{device['reserved_mb']} MB")
    table.add_row("Available for models", f"{device['available_for_models_mb']} MB")
    table.add_row("Allocated by models", f"{data['allocated_by_models_mb']} MB")
    console.print(table)

    loaded = data.get("loaded_models", [])
    if loaded:
        console.print(f"\nLoaded models: {', '.join(loaded)}")
    else:
        console.print("\nNo models currently loaded.")


@app.command("config")
def config(
    path: bool = typer.Option(False, "--path", help="Print config file path and exit"),
) -> None:
    """Show running configuration."""
    if path:
        console.print(str(user_config_path()))
        return
    data = get("/api/v1/system/config")
    from rich.pretty import pprint

    pprint(data)


@app.command("health")
def health() -> None:
    """Check if the server is running."""
    data = get("/api/v1/system/health")
    console.print(f"[green]{data.get('status', 'ok')}[/green]")
