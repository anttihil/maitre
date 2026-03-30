"""CLI sub-commands: maitre providers ..."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from maitre.cli._client import get

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command("list")
def list_providers() -> None:
    """List all registered providers."""
    providers = get("/api/v1/providers/")

    table = Table(title="Providers")
    table.add_column("Name", style="bold")
    table.add_column("Display Name")
    table.add_column("Installed")
    table.add_column("Enabled")
    table.add_column("Install Instructions")

    for p in providers:
        installed = "[green]yes[/green]" if p["installed"] else "[red]no[/red]"
        enabled = "[green]yes[/green]" if p["enabled"] else "[dim]no[/dim]"
        table.add_row(
            p["name"],
            p["display_name"],
            installed,
            enabled,
            p["install_instructions"],
        )
    console.print(table)
