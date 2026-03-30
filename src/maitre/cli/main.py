"""Top-level CLI application."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Optional

import typer

app = typer.Typer(
    name="maitre",
    help="Model manager service for local ML models.",
    no_args_is_help=True,
)


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


# ------------------------------------------------------------------
# serve
# ------------------------------------------------------------------


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Bind address")] = "0.0.0.0",
    port: Annotated[int, typer.Option(help="Bind port")] = 8741,
    config: Annotated[Optional[Path], typer.Option(help="Extra config file")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Start the maitre HTTP server."""
    _setup_logging(verbose)

    import uvicorn

    from maitre.server import create_app

    application = create_app(config)
    uvicorn.run(application, host=host, port=port)


# ------------------------------------------------------------------
# Sub-commands
# ------------------------------------------------------------------

from maitre.cli.models_cmd import app as models_app  # noqa: E402
from maitre.cli.providers_cmd import app as providers_app  # noqa: E402
from maitre.cli.system_cmd import app as system_app  # noqa: E402

app.add_typer(models_app, name="models", help="Manage models.")
app.add_typer(providers_app, name="providers", help="Manage providers.")
app.add_typer(system_app, name="system", help="System information.")
