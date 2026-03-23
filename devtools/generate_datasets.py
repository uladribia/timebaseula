"""Create the daily and monthly aggregated benchmark datasets under datasets/."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from devtools.benchmark_common import configure_logging
from devtools.benchmark_long_horizon import ensure_aggregated_datasets

app = typer.Typer(help="Prepare the cached benchmark datasets.")
console = Console()
LOG_PATH = Path("logs") / "generate_datasets.log"


@app.command()
def main(
    force_download: bool = typer.Option(
        False,
        "--force-download",
        help="Recreate cached aggregated datasets.",
    ),
) -> None:
    """Prepare the aggregated benchmark datasets and log the generated paths."""
    logger = configure_logging("generate_datasets", LOG_PATH)
    paths = ensure_aggregated_datasets(force_download=force_download)
    logger.info(
        "Prepared aggregated benchmark datasets",
        extra={"force_download": force_download, "count": len(paths)},
    )
    for path in paths:
        console.print(f"[green]ready[/green] {path}")


if __name__ == "__main__":
    app()
