"""Create the daily and monthly aggregated benchmark datasets under datasets/."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import typer
from rich.console import Console

from devtools.benchmark_long_horizon import ensure_aggregated_datasets

app = typer.Typer(help="Prepare the cached benchmark datasets.")
console = Console()
LOG_PATH = Path("logs") / "generate_datasets.log"


def configure_logging() -> logging.Logger:
    """Configure structured rotating logs for dataset generation."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("generate_datasets")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler(LOG_PATH, maxBytes=5 * 1024 * 1024, backupCount=1)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


@app.command()
def main(
    force_download: bool = typer.Option(
        False,
        "--force-download",
        help="Recreate cached aggregated datasets.",
    ),
) -> None:
    """Prepare the aggregated benchmark datasets and log the generated paths."""
    logger = configure_logging()
    paths = ensure_aggregated_datasets(force_download=force_download)
    logger.info(
        "Prepared aggregated benchmark datasets",
        extra={"force_download": force_download, "count": len(paths)},
    )
    for path in paths:
        console.print(f"[green]ready[/green] {path}")


if __name__ == "__main__":
    app()
