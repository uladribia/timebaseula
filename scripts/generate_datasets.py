"""Create the daily and monthly aggregated benchmark datasets under datasets/."""

from __future__ import annotations

from rich.console import Console

from scripts.benchmark_long_horizon import ensure_aggregated_datasets

console = Console()


def main() -> None:
    """Prepare the aggregated benchmark datasets once and reuse them afterwards."""
    paths = ensure_aggregated_datasets(force_download=False)
    for path in paths:
        console.print(f"[green]ready[/green] {path}")


if __name__ == "__main__":
    main()
