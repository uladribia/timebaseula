"""Prepare a Nixtla-ready long panel dataset from raw store-product sales data."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_INPUT_PATH = Path("data/danone_subset_250_pdvs_all_history.parquet.gzip")
DEFAULT_OUTPUT_DIR = Path("data/processed/danone_panel")
DEFAULT_TEST_RATIO = 0.2
DEFAULT_LOG_PATH = Path("logs/prepare_nixtla_panel.log")

REQUIRED_COLUMNS = ["fecha", "pdv", "sku", "so"]


@dataclass(frozen=True)
class SplitSummary:
    """Summary statistics for a global date-based train/test split."""

    n_rows: int
    n_series: int
    n_dates: int
    train_rows: int
    test_rows: int
    train_dates: int
    test_dates: int
    horizon: int
    cutoff_date: pd.Timestamp
    min_date: pd.Timestamp
    max_date: pd.Timestamp


def configure_logging(log_path: Path) -> logging.Logger:
    """Configure a rotating file logger for dataset preparation runs."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("prepare_nixtla_panel")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    handler = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=1)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def validate_raw_frame(frame: pd.DataFrame) -> None:
    """Validate that the raw frame contains the expected raw time-series columns."""
    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in frame.columns
    ]
    if missing_columns:
        msg = f"Missing required columns: {missing_columns}"
        raise ValueError(msg)

    if frame[REQUIRED_COLUMNS].isna().any().any():
        msg = "The raw dataset contains nulls in required columns."
        raise ValueError(msg)

    duplicate_count = int(frame.duplicated(["fecha", "pdv", "sku"]).sum())
    if duplicate_count:
        msg = f"Found {duplicate_count} duplicated (fecha, pdv, sku) rows."
        raise ValueError(msg)


def build_prepared_panel(frame: pd.DataFrame) -> pd.DataFrame:
    """Build a Nixtla-ready panel with detailed and aggregated time series."""
    validate_raw_frame(frame)
    base = frame.loc[:, REQUIRED_COLUMNS].rename(columns={"fecha": "ds", "so": "y"})

    unique_dates = pd.DataFrame({"ds": sorted(base["ds"].unique())})
    unique_pairs = base[["pdv", "sku"]].drop_duplicates().reset_index(drop=True)
    detailed_grid = unique_pairs.merge(unique_dates, how="cross")
    detailed = (
        detailed_grid.merge(
            base,
            on=["ds", "pdv", "sku"],
            how="left",
            validate="one_to_one",
        )
        .assign(
            y=lambda panel: panel["y"].fillna(0.0),
            unique_id=lambda panel: (
                panel["pdv"].astype(str) + "__" + panel["sku"].astype(str)
            ),
        )
        .loc[:, ["unique_id", "ds", "y", "pdv", "sku"]]
    )

    pdv_agg = (
        detailed.groupby(["ds", "pdv"], as_index=False)
        .agg(y=("y", "sum"))
        .assign(unique_id=lambda panel: "pdv__" + panel["pdv"].astype(str), sku=pd.NA)
        .loc[:, ["unique_id", "ds", "y", "pdv", "sku"]]
    )

    sku_agg = (
        detailed.groupby(["ds", "sku"], as_index=False)
        .agg(y=("y", "sum"))
        .assign(unique_id=lambda panel: "sku__" + panel["sku"].astype(str), pdv=pd.NA)
        .loc[:, ["unique_id", "ds", "y", "pdv", "sku"]]
    )

    total_agg = (
        detailed.groupby("ds", as_index=False)
        .agg(y=("y", "sum"))
        .assign(unique_id="total", pdv=pd.NA, sku=pd.NA)
        .loc[:, ["unique_id", "ds", "y", "pdv", "sku"]]
    )

    prepared = pd.concat([detailed, pdv_agg, sku_agg, total_agg], ignore_index=True)
    return prepared.sort_values(["unique_id", "ds"], kind="stable").reset_index(
        drop=True
    )


def split_panel_by_date_ratio(
    frame: pd.DataFrame,
    test_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, SplitSummary]:
    """Split a long panel by reserving the last ratio of unique dates for testing."""
    if not 0 < test_ratio < 1:
        msg = "test_ratio must be strictly between 0 and 1."
        raise ValueError(msg)

    unique_dates = pd.Index(sorted(pd.to_datetime(frame["ds"]).unique()))
    n_dates = len(unique_dates)
    if n_dates < 2:
        msg = "At least two unique dates are required to create a train/test split."
        raise ValueError(msg)

    test_dates = max(1, int(round(n_dates * test_ratio)))
    if test_dates >= n_dates:
        test_dates = n_dates - 1
    cutoff_date = pd.Timestamp(unique_dates[-test_dates - 1])

    train = frame.loc[frame["ds"] <= cutoff_date].copy()
    test = frame.loc[frame["ds"] > cutoff_date].copy()

    summary = SplitSummary(
        n_rows=int(len(frame)),
        n_series=int(frame["unique_id"].nunique()),
        n_dates=n_dates,
        train_rows=int(len(train)),
        test_rows=int(len(test)),
        train_dates=n_dates - test_dates,
        test_dates=test_dates,
        horizon=test_dates,
        cutoff_date=cutoff_date,
        min_date=pd.Timestamp(unique_dates[0]),
        max_date=pd.Timestamp(unique_dates[-1]),
    )
    return train.reset_index(drop=True), test.reset_index(drop=True), summary


def render_markdown_summary(summary: SplitSummary) -> str:
    """Render a markdown summary for the prepared panel dataset."""
    return f"""# Prepared daily panel dataset

## Overview
- Rows: `{summary.n_rows}`
- Series: `{summary.n_series}`
- Dates: `{summary.n_dates}`
- Train rows: `{summary.train_rows}`
- Test rows: `{summary.test_rows}`
- Train dates: `{summary.train_dates}`
- Test dates: `{summary.test_dates}`
- Horizon: `{summary.horizon}`
- Date span: `{summary.min_date.date()}` to `{summary.max_date.date()}`
- Global cutoff date: `{summary.cutoff_date.date()}`

## Notes
- Detailed `unique_id` values are built as `pdv__sku`.
- Aggregated series are also added at `pdv`, `sku`, and global `total` level.
- The split is global in time: the last `{summary.test_dates}` unique dates are reserved for test.
- This panel is ready to be consumed by `StatsForecast` and `NeuralForecast`.
"""


def prepare_dataset(
    input_path: Path,
    output_dir: Path,
    test_ratio: float,
    log_path: Path,
) -> SplitSummary:
    """Prepare the Nixtla-ready panel and persist train/test artifacts."""
    logger = configure_logging(log_path)
    logger.info("Reading raw dataset from %s", input_path)
    raw_frame = pd.read_parquet(input_path)
    logger.info(
        "Loaded raw dataset with %s rows and %s columns",
        len(raw_frame),
        len(raw_frame.columns),
    )

    prepared = build_prepared_panel(raw_frame)
    train, test, summary = split_panel_by_date_ratio(prepared, test_ratio=test_ratio)

    output_dir.mkdir(parents=True, exist_ok=True)
    full_output_path = output_dir / "panel.parquet"
    train_output_path = output_dir / "train.parquet"
    test_output_path = output_dir / "test.parquet"
    metadata_output_path = output_dir / "metadata.json"
    summary_output_path = output_dir / "README.md"

    prepared.to_parquet(full_output_path, index=False)
    train.to_parquet(train_output_path, index=False)
    test.to_parquet(test_output_path, index=False)
    metadata_output_path.write_text(
        json.dumps(asdict(summary), default=str, indent=2), encoding="utf-8"
    )
    summary_output_path.write_text(render_markdown_summary(summary), encoding="utf-8")

    logger.info("Prepared dataset written to %s", output_dir)
    return summary


def build_app() -> Any:
    """Build the Typer CLI application."""
    import typer
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    app = typer.Typer(
        help="Prepare a Nixtla-ready long panel dataset from raw parquet data."
    )

    @app.command("run")
    def run(
        input_path: Path = typer.Option(
            DEFAULT_INPUT_PATH, help="Raw parquet input path."
        ),
        output_dir: Path = typer.Option(
            DEFAULT_OUTPUT_DIR, help="Output directory for prepared artifacts."
        ),
        test_ratio: float = typer.Option(
            DEFAULT_TEST_RATIO,
            help="Approximate proportion of unique dates reserved for the test split.",
        ),
        log_path: Path = typer.Option(DEFAULT_LOG_PATH, help="Rotating log file path."),
        json_output: bool = typer.Option(
            False, "--json", help="Emit the split summary as JSON."
        ),
        quiet: bool = typer.Option(
            False, "--quiet", help="Suppress human-readable console output."
        ),
        verbose: bool = typer.Option(
            False, "--verbose", help="Show additional dataset details."
        ),
    ) -> None:
        """Prepare the daily panel and write full, train, and test parquet files."""
        console = Console(stderr=False, quiet=quiet or json_output)

        if not input_path.exists():
            raise typer.BadParameter(f"Input path does not exist: {input_path}")
        if not 0 < test_ratio < 1:
            raise typer.BadParameter("test_ratio must be strictly between 0 and 1")

        summary = prepare_dataset(
            input_path=input_path,
            output_dir=output_dir,
            test_ratio=test_ratio,
            log_path=log_path,
        )

        if json_output:
            typer.echo(json.dumps(asdict(summary), default=str, indent=2))
            return

        if quiet:
            return

        table = Table(title="Prepared panel summary")
        table.add_column("field")
        table.add_column("value")
        for key, value in asdict(summary).items():
            table.add_row(key, str(value))
        console.print(table)
        console.print(f"Prepared dataset written to {output_dir}")
        console.print(f"Log written to {log_path}")

        if verbose:
            console.print(
                Panel(
                    "\n".join(
                        [
                            f"Input: {input_path}",
                            f"Panel: {output_dir / 'panel.parquet'}",
                            f"Train: {output_dir / 'train.parquet'}",
                            f"Test: {output_dir / 'test.parquet'}",
                            f"Metadata: {output_dir / 'metadata.json'}",
                        ]
                    ),
                    title="Artifacts",
                )
            )

    return app


if __name__ == "__main__":
    build_app()()
