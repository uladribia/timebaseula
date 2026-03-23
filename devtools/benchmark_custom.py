"""Benchmark the custom monthly dataset with simple CSV and markdown output."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import typer
from neuralforecast import NeuralForecast
from neuralforecast.models import DLinear, NLinear
from rich.console import Console
from rich.table import Table
from statsforecast import StatsForecast
from statsforecast.models import AutoMFLES, SeasonalNaive

from devtools.benchmark_common import (
    build_markdown_report,
    configure_logging,
    count_params,
    evaluate_cv_results,
)
from timebaseula import TimeBase, TimeBaseTrend

app = typer.Typer(help="Benchmark the custom dataset and persist CSV/markdown outputs.")
console = Console(stderr=True)

DATASET_PATH = Path("datasets/custom/neuralforecast_monthly.csv")
LOG_PATH = Path("logs") / "benchmark_custom.log"
DEFAULT_OUTPUT_DIR = Path("logs") / "custom_dataset_benchmark"


def get_logger() -> logging.Logger:
    """Return the module logger."""
    return configure_logging("benchmark_custom", LOG_PATH)


def load_custom_dataset(dataset_path: str | Path) -> pd.DataFrame:
    """Load the custom dataset and keep only the core forecasting columns."""
    frame = pd.read_csv(dataset_path, parse_dates=["ds"])
    required_columns = ["unique_id", "ds", "y"]
    missing_columns = [column for column in required_columns if column not in frame]
    if missing_columns:
        missing_columns_text = ", ".join(missing_columns)
        msg = f"Dataset is missing required columns: {missing_columns_text}"
        raise ValueError(msg)
    return (
        frame[required_columns]
        .sort_values(["unique_id", "ds"])
        .reset_index(
            drop=True,
        )
    )


def validate_series_lengths(frame: pd.DataFrame, horizon: int) -> None:
    """Ensure every series has enough history for the requested horizon."""
    lengths = frame.groupby("unique_id").size()
    too_short = lengths[lengths <= horizon]
    if not too_short.empty:
        example_ids = ", ".join(too_short.index.astype(str).tolist()[:5])
        msg = (
            "Some series are too short for the requested horizon. "
            f"Need more than {horizon} rows, failing examples: {example_ids}"
        )
        raise ValueError(msg)


def _common_neural_kwargs(horizon: int, max_steps: int) -> dict[str, Any]:
    """Return shared CPU-first kwargs for neural models."""
    return {
        "input_size": max(2 * horizon, 8),
        "max_steps": max_steps,
        "learning_rate": 1e-3,
        "accelerator": "cpu",
        "devices": 1,
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "logger": False,
    }


def _run_neural_benchmark(
    frame: pd.DataFrame,
    horizon: int,
    max_steps: int,
    refit: bool,
) -> pd.DataFrame:
    """Run one native NeuralForecast cross-validation pass."""
    common_kwargs = _common_neural_kwargs(horizon, max_steps)
    models = [
        DLinear(h=horizon, **common_kwargs),
        NLinear(h=horizon, **common_kwargs),
        TimeBase(h=horizon, freq="MS", max_steps=max_steps, logger=False),
        TimeBaseTrend(h=horizon, freq="MS", max_steps=max_steps, logger=False),
    ]
    start_time = time.perf_counter()
    forecast = NeuralForecast(models=models, freq="MS").cross_validation(
        df=frame,
        n_windows=1,
        val_size=horizon,
        refit=refit,
    )
    elapsed = time.perf_counter() - start_time
    if "unique_id" not in forecast.columns:
        forecast = forecast.reset_index()

    model_names = [repr(model) for model in models]
    metrics = evaluate_cv_results(forecast, model_names)
    param_map = {repr(model): count_params(model) for model in models}
    metrics["params"] = metrics["model_name"].map(param_map)
    metrics["train_time"] = elapsed
    return metrics


def _run_stats_benchmark(
    frame: pd.DataFrame,
    horizon: int,
    refit: bool,
) -> pd.DataFrame:
    """Run one native StatsForecast cross-validation pass."""
    models = [
        SeasonalNaive(season_length=12),
        AutoMFLES(test_size=horizon, season_length=12),
    ]
    start_time = time.perf_counter()
    forecast = StatsForecast(models=models, freq="MS", verbose=False).cross_validation(
        df=frame,
        h=horizon,
        n_windows=1,
        refit=refit,
    )
    elapsed = time.perf_counter() - start_time
    if "unique_id" not in forecast.columns:
        forecast = forecast.reset_index()

    model_names = [repr(model) for model in models]
    metrics = evaluate_cv_results(forecast, model_names)
    metrics["model_name"] = metrics["model_name"].replace({"AutoMFLES": "MFLES"})
    metrics["params"] = 0
    metrics["train_time"] = metrics["model_name"].map(
        {"SeasonalNaive": 0.0, "MFLES": elapsed}
    )
    return metrics


def benchmark_custom_dataset(
    dataset_path: str | Path,
    horizon: int,
    max_steps: int,
    refit: bool,
) -> pd.DataFrame:
    """Run the simplified custom benchmark and return the leaderboard."""
    logger = get_logger()
    frame = load_custom_dataset(dataset_path)
    validate_series_lengths(frame, horizon)
    logger.info(
        "Running custom benchmark",
        extra={
            "rows": len(frame),
            "n_series": frame["unique_id"].nunique(),
            "horizon": horizon,
        },
    )
    neural_results = _run_neural_benchmark(
        frame=frame,
        horizon=horizon,
        max_steps=max_steps,
        refit=refit,
    )
    stats_results = _run_stats_benchmark(frame=frame, horizon=horizon, refit=refit)
    results = pd.concat([neural_results, stats_results], ignore_index=True)
    return results[["model_name", "mae", "rmse", "params", "train_time"]].sort_values(
        ["mae", "rmse"]
    )


def render_console_table(results_frame: pd.DataFrame) -> Table:
    """Build the Rich leaderboard table shown by the CLI."""
    table = Table(title="Custom dataset benchmark leaderboard")
    table.add_column("Model")
    table.add_column("MAE", justify="right")
    table.add_column("RMSE", justify="right")
    table.add_column("Params", justify="right")
    table.add_column("Train time (s)", justify="right")

    for row in results_frame.sort_values(["mae", "rmse"]).itertuples():
        table.add_row(
            row.model_name,
            f"{row.mae:.4f}",
            f"{row.rmse:.4f}",
            str(int(row.params)),
            f"{row.train_time:.2f}",
        )
    return table


@app.command()
def main(
    dataset_path: Path = typer.Option(
        DATASET_PATH,
        help="Path to the custom dataset CSV.",
    ),
    output_dir: Path = typer.Option(
        DEFAULT_OUTPUT_DIR,
        help="Directory for CSV and markdown outputs.",
    ),
    horizon: int = typer.Option(
        12,
        min=1,
        help="Forecast horizon.",
    ),
    max_steps: int = typer.Option(
        30,
        min=1,
        help="Maximum neural training steps.",
    ),
    refit: bool = typer.Option(
        False,
        "--refit/--no-refit",
        help="Whether to refit models during cross-validation.",
    ),
    quiet: bool = typer.Option(False, help="Suppress Rich progress output."),
) -> None:
    """Run the custom dataset benchmark and save CSV/markdown outputs."""
    results_frame = benchmark_custom_dataset(
        dataset_path=dataset_path,
        horizon=horizon,
        max_steps=max_steps,
        refit=refit,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    leaderboard_path = output_dir / "leaderboard.csv"
    report_path = output_dir / "report.md"
    results_frame.to_csv(leaderboard_path, index=False)
    report_path.write_text(
        build_markdown_report(
            title="Custom dataset benchmark report",
            source_label=str(leaderboard_path),
            results_frame=results_frame,
            slice_columns=[],
        ),
        encoding="utf-8",
    )

    if not quiet:
        console.print(render_console_table(results_frame))
        console.print(f"CSV leaderboard written to [bold]{leaderboard_path}[/bold]")
        console.print(f"Markdown report written to [bold]{report_path}[/bold]")


if __name__ == "__main__":
    app()
