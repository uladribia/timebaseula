"""Benchmark the custom monthly dataset with simple CSV, markdown, and plots."""

from __future__ import annotations

import logging
import os
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
    BASELINE_MODEL_NAME,
    BenchmarkArtifacts,
    build_dataset_summary,
    build_markdown_report,
    build_metrics_frame,
    build_plot_markdown,
    configure_logging,
    count_params,
    dataframe_to_markdown_table,
    evaluate_cv_results,
    merge_baseline_forecast,
    normalize_forecast_frame,
    save_representative_forecast_plots,
)
from timebaseula import TimeBase, TimeBaseTrend

app = typer.Typer(help="Benchmark the custom dataset and persist CSV/markdown outputs.")
console = Console(stderr=True)
os.environ.setdefault("NIXTLA_ID_AS_COL", "1")

DATASET_PATH = Path("datasets/custom/neuralforecast_monthly.csv")
LOG_PATH = Path("logs") / "benchmark_custom.log"
DEFAULT_OUTPUT_DIR = Path("logs") / "custom_dataset_benchmark"
DEFAULT_PLOT_LIMIT = 5


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


def _build_result_row(
    forecast_frame: pd.DataFrame,
    model_name: str,
    params: int,
    execution_time: float,
) -> pd.DataFrame:
    """Build one result row from a forecast frame using utilsforecast metrics."""
    model_names = (
        [BASELINE_MODEL_NAME]
        if model_name == BASELINE_MODEL_NAME
        else [BASELINE_MODEL_NAME, model_name]
    )
    metrics = evaluate_cv_results(
        forecast_frame=forecast_frame,
        model_names=model_names,
        baseline_model=BASELINE_MODEL_NAME,
    )
    result = metrics[metrics["model_name"] == model_name].reset_index(drop=True)
    result["params"] = params
    result["execution_time"] = execution_time
    return result


def _trim_forecast_frame(forecast_frame: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Keep only the columns required for reporting and plotting."""
    return forecast_frame[["unique_id", "ds", "cutoff", "y", model_name]].copy()


def _run_baseline_benchmark(
    frame: pd.DataFrame,
    horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the SeasonalNaive baseline once and keep its forecast frame."""
    model = SeasonalNaive(season_length=12)
    start_time = time.perf_counter()
    forecast = StatsForecast(models=[model], freq="MS", verbose=False).cross_validation(
        df=frame,
        h=horizon,
        n_windows=1,
        refit=True,
    )
    execution_time = time.perf_counter() - start_time
    normalized_forecast = normalize_forecast_frame(forecast)
    model_name = repr(model)
    result = _build_result_row(
        forecast_frame=normalized_forecast,
        model_name=model_name,
        params=0,
        execution_time=execution_time,
    )
    return result, _trim_forecast_frame(normalized_forecast, model_name)


def _run_stats_benchmark(
    frame: pd.DataFrame,
    horizon: int,
    baseline_forecast: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Run the non-baseline StatsForecast models individually."""
    model = AutoMFLES(test_size=horizon, season_length=12)
    start_time = time.perf_counter()
    forecast = StatsForecast(models=[model], freq="MS", verbose=False).cross_validation(
        df=frame,
        h=horizon,
        n_windows=1,
        refit=True,
    )
    execution_time = time.perf_counter() - start_time

    model_name = "MFLES"
    normalized_forecast = normalize_forecast_frame(forecast).rename(
        columns={repr(model): model_name}
    )
    merged_forecast = merge_baseline_forecast(normalized_forecast, baseline_forecast)
    result = _build_result_row(
        forecast_frame=merged_forecast,
        model_name=model_name,
        params=0,
        execution_time=execution_time,
    )
    return result, {model_name: _trim_forecast_frame(merged_forecast, model_name)}


def _run_neural_benchmark(
    frame: pd.DataFrame,
    horizon: int,
    max_steps: int,
    baseline_forecast: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Run the neural models individually and retain their forecasts."""
    common_kwargs = _common_neural_kwargs(horizon, max_steps)
    model_specs = [
        (DLinear, {"h": horizon, **common_kwargs}),
        (NLinear, {"h": horizon, **common_kwargs}),
        (TimeBase, {"h": horizon, "freq": "MS", **common_kwargs}),
        (TimeBaseTrend, {"h": horizon, "freq": "MS", **common_kwargs}),
    ]

    results: list[pd.DataFrame] = []
    forecasts: dict[str, pd.DataFrame] = {}
    for model_class, model_kwargs in model_specs:
        model = model_class(**model_kwargs)
        raw_model_name = repr(model)
        start_time = time.perf_counter()
        forecast = NeuralForecast(models=[model], freq="MS").cross_validation(
            df=frame,
            n_windows=1,
            val_size=horizon,
            refit=True,
        )
        execution_time = time.perf_counter() - start_time
        normalized_forecast = normalize_forecast_frame(forecast)
        merged_forecast = merge_baseline_forecast(
            normalized_forecast, baseline_forecast
        )
        results.append(
            _build_result_row(
                forecast_frame=merged_forecast,
                model_name=raw_model_name,
                params=count_params(model),
                execution_time=execution_time,
            )
        )
        forecasts[raw_model_name] = _trim_forecast_frame(
            merged_forecast,
            raw_model_name,
        )

    return pd.concat(results, ignore_index=True), forecasts


def run_custom_benchmark(
    dataset_path: str | Path,
    horizon: int,
    max_steps: int,
) -> BenchmarkArtifacts:
    """Run the custom benchmark and return results plus plot artifacts."""
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

    baseline_result, baseline_forecast = _run_baseline_benchmark(frame, horizon)
    neural_results, neural_forecasts = _run_neural_benchmark(
        frame=frame,
        horizon=horizon,
        max_steps=max_steps,
        baseline_forecast=baseline_forecast,
    )
    stats_results, stats_forecasts = _run_stats_benchmark(
        frame=frame,
        horizon=horizon,
        baseline_forecast=baseline_forecast,
    )

    results = (
        pd.concat(
            [baseline_result, neural_results, stats_results],
            ignore_index=True,
        )
        .sort_values(["mae", "rmse"])
        .reset_index(drop=True)
    )
    forecast_frames = {
        BASELINE_MODEL_NAME: baseline_forecast,
        **neural_forecasts,
        **stats_forecasts,
    }
    return BenchmarkArtifacts(
        results_frame=results[
            ["model_name", "mae", "rmse", "rmae", "params", "execution_time"]
        ],
        forecast_frames=forecast_frames,
        source_frame=frame,
    )


def benchmark_custom_dataset(
    dataset_path: str | Path,
    horizon: int,
    max_steps: int,
) -> pd.DataFrame:
    """Run the simplified custom benchmark and return only the leaderboard."""
    return run_custom_benchmark(
        dataset_path=dataset_path,
        horizon=horizon,
        max_steps=max_steps,
    ).results_frame


def render_console_table(results_frame: pd.DataFrame) -> Table:
    """Build the Rich leaderboard table shown by the CLI."""
    table = Table(title="Custom dataset benchmark leaderboard")
    table.add_column("Model")
    table.add_column("MAE", justify="right")
    table.add_column("RMSE", justify="right")
    table.add_column("RMAE", justify="right")
    table.add_column("Params", justify="right")
    table.add_column("Exec. time (s)", justify="right")

    for row in results_frame.sort_values(["mae", "rmse"]).itertuples():
        table.add_row(
            row.model_name,
            f"{row.mae:.4f}",
            f"{row.rmse:.4f}",
            f"{row.rmae:.4f}",
            str(int(row.params)),
            f"{row.execution_time:.2f}",
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
        help="Directory for CSV, markdown, and plot outputs.",
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
    quiet: bool = typer.Option(False, help="Suppress Rich progress output."),
) -> None:
    """Run the custom dataset benchmark and save CSV, markdown, and plots."""
    artifacts = run_custom_benchmark(
        dataset_path=dataset_path,
        horizon=horizon,
        max_steps=max_steps,
    )
    results_frame = artifacts.results_frame

    output_dir.mkdir(parents=True, exist_ok=True)
    leaderboard_path = output_dir / "leaderboard.csv"
    report_path = output_dir / "report.md"
    plots_dir = output_dir / "plots"
    results_frame.to_csv(leaderboard_path, index=False)

    saved_plots = save_representative_forecast_plots(
        frame=artifacts.source_frame,
        forecast_frames=artifacts.forecast_frames,
        results_frame=results_frame,
        output_dir=plots_dir,
        title_prefix=Path(dataset_path).stem,
        limit=DEFAULT_PLOT_LIMIT,
    )
    extra_sections = [
        ("Metrics", dataframe_to_markdown_table(build_metrics_frame())),
        (
            "Data summary",
            dataframe_to_markdown_table(
                build_dataset_summary(
                    frame=artifacts.source_frame,
                    label=Path(dataset_path).stem,
                    horizon=horizon,
                    frequency="MS",
                )
            ),
        ),
        (
            "Representative forecast plots",
            "\n".join(
                [
                    "Plots show train history, holdout targets, and model predictions.",
                    "Legend entries include RMAE and parameter counts.",
                    "",
                    build_plot_markdown(saved_plots, report_path),
                ]
            ),
        ),
    ]
    report_path.write_text(
        build_markdown_report(
            title="Custom dataset benchmark report",
            source_label=str(leaderboard_path),
            results_frame=results_frame,
            slice_columns=[],
            extra_sections=extra_sections,
        ),
        encoding="utf-8",
    )

    if not quiet:
        console.print(render_console_table(results_frame))
        console.print(f"CSV leaderboard written to [bold]{leaderboard_path}[/bold]")
        console.print(f"Markdown report written to [bold]{report_path}[/bold]")


if __name__ == "__main__":
    app()
