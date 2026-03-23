"""Benchmark long-horizon forecasting models with CSV, markdown, and plots."""

from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import pandas as pd
import typer
from datasetsforecast.long_horizon import LongHorizon
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
    save_markdown_pdf,
    save_representative_forecast_plots,
)
from timebaseula import AutoTimeBase, AutoTimeBaseTrend

app = typer.Typer(
    help="Benchmark long-horizon datasets and persist CSV/markdown outputs."
)
console = Console()
os.environ.setdefault("NIXTLA_ID_AS_COL", "1")

DATASETS_DIR = Path("datasets")
LOG_PATH = Path("logs") / "benchmark_long_horizon.log"
DEFAULT_MIN_SERIES = 200
DEFAULT_MAX_SERIES = 300
DEFAULT_PLOT_LIMIT = 5
MODE_TO_FREQUENCY = {"daily": "D", "monthly": "ME"}
DATASET_ALIASES = {
    "ECL": "ECL",
    "ecl": "ECL",
    "Traffic": "TrafficL",
    "traffic": "TrafficL",
    "TrafficL": "TrafficL",
    "trafficl": "TrafficL",
}


def get_logger() -> logging.Logger:
    """Return the module logger."""
    return configure_logging("benchmark_long_horizon", LOG_PATH)


def resolve_mode(mode: str) -> str:
    """Validate and normalize the benchmark mode."""
    normalized_mode = mode.lower()
    if normalized_mode not in MODE_TO_FREQUENCY:
        msg = f"Unsupported mode: {mode}"
        raise ValueError(msg)
    return normalized_mode


def resolve_mode_defaults(mode: str) -> dict[str, int | str]:
    """Return recommended defaults for daily or monthly benchmark runs."""
    normalized_mode = resolve_mode(mode)
    if normalized_mode == "daily":
        return {
            "freq": "D",
            "horizon": 14,
            "max_steps": 50,
            "report_name": "daily",
        }
    return {
        "freq": "ME",
        "horizon": 5,
        "max_steps": 30,
        "report_name": "monthly",
    }


def resolve_dataset_group(dataset: str) -> str:
    """Resolve user-provided dataset names to datasetsforecast group names."""
    try:
        return DATASET_ALIASES[dataset]
    except KeyError as error:
        msg = f"Unsupported dataset: {dataset}"
        raise ValueError(msg) from error


def normalize_frequency(freq: str) -> str:
    """Normalize a frequency alias to the pandas code used here."""
    normalized = freq.upper()
    if normalized == "M":
        return "ME"
    if normalized not in {"D", "ME"}:
        msg = f"Unsupported frequency: {freq}"
        raise ValueError(msg)
    return normalized


def get_aggregated_dataset_path(datasets_dir: Path, dataset: str, freq: str) -> Path:
    """Return the parquet path for an aggregated dataset."""
    dataset_name = resolve_dataset_group(dataset).lower()
    suffix = "daily" if normalize_frequency(freq) == "D" else "monthly"
    return datasets_dir / f"{dataset_name}_{suffix}.parquet"


def aggregate_frame(frame: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Aggregate a raw long-format frame to the requested frequency."""
    normalized_freq = normalize_frequency(freq)
    return (
        frame.assign(ds=pd.to_datetime(frame["ds"]))
        .set_index("ds")
        .groupby("unique_id")
        .resample(normalized_freq)["y"]
        .mean()
        .reset_index()
        .sort_values(["unique_id", "ds"])
        .reset_index(drop=True)[["unique_id", "ds", "y"]]
    )


def download_raw_dataset(dataset: str) -> pd.DataFrame:
    """Download a raw dataset into a temporary folder and return its frame."""
    dataset_group = resolve_dataset_group(dataset)
    with tempfile.TemporaryDirectory(prefix="timebaseula-longhorizon-") as temp_dir:
        frame, *_ = LongHorizon.load(temp_dir, dataset_group)
    return frame[["unique_id", "ds", "y"]].copy()


def load_or_create_aggregated_dataset(
    dataset: str,
    freq: str,
    force_download: bool = False,
) -> pd.DataFrame:
    """Load an aggregated dataset from disk or create it if missing."""
    logger = get_logger()
    output_path = get_aggregated_dataset_path(DATASETS_DIR, dataset, freq)
    if output_path.exists() and not force_download:
        logger.info("Loading aggregated dataset", extra={"path": str(output_path)})
        return pd.read_parquet(output_path)

    logger.info("Creating aggregated dataset", extra={"path": str(output_path)})
    raw_frame = download_raw_dataset(dataset)
    aggregated = aggregate_frame(raw_frame, freq)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    aggregated.to_parquet(output_path, index=False)
    return aggregated


def ensure_aggregated_datasets(force_download: bool = False) -> list[Path]:
    """Create the daily and monthly parquet files required for benchmarking."""
    logger = get_logger()
    generated_paths: list[Path] = []
    for dataset in ("ECL", "TrafficL"):
        dataset_paths = {
            freq: get_aggregated_dataset_path(DATASETS_DIR, dataset, freq)
            for freq in ("D", "ME")
        }
        missing_frequencies = [
            freq
            for freq, output_path in dataset_paths.items()
            if force_download or not output_path.exists()
        ]
        if not missing_frequencies:
            generated_paths.extend(dataset_paths.values())
            continue

        logger.info(
            "Creating aggregated dataset family",
            extra={
                "dataset": dataset,
                "frequencies": ",".join(missing_frequencies),
            },
        )
        raw_frame = download_raw_dataset(dataset)
        for freq, output_path in dataset_paths.items():
            if freq in missing_frequencies:
                aggregated = aggregate_frame(raw_frame, freq)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                aggregated.to_parquet(output_path, index=False)
            generated_paths.append(output_path)
    return generated_paths


def choose_series_count(available_series: int, requested_series: int | None) -> int:
    """Choose a broad series count, preferring 200-300 series when possible."""
    if requested_series is not None:
        return min(requested_series, available_series)
    if available_series >= DEFAULT_MAX_SERIES:
        return DEFAULT_MAX_SERIES
    if available_series >= DEFAULT_MIN_SERIES:
        return available_series
    return available_series


def select_series_subset(frame: pd.DataFrame, n_series: int | None) -> pd.DataFrame:
    """Restrict the benchmark to a broad leading slice of the available series."""
    available_series = int(frame["unique_id"].nunique())
    selected_count = choose_series_count(available_series, n_series)
    selected_ids = frame["unique_id"].drop_duplicates().head(selected_count)
    return frame[frame["unique_id"].isin(selected_ids)].reset_index(drop=True)


def _season_length(freq: str) -> int:
    """Return the naive season length used by the benchmark baselines."""
    return 7 if freq == "D" else 12


def _common_neural_kwargs(horizon: int, max_steps: int) -> dict[str, Any]:
    """Return shared CPU-first kwargs for neural models."""
    return {
        "input_size": max(2 * horizon, 8),
        "max_steps": max_steps,
        "val_check_steps": max_steps,
        "learning_rate": 1e-3,
        "accelerator": "cpu",
        "devices": 1,
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "logger": False,
    }


def _auto_config(horizon: int, max_steps: int) -> dict[str, Any]:
    """Return a fixed config for the auto wrappers used in benchmarks."""
    return {
        **_common_neural_kwargs(horizon, max_steps),
        "step_size": 1,
        "random_seed": 1,
    }


def _count_auto_model_params(
    model: AutoTimeBase | AutoTimeBaseTrend,
) -> int:
    """Count parameters for the wrapped base model selected by an auto model."""
    base_kwargs = {
        key: value
        for key, value in model.config.items()
        if key not in {"h", "loss", "valid_loss"}
    }
    base_model = model.cls_model(
        h=model.h,
        loss=model.loss,
        valid_loss=model.valid_loss,
        freq=model.freq,
        **base_kwargs,
    )
    return count_params(base_model)


def _build_neural_models(
    horizon: int,
    max_steps: int,
    freq: str,
) -> tuple[list[Any], dict[str, int]]:
    """Build the benchmark neural models and their parameter counts."""
    common_kwargs = _common_neural_kwargs(horizon, max_steps)
    models = [
        DLinear(h=horizon, **common_kwargs),
        NLinear(h=horizon, **common_kwargs),
        AutoTimeBase(
            h=horizon,
            freq=freq,
            config=_auto_config(horizon, max_steps),
            num_samples=1,
            cpus=1,
            gpus=0,
            verbose=False,
        ),
        AutoTimeBaseTrend(
            h=horizon,
            freq=freq,
            config=_auto_config(horizon, max_steps),
            num_samples=1,
            cpus=1,
            gpus=0,
            verbose=False,
        ),
    ]
    param_map = {
        repr(model): (
            _count_auto_model_params(model)
            if isinstance(model, AutoTimeBase | AutoTimeBaseTrend)
            else count_params(model)
        )
        for model in models
    }
    return models, param_map


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
    freq: str,
    horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the SeasonalNaive baseline once and keep its forecast frame."""
    model = SeasonalNaive(season_length=_season_length(freq))
    start_time = time.perf_counter()
    forecast = StatsForecast(models=[model], freq=freq, verbose=False).cross_validation(
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
    freq: str,
    horizon: int,
    baseline_forecast: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Run the non-baseline StatsForecast models individually."""
    model = AutoMFLES(test_size=horizon, season_length=_season_length(freq))
    start_time = time.perf_counter()
    forecast = StatsForecast(models=[model], freq=freq, verbose=False).cross_validation(
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
    freq: str,
    horizon: int,
    max_steps: int,
    baseline_forecast: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Run the neural models individually and retain their forecasts."""
    models, param_map = _build_neural_models(
        horizon=horizon,
        max_steps=max_steps,
        freq=freq,
    )

    results: list[pd.DataFrame] = []
    forecasts: dict[str, pd.DataFrame] = {}
    for model in models:
        model_name = repr(model)
        start_time = time.perf_counter()
        forecast = NeuralForecast(models=[model], freq=freq).cross_validation(
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
                model_name=model_name,
                params=param_map[model_name],
                execution_time=execution_time,
            )
        )
        forecasts[model_name] = _trim_forecast_frame(merged_forecast, model_name)

    return pd.concat(results, ignore_index=True), forecasts


def run_benchmark_block(
    dataset: str,
    freq: str,
    horizon: int,
    n_series: int | None,
    max_steps: int,
) -> BenchmarkArtifacts:
    """Benchmark one dataset block and return results plus plot artifacts."""
    logger = get_logger()
    normalized_freq = normalize_frequency(freq)
    frame = load_or_create_aggregated_dataset(dataset, normalized_freq)
    frame = select_series_subset(frame, n_series)
    logger.info(
        "Running benchmark block",
        extra={
            "dataset": dataset,
            "frequency": normalized_freq,
            "rows": len(frame),
            "n_series": frame["unique_id"].nunique(),
        },
    )

    baseline_result, baseline_forecast = _run_baseline_benchmark(
        frame=frame,
        freq=normalized_freq,
        horizon=horizon,
    )
    neural_results, neural_forecasts = _run_neural_benchmark(
        frame=frame,
        freq=normalized_freq,
        horizon=horizon,
        max_steps=max_steps,
        baseline_forecast=baseline_forecast,
    )
    stats_results, stats_forecasts = _run_stats_benchmark(
        frame=frame,
        freq=normalized_freq,
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
    results["dataset"] = resolve_dataset_group(dataset)
    results["frequency"] = normalized_freq
    ordered_columns = [
        "dataset",
        "frequency",
        "model_name",
        "mae",
        "rmse",
        "rmae",
        "params",
        "execution_time",
    ]
    forecast_frames = {
        BASELINE_MODEL_NAME: baseline_forecast,
        **neural_forecasts,
        **stats_forecasts,
    }
    return BenchmarkArtifacts(
        results_frame=results[ordered_columns],
        forecast_frames=forecast_frames,
        source_frame=frame,
    )


def benchmark_dataset(
    dataset: str,
    freq: str,
    horizon: int,
    n_series: int | None,
    max_steps: int,
) -> pd.DataFrame:
    """Benchmark one dataset/frequency block and return only the leaderboard."""
    return run_benchmark_block(
        dataset=dataset,
        freq=freq,
        horizon=horizon,
        n_series=n_series,
        max_steps=max_steps,
    ).results_frame


def render_results_table(results_frame: pd.DataFrame) -> None:
    """Render a Rich leaderboard table for the benchmark results."""
    table = Table(title="Long-horizon benchmark")
    table.add_column("Dataset", style="cyan")
    table.add_column("Freq")
    table.add_column("Model", style="magenta")
    table.add_column("MAE", justify="right", style="green")
    table.add_column("RMSE", justify="right", style="yellow")
    table.add_column("RMAE", justify="right")
    table.add_column("Params", justify="right")
    table.add_column("Exec.(s)", justify="right")

    for row in results_frame.sort_values(
        ["dataset", "frequency", "mae", "rmse"]
    ).itertuples():
        table.add_row(
            row.dataset,
            row.frequency,
            row.model_name,
            f"{row.mae:.4f}",
            f"{row.rmse:.4f}",
            f"{row.rmae:.4f}",
            f"{int(row.params)}",
            f"{row.execution_time:.2f}",
        )

    console.print(table)


@app.command()
def prepare_data(
    force_download: bool = typer.Option(
        False,
        "--force-download",
        help="Recreate cached aggregated datasets.",
    ),
) -> None:
    """Create the daily and monthly aggregated parquet datasets under datasets/."""
    paths = ensure_aggregated_datasets(force_download=force_download)
    for path in paths:
        console.print(f"[green]ready[/green] {path}")


@app.command()
def report(
    input_csv: Path = typer.Option(
        ..., help="Benchmark CSV produced by the run command."
    ),
    output_md: Path = typer.Option(
        Path("docs/benchmark.md"),
        help="Markdown report output path.",
    ),
    output_pdf: Path | None = typer.Option(None, help="Optional PDF report path."),
) -> None:
    """Generate a markdown benchmark report from a benchmark CSV."""
    frame = pd.read_csv(input_csv)
    report_text = build_markdown_report(
        title="Long-horizon benchmark report",
        source_label=str(input_csv),
        results_frame=frame,
        slice_columns=["dataset", "frequency"],
        extra_sections=[
            ("Metrics", dataframe_to_markdown_table(build_metrics_frame()))
        ],
    )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(report_text, encoding="utf-8")
    if output_pdf is not None:
        save_markdown_pdf(
            markdown_text=report_text,
            output_pdf=output_pdf,
            base_dir=output_md.parent,
        )
    console.print(f"[green]report saved[/green] {output_md}")
    if output_pdf is not None:
        console.print(f"[green]pdf report saved[/green] {output_pdf}")


@app.command("run")
def run_command(
    dataset: str = typer.Option("all", help="Dataset: ECL, TrafficL, or all."),
    mode: str = typer.Option("daily", help="Benchmark mode: daily or monthly."),
    freq: str | None = typer.Option(
        None,
        help="Optional single-frequency override: D for daily or M/ME for monthly.",
    ),
    horizon: int | None = typer.Option(None, help="Forecast horizon override."),
    n_series: int | None = typer.Option(
        None,
        help="Series count override. Default uses up to 300 series.",
    ),
    max_steps: int | None = typer.Option(
        None,
        help="Max training steps override for neural models.",
    ),
    output: Path | None = typer.Option(None, help="Optional CSV output path."),
    output_md: Path | None = typer.Option(None, help="Optional markdown report path."),
    output_pdf: Path | None = typer.Option(None, help="Optional PDF report path."),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress progress messages."),
    force_download: bool = typer.Option(
        False,
        "--force-download",
        help="Recreate cached aggregated datasets.",
    ),
) -> None:
    """Run the benchmark on aggregated ECL and Traffic datasets."""
    defaults = resolve_mode_defaults(mode)
    datasets = (
        ["ECL", "TrafficL"] if dataset == "all" else [resolve_dataset_group(dataset)]
    )
    resolved_freq = normalize_frequency(defaults["freq"] if freq is None else freq)
    resolved_horizon = int(defaults["horizon"] if horizon is None else horizon)
    resolved_max_steps = int(defaults["max_steps"] if max_steps is None else max_steps)
    report_name = str(defaults["report_name"])

    if output is None:
        output = Path(f"logs/benchmark_long_horizon_{report_name}.csv")
    if output_md is None:
        output_md = Path(f"logs/benchmark_long_horizon_{report_name}.md")
    plots_dir = output_md.parent / f"{output_md.stem}_plots"

    ensure_aggregated_datasets(force_download=force_download)
    benchmark_artifacts: list[BenchmarkArtifacts] = []
    summary_frames: list[pd.DataFrame] = []
    for current_dataset in datasets:
        if not quiet:
            console.print(
                f"[bold]Benchmarking[/bold] {current_dataset} at {resolved_freq} on CPU"
            )
        artifacts = run_benchmark_block(
            dataset=current_dataset,
            freq=resolved_freq,
            horizon=resolved_horizon,
            n_series=n_series,
            max_steps=resolved_max_steps,
        )
        benchmark_artifacts.append(artifacts)
        summary_frames.append(
            build_dataset_summary(
                frame=artifacts.source_frame,
                label=resolve_dataset_group(current_dataset),
                horizon=resolved_horizon,
                frequency=resolved_freq,
            )
        )

    results_frame = pd.concat(
        [artifact.results_frame for artifact in benchmark_artifacts],
        ignore_index=True,
    )
    render_results_table(results_frame)

    output.parent.mkdir(parents=True, exist_ok=True)
    results_frame.to_csv(output, index=False)

    saved_plots = []
    for artifact in benchmark_artifacts:
        dataset_label = str(artifact.results_frame["dataset"].iloc[0])
        frequency_label = str(artifact.results_frame["frequency"].iloc[0])
        saved_plots.extend(
            save_representative_forecast_plots(
                frame=artifact.source_frame,
                forecast_frames=artifact.forecast_frames,
                results_frame=artifact.results_frame,
                output_dir=plots_dir,
                title_prefix=f"{dataset_label} {frequency_label}",
                limit=DEFAULT_PLOT_LIMIT,
            )
        )

    report_text = build_markdown_report(
        title="Long-horizon benchmark report",
        source_label=str(output),
        results_frame=results_frame,
        slice_columns=["dataset", "frequency"],
        extra_sections=[
            ("Metrics", dataframe_to_markdown_table(build_metrics_frame())),
            (
                "Data summary",
                dataframe_to_markdown_table(
                    pd.concat(summary_frames, ignore_index=True)
                ),
            ),
            (
                "Representative forecast plots",
                "\n".join(
                    [
                        "Plots show train history, holdout targets, and model predictions.",
                        "Legend entries include RMAE and parameter counts.",
                        "",
                        build_plot_markdown(saved_plots, output_md),
                    ]
                ),
            ),
        ],
    )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(report_text, encoding="utf-8")
    if output_pdf is not None:
        save_markdown_pdf(
            markdown_text=report_text,
            output_pdf=output_pdf,
            base_dir=output_md.parent,
        )
    if not quiet:
        console.print(f"[green]saved[/green] {output}")
        console.print(f"[green]report saved[/green] {output_md}")
        if output_pdf is not None:
            console.print(f"[green]pdf report saved[/green] {output_pdf}")


if __name__ == "__main__":
    app()
