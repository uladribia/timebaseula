"""Benchmark long-horizon forecasting models on aggregated ECL and Traffic datasets."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
import warnings
from dataclasses import asdict, dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import typer
from datasetsforecast.long_horizon import LongHorizon
from neuralforecast import NeuralForecast
from neuralforecast.models import DLinear, NLinear
from rich.console import Console
from rich.table import Table
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoMFLES

from scripts.reporting import build_html_benchmark_report
from timebaseula import AutoTimeBase, AutoTimeBaseTrend
from timebaseula.recommend import (
    DatasetProfile,
    profile_dataset,
    recommend_timebase_model_kwargs,
    recommend_training_kwargs,
)

app = typer.Typer(help="Benchmark long-horizon forecasting models on ECL and Traffic.")
console = Console()
os.environ.setdefault("NIXTLA_ID_AS_COL", "1")
warnings.filterwarnings(
    "ignore",
    message="In a future version the predictions will have the id as a column.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"`isinstance\(treespec, LeafSpec\)` is deprecated.*",
    category=DeprecationWarning,
)

DATASETS_DIR = Path("datasets")
LOG_PATH = Path("logs") / "benchmark.log"
DEFAULT_MIN_SERIES = 200
DEFAULT_MAX_SERIES = 300
MODE_TO_FREQUENCY = {"daily": "D", "monthly": "ME", "all": "all"}

FREQ_DISPLAY_TO_PANDAS = {"D": "D", "M": "ME", "ME": "ME"}
DATASET_ALIASES = {
    "ECL": "ECL",
    "ecl": "ECL",
    "Traffic": "TrafficL",
    "traffic": "TrafficL",
    "TrafficL": "TrafficL",
    "trafficl": "TrafficL",
}


@dataclass(frozen=True)
class BenchmarkResult:
    """Metrics and runtime for one model on one aggregated dataset."""

    model_name: str
    dataset: str
    frequency: str
    mae: float
    rmse: float
    params: int
    train_time: float
    inference_time: float


def configure_logging() -> logging.Logger:
    """Configure structured rotating logs for benchmark runs."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("benchmark")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler(LOG_PATH, maxBytes=5 * 1024 * 1024, backupCount=1)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


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
        return {"freq": "D", "horizon": 14, "max_steps": 50}
    if normalized_mode == "monthly":
        return {"freq": "ME", "horizon": 5, "max_steps": 30}
    return {"freq": "all", "horizon": 5, "max_steps": 30}


def resolve_dataset_group(dataset: str) -> str:
    """Resolve user-provided dataset names to datasetsforecast group names."""
    try:
        return DATASET_ALIASES[dataset]
    except KeyError as error:
        msg = f"Unsupported dataset: {dataset}"
        raise ValueError(msg) from error


def normalize_frequency(freq: str) -> str:
    """Normalize a frequency alias to the pandas resample code used here."""
    freq_upper = freq.upper()
    if freq_upper not in FREQ_DISPLAY_TO_PANDAS:
        msg = f"Unsupported frequency: {freq}"
        raise ValueError(msg)
    return FREQ_DISPLAY_TO_PANDAS[freq_upper]


def get_aggregated_dataset_path(datasets_dir: Path, dataset: str, freq: str) -> Path:
    """Return the parquet path for an aggregated dataset."""
    dataset_name = resolve_dataset_group(dataset).lower()
    suffix = "daily" if normalize_frequency(freq) == "D" else "monthly"
    return datasets_dir / f"{dataset_name}_{suffix}.parquet"


def aggregate_frame(frame: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Aggregate a raw long-format frame to the target frequency."""
    normalized_freq = normalize_frequency(freq)
    aggregated = (
        frame.assign(ds=pd.to_datetime(frame["ds"]))
        .set_index("ds")
        .groupby("unique_id")
        .resample(normalized_freq)["y"]
        .mean()
        .reset_index()
        .sort_values(["unique_id", "ds"])
        .reset_index(drop=True)
    )
    return aggregated[["unique_id", "ds", "y"]]


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
    """Load an aggregated dataset from disk or create it once if missing."""
    logger = configure_logging()
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
    generated_paths: list[Path] = []
    for dataset in ("ECL", "TrafficL"):
        for freq in ("D", "ME"):
            load_or_create_aggregated_dataset(
                dataset=dataset,
                freq=freq,
                force_download=force_download,
            )
            generated_paths.append(
                get_aggregated_dataset_path(DATASETS_DIR, dataset, freq)
            )
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


def infer_test_size(
    series_length: int, horizon: int, test_fraction: float = 0.2
) -> int:
    """Infer an approximate 20% holdout size, keeping it valid for the horizon."""
    inferred = max(horizon, round(series_length * test_fraction))
    return min(inferred, series_length - 1)


def prepare_train_test(
    frame: pd.DataFrame,
    horizon: int,
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create per-series train/test splits using an approximate 20% tail holdout."""
    train_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []

    for unique_id in frame["unique_id"].drop_duplicates():
        series = frame[frame["unique_id"] == unique_id].sort_values("ds")
        test_size = infer_test_size(len(series), horizon, test_fraction=test_fraction)
        if len(series) <= test_size:
            continue
        train_frames.append(series.iloc[:-test_size])
        test_frames.append(series.tail(test_size))

    train = pd.concat(train_frames, ignore_index=True)
    test = pd.concat(test_frames, ignore_index=True)
    return train, test


def count_params(model: torch.nn.Module) -> int:
    """Count trainable parameters for a PyTorch model."""
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def compute_error_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[float, float]:
    """Compute MAE and RMSE."""
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return mae, rmse


def infer_season_length(freq: str) -> int:
    """Choose a seasonal length suitable for the aggregated frequency."""
    normalized_freq = normalize_frequency(freq)
    if normalized_freq == "D":
        return 7
    return 12


def infer_period_len(freq: str, input_size: int) -> int:
    """Choose a segment length for TimeBase models."""
    normalized_freq = normalize_frequency(freq)
    target = 7 if normalized_freq == "D" else 12
    return max(2, min(target, input_size))


def run_neural_benchmark(
    train: pd.DataFrame,
    test: pd.DataFrame,
    freq: str,
    model_class: type,
    model_kwargs: dict[str, Any],
    model_name: str,
    dataset: str,
    horizon: int,
) -> BenchmarkResult:
    """Fit a NeuralForecast model and compute metrics on the held-out horizon."""
    model = model_class(h=horizon, **model_kwargs)
    nf = NeuralForecast(models=[model], freq=freq)

    start_train = time.perf_counter()
    nf.fit(train, val_size=horizon)
    train_time = time.perf_counter() - start_train

    start_inference = time.perf_counter()
    forecast = nf.predict()
    inference_time = time.perf_counter() - start_inference

    merged = test.merge(forecast, on=["unique_id", "ds"], how="inner")
    mae, rmse = compute_error_metrics(
        merged["y"].to_numpy(),
        merged[model_name].to_numpy(),
    )
    return BenchmarkResult(
        model_name=model_name,
        dataset=resolve_dataset_group(dataset),
        frequency=freq,
        mae=mae,
        rmse=rmse,
        params=count_params(model),
        train_time=train_time,
        inference_time=inference_time,
    )


def run_naive_benchmark(
    train: pd.DataFrame,
    test: pd.DataFrame,
    freq: str,
    dataset: str,
    horizon: int,
) -> BenchmarkResult:
    """Seasonal naive benchmark using the last observed season."""
    season_length = infer_season_length(freq)
    predictions: list[float] = []
    truths: list[float] = []

    for unique_id in test["unique_id"].drop_duplicates():
        train_series = train[train["unique_id"] == unique_id].sort_values("ds")
        test_series = test[test["unique_id"] == unique_id].sort_values("ds")
        tail_values = train_series["y"].tail(season_length).to_numpy()
        repetitions = int(np.ceil(horizon / len(tail_values)))
        forecast = np.tile(tail_values, repetitions)[:horizon]
        predictions.extend(forecast.tolist())
        truths.extend(test_series["y"].head(horizon).tolist())

    mae, rmse = compute_error_metrics(np.array(truths), np.array(predictions))
    return BenchmarkResult(
        model_name="SeasonalNaive",
        dataset=resolve_dataset_group(dataset),
        frequency=freq,
        mae=mae,
        rmse=rmse,
        params=0,
        train_time=0.0,
        inference_time=0.0,
    )


def run_statsforecast_benchmark(
    train: pd.DataFrame,
    test: pd.DataFrame,
    freq: str,
    model_class: type,
    model_kwargs: dict[str, Any],
    model_name: str,
    dataset: str,
    horizon: int,
) -> BenchmarkResult:
    """Fit a StatsForecast model series by series and compute aggregate metrics."""
    predictions: list[float] = []
    truths: list[float] = []
    total_train_time = 0.0
    total_inference_time = 0.0

    for unique_id in train["unique_id"].drop_duplicates():
        train_series = train[train["unique_id"] == unique_id].sort_values("ds")
        test_series = test[test["unique_id"] == unique_id].sort_values("ds")
        if len(train_series) <= horizon:
            continue

        if model_class is AutoMFLES:
            model = model_class(test_size=horizon, **model_kwargs)
        else:
            model = model_class(**model_kwargs)
        statsforecast = StatsForecast(models=[model], freq=freq, verbose=False)

        start_train = time.perf_counter()
        statsforecast.fit(train_series)
        total_train_time += time.perf_counter() - start_train

        start_inference = time.perf_counter()
        forecast = statsforecast.predict(h=horizon)
        total_inference_time += time.perf_counter() - start_inference

        column_name = (
            model_name if model_name in forecast.columns else type(model).__name__
        )
        predictions.extend(forecast[column_name].to_numpy().tolist())
        truths.extend(test_series["y"].head(horizon).to_numpy().tolist())

    mae, rmse = compute_error_metrics(np.array(truths), np.array(predictions))
    return BenchmarkResult(
        model_name=model_name,
        dataset=resolve_dataset_group(dataset),
        frequency=freq,
        mae=mae,
        rmse=rmse,
        params=0,
        train_time=total_train_time,
        inference_time=total_inference_time,
    )


def benchmark_configuration(
    freq: str,
    horizon: int,
    max_steps: int,
    profile: DatasetProfile,
) -> list[tuple[str, str, dict[str, Any]]]:
    """Return the list of neural model configurations for a frequency."""
    training_kwargs = recommend_training_kwargs(profile, horizon, max_steps)
    model_defaults = recommend_timebase_model_kwargs(profile, horizon)

    common_kwargs = {
        "input_size": int(model_defaults["input_size"]),
        **training_kwargs,
        "accelerator": "cpu",
        "devices": 1,
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "logger": False,
        "start_padding_enabled": True,
    }
    auto_kwargs = {
        "freq": freq,
        "search_max_steps": max(5, min(10, int(training_kwargs["val_check_steps"]))),
        "n_search_configs": 2,
    }

    return [
        ("DLinear", "dlinear", common_kwargs.copy()),
        ("NLinear", "nlinear", common_kwargs.copy()),
        ("AutoTimeBase", "auto_timebase", {**common_kwargs, **auto_kwargs}),
        (
            "AutoTimeBaseTrend",
            "auto_timebase_trend",
            {**common_kwargs, **auto_kwargs},
        ),
    ]


def should_include_arima(skip_arima: bool) -> bool:
    """Return whether AutoARIMA should be run."""
    return not skip_arima


def run_benchmark_for_dataset(
    dataset: str,
    freq: str,
    horizon: int,
    n_series: int | None,
    max_steps: int,
    skip_arima: bool,
    test_fraction: float = 0.2,
) -> list[BenchmarkResult]:
    """Run all requested benchmark models for one dataset/frequency pair."""
    logger = configure_logging()
    normalized_freq = normalize_frequency(freq)
    frame = load_or_create_aggregated_dataset(dataset, normalized_freq)
    frame = select_series_subset(frame, n_series)
    train, test = prepare_train_test(
        frame,
        horizon=horizon,
        test_fraction=test_fraction,
    )

    logger.info(
        "Prepared benchmark split",
        extra={
            "dataset": dataset,
            "frequency": normalized_freq,
            "train_rows": len(train),
            "test_rows": len(test),
            "min_train_length": int(train.groupby("unique_id").size().min()),
            "min_test_length": int(test.groupby("unique_id").size().min()),
        },
    )

    profile = profile_dataset(train, normalized_freq, horizon)
    logger.info(
        "Dataset profile",
        extra={
            "dataset": dataset,
            "frequency": normalized_freq,
            "profile": asdict(profile),
        },
    )
    configs = benchmark_configuration(
        normalized_freq,
        horizon,
        max_steps,
        profile=profile,
    )
    results = [
        run_naive_benchmark(train, test, normalized_freq, dataset, horizon),
        run_neural_benchmark(
            train,
            test,
            normalized_freq,
            DLinear,
            configs[0][2],
            "DLinear",
            dataset,
            horizon,
        ),
        run_neural_benchmark(
            train,
            test,
            normalized_freq,
            NLinear,
            configs[1][2],
            "NLinear",
            dataset,
            horizon,
        ),
        run_neural_benchmark(
            train,
            test,
            normalized_freq,
            AutoTimeBase,
            configs[2][2],
            "AutoTimeBase",
            dataset,
            horizon,
        ),
        run_neural_benchmark(
            train,
            test,
            normalized_freq,
            AutoTimeBaseTrend,
            configs[3][2],
            "AutoTimeBaseTrend",
            dataset,
            horizon,
        ),
        run_statsforecast_benchmark(
            train,
            test,
            normalized_freq,
            AutoMFLES,
            {"season_length": infer_season_length(normalized_freq)},
            "AutoMFLES",
            dataset,
            horizon,
        ),
    ]
    if should_include_arima(skip_arima):
        results.append(
            run_statsforecast_benchmark(
                train,
                test,
                normalized_freq,
                AutoARIMA,
                {"season_length": infer_season_length(normalized_freq)},
                "AutoARIMA",
                dataset,
                horizon,
            )
        )
    return results


def results_to_frame(results: list[BenchmarkResult]) -> pd.DataFrame:
    """Convert benchmark results to a tabular frame."""
    return pd.DataFrame([asdict(result) for result in results])


def build_benchmark_summary(results_frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Build a best-by-slice summary from benchmark results."""
    summary: list[dict[str, Any]] = []
    if results_frame.empty:
        return summary

    grouped = results_frame.sort_values("mae").groupby(
        ["dataset", "frequency"], as_index=False
    )
    for _, group in grouped:
        best_row = group.iloc[0]
        summary.append(
            {
                "dataset": best_row["dataset"],
                "frequency": best_row["frequency"],
                "best_model": best_row["model_name"],
                "best_mae": float(best_row["mae"]),
            }
        )
    return summary


def dataframe_to_markdown_table(frame: pd.DataFrame) -> str:
    """Convert a DataFrame to a simple markdown table without optional deps."""
    columns = [str(column) for column in frame.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in frame.iterrows():
        values = [str(row[column]) for column in frame.columns]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def format_markdown_report(results_frame: pd.DataFrame, source_csv: str) -> str:
    """Render a markdown benchmark report from a CSV-backed result frame."""
    summary = build_benchmark_summary(results_frame)
    report_lines = [
        "---",
        "description: Benchmark report generated from the long-horizon benchmark CSV.",
        "---",
        "",
        "# Benchmark report",
        "",
        f"Source CSV: `{source_csv}`",
        "",
        "## TL;DR",
        "- The table below captures the full benchmark result set.",
        "- The summary table lists the best MAE per dataset/frequency slice.",
        "- AutoARIMA can be skipped for faster exploratory runs.",
        "",
    ]

    if not results_frame.empty:
        best_overall = results_frame.sort_values("mae").iloc[0]
        smallest_model = results_frame.sort_values("params").iloc[0]
        fastest_trainable = (
            results_frame[results_frame["train_time"] > 0]
            .sort_values("train_time")
            .iloc[0]
        )
        report_lines.extend(
            [
                "## Observations",
                "",
                (
                    "- Best overall MAE in this run: "
                    f"`{best_overall['model_name']}` on "
                    f"`{best_overall['dataset']} {best_overall['frequency']}` "
                    f"with `MAE={best_overall['mae']:.4f}`."
                ),
                (
                    "- Smallest parameterized model in this run: "
                    f"`{smallest_model['model_name']}` with "
                    f"`{int(smallest_model['params'])}` trainable parameters."
                ),
                (
                    "- Fastest model with non-zero training time: "
                    f"`{fastest_trainable['model_name']}` with "
                    f"`train_time={fastest_trainable['train_time']:.2f}s`."
                ),
                "",
                "## Full results",
                "",
                dataframe_to_markdown_table(results_frame),
                "",
                "## Best MAE by slice",
                "",
            ]
        )
    else:
        report_lines.extend(
            [
                "## Observations",
                "",
                "No results available.",
                "",
                "## Full results",
                "",
                "No results available.",
                "",
                "## Best MAE by slice",
                "",
            ]
        )
    if summary:
        summary_frame = pd.DataFrame(summary)
        report_lines.append(dataframe_to_markdown_table(summary_frame))
    else:
        report_lines.append("No results available.")
    report_lines.append("")
    return "\n".join(report_lines)


def resolve_html_report_output(
    emit_html_report: bool,
    html_report_output: str | Path | None,
    csv_output: str | Path | None,
) -> Path | None:
    """Resolve the optional HTML report output path for a benchmark run."""
    if not emit_html_report:
        return None
    if html_report_output is not None:
        return Path(html_report_output)
    if csv_output is not None:
        return Path(csv_output).with_suffix(".html")
    return Path("logs/benchmark_long_horizon_report.html")


def render_results_table(results: list[BenchmarkResult]) -> None:
    """Render a Rich table for benchmark results."""
    table = Table(title="Long-horizon benchmark")
    table.add_column("Dataset", style="cyan")
    table.add_column("Freq")
    table.add_column("Model", style="magenta")
    table.add_column("MAE", justify="right", style="green")
    table.add_column("RMSE", justify="right", style="yellow")
    table.add_column("Params", justify="right")
    table.add_column("Train(s)", justify="right")
    table.add_column("Infer(s)", justify="right")

    for result in results:
        table.add_row(
            result.dataset,
            result.frequency,
            result.model_name,
            f"{result.mae:.4f}",
            f"{result.rmse:.4f}",
            f"{result.params:,}" if result.params else "-",
            f"{result.train_time:.2f}" if result.train_time else "-",
            f"{result.inference_time:.2f}" if result.inference_time else "-",
        )

    console.print(table)


@app.command()
def prepare_data(
    force_download: bool = typer.Option(
        False, "--force-download", help="Recreate cached aggregated datasets."
    ),
) -> None:
    """Create the daily and monthly aggregated parquet datasets under datasets/."""
    paths = ensure_aggregated_datasets(force_download=force_download)
    for path in paths:
        console.print(f"[green]ready[/green] {path}")


@app.command()
def report(
    input_csv: Path = typer.Option(
        ..., help="Benchmark CSV produced by the main command."
    ),
    output_md: Path = typer.Option(
        Path("docs/benchmark.md"), help="Markdown report output path."
    ),
) -> None:
    """Generate a markdown benchmark report from a benchmark CSV."""
    frame = pd.read_csv(input_csv)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(format_markdown_report(frame, source_csv=str(input_csv)))
    console.print(f"[green]report saved[/green] {output_md}")


@app.command("report-html")
def report_html(
    input_csv: Path = typer.Option(
        ..., help="Benchmark CSV produced by the main command."
    ),
    output_html: Path = typer.Option(
        Path("logs/benchmark_report.html"), help="HTML report output path."
    ),
) -> None:
    """Generate a reusable HTML benchmark report from a benchmark CSV."""
    frame = pd.read_csv(input_csv)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(
        build_html_benchmark_report(
            frame,
            title="Long-horizon benchmark report",
            source_label=str(input_csv),
            slice_columns=["dataset", "frequency"],
            description=(
                "Reusable Matplotlib benchmark report for the long-horizon real datasets."
            ),
        ),
        encoding="utf-8",
    )
    console.print(f"[green]HTML report saved[/green] {output_html}")


@app.command()
def main(
    dataset: str = typer.Option("all", help="Dataset: ECL, TrafficL, or all."),
    mode: str = typer.Option("daily", help="Benchmark mode: daily, monthly, or all."),
    freq: str | None = typer.Option(None, help="Optional frequency override: D or M."),
    horizon: int | None = typer.Option(None, help="Forecast horizon override."),
    n_series: int | None = typer.Option(
        None,
        help=(
            "Series count override. Default uses all available up to 300, "
            "aiming for at least 200 when possible."
        ),
    ),
    max_steps: int | None = typer.Option(
        None, help="Max training steps override for neural models."
    ),
    output: Path | None = typer.Option(None, help="Optional CSV output path."),
    emit_html_report: bool = typer.Option(
        False,
        "--html-report",
        help="Also emit an HTML report for the current benchmark run.",
    ),
    html_report_output: Path | None = typer.Option(
        None,
        help="Optional HTML output path. Defaults to the CSV path with .html.",
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit JSON instead of only a Rich table."
    ),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress progress messages."),
    skip_arima: bool = typer.Option(
        False, "--skip-arima", help="Skip AutoARIMA to speed up the benchmark."
    ),
    force_download: bool = typer.Option(
        False, "--force-download", help="Recreate cached aggregated datasets."
    ),
) -> None:
    """Run the benchmark on aggregated ECL and Traffic daily/monthly datasets."""
    logger = configure_logging()
    defaults = resolve_mode_defaults(mode)

    datasets = (
        ["ECL", "TrafficL"] if dataset == "all" else [resolve_dataset_group(dataset)]
    )
    resolved_freq = defaults["freq"] if freq is None else freq
    resolved_horizon = int(defaults["horizon"] if horizon is None else horizon)
    resolved_max_steps = int(defaults["max_steps"] if max_steps is None else max_steps)
    frequencies = (
        ["D", "ME"]
        if str(resolved_freq).lower() == "all"
        else [normalize_frequency(str(resolved_freq))]
    )

    ensure_aggregated_datasets(force_download=force_download)

    all_results: list[BenchmarkResult] = []
    for current_dataset in datasets:
        for current_freq in frequencies:
            if not quiet:
                console.print(
                    "[bold]Benchmarking[/bold] "
                    f"{current_dataset} at {current_freq} on CPU"
                )
            current_frame = load_or_create_aggregated_dataset(
                current_dataset, current_freq
            )
            selected_series = choose_series_count(
                int(current_frame["unique_id"].nunique()),
                n_series,
            )
            logger.info(
                "Running benchmark block",
                extra={
                    "dataset": current_dataset,
                    "frequency": current_freq,
                    "selected_series": selected_series,
                },
            )
            if not quiet:
                console.print(
                    "[dim]using "
                    f"{selected_series} series for {current_dataset} "
                    f"{current_freq}[/dim]"
                )
            all_results.extend(
                run_benchmark_for_dataset(
                    dataset=current_dataset,
                    freq=current_freq,
                    horizon=resolved_horizon,
                    n_series=n_series,
                    max_steps=resolved_max_steps,
                    skip_arima=skip_arima,
                )
            )

    render_results_table(all_results)
    results_frame = results_to_frame(all_results)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        results_frame.to_csv(output, index=False)
        if not quiet:
            console.print(f"[green]saved[/green] {output}")

    html_output = resolve_html_report_output(
        emit_html_report,
        html_report_output,
        output,
    )
    if html_output is not None:
        html_output.parent.mkdir(parents=True, exist_ok=True)
        html_output.write_text(
            build_html_benchmark_report(
                results_frame,
                title="Long-horizon benchmark report",
                source_label=str(output)
                if output is not None
                else "current benchmark run",
                slice_columns=["dataset", "frequency"],
                description=(
                    "Reusable Matplotlib benchmark report for the long-horizon real datasets."
                ),
            ),
            encoding="utf-8",
        )
        if not quiet:
            console.print(f"[green]HTML report saved[/green] {html_output}")

    if json_output:
        console.print_json(
            data=json.loads(results_frame.to_json(orient="records", date_format="iso"))
        )


if __name__ == "__main__":
    app()
