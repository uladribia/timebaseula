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
from statsforecast.models import AutoMFLES, SeasonalNaive

from devtools.reporting import (
    build_html_benchmark_report,
    build_representative_forecast_sections,
)
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
MODE_TO_FREQUENCY = {"daily": "D", "monthly": "ME"}

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


def resolve_mode_defaults(mode: str) -> dict[str, int | float | str]:
    """Return recommended defaults for daily or monthly benchmark runs."""
    normalized_mode = resolve_mode(mode)
    if normalized_mode == "daily":
        return {
            "freq": "D",
            "horizon": 14,
            "max_steps": 50,
            "test_fraction": 0.2,
            "report_name": "daily",
        }
    return {
        "freq": "ME",
        "horizon": 5,
        "max_steps": 30,
        "test_fraction": 0.2,
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


def build_evaluation_target(test: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Return the full per-series holdout used for rolling evaluation.

    The benchmark evaluates the entire holdout window when it is longer than the
    forecast horizon. Short holdouts remain valid as-is.
    """
    del horizon
    return test.sort_values(["unique_id", "ds"]).reset_index(drop=True)


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


def average_cross_validation_predictions(
    forecast_frame: pd.DataFrame,
    prediction_column: str,
) -> pd.DataFrame:
    """Average overlapping cross-validation predictions per timestamp."""
    return (
        forecast_frame.groupby(["unique_id", "ds"], as_index=False)
        .agg(y=("y", "first"), **{prediction_column: (prediction_column, "mean")})
        .sort_values(["unique_id", "ds"])
        .reset_index(drop=True)
    )


def average_prediction_columns(
    forecast_frame: pd.DataFrame,
    prediction_columns: list[str],
) -> pd.DataFrame:
    """Average several prediction columns from one CV run."""
    averaged_frames = [
        average_cross_validation_predictions(
            forecast_frame[["unique_id", "ds", "y", prediction_column]].copy(),
            prediction_column,
        )
        for prediction_column in prediction_columns
    ]
    merged = averaged_frames[0]
    for frame in averaged_frames[1:]:
        prediction_column = next(
            column for column in frame.columns if column not in {"unique_id", "ds", "y"}
        )
        merged = merged.merge(
            frame[["unique_id", "ds", prediction_column]],
            on=["unique_id", "ds"],
        )
    return merged


def build_benchmark_result(
    forecast: pd.DataFrame,
    prediction_column: str,
    model_name: str,
    dataset: str,
    freq: str,
    params: int,
    train_time: float,
) -> BenchmarkResult:
    """Build a benchmark result from an averaged forecast frame."""
    mae, rmse = compute_error_metrics(
        forecast["y"].to_numpy(), forecast[prediction_column].to_numpy()
    )
    return BenchmarkResult(
        model_name=model_name,
        dataset=resolve_dataset_group(dataset),
        frequency=freq,
        mae=mae,
        rmse=rmse,
        params=params,
        train_time=train_time,
        inference_time=0.0,
    )


def run_neural_benchmark(
    train: pd.DataFrame,
    test: pd.DataFrame,
    freq: str,
    models: list[torch.nn.Module],
    dataset: str,
    horizon: int,
    refit: bool,
    return_forecast: bool = False,
) -> list[BenchmarkResult] | tuple[list[BenchmarkResult], dict[str, pd.DataFrame]]:
    """Run one joint NeuralForecast cross-validation pass for all neural models."""
    full_frame = pd.concat([train, test], ignore_index=True).sort_values(
        ["unique_id", "ds"]
    )
    holdout_size = int(test.groupby("unique_id").size().min())
    nf = NeuralForecast(models=models, freq=freq)

    start_cv = time.perf_counter()
    forecast = nf.cross_validation(
        df=full_frame,
        n_windows=None,
        val_size=horizon,
        test_size=holdout_size,
        step_size=1,
        refit=refit,
    )
    total_time = time.perf_counter() - start_cv
    if "unique_id" not in forecast.columns:
        forecast = forecast.reset_index()

    model_names = [repr(model) for model in models]
    averaged = average_prediction_columns(forecast, model_names)
    results: list[BenchmarkResult] = []
    forecast_frames: dict[str, pd.DataFrame] = {}
    for model in models:
        model_name = repr(model)
        results.append(
            build_benchmark_result(
                averaged,
                prediction_column=model_name,
                model_name=model_name,
                dataset=dataset,
                freq=freq,
                params=count_params(model),
                train_time=total_time,
            )
        )
        if return_forecast:
            forecast_frames[model_name] = averaged[
                ["unique_id", "ds", model_name]
            ].copy()
    if return_forecast:
        return results, forecast_frames
    return results


def run_statsforecast_benchmark(
    train: pd.DataFrame,
    test: pd.DataFrame,
    freq: str,
    models: list[Any],
    dataset: str,
    horizon: int,
    refit: bool,
    return_forecast: bool = False,
) -> list[BenchmarkResult] | tuple[list[BenchmarkResult], dict[str, pd.DataFrame]]:
    """Run one joint StatsForecast benchmark pass for all statistical models."""
    holdout_size = int(test.groupby("unique_id").size().min())
    statsforecast = StatsForecast(models=models, freq=freq, verbose=False)
    model_names = [repr(model) for model in models]

    start_cv = time.perf_counter()
    if not refit:
        forecast = statsforecast.forecast(df=train, h=holdout_size)
        if "unique_id" not in forecast.columns:
            forecast = forecast.reset_index()
        averaged = test[["unique_id", "ds", "y"]].merge(
            forecast[["unique_id", "ds", *model_names]],
            on=["unique_id", "ds"],
            how="inner",
        )
    else:
        full_frame = pd.concat([train, test], ignore_index=True).sort_values(
            ["unique_id", "ds"]
        )
        forecast = statsforecast.cross_validation(
            df=full_frame,
            h=horizon,
            test_size=holdout_size,
            step_size=1,
            n_windows=None,
            refit=refit,
        )
        if "unique_id" not in forecast.columns:
            forecast = forecast.reset_index()
        averaged = average_prediction_columns(forecast, model_names)
    total_time = time.perf_counter() - start_cv

    results: list[BenchmarkResult] = []
    forecast_frames: dict[str, pd.DataFrame] = {}
    for model_name in model_names:
        results.append(
            build_benchmark_result(
                averaged,
                prediction_column=model_name,
                model_name=model_name,
                dataset=dataset,
                freq=freq,
                params=0,
                train_time=0.0 if model_name == "SeasonalNaive" else total_time,
            )
        )
        if return_forecast:
            forecast_frames[model_name] = averaged[
                ["unique_id", "ds", model_name]
            ].copy()
    if return_forecast:
        return results, forecast_frames
    return results


def benchmark_configuration(
    freq: str,
    horizon: int,
    max_steps: int,
    profile: DatasetProfile,
) -> list[tuple[str, str, dict[str, Any]]]:
    """Return the list of neural model configurations for a frequency.

    The user-provided ``max_steps`` acts as a hard cap for smoke and benchmark
    runs, even when recommendation helpers would otherwise expand the budget.
    """
    training_kwargs = recommend_training_kwargs(profile, horizon, max_steps)
    model_defaults = recommend_timebase_model_kwargs(profile, horizon)
    bounded_max_steps = max(1, int(max_steps))
    bounded_val_check_steps = max(
        1,
        min(int(training_kwargs["val_check_steps"]), bounded_max_steps),
    )
    bounded_early_stop = max(
        bounded_val_check_steps,
        min(int(training_kwargs["early_stop_patience_steps"]), bounded_max_steps),
    )

    common_kwargs = {
        "input_size": int(model_defaults["input_size"]),
        "max_steps": bounded_max_steps,
        "learning_rate": float(training_kwargs["learning_rate"]),
        "early_stop_patience_steps": bounded_early_stop,
        "val_check_steps": bounded_val_check_steps,
        "accelerator": "cpu",
        "devices": 1,
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "logger": False,
        "start_padding_enabled": True,
    }
    auto_kwargs = {
        "freq": freq,
        "search_max_steps": max(1, min(10, bounded_max_steps)),
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


def run_benchmark_for_dataset(
    dataset: str,
    freq: str,
    horizon: int,
    n_series: int | None,
    max_steps: int,
    test_fraction: float = 0.2,
    refit: bool = True,
    return_forecasts: bool = False,
) -> (
    list[BenchmarkResult]
    | tuple[list[BenchmarkResult], pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]
):
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
    forecast_frames: dict[str, pd.DataFrame] = {}

    evaluation_target = build_evaluation_target(test, horizon)
    results: list[BenchmarkResult] = []

    neural_models = [
        DLinear(h=horizon, **configs[0][2]),
        NLinear(h=horizon, **configs[1][2]),
        AutoTimeBase(h=horizon, **configs[2][2]),
        AutoTimeBaseTrend(h=horizon, **configs[3][2]),
    ]
    neural_output = run_neural_benchmark(
        train,
        evaluation_target,
        normalized_freq,
        neural_models,
        dataset,
        horizon,
        refit=refit,
        return_forecast=return_forecasts,
    )
    if return_forecasts:
        neural_results, neural_forecasts = neural_output
        results.extend(neural_results)
        forecast_frames.update(neural_forecasts)
    else:
        results.extend(neural_output)

    stats_models: list[Any] = [
        SeasonalNaive(season_length=infer_season_length(normalized_freq)),
        AutoMFLES(
            test_size=horizon,
            season_length=infer_season_length(normalized_freq),
        ),
    ]

    stats_output = run_statsforecast_benchmark(
        train,
        evaluation_target,
        normalized_freq,
        stats_models,
        dataset,
        horizon,
        refit=refit,
        return_forecast=return_forecasts,
    )
    if return_forecasts:
        stats_results, stats_forecasts = stats_output
        results.extend(stats_results)
        forecast_frames.update(stats_forecasts)
    else:
        results.extend(stats_output)
    if return_forecasts:
        return results, train, test.rename(columns={"y": "y_true"}), forecast_frames
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
    return Path("logs/benchmark_long_horizon_daily.html")


def resolve_report_data_dir(
    report_data_dir: str | Path | None,
    csv_output: str | Path | None,
) -> Path | None:
    """Resolve the persisted report-data directory for benchmark artifacts."""
    if report_data_dir is not None:
        return Path(report_data_dir)
    if csv_output is None:
        return None
    csv_path = Path(csv_output)
    return csv_path.with_suffix("").with_name(f"{csv_path.stem}_report_data")


def infer_report_title(results_frame: pd.DataFrame) -> str:
    """Infer a daily/monthly report title from a single-frequency result frame."""
    frequencies = sorted(results_frame["frequency"].dropna().astype(str).unique())
    if frequencies == ["D"]:
        return "Long-horizon daily benchmark report"
    if frequencies == ["ME"]:
        return "Long-horizon monthly benchmark report"
    msg = (
        "Long-horizon reports must be generated from a single frequency regime. "
        f"Found frequencies: {frequencies}"
    )
    raise ValueError(msg)


def save_report_data(
    report_data_dir: Path,
    source_frames: list[tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]],
) -> None:
    """Persist representative-series source frames for later HTML rendering."""
    report_data_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for index, (train_frame, target_frame, forecast_frames) in enumerate(source_frames):
        block = {
            "name": f"block_{index:02d}",
            "train": f"block_{index:02d}_train.parquet",
            "target": f"block_{index:02d}_target.parquet",
            "forecasts": {},
        }
        train_frame.to_parquet(report_data_dir / block["train"], index=False)
        target_frame.to_parquet(report_data_dir / block["target"], index=False)
        for model_name, forecast_frame in forecast_frames.items():
            filename = f"block_{index:02d}_{model_name}.parquet"
            forecast_frame.to_parquet(report_data_dir / filename, index=False)
            block["forecasts"][model_name] = filename
        manifest.append(block)
    (report_data_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def load_report_data(
    report_data_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]] | None:
    """Load persisted representative-series source frames for HTML reporting."""
    manifest_path = report_data_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_parts: list[pd.DataFrame] = []
    target_parts: list[pd.DataFrame] = []
    forecast_parts: dict[str, list[pd.DataFrame]] = {}
    for block in manifest:
        train_parts.append(pd.read_parquet(report_data_dir / block["train"]))
        target_parts.append(pd.read_parquet(report_data_dir / block["target"]))
        for model_name, filename in block["forecasts"].items():
            forecast_parts.setdefault(model_name, []).append(
                pd.read_parquet(report_data_dir / filename)
            )
    return (
        pd.concat(train_parts, ignore_index=True),
        pd.concat(target_parts, ignore_index=True),
        {
            model_name: pd.concat(parts, ignore_index=True)
            for model_name, parts in forecast_parts.items()
        },
    )


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
        Path("logs/benchmark_long_horizon_daily.html"),
        help="HTML report output path. Use a daily/monthly-specific filename.",
    ),
    report_data_dir: Path | None = typer.Option(
        None,
        help=(
            "Optional directory with persisted representative-series inputs. "
            "Defaults to a sibling directory derived from the CSV path."
        ),
    ),
) -> None:
    """Generate a reusable HTML benchmark report from a benchmark CSV."""
    frame = pd.read_csv(input_csv)
    representative_sections: list[str] | None = None
    title = infer_report_title(frame)
    resolved_report_data_dir = resolve_report_data_dir(report_data_dir, input_csv)
    if resolved_report_data_dir is not None:
        loaded = load_report_data(resolved_report_data_dir)
        if loaded is not None:
            observed_frame, target_frame, forecast_frames = loaded
            representative_sections = build_representative_forecast_sections(
                observed_frame,
                target_frame,
                forecast_frames,
                slice_columns=["dataset", "frequency"],
                n_examples=5,
            )
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(
        build_html_benchmark_report(
            frame,
            title=title,
            source_label=str(input_csv),
            slice_columns=["dataset", "frequency"],
            description=(
                "Reusable Matplotlib benchmark report for the long-horizon real datasets."
            ),
            representative_sections=representative_sections,
        ),
        encoding="utf-8",
    )
    console.print(f"[green]HTML report saved[/green] {output_html}")


def run_command(
    dataset: str = typer.Option("all", help="Dataset: ECL, TrafficL, or all."),
    mode: str = typer.Option(
        "daily",
        help=(
            "Benchmark mode: daily or monthly. Daily and monthly runs are "
            "kept completely separate."
        ),
    ),
    freq: str | None = typer.Option(
        None,
        help="Optional single-frequency override: D for daily or M/ME for monthly.",
    ),
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
    report_data_dir: Path | None = typer.Option(
        None,
        help=(
            "Directory used to persist representative-series inputs so reports "
            "can be regenerated without rerunning benchmarks."
        ),
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit JSON instead of only a Rich table."
    ),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress progress messages."),
    force_download: bool = typer.Option(
        False, "--force-download", help="Recreate cached aggregated datasets."
    ),
    refit: bool = typer.Option(
        False,
        "--refit/--no-refit",
        help="Whether to refit models at each cross-validation window.",
    ),
) -> None:
    """Run the benchmark on aggregated ECL and Traffic datasets for one frequency."""
    logger = configure_logging()
    defaults = resolve_mode_defaults(mode)

    datasets = (
        ["ECL", "TrafficL"] if dataset == "all" else [resolve_dataset_group(dataset)]
    )
    resolved_freq = defaults["freq"] if freq is None else freq
    normalized_frequency = normalize_frequency(str(resolved_freq))
    expected_mode_frequency = str(defaults["freq"])
    if normalized_frequency != expected_mode_frequency:
        msg = (
            "Long-horizon benchmark runs must stay within one frequency regime. "
            f"Mode '{mode}' expects frequency {expected_mode_frequency}, got {normalized_frequency}."
        )
        raise ValueError(msg)

    resolved_horizon = int(defaults["horizon"] if horizon is None else horizon)
    resolved_max_steps = int(defaults["max_steps"] if max_steps is None else max_steps)
    resolved_test_fraction = float(defaults["test_fraction"])
    report_name = str(defaults["report_name"])
    frequencies = [normalized_frequency]

    if output is None and not json_output:
        output = Path(f"logs/benchmark_long_horizon_{report_name}.csv")
    if html_report_output is None and emit_html_report:
        html_report_output = Path(f"logs/benchmark_long_horizon_{report_name}.html")
    if report_data_dir is None and output is not None:
        report_data_dir = output.with_suffix("").with_name(f"{output.stem}_report_data")

    ensure_aggregated_datasets(force_download=force_download)

    all_results: list[BenchmarkResult] = []
    representative_source_frames: list[
        tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]
    ] = []
    resolved_report_data_dir = resolve_report_data_dir(report_data_dir, output)
    persist_report_data_enabled = (
        emit_html_report or resolved_report_data_dir is not None
    )
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
            benchmark_output = run_benchmark_for_dataset(
                dataset=current_dataset,
                freq=current_freq,
                horizon=resolved_horizon,
                n_series=n_series,
                max_steps=resolved_max_steps,
                test_fraction=resolved_test_fraction,
                refit=refit,
                return_forecasts=persist_report_data_enabled,
            )
            if persist_report_data_enabled:
                block_results, train_frame, target_frame, forecast_frames = (
                    benchmark_output
                )
                all_results.extend(block_results)
                representative_source_frames.append(
                    (
                        train_frame.assign(
                            dataset=current_dataset, frequency=current_freq
                        ),
                        target_frame.assign(
                            dataset=current_dataset, frequency=current_freq
                        ),
                        {
                            name: frame.assign(
                                dataset=current_dataset,
                                frequency=current_freq,
                            )
                            for name, frame in forecast_frames.items()
                        },
                    )
                )
            else:
                all_results.extend(benchmark_output)

    render_results_table(all_results)
    results_frame = results_to_frame(all_results)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        results_frame.to_csv(output, index=False)
        if not quiet:
            console.print(f"[green]saved[/green] {output}")

    if resolved_report_data_dir is not None and representative_source_frames:
        save_report_data(resolved_report_data_dir, representative_source_frames)
        if not quiet:
            console.print(
                f"[green]report data saved[/green] {resolved_report_data_dir}"
            )

    html_output = resolve_html_report_output(
        emit_html_report,
        html_report_output,
        output,
    )
    if html_output is not None:
        html_output.parent.mkdir(parents=True, exist_ok=True)
        representative_sections: list[str] | None = None
        if resolved_report_data_dir is not None:
            loaded = load_report_data(resolved_report_data_dir)
            if loaded is not None:
                observed_frame, target_frame, forecast_frames = loaded
                representative_sections = build_representative_forecast_sections(
                    observed_frame,
                    target_frame,
                    forecast_frames,
                    slice_columns=["dataset", "frequency"],
                )
        html_output.write_text(
            build_html_benchmark_report(
                results_frame,
                title=infer_report_title(results_frame),
                source_label=str(output)
                if output is not None
                else "current benchmark run",
                slice_columns=["dataset", "frequency"],
                description=(
                    "Reusable Matplotlib benchmark report for the long-horizon real datasets."
                ),
                representative_sections=representative_sections,
            ),
            encoding="utf-8",
        )
        if not quiet:
            console.print(f"[green]HTML report saved[/green] {html_output}")

    if json_output:
        console.print_json(
            data=json.loads(results_frame.to_json(orient="records", date_format="iso"))
        )


app.command("run")(run_command)


if __name__ == "__main__":
    app()
