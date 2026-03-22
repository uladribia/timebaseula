"""Benchmark forecasting models on the custom monthly dataset and render HTML output."""

from __future__ import annotations

import base64
import json
import logging
import time
from html import escape
from io import BytesIO
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import typer
from matplotlib import pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.models import DLinear, NLinear
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from rich.console import Console
from rich.table import Table
from statsforecast import StatsForecast
from statsforecast.models import AutoMFLES

from timebaseula import AutoTimeBase, AutoTimeBaseTrend
from timebaseula.recommend import (
    profile_dataset,
    recommend_timebase_model_kwargs,
    recommend_training_kwargs,
)

app = typer.Typer(help="Benchmark the custom monthly dataset and build an HTML report.")
console = Console(stderr=True)

DATASET_PATH = Path("datasets/custom/neuralforecast_monthly.csv")
LOG_PATH = Path("logs") / "custom_dataset_benchmark.log"
DEFAULT_OUTPUT_DIR = Path("logs") / "custom_dataset_benchmark"
MODEL_NAMES = (
    "SeasonalNaive",
    "MFLES",
    "AutoTimeBase",
    "AutoTimeBaseTrend",
    "NLinear",
    "DLinear",
)
MODEL_COLORS = {
    "SeasonalNaive": "#7dd3fc",
    "MFLES": "#34d399",
    "AutoTimeBase": "#f59e0b",
    "AutoTimeBaseTrend": "#f97316",
    "TimeBase": "#f59e0b",
    "TimeBaseTrend": "#f97316",
    "NLinear": "#a78bfa",
    "DLinear": "#f472b6",
    "Observed": "#e5e7eb",
    "Holdout": "#ffffff",
}
LOGO_PATH = Path("docs/img/logo_dribia_d_blanc.png")

plt.switch_backend("Agg")


def configure_logging() -> logging.Logger:
    """Configure structured rotating logs for benchmark execution."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("custom_dataset_benchmark")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler(LOG_PATH, maxBytes=5 * 1024 * 1024, backupCount=1)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


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


def prepare_train_test(
    frame: pd.DataFrame,
    horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split each series into a train prefix and a horizon-length holdout."""
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for _, series in frame.groupby("unique_id", sort=False):
        ordered = series.sort_values("ds")
        train_parts.append(ordered.iloc[:-horizon])
        test_parts.append(ordered.tail(horizon))

    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    return train, test


def build_holdout_target(test_frame: pd.DataFrame) -> pd.DataFrame:
    """Return the target holdout frame with a stable truth column name."""
    return test_frame.rename(columns={"y": "y_true"}).reset_index(drop=True)


def choose_common_model_kwargs(
    train_frame: pd.DataFrame,
    freq: str,
    horizon: int,
    max_steps: int,
) -> dict[str, Any]:
    """Choose shared neural-model parameters from generic dataset recommendations."""
    profile = profile_dataset(train_frame, freq=freq, horizon=horizon)
    model_defaults = recommend_timebase_model_kwargs(profile, horizon=horizon)
    training_defaults = recommend_training_kwargs(
        profile,
        horizon=horizon,
        max_steps=max_steps,
    )
    return {
        "input_size": int(model_defaults["input_size"]),
        "max_steps": int(training_defaults["max_steps"]),
        "learning_rate": float(training_defaults["learning_rate"]),
        "early_stop_patience_steps": int(
            training_defaults["early_stop_patience_steps"],
        ),
        "val_check_steps": int(training_defaults["val_check_steps"]),
    }


def compute_error_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[float, float]:
    """Compute MAE and RMSE for aligned arrays."""
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return mae, rmse


def count_params(model: torch.nn.Module | None) -> int:
    """Count trainable parameters for a PyTorch model."""
    if model is None:
        return 0
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def evaluate_prediction_frame(
    target_frame: pd.DataFrame,
    forecast_frame: pd.DataFrame,
    prediction_column: str,
    model_name: str,
    train_time: float,
    inference_time: float,
    params: int,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    """Compute aggregate and per-series metrics for one model."""
    merged = target_frame.merge(forecast_frame, on=["unique_id", "ds"], how="inner")
    if merged.empty:
        msg = f"No aligned predictions found for model {model_name}"
        raise ValueError(msg)

    mae, rmse = compute_error_metrics(
        merged["y_true"].to_numpy(dtype=float),
        merged[prediction_column].to_numpy(dtype=float),
    )
    per_series = (
        merged.assign(
            abs_error=lambda df: (df["y_true"] - df[prediction_column]).abs(),
            sq_error=lambda df: (df["y_true"] - df[prediction_column]) ** 2,
        )
        .groupby("unique_id", as_index=False)
        .agg(
            mae=("abs_error", "mean"),
            rmse=("sq_error", lambda s: float(np.sqrt(s.mean()))),
        )
        .assign(model_name=model_name)[["unique_id", "model_name", "mae", "rmse"]]
    )
    aggregate = {
        "model_name": model_name,
        "overall_mae": mae,
        "overall_rmse": rmse,
        "mean_series_mae": float(per_series["mae"].mean()),
        "median_series_mae": float(per_series["mae"].median()),
        "params": int(params),
        "win_count": 0,
        "train_time": train_time,
        "inference_time": inference_time,
    }
    return aggregate, per_series


def build_seasonal_naive_forecast(
    train_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    horizon: int,
    season_length: int = 12,
) -> pd.DataFrame:
    """Build a seasonal naive forecast by repeating the last seasonal block."""
    forecast_parts: list[pd.DataFrame] = []
    for unique_id, series_train in train_frame.groupby("unique_id", sort=False):
        tail_values = series_train.sort_values("ds")["y"].tail(season_length).to_numpy()
        repeats = int(np.ceil(horizon / len(tail_values)))
        predicted = np.tile(tail_values, repeats)[:horizon]
        future_index = target_frame[target_frame["unique_id"] == unique_id][
            ["ds"]
        ].copy()
        future_index["unique_id"] = unique_id
        future_index["SeasonalNaive"] = predicted
        forecast_parts.append(future_index[["unique_id", "ds", "SeasonalNaive"]])
    return pd.concat(forecast_parts, ignore_index=True)


def run_seasonal_naive_model(
    train_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    horizon: int,
    logger: logging.Logger,
) -> tuple[dict[str, float | int | str], pd.DataFrame, pd.DataFrame]:
    """Evaluate a seasonal naive baseline on the common holdout horizon."""
    start_inference = time.perf_counter()
    forecast_frame = build_seasonal_naive_forecast(
        train_frame=train_frame,
        target_frame=target_frame,
        horizon=horizon,
        season_length=12,
    )
    inference_time = time.perf_counter() - start_inference
    aggregate, per_series = evaluate_prediction_frame(
        target_frame=target_frame,
        forecast_frame=forecast_frame,
        prediction_column="SeasonalNaive",
        model_name="SeasonalNaive",
        train_time=0.0,
        inference_time=inference_time,
        params=0,
    )
    logger.info(
        "Completed seasonal naive benchmark",
        extra={
            "model_name": "SeasonalNaive",
            "overall_mae": aggregate["overall_mae"],
            "inference_time": inference_time,
        },
    )
    return aggregate, per_series, forecast_frame


def align_forecast_frame(
    forecast_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    prediction_column: str,
) -> pd.DataFrame:
    """Ensure a forecast frame has unique_id and ds aligned to the target horizon."""
    if {"unique_id", "ds", prediction_column}.issubset(forecast_frame.columns):
        return forecast_frame[["unique_id", "ds", prediction_column]].copy()
    if {"ds", prediction_column}.issubset(forecast_frame.columns) and len(
        forecast_frame
    ) == len(
        target_frame,
    ):
        aligned = target_frame[["unique_id", "ds"]].reset_index(drop=True).copy()
        aligned[prediction_column] = forecast_frame[prediction_column].to_numpy()
        return aligned
    msg = (
        f"Forecast frame for {prediction_column} is missing required alignment columns"
    )
    raise ValueError(msg)


def extract_training_curve(training_log_path: Path, model_name: str) -> pd.DataFrame:
    """Extract train/validation losses from a Lightning CSV logger."""
    resolved_path = training_log_path
    if training_log_path.is_dir():
        candidates = sorted(training_log_path.glob("version_*/metrics.csv"))
        if not candidates:
            return pd.DataFrame(columns=["model_name", "step", "split", "loss"])
        resolved_path = candidates[-1]
    if not resolved_path.exists():
        return pd.DataFrame(columns=["model_name", "step", "split", "loss"])
    metrics = pd.read_csv(resolved_path)
    value_columns = [
        column
        for column in ["train_loss_epoch", "train_loss_step", "valid_loss", "val_loss"]
        if column in metrics.columns
    ]
    if not value_columns:
        return pd.DataFrame(columns=["model_name", "step", "split", "loss"])
    long_frame = metrics[
        [column for column in ["step", *value_columns] if column in metrics.columns]
    ].copy()
    long_frame["step"] = long_frame.get("step", pd.Series(np.arange(len(long_frame))))
    melted = long_frame.melt(
        id_vars=["step"], var_name="metric", value_name="loss"
    ).dropna()
    if melted.empty:
        return pd.DataFrame(columns=["model_name", "step", "split", "loss"])
    melted["split"] = np.where(
        melted["metric"].str.contains("valid|val", case=False), "Validation", "Train"
    )
    deduped = (
        melted.sort_values(["split", "step", "metric"])
        .groupby(["split", "step"], as_index=False)
        .tail(1)
        .assign(model_name=model_name)
    )
    return deduped[["model_name", "step", "split", "loss"]].reset_index(drop=True)


def run_neural_model(
    train_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    freq: str,
    model: torch.nn.Module,
    prediction_column: str,
    logger: logging.Logger,
    output_dir: Path,
) -> tuple[dict[str, float | int | str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit a NeuralForecast model and evaluate it on the holdout horizon."""
    training_logs_dir = output_dir / "training_logs"
    checkpoints_dir = output_dir / "checkpoints"
    training_logs_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    csv_logger = CSVLogger(save_dir=str(training_logs_dir), name=prediction_column)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoints_dir / prediction_column,
        filename="best-{step}",
        monitor="ptl/val_loss",
        mode="min",
        save_top_k=1,
    )
    existing_callbacks = list(getattr(model, "trainer_kwargs", {}).get("callbacks", []))
    model.trainer_kwargs = {
        **getattr(model, "trainer_kwargs", {}),
        "logger": csv_logger,
        "callbacks": [*existing_callbacks, checkpoint_callback],
        "enable_checkpointing": True,
        "enable_progress_bar": False,
        "enable_model_summary": False,
    }
    nf = NeuralForecast(models=[model], freq=freq)

    start_train = time.perf_counter()
    nf.fit(df=train_frame, val_size=model.h)
    train_time = time.perf_counter() - start_train

    can_reload_checkpoint = not isinstance(model, (AutoTimeBase, AutoTimeBaseTrend))
    if checkpoint_callback.best_model_path and can_reload_checkpoint:
        best_model = type(model).load_from_checkpoint(
            checkpoint_callback.best_model_path
        )
        best_model.trainer_kwargs = {
            **getattr(model, "trainer_kwargs", {}),
            "logger": csv_logger,
            "enable_checkpointing": False,
            "enable_progress_bar": False,
            "enable_model_summary": False,
        }
        nf.models = [best_model]

    start_inference = time.perf_counter()
    raw_forecast_frame = nf.predict()
    inference_time = time.perf_counter() - start_inference
    forecast_frame = align_forecast_frame(
        forecast_frame=raw_forecast_frame,
        target_frame=target_frame,
        prediction_column=prediction_column,
    )

    aggregate, per_series = evaluate_prediction_frame(
        target_frame=target_frame,
        forecast_frame=forecast_frame,
        prediction_column=prediction_column,
        model_name=prediction_column,
        train_time=train_time,
        inference_time=inference_time,
        params=count_params(model),
    )
    training_curve = extract_training_curve(
        training_logs_dir / prediction_column,
        prediction_column,
    )
    logger.info(
        "Completed neural benchmark",
        extra={
            "model_name": prediction_column,
            "overall_mae": aggregate["overall_mae"],
            "train_time": train_time,
            "inference_time": inference_time,
        },
    )
    return aggregate, per_series, forecast_frame, training_curve


def run_mfles_model(
    train_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    freq: str,
    horizon: int,
    logger: logging.Logger,
) -> tuple[dict[str, float | int | str], pd.DataFrame, pd.DataFrame]:
    """Fit AutoMFLES series by series and evaluate on the common holdout horizon."""
    start_train = time.perf_counter()
    forecast_parts: list[pd.DataFrame] = []

    for unique_id, series_train in train_frame.groupby("unique_id", sort=False):
        sf = StatsForecast(
            models=[AutoMFLES(test_size=horizon, season_length=12)],
            freq=freq,
            verbose=False,
        )
        sf.fit(series_train)
        forecast = sf.predict(h=horizon)
        forecast_parts.append(forecast.assign(unique_id=unique_id))

    train_time = time.perf_counter() - start_train
    start_inference = time.perf_counter()
    forecast_frame = pd.concat(forecast_parts, ignore_index=True)
    inference_time = time.perf_counter() - start_inference

    aggregate, per_series = evaluate_prediction_frame(
        target_frame=target_frame,
        forecast_frame=forecast_frame,
        prediction_column="AutoMFLES",
        model_name="MFLES",
        train_time=train_time,
        inference_time=inference_time,
        params=0,
    )
    aggregate["model_name"] = "MFLES"
    per_series["model_name"] = "MFLES"
    forecast_frame = forecast_frame.rename(columns={"AutoMFLES": "MFLES"})
    logger.info(
        "Completed MFLES benchmark",
        extra={
            "model_name": "MFLES",
            "overall_mae": aggregate["overall_mae"],
            "train_time": train_time,
            "inference_time": inference_time,
        },
    )
    return aggregate, per_series, forecast_frame


def add_win_counts(
    aggregate_results: pd.DataFrame,
    per_series_results: pd.DataFrame,
) -> pd.DataFrame:
    """Count per-series MAE wins for each model."""
    winners = per_series_results.loc[
        per_series_results.groupby("unique_id")["mae"].idxmin()
    ]
    win_counts = winners["model_name"].value_counts()
    result = aggregate_results.copy()
    result["win_count"] = result["model_name"].map(win_counts).fillna(0).astype(int)
    return result


def add_average_ranks(
    aggregate_results: pd.DataFrame,
    per_series_results: pd.DataFrame,
) -> pd.DataFrame:
    """Add the average per-series MAE rank for each model."""
    ranked = per_series_results.copy()
    ranked["rank"] = ranked.groupby("unique_id")["mae"].rank(method="average")
    average_ranks = ranked.groupby("model_name", as_index=False)["rank"].mean()
    average_ranks = average_ranks.rename(columns={"rank": "average_rank"})
    return aggregate_results.copy().merge(
        average_ranks,
        on="model_name",
        how="left",
    )


def add_relative_mae(
    per_series_results: pd.DataFrame,
    full_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Add relative MAE using each series mean count as the denominator."""
    series_means = (
        full_frame.groupby("unique_id", as_index=False)["y"]
        .mean()
        .rename(columns={"y": "series_mean"})
    )
    result = per_series_results.merge(series_means, on="unique_id", how="left")
    result["rmae"] = result["mae"] / result["series_mean"].clip(lower=1e-12)
    return result


def add_relative_mae_summary(
    aggregate_results: pd.DataFrame,
    per_series_results: pd.DataFrame,
) -> pd.DataFrame:
    """Add mean and median relative MAE summary columns to the aggregate results."""
    relative_summary = per_series_results.groupby("model_name", as_index=False).agg(
        mean_series_rmae=("rmae", "mean"),
        median_series_rmae=("rmae", "median"),
    )
    return aggregate_results.merge(relative_summary, on="model_name", how="left")


def summarise_dataset(frame: pd.DataFrame, freq: str, horizon: int) -> dict[str, Any]:
    """Build a compact dataset summary for console and HTML output."""
    lengths = frame.groupby("unique_id").size()
    return {
        "dataset_name": "custom",
        "freq": freq,
        "horizon": horizon,
        "n_series": int(frame["unique_id"].nunique()),
        "n_rows": len(frame),
        "min_length": int(lengths.min()),
        "median_length": int(lengths.median()),
        "max_length": int(lengths.max()),
        "date_min": frame["ds"].min().strftime("%Y-%m-%d"),
        "date_max": frame["ds"].max().strftime("%Y-%m-%d"),
    }


def choose_representative_series(
    per_series_results: pd.DataFrame,
    anchor_model: str,
    n_examples: int = 5,
) -> list[str]:
    """Choose representative series spread across the anchor-model ranking."""
    anchor = per_series_results[per_series_results["model_name"] == anchor_model]
    ranked = anchor.sort_values(["mae", "unique_id"]).reset_index(drop=True)
    if ranked.empty:
        return []
    target_count = min(n_examples, len(ranked))
    positions = np.linspace(0, len(ranked) - 1, num=target_count)
    selected_ids: list[str] = []
    for position in positions:
        unique_id = str(ranked.iloc[round(position)]["unique_id"])
        if unique_id not in selected_ids:
            selected_ids.append(unique_id)
    return selected_ids


def build_logo_data_uri() -> str:
    """Return the embedded logo as a data URI."""
    encoded = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render_matplotlib_figure(fig: plt.Figure, alt_text: str) -> str:
    """Render a Matplotlib figure as an embeddable HTML image fragment."""
    buffer = BytesIO()
    fig.tight_layout()
    fig.savefig(
        buffer,
        format="png",
        dpi=160,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f'<img class="plot-image" src="data:image/png;base64,{encoded}" alt="{escape(alt_text)}">'


def build_representative_plot_sections(
    full_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    forecast_frames: dict[str, pd.DataFrame],
    representative_ids: list[str],
    history_points: int = 36,
) -> list[str]:
    """Build HTML sections with representative forecast plots."""
    sections: list[str] = []
    plot_order = ["Observed", "Holdout", *MODEL_NAMES]
    for unique_id in representative_ids:
        holdout_length = len(target_frame[target_frame["unique_id"] == unique_id])
        history = (
            full_frame[full_frame["unique_id"] == unique_id]
            .sort_values("ds")
            .tail(history_points + holdout_length)[["ds", "y"]]
            .assign(series="Observed", value=lambda df: df["y"])[
                ["ds", "series", "value"]
            ]
        )
        holdout = (
            target_frame[target_frame["unique_id"] == unique_id]
            .sort_values("ds")[["ds", "y_true"]]
            .assign(series="Holdout", value=lambda df: df["y_true"])[
                ["ds", "series", "value"]
            ]
        )
        forecast_parts = [history, holdout]
        for model_name, forecast_frame in forecast_frames.items():
            series_forecast = (
                forecast_frame[forecast_frame["unique_id"] == unique_id]
                .sort_values("ds")[["ds", model_name]]
                .assign(series=model_name, value=lambda df, c=model_name: df[c])[
                    ["ds", "series", "value"]
                ]
            )
            forecast_parts.append(series_forecast)
        plot_frame = pd.concat(forecast_parts, ignore_index=True)
        value_cap = (
            float(plot_frame["value"].quantile(0.995)) if not plot_frame.empty else 1.0
        )
        fig, ax = plt.subplots(figsize=(10.5, 3.8), facecolor="#0f172a")
        ax.set_facecolor("#0f172a")
        for series_name in plot_order:
            series_frame = plot_frame[plot_frame["series"] == series_name]
            if series_frame.empty:
                continue
            style = "--" if series_name == "Holdout" else "-"
            linewidth = 2.4 if series_name in {"Observed", "Holdout"} else 1.8
            ax.plot(
                series_frame["ds"],
                series_frame["value"],
                label=series_name,
                color=MODEL_COLORS.get(series_name, "#cbd5e1"),
                linestyle=style,
                linewidth=linewidth,
                alpha=0.95,
            )
        ax.set_title(f"Forecast plot for {unique_id}", color="#e2e8f0")
        ax.set_xlabel("Date", color="#cbd5e1")
        ax.set_ylabel("Traffic", color="#cbd5e1")
        ax.set_ylim(0, max(value_cap, 1.0))
        ax.tick_params(colors="#cbd5e1")
        ax.grid(True, alpha=0.18, color="#94a3b8")
        legend = ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper left")
        for text in legend.get_texts():
            text.set_color("#e2e8f0")
        sections.append(
            f"<figure><h3>{escape(unique_id)}</h3>{render_matplotlib_figure(fig, f'Forecast plot for {unique_id}')}</figure>",
        )
    return sections


def build_mae_distribution_section(per_series_results: pd.DataFrame) -> str:
    """Build an HTML section with relative-MAE distribution plots."""
    clamped = per_series_results.copy()
    upper_clip = float(clamped["rmae"].quantile(0.99)) if not clamped.empty else 1.0
    clamped["rmae_clamped"] = clamped["rmae"].clip(upper=upper_clip)
    fig, ax = plt.subplots(figsize=(10.5, 4.0), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")
    bins = np.linspace(0, max(upper_clip, 1e-6), 40)
    for model_name in MODEL_NAMES:
        model_values = clamped.loc[
            clamped["model_name"] == model_name, "rmae_clamped"
        ].to_numpy()
        if len(model_values) == 0:
            continue
        counts, edges = np.histogram(model_values, bins=bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.fill_between(centers, counts, alpha=0.15, color=MODEL_COLORS[model_name])
        ax.plot(
            centers,
            counts,
            linewidth=2.0,
            color=MODEL_COLORS[model_name],
            label=model_name,
        )
        ax.axvline(
            float(np.median(model_values)),
            color=MODEL_COLORS[model_name],
            linestyle="--",
            linewidth=1.6,
        )
    ax.set_title("Distribution of relative MAE across series", color="#e2e8f0")
    ax.set_xlabel("Relative MAE (MAE / mean series count)", color="#cbd5e1")
    ax.set_ylabel("Density", color="#cbd5e1")
    ax.tick_params(colors="#cbd5e1")
    ax.grid(True, alpha=0.18, color="#94a3b8")
    legend = ax.legend(ncol=3, frameon=False, fontsize=8)
    for text in legend.get_texts():
        text.set_color("#e2e8f0")
    return (
        '<section class="card">'
        "<h2>Distribution of relative MAE across series</h2>"
        "<p>Filled density-style histograms per model over relative MAE, with dashed vertical markers for each model's median RMAE.</p>"
        f"{render_matplotlib_figure(fig, 'Distribution of relative MAE across series')}"
        "<p><strong>Median RMAE</strong> markers are shown as dashed vertical lines.</p>"
        "</section>"
    )


def build_summary_section(dataset_summary: dict[str, Any]) -> str:
    """Build a compact non-tabular dataset summary section."""
    cards = "".join(
        f'<div class="kpi"><strong>{escape(str(key).replace("_", " ").title())}:</strong> {escape(str(value))}</div>'
        for key, value in dataset_summary.items()
    )
    return f'<section class="card"><h2>Dataset summary</h2><div>{cards}</div></section>'


def build_leaderboard_section(aggregate_results: pd.DataFrame) -> str:
    """Build a static HTML leaderboard table."""
    leaderboard = aggregate_results.sort_values(["overall_mae", "overall_rmse"]).copy()
    rows = []
    for row in leaderboard.itertuples():
        rows.append(
            "<tr>"
            f"<td>{escape(str(row.model_name))}</td>"
            f"<td>{row.overall_mae:.4f}</td>"
            f"<td>{row.overall_rmse:.4f}</td>"
            f"<td>{row.mean_series_rmae:.4f}</td>"
            f"<td>{row.median_series_rmae:.4f}</td>"
            f"<td>{row.average_rank:.3f}</td>"
            f"<td>{int(row.win_count)}</td>"
            f"<td>{int(row.params)}</td>"
            f"<td>{row.train_time:.3f}</td>"
            f"<td>{row.inference_time:.3f}</td>"
            "</tr>"
        )
    table_html = (
        '<table class="leaderboard-table">'
        "<thead><tr>"
        "<th>Model</th>"
        '<th>Overall MAE <span class="sort-pill">primary ↓</span></th>'
        '<th>Overall RMSE <span class="sort-pill">tiebreak ↓</span></th>'
        "<th>Mean RMAE</th><th>Median RMAE</th><th>Average Rank</th>"
        "<th>Wins</th><th>Params</th><th>Train time (s)</th><th>Inference time (s)</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )
    return (
        '<section class="card">'
        "<h2>Leaderboard</h2>"
        "<p>Static summary table of MAE, RMSE, relative MAE to the series mean, Average Rank, wins, and runtime.</p>"
        f"{table_html}"
        "</section>"
    )


def build_win_counts_section(aggregate_results: pd.DataFrame) -> str:
    """Build a Matplotlib win-count summary."""
    wins = aggregate_results[["model_name", "win_count"]].sort_values(
        ["win_count", "model_name"],
        ascending=[True, True],
    )
    fig, ax = plt.subplots(figsize=(6.2, 3.8), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")
    ax.barh(
        wins["model_name"],
        wins["win_count"],
        color=[MODEL_COLORS.get(name, "#cbd5e1") for name in wins["model_name"]],
    )
    ax.set_title("Per-series win counts", color="#e2e8f0")
    ax.set_xlabel("Per-series MAE wins", color="#cbd5e1")
    ax.tick_params(colors="#cbd5e1")
    ax.grid(True, axis="x", alpha=0.18, color="#94a3b8")
    return f'<section class="card"><h2>Per-series win counts</h2>{render_matplotlib_figure(fig, "Per-series win counts")}</section>'


def build_matrix_section(per_series_results: pd.DataFrame) -> str:
    """Build a hidden Matplotlib heatmap for per-series MAE."""
    clipped = per_series_results.copy()
    upper_mae = float(clipped["mae"].quantile(0.99)) if not clipped.empty else 1.0
    clipped["mae_clamped"] = clipped["mae"].clip(upper=upper_mae)
    matrix = (
        clipped.pivot(index="unique_id", columns="model_name", values="mae_clamped")
        .reindex(
            columns=[
                name for name in MODEL_NAMES if name in clipped["model_name"].unique()
            ]
        )
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(8.0, 10.0), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_title("Per-series MAE matrix", color="#e2e8f0")
    ax.set_xlabel("Model", color="#cbd5e1")
    ax.set_ylabel("Series", color="#cbd5e1")
    ax.set_xticks(
        range(len(matrix.columns)), labels=matrix.columns, rotation=45, ha="right"
    )
    ax.tick_params(colors="#cbd5e1")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("MAE (clamped 99th pct)", color="#cbd5e1")
    colorbar.ax.yaxis.set_tick_params(color="#cbd5e1")
    plt.setp(colorbar.ax.get_yticklabels(), color="#cbd5e1")
    return (
        '<section class="card">'
        "<h2>Per-series MAE matrix</h2>"
        "<details><summary>Show per-series MAE matrix</summary>"
        f"{render_matplotlib_figure(fig, 'Per-series MAE matrix')}"
        "</details></section>"
    )


def build_model_comparison_section(
    aggregate_results: pd.DataFrame,
    combined_forecasts: pd.DataFrame,
) -> str:
    """Build extra model-comparison views for detailed differences."""
    residual_parts: list[pd.DataFrame] = []
    for model_name in MODEL_NAMES:
        if model_name not in combined_forecasts.columns:
            continue
        frame = combined_forecasts[["y_true", model_name]].copy()
        frame = frame.rename(columns={model_name: "y_pred"})
        frame["model_name"] = model_name
        frame["residual"] = frame["y_pred"] - frame["y_true"]
        frame["abs_error"] = frame["residual"].abs()
        residual_parts.append(frame)
    residual_frame = pd.concat(residual_parts, ignore_index=True)
    residual_cap = (
        float(np.quantile(np.abs(residual_frame["residual"]), 0.99))
        if not residual_frame.empty
        else 1.0
    )
    residual_frame["residual_clamped"] = residual_frame["residual"].clip(
        -residual_cap, residual_cap
    )

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), facecolor="#0f172a")
    for ax in axes.flat:
        ax.set_facecolor("#0f172a")
        ax.tick_params(colors="#cbd5e1")
        ax.grid(True, alpha=0.18, color="#94a3b8")

    ranked = aggregate_results.sort_values("average_rank")
    axes[0, 0].barh(
        ranked["model_name"],
        ranked["average_rank"],
        color=[MODEL_COLORS[name] for name in ranked["model_name"]],
    )
    axes[0, 0].set_title("Average rank by model", color="#e2e8f0")
    axes[0, 0].set_xlabel("Average rank per series", color="#cbd5e1")

    for row in aggregate_results.itertuples():
        axes[0, 1].scatter(
            row.train_time,
            row.mean_series_rmae,
            s=max(60, int(row.win_count) * 6 + 60),
            color=MODEL_COLORS.get(str(row.model_name), "#cbd5e1"),
            alpha=0.85,
            label=str(row.model_name),
        )
        axes[0, 1].annotate(
            str(row.model_name),
            (row.train_time, row.mean_series_rmae),
            color="#e2e8f0",
            fontsize=8,
        )
    axes[0, 1].set_title("Accuracy-speed trade-off", color="#e2e8f0")
    axes[0, 1].set_xlabel("Train time (s)", color="#cbd5e1")
    axes[0, 1].set_ylabel("Mean series RMAE", color="#cbd5e1")

    bins = np.linspace(-max(residual_cap, 1e-6), max(residual_cap, 1e-6), 40)
    for model_name in MODEL_NAMES:
        values = residual_frame.loc[
            residual_frame["model_name"] == model_name, "residual_clamped"
        ].to_numpy()
        if len(values) == 0:
            continue
        counts, edges = np.histogram(values, bins=bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        axes[1, 0].plot(
            centers,
            counts,
            linewidth=2.0,
            color=MODEL_COLORS[model_name],
            label=model_name,
        )
    axes[1, 0].set_title(
        "Holdout residual distributions across all models", color="#e2e8f0"
    )
    axes[1, 0].set_xlabel(
        "Holdout residual (prediction - observed, clamped 99th pct)", color="#cbd5e1"
    )
    axes[1, 0].set_ylabel("Density", color="#cbd5e1")

    for model_name in MODEL_NAMES:
        frame = residual_frame[residual_frame["model_name"] == model_name]
        if frame.empty:
            continue
        axes[1, 1].scatter(
            frame["y_true"],
            frame["abs_error"],
            s=14,
            alpha=0.35,
            color=MODEL_COLORS[model_name],
            label=model_name,
        )
    axes[1, 1].set_title("Absolute holdout error vs observed value", color="#e2e8f0")
    axes[1, 1].set_xlabel("Observed holdout value", color="#cbd5e1")
    axes[1, 1].set_ylabel("Absolute holdout error", color="#cbd5e1")

    handles, labels = axes[1, 0].get_legend_handles_labels()
    legend = fig.legend(
        handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=8
    )
    for text in legend.get_texts():
        text.set_color("#e2e8f0")
    return (
        '<section class="card"><h2>Model comparison details</h2>'
        "<p>Use these views to inspect rank stability, the speed-accuracy trade-off, and holdout residual behavior across all models.</p>"
        f"{render_matplotlib_figure(fig, 'Model comparison details')}"
        "</section>"
    )


def build_neural_diagnostics_section(training_curves: pd.DataFrame) -> str:
    """Build neural-model training diagnostics for over/underfitting checks."""
    if training_curves.empty:
        return (
            '<section class="card"><h2>Neural model diagnostics</h2>'
            "<p>No neural training curves were captured.</p></section>"
        )
    neural_models = ["DLinear", "NLinear", "AutoTimeBase", "AutoTimeBaseTrend"]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2), facecolor="#0f172a")
    for ax in axes:
        ax.set_facecolor("#0f172a")
        ax.tick_params(colors="#cbd5e1")
        ax.grid(True, alpha=0.18, color="#94a3b8")

    for model_name in neural_models:
        model_frame = training_curves[training_curves["model_name"] == model_name]
        if model_frame.empty:
            continue
        for split, style in [("Train", "-"), ("Validation", "--")]:
            split_frame = model_frame[model_frame["split"] == split].sort_values("step")
            if split_frame.empty:
                continue
            axes[0].plot(
                split_frame["step"],
                split_frame["loss"],
                linestyle=style,
                linewidth=2.0,
                color=MODEL_COLORS[model_name],
                label=f"{model_name} - {split}",
            )
    axes[0].set_title("Training and validation loss curves", color="#e2e8f0")
    axes[0].set_xlabel("Training step", color="#cbd5e1")
    axes[0].set_ylabel("Loss", color="#cbd5e1")

    final_losses = (
        training_curves.sort_values("step")
        .groupby(["model_name", "split"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    gap_pivot = final_losses.pivot(
        index="model_name", columns="split", values="loss"
    ).reset_index()
    gap_html = ""
    if {"Train", "Validation"}.issubset(gap_pivot.columns):
        gap_pivot["generalization_gap"] = gap_pivot["Validation"] - gap_pivot["Train"]
        ordered = (
            gap_pivot.set_index("model_name")
            .reindex(neural_models)
            .dropna(subset=["generalization_gap"])
            .reset_index()
        )
        axes[1].barh(
            ordered["model_name"],
            ordered["generalization_gap"],
            color=[MODEL_COLORS[name] for name in ordered["model_name"]],
        )
        axes[1].set_title("Final generalization gap", color="#e2e8f0")
        axes[1].set_xlabel("Final validation - train loss", color="#cbd5e1")
        gap_html = "<p><strong>Generalization gap</strong> compares the final validation loss against the final training loss.</p>"
    else:
        axes[1].text(
            0.5,
            0.5,
            "Validation loss not logged for all models",
            ha="center",
            va="center",
            color="#e2e8f0",
        )
        axes[1].set_axis_off()
    legend = axes[0].legend(ncol=2, frameon=False, fontsize=8)
    for text in legend.get_texts():
        text.set_color("#e2e8f0")
    return (
        '<section class="card"><h2>Neural model diagnostics</h2>'
        "<p>Use these training curves to look for overfitting or underfitting: widening train/validation gaps suggest overfitting, while flat and high losses suggest underfitting.</p>"
        f"{render_matplotlib_figure(fig, 'Neural model diagnostics')}{gap_html}</section>"
    )


def build_html_report(
    aggregate_results: pd.DataFrame,
    per_series_results: pd.DataFrame,
    dataset_summary: dict[str, Any],
    representative_plot_sections: list[str],
    mae_distribution_section: str,
    model_comparison_section: str,
    neural_diagnostics_section: str,
) -> str:
    """Render a self-contained HTML benchmark report."""
    leaderboard = aggregate_results.sort_values(["overall_mae", "overall_rmse"])
    summary_section = build_summary_section(dataset_summary)
    logo_data_uri = build_logo_data_uri()
    leaderboard_section = build_leaderboard_section(leaderboard)
    win_counts_section = build_win_counts_section(leaderboard)
    matrix_section = build_matrix_section(per_series_results)
    plot_html = "".join(representative_plot_sections)

    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>Custom dataset benchmark report</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #0f172a;
      --panel: rgba(15, 23, 42, 0.92);
      --panel-2: rgba(30, 41, 59, 0.96);
      --text: #e2e8f0;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --accent-soft: rgba(56, 189, 248, 0.14);
      --border: rgba(148, 163, 184, 0.22);
    }}
    body {{ margin: 0; font-family: Roboto, Inter, ui-sans-serif, system-ui, sans-serif; background: linear-gradient(180deg, #0b1220 0%, #111827 100%); color: var(--text); }}
    main {{ max-width: 1480px; margin: 0 auto; padding: 32px 24px 72px; }}
    h1, h2, h3 {{ margin: 0 0 12px; line-height: 1.2; }}
    p {{ color: var(--muted); line-height: 1.5; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 20px; margin: 24px 0; align-items: start; }}
    .card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 18px; padding: 22px; box-shadow: 0 14px 40px rgba(0, 0, 0, 0.28); overflow-x: auto; }}
    .plot-grid {{ display: grid; grid-template-columns: 1fr; gap: 24px; }}
    .plot-image {{ width: 100%; height: auto; display: block; border-radius: 12px; }}
    .full-width {{ width: 100%; margin: 24px 0; }}
    .hero {{ margin-bottom: 24px; padding: 28px; border: 1px solid var(--border); border-radius: 20px; background: linear-gradient(180deg, rgba(30, 41, 59, 0.96) 0%, rgba(15, 23, 42, 0.96) 100%); }}
    .hero-top {{ display: flex; gap: 18px; align-items: center; margin-bottom: 14px; }}
    .hero-logo {{ width: 56px; height: 56px; object-fit: contain; }}
    .kpi {{ display: inline-block; margin-right: 12px; margin-top: 10px; padding: 12px 16px; border-radius: 14px; background: var(--accent-soft); border: 1px solid rgba(56, 189, 248, 0.24); }}
    figure {{ margin: 0; }}
    .tabs {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 20px 0 28px; }}
    .tab-button {{ background: var(--panel-2); color: var(--text); border: 1px solid var(--border); border-radius: 999px; padding: 10px 16px; cursor: pointer; font-weight: 600; }}
    .tab-button.active {{ background: var(--accent-soft); border-color: rgba(56, 189, 248, 0.55); color: #e0f2fe; }}
    .tab-panel {{ display: none; }}
    .tab-panel.active {{ display: block; }}
    .leaderboard-table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    .leaderboard-table th, .leaderboard-table td {{ padding: 10px 12px; border-bottom: 1px solid rgba(148, 163, 184, 0.16); text-align: left; white-space: nowrap; }}
    .leaderboard-table th {{ color: #bae6fd; position: sticky; top: 0; background: rgba(15, 23, 42, 0.98); }}
    .sort-pill {{ display: inline-block; margin-left: 6px; padding: 2px 8px; border-radius: 999px; font-size: 11px; color: #e0f2fe; background: rgba(56, 189, 248, 0.18); border: 1px solid rgba(56, 189, 248, 0.28); }}
    summary {{ cursor: pointer; color: #bae6fd; font-weight: 600; margin-bottom: 12px; }}
  </style>
</head>
<body>
  <main>
    <section class=\"hero\">
      <div class=\"hero-top\">
        <img class=\"hero-logo\" src=\"{logo_data_uri}\" alt=\"TimeBaseUla logo\">
        <div>
          <h1>Custom dataset benchmark report</h1>
          <p>Professional benchmark comparison for the custom monthly toll dataset.</p>
        </div>
      </div>
      <p>Benchmark of SeasonalNaive, MFLES, AutoTimeBase, AutoTimeBaseTrend, NLinear, and DLinear on the custom monthly dataset using the same trailing holdout horizon for every series. The goal is to compare absolute accuracy, relative accuracy, stability across series, and diagnostic signs of misfit.</p>
      <div class=\"kpi\"><strong>Best model:</strong> {escape(str(leaderboard.iloc[0]["model_name"]))}</div>
      <div class=\"kpi\"><strong>Best overall MAE:</strong> {leaderboard.iloc[0]["overall_mae"]:.4f}</div>
      <div class=\"kpi\"><strong>Series count:</strong> {dataset_summary["n_series"]}</div>
      <div class=\"kpi\"><strong>Horizon:</strong> {dataset_summary["horizon"]}</div>
    </section>
    <div class=\"tabs\">
      <button class=\"tab-button active\" data-tab=\"general\">General</button>
      <button class=\"tab-button\" data-tab=\"representative\">Representative series</button>
      <button class=\"tab-button\" data-tab=\"distribution\">RMAE and model differences</button>
      <button class=\"tab-button\" data-tab=\"diagnostics\">Neural diagnostics</button>
    </div>
    <section id=\"tab-general\" class=\"tab-panel active\">
      <div class=\"grid\">{summary_section}{leaderboard_section}</div>
    </section>
    <section id=\"tab-representative\" class=\"tab-panel\">
      <section class=\"card\">
        <h2>Representative forecast plots</h2>
        <p>Five representative series spread across the anchor-model ranking.</p>
        <div class=\"plot-grid\">{plot_html}</div>
      </section>
    </section>
    <section id=\"tab-distribution\" class=\"tab-panel\">
      {mae_distribution_section}
      <div class=\"grid\">{win_counts_section}</div>
      <div class=\"full-width\">{model_comparison_section}</div>
      {matrix_section}
    </section>
    <section id=\"tab-diagnostics\" class=\"tab-panel\">
      {neural_diagnostics_section}
    </section>
  </main>
  <script>
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabPanels = document.querySelectorAll('.tab-panel');
    tabButtons.forEach((button) => {{
      button.addEventListener('click', () => {{
        const selected = button.dataset.tab;
        tabButtons.forEach((item) => item.classList.toggle('active', item === button));
        tabPanels.forEach((panel) => panel.classList.toggle('active', panel.id === `tab-${{selected}}`));
      }});
    }});
  </script>
</body>
</html>
"""


def render_console_table(aggregate_results: pd.DataFrame) -> Table:
    """Build the Rich leaderboard table shown by the CLI."""
    table = Table(title="Custom dataset benchmark leaderboard")
    table.add_column("Model")
    table.add_column("Overall MAE", justify="right")
    table.add_column("Overall RMSE", justify="right")
    table.add_column("Params", justify="right")
    table.add_column("Mean series RMAE", justify="right")
    table.add_column("Average Rank", justify="right")
    table.add_column("Mean series MAE", justify="right")
    table.add_column("Wins", justify="right")
    table.add_column("Train time (s)", justify="right")
    table.add_column("Inference time (s)", justify="right")

    for row in aggregate_results.sort_values(
        ["overall_mae", "overall_rmse"]
    ).itertuples():
        table.add_row(
            row.model_name,
            f"{row.overall_mae:.4f}",
            f"{row.overall_rmse:.4f}",
            str(row.params),
            f"{row.mean_series_rmae:.4f}",
            f"{row.average_rank:.3f}",
            f"{row.mean_series_mae:.4f}",
            str(row.win_count),
            f"{row.train_time:.2f}",
            f"{row.inference_time:.2f}",
        )
    return table


@app.command()
def main(
    dataset_path: Path = typer.Option(
        DATASET_PATH, help="Path to the custom dataset CSV."
    ),
    output_dir: Path = typer.Option(
        DEFAULT_OUTPUT_DIR, help="Directory for CSV, JSON, and HTML outputs."
    ),
    horizon: int = typer.Option(
        12, min=1, help="Forecast horizon and test size per series."
    ),
    max_steps: int = typer.Option(
        30, min=1, help="Maximum neural training steps before recommendation clipping."
    ),
    representative_series_count: int = typer.Option(
        5, min=1, help="Number of representative series plots to include."
    ),
    quiet: bool = typer.Option(False, help="Suppress Rich progress output."),
    json_output: bool = typer.Option(
        False, "--json", help="Print a JSON summary to stdout."
    ),
) -> None:
    """Run the custom dataset benchmark and save a self-contained HTML report."""
    logger = configure_logging()
    frame = load_custom_dataset(dataset_path)
    validate_series_lengths(frame, horizon=horizon)
    dataset_summary = summarise_dataset(frame, freq="MS", horizon=horizon)
    train_frame, test_frame = prepare_train_test(frame, horizon=horizon)
    target_frame = build_holdout_target(test_frame)

    common_kwargs = choose_common_model_kwargs(
        train_frame=train_frame,
        freq="MS",
        horizon=horizon,
        max_steps=max_steps,
    )
    if not quiet and not json_output:
        console.print(
            f"Running benchmark on {dataset_summary['n_series']} series, {dataset_summary['n_rows']} rows, horizon={horizon}.",
        )

    aggregate_rows: list[dict[str, float | int | str]] = []
    per_series_parts: list[pd.DataFrame] = []
    forecast_frames: dict[str, pd.DataFrame] = {}
    training_curve_parts: list[pd.DataFrame] = []

    seasonal_naive_aggregate, seasonal_naive_per_series, seasonal_naive_forecast = (
        run_seasonal_naive_model(
            train_frame=train_frame,
            target_frame=target_frame,
            horizon=horizon,
            logger=logger,
        )
    )
    aggregate_rows.append(seasonal_naive_aggregate)
    per_series_parts.append(seasonal_naive_per_series)
    forecast_frames["SeasonalNaive"] = seasonal_naive_forecast

    mfles_aggregate, mfles_per_series, mfles_forecast = run_mfles_model(
        train_frame=train_frame,
        target_frame=target_frame,
        freq="MS",
        horizon=horizon,
        logger=logger,
    )
    aggregate_rows.append(mfles_aggregate)
    per_series_parts.append(mfles_per_series)
    forecast_frames["MFLES"] = mfles_forecast[["unique_id", "ds", "MFLES"]]

    neural_models = [
        DLinear(h=horizon, **common_kwargs),
        NLinear(h=horizon, **common_kwargs),
        AutoTimeBase(h=horizon, freq="MS", max_steps=max_steps, search_max_steps=10),
        AutoTimeBaseTrend(
            h=horizon,
            freq="MS",
            max_steps=max_steps,
            search_max_steps=10,
        ),
    ]
    prediction_columns = [
        "DLinear",
        "NLinear",
        "AutoTimeBase",
        "AutoTimeBaseTrend",
    ]

    for model, prediction_column in zip(neural_models, prediction_columns, strict=True):
        aggregate, per_series, forecast_frame, training_curve = run_neural_model(
            train_frame=train_frame,
            target_frame=target_frame,
            freq="MS",
            model=model,
            prediction_column=prediction_column,
            logger=logger,
            output_dir=output_dir,
        )
        aggregate_rows.append(aggregate)
        per_series_parts.append(per_series)
        forecast_frames[prediction_column] = forecast_frame[
            ["unique_id", "ds", prediction_column]
        ]
        training_curve_parts.append(training_curve)

    aggregate_results = pd.DataFrame(aggregate_rows)
    per_series_results = pd.concat(per_series_parts, ignore_index=True)
    per_series_results = add_relative_mae(per_series_results, frame)
    aggregate_results = add_win_counts(aggregate_results, per_series_results)
    aggregate_results = add_average_ranks(aggregate_results, per_series_results)
    aggregate_results = add_relative_mae_summary(aggregate_results, per_series_results)
    aggregate_results["model_name"] = pd.Categorical(
        aggregate_results["model_name"],
        categories=MODEL_NAMES,
        ordered=True,
    )
    aggregate_results = aggregate_results.sort_values(["overall_mae", "overall_rmse"])

    representative_ids = choose_representative_series(
        per_series_results=per_series_results,
        anchor_model=str(aggregate_results.iloc[0]["model_name"]),
        n_examples=representative_series_count,
    )
    representative_plot_sections = build_representative_plot_sections(
        full_frame=frame,
        target_frame=target_frame,
        forecast_frames=forecast_frames,
        representative_ids=representative_ids,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_forecasts = target_frame[["unique_id", "ds", "y_true"]].copy()
    for _model_name, forecast_frame in forecast_frames.items():
        combined_forecasts = combined_forecasts.merge(
            forecast_frame,
            on=["unique_id", "ds"],
            how="left",
        )
    training_curves = (
        pd.concat(training_curve_parts, ignore_index=True)
        if training_curve_parts
        else pd.DataFrame()
    )
    mae_distribution_section = build_mae_distribution_section(per_series_results)
    model_comparison_section = build_model_comparison_section(
        aggregate_results, combined_forecasts
    )

    leaderboard_path = output_dir / "leaderboard.csv"
    per_series_path = output_dir / "per_series.csv"
    summary_path = output_dir / "summary.json"
    forecasts_path = output_dir / "forecasts.csv"
    html_path = output_dir / "report.html"

    aggregate_results.to_csv(leaderboard_path, index=False)
    per_series_results.to_csv(per_series_path, index=False)
    combined_forecasts.to_csv(forecasts_path, index=False)
    summary_path.write_text(json.dumps(dataset_summary, indent=2), encoding="utf-8")
    neural_diagnostics_section = build_neural_diagnostics_section(training_curves)
    html_path.write_text(
        build_html_report(
            aggregate_results=aggregate_results,
            per_series_results=per_series_results,
            dataset_summary=dataset_summary,
            representative_plot_sections=representative_plot_sections,
            mae_distribution_section=mae_distribution_section,
            model_comparison_section=model_comparison_section,
            neural_diagnostics_section=neural_diagnostics_section,
        ),
        encoding="utf-8",
    )

    if json_output:
        payload = {
            "dataset_summary": dataset_summary,
            "leaderboard": aggregate_results.to_dict(orient="records"),
            "representative_series": representative_ids,
            "files": {
                "leaderboard_csv": str(leaderboard_path),
                "per_series_csv": str(per_series_path),
                "forecasts_csv": str(forecasts_path),
                "summary_json": str(summary_path),
                "report_html": str(html_path),
            },
        }
        typer.echo(json.dumps(payload, indent=2, default=str))
        return

    if not quiet:
        console.print(render_console_table(aggregate_results))
        console.print(f"HTML report written to [bold]{html_path}[/bold]")


if __name__ == "__main__":
    app()
