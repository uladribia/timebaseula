"""Benchmark TimeBaseUla models on NeuralForecast's AirPassengersPanel data."""

from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_HORIZON = 12
DEFAULT_INPUT_SIZE = 24
DEFAULT_MAX_STEPS = 30
DEFAULT_OUTPUT_MARKDOWN = Path("docs/benchmark.md")
DEFAULT_OUTPUT_PLOT = Path("docs/img/airpassengers-benchmark.png")
DEFAULT_OUTPUT_CONFORMAL_PLOT = Path(
    "docs/img/airpassengers-timebasetrend-conformal.png"
)
DEFAULT_LOG_PATH = Path("logs/benchmark_airpassengers.log")
DEFAULT_FREQ = "ME"
BASELINE_MODEL_NAME = "Naive"


def split_train_test(
    frame: pd.DataFrame, horizon: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a long-format panel into train and test partitions per series."""
    sorted_frame = frame.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    train = sorted_frame.groupby("unique_id", group_keys=False).head(-horizon)
    test = sorted_frame.groupby("unique_id", group_keys=False).tail(horizon)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def count_trainable_parameters(model: Any) -> int:
    """Count trainable parameters for torch-style models."""
    if not hasattr(model, "parameters"):
        return 0
    return int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
    )


def build_metrics_table(
    actual: pd.DataFrame,
    forecasts: dict[str, pd.DataFrame],
    runtimes: dict[str, float],
    parameter_counts: dict[str, int],
) -> pd.DataFrame:
    """Build the benchmark metrics table for all evaluated models."""
    metrics_rows: list[dict[str, float | int | str]] = []
    naive_mae: float | None = None

    for model_name, forecast in forecasts.items():
        merged = actual.merge(
            forecast, on=["unique_id", "ds"], how="inner", validate="one_to_one"
        )
        errors = merged["y"] - merged["y_hat"]
        mae = float(np.abs(errors).mean())
        rmse = float(np.sqrt(np.square(errors).mean()))
        if model_name == BASELINE_MODEL_NAME:
            naive_mae = mae
        metrics_rows.append(
            {
                "model": model_name,
                "mae": mae,
                "rmse": rmse,
                "parameters": int(parameter_counts.get(model_name, 0)),
                "runtime_seconds": float(runtimes[model_name]),
            }
        )

    if naive_mae is None:
        msg = f"Missing baseline forecast for '{BASELINE_MODEL_NAME}'."
        raise ValueError(msg)

    metrics = pd.DataFrame(metrics_rows)
    metrics["rmae"] = metrics["mae"] / naive_mae
    metrics = metrics[["model", "mae", "rmse", "rmae", "parameters", "runtime_seconds"]]
    metrics = metrics.sort_values(["mae", "runtime_seconds", "model"]).reset_index(
        drop=True
    )
    numeric_columns = ["mae", "rmse", "rmae", "runtime_seconds"]
    metrics[numeric_columns] = metrics[numeric_columns].round(4)
    return metrics


def render_markdown_table(frame: pd.DataFrame) -> str:
    """Render a DataFrame as a markdown table without extra dependencies."""
    headers = list(frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = [str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def render_settings_snippet(model_settings: dict[str, dict[str, int | float]]) -> str:
    """Render model settings as a formatted JSON code block."""
    return (
        "```python\nMODEL_SETTINGS = " + json.dumps(model_settings, indent=2) + "\n```"
    )


def render_markdown_report(
    metrics: pd.DataFrame,
    horizon: int,
    plot_path: str,
    model_settings: dict[str, dict[str, int | float]],
) -> str:
    """Render the benchmark report as markdown suitable for the docs site."""
    best_model = metrics.iloc[0]["model"]
    return f"""---
description: Benchmark report for TimeBaseUla on the AirPassengersPanel dataset.
---

# AirPassengers benchmark

## TL;DR
- Dataset: `AirPassengersPanel` from `neuralforecast.utils`
- Horizon: `{horizon}` months per series
- Series benchmarked: `2`
- Models: `TimeBase`, `TimeBaseTrend`, `NLinear`, `DLinear`, `AutoMFLES`, `Naive`
- Best MAE in this run: `{best_model}`

## Setup
- The benchmark uses the last `{horizon}` observations of each series as test data.
- Neural models use small model-specific settings tuned for this dataset.
- `RMAE` is computed relative to `Naive`.
- Statistical baselines report `0` trainable parameters.

!!! tip "Interpret these results with care"
    `AirPassengersPanel` is a difficult benchmark for neural networks because it contains only **two series** and a **short monthly history**.
    That means there is much less cross-series information and much less temporal evidence than in larger panel datasets.
    Neural models can still work well here, but this benchmark should be read as a **small-data stress test**, not as a broad ranking of neural forecasting performance.

## Metrics

{render_markdown_table(metrics)}

## Reproducible model settings

{render_settings_snippet(model_settings)}

## Forecast plot

![AirPassengers benchmark]({plot_path})
"""


def configure_logging(log_path: Path) -> logging.Logger:
    """Configure a rotating file logger for benchmark runs."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("benchmark_airpassengers")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    handler = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=1)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def _trainer_overrides(max_steps: int, learning_rate: float) -> dict[str, Any]:
    """Return compact CPU-first trainer settings for the benchmark."""
    return {
        "max_steps": max_steps,
        "val_check_steps": max(10, min(max_steps, 25)),
        "learning_rate": learning_rate,
        "batch_size": 32,
        "windows_batch_size": 128,
        "random_seed": 1,
        "accelerator": "cpu",
        "devices": 1,
        "logger": False,
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "log_every_n_steps": 1,
    }


def resolve_benchmark_loss(loss_name: str):
    """Resolve a benchmark loss name to a NeuralForecast loss instance."""
    from neuralforecast.losses.pytorch import DistributionLoss, MAE

    normalized_name = loss_name.strip().lower()
    if normalized_name == "mae":
        return MAE()
    if normalized_name in {"normal", "gaussian"}:
        return DistributionLoss("Normal", level=[80, 95])
    if normalized_name == "poisson":
        return DistributionLoss("Poisson", level=[80, 95])
    msg = f"Unsupported benchmark loss: {loss_name}"
    raise ValueError(msg)


def get_neural_model_configs(
    input_size: int, max_steps: int
) -> dict[str, dict[str, int | float]]:
    """Return tuned per-model settings for the AirPassengers benchmark.

    The input arguments act as fallbacks for ad-hoc runs, while the defaults
    are intentionally overridden by model-specific settings tuned on this small
    monthly benchmark.
    """
    del input_size, max_steps
    return {
        "TimeBase": {
            "input_size": 48,
            "max_steps": 30,
            "learning_rate": 1e-2,
            "basis_num": 6,
            "period_len": 12,
        },
        "TimeBaseTrend": {
            "input_size": 48,
            "max_steps": 100,
            "learning_rate": 1e-2,
            "basis_num": 6,
            "period_len": 6,
            "moving_avg_window": 25,
        },
        "NLinear": {
            "input_size": 36,
            "max_steps": 100,
            "learning_rate": 5e-3,
        },
        "DLinear": {
            "input_size": 12,
            "max_steps": 100,
            "learning_rate": 1e-2,
        },
    }


def _normalize_forecast_frame(forecast: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Normalize model forecasts to a common three-column schema."""
    normalized = forecast.reset_index()
    return normalized[["unique_id", "ds", model_name]].rename(
        columns={model_name: "y_hat"}
    )


def run_neuralforecast_models(
    train_df: pd.DataFrame,
    horizon: int,
    input_size: int,
    max_steps: int,
    neural_loss_name: str,
    logger: logging.Logger,
) -> tuple[dict[str, pd.DataFrame], dict[str, float], dict[str, int]]:
    """Fit and predict the NeuralForecast models used in the benchmark."""
    import warnings

    from neuralforecast import NeuralForecast
    from neuralforecast.models import DLinear, NLinear

    from timebaseula import TimeBase, TimeBaseTrend

    model_configs = get_neural_model_configs(input_size=input_size, max_steps=max_steps)
    neural_loss = resolve_benchmark_loss(neural_loss_name)
    models = [
        TimeBase(
            h=horizon,
            freq=DEFAULT_FREQ,
            input_size=int(model_configs["TimeBase"]["input_size"]),
            basis_num=int(model_configs["TimeBase"]["basis_num"]),
            period_len=int(model_configs["TimeBase"]["period_len"]),
            loss=neural_loss,
            **_trainer_overrides(
                max_steps=int(model_configs["TimeBase"]["max_steps"]),
                learning_rate=float(model_configs["TimeBase"]["learning_rate"]),
            ),
        ),
        TimeBaseTrend(
            h=horizon,
            freq=DEFAULT_FREQ,
            input_size=int(model_configs["TimeBaseTrend"]["input_size"]),
            basis_num=int(model_configs["TimeBaseTrend"]["basis_num"]),
            period_len=int(model_configs["TimeBaseTrend"]["period_len"]),
            moving_avg_window=int(model_configs["TimeBaseTrend"]["moving_avg_window"]),
            loss=neural_loss,
            **_trainer_overrides(
                max_steps=int(model_configs["TimeBaseTrend"]["max_steps"]),
                learning_rate=float(model_configs["TimeBaseTrend"]["learning_rate"]),
            ),
        ),
        NLinear(
            h=horizon,
            input_size=int(model_configs["NLinear"]["input_size"]),
            loss=neural_loss,
            **_trainer_overrides(
                max_steps=int(model_configs["NLinear"]["max_steps"]),
                learning_rate=float(model_configs["NLinear"]["learning_rate"]),
            ),
        ),
        DLinear(
            h=horizon,
            input_size=int(model_configs["DLinear"]["input_size"]),
            loss=neural_loss,
            **_trainer_overrides(
                max_steps=int(model_configs["DLinear"]["max_steps"]),
                learning_rate=float(model_configs["DLinear"]["learning_rate"]),
            ),
        ),
    ]

    forecasts: dict[str, pd.DataFrame] = {}
    runtimes: dict[str, float] = {}
    parameter_counts: dict[str, int] = {}

    for model in models:
        model_name = type(model).__name__
        parameter_counts[model_name] = count_trainable_parameters(model)
        logger.info("Running neural model %s", model_name)
        nf = NeuralForecast(models=[model], freq=DEFAULT_FREQ)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=Warning)
            start_time = perf_counter()
            nf.fit(train_df, val_size=horizon)
            prediction = nf.predict()
            runtimes[model_name] = perf_counter() - start_time
        forecasts[model_name] = _normalize_forecast_frame(prediction, model_name)

    return forecasts, runtimes, parameter_counts


def get_reproducible_model_settings(
    horizon: int,
    input_size: int,
    max_steps: int,
) -> dict[str, dict[str, int | float]]:
    """Return the exact model settings published in the benchmark report."""
    settings = get_neural_model_configs(
        input_size=input_size, max_steps=max_steps
    ).copy()
    settings["AutoMFLES"] = {
        "test_size": horizon,
        "season_length": 12,
    }
    settings["Naive"] = {}
    return settings


def run_statsforecast_models(
    train_df: pd.DataFrame,
    horizon: int,
    logger: logging.Logger,
) -> tuple[dict[str, pd.DataFrame], dict[str, float], dict[str, int]]:
    """Fit and predict the StatsForecast models used in the benchmark."""
    from statsforecast import StatsForecast
    from statsforecast.models import AutoMFLES, Naive

    models = [
        AutoMFLES(test_size=horizon, season_length=12),
        Naive(),
    ]

    forecasts: dict[str, pd.DataFrame] = {}
    runtimes: dict[str, float] = {}
    parameter_counts = {"AutoMFLES": 0, "Naive": 0}

    for model in models:
        model_name = model.alias
        logger.info("Running statistical model %s", model_name)
        start_time = perf_counter()
        sf = StatsForecast(models=[model], freq=DEFAULT_FREQ, n_jobs=1)
        prediction = sf.forecast(df=train_df, h=horizon)
        runtimes[model_name] = perf_counter() - start_time
        forecasts[model_name] = prediction[["unique_id", "ds", model_name]].rename(
            columns={model_name: "y_hat"}
        )

    return forecasts, runtimes, parameter_counts


def save_forecast_plot(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    forecasts: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    """Save a two-panel plot with actual values and model forecasts."""
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    series_ids = sorted(test_df["unique_id"].unique())
    colors = {
        "TimeBase": "#1f77b4",
        "TimeBaseTrend": "#ff7f0e",
        "NLinear": "#2ca02c",
        "DLinear": "#d62728",
        "AutoMFLES": "#9467bd",
        "Naive": "#8c564b",
    }

    figure, axes = plt.subplots(len(series_ids), 1, figsize=(12, 8), sharex=True)
    if len(series_ids) == 1:
        axes = [axes]

    for axis, series_id in zip(axes, series_ids, strict=True):
        series_train = train_df.loc[train_df["unique_id"] == series_id]
        series_test = test_df.loc[test_df["unique_id"] == series_id]

        axis.plot(
            series_train["ds"],
            series_train["y"],
            color="black",
            linewidth=1.5,
            label="train",
        )
        axis.plot(
            series_test["ds"],
            series_test["y"],
            color="black",
            linestyle="--",
            linewidth=2.0,
            marker="o",
            label="test",
        )

        for model_name, forecast in forecasts.items():
            series_forecast = forecast.loc[forecast["unique_id"] == series_id]
            axis.plot(
                series_forecast["ds"],
                series_forecast["y_hat"],
                label=model_name,
                color=colors[model_name],
                linewidth=1.8,
                marker="o",
            )

        axis.set_title(series_id)
        axis.set_ylabel("Passengers")
        axis.grid(alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def run_timebasetrend_conformal_example(
    train_df: pd.DataFrame,
    horizon: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Fit TimeBaseTrend with conformal intervals on the benchmark split."""
    import warnings

    from neuralforecast import NeuralForecast
    from neuralforecast.utils import PredictionIntervals

    from timebaseula import TimeBaseTrend

    model_configs = get_neural_model_configs(
        input_size=DEFAULT_INPUT_SIZE,
        max_steps=DEFAULT_MAX_STEPS,
    )
    logger.info("Running TimeBaseTrend conformal interval example")
    model = TimeBaseTrend(
        h=horizon,
        freq=DEFAULT_FREQ,
        input_size=int(model_configs["TimeBaseTrend"]["input_size"]),
        basis_num=int(model_configs["TimeBaseTrend"]["basis_num"]),
        period_len=int(model_configs["TimeBaseTrend"]["period_len"]),
        moving_avg_window=int(model_configs["TimeBaseTrend"]["moving_avg_window"]),
        **_trainer_overrides(
            max_steps=int(model_configs["TimeBaseTrend"]["max_steps"]),
            learning_rate=float(model_configs["TimeBaseTrend"]["learning_rate"]),
        ),
    )
    nf = NeuralForecast(models=[model], freq=DEFAULT_FREQ)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=Warning)
        nf.fit(
            train_df,
            val_size=horizon,
            prediction_intervals=PredictionIntervals(
                n_windows=2,
                method="conformal_error",
            ),
        )
        prediction = nf.predict(level=[80, 95])
    return prediction.reset_index()


def save_conformal_interval_plot(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    conformal_forecast: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save a plot of TimeBaseTrend forecasts with conformal error bands."""
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    series_ids = sorted(test_df["unique_id"].unique())
    figure, axes = plt.subplots(len(series_ids), 1, figsize=(12, 8), sharex=True)
    if len(series_ids) == 1:
        axes = [axes]

    for axis, series_id in zip(axes, series_ids, strict=True):
        series_train = train_df.loc[train_df["unique_id"] == series_id]
        series_test = test_df.loc[test_df["unique_id"] == series_id]
        series_forecast = conformal_forecast.loc[
            conformal_forecast["unique_id"] == series_id
        ]

        axis.plot(
            series_train["ds"],
            series_train["y"],
            color="black",
            linewidth=1.5,
            label="train",
        )
        axis.plot(
            series_test["ds"],
            series_test["y"],
            color="black",
            linestyle="--",
            linewidth=2.0,
            marker="o",
            label="test",
        )
        axis.plot(
            series_forecast["ds"],
            series_forecast["TimeBaseTrend"],
            color="#ff7f0e",
            linewidth=2.0,
            marker="o",
            label="TimeBaseTrend",
        )
        axis.fill_between(
            series_forecast["ds"],
            series_forecast["TimeBaseTrend-lo-95"],
            series_forecast["TimeBaseTrend-hi-95"],
            color="#ff7f0e",
            alpha=0.15,
            label="95% conformal band",
        )
        axis.fill_between(
            series_forecast["ds"],
            series_forecast["TimeBaseTrend-lo-80"],
            series_forecast["TimeBaseTrend-hi-80"],
            color="#ff7f0e",
            alpha=0.30,
            label="80% conformal band",
        )
        axis.set_title(series_id)
        axis.set_ylabel("Passengers")
        axis.grid(alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def benchmark_airpassengers(
    horizon: int,
    input_size: int,
    max_steps: int,
    neural_loss_name: str,
    output_markdown: Path,
    output_plot: Path,
    output_conformal_plot: Path,
    log_path: Path,
) -> pd.DataFrame:
    """Run the full AirPassengers benchmark and persist the report artifacts."""
    from neuralforecast.utils import AirPassengersPanel

    logger = configure_logging(log_path)
    logger.info(
        "Starting AirPassengers benchmark with neural loss %s", neural_loss_name
    )

    frame = AirPassengersPanel[["unique_id", "ds", "y"]].copy()
    train_df, test_df = split_train_test(frame, horizon=horizon)

    neural_forecasts, neural_runtimes, neural_parameters = run_neuralforecast_models(
        train_df=train_df,
        horizon=horizon,
        input_size=input_size,
        max_steps=max_steps,
        neural_loss_name=neural_loss_name,
        logger=logger,
    )
    stats_forecasts, stats_runtimes, stats_parameters = run_statsforecast_models(
        train_df=train_df,
        horizon=horizon,
        logger=logger,
    )

    forecasts = neural_forecasts | stats_forecasts
    runtimes = neural_runtimes | stats_runtimes
    parameter_counts = neural_parameters | stats_parameters

    metrics = build_metrics_table(
        actual=test_df,
        forecasts=forecasts,
        runtimes=runtimes,
        parameter_counts=parameter_counts,
    )

    save_forecast_plot(
        train_df=train_df,
        test_df=test_df,
        forecasts=forecasts,
        output_path=output_plot,
    )
    conformal_forecast = run_timebasetrend_conformal_example(
        train_df=train_df,
        horizon=horizon,
        logger=logger,
    )
    save_conformal_interval_plot(
        train_df=train_df,
        test_df=test_df,
        conformal_forecast=conformal_forecast,
        output_path=output_conformal_plot,
    )

    report = render_markdown_report(
        metrics=metrics,
        horizon=horizon,
        plot_path=output_plot.relative_to(output_markdown.parent).as_posix(),
        model_settings=get_reproducible_model_settings(
            horizon=horizon,
            input_size=input_size,
            max_steps=max_steps,
        ),
    )
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text(report, encoding="utf-8")

    logger.info("Finished AirPassengers benchmark")
    return metrics


def build_app() -> Any:
    """Build the Typer CLI application."""
    import typer
    from rich.console import Console
    from rich.table import Table

    app = typer.Typer(help="Benchmark TimeBaseUla on the AirPassengersPanel dataset.")

    @app.callback()
    def benchmark_airpassengers_callback() -> None:
        """Run subcommands for the AirPassengers benchmark CLI."""

    @app.command("run")
    def run(
        horizon: int = typer.Option(
            DEFAULT_HORIZON, help="Forecast horizon per series."
        ),
        input_size: int = typer.Option(
            DEFAULT_INPUT_SIZE, help="Input window size for neural models."
        ),
        max_steps: int = typer.Option(
            DEFAULT_MAX_STEPS, help="Maximum training steps for neural models."
        ),
        neural_loss_name: str = typer.Option(
            "mae",
            "--neural-loss",
            help="Loss for neural models: mae, normal, gaussian, or poisson.",
        ),
        output_markdown: Path = typer.Option(
            DEFAULT_OUTPUT_MARKDOWN, help="Markdown report output path."
        ),
        output_plot: Path = typer.Option(
            DEFAULT_OUTPUT_PLOT, help="Forecast plot output path."
        ),
        output_conformal_plot: Path = typer.Option(
            DEFAULT_OUTPUT_CONFORMAL_PLOT,
            help="TimeBaseTrend conformal interval plot output path.",
        ),
        log_path: Path = typer.Option(DEFAULT_LOG_PATH, help="Log file output path."),
        json_output: bool = typer.Option(
            False, "--json", help="Emit the metrics table as JSON."
        ),
        quiet: bool = typer.Option(
            False, "--quiet", help="Suppress human-readable console output."
        ),
    ) -> None:
        """Run the AirPassengers benchmark and write the docs-ready markdown report."""
        console = Console(stderr=False, quiet=quiet or json_output)
        metrics = benchmark_airpassengers(
            horizon=horizon,
            input_size=input_size,
            max_steps=max_steps,
            neural_loss_name=neural_loss_name,
            output_markdown=output_markdown,
            output_plot=output_plot,
            output_conformal_plot=output_conformal_plot,
            log_path=log_path,
        )

        if json_output:
            typer.echo(json.dumps(metrics.to_dict(orient="records"), indent=2))
            return

        if quiet:
            return

        table = Table(title="AirPassengers benchmark")
        for column in metrics.columns:
            table.add_column(column)
        for row in metrics.itertuples(index=False, name=None):
            table.add_row(*[str(value) for value in row])
        console.print(table)
        console.print(f"Report written to {output_markdown}")
        console.print(f"Plot written to {output_plot}")
        console.print(f"Conformal plot written to {output_conformal_plot}")

    return app


if __name__ == "__main__":
    build_app()()
