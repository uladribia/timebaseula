"""Evaluate DLinear MAE on synthetic scenarios."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from neuralforecast import NeuralForecast
from neuralforecast.models import DLinear
from rich.console import Console

from devtools.benchmark_synthetic import (
    average_cross_validation_predictions,
    build_multivariate_frame,
    synthetic_scenarios,
)
from timebaseula.synthetic import make_synthetic_series

app = typer.Typer(help="Evaluate DLinear MAE on synthetic scenarios.")
console = Console()

LOG_PATH = Path("logs") / "benchmark_synthetic_dlinear.log"


def configure_logging() -> logging.Logger:
    """Configure structured logging for script execution."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("benchmark_synthetic_dlinear")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler(LOG_PATH, maxBytes=5 * 1024 * 1024, backupCount=1)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def evaluate_scenario(
    frame: pd.DataFrame,
    h: int,
    input_size: int,
    test_size: int,
    refit: bool,
) -> float:
    """Evaluate DLinear with native NeuralForecast cross-validation."""
    model = DLinear(h=h, input_size=input_size, max_steps=200, learning_rate=1e-2)
    nf = NeuralForecast(models=[model], freq="D")
    forecast = nf.cross_validation(
        df=frame,
        n_windows=None,
        val_size=h,
        test_size=test_size,
        step_size=1,
        refit=refit,
    )
    if "unique_id" not in forecast.columns:
        forecast = forecast.reset_index()
    averaged = average_cross_validation_predictions(
        forecast[["unique_id", "ds", "y", "DLinear"]],
        "DLinear",
    )
    mae = np.mean(np.abs(averaged["y"] - averaged["DLinear"]))
    return float(mae)


@app.command()
def main(
    length: int = typer.Option(360, help="Length of each synthetic series."),
    h: int = typer.Option(24, help="Forecast horizon."),
    input_size: int = typer.Option(48, help="Input window size."),
    val_size: int = typer.Option(24, help="Validation window size."),
    test_size: int = typer.Option(24, help="Test window size."),
    refit: bool = typer.Option(
        False,
        "--refit/--no-refit",
        help="Whether to refit the model at each cross-validation window.",
    ),
) -> None:
    """Run DLinear evaluations for easy/medium/hard synthetic scenarios."""
    logger = configure_logging()

    del val_size
    scenarios = synthetic_scenarios()

    variations = [
        ("series_a", 1.0, 0.0),
        ("series_b", 1.15, 0.5),
        ("series_c", 0.85, -0.3),
    ]

    results: dict[str, float] = {}
    for name, params in scenarios.items():
        base_frame = make_synthetic_series(
            length=length,
            noise_std=params["noise_std"],
            include_trend=True,
            include_seasonality=True,
            season_period=24,
            amplitude_period=params["amplitude_period"],
            amplitude_strength=params["amplitude_strength"],
            amplitude_growth_rate=params["amplitude_growth_rate"],
        )
        frame = build_multivariate_frame(base_frame, variations)
        mae = evaluate_scenario(frame, h, input_size, test_size, refit)
        results[name] = mae
        logger.info("Scenario MAE", extra={"scenario": name, "mae": mae})

    for name, mae in results.items():
        console.print(f"{name}: MAE={mae:.4f}")


if __name__ == "__main__":
    app()
