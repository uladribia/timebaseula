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

from timebaseula.synthetic import make_synthetic_series

app = typer.Typer(help="Evaluate DLinear MAE on synthetic scenarios.")
console = Console()

LOG_PATH = Path("logs") / "dlinear_mae.log"


def configure_logging() -> logging.Logger:
    """Configure structured logging for script execution."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("dlinear_mae")
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


def build_multivariate_frame(
    base_frame: pd.DataFrame,
    variations: list[tuple[str, float, float]],
) -> pd.DataFrame:
    """Create a multivariate dataset by scaling and shifting a base frame.

    Args:
        base_frame: Base synthetic frame with columns [unique_id, ds, y].
        variations: List of (unique_id, scale, offset) tuples.

    Returns:
        Concatenated DataFrame for multiple series.
    """
    frames: list[pd.DataFrame] = []
    for unique_id, scale, offset in variations:
        frame = base_frame.copy()
        frame["unique_id"] = unique_id
        frame["y"] = frame["y"] * scale + offset
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def evaluate_scenario(
    scenario_name: str,
    frame: pd.DataFrame,
    h: int,
    input_size: int,
    val_size: int,
    test_size: int,
) -> float:
    """Fit DLinear on a multivariate frame and compute MAE on the test split."""
    model = DLinear(h=h, input_size=input_size, max_steps=200, learning_rate=1e-2)
    nf = NeuralForecast(models=[model], freq="D")

    train_frame = frame.groupby("unique_id", group_keys=False).apply(
        lambda df: df.iloc[:-test_size], include_groups=False
    )
    target = frame.groupby("unique_id").tail(test_size).rename(columns={"y": "y_true"})

    nf.fit(train_frame, val_size=val_size)
    forecast = nf.predict()

    merged = target.merge(forecast, on=["unique_id", "ds"], how="inner")
    mae = np.mean(np.abs(merged["y_true"] - merged["DLinear"]))
    return float(mae)


@app.command()
def main(
    length: int = typer.Option(360, help="Length of each synthetic series."),
    h: int = typer.Option(24, help="Forecast horizon."),
    input_size: int = typer.Option(48, help="Input window size."),
    val_size: int = typer.Option(24, help="Validation window size."),
    test_size: int = typer.Option(24, help="Test window size."),
) -> None:
    """Run DLinear evaluations for easy/medium/hard synthetic scenarios."""
    logger = configure_logging()

    scenarios = {
        "easy": {
            "noise_std": 0.15,
            "amplitude_period": None,
            "amplitude_strength": 0.0,
            "amplitude_growth_rate": 0.0,
        },
        "medium": {
            "noise_std": 0.15,
            "amplitude_period": 48,
            "amplitude_strength": 0.4,
            "amplitude_growth_rate": 0.0,
        },
        "hard": {
            "noise_std": 0.15,
            "amplitude_period": 96,
            "amplitude_strength": 0.9,
            "amplitude_growth_rate": 1.2,
        },
    }

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
        mae = evaluate_scenario(name, frame, h, input_size, val_size, test_size)
        results[name] = mae
        logger.info("Scenario MAE", extra={"scenario": name, "mae": mae})

    for name, mae in results.items():
        console.print(f"{name}: MAE={mae:.4f}")


if __name__ == "__main__":
    app()
