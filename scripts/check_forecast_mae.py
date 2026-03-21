"""Check MAE across naive, DLinear, TimeBase, and MFLES forecasts."""

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
from rich.table import Table
from statsforecast import StatsForecast
from statsforecast.models import AutoMFLES

from tests.utils.synthetic_series import make_synthetic_series
from timebaseula.models.timebase import TimeBase, TimeBaseTrend

app = typer.Typer(help="Compare MAE across baseline and model forecasts.")
console = Console()

LOG_PATH = Path("logs") / "forecast_mae_check.log"


def configure_logging() -> logging.Logger:
    """Configure structured logging for script execution."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("forecast_mae_check")
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
    """Create a multivariate dataset by scaling and shifting a base frame."""
    frames: list[pd.DataFrame] = []
    for unique_id, scale, offset in variations:
        frame = base_frame.copy()
        frame["unique_id"] = unique_id
        frame["y"] = frame["y"] * scale + offset
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def build_naive_forecast(target: pd.DataFrame, last_values: pd.Series) -> pd.Series:
    """Build a naive forecast that repeats the last observed value."""
    return target["unique_id"].map(last_values)


def evaluate_models(
    frame: pd.DataFrame,
    h: int,
    input_size: int,
    val_size: int,
    test_size: int,
) -> dict[str, float]:
    """Fit models and compute MAE for each forecast."""
    group_sizes = frame.groupby("unique_id")["ds"].transform("size")
    positions = frame.groupby("unique_id").cumcount()
    train_frame = frame[positions < (group_sizes - test_size)]
    target = frame.groupby("unique_id").tail(test_size).rename(columns={"y": "y_true"})

    last_values = (
        train_frame.groupby("unique_id", as_index=False)
        .tail(1)
        .set_index("unique_id")["y"]
    )
    naive = build_naive_forecast(target, last_values)

    models = [
        DLinear(h=h, input_size=input_size, max_steps=200, learning_rate=1e-2),
        TimeBase(
            h=h,
            input_size=input_size,
            period_len=24,
            basis_num=6,
            max_steps=200,
            learning_rate=1e-2,
        ),
        TimeBaseTrend(
            h=h,
            input_size=input_size,
            period_len=24,
            basis_num=6,
            moving_avg_window=25,
            max_steps=200,
            learning_rate=1e-2,
        ),
    ]
    nf = NeuralForecast(models=models, freq="D")
    nf.fit(train_frame, val_size=val_size)
    forecast = nf.predict()

    merged = target.merge(forecast, on=["unique_id", "ds"], how="inner")
    return {
        "naive": float(np.mean(np.abs(target["y_true"] - naive))),
        "dlinear": float(np.mean(np.abs(merged["y_true"] - merged["DLinear"]))),
        "timebase": float(np.mean(np.abs(merged["y_true"] - merged["TimeBase"]))),
        "timebase_trend": float(
            np.mean(np.abs(merged["y_true"] - merged["TimeBaseTrend"]))
        ),
    }


def evaluate_mfles(
    frame: pd.DataFrame,
    h: int,
    test_size: int,
    season_length: int = 24,
) -> dict[str, float]:
    """Fit MFLES model and compute MAE for each series."""
    group_sizes = frame.groupby("unique_id")["ds"].transform("size")
    positions = frame.groupby("unique_id").cumcount()
    train_frame = frame[positions < (group_sizes - test_size)]
    target = frame.groupby("unique_id").tail(test_size).rename(columns={"y": "y_true"})

    # MFLES doesn't support multiple series, so process each separately
    all_errors: list[float] = []
    for series_id in train_frame["unique_id"].unique():
        series_train = train_frame[train_frame["unique_id"] == series_id].copy()
        series_target = target[target["unique_id"] == series_id].copy()

        mfles_model = AutoMFLES(test_size=test_size, season_length=season_length)
        sf = StatsForecast(models=[mfles_model], freq="D", verbose=False)
        sf.fit(series_train)
        mfles_forecast = sf.predict(h=h)

        merged = series_target.merge(
            mfles_forecast, on=["unique_id", "ds"], how="inner"
        )
        errors = np.abs(merged["y_true"] - merged["AutoMFLES"])
        all_errors.extend(errors.tolist())

    return {
        "mfles": float(np.mean(all_errors)),
    }


@app.command()
def main(
    length: int = typer.Option(360, help="Length of each synthetic series."),
    h: int = typer.Option(24, help="Forecast horizon."),
    input_size: int = typer.Option(48, help="Input window size."),
    val_size: int = typer.Option(24, help="Validation window size."),
    test_size: int = typer.Option(24, help="Test window size."),
) -> None:
    """Run MAE checks across synthetic scenarios."""
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

    table = Table(title="Synthetic scenario MAE")
    table.add_column("Scenario")
    table.add_column("Naive", justify="right")
    table.add_column("DLinear", justify="right")
    table.add_column("TimeBase", justify="right")
    table.add_column("TimeBaseTrend", justify="right")
    table.add_column("MFLES", justify="right")

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
        results = evaluate_models(frame, h, input_size, val_size, test_size)
        mfles_results = evaluate_mfles(frame, h, test_size)
        results.update(mfles_results)
        logger.info("Scenario MAE", extra={"scenario": name, **results})
        table.add_row(
            name,
            f"{results['naive']:.4f}",
            f"{results['dlinear']:.4f}",
            f"{results['timebase']:.4f}",
            f"{results['timebase_trend']:.4f}",
            f"{results['mfles']:.4f}",
        )

    console.print(table)


if __name__ == "__main__":
    app()
