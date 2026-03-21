"""Generate a synthetic series plot for documentation."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import typer
from neuralforecast import NeuralForecast
from neuralforecast.models import DLinear
from rich.console import Console
from statsforecast import StatsForecast
from statsforecast.models import AutoMFLES

from tests.utils.synthetic_series import make_synthetic_series
from timebaseula.models.timebase import TimeBase, TimeBaseTrend

app = typer.Typer(help="Generate a synthetic series plot for documentation.")
console = Console()

LOG_PATH = Path("logs") / "synthetic_plot.log"


def configure_logging() -> logging.Logger:
    """Configure structured logging for script execution."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("synthetic_plot")
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


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@app.command()
def main(
    output: Path = typer.Option(
        Path("docs/img/synthetic_series.png"),
        help="Path where the plot image will be saved.",
    ),
    title: str = typer.Option(
        "Synthetic series",
        help="Plot title for the generated series.",
    ),
    length: int = typer.Option(360, help="Number of points in the series."),
    noise_std: float = typer.Option(0.05, help="Noise standard deviation."),
    include_trend: bool = typer.Option(True, help="Include trend component."),
    include_seasonality: bool = typer.Option(True, help="Include seasonality."),
    season_period: int = typer.Option(24, help="Seasonal period length."),
    amplitude_period: int = typer.Option(
        48,
        help="Amplitude modulation period (0 to disable).",
    ),
    amplitude_strength: float = typer.Option(
        0.3,
        help="Amplitude modulation strength.",
    ),
    amplitude_growth_rate: float = typer.Option(
        0.0,
        help="Linear growth rate for amplitude modulation.",
    ),
    forecast_horizon: int = typer.Option(
        24,
        help="Forecast horizon for reference models.",
    ),
    include_reference: bool = typer.Option(
        True,
        help="Overlay a DLinear reference forecast.",
    ),
    include_timebase: bool = typer.Option(
        True,
        help="Overlay a TimeBase reference forecast.",
    ),
    include_timebase_trend: bool = typer.Option(
        True,
        help="Overlay a TimeBaseTrend reference forecast.",
    ),
    include_mfles: bool = typer.Option(
        True,
        help="Overlay an AutoMFLES reference forecast.",
    ),
) -> None:
    """Generate and save a synthetic series plot."""
    logger = configure_logging()
    matplotlib.use("Agg")

    logger.info("Generating synthetic series plot", extra={"output": str(output)})

    resolved_amplitude_period = None if amplitude_period <= 0 else amplitude_period

    frame = make_synthetic_series(
        length=length,
        noise_std=noise_std,
        include_trend=include_trend,
        include_seasonality=include_seasonality,
        season_period=season_period,
        amplitude_period=resolved_amplitude_period,
        amplitude_strength=amplitude_strength,
        amplitude_growth_rate=amplitude_growth_rate,
    )

    output.parent.mkdir(parents=True, exist_ok=True)

    forecast_frame = None
    mfles_forecast = None
    if (
        include_reference or include_timebase or include_timebase_trend
    ) and length > forecast_horizon:
        input_size = min(48, length - forecast_horizon)
        models = []
        if include_reference:
            models.append(
                DLinear(
                    h=forecast_horizon,
                    input_size=input_size,
                    max_steps=200,
                    learning_rate=1e-2,
                )
            )
        if include_timebase:
            models.append(
                TimeBase(
                    h=forecast_horizon,
                    input_size=input_size,
                    period_len=24,
                    basis_num=6,
                    max_steps=200,
                    learning_rate=1e-2,
                )
            )
        if include_timebase_trend:
            models.append(
                TimeBaseTrend(
                    h=forecast_horizon,
                    input_size=input_size,
                    period_len=24,
                    basis_num=6,
                    moving_avg_window=25,
                    max_steps=200,
                    learning_rate=1e-2,
                )
            )
        nf = NeuralForecast(models=models, freq="D")
        train_frame = frame.iloc[:-forecast_horizon]
        nf.fit(train_frame, val_size=forecast_horizon)
        forecast_frame = nf.predict()

    # MFLES forecast using statsforecast
    if include_mfles and length > forecast_horizon:
        train_frame = frame.iloc[:-forecast_horizon].copy()
        mfles_model = AutoMFLES(test_size=forecast_horizon, season_length=season_period)
        sf = StatsForecast(models=[mfles_model], freq="D", verbose=False)
        sf.fit(train_frame)
        mfles_pred = sf.predict(h=forecast_horizon)
        mfles_forecast = mfles_pred.reset_index(drop=True)
        mfles_forecast["ds"] = frame.tail(forecast_horizon)["ds"].values

    plt.figure(figsize=(10, 4))
    plt.plot(frame["ds"], frame["y"], label="synthetic_series")
    if forecast_frame is not None:
        target = frame.tail(forecast_horizon).set_index("ds")
        if "DLinear" in forecast_frame.columns:
            dlinear_pred = forecast_frame.set_index("ds")["DLinear"]
            dlinear_mae = float(
                np.mean(np.abs(target["y"] - dlinear_pred.loc[target.index]))
            )
            dlinear_params = sum(
                p.numel() for p in nf.models[0].parameters() if p.requires_grad
            )
            plt.plot(
                forecast_frame["ds"],
                forecast_frame["DLinear"],
                label=f"dlinear (MAE {dlinear_mae:.3f}, {dlinear_params} params)",
                linestyle="--",
            )
        if "TimeBase" in forecast_frame.columns:
            timebase_pred = forecast_frame.set_index("ds")["TimeBase"]
            timebase_mae = float(
                np.mean(np.abs(target["y"] - timebase_pred.loc[target.index]))
            )
            # Find TimeBase model index
            timebase_idx = 1 if "DLinear" in forecast_frame.columns else 0
            timebase_params = sum(
                p.numel()
                for p in nf.models[timebase_idx].parameters()
                if p.requires_grad
            )
            plt.plot(
                forecast_frame["ds"],
                forecast_frame["TimeBase"],
                label=f"timebase (MAE {timebase_mae:.3f}, {timebase_params} params)",
                linestyle=":",
            )
        if "TimeBaseTrend" in forecast_frame.columns:
            timebase_trend_pred = forecast_frame.set_index("ds")["TimeBaseTrend"]
            timebase_trend_mae = float(
                np.mean(np.abs(target["y"] - timebase_trend_pred.loc[target.index]))
            )
            # Find TimeBaseTrend model index
            model_count = len(
                [
                    m
                    for m in [
                        include_reference,
                        include_timebase,
                        include_timebase_trend,
                    ]
                    if m
                ]
            )
            timebase_trend_idx = model_count - 1
            timebase_trend_params = sum(
                p.numel()
                for p in nf.models[timebase_trend_idx].parameters()
                if p.requires_grad
            )
            plt.plot(
                forecast_frame["ds"],
                forecast_frame["TimeBaseTrend"],
                label=f"tbt (MAE {timebase_trend_mae:.3f}, {timebase_trend_params}p)",
                linestyle="-.",
            )
    if mfles_forecast is not None and "AutoMFLES" in mfles_forecast.columns:
        target = frame.tail(forecast_horizon).set_index("ds")
        mfles_pred = mfles_forecast.set_index("ds")["AutoMFLES"]
        mfles_mae = float(np.mean(np.abs(target["y"] - mfles_pred.loc[target.index])))
        plt.plot(
            mfles_forecast["ds"],
            mfles_forecast["AutoMFLES"],
            label=f"mfles (MAE {mfles_mae:.3f})",
            linestyle="--",
            color="purple",
        )
    plt.title(title)
    plt.xlabel("date")
    plt.ylabel("value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()

    logger.info("Saved synthetic series plot", extra={"output": str(output)})
    console.print(f"Saved synthetic series plot to [bold]{output}[/bold]")


if __name__ == "__main__":
    app()
