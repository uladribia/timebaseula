"""Generate a synthetic series visualization for documentation using Matplotlib."""

from __future__ import annotations

import base64
import logging
from io import BytesIO
from logging.handlers import RotatingFileHandler
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import typer
from matplotlib import pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.models import DLinear
from rich.console import Console
from statsforecast import StatsForecast
from statsforecast.models import AutoMFLES

from timebaseula import recommend_timebase_kwargs, recommend_timebase_trend_kwargs
from timebaseula.models.timebase import TimeBase, TimeBaseTrend
from timebaseula.synthetic import make_synthetic_series

plt.switch_backend("Agg")

app = typer.Typer(help="Generate a synthetic series visualization for documentation.")
console = Console()

LOG_PATH = Path("logs") / "synthetic_plot.log"
DEFAULT_OUTPUT = Path("docs/img/synthetic_series.html")


def configure_logging() -> logging.Logger:
    """Configure structured logging for script execution."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("synthetic_plot")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler(LOG_PATH, maxBytes=5 * 1024 * 1024, backupCount=1)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logger.addHandler(handler)
    return logger


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def build_series_layer(
    frame: pd.DataFrame, label: str, value_column: str
) -> pd.DataFrame:
    """Convert a forecast or observed frame to a common plotting schema."""
    return (
        frame[["ds", value_column]]
        .rename(columns={value_column: "value"})
        .assign(series=label)
    )


def render_chart_html(fig: plt.Figure, title: str) -> str:
    """Render a Matplotlib figure as a standalone HTML document."""
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
    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>{title}</title>
  <style>
    body {{ margin: 0; padding: 24px; font-family: Roboto, Arial, sans-serif; background: #ffffff; color: #111827; }}
    main {{ max-width: 1100px; margin: 0 auto; }}
    img {{ width: 100%; height: auto; display: block; }}
  </style>
</head>
<body>
  <main>
    <img src=\"data:image/png;base64,{encoded}\" alt=\"{title}\">
  </main>
</body>
</html>
"""


@app.command()
def main(
    output: Path = typer.Option(
        DEFAULT_OUTPUT, help="Path where the HTML visualization will be saved."
    ),
    title: str = typer.Option(
        "Synthetic series", help="Chart title for the generated series."
    ),
    length: int = typer.Option(360, help="Number of points in the series."),
    noise_std: float = typer.Option(0.05, help="Noise standard deviation."),
    include_trend: bool = typer.Option(True, help="Include trend component."),
    include_seasonality: bool = typer.Option(True, help="Include seasonality."),
    season_period: int = typer.Option(24, help="Seasonal period length."),
    amplitude_period: int = typer.Option(
        48, help="Amplitude modulation period (0 to disable)."
    ),
    amplitude_strength: float = typer.Option(
        0.3, help="Amplitude modulation strength."
    ),
    amplitude_growth_rate: float = typer.Option(
        0.0, help="Linear growth rate for amplitude modulation."
    ),
    forecast_horizon: int = typer.Option(
        24, help="Forecast horizon for reference models."
    ),
    include_reference: bool = typer.Option(
        True, help="Overlay a DLinear reference forecast."
    ),
    include_timebase: bool = typer.Option(
        True, help="Overlay a TimeBase reference forecast."
    ),
    include_timebase_trend: bool = typer.Option(
        True, help="Overlay a TimeBaseTrend reference forecast."
    ),
    include_mfles: bool = typer.Option(
        True, help="Overlay an AutoMFLES reference forecast."
    ),
) -> None:
    """Generate and save a synthetic series chart."""
    logger = configure_logging()
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
    plot_parts = [build_series_layer(frame, label="synthetic_series", value_column="y")]

    if (
        include_reference or include_timebase or include_timebase_trend
    ) and length > forecast_horizon:
        train_frame = frame.iloc[:-forecast_horizon].copy()
        target = frame.tail(forecast_horizon).set_index("ds")
        timebase_kwargs = recommend_timebase_kwargs(
            train_frame, freq="D", horizon=forecast_horizon, max_steps=200
        )
        timebase_trend_kwargs = recommend_timebase_trend_kwargs(
            train_frame, freq="D", horizon=forecast_horizon, max_steps=200
        )
        models: list[torch.nn.Module] = []
        labels: list[str] = []
        if include_reference:
            models.append(
                DLinear(
                    h=forecast_horizon,
                    input_size=int(timebase_kwargs["input_size"]),
                    max_steps=200,
                    learning_rate=1e-2,
                )
            )
            labels.append("DLinear")
        if include_timebase:
            models.append(TimeBase(h=forecast_horizon, **timebase_kwargs))
            labels.append("TimeBase")
        if include_timebase_trend:
            models.append(TimeBaseTrend(h=forecast_horizon, **timebase_trend_kwargs))
            labels.append("TimeBaseTrend")
        nf = NeuralForecast(models=models, freq="D")
        nf.fit(train_frame, val_size=forecast_horizon)
        forecast_frame = nf.predict()
        for model, label in zip(models, labels, strict=True):
            pred = forecast_frame.set_index("ds")[label]
            mae = float(np.mean(np.abs(target["y"] - pred.loc[target.index])))
            plot_parts.append(
                build_series_layer(
                    forecast_frame.assign(**{label: forecast_frame[label]}),
                    label=f"{label} (MAE {mae:.3f}, {count_parameters(model)} params)",
                    value_column=label,
                )
            )

    if include_mfles and length > forecast_horizon:
        train_frame = frame.iloc[:-forecast_horizon].copy()
        sf = StatsForecast(
            models=[AutoMFLES(test_size=forecast_horizon, season_length=season_period)],
            freq="D",
            verbose=False,
        )
        sf.fit(train_frame)
        mfles_forecast = sf.predict(h=forecast_horizon).reset_index(drop=True)
        mfles_forecast["ds"] = frame.tail(forecast_horizon)["ds"].to_numpy()
        target = frame.tail(forecast_horizon).set_index("ds")
        mfles_pred = mfles_forecast.set_index("ds")["AutoMFLES"]
        mfles_mae = float(np.mean(np.abs(target["y"] - mfles_pred.loc[target.index])))
        plot_parts.append(
            build_series_layer(
                mfles_forecast,
                label=f"MFLES (MAE {mfles_mae:.3f})",
                value_column="AutoMFLES",
            )
        )

    plot_frame = pd.concat(plot_parts, ignore_index=True)
    fig, ax = plt.subplots(figsize=(11, 4.3), facecolor="#ffffff")
    ax.set_facecolor("#ffffff")
    for series_name, series_frame in plot_frame.groupby("series", sort=False):
        ax.plot(
            series_frame["ds"], series_frame["value"], linewidth=2.0, label=series_name
        )
    ax.set_title(title)
    ax.set_xlabel("date")
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    output.write_text(render_chart_html(fig, title), encoding="utf-8")
    logger.info("Saved synthetic series plot", extra={"output": str(output)})
    console.print(f"Saved synthetic series plot to [bold]{output}[/bold]")


if __name__ == "__main__":
    app()
