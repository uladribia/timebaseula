"""Check MAE across naive, DLinear, TimeBase, and MFLES forecasts."""

from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import typer
from neuralforecast import NeuralForecast
from neuralforecast.models import DLinear
from rich.console import Console
from rich.table import Table
from statsforecast import StatsForecast
from statsforecast.models import AutoMFLES

from devtools.reporting import (
    build_html_benchmark_report,
    build_representative_forecast_sections,
)
from timebaseula.models.timebase import AutoTimeBase, AutoTimeBaseTrend
from timebaseula.synthetic import make_synthetic_series

app = typer.Typer(help="Compare MAE across baseline and model forecasts.")
console = Console()

LOG_PATH = Path("logs") / "benchmark_synthetic.log"
DEFAULT_OUTPUT_CSV = Path("logs/synthetic_benchmark_results.csv")
DEFAULT_OUTPUT_HTML = Path("logs/synthetic_benchmark_report.html")


def synthetic_scenarios() -> dict[str, dict[str, float | int | None]]:
    """Return the named synthetic benchmark scenarios."""
    return {
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


def configure_logging() -> logging.Logger:
    """Configure structured logging for script execution."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("benchmark_synthetic")
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
    max_steps: int,
    return_forecasts: bool = False,
) -> dict[str, float] | tuple[dict[str, float], pd.DataFrame, dict[str, pd.DataFrame]]:
    """Fit neural and naive models and compute MAE for each forecast."""
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

    bounded_search_steps = max(1, min(10, max_steps))
    models = [
        DLinear(h=h, input_size=input_size, max_steps=max_steps, learning_rate=1e-2),
        AutoTimeBase(
            h=h,
            freq="D",
            max_steps=max_steps,
            search_max_steps=bounded_search_steps,
        ),
        AutoTimeBaseTrend(
            h=h,
            freq="D",
            max_steps=max_steps,
            search_max_steps=bounded_search_steps,
        ),
    ]
    nf = NeuralForecast(models=models, freq="D")
    nf.fit(train_frame, val_size=val_size)
    forecast = nf.predict()
    if "unique_id" not in forecast.columns:
        forecast = forecast.reset_index()

    merged = target.merge(forecast, on=["unique_id", "ds"], how="inner")
    results = {
        "naive": float(np.mean(np.abs(target["y_true"] - naive))),
        "dlinear": float(np.mean(np.abs(merged["y_true"] - merged["DLinear"]))),
        "timebase": float(np.mean(np.abs(merged["y_true"] - merged["AutoTimeBase"]))),
        "timebase_trend": float(
            np.mean(np.abs(merged["y_true"] - merged["AutoTimeBaseTrend"]))
        ),
    }
    if not return_forecasts:
        return results
    forecast_frames = {
        "Naive": target[["unique_id", "ds"]].assign(Naive=naive.to_numpy()),
        "DLinear": forecast[["unique_id", "ds", "DLinear"]].copy(),
        "AutoTimeBase": forecast[["unique_id", "ds", "AutoTimeBase"]].copy(),
        "AutoTimeBaseTrend": forecast[["unique_id", "ds", "AutoTimeBaseTrend"]].copy(),
    }
    return results, target[["unique_id", "ds", "y_true"]].copy(), forecast_frames


def evaluate_mfles(
    frame: pd.DataFrame,
    h: int,
    test_size: int,
    season_length: int = 24,
    return_forecasts: bool = False,
) -> dict[str, float] | tuple[dict[str, float], pd.DataFrame]:
    """Fit MFLES model and compute MAE for each series."""
    group_sizes = frame.groupby("unique_id")["ds"].transform("size")
    positions = frame.groupby("unique_id").cumcount()
    train_frame = frame[positions < (group_sizes - test_size)]
    target = frame.groupby("unique_id").tail(test_size).rename(columns={"y": "y_true"})

    all_errors: list[float] = []
    forecast_parts: list[pd.DataFrame] = []
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
        forecast_parts.append(merged[["unique_id", "ds", "AutoMFLES"]].copy())

    result = {"mfles": float(np.mean(all_errors))}
    if not return_forecasts:
        return result
    return result, pd.concat(forecast_parts, ignore_index=True)


def run_synthetic_benchmark(
    length: int,
    h: int,
    input_size: int,
    val_size: int,
    test_size: int,
    max_steps: int,
    return_artifacts: bool = False,
) -> (
    pd.DataFrame
    | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]
):
    """Run the synthetic benchmark and optionally return report artifacts."""
    variations = [
        ("series_a", 1.0, 0.0),
        ("series_b", 1.15, 0.5),
        ("series_c", 0.85, -0.3),
    ]
    rows: list[dict[str, float | str | int]] = []
    observed_parts: list[pd.DataFrame] = []
    target_parts: list[pd.DataFrame] = []
    forecast_parts: dict[str, list[pd.DataFrame]] = {}

    for name, params in synthetic_scenarios().items():
        base_frame = make_synthetic_series(
            length=length,
            noise_std=float(params["noise_std"]),
            include_trend=True,
            include_seasonality=True,
            season_period=24,
            amplitude_period=params["amplitude_period"],
            amplitude_strength=float(params["amplitude_strength"]),
            amplitude_growth_rate=float(params["amplitude_growth_rate"]),
        )
        frame = build_multivariate_frame(base_frame, variations)
        if return_artifacts:
            model_results, target_frame, model_forecasts = evaluate_models(
                frame,
                h,
                input_size,
                val_size,
                test_size,
                max_steps,
                return_forecasts=True,
            )
            mfles_results, mfles_forecast = evaluate_mfles(
                frame,
                h,
                test_size,
                return_forecasts=True,
            )
            observed_parts.append(frame.assign(scenario=name))
            target_parts.append(target_frame.assign(scenario=name))
            for model_name, forecast_frame in {
                **model_forecasts,
                "MFLES": mfles_forecast.rename(columns={"AutoMFLES": "MFLES"}),
            }.items():
                forecast_parts.setdefault(model_name, []).append(
                    forecast_frame.assign(scenario=name)
                )
        else:
            model_results = evaluate_models(
                frame,
                h,
                input_size,
                val_size,
                test_size,
                max_steps,
            )
            mfles_results = evaluate_mfles(frame, h, test_size)

        model_results.update(mfles_results)
        rows.extend(
            {
                "scenario": name,
                "model_name": model_name,
                "mae": mae,
            }
            for model_name, mae in [
                ("Naive", model_results["naive"]),
                ("DLinear", model_results["dlinear"]),
                ("AutoTimeBase", model_results["timebase"]),
                ("AutoTimeBaseTrend", model_results["timebase_trend"]),
                ("MFLES", model_results["mfles"]),
            ]
        )

    results_frame = pd.DataFrame(rows)
    if not return_artifacts:
        return results_frame
    return (
        results_frame,
        pd.concat(observed_parts, ignore_index=True),
        pd.concat(target_parts, ignore_index=True),
        {
            model_name: pd.concat(parts, ignore_index=True)
            for model_name, parts in forecast_parts.items()
        },
    )


def run_synthetic_mae_checks(
    length: int,
    h: int,
    input_size: int,
    val_size: int,
    test_size: int,
    max_steps: int = 200,
) -> pd.DataFrame:
    """Run the synthetic scenarios and return a long-format result table."""
    return run_synthetic_benchmark(
        length,
        h,
        input_size,
        val_size,
        test_size,
        max_steps,
        return_artifacts=False,
    )


def resolve_report_data_dir(
    report_data_dir: Path | None,
    csv_path: Path | None,
) -> Path | None:
    """Resolve the persisted report-data directory for synthetic runs."""
    if report_data_dir is not None:
        return report_data_dir
    if csv_path is None:
        return None
    return csv_path.with_suffix("").with_name(f"{csv_path.stem}_report_data")


def save_report_data(
    report_data_dir: Path,
    observed_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    forecast_frames: dict[str, pd.DataFrame],
) -> None:
    """Persist synthetic report inputs for later HTML rendering."""
    report_data_dir.mkdir(parents=True, exist_ok=True)
    observed_frame.to_parquet(report_data_dir / "observed.parquet", index=False)
    target_frame.to_parquet(report_data_dir / "target.parquet", index=False)
    manifest = {
        "observed": "observed.parquet",
        "target": "target.parquet",
        "forecasts": {},
    }
    for model_name, forecast_frame in forecast_frames.items():
        filename = f"forecast_{model_name}.parquet"
        forecast_frame.to_parquet(report_data_dir / filename, index=False)
        manifest["forecasts"][model_name] = filename
    (report_data_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def load_report_data(
    report_data_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]] | None:
    """Load persisted synthetic report inputs."""
    manifest_path = report_data_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    observed_frame = pd.read_parquet(report_data_dir / manifest["observed"])
    target_frame = pd.read_parquet(report_data_dir / manifest["target"])
    forecast_frames = {
        model_name: pd.read_parquet(report_data_dir / filename)
        for model_name, filename in manifest["forecasts"].items()
    }
    return observed_frame, target_frame, forecast_frames


def render_results_table(results_frame: pd.DataFrame) -> None:
    """Render the synthetic benchmark results as a Rich table."""
    pivoted = (
        results_frame.pivot(index="scenario", columns="model_name", values="mae")
        .reset_index()
        .rename_axis(columns=None)
    )

    table = Table(title="Synthetic scenario MAE")
    table.add_column("Scenario")
    table.add_column("Naive", justify="right")
    table.add_column("DLinear", justify="right")
    table.add_column("AutoTimeBase", justify="right")
    table.add_column("AutoTimeBaseTrend", justify="right")
    table.add_column("MFLES", justify="right")

    logger = configure_logging()
    for row in pivoted.itertuples(index=False):
        logger.info(
            "Scenario MAE",
            extra={
                "scenario": row.scenario,
                "naive": row.Naive,
                "dlinear": row.DLinear,
                "timebase": row.AutoTimeBase,
                "timebase_trend": row.AutoTimeBaseTrend,
                "mfles": row.MFLES,
            },
        )
        table.add_row(
            row.scenario,
            f"{row.Naive:.4f}",
            f"{row.DLinear:.4f}",
            f"{row.AutoTimeBase:.4f}",
            f"{row.AutoTimeBaseTrend:.4f}",
            f"{row.MFLES:.4f}",
        )

    console.print(table)


def run_command(
    length: int = typer.Option(360, help="Length of each synthetic series."),
    h: int = typer.Option(24, help="Forecast horizon."),
    input_size: int = typer.Option(48, help="Input window size."),
    val_size: int = typer.Option(24, help="Validation window size."),
    test_size: int = typer.Option(24, help="Test window size."),
    max_steps: int = typer.Option(200, help="Hard cap on neural training steps."),
    output_csv: Path | None = typer.Option(
        None,
        help="Optional CSV output path for benchmark results.",
    ),
    report_data_dir: Path | None = typer.Option(
        None,
        help=(
            "Directory used to persist representative-series inputs so reports "
            "can be regenerated without rerunning the benchmark."
        ),
    ),
) -> None:
    """Run MAE checks across synthetic scenarios."""
    resolved_report_data_dir = resolve_report_data_dir(report_data_dir, output_csv)
    if resolved_report_data_dir is not None:
        results_frame, observed_frame, target_frame, forecast_frames = (
            run_synthetic_benchmark(
                length,
                h,
                input_size,
                val_size,
                test_size,
                max_steps,
                return_artifacts=True,
            )
        )
        save_report_data(
            resolved_report_data_dir,
            observed_frame,
            target_frame,
            forecast_frames,
        )
        console.print(f"[green]report data saved[/green] {resolved_report_data_dir}")
    else:
        results_frame = run_synthetic_mae_checks(
            length,
            h,
            input_size,
            val_size,
            test_size,
            max_steps,
        )

    render_results_table(results_frame)
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        results_frame.to_csv(output_csv, index=False)
        console.print(f"[green]saved[/green] {output_csv}")


app.command("run")(run_command)


@app.command("report-html")
def report_html(
    input_csv: Path = typer.Option(
        DEFAULT_OUTPUT_CSV,
        help="CSV output path for the synthetic benchmark table.",
    ),
    output_html: Path = typer.Option(
        DEFAULT_OUTPUT_HTML,
        help="HTML report output path for the synthetic benchmark table.",
    ),
    report_data_dir: Path | None = typer.Option(
        None,
        help=(
            "Optional directory with persisted representative-series inputs. "
            "Defaults to a sibling directory derived from the CSV path."
        ),
    ),
) -> None:
    """Generate a reusable HTML report from persisted synthetic benchmark outputs."""
    results_frame = pd.read_csv(input_csv)
    representative_sections: list[str] | None = None
    resolved_report_data_dir = resolve_report_data_dir(report_data_dir, input_csv)
    if resolved_report_data_dir is not None:
        loaded = load_report_data(resolved_report_data_dir)
        if loaded is not None:
            observed_frame, target_frame, forecast_frames = loaded
            representative_sections = build_representative_forecast_sections(
                observed_frame,
                target_frame,
                forecast_frames,
                slice_columns=["scenario"],
                n_examples=5,
            )

    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(
        build_html_benchmark_report(
            results_frame,
            title="Synthetic benchmark report",
            source_label=str(input_csv),
            slice_columns=["scenario"],
            description="Reusable Matplotlib report for synthetic MAE benchmark scenarios.",
            representative_sections=representative_sections,
        ),
        encoding="utf-8",
    )
    console.print(f"[green]HTML report saved[/green] {output_html}")


if __name__ == "__main__":
    app()
