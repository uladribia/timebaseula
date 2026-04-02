"""Tune neural models on the aggregated daily panel and optionally benchmark them."""

from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.benchmark_nixtla_panel import (
    DEFAULT_FREQ,
    DEFAULT_HORIZON,
    DEFAULT_INPUT_PATH,
    DEFAULT_MIN_COVERAGE,
    DEFAULT_MIN_TEST_POINTS,
    DEFAULT_MIN_TRAIN_POINTS,
    DEFAULT_TEST_RATIO,
    benchmark_daily_panel,
    estimate_runtime,
    filter_series_scope,
)

DEFAULT_OUTPUT_DIR = Path("artifacts/tuning/aggregated")
DEFAULT_TUNED_CONFIG_PATH = DEFAULT_OUTPUT_DIR / "best_configs.json"
DEFAULT_REPORT_PATH = DEFAULT_OUTPUT_DIR / "report.json"
DEFAULT_BENCHMARK_MARKDOWN = Path("docs/daily-panel-aggregated-benchmark.md")
DEFAULT_BENCHMARK_DIR = Path("docs/img/daily-panel-aggregated-benchmark")
DEFAULT_LOG_PATH = Path("logs/tune_nixtla_panel_aggregated.log")
DEFAULT_MAX_SERIES = 64
DEFAULT_NUM_SAMPLES = {"smoke": 1, "normal": 3, "heavy": 6}


def configure_logging(log_path: Path) -> logging.Logger:
    """Configure a rotating file logger for tuning runs."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("tune_nixtla_panel_aggregated")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=1)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def build_timebase_candidate_configs(profile: str) -> list[dict[str, int | float]]:
    """Return CPU-aware candidate settings for TimeBase tuning."""
    smoke = [
        {
            "input_size": 56,
            "basis_num": 6,
            "period_len": 7,
            "learning_rate": 1e-3,
            "max_steps": 80,
        },
        {
            "input_size": 84,
            "basis_num": 6,
            "period_len": 7,
            "learning_rate": 1e-3,
            "max_steps": 100,
        },
    ]
    normal = smoke + [
        {
            "input_size": 112,
            "basis_num": 8,
            "period_len": 7,
            "learning_rate": 7e-4,
            "max_steps": 140,
        },
        {
            "input_size": 84,
            "basis_num": 8,
            "period_len": 14,
            "learning_rate": 1e-3,
            "max_steps": 140,
        },
    ]
    heavy = normal + [
        {
            "input_size": 140,
            "basis_num": 8,
            "period_len": 7,
            "learning_rate": 5e-4,
            "max_steps": 220,
        },
        {
            "input_size": 112,
            "basis_num": 10,
            "period_len": 14,
            "learning_rate": 7e-4,
            "max_steps": 240,
        },
    ]
    return {"smoke": smoke, "normal": normal, "heavy": heavy}[profile]


def build_timebasetrend_candidate_configs(profile: str) -> list[dict[str, int | float]]:
    """Return CPU-aware candidate settings for TimeBaseTrend tuning."""
    smoke = [
        {
            "input_size": 84,
            "basis_num": 6,
            "period_len": 7,
            "moving_avg_window": 21,
            "learning_rate": 1e-3,
            "max_steps": 100,
        },
        {
            "input_size": 112,
            "basis_num": 6,
            "period_len": 7,
            "moving_avg_window": 29,
            "learning_rate": 1e-3,
            "max_steps": 120,
        },
    ]
    normal = smoke + [
        {
            "input_size": 112,
            "basis_num": 8,
            "period_len": 14,
            "moving_avg_window": 21,
            "learning_rate": 7e-4,
            "max_steps": 160,
        },
        {
            "input_size": 140,
            "basis_num": 8,
            "period_len": 7,
            "moving_avg_window": 35,
            "learning_rate": 7e-4,
            "max_steps": 180,
        },
    ]
    heavy = normal + [
        {
            "input_size": 168,
            "basis_num": 8,
            "period_len": 14,
            "moving_avg_window": 35,
            "learning_rate": 5e-4,
            "max_steps": 260,
        },
        {
            "input_size": 140,
            "basis_num": 10,
            "period_len": 7,
            "moving_avg_window": 21,
            "learning_rate": 7e-4,
            "max_steps": 240,
        },
    ]
    return {"smoke": smoke, "normal": normal, "heavy": heavy}[profile]


def sanitize_auto_config(
    raw_config: dict[str, Any],
    model_name: str,
) -> dict[str, int | float | str | bool | None]:
    """Reduce auto-model configs to benchmark-ready fields."""
    sanitized: dict[str, int | float | str | bool | None] = {
        "input_size": int(raw_config["input_size"]),
        "learning_rate": float(raw_config["learning_rate"]),
        "max_steps": int(raw_config["max_steps"]),
        "step_size": int(raw_config["step_size"]),
        "scaler_type": raw_config.get("scaler_type"),
    }
    if model_name in {"AutoTimeBase", "AutoTimeBaseTrend"}:
        sanitized["basis_num"] = int(raw_config["basis_num"])
        sanitized["period_len"] = int(raw_config["period_len"])
    if model_name in {"AutoDLinear", "AutoTimeBaseTrend"}:
        sanitized["moving_avg_window"] = int(raw_config["moving_avg_window"])
    return sanitized


def select_best_tuning_result(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Select the best result using normalized error first and raw MAE second."""
    return min(results, key=lambda row: (row["avg_mean_scaled_mae"], row["avg_mae"]))


def _build_auto_model_config(
    candidates: list[dict[str, int | float]],
    model_name: str,
) -> dict[str, Any]:
    """Convert compact benchmark candidates into a Ray Tune search space."""
    from ray import tune

    unique_values = {
        key: sorted({candidate[key] for candidate in candidates})
        for key in candidates[0]
    }
    config: dict[str, Any] = {
        "input_size": tune.choice(
            [int(value) for value in unique_values["input_size"]]
        ),
        "basis_num": tune.choice([int(value) for value in unique_values["basis_num"]]),
        "period_len": tune.choice(
            [int(value) for value in unique_values["period_len"]]
        ),
        "learning_rate": tune.choice(
            [float(value) for value in unique_values["learning_rate"]]
        ),
        "max_steps": tune.choice([int(value) for value in unique_values["max_steps"]]),
        "step_size": tune.choice([1, int(max(unique_values["period_len"]))]),
        "scaler_type": tune.choice(["identity"]),
        "batch_size": tune.choice([32, 64]),
        "windows_batch_size": tune.choice([256, 512, 1024]),
        "random_seed": tune.randint(lower=1, upper=20),
    }
    if model_name == "AutoTimeBaseTrend":
        config["moving_avg_window"] = tune.choice(
            [int(value) for value in unique_values["moving_avg_window"]]
        )
    return config


def _tune_auto_timebase_model(
    train_df: pd.DataFrame,
    horizon: int,
    profile: str,
    model_name: str,
    logger: logging.Logger,
) -> dict[str, int | float | str | bool | None]:
    """Tune one TimeBase-family model through NeuralForecast native auto utilities."""
    from neuralforecast import NeuralForecast
    from timebaseula import AutoTimeBase, AutoTimeBaseTrend

    model_map = {
        "AutoTimeBase": AutoTimeBase,
        "AutoTimeBaseTrend": AutoTimeBaseTrend,
    }
    candidate_map = {
        "AutoTimeBase": build_timebase_candidate_configs(profile),
        "AutoTimeBaseTrend": build_timebasetrend_candidate_configs(profile),
    }
    model_cls = model_map[model_name]
    config = _build_auto_model_config(candidate_map[model_name], model_name=model_name)
    num_samples = min(DEFAULT_NUM_SAMPLES[profile], len(candidate_map[model_name]))
    logger.info("Tuning native auto model %s with %s samples", model_name, num_samples)
    model = model_cls(
        h=horizon,
        config=config,
        num_samples=num_samples,
        cpus=1,
        gpus=0,
        verbose=False,
        alias=model_name,
        backend="ray",
    )
    nf = NeuralForecast(models=[model], freq=DEFAULT_FREQ)
    nf.fit(train_df, val_size=horizon)
    fitted_model = nf.models[0]
    return sanitize_auto_config(
        fitted_model.results.get_best_result().config,
        model_name=model_name,
    )


def tune_timebase_family(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    horizon: int,
    profile: str,
    logger: logging.Logger,
) -> dict[str, dict[str, int | float | str | bool | None]]:
    """Tune TimeBase wrappers with NeuralForecast native auto utilities."""
    del test_df
    return {
        model_name: _tune_auto_timebase_model(
            train_df=train_df,
            horizon=horizon,
            profile=profile,
            model_name=model_name,
            logger=logger,
        )
        for model_name in ("AutoTimeBase", "AutoTimeBaseTrend")
    }


def tune_native_auto_models(
    train_df: pd.DataFrame,
    horizon: int,
    profile: str,
    logger: logging.Logger,
) -> dict[str, dict[str, int | float | str | bool | None]]:
    """Tune DLinear and NLinear using native NeuralForecast auto models."""
    from neuralforecast import NeuralForecast
    from neuralforecast.auto import AutoDLinear, AutoNLinear

    num_samples = DEFAULT_NUM_SAMPLES[profile]
    best_configs: dict[str, dict[str, int | float | str | bool | None]] = {}
    for model_name, model_cls in {
        "AutoDLinear": AutoDLinear,
        "AutoNLinear": AutoNLinear,
    }.items():
        logger.info(
            "Tuning native auto model %s with %s samples", model_name, num_samples
        )
        model = model_cls(
            h=horizon,
            num_samples=num_samples,
            cpus=1,
            gpus=0,
            verbose=False,
            alias=model_name,
            backend="ray",
        )
        nf = NeuralForecast(models=[model], freq=DEFAULT_FREQ)
        nf.fit(train_df, val_size=horizon)
        fitted_model = nf.models[0]
        best_configs[model_name] = sanitize_auto_config(
            fitted_model.results.get_best_result().config,
            model_name=model_name,
        )
    return best_configs


def tune_aggregated_models(
    input_path: Path,
    output_dir: Path,
    tuned_config_path: Path,
    report_path: Path,
    test_ratio: float,
    horizon: int,
    max_series: int,
    min_train_points: int,
    min_test_points: int,
    min_coverage: float,
    profile: str,
    run_benchmark: bool,
    benchmark_markdown: Path,
    benchmark_dir: Path,
    log_path: Path,
) -> dict[str, dict[str, int | float | str | bool | None]]:
    """Tune aggregated models and optionally publish a benchmark with them."""
    from scripts.benchmark_nixtla_panel import select_benchmark_panel

    logger = configure_logging(log_path)
    logger.info("Reading prepared panel from %s", input_path)
    frame = pd.read_parquet(input_path)
    frame["ds"] = pd.to_datetime(frame["ds"])
    frame = filter_series_scope(frame, series_scope="aggregated")
    panel_df, holdout_train_df, holdout_test_df, dataset_summary = (
        select_benchmark_panel(
            frame=frame,
            test_ratio=test_ratio,
            horizon=horizon,
            max_series=max_series,
            min_train_points=min_train_points,
            min_test_points=min_test_points,
            min_coverage=min_coverage,
            series_scope="aggregated",
        )
    )
    del panel_df, dataset_summary
    model_train_df = holdout_train_df[["unique_id", "ds", "y"]].copy()
    model_test_df = holdout_test_df[["unique_id", "ds", "y"]].copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    best_configs = tune_timebase_family(
        train_df=model_train_df,
        test_df=model_test_df,
        horizon=horizon,
        profile=profile,
        logger=logger,
    )
    best_configs |= tune_native_auto_models(
        train_df=model_train_df,
        horizon=horizon,
        profile=profile,
        logger=logger,
    )

    tuned_config_path.write_text(json.dumps(best_configs, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(best_configs, indent=2), encoding="utf-8")
    logger.info("Wrote tuned configs to %s", tuned_config_path)

    if run_benchmark:
        benchmark_daily_panel(
            input_path=input_path,
            output_markdown=benchmark_markdown,
            output_dir=benchmark_dir,
            test_ratio=test_ratio,
            horizon=horizon,
            max_series=max_series,
            min_train_points=min_train_points,
            min_test_points=min_test_points,
            min_coverage=min_coverage,
            profile=profile,
            series_scope="aggregated",
            include_autotheta=True,
            tuned_config_path=tuned_config_path,
            neural_loss_name="mae",
            log_path=Path("logs/benchmark_nixtla_panel.log"),
        )

    return best_configs


def build_app() -> Any:
    """Build the Typer CLI application."""
    import typer
    from rich.console import Console
    from rich.panel import Panel

    app = typer.Typer(help="Tune aggregated-only daily benchmark models.")

    @app.command("run")
    def run(
        input_path: Path = typer.Option(
            DEFAULT_INPUT_PATH, help="Prepared Nixtla panel parquet path."
        ),
        output_dir: Path = typer.Option(
            DEFAULT_OUTPUT_DIR, help="Directory for tuned config artifacts."
        ),
        tuned_config_path: Path = typer.Option(
            DEFAULT_TUNED_CONFIG_PATH, help="Output JSON path for best tuned configs."
        ),
        report_path: Path = typer.Option(
            DEFAULT_REPORT_PATH,
            help="Output JSON path for the tuning summary artifact.",
        ),
        benchmark_markdown: Path = typer.Option(
            DEFAULT_BENCHMARK_MARKDOWN,
            help="Benchmark markdown to refresh after tuning.",
        ),
        benchmark_dir: Path = typer.Option(
            DEFAULT_BENCHMARK_DIR,
            help="Benchmark plot directory to refresh after tuning.",
        ),
        test_ratio: float = typer.Option(
            DEFAULT_TEST_RATIO,
            help="Approximate proportion of unique dates reserved for rolling test windows.",
        ),
        horizon: int = typer.Option(DEFAULT_HORIZON, help="Forecast horizon in days."),
        max_series: int = typer.Option(
            DEFAULT_MAX_SERIES, help="Maximum aggregated series to tune on."
        ),
        min_train_points: int = typer.Option(
            DEFAULT_MIN_TRAIN_POINTS, help="Minimum train observations per series."
        ),
        min_test_points: int = typer.Option(
            DEFAULT_MIN_TEST_POINTS, help="Minimum test observations per series."
        ),
        min_coverage: float = typer.Option(
            DEFAULT_MIN_COVERAGE, help="Minimum observed-date coverage."
        ),
        profile: str = typer.Option(
            "smoke", help="Tuning profile: smoke, normal, or heavy."
        ),
        run_benchmark: bool = typer.Option(
            True,
            "--run-benchmark/--no-run-benchmark",
            help="Refresh the aggregated benchmark with tuned configs after tuning.",
        ),
        log_path: Path = typer.Option(DEFAULT_LOG_PATH, help="Rotating log file path."),
        quiet: bool = typer.Option(
            False, "--quiet", help="Suppress human-readable console output."
        ),
        verbose: bool = typer.Option(
            False, "--verbose", help="Show benchmark-oriented tuning details."
        ),
    ) -> None:
        console = Console(stderr=False, quiet=quiet)
        if profile not in {"smoke", "normal", "heavy"}:
            raise typer.BadParameter("profile must be one of: smoke, normal, heavy")

        if verbose and not quiet:
            console.print(
                Panel(
                    "\n".join(
                        [
                            f"Input dataset: {input_path}",
                            f"Forecast horizon: {horizon} days",
                            f"Training profile: {profile}",
                            f"Aggregated max series: {max_series}",
                            f"Estimated runtime: {estimate_runtime(max_series=max_series, cv_windows=6, profile=profile)} plus auto-model tuning overhead.",
                            f"Run benchmark after tuning: {run_benchmark}",
                        ]
                    ),
                    title="Aggregated tuning configuration",
                )
            )

        best_configs = tune_aggregated_models(
            input_path=input_path,
            output_dir=output_dir,
            tuned_config_path=tuned_config_path,
            report_path=report_path,
            test_ratio=test_ratio,
            horizon=horizon,
            max_series=max_series,
            min_train_points=min_train_points,
            min_test_points=min_test_points,
            min_coverage=min_coverage,
            profile=profile,
            run_benchmark=run_benchmark,
            benchmark_markdown=benchmark_markdown,
            benchmark_dir=benchmark_dir,
            log_path=log_path,
        )
        if not quiet:
            console.print_json(json.dumps(best_configs, indent=2))

    return app


app = build_app()


if __name__ == "__main__":
    app()
