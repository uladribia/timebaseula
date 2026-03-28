"""Benchmark TimeBaseUla and baseline models on a prepared daily panel dataset."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_INPUT_PATH = Path("data/processed/danone_panel/panel.parquet")
DEFAULT_OUTPUT_MARKDOWN = Path("docs/daily-panel-benchmark.md")
DEFAULT_OUTPUT_DIR = Path("docs/img/daily-panel-benchmark")
DEFAULT_LOG_PATH = Path("logs/benchmark_nixtla_panel.log")
DEFAULT_FREQ = "D"
DEFAULT_TEST_RATIO = 0.2
DEFAULT_HORIZON = 28
DEFAULT_MAX_SERIES = 256
DEFAULT_MIN_TRAIN_POINTS = 365
DEFAULT_MIN_TEST_POINTS = 90
DEFAULT_MIN_COVERAGE = 0.8
BASELINE_MODEL_NAME = "Naive"
SUMMARY_PLOT_NAME = "summary.png"
DISTRIBUTION_PLOT_NAME = "distribution.png"
FORECAST_PLOT_NAME = "forecast_examples.png"


@dataclass(frozen=True)
class BenchmarkDatasetSummary:
    """Summary metadata for the benchmark-ready panel subset."""

    n_rows: int
    n_series: int
    n_dates: int
    train_rows: int
    test_rows: int
    horizon: int
    cv_windows: int
    cv_test_size: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    selected_from_series: int


def configure_logging(log_path: Path) -> logging.Logger:
    """Configure a rotating file logger for benchmark runs."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("benchmark_nixtla_panel")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    handler = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=1)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def render_markdown_table(frame: pd.DataFrame) -> str:
    """Render a DataFrame as a markdown table without extra dependencies."""

    def _format_value(value: Any) -> str:
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)

    headers = list(frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_format_value(value) for value in row) + " |")
    return "\n".join(lines)


def split_panel_by_date_ratio(
    frame: pd.DataFrame,
    test_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, int, pd.Timestamp]:
    """Split the panel with a global date cutoff based on the requested ratio."""
    if not 0 < test_ratio < 1:
        msg = "test_ratio must be strictly between 0 and 1."
        raise ValueError(msg)

    unique_dates = pd.Index(sorted(pd.to_datetime(frame["ds"]).unique()))
    n_dates = len(unique_dates)
    if n_dates < 2:
        msg = "At least two unique dates are required to create a train/test split."
        raise ValueError(msg)

    horizon = max(1, int(round(n_dates * test_ratio)))
    if horizon >= n_dates:
        horizon = n_dates - 1
    cutoff_date = pd.Timestamp(unique_dates[-horizon - 1])
    train = frame.loc[frame["ds"] <= cutoff_date].copy()
    test = frame.loc[frame["ds"] > cutoff_date].copy()
    return (
        train.reset_index(drop=True),
        test.reset_index(drop=True),
        horizon,
        cutoff_date,
    )


def split_panel_by_horizon(
    frame: pd.DataFrame,
    horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep the last fixed horizon dates as the final holdout window."""
    unique_dates = pd.Index(sorted(pd.to_datetime(frame["ds"]).unique()))
    if horizon <= 0 or horizon >= len(unique_dates):
        msg = "horizon must be positive and smaller than the number of unique dates."
        raise ValueError(msg)
    cutoff_date = pd.Timestamp(unique_dates[-horizon - 1])
    train = frame.loc[frame["ds"] <= cutoff_date].copy()
    test = frame.loc[frame["ds"] > cutoff_date].copy()
    return train.reset_index(drop=True), test.reset_index(drop=True)


def filter_series_scope(frame: pd.DataFrame, series_scope: str) -> pd.DataFrame:
    """Filter the benchmark panel according to the requested series scope."""
    if series_scope == "all":
        return frame.copy()
    if series_scope == "aggregated":
        mask = frame["unique_id"].str.startswith(("pdv__", "sku__")) | (
            frame["unique_id"] == "total"
        )
        return frame.loc[mask].copy()
    if series_scope == "detailed":
        aggregate_mask = frame["unique_id"].str.startswith(("pdv__", "sku__")) | (
            frame["unique_id"] == "total"
        )
        return frame.loc[~aggregate_mask].copy()
    msg = f"Unsupported series_scope: {series_scope}"
    raise ValueError(msg)


def regularize_benchmark_panel(
    frame: pd.DataFrame,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """Regularize each series to a complete daily grid and fill missing targets."""
    regularized_frames: list[pd.DataFrame] = []

    for unique_id, series in frame.groupby("unique_id", sort=True):
        series = series.sort_values("ds").reset_index(drop=True)
        series_start = pd.Timestamp(start_date or series["ds"].min())
        series_end = pd.Timestamp(end_date or series["ds"].max())
        dense_index = pd.date_range(series_start, series_end, freq=DEFAULT_FREQ)
        dense = pd.DataFrame({"ds": dense_index})
        dense = dense.merge(series[["ds", "y"]], on="ds", how="left")
        dense["unique_id"] = unique_id
        dense["y"] = dense["y"].fillna(fill_value)
        dense["pdv"] = series["pdv"].iloc[0]
        dense["sku"] = series["sku"].iloc[0]
        regularized_frames.append(dense[["unique_id", "ds", "y", "pdv", "sku"]])

    return pd.concat(regularized_frames, ignore_index=True)


def select_benchmark_panel(
    frame: pd.DataFrame,
    test_ratio: float,
    horizon: int,
    max_series: int,
    min_train_points: int,
    min_test_points: int,
    min_coverage: float,
    series_scope: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, BenchmarkDatasetSummary]:
    """Filter and densify a manageable benchmark subset from the prepared panel."""
    scoped_frame = filter_series_scope(frame, series_scope=series_scope)
    ratio_train, ratio_test, ratio_horizon, cutoff_date = split_panel_by_date_ratio(
        scoped_frame, test_ratio=test_ratio
    )
    del ratio_train
    total_dates = int(scoped_frame["ds"].nunique())
    cv_windows = max(1, ratio_horizon // horizon)
    cv_test_size = cv_windows * horizon
    cv_cutoff_date = pd.Timestamp(
        sorted(scoped_frame["ds"].unique())[-cv_test_size - 1]
    )

    series_stats = (
        scoped_frame.groupby("unique_id")
        .agg(
            total_points=("ds", "size"),
            train_points=("ds", lambda values: int((values <= cv_cutoff_date).sum())),
            test_points=("ds", lambda values: int((values > cv_cutoff_date).sum())),
            total_y=("y", "sum"),
        )
        .assign(coverage=lambda stats: stats["total_points"] / total_dates)
    )

    eligible = series_stats.loc[
        (series_stats["train_points"] >= min_train_points)
        & (series_stats["test_points"] >= min_test_points)
        & (series_stats["coverage"] >= min_coverage)
    ].copy()
    if eligible.empty:
        msg = "No series satisfy the requested benchmark filters."
        raise ValueError(msg)

    selected_ids = (
        eligible.sort_values(
            ["coverage", "train_points", "test_points", "total_y"],
            ascending=[False, False, False, False],
        )
        .head(max_series)
        .index.tolist()
    )

    selected_frame = scoped_frame.loc[
        scoped_frame["unique_id"].isin(selected_ids)
    ].copy()
    min_date = pd.Timestamp(scoped_frame["ds"].min())
    max_date = pd.Timestamp(scoped_frame["ds"].max())
    regularized = regularize_benchmark_panel(
        selected_frame,
        start_date=min_date,
        end_date=max_date,
        fill_value=0.0,
    )
    final_train, final_test = split_panel_by_horizon(regularized, horizon=horizon)
    cv_train = regularized.loc[regularized["ds"] <= cv_cutoff_date].copy()
    cv_test = regularized.loc[regularized["ds"] > cv_cutoff_date].copy()

    summary = BenchmarkDatasetSummary(
        n_rows=int(len(regularized)),
        n_series=int(regularized["unique_id"].nunique()),
        n_dates=int(regularized["ds"].nunique()),
        train_rows=int(len(cv_train)),
        test_rows=int(len(cv_test)),
        horizon=int(horizon),
        cv_windows=int(cv_windows),
        cv_test_size=int(cv_test_size),
        train_start=pd.Timestamp(cv_train["ds"].min()),
        train_end=pd.Timestamp(cv_train["ds"].max()),
        test_start=pd.Timestamp(cv_test["ds"].min()),
        test_end=pd.Timestamp(cv_test["ds"].max()),
        selected_from_series=int(len(eligible)),
    )
    return regularized, final_train, final_test, summary


def _trainer_overrides(max_steps: int, learning_rate: float) -> dict[str, Any]:
    """Return compact CPU-first trainer settings for daily-panel experiments."""
    return {
        "max_steps": max_steps,
        "val_check_steps": max(10, min(max_steps, 25)),
        "learning_rate": learning_rate,
        "batch_size": 64,
        "windows_batch_size": 128,
        "random_seed": 1,
        "accelerator": "cpu",
        "devices": 1,
        "logger": False,
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "log_every_n_steps": 1,
    }


def get_daily_model_configs(
    profile: str,
    n_series: int,
    cv_windows: int,
    include_autotheta: bool = True,
) -> dict[str, dict[str, int | float]]:
    """Return simple profile-based model settings for daily benchmarks."""
    profile_steps = {
        "smoke": {"TimeBase": 40, "TimeBaseTrend": 50, "NLinear": 50, "DLinear": 50},
        "normal": {
            "TimeBase": 150,
            "TimeBaseTrend": 190,
            "NLinear": 150,
            "DLinear": 150,
        },
        "heavy": {
            "TimeBase": 320,
            "TimeBaseTrend": 380,
            "NLinear": 300,
            "DLinear": 300,
        },
    }
    if profile not in profile_steps:
        msg = f"Unsupported profile: {profile}"
        raise ValueError(msg)

    workload_scale = 1.0
    if n_series <= 16 and cv_windows <= 2:
        workload_scale = 1.0
    elif n_series <= 64 and cv_windows <= 4:
        workload_scale = 0.9
    elif n_series <= 256 and cv_windows <= 6:
        workload_scale = 0.8
    else:
        workload_scale = 0.7

    steps = {
        model_name: max(30, int(round(base_steps * workload_scale)))
        for model_name, base_steps in profile_steps[profile].items()
    }
    settings: dict[str, dict[str, int | float]] = {
        "TimeBase": {
            "input_size": 56,
            "max_steps": steps["TimeBase"],
            "learning_rate": 1e-3,
            "basis_num": 6,
            "period_len": 7,
        },
        "TimeBaseTrend": {
            "input_size": 84,
            "max_steps": steps["TimeBaseTrend"],
            "learning_rate": 1e-3,
            "basis_num": 6,
            "period_len": 7,
            "moving_avg_window": 21,
        },
        "NLinear": {
            "input_size": 56,
            "max_steps": steps["NLinear"],
            "learning_rate": 2e-3,
        },
        "DLinear": {
            "input_size": 56,
            "max_steps": steps["DLinear"],
            "learning_rate": 2e-3,
        },
        "AutoMFLES": {
            "season_length": 7,
        },
        "Naive": {},
    }
    if include_autotheta:
        settings["AutoTheta"] = {"season_length": 7}
    return settings


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


def _to_pandas_frame(frame: Any) -> pd.DataFrame:
    """Convert dataframe-like outputs from Nixtla libraries to pandas."""
    if isinstance(frame, pd.DataFrame):
        return frame
    if hasattr(frame, "to_pandas"):
        return frame.to_pandas()
    msg = f"Unsupported dataframe type: {type(frame)!r}"
    raise TypeError(msg)


def _normalize_forecast_frame(forecast: Any, model_name: str) -> pd.DataFrame:
    """Normalize model forecasts to a common three-column schema."""
    normalized = _to_pandas_frame(forecast).reset_index()
    return normalized[["unique_id", "ds", model_name]].rename(
        columns={model_name: "y_hat"}
    )


def _normalize_cv_frame(
    forecast: Any, model_name: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize cross-validation outputs to common schemas for actuals and forecasts."""
    normalized = _to_pandas_frame(forecast).reset_index()
    actual = normalized[["unique_id", "ds", "cutoff", "y"]].copy()
    prediction = normalized[["unique_id", "ds", "cutoff", model_name]].rename(
        columns={model_name: "y_hat"}
    )
    return actual, prediction


def load_tuned_model_configs(
    tuned_config_path: Path | None,
) -> dict[str, dict[str, Any]]:
    """Load optional tuned model configs from a JSON artifact."""
    if tuned_config_path is None:
        return {}
    return json.loads(tuned_config_path.read_text(encoding="utf-8"))


def _model_display_name(model: Any) -> str:
    """Return a stable display name for benchmark reporting."""
    alias = getattr(model, "alias", None)
    if isinstance(alias, str) and alias:
        return alias
    return type(model).__name__


def build_neural_models(
    horizon: int,
    settings: dict[str, dict[str, int | float]],
    tuned_model_configs: dict[str, dict[str, Any]],
) -> list[Any]:
    """Build the neural benchmark models, optionally including tuned variants."""
    from neuralforecast.models import DLinear, NLinear

    from timebaseula import TimeBase, TimeBaseTrend

    models: list[Any] = [
        TimeBase(
            h=horizon,
            freq=DEFAULT_FREQ,
            input_size=int(settings["TimeBase"]["input_size"]),
            basis_num=int(settings["TimeBase"]["basis_num"]),
            period_len=int(settings["TimeBase"]["period_len"]),
            **_trainer_overrides(
                max_steps=int(settings["TimeBase"]["max_steps"]),
                learning_rate=float(settings["TimeBase"]["learning_rate"]),
            ),
        ),
        TimeBaseTrend(
            h=horizon,
            freq=DEFAULT_FREQ,
            input_size=int(settings["TimeBaseTrend"]["input_size"]),
            basis_num=int(settings["TimeBaseTrend"]["basis_num"]),
            period_len=int(settings["TimeBaseTrend"]["period_len"]),
            moving_avg_window=int(settings["TimeBaseTrend"]["moving_avg_window"]),
            **_trainer_overrides(
                max_steps=int(settings["TimeBaseTrend"]["max_steps"]),
                learning_rate=float(settings["TimeBaseTrend"]["learning_rate"]),
            ),
        ),
        NLinear(
            h=horizon,
            input_size=int(settings["NLinear"]["input_size"]),
            **_trainer_overrides(
                max_steps=int(settings["NLinear"]["max_steps"]),
                learning_rate=float(settings["NLinear"]["learning_rate"]),
            ),
        ),
        DLinear(
            h=horizon,
            input_size=int(settings["DLinear"]["input_size"]),
            **_trainer_overrides(
                max_steps=int(settings["DLinear"]["max_steps"]),
                learning_rate=float(settings["DLinear"]["learning_rate"]),
            ),
        ),
    ]

    if "AutoTimeBase" in tuned_model_configs:
        config = tuned_model_configs["AutoTimeBase"]
        model = TimeBase(
            h=horizon,
            freq=DEFAULT_FREQ,
            input_size=int(config["input_size"]),
            basis_num=int(config["basis_num"]),
            period_len=int(config["period_len"]),
            **_trainer_overrides(
                max_steps=int(config["max_steps"]),
                learning_rate=float(config["learning_rate"]),
            ),
        )
        model.alias = "AutoTimeBase"
        models.append(model)
    if "AutoTimeBaseTrend" in tuned_model_configs:
        config = tuned_model_configs["AutoTimeBaseTrend"]
        model = TimeBaseTrend(
            h=horizon,
            freq=DEFAULT_FREQ,
            input_size=int(config["input_size"]),
            basis_num=int(config["basis_num"]),
            period_len=int(config["period_len"]),
            moving_avg_window=int(config["moving_avg_window"]),
            **_trainer_overrides(
                max_steps=int(config["max_steps"]),
                learning_rate=float(config["learning_rate"]),
            ),
        )
        model.alias = "AutoTimeBaseTrend"
        models.append(model)
    if "AutoNLinear" in tuned_model_configs:
        config = tuned_model_configs["AutoNLinear"]
        models.append(
            NLinear(
                h=horizon,
                input_size=int(config["input_size"]),
                alias="AutoNLinear",
                scaler_type=config["scaler_type"],
                step_size=int(config["step_size"]),
                **_trainer_overrides(
                    max_steps=int(config["max_steps"]),
                    learning_rate=float(config["learning_rate"]),
                ),
            )
        )
    if "AutoDLinear" in tuned_model_configs:
        config = tuned_model_configs["AutoDLinear"]
        models.append(
            DLinear(
                h=horizon,
                input_size=int(config["input_size"]),
                alias="AutoDLinear",
                scaler_type=config["scaler_type"],
                step_size=int(config["step_size"]),
                moving_avg_window=int(config["moving_avg_window"]),
                **_trainer_overrides(
                    max_steps=int(config["max_steps"]),
                    learning_rate=float(config["learning_rate"]),
                ),
            )
        )
    return models


def run_neuralforecast_models(
    train_df: pd.DataFrame,
    horizon: int,
    settings: dict[str, dict[str, int | float]],
    tuned_model_configs: dict[str, dict[str, Any]],
    logger: logging.Logger,
) -> tuple[dict[str, pd.DataFrame], dict[str, float], dict[str, float], dict[str, int]]:
    """Fit and predict the neural models used in the benchmark."""
    import warnings

    from neuralforecast import NeuralForecast

    models = build_neural_models(
        horizon=horizon,
        settings=settings,
        tuned_model_configs=tuned_model_configs,
    )

    forecasts: dict[str, pd.DataFrame] = {}
    training_times: dict[str, float] = {}
    inference_times: dict[str, float] = {}
    parameter_counts: dict[str, int] = {}

    for model in models:
        model_name = _model_display_name(model)
        parameter_counts[model_name] = count_trainable_parameters(model)
        logger.info("Training neural model %s for final 28-day holdout", model_name)
        nf = NeuralForecast(models=[model], freq=DEFAULT_FREQ)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=Warning)
            fit_start = perf_counter()
            nf.fit(train_df, val_size=horizon)
            training_times[model_name] = perf_counter() - fit_start
            predict_start = perf_counter()
            prediction = nf.predict()
            inference_times[model_name] = perf_counter() - predict_start
        forecasts[model_name] = _normalize_forecast_frame(prediction, model_name)

    return forecasts, training_times, inference_times, parameter_counts


def run_statsforecast_models(
    train_df: pd.DataFrame,
    horizon: int,
    settings: dict[str, dict[str, int | float]],
    logger: logging.Logger,
) -> tuple[dict[str, pd.DataFrame], dict[str, float], dict[str, float], dict[str, int]]:
    """Fit and predict the statistical models used in the benchmark."""
    from statsforecast import StatsForecast
    from statsforecast.models import AutoMFLES, AutoTheta, Naive

    models: list[Any] = [
        AutoMFLES(
            test_size=horizon,
            season_length=int(settings["AutoMFLES"]["season_length"]),
        )
    ]
    if "AutoTheta" in settings:
        models.append(
            AutoTheta(season_length=int(settings["AutoTheta"]["season_length"]))
        )
    models.append(Naive())

    forecasts: dict[str, pd.DataFrame] = {}
    training_times: dict[str, float] = {}
    inference_times: dict[str, float] = {}
    parameter_counts = {name: 0 for name in ("AutoMFLES", "Naive")}
    if "AutoTheta" in settings:
        parameter_counts["AutoTheta"] = 0

    for model in models:
        model_name = model.alias
        logger.info(
            "Training statistical model %s for final 28-day holdout", model_name
        )
        sf = StatsForecast(models=[model], freq=DEFAULT_FREQ, n_jobs=1)
        fit_start = perf_counter()
        sf.fit(train_df)
        training_times[model_name] = perf_counter() - fit_start
        predict_start = perf_counter()
        prediction = sf.predict(h=horizon)
        inference_times[model_name] = perf_counter() - predict_start
        forecasts[model_name] = prediction[["unique_id", "ds", model_name]].rename(
            columns={model_name: "y_hat"}
        )

    return forecasts, training_times, inference_times, parameter_counts


def run_neuralforecast_cross_validation(
    panel_df: pd.DataFrame,
    horizon: int,
    cv_windows: int,
    cv_test_size: int,
    settings: dict[str, dict[str, int | float]],
    tuned_model_configs: dict[str, dict[str, Any]],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Run rolling 28-day cross-validation for the neural models."""
    import warnings

    from neuralforecast import NeuralForecast

    models = build_neural_models(
        horizon=horizon,
        settings=settings,
        tuned_model_configs=tuned_model_configs,
    )

    actual: pd.DataFrame | None = None
    forecasts: dict[str, pd.DataFrame] = {}

    for model in models:
        model_name = _model_display_name(model)
        logger.info("Running neural cross-validation for %s", model_name)
        nf = NeuralForecast(models=[model], freq=DEFAULT_FREQ)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=Warning)
            cv_frame = nf.cross_validation(
                df=panel_df,
                n_windows=cv_windows,
                step_size=horizon,
                refit=False,
                verbose=False,
            )
        model_actual, model_forecast = _normalize_cv_frame(cv_frame, model_name)
        forecasts[model_name] = model_forecast
        if actual is None:
            actual = model_actual

    if actual is None:
        msg = "Neural cross-validation produced no outputs."
        raise ValueError(msg)
    return actual, forecasts


def run_statsforecast_cross_validation(
    panel_df: pd.DataFrame,
    horizon: int,
    cv_windows: int,
    cv_test_size: int,
    settings: dict[str, dict[str, int | float]],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Run rolling 28-day cross-validation for the statistical models."""
    from statsforecast import StatsForecast
    from statsforecast.models import AutoMFLES, AutoTheta, Naive

    models: list[Any] = [
        AutoMFLES(
            test_size=horizon,
            season_length=int(settings["AutoMFLES"]["season_length"]),
        )
    ]
    if "AutoTheta" in settings:
        models.append(
            AutoTheta(season_length=int(settings["AutoTheta"]["season_length"]))
        )
    models.append(Naive())

    actual: pd.DataFrame | None = None
    forecasts: dict[str, pd.DataFrame] = {}

    for model in models:
        model_name = model.alias
        logger.info("Running statistical cross-validation for %s", model_name)
        sf = StatsForecast(models=[model], freq=DEFAULT_FREQ, n_jobs=1)
        try:
            cv_frame = sf.cross_validation(
                df=panel_df,
                h=horizon,
                n_windows=cv_windows,
                step_size=horizon,
                refit=False,
            )
        except ValueError as error:
            logger.warning(
                "StatsForecast model %s does not support refit=False in cross_validation; falling back to refit=True. Error: %s",
                model_name,
                error,
            )
            cv_frame = sf.cross_validation(
                df=panel_df,
                h=horizon,
                n_windows=cv_windows,
                step_size=horizon,
                refit=True,
            )
        model_actual, model_forecast = _normalize_cv_frame(cv_frame, model_name)
        forecasts[model_name] = model_forecast
        if actual is None:
            actual = model_actual

    if actual is None:
        msg = "Statistical cross-validation produced no outputs."
        raise ValueError(msg)
    return actual, forecasts


def build_model_summary_table(
    actual: pd.DataFrame,
    forecasts: dict[str, pd.DataFrame],
    training_times: dict[str, float],
    inference_times: dict[str, float],
    parameter_counts: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate Nixtla metrics plus simple custom summaries into one table."""
    from utilsforecast.losses import mae, rmse, smape

    merge_keys = ["unique_id", "ds"]
    task_keys = ["unique_id"]
    if "cutoff" in actual.columns:
        merge_keys.append("cutoff")
        task_keys.append("cutoff")

    combined = actual.loc[:, merge_keys + ["y"]].copy()
    model_names = sorted(forecasts)
    for model_name in model_names:
        model_forecast = forecasts[model_name].rename(columns={"y_hat": model_name})
        combined = combined.merge(
            model_forecast,
            on=merge_keys,
            how="inner",
            validate="one_to_one",
        )

    if combined.empty:
        msg = "No overlapping predictions found across the evaluated models."
        raise ValueError(msg)

    metric_frames = []
    for metric_name, metric_function in [
        ("mae", mae),
        ("rmse", rmse),
        ("smape", smape),
    ]:
        metric_frame = _to_pandas_frame(metric_function(combined, models=model_names))
        metric_frames.append(
            metric_frame.melt(
                id_vars=task_keys,
                value_vars=model_names,
                var_name="model",
                value_name=metric_name,
            )
        )

    per_series_metrics = metric_frames[0]
    for metric_frame in metric_frames[1:]:
        per_series_metrics = per_series_metrics.merge(
            metric_frame,
            on=task_keys + ["model"],
            how="inner",
            validate="one_to_one",
        )

    mean_target = combined.groupby(task_keys, as_index=False).agg(y_mean=("y", "mean"))
    per_series_metrics = per_series_metrics.merge(
        mean_target,
        on=task_keys,
        how="left",
        validate="many_to_one",
    )
    per_series_metrics["mean_scaled_mae"] = np.where(
        per_series_metrics["y_mean"] > 0,
        per_series_metrics["mae"] / per_series_metrics["y_mean"],
        np.nan,
    )
    per_series_metrics["rank"] = per_series_metrics.groupby(task_keys)["mae"].rank(
        method="average",
        ascending=True,
    )

    summary = (
        per_series_metrics.groupby("model", as_index=False)
        .agg(
            avg_mae=("mae", "mean"),
            median_mae=("mae", "median"),
            avg_mean_scaled_mae=("mean_scaled_mae", "mean"),
            median_mean_scaled_mae=("mean_scaled_mae", "median"),
            avg_rmse=("rmse", "mean"),
            median_rmse=("rmse", "median"),
            avg_smape=("smape", "mean"),
            median_smape=("smape", "median"),
            avg_rank=("rank", "mean"),
            median_rank=("rank", "median"),
            wins=("rank", lambda values: int(np.sum(np.isclose(values, 1.0)))),
        )
        .assign(
            training_time_seconds=lambda frame: (
                frame["model"].map(training_times).astype(float)
            ),
            inference_time_seconds=lambda frame: (
                frame["model"].map(inference_times).astype(float)
            ),
            parameters=lambda frame: frame["model"].map(parameter_counts).astype(int),
        )
        .loc[
            :,
            [
                "model",
                "training_time_seconds",
                "inference_time_seconds",
                "parameters",
                "avg_mae",
                "median_mae",
                "avg_mean_scaled_mae",
                "median_mean_scaled_mae",
                "avg_rmse",
                "median_rmse",
                "avg_smape",
                "median_smape",
                "avg_rank",
                "median_rank",
                "wins",
            ],
        ]
        .sort_values(["avg_rank", "avg_mean_scaled_mae", "avg_mae", "model"])
        .reset_index(drop=True)
    )

    numeric_columns = [
        "training_time_seconds",
        "inference_time_seconds",
        "avg_mae",
        "median_mae",
        "avg_mean_scaled_mae",
        "median_mean_scaled_mae",
        "avg_rmse",
        "median_rmse",
        "avg_smape",
        "median_smape",
        "avg_rank",
        "median_rank",
    ]
    summary[numeric_columns] = summary[numeric_columns].round(4)
    per_series_metrics[["mae", "rmse", "mean_scaled_mae", "smape", "rank"]] = (
        per_series_metrics[["mae", "rmse", "mean_scaled_mae", "smape", "rank"]].round(4)
    )
    return summary, per_series_metrics


def estimate_runtime(max_series: int, cv_windows: int, profile: str) -> str:
    """Return a simple runtime estimate based on workload and training profile."""
    workload = max_series * max(cv_windows, 1)
    if profile == "smoke":
        if workload <= 32:
            return "roughly 2-10 minutes"
        if workload <= 128:
            return "roughly 10-25 minutes"
        return "roughly 25-60 minutes"
    if profile == "normal":
        if workload <= 64:
            return "roughly 10-30 minutes"
        if workload <= 256:
            return "roughly 30-90 minutes"
        return "roughly 1.5-4 hours"
    if workload <= 64:
        return "roughly 20-45 minutes"
    if workload <= 256:
        return "roughly 1-3 hours"
    return "roughly 3-8 hours"


def generate_report_comments(summary: pd.DataFrame) -> list[str]:
    """Generate a few concise comments from the benchmark summary table."""
    best_avg_rank = summary.iloc[0]
    fastest_train = summary.sort_values("training_time_seconds").iloc[0]
    fastest_inference = summary.sort_values("inference_time_seconds").iloc[0]
    best_mean_scaled_mae = summary.sort_values("avg_mean_scaled_mae").iloc[0]
    return [
        f"Best overall trade-off in this run: {best_avg_rank['model']} (average rank {best_avg_rank['avg_rank']}, wins {int(best_avg_rank['wins'])}).",
        f"Fastest training model on the final 28-day holdout: {fastest_train['model']} ({fastest_train['training_time_seconds']} s).",
        f"Fastest inference model on the final 28-day holdout: {fastest_inference['model']} ({fastest_inference['inference_time_seconds']} s).",
        f"Best average mean-scaled MAE across rolling 28-day windows: {best_mean_scaled_mae['model']} ({best_mean_scaled_mae['avg_mean_scaled_mae']}).",
    ]


def filter_plot_window(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    context_points: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep the full test window plus a short train context for readable plots."""
    if context_points <= 0:
        return train_df.iloc[0:0].copy(), test_df.copy()
    train_window = train_df.sort_values("ds").tail(context_points).copy()
    test_window = test_df.sort_values("ds").copy()
    return train_window, test_window


def _plot_scale_factor(unique_id: str) -> float:
    """Return a deterministic pseudo-random scale factor for anonymized plots."""
    digest = hashlib.sha256(unique_id.encode("utf-8")).hexdigest()
    raw_value = int(digest[:8], 16) / 0xFFFFFFFF
    return 0.5 + (2.5 * raw_value)


def anonymize_series_values(frame: pd.DataFrame, value_column: str) -> pd.DataFrame:
    """Scale plotted values by a deterministic factor to hide real units."""
    anonymized = frame.copy()
    anonymized[value_column] = anonymized[value_column] * _plot_scale_factor(
        str(anonymized["unique_id"].iloc[0])
    )
    return anonymized


def build_series_aliases(unique_ids: list[str]) -> dict[str, str]:
    """Map raw series identifiers to generic labels for public plots."""
    return {
        unique_id: f"Series {index}"
        for index, unique_id in enumerate(unique_ids, start=1)
    }


def save_summary_plots(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save summary plots for rank, wins, runtime, and accuracy-vs-time."""
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = summary.sort_values("avg_rank").reset_index(drop=True)
    models = ordered["model"].tolist()

    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].bar(models, ordered["avg_rank"], color="#1f77b4")
    axes[0, 0].set_title("Average rank")
    axes[0, 0].tick_params(axis="x", rotation=30)

    axes[0, 1].bar(models, ordered["wins"], color="#2ca02c")
    axes[0, 1].set_title("Number of wins")
    axes[0, 1].tick_params(axis="x", rotation=30)

    axes[1, 0].scatter(
        ordered["training_time_seconds"],
        ordered["avg_mae"],
        color="#ff7f0e",
        s=70,
    )
    for _, row in ordered.iterrows():
        axes[1, 0].annotate(
            row["model"],
            (row["training_time_seconds"], row["avg_mae"]),
            textcoords="offset points",
            xytext=(4, 4),
        )
    axes[1, 0].set_title("Accuracy vs training time")
    axes[1, 0].set_xlabel("Training time (s)")
    axes[1, 0].set_ylabel("Average MAE")

    axes[1, 1].bar(models, ordered["inference_time_seconds"], color="#9467bd")
    axes[1, 1].set_title("Inference time (s)")
    axes[1, 1].tick_params(axis="x", rotation=30)

    for axis in axes.flat:
        axis.grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def save_distribution_plot(per_series_metrics: pd.DataFrame, output_path: Path) -> None:
    """Save a boxplot for per-task mean-scaled MAE distributions by model."""
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_models = (
        per_series_metrics.groupby("model")["rank"].mean().sort_values().index.tolist()
    )
    cleaned_data: list[np.ndarray] = []
    cleaned_labels: list[str] = []
    for model_name in ordered_models:
        values = (
            per_series_metrics.loc[
                per_series_metrics["model"] == model_name, "mean_scaled_mae"
            ]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .to_numpy()
        )
        if len(values) == 0:
            continue
        cleaned_data.append(values)
        cleaned_labels.append(model_name)

    figure, axis = plt.subplots(figsize=(12, 6))
    if cleaned_data:
        axis.boxplot(cleaned_data, tick_labels=cleaned_labels, showfliers=False)
    else:
        axis.text(
            0.5,
            0.5,
            "No valid mean-scaled MAE values to plot.",
            ha="center",
            va="center",
        )
        axis.set_xticks([])
    axis.set_title("Rolling 28-day mean-scaled MAE distribution")
    axis.set_ylabel("mean-scaled MAE")
    axis.tick_params(axis="x", rotation=30)
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def save_forecast_examples_plot(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    forecasts: dict[str, pd.DataFrame],
    output_path: Path,
    max_series: int = 4,
    context_points: int = 56,
) -> None:
    """Save forecast examples focused on the holdout and nearby train context."""
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected_ids = (
        test_df.groupby("unique_id")["y"]
        .sum()
        .sort_values(ascending=False)
        .head(max_series)
        .index.tolist()
    )
    figure, axes = plt.subplots(
        len(selected_ids), 1, figsize=(14, 3.5 * len(selected_ids)), sharex=True
    )
    if len(selected_ids) == 1:
        axes = [axes]

    series_aliases = build_series_aliases(selected_ids)
    colors = {
        "TimeBase": "#1f77b4",
        "TimeBaseTrend": "#ff7f0e",
        "NLinear": "#2ca02c",
        "DLinear": "#d62728",
        "AutoTimeBase": "#1a55a5",
        "AutoTimeBaseTrend": "#ff9f40",
        "AutoNLinear": "#1f8a3d",
        "AutoDLinear": "#c11f27",
        "AutoMFLES": "#9467bd",
        "AutoTheta": "#17becf",
        "Naive": "#8c564b",
    }

    for axis, unique_id in zip(axes, selected_ids, strict=True):
        series_train = train_df.loc[train_df["unique_id"] == unique_id]
        series_test = test_df.loc[test_df["unique_id"] == unique_id]
        train_window, test_window = filter_plot_window(
            series_train,
            series_test,
            context_points=context_points,
        )
        anonymized_train_window = anonymize_series_values(
            train_window, value_column="y"
        )
        anonymized_test_window = anonymize_series_values(test_window, value_column="y")
        axis.plot(
            anonymized_train_window["ds"],
            anonymized_train_window["y"],
            color="black",
            linewidth=1.2,
            label="train",
        )
        axis.plot(
            anonymized_test_window["ds"],
            anonymized_test_window["y"],
            color="black",
            linestyle="--",
            linewidth=1.8,
            label="test",
        )

        for model_name, forecast in forecasts.items():
            series_forecast = forecast.loc[forecast["unique_id"] == unique_id]
            anonymized_forecast = anonymize_series_values(
                series_forecast,
                value_column="y_hat",
            )
            axis.plot(
                anonymized_forecast["ds"],
                anonymized_forecast["y_hat"],
                label=model_name,
                color=colors[model_name],
                linewidth=1.1,
            )

        axis.set_title(
            f"{series_aliases[unique_id]} (last {len(train_window)} train points + holdout)"
        )
        axis.grid(alpha=0.3)
        axis.set_ylabel("anonymized units")

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def render_model_settings_snippet(settings: dict[str, dict[str, int | float]]) -> str:
    """Render model settings as a JSON code block."""
    return "```python\nMODEL_SETTINGS = " + json.dumps(settings, indent=2) + "\n```"


def render_model_metrics_matrix(summary: pd.DataFrame) -> str:
    """Render the benchmark summary with models as columns and metrics as rows."""
    matrix = (
        summary.set_index("model")
        .transpose()
        .reset_index()
        .rename(columns={"index": "metric"})
    )
    integer_metrics = {"parameters", "wins"}
    for row_index, metric_name in enumerate(matrix["metric"]):
        if metric_name in integer_metrics:
            matrix.iloc[row_index, 1:] = matrix.iloc[row_index, 1:].astype(int)
    return render_markdown_table(matrix)


def render_markdown_report(
    summary: pd.DataFrame,
    dataset_summary: dict[str, Any],
    plot_paths: list[Path],
    comments: list[str],
    settings: dict[str, dict[str, Any]],
    profile: str,
) -> str:
    """Render the benchmark report as markdown suitable for repository docs."""
    best_model = summary.iloc[0]["model"]
    plot_section = "\n\n".join(
        f"![Benchmark plot]({plot_path.as_posix()})" for plot_path in plot_paths
    )
    comments_section = "\n".join(f"- {comment}" for comment in comments)
    cv_windows = dataset_summary.get("cv_windows", "n/a")
    cv_test_size = dataset_summary.get(
        "cv_test_size", dataset_summary.get("test_rows", "n/a")
    )
    n_rows = dataset_summary.get("n_rows", dataset_summary.get("train_rows", "n/a"))
    n_dates = dataset_summary.get("n_dates", "n/a")
    train_start = dataset_summary.get("train_start")
    train_end = dataset_summary.get("train_end")
    test_start = dataset_summary.get("test_start")
    test_end = dataset_summary.get("test_end")
    train_window = (
        f"{pd.Timestamp(train_start).date()} to {pd.Timestamp(train_end).date()}"
        if train_start is not None and train_end is not None
        else "n/a"
    )
    test_window = (
        f"{pd.Timestamp(test_start).date()} to {pd.Timestamp(test_end).date()}"
        if test_start is not None and test_end is not None
        else "n/a"
    )
    return f"""---
description: Benchmark report for a daily panel dataset using TimeBaseUla and baseline models.
---

# Daily panel benchmark

## TL;DR
- Best model in this run: `{best_model}`
- Benchmarked series: `{dataset_summary["n_series"]}`
- Rolling evaluation windows: `{cv_windows}`
- Rolling test size: `{cv_test_size}` days
- Forecast horizon: `{dataset_summary["horizon"]}` daily steps

## Dataset summary
- Total regularized rows: `{n_rows}`
- Total unique dates: `{n_dates}`
- Cross-validation train window: `{train_window}`
- Cross-validation test window: `{test_window}`
- Training profile: `{profile}`
- Training and inference times are measured on the final single `{dataset_summary["horizon"]}`-day holdout.
- Accuracy metrics are aggregated across rolling `{dataset_summary["horizon"]}`-day cross-validation windows.

## Aggregate metrics

{render_model_metrics_matrix(summary)}

## Reproducible model settings

{render_model_settings_snippet(settings)}

## Comments
{comments_section}

## Plots

{plot_section}
"""


def benchmark_daily_panel(
    input_path: Path,
    output_markdown: Path,
    output_dir: Path,
    test_ratio: float,
    horizon: int,
    max_series: int,
    min_train_points: int,
    min_test_points: int,
    min_coverage: float,
    profile: str,
    series_scope: str,
    include_autotheta: bool,
    tuned_config_path: Path | None,
    log_path: Path,
) -> tuple[pd.DataFrame, BenchmarkDatasetSummary, list[Path], list[str]]:
    """Run the benchmark and persist markdown and plot artifacts."""
    logger = configure_logging(log_path)
    logger.info("Reading prepared panel from %s", input_path)
    frame = pd.read_parquet(input_path)
    tuned_model_configs = load_tuned_model_configs(tuned_config_path)
    frame["ds"] = pd.to_datetime(frame["ds"])

    panel_df, holdout_train_df, holdout_test_df, dataset_summary = (
        select_benchmark_panel(
            frame=frame,
            test_ratio=test_ratio,
            horizon=horizon,
            max_series=max_series,
            min_train_points=min_train_points,
            min_test_points=min_test_points,
            min_coverage=min_coverage,
            series_scope=series_scope,
        )
    )
    logger.info(
        "Benchmark subset contains %s series, %s CV windows, and %s total regularized rows",
        dataset_summary.n_series,
        dataset_summary.cv_windows,
        dataset_summary.n_rows,
    )
    settings = get_daily_model_configs(
        profile=profile,
        n_series=dataset_summary.n_series,
        cv_windows=dataset_summary.cv_windows,
        include_autotheta=include_autotheta,
    )

    model_holdout_train_df = holdout_train_df[["unique_id", "ds", "y"]].copy()
    model_panel_df = panel_df[["unique_id", "ds", "y"]].copy()

    neural_holdout_forecasts, neural_training, neural_inference, neural_parameters = (
        run_neuralforecast_models(
            train_df=model_holdout_train_df,
            horizon=horizon,
            settings=settings,
            tuned_model_configs=tuned_model_configs,
            logger=logger,
        )
    )
    stats_holdout_forecasts, stats_training, stats_inference, stats_parameters = (
        run_statsforecast_models(
            train_df=model_holdout_train_df,
            horizon=horizon,
            settings=settings,
            logger=logger,
        )
    )
    holdout_forecasts = neural_holdout_forecasts | stats_holdout_forecasts
    training_times = neural_training | stats_training
    inference_times = neural_inference | stats_inference
    parameter_counts = neural_parameters | stats_parameters

    neural_actual, neural_cv_forecasts = run_neuralforecast_cross_validation(
        panel_df=model_panel_df,
        horizon=horizon,
        cv_windows=dataset_summary.cv_windows,
        cv_test_size=dataset_summary.cv_test_size,
        settings=settings,
        tuned_model_configs=tuned_model_configs,
        logger=logger,
    )
    stats_actual, stats_cv_forecasts = run_statsforecast_cross_validation(
        panel_df=model_panel_df,
        horizon=horizon,
        cv_windows=dataset_summary.cv_windows,
        cv_test_size=dataset_summary.cv_test_size,
        settings=settings,
        logger=logger,
    )
    if not neural_actual.equals(stats_actual):
        logger.info(
            "Neural and statistical cross-validation actual frames differ slightly; using neural actual frame as reference"
        )
    cv_forecasts = neural_cv_forecasts | stats_cv_forecasts

    summary, per_series_metrics = build_model_summary_table(
        actual=neural_actual,
        forecasts=cv_forecasts,
        training_times=training_times,
        inference_times=inference_times,
        parameter_counts=parameter_counts,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_plot = output_dir / SUMMARY_PLOT_NAME
    distribution_plot = output_dir / DISTRIBUTION_PLOT_NAME
    forecast_plot = output_dir / FORECAST_PLOT_NAME
    save_summary_plots(summary, summary_plot)
    save_distribution_plot(per_series_metrics, distribution_plot)
    save_forecast_examples_plot(
        holdout_train_df, holdout_test_df, holdout_forecasts, forecast_plot
    )

    comments = generate_report_comments(summary)
    plot_paths = [
        summary_plot.relative_to(output_markdown.parent),
        distribution_plot.relative_to(output_markdown.parent),
        forecast_plot.relative_to(output_markdown.parent),
    ]
    report = render_markdown_report(
        summary=summary,
        dataset_summary=asdict(dataset_summary),
        plot_paths=plot_paths,
        comments=comments,
        settings=settings | tuned_model_configs,
        profile=profile,
    )
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text(report, encoding="utf-8")

    logger.info("Benchmark report written to %s", output_markdown)
    return summary, dataset_summary, plot_paths, comments


def build_app() -> Any:
    """Build the Typer CLI application."""
    import typer
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    app = typer.Typer(
        help="Benchmark TimeBaseUla and baselines on a prepared daily panel dataset."
    )

    @app.command("run")
    def run(
        input_path: Path = typer.Option(
            DEFAULT_INPUT_PATH, help="Prepared Nixtla panel parquet path."
        ),
        output_markdown: Path = typer.Option(
            DEFAULT_OUTPUT_MARKDOWN, help="Markdown report output path."
        ),
        output_dir: Path = typer.Option(
            DEFAULT_OUTPUT_DIR, help="Directory for generated Matplotlib plots."
        ),
        test_ratio: float = typer.Option(
            DEFAULT_TEST_RATIO,
            help="Approximate proportion of unique dates reserved for rolling test windows.",
        ),
        horizon: int = typer.Option(
            DEFAULT_HORIZON,
            help="Forecast horizon in days. The benchmark always evaluates rolling windows of this size.",
        ),
        max_series: int = typer.Option(
            DEFAULT_MAX_SERIES, help="Maximum number of dense series to benchmark."
        ),
        min_train_points: int = typer.Option(
            DEFAULT_MIN_TRAIN_POINTS,
            help="Minimum number of train observations required per series before densification.",
        ),
        min_test_points: int = typer.Option(
            DEFAULT_MIN_TEST_POINTS,
            help="Minimum number of test observations required per series before densification.",
        ),
        min_coverage: float = typer.Option(
            DEFAULT_MIN_COVERAGE,
            help="Minimum observed-date coverage required before densification.",
        ),
        series_scope: str = typer.Option(
            "all",
            help="Series scope to benchmark: all, aggregated, or detailed.",
        ),
        profile: str = typer.Option(
            "normal",
            help="Training profile controlling neural iteration budgets: smoke, normal, or heavy.",
        ),
        include_autotheta: bool = typer.Option(
            True,
            "--include-autotheta/--no-include-autotheta",
            help="Whether to include AutoTheta in the statistical benchmark models.",
        ),
        tuned_config_path: Path | None = typer.Option(
            None,
            help="Optional JSON file with tuned neural model configs to add to the benchmark.",
        ),
        log_path: Path = typer.Option(DEFAULT_LOG_PATH, help="Rotating log file path."),
        json_output: bool = typer.Option(
            False, "--json", help="Emit the summary table as JSON."
        ),
        quiet: bool = typer.Option(
            False, "--quiet", help="Suppress human-readable console output."
        ),
        verbose: bool = typer.Option(
            False,
            "--verbose",
            help="Show additional benchmark details before and after the run.",
        ),
    ) -> None:
        """Run a CPU-first benchmark on a prepared daily panel dataset."""
        console = Console(stderr=False, quiet=quiet or json_output)

        if not input_path.exists():
            raise typer.BadParameter(f"Input path does not exist: {input_path}")
        if not 0 < test_ratio < 1:
            raise typer.BadParameter("test_ratio must be strictly between 0 and 1")
        if horizon <= 0:
            raise typer.BadParameter("horizon must be positive")
        if max_series <= 0:
            raise typer.BadParameter("max_series must be positive")
        if min_train_points <= 0 or min_test_points <= 0:
            raise typer.BadParameter("Minimum train and test points must be positive")
        if not 0 < min_coverage <= 1:
            raise typer.BadParameter("min_coverage must be in the interval (0, 1]")
        if series_scope not in {"all", "aggregated", "detailed"}:
            raise typer.BadParameter(
                "series_scope must be one of: all, aggregated, detailed"
            )
        if profile not in {"smoke", "normal", "heavy"}:
            raise typer.BadParameter("profile must be one of: smoke, normal, heavy")

        if verbose and not (quiet or json_output):
            ds_frame = pd.read_parquet(input_path, columns=["ds"])
            projected_dates = int(pd.to_datetime(ds_frame["ds"]).nunique())
            projected_windows = max(
                1, int(round(projected_dates * test_ratio)) // horizon
            )
            console.print(
                Panel(
                    "\n".join(
                        [
                            f"Input dataset: {input_path}",
                            f"Rolling test ratio: {test_ratio}",
                            f"Forecast horizon: {horizon} days",
                            f"Projected unique dates: {projected_dates}",
                            f"Projected rolling windows: {projected_windows}",
                            f"Training profile: {profile}",
                            f"Series scope: {series_scope}",
                            f"Include AutoTheta: {include_autotheta}",
                            f"Tuned config path: {tuned_config_path}",
                            f"Max series: {max_series}",
                            f"Minimum train points: {min_train_points}",
                            f"Minimum test points: {min_test_points}",
                            f"Minimum coverage: {min_coverage}",
                            f"Estimated runtime: {estimate_runtime(max_series=max_series, cv_windows=projected_windows, profile=profile)} on a 16 GB CPU-only machine.",
                        ]
                    ),
                    title="Benchmark configuration",
                )
            )

        summary, dataset_summary, plot_paths, comments = benchmark_daily_panel(
            input_path=input_path,
            output_markdown=output_markdown,
            output_dir=output_dir,
            test_ratio=test_ratio,
            horizon=horizon,
            max_series=max_series,
            min_train_points=min_train_points,
            min_test_points=min_test_points,
            min_coverage=min_coverage,
            profile=profile,
            series_scope=series_scope,
            include_autotheta=include_autotheta,
            tuned_config_path=tuned_config_path,
            log_path=log_path,
        )

        if json_output:
            typer.echo(json.dumps(summary.to_dict(orient="records"), indent=2))
            return

        if quiet:
            return

        table = Table(title="Daily panel benchmark")
        for column in summary.columns:
            table.add_column(column)
        for row in summary.itertuples(index=False, name=None):
            table.add_row(*[str(value) for value in row])
        console.print(table)
        console.print(f"Report written to {output_markdown}")
        for plot_path in plot_paths:
            console.print(f"Plot written to {output_markdown.parent / plot_path}")
        console.print(f"Log written to {log_path}")

        if verbose:
            summary_table = Table(title="Selected benchmark subset")
            summary_table.add_column("field")
            summary_table.add_column("value")
            for key, value in asdict(dataset_summary).items():
                summary_table.add_row(key, str(value))
            console.print(summary_table)
            console.print(
                Panel(
                    "\n".join(f"- {comment}" for comment in comments),
                    title="Auto-comments",
                )
            )

    return app


if __name__ == "__main__":
    build_app()()
