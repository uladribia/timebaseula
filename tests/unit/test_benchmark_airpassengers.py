"""Tests for the AirPassengers benchmark script helpers."""

from __future__ import annotations

import pandas as pd

from scripts.benchmark_airpassengers import (
    DEFAULT_INPUT_SIZE,
    DEFAULT_MAX_STEPS,
    build_metrics_table,
    get_reproducible_model_settings,
    get_neural_model_configs,
    render_markdown_report,
    split_train_test,
)


def test_split_train_test_keeps_last_horizon_per_series() -> None:
    """The split helper should keep the last horizon in the test partition."""
    frame = pd.DataFrame(
        {
            "unique_id": ["a"] * 5 + ["b"] * 5,
            "ds": pd.date_range("2024-01-01", periods=5, freq="D").tolist() * 2,
            "y": list(range(5)) + list(range(10, 15)),
        }
    )

    train, test = split_train_test(frame, horizon=2)

    assert train.groupby("unique_id").size().to_dict() == {"a": 3, "b": 3}
    assert test.groupby("unique_id").size().to_dict() == {"a": 2, "b": 2}
    assert test.groupby("unique_id")["y"].apply(list).to_dict() == {
        "a": [3, 4],
        "b": [13, 14],
    }


def test_build_metrics_table_computes_error_metrics_and_rmae() -> None:
    """The metrics table should include MAE, RMSE, and RMAE."""
    actual = pd.DataFrame(
        {
            "unique_id": ["a", "a", "b", "b"],
            "ds": pd.date_range("2024-01-01", periods=4, freq="D"),
            "y": [10.0, 20.0, 30.0, 40.0],
        }
    )
    forecasts = {
        "Naive": pd.DataFrame(
            {
                "unique_id": ["a", "a", "b", "b"],
                "ds": actual["ds"],
                "y_hat": [12.0, 22.0, 32.0, 42.0],
            }
        ),
        "TimeBase": pd.DataFrame(
            {
                "unique_id": ["a", "a", "b", "b"],
                "ds": actual["ds"],
                "y_hat": [11.0, 21.0, 31.0, 41.0],
            }
        ),
    }
    runtimes = {"Naive": 0.1, "TimeBase": 0.2}
    parameter_counts = {"Naive": 0, "TimeBase": 25}

    metrics = build_metrics_table(actual, forecasts, runtimes, parameter_counts)

    naive_row = metrics.loc[metrics["model"] == "Naive"].iloc[0]
    timebase_row = metrics.loc[metrics["model"] == "TimeBase"].iloc[0]

    assert naive_row["mae"] == 2.0
    assert naive_row["rmse"] == 2.0
    assert naive_row["rmae"] == 1.0
    assert timebase_row["mae"] == 1.0
    assert timebase_row["rmse"] == 1.0
    assert timebase_row["rmae"] == 0.5
    assert timebase_row["parameters"] == 25


def test_render_markdown_report_embeds_plot_metrics_and_settings() -> None:
    """The report renderer should include both plots, metrics, and settings."""
    metrics = pd.DataFrame(
        {
            "model": ["Naive", "TimeBase"],
            "mae": [2.0, 1.0],
            "rmse": [2.0, 1.0],
            "rmae": [1.0, 0.5],
            "parameters": [0, 25],
            "runtime_seconds": [0.1, 0.2],
        }
    )

    markdown = render_markdown_report(
        metrics=metrics,
        horizon=12,
        plot_path="img/airpassengers-benchmark.png",
        conformal_plot_path="img/airpassengers-timebasetrend-conformal.png",
        model_settings={"TimeBase": {"input_size": 48}},
    )

    assert "# AirPassengers benchmark" in markdown
    assert "![AirPassengers benchmark](img/airpassengers-benchmark.png)" in markdown
    assert (
        "![TimeBaseTrend conformal intervals](img/airpassengers-timebasetrend-conformal.png)"
        in markdown
    )
    assert "## TimeBaseTrend conformal intervals" in markdown
    assert "conformal_error" in markdown
    assert "| model | mae | rmse | rmae | parameters | runtime_seconds |" in markdown
    assert "## Reproducible model settings" in markdown
    assert '"TimeBase": {' in markdown


def test_get_neural_model_configs_returns_tuned_model_specific_settings() -> None:
    """The benchmark should use per-model settings tuned for this dataset."""
    configs = get_neural_model_configs(
        input_size=DEFAULT_INPUT_SIZE, max_steps=DEFAULT_MAX_STEPS
    )

    assert configs["TimeBase"] == {
        "input_size": 48,
        "max_steps": 30,
        "learning_rate": 1e-2,
        "basis_num": 6,
        "period_len": 12,
    }
    assert configs["TimeBaseTrend"] == {
        "input_size": 48,
        "max_steps": 100,
        "learning_rate": 1e-2,
        "basis_num": 6,
        "period_len": 6,
        "moving_avg_window": 25,
    }
    assert configs["NLinear"] == {
        "input_size": 36,
        "max_steps": 100,
        "learning_rate": 5e-3,
    }
    assert configs["DLinear"] == {
        "input_size": 12,
        "max_steps": 100,
        "learning_rate": 1e-2,
    }


def test_get_reproducible_model_settings_includes_all_benchmarked_models() -> None:
    """The published settings helper should include neural and statistical models."""
    settings = get_reproducible_model_settings(
        horizon=12,
        input_size=DEFAULT_INPUT_SIZE,
        max_steps=DEFAULT_MAX_STEPS,
    )

    assert set(settings) == {
        "TimeBase",
        "TimeBaseTrend",
        "NLinear",
        "DLinear",
        "AutoMFLES",
        "Naive",
    }
    assert settings["AutoMFLES"]["season_length"] == 12
    assert settings["Naive"] == {}
