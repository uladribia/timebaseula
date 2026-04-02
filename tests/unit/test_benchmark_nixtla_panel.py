"""Tests for the Nixtla panel benchmark helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from neuralforecast.losses.pytorch import DistributionLoss, MAE

from scripts.benchmark_nixtla_panel import (
    build_model_summary_table,
    filter_plot_window,
    regularize_benchmark_panel,
    render_markdown_report,
    resolve_benchmark_loss,
)


def test_regularize_benchmark_panel_fills_missing_dates_with_zeroes() -> None:
    """The benchmark panel should be densified on a daily grid per selected series."""
    frame = pd.DataFrame(
        {
            "unique_id": ["a", "a", "b"],
            "ds": pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-02"]),
            "y": [1.0, 3.0, 5.0],
            "pdv": [1, 1, 2],
            "sku": [10, 10, 20],
        }
    )

    regularized = regularize_benchmark_panel(frame)

    series_a = regularized.loc[regularized["unique_id"] == "a", ["ds", "y"]]
    assert (
        series_a["ds"].tolist()
        == pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]).tolist()
    )
    assert series_a["y"].tolist() == [1.0, 0.0, 3.0]


def test_build_model_summary_table_aggregates_per_series_metrics() -> None:
    """The summary table should expose aggregate accuracy and ranking metrics."""
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
                "y_hat": [11.0, 21.0, 29.0, 39.0],
            }
        ),
    }
    training_times = {"Naive": 0.0, "TimeBase": 1.5}
    inference_times = {"Naive": 0.1, "TimeBase": 0.2}
    parameter_counts = {"Naive": 0, "TimeBase": 42}

    summary, per_series = build_model_summary_table(
        actual=actual,
        forecasts=forecasts,
        training_times=training_times,
        inference_times=inference_times,
        parameter_counts=parameter_counts,
    )

    naive_row = summary.loc[summary["model"] == "Naive"].iloc[0]
    timebase_row = summary.loc[summary["model"] == "TimeBase"].iloc[0]

    assert naive_row["avg_mae"] == 2.0
    assert naive_row["avg_mean_scaled_mae"] == 0.0952
    assert naive_row["avg_smape"] == 0.0488
    assert naive_row["wins"] == 0
    assert timebase_row["avg_mae"] == 1.0
    assert timebase_row["parameters"] == 42
    assert timebase_row["median_rmse"] == 1.0
    assert timebase_row["avg_mean_scaled_mae"] == 0.0476
    assert timebase_row["avg_smape"] == 0.0254
    assert timebase_row["avg_rank"] == 1.0
    assert timebase_row["wins"] == 2
    assert set(per_series.columns) >= {
        "model",
        "unique_id",
        "mae",
        "mean_scaled_mae",
        "rmse",
        "smape",
        "rank",
    }


def test_filter_plot_window_keeps_test_window_and_recent_train_context() -> None:
    """Forecast plots should focus on the holdout plus a small train context."""
    train = pd.DataFrame(
        {
            "unique_id": ["a"] * 10,
            "ds": pd.date_range("2024-01-01", periods=10, freq="D"),
            "y": range(10),
        }
    )
    test = pd.DataFrame(
        {
            "unique_id": ["a"] * 4,
            "ds": pd.date_range("2024-01-11", periods=4, freq="D"),
            "y": range(4),
        }
    )

    train_window, test_window = filter_plot_window(train, test, context_points=3)

    assert (
        train_window["ds"].tolist()
        == pd.date_range("2024-01-08", periods=3, freq="D").tolist()
    )
    assert (
        test_window["ds"].tolist()
        == pd.date_range("2024-01-11", periods=4, freq="D").tolist()
    )


def test_render_markdown_report_references_generated_artifacts() -> None:
    """The markdown report should list metrics and embed plot paths."""
    summary = pd.DataFrame(
        {
            "model": ["TimeBase", "Naive"],
            "training_time_seconds": [1.5, 0.0],
            "inference_time_seconds": [0.2, 0.1],
            "parameters": [42, 0],
            "avg_mae": [1.0, 2.0],
            "median_mae": [1.0, 2.0],
            "avg_mean_scaled_mae": [0.5, 1.0],
            "median_mean_scaled_mae": [0.5, 1.0],
            "avg_rmse": [1.0, 2.0],
            "median_rmse": [1.0, 2.0],
            "avg_smape": [4.0, 8.0],
            "median_smape": [4.0, 8.0],
            "avg_rank": [1.0, 2.0],
            "median_rank": [1.0, 2.0],
            "wins": [2, 0],
        }
    )
    dataset_summary = {
        "n_series": 2,
        "train_rows": 8,
        "test_rows": 4,
        "horizon": 2,
    }

    markdown = render_markdown_report(
        summary=summary,
        dataset_summary=dataset_summary,
        plot_paths=[Path("img/rank.png"), Path("img/times.png")],
        comments=["TimeBase wins on both series."],
        settings={"TimeBase": {"max_steps": 120}},
        profile="normal",
    )

    assert "# Daily panel benchmark" in markdown
    assert "| metric | TimeBase | Naive |" in markdown
    assert "| parameters | 42 | 0 |" in markdown
    assert "![Benchmark plot](img/rank.png)" in markdown
    assert "TimeBase wins on both series." in markdown


def test_resolve_benchmark_loss_supports_point_and_distribution_losses() -> None:
    """Daily benchmark helpers should resolve both point and probabilistic losses."""
    assert isinstance(resolve_benchmark_loss("mae"), MAE)

    normal_loss = resolve_benchmark_loss("normal")
    poisson_loss = resolve_benchmark_loss("poisson")

    assert isinstance(normal_loss, DistributionLoss)
    assert normal_loss.distribution == "Normal"
    assert isinstance(poisson_loss, DistributionLoss)
    assert poisson_loss.distribution == "Poisson"
