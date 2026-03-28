"""Tests for preparing Nixtla-ready panel datasets."""

from __future__ import annotations

import pandas as pd

from scripts.prepare_nixtla_panel import (
    build_prepared_panel,
    split_panel_by_date_ratio,
)


def test_build_prepared_panel_creates_nixtla_columns_and_unique_ids() -> None:
    """The prepared panel should expose Nixtla's canonical schema."""
    raw = pd.DataFrame(
        {
            "fecha": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-01",
                    "2024-01-01",
                ]
            ),
            "pdv": [10, 10, 11],
            "sku": [100, 100, 101],
            "so": [2.5, 1.5, 3.0],
        }
    )

    prepared = build_prepared_panel(raw)

    assert prepared.columns.tolist() == ["unique_id", "ds", "y", "pdv", "sku"]
    detailed = prepared.loc[
        prepared["unique_id"].str.contains("__")
        & ~prepared["unique_id"].str.startswith(("pdv__", "sku__"))
    ]
    assert detailed["unique_id"].tolist() == [
        "10__100",
        "10__100",
        "11__101",
        "11__101",
    ]
    assert (
        detailed["ds"].tolist()
        == pd.to_datetime(
            ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"]
        ).tolist()
    )
    assert detailed["y"].tolist() == [1.5, 2.5, 3.0, 0.0]


def test_build_prepared_panel_fills_missing_pdv_sku_dates_with_zeroes() -> None:
    """Missing raw observations should become zero-valued detailed series rows."""
    raw = pd.DataFrame(
        {
            "fecha": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "pdv": [10, 10],
            "sku": [100, 100],
            "so": [1.0, 3.0],
        }
    )

    prepared = build_prepared_panel(raw)

    detailed = prepared.loc[prepared["unique_id"] == "10__100", ["ds", "y"]]
    assert (
        detailed["ds"].tolist() == pd.to_datetime(["2024-01-01", "2024-01-02"]).tolist()
    )
    assert detailed["y"].tolist() == [1.0, 3.0]


def test_build_prepared_panel_adds_aggregated_series() -> None:
    """The prepared panel should include pdv, sku, and total aggregates."""
    raw = pd.DataFrame(
        {
            "fecha": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-02",
                ]
            ),
            "pdv": [10, 10, 10, 11],
            "sku": [100, 101, 100, 100],
            "so": [1.0, 2.0, 3.0, 4.0],
        }
    )

    prepared = build_prepared_panel(raw)

    expected_ids = {
        "10__100",
        "10__101",
        "11__100",
        "pdv__10",
        "pdv__11",
        "sku__100",
        "sku__101",
        "total",
    }
    assert expected_ids.issubset(set(prepared["unique_id"]))
    missing_combo = prepared.loc[
        prepared["unique_id"] == "10__101", ["ds", "y"]
    ].sort_values("ds")
    assert missing_combo["y"].tolist() == [2.0, 0.0]
    total_series = prepared.loc[prepared["unique_id"] == "total", ["ds", "y"]]
    assert total_series["y"].tolist() == [3.0, 7.0]


def test_split_panel_by_date_ratio_uses_a_global_temporal_cutoff() -> None:
    """The split helper should reserve the last proportion of dates for testing."""
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    panel = pd.DataFrame(
        {
            "unique_id": ["a"] * 10 + ["b"] * 10,
            "ds": dates.tolist() * 2,
            "y": list(range(10)) + list(range(10, 20)),
            "pdv": [1] * 10 + [2] * 10,
            "sku": [100] * 10 + [200] * 10,
        }
    )

    train, test, summary = split_panel_by_date_ratio(panel, test_ratio=0.2)

    assert summary.train_dates == 8
    assert summary.test_dates == 2
    assert summary.horizon == 2
    assert summary.cutoff_date == pd.Timestamp("2024-01-08")
    assert train["ds"].max() == pd.Timestamp("2024-01-08")
    assert test["ds"].min() == pd.Timestamp("2024-01-09")
    assert test.groupby("unique_id").size().to_dict() == {"a": 2, "b": 2}
