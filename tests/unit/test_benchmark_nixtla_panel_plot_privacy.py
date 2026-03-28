"""Tests for anonymized benchmark plotting helpers."""

from __future__ import annotations

import pandas as pd

from scripts.benchmark_nixtla_panel import anonymize_series_values, build_series_aliases


def test_anonymize_series_values_preserves_shape_with_deterministic_scaling() -> None:
    """Anonymized plotting should use deterministic positive scaling factors."""
    series = pd.DataFrame(
        {
            "unique_id": ["a", "a", "a"],
            "ds": pd.date_range("2024-01-01", periods=3, freq="D"),
            "y": [1.0, 2.0, 4.0],
        }
    )

    anonymized_first = anonymize_series_values(series, value_column="y")
    anonymized_second = anonymize_series_values(series, value_column="y")

    ratio_first = anonymized_first["y"].iloc[1] / anonymized_first["y"].iloc[0]
    ratio_second = anonymized_first["y"].iloc[2] / anonymized_first["y"].iloc[1]

    assert anonymized_first["y"].tolist() == anonymized_second["y"].tolist()
    assert ratio_first == 2.0
    assert ratio_second == 2.0
    assert all(value > 0 for value in anonymized_first["y"])


def test_build_series_aliases_removes_raw_identifiers_from_plot_titles() -> None:
    """Plot aliases should use generic names instead of raw unique identifiers."""
    aliases = build_series_aliases(["350039371__2604", "sku__100", "total"])

    assert aliases == {
        "350039371__2604": "Series 1",
        "sku__100": "Series 2",
        "total": "Series 3",
    }
