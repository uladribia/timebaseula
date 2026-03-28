"""Tests for detailed-only benchmark series filtering."""

from __future__ import annotations

import pandas as pd

from scripts.benchmark_nixtla_panel import filter_series_scope


def test_filter_series_scope_detailed_keeps_only_granular_series() -> None:
    """The detailed scope should exclude aggregate and total series."""
    frame = pd.DataFrame(
        {
            "unique_id": ["10__100", "11__101", "pdv__10", "sku__100", "total"],
            "ds": pd.to_datetime(["2024-01-01"] * 5),
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    filtered = filter_series_scope(frame, series_scope="detailed")

    assert filtered["unique_id"].tolist() == ["10__100", "11__101"]
