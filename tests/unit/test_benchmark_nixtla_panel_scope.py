"""Tests for benchmark series-scope filtering."""

from __future__ import annotations

import pandas as pd

from scripts.benchmark_nixtla_panel import filter_series_scope


def test_filter_series_scope_aggregated_excludes_detailed_series() -> None:
    """The aggregated scope should keep only aggregate and total series."""
    frame = pd.DataFrame(
        {
            "unique_id": ["10__100", "pdv__10", "sku__100", "total"],
            "ds": pd.to_datetime(["2024-01-01"] * 4),
            "y": [1.0, 2.0, 3.0, 4.0],
        }
    )

    filtered = filter_series_scope(frame, series_scope="aggregated")

    assert filtered["unique_id"].tolist() == ["pdv__10", "sku__100", "total"]
