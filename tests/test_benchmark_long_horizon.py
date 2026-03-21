"""Tests for long-horizon benchmark dataset helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pandas as pd

from scripts.benchmark_long_horizon import (
    aggregate_frame,
    choose_series_count,
    get_aggregated_dataset_path,
    resolve_dataset_group,
)


class TestBenchmarkDatasetHelpers:
    """Validate dataset helper behavior for the benchmark script."""

    def test_resolve_dataset_group_aliases(self) -> None:
        """Dataset aliases should map to datasetsforecast group names."""
        assert resolve_dataset_group("ecl") == "ECL"
        assert resolve_dataset_group("TrafficL") == "TrafficL"
        assert resolve_dataset_group("traffic") == "TrafficL"

    def test_get_aggregated_dataset_path(self) -> None:
        """Aggregated dataset files should live under the datasets directory."""
        path = get_aggregated_dataset_path(Path("datasets"), "ECL", "D")
        assert path == Path("datasets/ecl_daily.parquet")

        monthly = get_aggregated_dataset_path(Path("datasets"), "TrafficL", "ME")
        assert monthly == Path("datasets/trafficl_monthly.parquet")

    def test_choose_series_count_prefers_broad_slice(self) -> None:
        """Automatic selection should prefer 200-300 series when available."""
        assert choose_series_count(350, None) == 300
        assert choose_series_count(240, None) == 240
        assert choose_series_count(120, None) == 120
        assert choose_series_count(350, 220) == 220

    def test_aggregate_frame_daily(self) -> None:
        """Daily aggregation should preserve unique_id and mean by day."""
        frame = pd.DataFrame(
            {
                "unique_id": ["a", "a", "a", "a"],
                "ds": pd.to_datetime(
                    [
                        "2024-01-01 00:00:00",
                        "2024-01-01 12:00:00",
                        "2024-01-02 00:00:00",
                        "2024-01-02 12:00:00",
                    ]
                ),
                "y": [1.0, 3.0, 2.0, 6.0],
            }
        )

        result = aggregate_frame(frame, "D")

        assert list(result.columns) == ["unique_id", "ds", "y"]
        assert len(result) == 2
        assert result["y"].tolist() == [2.0, 4.0]

    def test_aggregate_frame_monthly(self) -> None:
        """Monthly aggregation should use month-end buckets."""
        frame = pd.DataFrame(
            {
                "unique_id": ["a", "a", "a"],
                "ds": pd.to_datetime(["2024-01-01", "2024-01-15", "2024-02-01"]),
                "y": [1.0, 3.0, 10.0],
            }
        )

        result = aggregate_frame(frame, "ME")

        assert len(result) == 2
        assert result["y"].tolist() == [2.0, 10.0]

    def test_cached_dataset_avoids_redownload(
        self, tmp_path: Path, monkeypatch: Mock
    ) -> None:
        """Existing aggregated files should be reused instead of downloaded again."""
        from scripts import benchmark_long_horizon as benchmark

        cached = pd.DataFrame(
            {
                "unique_id": ["a"],
                "ds": pd.to_datetime(["2024-01-31"]),
                "y": [1.0],
            }
        )
        cache_path = tmp_path / "ecl_daily.parquet"
        cached.to_parquet(cache_path)

        load_mock = Mock(side_effect=AssertionError("should not download"))
        monkeypatch.setattr(benchmark, "DATASETS_DIR", tmp_path)
        monkeypatch.setattr(benchmark.LongHorizon, "load", load_mock)

        result = benchmark.load_or_create_aggregated_dataset("ECL", "D")

        assert result.equals(cached)
        load_mock.assert_not_called()
