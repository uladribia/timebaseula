"""Tests for long-horizon benchmark dataset helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pandas as pd

from scripts.benchmark_long_horizon import (
    aggregate_frame,
    benchmark_configuration,
    choose_series_count,
    get_aggregated_dataset_path,
    infer_test_size,
    prepare_train_test,
    profile_dataset,
    recommend_timebase_kwargs,
    recommend_training_kwargs,
    resolve_dataset_group,
    resolve_mode_defaults,
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

    def test_resolve_mode_defaults(self) -> None:
        """Mode defaults should provide sensible daily and monthly settings."""
        assert resolve_mode_defaults("daily") == {
            "freq": "D",
            "horizon": 14,
            "max_steps": 50,
        }
        assert resolve_mode_defaults("monthly") == {
            "freq": "ME",
            "horizon": 5,
            "max_steps": 30,
        }

    def test_choose_series_count_prefers_broad_slice(self) -> None:
        """Automatic selection should prefer 200-300 series when available."""
        assert choose_series_count(350, None) == 300
        assert choose_series_count(240, None) == 240
        assert choose_series_count(120, None) == 120
        assert choose_series_count(350, 220) == 220

    def test_infer_test_size_uses_approximate_20_percent(self) -> None:
        """Holdout size should be about 20% while respecting the horizon."""
        assert infer_test_size(100, horizon=5) == 20
        assert infer_test_size(25, horizon=5) == 5
        assert infer_test_size(37, horizon=5) == 7

    def test_prepare_train_test_uses_fractional_tail(self) -> None:
        """Train/test split should use the approximate 20% tail per series."""
        frame = pd.DataFrame(
            {
                "unique_id": ["a"] * 10 + ["b"] * 10,
                "ds": pd.date_range("2024-01-01", periods=10, freq="D").tolist() * 2,
                "y": list(range(10)) + list(range(10)),
            }
        )

        train, test = prepare_train_test(frame, horizon=2)

        assert len(train) == 16
        assert len(test) == 4
        assert test.groupby("unique_id").size().tolist() == [2, 2]

    def test_profile_dataset_detects_short_monthly_regime(self) -> None:
        """Profiler should identify short-history monthly datasets."""
        rows = []
        for unique_id in ("a", "b", "c"):
            for step in range(24):
                rows.append(
                    {
                        "unique_id": unique_id,
                        "ds": pd.Timestamp("2020-01-31") + pd.offsets.MonthEnd(step),
                        "y": float(step % 6),
                    }
                )
        frame = pd.DataFrame(rows)

        profile = profile_dataset(frame, freq="ME", horizon=6)
        recommendation = recommend_timebase_kwargs(profile, freq="ME", horizon=6)
        training = recommend_training_kwargs(profile, horizon=6, max_steps=200)

        assert profile.short_history is True
        assert profile.dominant_period in {3, 6, 12}
        assert recommendation["basis_num"] <= 3
        assert training["max_steps"] == 100
        assert training["learning_rate"] == 5e-3

    def test_profile_dataset_detects_long_daily_regime(self) -> None:
        """Profiler should recommend larger budgets for long daily datasets."""
        rows = []
        for unique_id in ("a", "b"):
            for step in range(200):
                rows.append(
                    {
                        "unique_id": unique_id,
                        "ds": pd.Timestamp("2024-01-01") + pd.Timedelta(days=step),
                        "y": float(step % 7),
                    }
                )
        frame = pd.DataFrame(rows)

        profile = profile_dataset(frame, freq="D", horizon=28)
        configs = benchmark_configuration(
            "D", horizon=28, max_steps=50, profile=profile
        )

        assert profile.long_history is True
        assert profile.dominant_period in {7, 14, 28}
        dlinear_kwargs = configs[0][2]
        timebase_kwargs = configs[2][2]
        assert dlinear_kwargs["max_steps"] >= 150
        assert timebase_kwargs["period_len"] in {7, 14, 28}
        assert timebase_kwargs["basis_num"] >= 6

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
