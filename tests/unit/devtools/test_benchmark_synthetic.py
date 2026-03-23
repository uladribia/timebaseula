"""Tests for synthetic benchmark result helpers."""

from __future__ import annotations

import pandas as pd
from pytest import MonkeyPatch

from devtools import benchmark_synthetic as synthetic_benchmark


class TestSyntheticBenchmarkHelpers:
    """Validate synthetic benchmark table helpers."""

    def test_run_synthetic_mae_checks_returns_long_format_frame(
        self,
        monkeypatch: MonkeyPatch,
    ) -> None:
        """Synthetic benchmark helper should return one row per scenario/model pair."""
        monkeypatch.setattr(
            synthetic_benchmark,
            "make_synthetic_series",
            lambda **_: pd.DataFrame(
                {
                    "unique_id": ["base"] * 4,
                    "ds": pd.date_range("2024-01-01", periods=4, freq="D"),
                    "y": [1.0, 2.0, 3.0, 4.0],
                }
            ),
        )
        monkeypatch.setattr(
            synthetic_benchmark,
            "evaluate_models",
            lambda *args, **kwargs: {
                "dlinear": 0.5,
                "timebase": 0.6,
                "timebase_trend": 0.7,
            },
        )
        monkeypatch.setattr(
            synthetic_benchmark,
            "evaluate_statsforecast_models",
            lambda *args, **kwargs: {"naive": 1.0, "mfles": 0.4},
        )

        result = synthetic_benchmark.run_synthetic_mae_checks(
            length=16,
            h=4,
            input_size=8,
            val_size=4,
            test_size=4,
            max_steps=3,
        )

        assert set(result.columns) == {"scenario", "model_name", "mae"}
        assert len(result) == 15
        assert set(result["scenario"]) == {"easy", "medium", "hard"}
        assert set(result["model_name"]) == {
            "Naive",
            "DLinear",
            "AutoTimeBase",
            "AutoTimeBaseTrend",
            "MFLES",
        }

    def test_run_synthetic_mae_checks_passes_refit_flag(
        self,
        monkeypatch: MonkeyPatch,
    ) -> None:
        """Synthetic benchmark orchestration should forward the refit setting."""
        captured: dict[str, object] = {}

        monkeypatch.setattr(
            synthetic_benchmark,
            "run_synthetic_benchmark",
            lambda *args, **kwargs: captured.update(kwargs) or pd.DataFrame(),
        )

        synthetic_benchmark.run_synthetic_mae_checks(
            length=16,
            h=4,
            input_size=8,
            val_size=4,
            test_size=4,
            max_steps=3,
            refit=False,
        )

        assert captured["refit"] is False
