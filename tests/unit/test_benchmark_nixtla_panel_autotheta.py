"""Tests for optional AutoTheta inclusion in the daily benchmark."""

from __future__ import annotations

from scripts.benchmark_nixtla_panel import get_daily_model_configs


def test_get_daily_model_configs_can_exclude_autotheta() -> None:
    """The benchmark settings should support disabling AutoTheta."""
    settings = get_daily_model_configs(
        profile="normal",
        n_series=256,
        cv_windows=6,
        include_autotheta=False,
    )

    assert "AutoMFLES" in settings
    assert "Naive" in settings
    assert "AutoTheta" not in settings
