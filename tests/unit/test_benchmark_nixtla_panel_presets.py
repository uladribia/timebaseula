"""Tests for benchmark runtime presets and adaptive iteration settings."""

from __future__ import annotations

from scripts.benchmark_nixtla_panel import (
    estimate_runtime,
    get_daily_model_configs,
)


def test_get_daily_model_configs_scales_iterations_by_profile() -> None:
    """Training profiles should increase neural iterations monotonically."""
    smoke = get_daily_model_configs(profile="smoke", n_series=8, cv_windows=2)
    normal = get_daily_model_configs(profile="normal", n_series=64, cv_windows=4)
    heavy = get_daily_model_configs(profile="heavy", n_series=256, cv_windows=6)

    assert smoke["TimeBase"]["max_steps"] < normal["TimeBase"]["max_steps"]
    assert normal["TimeBase"]["max_steps"] < heavy["TimeBase"]["max_steps"]
    assert smoke["DLinear"]["max_steps"] < normal["DLinear"]["max_steps"]
    assert normal["DLinear"]["max_steps"] <= heavy["DLinear"]["max_steps"]
    assert normal["TimeBase"]["max_steps"] >= 130
    assert heavy["TimeBaseTrend"]["max_steps"] >= 220


def test_estimate_runtime_depends_on_profile_and_workload() -> None:
    """Runtime estimates should get larger for heavier profiles and workloads."""
    smoke = estimate_runtime(max_series=8, cv_windows=2, profile="smoke")
    normal = estimate_runtime(max_series=64, cv_windows=4, profile="normal")
    heavy = estimate_runtime(max_series=256, cv_windows=6, profile="heavy")

    assert smoke != normal
    assert normal != heavy
