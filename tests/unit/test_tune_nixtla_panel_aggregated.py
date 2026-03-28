"""Tests for aggregated tuning helpers."""

from __future__ import annotations

from scripts.tune_nixtla_panel_aggregated import (
    build_timebase_candidate_configs,
    build_timebasetrend_candidate_configs,
    sanitize_auto_config,
    select_best_tuning_result,
)


def test_build_timebase_candidate_configs_smoke_is_small() -> None:
    """Smoke tuning should use a compact TimeBase grid."""
    configs = build_timebase_candidate_configs(profile="smoke")

    assert len(configs) > 0
    assert len(configs) <= 4
    assert {
        "input_size",
        "basis_num",
        "period_len",
        "learning_rate",
        "max_steps",
    } <= set(configs[0])


def test_build_timebasetrend_candidate_configs_heavy_is_larger() -> None:
    """Heavy tuning should explore more TimeBaseTrend candidates than smoke."""
    smoke = build_timebasetrend_candidate_configs(profile="smoke")
    heavy = build_timebasetrend_candidate_configs(profile="heavy")

    assert len(heavy) > len(smoke)


def test_sanitize_auto_config_keeps_expected_dlinear_fields() -> None:
    """Native auto-model configs should be reduced to benchmark-ready fields."""
    raw_config = {
        "input_size": 84,
        "learning_rate": 0.001,
        "max_steps": 700.0,
        "step_size": 28,
        "scaler_type": "robust",
        "moving_avg_window": 25,
        "random_seed": 7,
    }

    sanitized = sanitize_auto_config(raw_config, model_name="AutoDLinear")

    assert sanitized == {
        "input_size": 84,
        "learning_rate": 0.001,
        "max_steps": 700,
        "step_size": 28,
        "scaler_type": "robust",
        "moving_avg_window": 25,
    }


def test_select_best_tuning_result_prioritizes_mean_scaled_mae_then_mae() -> None:
    """Best tuned config selection should prioritize normalized accuracy."""
    results = [
        {"model": "a", "avg_mean_scaled_mae": 0.20, "avg_mae": 12.0},
        {"model": "b", "avg_mean_scaled_mae": 0.15, "avg_mae": 20.0},
        {"model": "c", "avg_mean_scaled_mae": 0.15, "avg_mae": 10.0},
    ]

    best = select_best_tuning_result(results)

    assert best["model"] == "c"
