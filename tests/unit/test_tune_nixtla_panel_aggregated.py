"""Tests for aggregated tuning helpers."""

from __future__ import annotations

import logging
import sys
from types import SimpleNamespace

import pandas as pd

from scripts.tune_nixtla_panel_aggregated import (
    build_timebase_candidate_configs,
    build_timebasetrend_candidate_configs,
    sanitize_auto_config,
    select_best_tuning_result,
    tune_timebase_family,
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


def test_sanitize_auto_config_keeps_expected_timebase_fields() -> None:
    """AutoTimeBase configs should keep the model-specific tuning fields."""
    raw_config = {
        "input_size": 84,
        "basis_num": 8,
        "period_len": 7,
        "learning_rate": 0.001,
        "max_steps": 200.0,
        "step_size": 28,
        "scaler_type": "identity",
        "batch_size": 32,
    }

    sanitized = sanitize_auto_config(raw_config, model_name="AutoTimeBase")

    assert sanitized == {
        "input_size": 84,
        "basis_num": 8,
        "period_len": 7,
        "learning_rate": 0.001,
        "max_steps": 200,
        "step_size": 28,
        "scaler_type": "identity",
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


def test_tune_timebase_family_uses_native_auto_wrappers(monkeypatch) -> None:
    """Aggregated tuning should use AutoTimeBase wrappers through NeuralForecast."""

    class _BestResult:
        def __init__(self, config):
            self.config = config

    class _Results:
        def __init__(self, config):
            self._config = config

        def get_best_result(self):
            return _BestResult(self._config)

    class _AutoModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.results = _Results(
                {
                    "input_size": 84,
                    "basis_num": 8,
                    "period_len": 7,
                    "moving_avg_window": 21,
                    "learning_rate": 0.001,
                    "max_steps": 200.0,
                    "step_size": 28,
                    "scaler_type": "identity",
                }
            )

    class _NeuralForecast:
        def __init__(self, models, freq):
            self.models = models
            self.freq = freq

        def fit(self, df, val_size):
            assert isinstance(df, pd.DataFrame)
            assert val_size == 28

    monkeypatch.setitem(
        sys.modules,
        "timebaseula",
        SimpleNamespace(AutoTimeBase=_AutoModel, AutoTimeBaseTrend=_AutoModel),
    )
    monkeypatch.setitem(
        sys.modules,
        "neuralforecast",
        SimpleNamespace(NeuralForecast=_NeuralForecast),
    )

    train_df = pd.DataFrame(
        {
            "unique_id": ["s1"] * 56,
            "ds": pd.date_range("2024-01-01", periods=56, freq="D"),
            "y": range(56),
        }
    )

    best = tune_timebase_family(
        train_df=train_df,
        test_df=train_df.tail(28),
        horizon=28,
        profile="smoke",
        logger=logging.getLogger("test"),
    )

    assert best["AutoTimeBase"]["basis_num"] == 8
    assert best["AutoTimeBaseTrend"]["moving_avg_window"] == 21
