"""Tests for the NeuralForecast auto wrappers."""

from __future__ import annotations

from neuralforecast.losses.pytorch import MAE

from timebaseula.models.auto import AutoTimeBase, AutoTimeBaseTrend
from timebaseula.models.timebase import TimeBase, TimeBaseTrend


def test_autotimebase_default_config_exposes_timebase_search_space() -> None:
    """AutoTimeBase should expose the TimeBase hyperparameters to tune."""
    config = AutoTimeBase.get_default_config(h=28, backend="ray")

    assert {
        "input_size",
        "basis_num",
        "period_len",
        "learning_rate",
        "max_steps",
    } <= set(config)
    assert "moving_avg_window" not in config


def test_autotimebasetrend_default_config_exposes_trend_search_space() -> None:
    """AutoTimeBaseTrend should include the trend decomposition hyperparameters."""
    config = AutoTimeBaseTrend.get_default_config(h=28, backend="ray")

    assert {
        "input_size",
        "basis_num",
        "period_len",
        "moving_avg_window",
        "learning_rate",
        "max_steps",
    } <= set(config)


def test_autotimebase_uses_timebase_model_class() -> None:
    """AutoTimeBase should wrap the explicit TimeBase model."""
    model = AutoTimeBase(h=14, loss=MAE(), num_samples=1, cpus=1, gpus=0, backend="ray")

    assert model.cls_model is TimeBase
    assert model.h == 14


def test_autotimebasetrend_uses_timebasetrend_model_class() -> None:
    """AutoTimeBaseTrend should wrap the explicit TimeBaseTrend model."""
    model = AutoTimeBaseTrend(
        h=14,
        loss=MAE(),
        num_samples=1,
        cpus=1,
        gpus=0,
        backend="ray",
    )

    assert model.cls_model is TimeBaseTrend
    assert model.h == 14
