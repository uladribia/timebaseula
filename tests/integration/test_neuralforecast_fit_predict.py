"""Integration tests for NeuralForecast compatibility."""

from __future__ import annotations

import warnings
from typing import Any, cast

import pandas as pd
import pytest

from timebaseula.models.timebase import AutoTimeBase, TimeBase


@pytest.mark.integration
def test_neuralforecast_fit_predict() -> None:
    """Minimal NeuralForecast fit/predict works."""
    pytest.importorskip("neuralforecast")
    from neuralforecast import NeuralForecast

    np = pytest.importorskip("numpy")
    np.random.seed(42)
    n = 100
    dates = pd.date_range(start="2020-01-01", periods=n, freq="D")
    y = np.random.randn(n).cumsum() + 10
    df = pd.DataFrame({"ds": dates, "y": y, "unique_id": "series_1"})

    model = TimeBase(
        h=8,
        input_size=16,
        period_len=8,
        basis_num=4,
        max_steps=100,
        val_check_steps=50,
        learning_rate=1e-2,
        num_workers_loader=13,
        logger=cast(Any, False),
        enable_progress_bar=cast(Any, False),
        enable_model_summary=cast(Any, False),
        log_every_n_steps=cast(Any, 1),
    )

    nf = NeuralForecast(models=[model], freq="D")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"`isinstance\(treespec, LeafSpec\)` is deprecated.*",
            category=FutureWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"The 'val_dataloader' does not have many workers.*",
        )
        warnings.filterwarnings(
            "ignore",
            message=r"The 'train_dataloader' does not have many workers.*",
        )
        warnings.filterwarnings("ignore", category=Warning)
        nf.fit(df, val_size=8)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"`isinstance\(treespec, LeafSpec\)` is deprecated.*",
            category=FutureWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"The 'predict_dataloader' does not have many workers.*",
        )
        warnings.filterwarnings("ignore", category=Warning)
        pred = nf.predict()

    assert len(pred) == 8
    assert "TimeBase" in pred.columns


@pytest.mark.integration
def test_auto_timebase_fit_predict() -> None:
    """AutoTimeBase should fit and predict through NeuralForecast."""
    pytest.importorskip("neuralforecast")
    from neuralforecast import NeuralForecast

    np = pytest.importorskip("numpy")
    np.random.seed(42)
    n = 100
    dates = pd.date_range(start="2020-01-01", periods=n, freq="D")
    y = np.random.randn(n).cumsum() + 10
    df = pd.DataFrame({"ds": dates, "y": y, "unique_id": "series_1"})

    model = AutoTimeBase(
        h=8,
        freq="D",
        max_steps=40,
        search_enabled=True,
        search_max_steps=5,
        n_search_configs=2,
        logger=cast(Any, False),
        enable_progress_bar=cast(Any, False),
        enable_model_summary=cast(Any, False),
        log_every_n_steps=cast(Any, 1),
    )

    nf = NeuralForecast(models=[model], freq="D")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=Warning)
        nf.fit(df, val_size=8)
        pred = nf.predict()

    assert len(pred) == 8
    assert "AutoTimeBase" in pred.columns
    assert model.selected_config_["input_size"] >= 8


@pytest.mark.integration
def test_multiseries_training_can_predict_only_one_series() -> None:
    """A model trained on multiple series should still predict one series."""
    pytest.importorskip("neuralforecast")
    from neuralforecast import NeuralForecast

    np = pytest.importorskip("numpy")
    np.random.seed(42)
    n = 100
    frames = []
    for index in range(3):
        dates = pd.date_range(start="2020-01-01", periods=n, freq="D")
        y = np.random.randn(n).cumsum() + 10 + index
        frames.append(
            pd.DataFrame({"ds": dates, "y": y, "unique_id": f"series_{index}"})
        )

    df = pd.concat(frames)
    model = TimeBase(
        h=8,
        input_size=16,
        period_len=8,
        basis_num=4,
        max_steps=100,
        val_check_steps=50,
        num_workers_loader=13,
        logger=cast(Any, False),
        enable_progress_bar=cast(Any, False),
        enable_model_summary=cast(Any, False),
        log_every_n_steps=cast(Any, 1),
    )

    nf = NeuralForecast(models=[model], freq="D")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"`isinstance\(treespec, LeafSpec\)` is deprecated.*",
            category=FutureWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"The 'val_dataloader' does not have many workers.*",
        )
        warnings.filterwarnings(
            "ignore",
            message=r"The 'train_dataloader' does not have many workers.*",
        )
        warnings.filterwarnings("ignore", category=Warning)
        nf.fit(df, val_size=8)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"`isinstance\(treespec, LeafSpec\)` is deprecated.*",
            category=FutureWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"The 'predict_dataloader' does not have many workers.*",
        )
        warnings.filterwarnings("ignore", category=Warning)
        pred = nf.predict(df=df[df["unique_id"] == "series_1"])

    assert pred["unique_id"].eq("series_1").all()
    assert len(pred) == 8
