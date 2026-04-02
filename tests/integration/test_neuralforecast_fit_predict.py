"""Integration tests for NeuralForecast compatibility."""

from __future__ import annotations

import warnings
from typing import Any, cast

import pandas as pd
import pytest
from neuralforecast.losses.pytorch import DistributionLoss

from timebaseula.models.timebase import TimeBase, TimeBaseTrend


def _disabled_trainer_kwargs() -> dict[str, Any]:
    """Return quiet CPU-first trainer kwargs for integration tests."""
    return {
        "num_workers_loader": 13,
        "logger": cast(Any, False),
        "enable_progress_bar": cast(Any, False),
        "enable_model_summary": cast(Any, False),
        "log_every_n_steps": cast(Any, 1),
    }


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
        freq="D",
        max_steps=40,
        val_check_steps=20,
        learning_rate=1e-2,
        **_disabled_trainer_kwargs(),
    )

    nf = NeuralForecast(models=[model], freq="D")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=Warning)
        nf.fit(df, val_size=8)
        pred = nf.predict()

    assert len(pred) == 8
    assert "TimeBase" in pred.columns


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
        freq="D",
        max_steps=40,
        val_check_steps=20,
        **_disabled_trainer_kwargs(),
    )

    nf = NeuralForecast(models=[model], freq="D")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=Warning)
        nf.fit(df, val_size=8)
        pred = nf.predict(df=df[df["unique_id"] == "series_1"])

    if "unique_id" in pred.columns:
        assert pred["unique_id"].eq("series_1").all()
    assert len(pred) == 8


@pytest.mark.integration
@pytest.mark.parametrize(
    ("model_cls", "model_name"),
    [(TimeBase, "TimeBase"), (TimeBaseTrend, "TimeBaseTrend")],
)
def test_models_support_conformal_prediction_intervals(
    model_cls: type[TimeBase] | type[TimeBaseTrend],
    model_name: str,
) -> None:
    """TimeBase variants should work with NeuralForecast conformal intervals."""
    pytest.importorskip("neuralforecast")
    from neuralforecast import NeuralForecast
    from neuralforecast.utils import PredictionIntervals

    np = pytest.importorskip("numpy")
    np.random.seed(42)
    n = 40
    dates = pd.date_range(start="2020-01-01", periods=n, freq="D")
    y = np.random.randn(n).cumsum() + 10
    df = pd.DataFrame({"ds": dates, "y": y, "unique_id": "series_1"})

    model = model_cls(
        h=4,
        input_size=8,
        period_len=4,
        basis_num=2,
        freq="D",
        max_steps=1,
        val_check_steps=1,
        **_disabled_trainer_kwargs(),
    )
    nf = NeuralForecast(models=[model], freq="D")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=Warning)
        nf.fit(
            df,
            val_size=0,
            prediction_intervals=PredictionIntervals(n_windows=2),
        )
        pred = nf.predict(level=[80])

    assert len(pred) == 4
    assert model_name in pred.columns
    assert f"{model_name}-lo-80" in pred.columns
    assert f"{model_name}-hi-80" in pred.columns


@pytest.mark.integration
@pytest.mark.parametrize(
    ("model_cls", "loss"),
    [
        (TimeBase, DistributionLoss("Normal", level=[80])),
        (TimeBaseTrend, DistributionLoss("Poisson", level=[80])),
    ],
)
def test_models_support_distribution_loss_fit_predict(
    model_cls: type[TimeBase] | type[TimeBaseTrend],
    loss: DistributionLoss,
) -> None:
    """TimeBase variants should fit and predict with NeuralForecast distribution losses."""
    pytest.importorskip("neuralforecast")
    from neuralforecast import NeuralForecast

    np = pytest.importorskip("numpy")
    np.random.seed(42)
    n = 80
    dates = pd.date_range(start="2020-01-01", periods=n, freq="D")
    y = np.abs(np.random.randn(n).cumsum()) + 1
    df = pd.DataFrame({"ds": dates, "y": y, "unique_id": "series_1"})

    model = model_cls(
        h=4,
        input_size=12,
        period_len=4,
        basis_num=2,
        freq="D",
        loss=loss,
        max_steps=5,
        val_check_steps=5,
        **_disabled_trainer_kwargs(),
    )
    nf = NeuralForecast(models=[model], freq="D")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=Warning)
        nf.fit(df, val_size=4)
        pred = nf.predict()

    model_name = type(model).__name__
    assert len(pred) == 4
    assert model_name in pred.columns
