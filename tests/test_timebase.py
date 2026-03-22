"""Tests for the TimeBase model."""

from __future__ import annotations

import warnings
from typing import Any, cast

import pandas as pd
import pytest
import torch

from timebaseula.models.timebase import (
    AutoTimeBase,
    AutoTimeBaseTrend,
    TimeBase,
    TimeBaseTrend,
)
from timebaseula.recommend import (
    recommend_timebase_kwargs,
    trim_frame_for_recommendation,
)


class TestTimeBase:
    """Validate TimeBase forward behavior."""

    def test_recommend_defaults(self) -> None:
        """TimeBase should expose class helpers for default selection."""
        frame = pd.DataFrame(
            {
                "unique_id": ["a"] * 60 + ["b"] * 60,
                "ds": list(pd.date_range("2024-01-01", periods=60, freq="D")) * 2,
                "y": [float(step % 7) for step in range(60)] * 2,
            }
        )

        profile = TimeBase.profile_dataset(frame, freq="D", horizon=14)
        defaults = TimeBase.recommend_defaults(
            frame,
            freq="D",
            horizon=14,
            max_steps=120,
        )

        assert profile.dominant_period in {7, 14, 28}
        assert defaults["input_size"] >= 16
        assert defaults["period_len"] >= 2
        assert defaults["max_steps"] <= 120

    def test_forward_shape(self) -> None:
        """The forward output should match (batch, h, 1)."""
        model = TimeBase(h=12, input_size=24, period_len=6, basis_num=4)
        windows_batch = {"insample_y": torch.ones((2, 24))}
        output = model(windows_batch)
        assert output.shape == (2, 12)

    def test_padding_output_shape(self) -> None:
        """Padding should still yield the requested horizon length."""
        model = TimeBase(h=10, input_size=25, period_len=6, basis_num=4)
        windows_batch = {"insample_y": torch.zeros((3, 25))}
        output = model(windows_batch)
        assert output.shape == (3, 10)

    def test_orthogonal_loss_zero(self) -> None:
        """Orthogonal basis should yield near-zero orthogonal loss."""
        model = TimeBase(h=4, input_size=8, period_len=4, basis_num=4)
        basis = torch.eye(4).unsqueeze(0)
        loss = model._compute_orthogonal_loss(basis)
        assert torch.isclose(loss, torch.tensor(0.0))

    def test_deterministic_behavior(self) -> None:
        """Same seed should produce same output."""
        torch.manual_seed(42)
        model1 = TimeBase(h=8, input_size=16, period_len=4, basis_num=4)
        windows_batch = {"insample_y": torch.randn(2, 16)}
        output1 = model1(windows_batch)

        torch.manual_seed(42)
        model2 = TimeBase(h=8, input_size=16, period_len=4, basis_num=4)
        output2 = model2(windows_batch)

        assert torch.allclose(output1, output2)


class TestTimeBaseTrend:
    """Validate TimeBaseTrend forward behavior."""

    def test_recommend_defaults(self) -> None:
        """TimeBaseTrend should expose class helpers for default selection."""
        frame = pd.DataFrame(
            {
                "unique_id": ["a"] * 36 + ["b"] * 36,
                "ds": list(pd.date_range("2024-01-31", periods=36, freq="ME")) * 2,
                "y": [float(step % 6) for step in range(36)] * 2,
            }
        )

        profile = TimeBaseTrend.profile_dataset(frame, freq="ME", horizon=6)
        defaults = TimeBaseTrend.recommend_defaults(
            frame,
            freq="ME",
            horizon=6,
            max_steps=120,
        )

        assert profile.short_history is True
        assert defaults["moving_avg_window"] % 2 == 1
        assert defaults["basis_num"] <= 3
        assert defaults["max_steps"] <= 100

    def test_forward_shape(self) -> None:
        """The forward output should match (batch, h)."""
        model = TimeBaseTrend(h=12, input_size=24, period_len=6, basis_num=4)
        windows_batch = {"insample_y": torch.ones((2, 24))}
        output = model(windows_batch)
        assert output.shape == (2, 12)

    def test_linear_trend_head_is_present(self) -> None:
        """The linear_trend layer should be present."""
        model = TimeBaseTrend(h=4, input_size=8, period_len=4, basis_num=4)
        assert hasattr(model, "linear_trend")
        assert isinstance(model.linear_trend, torch.nn.Linear)

    def test_invalid_moving_avg_window_even(self) -> None:
        """Even moving_avg_window should raise ValueError."""
        with pytest.raises(ValueError):
            TimeBaseTrend(
                h=4, input_size=8, period_len=4, basis_num=4, moving_avg_window=4
            )

    def test_valid_moving_avg_window_odd(self) -> None:
        """Odd moving_avg_window should be accepted."""
        model = TimeBaseTrend(
            h=4, input_size=8, period_len=4, basis_num=4, moving_avg_window=5
        )
        assert model.moving_avg_window == 5

    def test_has_decomposition(self) -> None:
        """TimeBaseTrend should have SeriesDecomp for trend extraction."""
        model = TimeBaseTrend(h=4, input_size=8, period_len=4, basis_num=4)
        assert hasattr(model, "decomp")


class TestAutoTimeBase:
    """Validate auto-configured TimeBase wrappers."""

    def test_recommendation_helpers_can_emit_iteration_recommendation(self) -> None:
        """Recommendation helpers should optionally include iteration guidance."""
        frame = pd.DataFrame(
            {
                "unique_id": ["a"] * 80 + ["b"] * 80,
                "ds": list(pd.date_range("2024-01-01", periods=80, freq="D")) * 2,
                "y": [float(step % 7) for step in range(80)] * 2,
            }
        )

        recommendation = recommend_timebase_kwargs(
            frame=frame,
            freq="D",
            horizon=14,
            max_steps=120,
            include_iteration_recommendation=True,
        )

        assert recommendation["recommended_training_iterations"] >= 1
        assert (
            recommendation["recommended_training_iterations"]
            >= recommendation["max_steps"]
        )

    def test_trim_frame_for_recommendation_excludes_tail_per_series(self) -> None:
        """Recommendation trimming should remove the requested tail from each series."""
        frame = pd.DataFrame(
            {
                "unique_id": ["a"] * 6 + ["b"] * 6,
                "ds": pd.date_range("2024-01-01", periods=6, freq="D").tolist() * 2,
                "y": list(range(6)) + list(range(6)),
            }
        )

        trimmed = trim_frame_for_recommendation(frame, holdout_size=2)

        assert trimmed.groupby("unique_id").size().tolist() == [4, 4]
        assert trimmed.groupby("unique_id")["y"].max().tolist() == [3, 3]

    def test_auto_timebase_recommend_defaults(self) -> None:
        """AutoTimeBase should expose heuristic defaults before any search."""
        frame = pd.DataFrame(
            {
                "unique_id": ["a"] * 60 + ["b"] * 60,
                "ds": list(pd.date_range("2024-01-01", periods=60, freq="D")) * 2,
                "y": [float(step % 7) for step in range(60)] * 2,
            }
        )

        defaults = AutoTimeBase.recommend_defaults(
            frame,
            freq="D",
            horizon=14,
            max_steps=120,
            include_iteration_recommendation=True,
        )

        assert defaults["input_size"] >= 16
        assert defaults["period_len"] >= 2
        assert defaults["recommended_training_iterations"] >= defaults["max_steps"]

    def test_auto_timebase_trend_recommend_defaults(self) -> None:
        """AutoTimeBaseTrend should include decomposition-aware defaults."""
        frame = pd.DataFrame(
            {
                "unique_id": ["a"] * 36 + ["b"] * 36,
                "ds": list(pd.date_range("2024-01-31", periods=36, freq="ME")) * 2,
                "y": [float(step % 12) for step in range(36)] * 2,
            }
        )

        defaults = AutoTimeBaseTrend.recommend_defaults(
            frame,
            freq="ME",
            horizon=6,
            max_steps=80,
            include_iteration_recommendation=True,
        )

        assert defaults["moving_avg_window"] % 2 == 1
        assert defaults["recommended_training_iterations"] >= defaults["max_steps"]


class TestIntegration:
    """Integration tests with NeuralForecast.

    Note: These tests require full NeuralForecast training which can be slow.
    They are marked as integration tests and may be skipped in CI.
    """

    @pytest.mark.integration
    def test_neuralforecast_fit_predict(self) -> None:
        """Minimal NeuralForecast fit/predict works."""
        pytest.importorskip("neuralforecast")
        import pandas as pd
        from neuralforecast import NeuralForecast

        # Create simple test data
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
            warnings.filterwarnings(
                "ignore",
                category=Warning,
            )
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
    def test_auto_timebase_fit_predict(self) -> None:
        """AutoTimeBase should fit and predict through NeuralForecast."""
        pytest.importorskip("neuralforecast")
        import pandas as pd
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
    def test_multiseries_training_can_predict_only_one_series(self) -> None:
        """A model trained on multiple series should still predict one series."""
        pytest.importorskip("neuralforecast")
        import pandas as pd
        from neuralforecast import NeuralForecast

        np = pytest.importorskip("numpy")
        np.random.seed(42)
        n = 100
        n_series = 3

        frames = []
        for i in range(n_series):
            dates = pd.date_range(start="2020-01-01", periods=n, freq="D")
            y = np.random.randn(n).cumsum() + 10 + i
            frames.append(
                pd.DataFrame({"ds": dates, "y": y, "unique_id": f"series_{i}"})
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
            warnings.filterwarnings(
                "ignore",
                category=Warning,
            )
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
