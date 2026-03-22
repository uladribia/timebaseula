"""Tests for the TimeBase model."""

from __future__ import annotations

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
