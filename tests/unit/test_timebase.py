"""Tests for the TimeBase models."""

from __future__ import annotations

import pytest
import torch

from timebaseula.models.timebase import TimeBase, TimeBaseTrend


class TestTimeBase:
    """Validate TimeBase behavior."""

    def test_default_init_uses_horizon_based_defaults(self) -> None:
        """TimeBase should provide deterministic defaults without profiling."""
        model = TimeBase(h=12)

        assert model.input_size == 24
        assert model.core.period_len == 12
        assert model.core.basis_num == 6
        assert model.trainer_kwargs["accelerator"] == "cpu"
        assert model.trainer_kwargs["devices"] == 1

    def test_daily_frequency_prefers_weekly_period_default(self) -> None:
        """Daily models should default to a weekly period when freq is available."""
        model = TimeBase(h=14, freq="D")

        assert model.input_size == 28
        assert model.core.period_len == 7

    def test_forward_shape(self) -> None:
        """The forward output should match (batch, horizon)."""
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
    """Validate TimeBaseTrend behavior."""

    def test_default_init_uses_simple_defaults(self) -> None:
        """TimeBaseTrend should expose deterministic defaults."""
        model = TimeBaseTrend(h=12)

        assert model.input_size == 24
        assert model.core.period_len == 12
        assert model.core.basis_num == 6
        assert model.moving_avg_window == 25
        assert model.trainer_kwargs["accelerator"] == "cpu"
        assert model.trainer_kwargs["devices"] == 1

    def test_monthly_frequency_prefers_yearly_period_default(self) -> None:
        """Monthly models should default to yearly seasonality when freq is provided."""
        model = TimeBaseTrend(h=6, freq="ME")

        assert model.input_size == 12
        assert model.core.period_len == 12
        assert model.moving_avg_window % 2 == 1

    def test_forward_shape(self) -> None:
        """The forward output should match (batch, horizon)."""
        model = TimeBaseTrend(h=12, input_size=24, period_len=6, basis_num=4)
        windows_batch = {"insample_y": torch.ones((2, 24))}
        output = model(windows_batch)
        assert output.shape == (2, 12)

    def test_linear_trend_head_is_present(self) -> None:
        """The linear trend layer should be present."""
        model = TimeBaseTrend(h=4, input_size=8, period_len=4, basis_num=4)
        assert hasattr(model, "linear_trend")
        assert isinstance(model.linear_trend, torch.nn.Linear)

    def test_invalid_moving_avg_window_even(self) -> None:
        """Even moving_avg_window should raise ValueError."""
        with pytest.raises(ValueError):
            TimeBaseTrend(
                h=4,
                input_size=8,
                period_len=4,
                basis_num=4,
                moving_avg_window=4,
            )

    def test_has_decomposition(self) -> None:
        """TimeBaseTrend should have SeriesDecomp for trend extraction."""
        model = TimeBaseTrend(h=4, input_size=8, period_len=4, basis_num=4)
        assert hasattr(model, "decomp")
