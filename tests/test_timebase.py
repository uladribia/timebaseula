"""Tests for the TimeBase model."""

from __future__ import annotations

import pytest
import torch

from timebaseula.models.timebase import TimeBase, TimeBaseTrend


class TestTimeBase:
    """Validate TimeBase forward behavior."""

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

    def test_forward_shape(self) -> None:
        """The forward output should match (batch, h)."""
        model = TimeBaseTrend(h=12, input_size=24, period_len=6, basis_num=4)
        windows_batch = {"insample_y": torch.ones((2, 24))}
        output = model(windows_batch)
        assert output.shape == (2, 12)

    def test_trend_weight_is_learnable(self) -> None:
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
        )

        nf = NeuralForecast(models=[model], freq="D")
        nf.fit(df, val_size=8)

        pred = nf.predict()
        assert len(pred) == 8
        assert "TimeBase" in pred.columns

    @pytest.mark.integration
    def test_multivariate_fit(self) -> None:
        """Multivariate training with multiple series works."""
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
        )

        nf = NeuralForecast(models=[model], freq="D")
        nf.fit(df, val_size=8)

        pred = nf.predict()
        assert len(pred) == 8 * n_series  # Predictions for all series
