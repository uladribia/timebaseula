"""Tests for the synthetic series generator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.utils.synthetic_series import make_synthetic_series


class TestSyntheticSeries:
    """Validate synthetic series generation."""

    def test_generates_expected_columns(self) -> None:
        """The generator should return the NeuralForecast-ready schema."""
        frame = make_synthetic_series(length=48, noise_std=0.0)
        assert list(frame.columns) == ["unique_id", "ds", "y"]
        assert len(frame) == 48
        first_id = frame["unique_id"].iloc[0]
        assert frame["unique_id"].eq(first_id).all()
        assert pd.api.types.is_datetime64_any_dtype(frame["ds"])

    def test_no_components_returns_zero_series(self) -> None:
        """Disabling trend and seasonality should yield zeros (no noise)."""
        frame = make_synthetic_series(
            length=12,
            noise_std=0.0,
            include_trend=False,
            include_seasonality=False,
        )
        assert np.allclose(frame["y"].to_numpy(), 0.0)

    def test_trend_only_is_monotonic(self) -> None:
        """Trend-only series should be non-decreasing without noise."""
        frame = make_synthetic_series(
            length=20,
            noise_std=0.0,
            include_trend=True,
            include_seasonality=False,
        )
        diffs = np.diff(frame["y"].to_numpy())
        assert np.all(diffs >= 0)

    def test_seasonality_repeats(self) -> None:
        """Seasonal series should repeat at the specified period."""
        frame = make_synthetic_series(
            length=48,
            noise_std=0.0,
            include_trend=False,
            include_seasonality=True,
            season_period=12,
        )
        values = frame["y"].to_numpy()
        assert np.isclose(values[0], values[12])
        assert np.isclose(values[5], values[17])

    def test_noise_is_deterministic(self) -> None:
        """Noise should be deterministic for a fixed seed."""
        frame_a = make_synthetic_series(length=30, noise_std=0.1, seed=42)
        frame_b = make_synthetic_series(length=30, noise_std=0.1, seed=42)
        assert np.allclose(frame_a["y"].to_numpy(), frame_b["y"].to_numpy())

    def test_amplitude_modulation_changes_scale(self) -> None:
        """Amplitude modulation should alter the seasonal scale over time."""
        frame = make_synthetic_series(
            length=48,
            noise_std=0.0,
            include_trend=False,
            include_seasonality=True,
            season_period=12,
            amplitude_period=24,
            amplitude_strength=0.5,
        )
        values = frame["y"].to_numpy()
        first_window = np.abs(values[:12]).max()
        second_window = np.abs(values[12:24]).max()
        assert not np.isclose(first_window, second_window)

    def test_amplitude_growth_increases_scale(self) -> None:
        """Amplitude growth should increase oscillation magnitude over time."""
        frame = make_synthetic_series(
            length=60,
            noise_std=0.0,
            include_trend=False,
            include_seasonality=True,
            season_period=12,
            amplitude_growth_rate=1.0,
        )
        values = frame["y"].to_numpy()
        early = np.abs(values[:12]).max()
        late = np.abs(values[-12:]).max()
        assert late > early

    @pytest.mark.parametrize("bad_length", [0, -5])
    def test_invalid_length(self, bad_length: int) -> None:
        """Invalid lengths should raise a ValueError."""
        with pytest.raises(ValueError, match="length must be positive"):
            make_synthetic_series(length=bad_length, noise_std=0.1)

    def test_invalid_noise(self) -> None:
        """Negative noise should raise a ValueError."""
        with pytest.raises(ValueError, match="noise_std must be non-negative"):
            make_synthetic_series(length=10, noise_std=-0.1)

    def test_invalid_season_period(self) -> None:
        """Invalid season period should raise a ValueError."""
        with pytest.raises(ValueError, match="season_period must be positive"):
            make_synthetic_series(length=10, noise_std=0.1, season_period=0)

    def test_invalid_amplitude_period(self) -> None:
        """Invalid amplitude period should raise a ValueError."""
        with pytest.raises(ValueError, match="amplitude_period must be positive"):
            make_synthetic_series(
                length=10,
                noise_std=0.1,
                amplitude_period=0,
            )

    def test_invalid_amplitude_strength(self) -> None:
        """Invalid amplitude strength should raise a ValueError."""
        with pytest.raises(ValueError, match="amplitude_strength must be non-negative"):
            make_synthetic_series(
                length=10,
                noise_std=0.1,
                amplitude_strength=-0.1,
            )

    def test_invalid_amplitude_growth_rate(self) -> None:
        """Invalid amplitude growth rate should raise a ValueError."""
        with pytest.raises(
            ValueError, match="amplitude_growth_rate must be non-negative"
        ):
            make_synthetic_series(
                length=10,
                noise_std=0.1,
                amplitude_growth_rate=-0.2,
            )
