"""Tests for the TimeBase models."""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st
import pytest
import torch
from neuralforecast.losses.pytorch import DistributionLoss, MQLoss

from tests.property_strategies import (
    DistributionCase,
    ModelCase,
    distribution_cases,
    even_integers,
    multivariate_model_cases,
    padded_model_cases,
    univariate_model_cases,
)
from timebaseula.models.decomposition import SeriesDecomposition
from timebaseula.models.timebase import TimeBase, TimeBaseTrend


class TestTimeBase:
    """Validate TimeBase behavior."""

    @given(h=st.integers(min_value=1, max_value=32))
    def test_default_init_uses_horizon_based_defaults(self, h: int) -> None:
        """TimeBase should provide deterministic defaults without profiling."""
        model = TimeBase(h=h)

        assert model.input_size == max(2 * h, 8)
        assert model.core.period_len == min(max(2, h), model.input_size)
        assert model.core.basis_num == 6
        assert model.trainer_kwargs["accelerator"] == "cpu"
        assert model.trainer_kwargs["devices"] == 1

    @given(h=st.integers(min_value=1, max_value=32))
    def test_daily_frequency_prefers_weekly_period_default(self, h: int) -> None:
        """Daily models should default to a weekly period when freq is available."""
        model = TimeBase(h=h, freq="D")

        assert model.input_size == max(2 * h, 8)
        assert model.core.period_len == min(7, model.input_size)

    @given(case=univariate_model_cases())
    def test_forward_shape(
        self,
        case: ModelCase,
    ) -> None:
        """The forward output should match (batch, horizon)."""
        model = TimeBase(
            h=case.h,
            input_size=case.input_size,
            period_len=case.period_len,
            basis_num=case.basis_num,
        )
        windows_batch = {"insample_y": case.insample_y}
        output = model(windows_batch)
        assert output.shape == (case.insample_y.shape[0], case.h)

    @given(case=multivariate_model_cases())
    def test_forward_multivariate_shape(
        self,
        case: ModelCase,
    ) -> None:
        """The forward output should keep a trailing series dimension."""
        model = TimeBase(
            h=case.h,
            input_size=case.input_size,
            period_len=case.period_len,
            basis_num=case.basis_num,
        )
        windows_batch = {"insample_y": case.insample_y}
        output = model(windows_batch)
        assert output.shape == (
            case.insample_y.shape[0],
            case.h,
            case.insample_y.shape[-1],
        )

    @given(case=padded_model_cases())
    def test_padding_output_shape(
        self,
        case: ModelCase,
    ) -> None:
        """Padding should still yield the requested horizon length."""
        model = TimeBase(
            h=case.h,
            input_size=case.input_size,
            period_len=case.period_len,
            basis_num=case.basis_num,
        )
        windows_batch = {"insample_y": case.insample_y}
        output = model(windows_batch)
        assert case.input_size % case.period_len != 0
        assert output.shape == (case.insample_y.shape[0], case.h)

    @given(rank=st.integers(min_value=1, max_value=8))
    def test_orthogonal_loss_zero(self, rank: int) -> None:
        """Orthogonal basis should yield near-zero orthogonal loss."""
        model = TimeBase(
            h=rank, input_size=max(2 * rank, 8), period_len=rank, basis_num=rank
        )
        basis = torch.eye(rank).unsqueeze(0)
        loss = model._compute_orthogonal_loss(basis)
        assert torch.isclose(loss, torch.tensor(0.0))

    @given(
        case=univariate_model_cases(),
        seed=st.integers(min_value=0, max_value=10_000),
    )
    def test_deterministic_behavior(
        self,
        case: ModelCase,
        seed: int,
    ) -> None:
        """Same seed should produce same output."""
        windows_batch = {"insample_y": case.insample_y}

        torch.manual_seed(seed)
        model1 = TimeBase(
            h=case.h,
            input_size=case.input_size,
            period_len=case.period_len,
            basis_num=case.basis_num,
        )
        output1 = model1(windows_batch)

        torch.manual_seed(seed)
        model2 = TimeBase(
            h=case.h,
            input_size=case.input_size,
            period_len=case.period_len,
            basis_num=case.basis_num,
        )
        output2 = model2(windows_batch)

        assert torch.allclose(output1, output2)

    @pytest.mark.parametrize(
        ("loss", "expected_shapes"),
        [
            (DistributionLoss("Normal"), ((2, 4), (2, 4))),
            (DistributionLoss("Poisson"), ((2, 4),)),
            (DistributionLoss("StudentT"), ((2, 4), (2, 4), (2, 4))),
            (DistributionLoss("NegativeBinomial"), ((2, 4), (2, 4))),
            (DistributionLoss("Tweedie"), ((2, 4),)),
        ],
    )
    def test_forward_supports_distribution_losses(
        self,
        loss: DistributionLoss,
        expected_shapes: tuple[tuple[int, ...], ...],
    ) -> None:
        """TimeBase should emit parameter tensors compatible with distribution losses."""
        model = TimeBase(h=4, input_size=8, period_len=4, basis_num=4, loss=loss)

        output = model({"insample_y": torch.ones((2, 8))})

        assert isinstance(output, tuple)
        assert tuple(component.shape for component in output) == expected_shapes

    @given(
        case=distribution_cases(),
        model_case=univariate_model_cases(),
    )
    def test_forward_distribution_losses_preserve_batch_and_horizon_contract(
        self,
        case: DistributionCase,
        model_case: ModelCase,
    ) -> None:
        """Supported distribution losses should preserve batch and horizon dimensions."""
        model = case.model_cls(
            h=model_case.h,
            input_size=model_case.input_size,
            period_len=model_case.period_len,
            basis_num=model_case.basis_num,
            loss=case.loss,
        )

        output = model({"insample_y": model_case.insample_y})

        assert isinstance(output, tuple)
        assert len(output) == case.parameter_count
        assert all(
            component.shape == (model_case.insample_y.shape[0], model_case.h)
            for component in output
        )

    @given(
        model_cls=st.sampled_from([TimeBase, TimeBaseTrend]),
        model_case=univariate_model_cases(),
    )
    def test_forward_supports_multi_quantile_loss(
        self,
        model_cls: type[TimeBase] | type[TimeBaseTrend],
        model_case: ModelCase,
    ) -> None:
        """TimeBase should emit multi-output tensors for quantile-style losses."""
        model = model_cls(
            h=model_case.h,
            input_size=model_case.input_size,
            period_len=model_case.period_len,
            basis_num=model_case.basis_num,
            loss=MQLoss(),
        )

        output = model({"insample_y": model_case.insample_y})

        assert output.shape == (
            model_case.insample_y.shape[0],
            model_case.h,
            model.loss.outputsize_multiplier,
        )

    def test_sampling_type_is_multivariate(self) -> None:
        """TimeBase should use NeuralForecast multivariate sampling."""
        model = TimeBase(h=4, input_size=8, period_len=4, basis_num=4)

        assert model.SAMPLING_TYPE == "multivariate"


class TestTimeBaseTrend:
    """Validate TimeBaseTrend behavior."""

    @given(h=st.integers(min_value=1, max_value=32))
    def test_default_init_uses_simple_defaults(self, h: int) -> None:
        """TimeBaseTrend should expose deterministic defaults."""
        model = TimeBaseTrend(h=h)

        assert model.input_size == max(2 * h, 8)
        assert model.core.period_len == min(max(2, h), model.input_size)
        assert model.core.basis_num == 6
        assert model.moving_avg_window == 25
        assert model.trainer_kwargs["accelerator"] == "cpu"
        assert model.trainer_kwargs["devices"] == 1

    @given(h=st.integers(min_value=1, max_value=32))
    def test_monthly_frequency_prefers_yearly_period_default(self, h: int) -> None:
        """Monthly models should default to yearly seasonality when freq is provided."""
        model = TimeBaseTrend(h=h, freq="ME")

        assert model.input_size == max(2 * h, 8)
        assert model.core.period_len == min(12, model.input_size)
        assert model.moving_avg_window % 2 == 1

    @given(case=univariate_model_cases())
    def test_forward_shape(
        self,
        case: ModelCase,
    ) -> None:
        """The forward output should match (batch, horizon)."""
        model = TimeBaseTrend(
            h=case.h,
            input_size=case.input_size,
            period_len=case.period_len,
            basis_num=case.basis_num,
        )
        windows_batch = {"insample_y": case.insample_y}
        output = model(windows_batch)
        assert output.shape == (case.insample_y.shape[0], case.h)

    @given(case=multivariate_model_cases())
    def test_forward_multivariate_shape(
        self,
        case: ModelCase,
    ) -> None:
        """The trend model should keep a trailing series dimension."""
        model = TimeBaseTrend(
            h=case.h,
            input_size=case.input_size,
            period_len=case.period_len,
            basis_num=case.basis_num,
        )
        windows_batch = {"insample_y": case.insample_y}
        output = model(windows_batch)
        assert output.shape == (
            case.insample_y.shape[0],
            case.h,
            case.insample_y.shape[-1],
        )

    def test_linear_trend_head_is_present(self) -> None:
        """The linear trend layer should be present."""
        model = TimeBaseTrend(h=4, input_size=8, period_len=4, basis_num=4)
        assert hasattr(model, "linear_trend")
        assert isinstance(model.linear_trend, torch.nn.Linear)

    @given(moving_avg_window=even_integers(min_value=1, max_value=16))
    def test_invalid_moving_avg_window_even(self, moving_avg_window: int) -> None:
        """Even moving_avg_window should raise ValueError."""
        with pytest.raises(ValueError):
            TimeBaseTrend(
                h=4,
                input_size=8,
                period_len=4,
                basis_num=4,
                moving_avg_window=moving_avg_window,
            )

    def test_has_local_decomposition_module(self) -> None:
        """TimeBaseTrend should use the local Torch decomposition helper."""
        model = TimeBaseTrend(h=4, input_size=8, period_len=4, basis_num=4)
        assert hasattr(model, "decomp")
        assert isinstance(model.decomp, SeriesDecomposition)

    @pytest.mark.parametrize(
        ("loss", "expected_shapes"),
        [
            (DistributionLoss("Normal"), ((2, 4), (2, 4))),
            (DistributionLoss("Poisson"), ((2, 4),)),
            (DistributionLoss("StudentT"), ((2, 4), (2, 4), (2, 4))),
            (DistributionLoss("NegativeBinomial"), ((2, 4), (2, 4))),
            (DistributionLoss("Tweedie"), ((2, 4),)),
        ],
    )
    def test_forward_supports_distribution_losses(
        self,
        loss: DistributionLoss,
        expected_shapes: tuple[tuple[int, ...], ...],
    ) -> None:
        """TimeBaseTrend should emit parameter tensors compatible with distribution losses."""
        model = TimeBaseTrend(
            h=4,
            input_size=8,
            period_len=4,
            basis_num=4,
            loss=loss,
        )

        output = model({"insample_y": torch.ones((2, 8))})

        assert isinstance(output, tuple)
        assert tuple(component.shape for component in output) == expected_shapes

    def test_sampling_type_is_multivariate(self) -> None:
        """TimeBaseTrend should use NeuralForecast multivariate sampling."""
        model = TimeBaseTrend(h=4, input_size=8, period_len=4, basis_num=4)

        assert model.SAMPLING_TYPE == "multivariate"
