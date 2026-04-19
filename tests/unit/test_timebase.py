"""Tests for the TimeBase models."""

from __future__ import annotations

from typing import Any, cast

from hypothesis import given
from hypothesis import strategies as st
import pytest
import torch
from neuralforecast.losses.pytorch import DistributionLoss, MQLoss

from timebaseula.models.decomposition import SeriesDecomposition
from timebaseula.models.timebase import TimeBase, TimeBaseTrend

FINITE_FLOATS = st.floats(
    min_value=-50,
    max_value=50,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)


def _odd_ints(min_value: int, max_value: int) -> st.SearchStrategy[int]:
    return st.integers(min_value=min_value, max_value=max_value).map(
        lambda value: 2 * value - 1
    )


def _even_ints(min_value: int, max_value: int) -> st.SearchStrategy[int]:
    return st.integers(min_value=min_value, max_value=max_value).map(
        lambda value: 2 * value
    )


@st.composite
def _tensor_2d_for_length(draw: st.DrawFn, length: int) -> torch.Tensor:
    batch_size = draw(st.integers(min_value=1, max_value=4))
    values = draw(
        st.lists(
            st.lists(FINITE_FLOATS, min_size=length, max_size=length),
            min_size=batch_size,
            max_size=batch_size,
        )
    )
    return torch.tensor(values, dtype=torch.float32)


@st.composite
def _tensor_3d_for_length(draw: st.DrawFn, length: int) -> torch.Tensor:
    batch_size = draw(st.integers(min_value=1, max_value=3))
    n_series = draw(st.integers(min_value=1, max_value=4))
    values = draw(
        st.lists(
            st.lists(
                st.lists(FINITE_FLOATS, min_size=n_series, max_size=n_series),
                min_size=length,
                max_size=length,
            ),
            min_size=batch_size,
            max_size=batch_size,
        )
    )
    return torch.tensor(values, dtype=torch.float32)


def _tensor_2d_strategy_for_length(length: int) -> st.SearchStrategy[torch.Tensor]:
    return cast(Any, _tensor_2d_for_length)(length)


def _tensor_3d_strategy_for_length(length: int) -> st.SearchStrategy[torch.Tensor]:
    return cast(Any, _tensor_3d_for_length)(length)


@st.composite
def _model_cases(
    draw: st.DrawFn,
    *,
    multivariate: bool,
) -> tuple[int, int, int, int, torch.Tensor]:
    h = draw(st.integers(min_value=1, max_value=10))
    input_size = draw(st.integers(min_value=1, max_value=24))
    period_len = draw(st.integers(min_value=1, max_value=min(input_size, 8)))
    basis_num = draw(st.integers(min_value=1, max_value=6))
    if multivariate:
        insample_y = draw(_tensor_3d_strategy_for_length(input_size))
    else:
        insample_y = draw(_tensor_2d_strategy_for_length(input_size))
    return h, input_size, period_len, basis_num, insample_y


@st.composite
def _padded_model_cases(draw: st.DrawFn) -> tuple[int, int, int, int, torch.Tensor]:
    h = draw(st.integers(min_value=1, max_value=10))
    period_len = draw(st.integers(min_value=2, max_value=8))
    quotient = draw(st.integers(min_value=1, max_value=3))
    remainder = draw(st.integers(min_value=1, max_value=period_len - 1))
    input_size = quotient * period_len + remainder
    basis_num = draw(st.integers(min_value=1, max_value=6))
    insample_y = draw(_tensor_2d_strategy_for_length(input_size))
    return h, input_size, period_len, basis_num, insample_y


@st.composite
def _distribution_cases(
    draw: st.DrawFn,
) -> tuple[type[TimeBase] | type[TimeBaseTrend], DistributionLoss, int]:
    model_cls = draw(st.sampled_from([TimeBase, TimeBaseTrend]))
    loss_name = draw(
        st.sampled_from(
            [
                ("Normal", 2),
                ("Poisson", 1),
                ("StudentT", 3),
                ("NegativeBinomial", 2),
                ("Tweedie", 1),
            ]
        )
    )
    return model_cls, DistributionLoss(loss_name[0]), loss_name[1]


def _univariate_model_case_strategy() -> st.SearchStrategy[
    tuple[int, int, int, int, torch.Tensor]
]:
    return cast(Any, _model_cases)(multivariate=False)


def _multivariate_model_case_strategy() -> st.SearchStrategy[
    tuple[int, int, int, int, torch.Tensor]
]:
    return cast(Any, _model_cases)(multivariate=True)


def _padded_model_case_strategy() -> st.SearchStrategy[
    tuple[int, int, int, int, torch.Tensor]
]:
    return cast(Any, _padded_model_cases)()


def _distribution_case_strategy() -> st.SearchStrategy[
    tuple[type[TimeBase] | type[TimeBaseTrend], DistributionLoss, int]
]:
    return cast(Any, _distribution_cases)()


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

    @given(case=_univariate_model_case_strategy())
    def test_forward_shape(
        self,
        case: tuple[int, int, int, int, torch.Tensor],
    ) -> None:
        """The forward output should match (batch, horizon)."""
        h, input_size, period_len, basis_num, insample_y = case
        model = TimeBase(
            h=h,
            input_size=input_size,
            period_len=period_len,
            basis_num=basis_num,
        )
        windows_batch = {"insample_y": insample_y}
        output = model(windows_batch)
        assert output.shape == (insample_y.shape[0], h)

    @given(case=_multivariate_model_case_strategy())
    def test_forward_multivariate_shape(
        self,
        case: tuple[int, int, int, int, torch.Tensor],
    ) -> None:
        """The forward output should keep a trailing series dimension."""
        h, input_size, period_len, basis_num, insample_y = case
        model = TimeBase(
            h=h,
            input_size=input_size,
            period_len=period_len,
            basis_num=basis_num,
        )
        windows_batch = {"insample_y": insample_y}
        output = model(windows_batch)
        assert output.shape == (insample_y.shape[0], h, insample_y.shape[-1])

    @given(case=_padded_model_case_strategy())
    def test_padding_output_shape(
        self,
        case: tuple[int, int, int, int, torch.Tensor],
    ) -> None:
        """Padding should still yield the requested horizon length."""
        h, input_size, period_len, basis_num, insample_y = case
        model = TimeBase(
            h=h,
            input_size=input_size,
            period_len=period_len,
            basis_num=basis_num,
        )
        windows_batch = {"insample_y": insample_y}
        output = model(windows_batch)
        assert input_size % period_len != 0
        assert output.shape == (insample_y.shape[0], h)

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
        case=_univariate_model_case_strategy(),
        seed=st.integers(min_value=0, max_value=10_000),
    )
    def test_deterministic_behavior(
        self,
        case: tuple[int, int, int, int, torch.Tensor],
        seed: int,
    ) -> None:
        """Same seed should produce same output."""
        h, input_size, period_len, basis_num, insample_y = case
        windows_batch = {"insample_y": insample_y}

        torch.manual_seed(seed)
        model1 = TimeBase(
            h=h,
            input_size=input_size,
            period_len=period_len,
            basis_num=basis_num,
        )
        output1 = model1(windows_batch)

        torch.manual_seed(seed)
        model2 = TimeBase(
            h=h,
            input_size=input_size,
            period_len=period_len,
            basis_num=basis_num,
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
        case=_distribution_case_strategy(),
        model_case=_univariate_model_case_strategy(),
    )
    def test_forward_distribution_losses_preserve_batch_and_horizon_contract(
        self,
        case: tuple[type[TimeBase] | type[TimeBaseTrend], DistributionLoss, int],
        model_case: tuple[int, int, int, int, torch.Tensor],
    ) -> None:
        """Supported distribution losses should preserve batch and horizon dimensions."""
        model_cls, loss, parameter_count = case
        h, input_size, period_len, basis_num, insample_y = model_case
        model = model_cls(
            h=h,
            input_size=input_size,
            period_len=period_len,
            basis_num=basis_num,
            loss=loss,
        )

        output = model({"insample_y": insample_y})

        assert isinstance(output, tuple)
        assert len(output) == parameter_count
        assert all(component.shape == (insample_y.shape[0], h) for component in output)

    @given(
        model_cls=st.sampled_from([TimeBase, TimeBaseTrend]),
        model_case=_univariate_model_case_strategy(),
    )
    def test_forward_supports_multi_quantile_loss(
        self,
        model_cls: type[TimeBase] | type[TimeBaseTrend],
        model_case: tuple[int, int, int, int, torch.Tensor],
    ) -> None:
        """TimeBase should emit multi-output tensors for quantile-style losses."""
        h, input_size, period_len, basis_num, insample_y = model_case
        model = model_cls(
            h=h,
            input_size=input_size,
            period_len=period_len,
            basis_num=basis_num,
            loss=MQLoss(),
        )

        output = model({"insample_y": insample_y})

        assert output.shape == (
            insample_y.shape[0],
            h,
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

    @given(case=_univariate_model_case_strategy())
    def test_forward_shape(
        self,
        case: tuple[int, int, int, int, torch.Tensor],
    ) -> None:
        """The forward output should match (batch, horizon)."""
        h, input_size, period_len, basis_num, insample_y = case
        model = TimeBaseTrend(
            h=h,
            input_size=input_size,
            period_len=period_len,
            basis_num=basis_num,
        )
        windows_batch = {"insample_y": insample_y}
        output = model(windows_batch)
        assert output.shape == (insample_y.shape[0], h)

    @given(case=_multivariate_model_case_strategy())
    def test_forward_multivariate_shape(
        self,
        case: tuple[int, int, int, int, torch.Tensor],
    ) -> None:
        """The trend model should keep a trailing series dimension."""
        h, input_size, period_len, basis_num, insample_y = case
        model = TimeBaseTrend(
            h=h,
            input_size=input_size,
            period_len=period_len,
            basis_num=basis_num,
        )
        windows_batch = {"insample_y": insample_y}
        output = model(windows_batch)
        assert output.shape == (insample_y.shape[0], h, insample_y.shape[-1])

    def test_linear_trend_head_is_present(self) -> None:
        """The linear trend layer should be present."""
        model = TimeBaseTrend(h=4, input_size=8, period_len=4, basis_num=4)
        assert hasattr(model, "linear_trend")
        assert isinstance(model.linear_trend, torch.nn.Linear)

    @given(moving_avg_window=_even_ints(min_value=1, max_value=16))
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
