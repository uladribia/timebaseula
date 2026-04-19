# Property-Based Testing

This repository uses [Hypothesis](https://hypothesis.readthedocs.io/) for
invariant-driven tests in the main library.

## Current Status

Property-based tests are implemented for the library areas where broad input-domain
validation is more useful than one-off examples.

Current coverage focus:

- `tests/unit/test_model_decomposition.py`
- `tests/unit/test_model_core.py`
- `tests/unit/test_model_defaults.py`
- `tests/unit/test_model_factories.py`
- invariant-driven parts of `tests/unit/test_timebase.py`
- horizon-dependent auto-wrapper search-space checks in `tests/unit/test_auto.py`

Shared reusable strategies live in `tests/property_strategies.py`.

## What We Use Hypothesis For

Use Hypothesis when the behavior is best expressed as a contract over many valid
inputs, especially for:

- tensor shape preservation
- reconstruction identities up to numeric tolerance
- default-resolution rules
- factory/config consistency
- deterministic behavior under identical seeds
- bounded search-space invariants

## What Stays Example-Based

Keep example-based tests for:

- public exports and import behavior
- object identity and fixed API facts
- finite supported sets where named examples improve readability
- heavy integration behavior such as real NeuralForecast fit/predict tests

## Strategy Policy

- Keep generated domains small, finite, and aligned with the real implementation
  contract.
- Prefer bounded tensor sizes and finite `float32` values.
- Avoid `assume(...)`-heavy generators when direct bounded strategies are possible.
- Keep the shared strategy plumbing in `tests/property_strategies.py` rather than
  duplicating it across test files.

## Implemented Test Split

The current property-based adoption for the library suite is:

- replace about 65% of the library unit tests with property-based checks
- expand about 11% more with additional property coverage
- keep the remaining quarter example-based where examples are clearer or cheaper

This keeps Hypothesis focused on the parts of the library where it adds real signal.

## Side-By-Side Examples

### Decomposition shape preservation

Previous example-style test:

```python
def test_moving_average_preserves_expected_shape() -> None:
    moving_average = MovingAverage(kernel_size=5)

    trend = moving_average(torch.arange(12, dtype=torch.float32).reshape(2, 6))

    assert trend.shape == (2, 6)
```

Current property-based test shape:

```python
@given(kernel_size=odd_integers(1, 17), series=tensor_2d())
def test_moving_average_preserves_expected_shape(
    kernel_size: int,
    series: torch.Tensor,
) -> None:
    moving_average = MovingAverage(kernel_size=kernel_size)

    trend = moving_average(series)

    assert trend.shape == series.shape
```

### Core forward contract

Previous example-style test:

```python
def test_timebase_core_forward_returns_forecast_and_basis() -> None:
    core = TimeBaseCore(
        config=TimeBaseConfig(
            input_size=24,
            period_len=6,
            basis_num=4,
            use_period_norm=True,
        ),
        horizon=12,
    )

    forecast, basis = core(torch.ones((2, 24)))

    assert forecast.shape == (2, 12)
    assert basis.shape == (2, 6, 4)
```

Current property-based test shape:

```python
@given(case=core_univariate_cases())
def test_timebase_core_forward_returns_forecast_and_basis(case: CoreCase) -> None:
    forecast, basis = case.core(case.series)

    assert forecast.shape == (case.series.shape[0], case.horizon)
    assert basis.shape == (case.series.shape[0], case.period_len, case.core.basis_num)
```

### Explicit-model forward shape

Previous example-style test:

```python
def test_forward_shape() -> None:
    model = TimeBase(h=12, input_size=24, period_len=6, basis_num=4)
    windows_batch = {"insample_y": torch.ones((2, 24))}

    output = model(windows_batch)

    assert output.shape == (2, 12)
```

Current property-based test shape:

```python
@given(case=univariate_model_cases())
def test_forward_shape(case: ModelCase) -> None:
    model = TimeBase(
        h=case.h,
        input_size=case.input_size,
        period_len=case.period_len,
        basis_num=case.basis_num,
    )

    output = model({"insample_y": case.insample_y})

    assert output.shape == (case.insample_y.shape[0], case.h)
```

## Maintenance Notes

- If a strategy is reused across files, add it to `tests/property_strategies.py`.
- If a property depends on floating-point reconstruction, use an explicit tolerance.
- If a property only holds for a narrower valid domain, restrict the strategy instead
  of weakening the assertion globally.
