# Plan: TimeBaseUla Integration (TimeBase → NeuralForecast-compatible)

## Goal
Design a clean, maintainable implementation of the TimeBase forecasting method inside this library, compatible with Nixtla’s NeuralForecast API, following the repo’s coding standards (CPU-first, ruff/ty/pytest, MkDocs). **No implementation in this phase.**

---

## 1) Synthetic Test Series (first step)

**Goal:** create a lightweight, deterministic synthetic time series generator used throughout development to validate correctness.

**Planned artifact:** `tests/utils/synthetic_series.py`
- Function: `make_synthetic_series(length, noise_std, include_trend=True, include_seasonality=True, season_period=24, seed=0)`.
- Produces a pandas DataFrame or numpy array suitable for NeuralForecast (e.g., `ds`, `y`, `unique_id`).
- Configurable components:
  - Trend (linear slope)
  - Seasonality (sinusoidal)
  - Gaussian noise with tunable `noise_std`
- Keep series length small (e.g., 500–2000 points) for fast CPU tests.

**Success criterion for "works":**
- Achieve MAE below a defined threshold on the synthetic dataset with low noise (e.g., `noise_std <= 0.05`).
- Use this dataset as the primary smoke test until all models pass reliably.

---

## 2) Paper Findings (docs/huang25az.pdf)

**Core idea:**
- TimeBase is a minimalistic LTSF model that leverages **segment-level forecasting** and **basis extraction** with **two linear layers**.
- Time series are segmented by a period length **P**, then a low-rank basis of size **R** is learned to represent segment patterns.
- Forecasting is done on segments, then flattened back to point-level predictions.

**Key equations and components:**
- Segment the history: `Xhis = Segment[N,P](X)` where `N = ceil(T / P)`.
- Basis extraction: `Xbasis = BasisExtract(Xhis)` (linear layer).
- Segment-level forecasting: `Xpred = SegmentForecast(Xbasis)` (linear layer).
- Flatten to final horizon: `Y = Flatten(Xpred)[:L]`.
- **Orthogonal regularization** for basis:
  - Gram matrix `G = Xbasis^T Xbasis`.
  - `Lorth = ||G - diag(G)||_F^2`.
  - Total loss: `L = Lprediction + λorth * Lorth`.

**Implementation notes from paper:**
- Use **Channel Independence** for multivariate series (treat each channel as separate univariate forecasting task).
- `P` should be the natural period (e.g., 24 for hourly daily seasonality); use FFT if no clear period.
- Typical `R` is small (e.g., 6).
- Reported as ultra-lightweight, competitive accuracy.

---

## 3) TimeBase Code Inspection (../TimeBase)

**Model architecture (models/TimeBase.py):**
- Inputs `x` shape: `(batch, time, channels)`.
- Padding to nearest multiple of `period_len` for both history and horizon.
- Reshape: `(batch, channels, seg_num_x, period_len)` → `(batch*channels, period_len, seg_num_x)`.
- **Normalization options:**
  - `use_period_norm`: subtract per-period mean.
  - else: subtract overall mean per series.
- **Basis extraction:** linear mapping from `seg_num_x → basis_num`.
- **Forecasting:** linear mapping from `basis_num → seg_num_y`.
- **Orthogonal loss:** computed on basis `x_basis` if enabled.
- Output reshaped back to `(batch, pred_len, channels)`.

**Other notes:**
- `individual` flag creates per-channel linear layers.
- Training script adds orthogonal loss via `orthogonal_weight`.
- Uses CUDA-specific tensors in augmentations and tools (needs CPU-safe refactor).

---

## 4) NeuralForecast Inspection (via uv pip)

**Key integration points:**
- **BaseWindows** (common/_base_windows.py):
  - Expects `forward(windows_batch)` to return `(batch, h, outputsize_multiplier)`.
  - Handles windowing, normalization (TemporalNorm), and loss calculation.
  - `training_step` computes `loss = self.loss(...)` on output.
- **BaseMultivariate** for multi-series outputs, with different window shape.
- Standard models (DLinear, NLinear) subclass BaseWindows; they use `insample_y` from `windows_batch`.

**Implications:**
- TimeBase should be implemented as a **BaseWindows** model (univariate per series) unless we explicitly add a multivariate variant.
- Orthogonal regularization requires overriding `training_step` to add `λorth * orth_loss`.
- Internal normalization in TimeBase should be coordinated with `TemporalNorm` to avoid double-normalization.

---

## 5) Proposed Design for TimeBaseUla

### 4.1 Module layout
- `timebaseula/models/timebase.py`
  - `TimeBaseCore(nn.Module)` for reusable segment/basis logic.
  - `TimeBase(BaseWindows)` wrapper for NeuralForecast compatibility.
  - `TimeBaseTrend(BaseWindows)` variant mixing DLinear + TimeBase for trend-heavy series.
- `timebaseula/__init__.py` exports `TimeBase`, `TimeBaseTrend`, and version.
- `docs/usage.md` + `docs/index.md` updated with API usage and examples.

### 4.2 API and hyperparameters (NeuralForecast-style)
- **Required:** `h`, `input_size`.
- **TimeBase params:**
  - `period_len` (P)
  - `basis_num` (R)
  - `use_period_norm` (bool)
  - `use_orthogonal` (bool)
  - `orthogonal_weight` (λorth)
  - `channel_independence` (default True; for multivariate use cases)
- **TimeBaseTrend params (adds trend branch):**
  - `moving_avg_window` (int, odd) for DLinear-style trend/seasonal split.
  - `trend_weight` (float, optional learnable or fixed) to blend trend and base outputs.
- **NeuralForecast standard params:** loss, valid_loss, learning_rate, max_steps, val_check_steps, batch_size, etc.

### 4.3 Core algorithm (TimeBaseCore)
- Input `x` as `(batch, time)` (univariate) or `(batch, time, channels)` (optional).
- Pad history to nearest multiple of `period_len`.
- Segment into `(batch, period_len, seg_num_x)`.
- Normalize (`period` or `series` mean) and store stats.
- Linear map `seg_num_x → basis_num` (basis extraction).
- Linear map `basis_num → seg_num_y` (segment forecasting).
- Denormalize and reshape to `(batch, h)` or `(batch, h, channels)`.
- Orthogonal loss computed from basis tensor (Gram matrix), if enabled.

### 4.4 NeuralForecast wrapper (TimeBase)
- Subclass `BaseWindows`.
- `forward(windows_batch)`:
  - Use `insample_y = windows_batch["insample_y"]`.
  - Call `TimeBaseCore` to produce `(batch, h)`.
  - Reshape to `(batch, h, outputsize_multiplier)` and apply `loss.domain_map`.
- Override `training_step` to:
  - Call parent logic to get `outsample_y` and base loss.
  - Add `orthogonal_weight * orthogonal_loss` when enabled.
  - Keep validation/prediction behavior aligned with BaseWindows.

### 4.5 NeuralForecast wrapper (TimeBaseTrend)
- Subclass `BaseWindows`.
- **Architecture:** DLinear-style decomposition + TimeBase basis forecast.
  - Use `SeriesDecomp` (moving average) to split `insample_y` into `trend_init` and `seasonal_init`.
  - Apply **TimeBaseCore** to `seasonal_init` (segment/basis forecast).
  - Apply **linear trend head** to `trend_init` (mirrors DLinear trend branch).
  - Combine outputs: `forecast = trend_weight * trend_forecast + (1 - trend_weight) * timebase_forecast`.
- Training_step mirrors TimeBase with optional orthogonal loss; trend branch uses same output loss.

### 4.6 Normalization policy
- Default `scaler_type="identity"` to avoid double normalization.
- Expose `scaler_type` for advanced users.
- Ensure TimeBase internal normalization mirrors the paper (and applies consistently to TimeBaseTrend).

### 4.7 CPU-only compliance
- No CUDA-specific calls (`torch.cuda.*`) in core logic.
- Use device-agnostic tensors (`x.device`).

---

## 6) Testing Strategy (pytest)

**Test modules:** `tests/test_timebase.py`
- **Synthetic data fixtures:**
  - Use `make_synthetic_series` to generate train/val/test splits.
  - Cover low-noise (primary) and moderate-noise scenarios.
- **Shape tests:**
  - Input `(batch, input_size)` → output `(batch, h, 1)`.
  - Padding behavior for non-multiple `period_len`.
- **Orthogonal loss tests:**
  - Enabled/disabled behavior; ensure scalar loss added.
- **TimeBaseTrend tests:**
  - Trend-heavy synthetic data shows stable output shapes.
  - Moving average window validation (odd window size).
- **Deterministic behavior:**
  - Fixed seed returns repeatable outputs.
- **Integration smoke test:**
  - Minimal `NeuralForecast` fit/predict with synthetic data (CPU only).
  - Assert MAE below threshold for low-noise dataset.

---

## 7) Documentation Plan

- Update `README.md` and `docs/usage.md` with:
  - Installation (uv)
  - Example usage with NeuralForecast API
  - Hyperparameter descriptions (P, R, λorth) and TimeBaseTrend additions
  - Note on internal normalization vs. scaler_type
- Update `docs/index.md` “Key features” to include TimeBase and TimeBaseTrend.
- Mention agent-made changes when applicable.

---

## 8) Implementation Checklist (when execution begins)

1. Create `tests/utils/synthetic_series.py` with `make_synthetic_series` and fixtures.
2. Add `timebaseula/models/timebase.py` with `TimeBaseCore`, `TimeBase`, and `TimeBaseTrend`.
3. Export in `timebaseula/__init__.py`.
4. Write tests (TDD: red → green), including MAE threshold checks on low-noise synthetic data.
5. Run `make format`, `make lint`, `make test`.
6. Update docs with `write-docs` skill.
7. Commit with Conventional Commits via `commit` skill.

---

## Open Questions / Decisions to Confirm
- What MAE threshold should be considered "small enough" for low-noise synthetic data (e.g., <= 0.05 or <= 0.02)? Around 0.1
- Should we provide a **multivariate** variant (BaseMultivariate) or keep univariate per-series only? Both options are needed. Also, note that an option should be given to, even if a multivariate model has been trained, to generate predictions for a single series.
- Should `use_period_norm` be default True (paper-style) or False (to rely on TemporalNorm)? Period norm is ok, but make sure to properly document it.
- Should `basis_num` be configurable per-series or global? (Likely global for simplicity.) Global is ok.
- Should `trend_weight` be fixed (hyperparameter) or learnable (nn.Parameter) in TimeBaseTrend? It should be learnable.

---

**No code changes beyond this plan.**
