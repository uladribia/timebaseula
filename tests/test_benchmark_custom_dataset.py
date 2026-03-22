"""Tests for the custom dataset benchmark script."""

from __future__ import annotations

import pandas as pd

from scripts.benchmark_custom_dataset import (
    add_average_ranks,
    add_relative_mae,
    add_relative_mae_summary,
    build_holdout_target,
    build_html_report,
    build_mae_distribution_section,
    build_model_comparison_section,
    build_neural_diagnostics_section,
    build_representative_plot_sections,
    build_seasonal_naive_forecast,
    choose_common_model_kwargs,
    choose_representative_series,
    load_custom_dataset,
    prepare_train_test,
)


class TestCustomDatasetBenchmark:
    """Validate helpers used by the custom dataset benchmark."""

    def test_load_custom_dataset_keeps_core_columns(self) -> None:
        frame = load_custom_dataset("datasets/custom/neuralforecast_monthly.csv")
        assert list(frame.columns) == ["unique_id", "ds", "y"]
        assert pd.api.types.is_datetime64_any_dtype(frame["ds"])
        assert frame["unique_id"].nunique() == 280

    def test_prepare_train_test_reserves_last_horizon_per_series(self) -> None:
        frame = pd.DataFrame(
            {
                "unique_id": ["a"] * 6 + ["b"] * 6,
                "ds": pd.date_range("2024-01-01", periods=6, freq="MS").tolist() * 2,
                "y": list(range(6)) + list(range(6)),
            }
        )
        train, test = prepare_train_test(frame, horizon=2)
        assert train.groupby("unique_id").size().tolist() == [4, 4]
        assert test.groupby("unique_id").size().tolist() == [2, 2]

    def test_build_holdout_target_renames_truth_column(self) -> None:
        frame = pd.DataFrame(
            {
                "unique_id": ["a", "a"],
                "ds": pd.to_datetime(["2024-05-01", "2024-06-01"]),
                "y": [1.0, 2.0],
            }
        )
        target = build_holdout_target(frame)
        assert list(target.columns) == ["unique_id", "ds", "y_true"]
        assert target["y_true"].tolist() == [1.0, 2.0]

    def test_choose_common_model_kwargs_uses_recommendations(self) -> None:
        rows = []
        for unique_id in ("a", "b"):
            for step in range(36):
                rows.append(
                    {
                        "unique_id": unique_id,
                        "ds": pd.Timestamp("2022-01-01") + pd.offsets.MonthBegin(step),
                        "y": float(step % 12),
                    }
                )
        frame = pd.DataFrame(rows)
        kwargs = choose_common_model_kwargs(frame, freq="MS", horizon=6, max_steps=30)
        assert kwargs["input_size"] >= 12
        assert kwargs["max_steps"] <= 30
        assert "learning_rate" in kwargs

    def test_choose_common_model_kwargs_hard_caps_training_budget(self) -> None:
        rows = []
        for unique_id in ("a", "b"):
            for step in range(120):
                rows.append(
                    {
                        "unique_id": unique_id,
                        "ds": pd.Timestamp("2020-01-01") + pd.offsets.MonthBegin(step),
                        "y": float(step % 12),
                    }
                )
        frame = pd.DataFrame(rows)

        kwargs = choose_common_model_kwargs(frame, freq="MS", horizon=12, max_steps=5)

        assert kwargs["max_steps"] == 5
        assert kwargs["val_check_steps"] <= 5
        assert kwargs["early_stop_patience_steps"] <= 5

    def test_build_seasonal_naive_forecast_repeats_last_season(self) -> None:
        train_frame = pd.DataFrame(
            {
                "unique_id": ["a"] * 6,
                "ds": pd.date_range("2024-01-01", periods=6, freq="MS"),
                "y": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            }
        )
        target_frame = pd.DataFrame(
            {
                "unique_id": ["a"] * 4,
                "ds": pd.date_range("2024-07-01", periods=4, freq="MS"),
                "y_true": [0.0, 0.0, 0.0, 0.0],
            }
        )
        forecast = build_seasonal_naive_forecast(
            train_frame, target_frame, horizon=4, season_length=3
        )
        assert forecast["SeasonalNaive"].tolist() == [40.0, 50.0, 60.0, 40.0]

    def test_choose_representative_series_prefers_long_var_trend_and_random(
        self,
    ) -> None:
        frame = pd.DataFrame(
            {
                "unique_id": (
                    ["a"] * 4 + ["b"] * 6 + ["c"] * 5 + ["d"] * 5 + ["e"] * 5
                ),
                "ds": pd.date_range("2024-01-01", periods=25, freq="D"),
                "y": [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                    0.0,
                    10.0,
                    0.0,
                    10.0,
                    0.0,
                    1.0,
                    2.0,
                    4.0,
                    7.0,
                    11.0,
                    3.0,
                    3.0,
                    4.0,
                    4.0,
                    5.0,
                ],
            }
        )
        selected = choose_representative_series(frame, n_examples=5)
        assert "b" in selected
        assert "c" in selected
        assert "d" in selected
        assert len(selected) == 5

    def test_add_average_ranks_computes_mean_rank_per_model(self) -> None:
        aggregate = pd.DataFrame({"model_name": ["A", "B"], "overall_mae": [1.0, 2.0]})
        per_series = pd.DataFrame(
            {
                "unique_id": ["s1", "s1", "s2", "s2"],
                "model_name": ["A", "B", "A", "B"],
                "mae": [1.0, 2.0, 3.0, 1.0],
                "rmse": [1.0, 2.0, 3.0, 1.0],
            }
        )
        ranked = add_average_ranks(aggregate, per_series)
        assert ranked.loc[ranked["model_name"] == "A", "average_rank"].item() == 1.5
        assert ranked.loc[ranked["model_name"] == "B", "average_rank"].item() == 1.5

    def test_add_relative_mae_computes_rmae_from_series_means(self) -> None:
        per_series = pd.DataFrame(
            {
                "unique_id": ["a", "b"],
                "model_name": ["TimeBase", "TimeBase"],
                "mae": [2.0, 4.0],
                "rmse": [2.5, 4.5],
            }
        )
        full_frame = pd.DataFrame(
            {
                "unique_id": ["a", "a", "b", "b"],
                "ds": pd.date_range("2024-01-01", periods=4, freq="D"),
                "y": [2.0, 2.0, 8.0, 8.0],
            }
        )
        result = add_relative_mae(per_series, full_frame)
        assert result.loc[result["unique_id"] == "a", "rmae"].item() == 1.0
        assert result.loc[result["unique_id"] == "b", "rmae"].item() == 0.5

    def test_add_relative_mae_summary_adds_aggregate_columns(self) -> None:
        aggregate = pd.DataFrame({"model_name": ["A", "B"]})
        per_series = pd.DataFrame(
            {
                "unique_id": ["s1", "s1", "s2", "s2"],
                "model_name": ["A", "B", "A", "B"],
                "mae": [1.0, 2.0, 3.0, 1.0],
                "rmse": [1.0, 2.0, 3.0, 1.0],
                "rmae": [0.1, 0.2, 0.3, 0.1],
            }
        )
        result = add_relative_mae_summary(aggregate, per_series)
        assert "mean_series_rmae" in result.columns
        assert "median_series_rmae" in result.columns
        assert result.loc[result["model_name"] == "A", "mean_series_rmae"].item() == 0.2

    def test_build_mae_distribution_section_uses_embedded_png_markup(self) -> None:
        per_series = pd.DataFrame(
            {
                "unique_id": ["a", "a", "b", "b"],
                "model_name": [
                    "TimeBase",
                    "SeasonalNaive",
                    "TimeBase",
                    "SeasonalNaive",
                ],
                "mae": [1.2, 1.0, 1.0, 0.8],
                "rmse": [1.3, 1.1, 1.1, 0.9],
                "rmae": [0.12, 0.10, 0.20, 0.16],
            }
        )
        section = build_mae_distribution_section(per_series)
        assert "Distribution of relative MAE across series" in section
        assert "data:image/png;base64," in section
        assert "Median RMAE" in section

    def test_build_model_comparison_section_returns_embedded_png_markup(self) -> None:
        aggregate = pd.DataFrame(
            {
                "model_name": ["SeasonalNaive", "MFLES"],
                "overall_mae": [1.0, 2.0],
                "overall_rmse": [1.5, 2.5],
                "mean_series_rmae": [0.1, 0.2],
                "average_rank": [1.0, 2.0],
                "win_count": [2, 0],
                "train_time": [0.1, 0.2],
            }
        )
        combined = pd.DataFrame(
            {
                "y_true": [1.0, 2.0],
                "SeasonalNaive": [1.1, 2.1],
                "MFLES": [0.9, 1.9],
                "TimeBase": [1.0, 2.0],
                "TimeBaseTrend": [1.2, 2.2],
                "NLinear": [1.1, 2.0],
                "DLinear": [0.8, 2.4],
            }
        )
        section = build_model_comparison_section(aggregate, combined)
        assert "data:image/png;base64," in section
        assert "Model comparison details" in section

    def test_build_neural_diagnostics_section_returns_embedded_png_markup(self) -> None:
        training_curves = pd.DataFrame(
            {
                "model_name": ["DLinear", "DLinear", "NLinear", "NLinear"],
                "step": [1, 1, 1, 1],
                "split": ["Train", "Validation", "Train", "Validation"],
                "loss": [0.5, 0.7, 0.4, 0.6],
            }
        )
        section = build_neural_diagnostics_section(training_curves)
        assert "data:image/png;base64," in section
        assert "Neural model diagnostics" in section
        assert "generalization gap" in section.lower()

    def test_build_representative_plot_sections_use_embedded_png_markup(self) -> None:
        full_frame = pd.DataFrame(
            {
                "unique_id": ["a"] * 6,
                "ds": pd.date_range("2024-01-01", periods=6, freq="MS"),
                "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        )
        target_frame = pd.DataFrame(
            {
                "unique_id": ["a"] * 2,
                "ds": pd.date_range("2024-05-01", periods=2, freq="MS"),
                "y_true": [5.0, 6.0],
            }
        )
        forecast_frames = {
            "SeasonalNaive": pd.DataFrame(
                {
                    "unique_id": ["a"] * 2,
                    "ds": pd.date_range("2024-05-01", periods=2, freq="MS"),
                    "SeasonalNaive": [4.5, 5.5],
                }
            ),
            "TimeBase": pd.DataFrame(
                {
                    "unique_id": ["a"] * 2,
                    "ds": pd.date_range("2024-05-01", periods=2, freq="MS"),
                    "TimeBase": [5.1, 6.1],
                }
            ),
        }
        sections = build_representative_plot_sections(
            full_frame, target_frame, forecast_frames, ["a"], history_points=4
        )
        assert len(sections) == 1
        assert "data:image/png;base64," in sections[0]
        assert "Forecast plot for a" in sections[0]

    def test_build_html_report_includes_tabbed_sections(self) -> None:
        aggregate = pd.DataFrame(
            {
                "model_name": ["TimeBase", "SeasonalNaive"],
                "overall_mae": [1.2, 1.0],
                "overall_rmse": [1.5, 1.3],
                "mean_series_mae": [1.1, 0.9],
                "median_series_mae": [1.0, 0.8],
                "mean_series_rmae": [0.11, 0.09],
                "median_series_rmae": [0.10, 0.08],
                "average_rank": [2.0, 1.0],
                "params": [100, 0],
                "win_count": [0, 1],
                "train_time": [2.0, 0.0],
                "inference_time": [0.5, 0.0],
            }
        )
        per_series = pd.DataFrame(
            {
                "unique_id": ["a", "a", "b", "b"],
                "model_name": [
                    "TimeBase",
                    "SeasonalNaive",
                    "TimeBase",
                    "SeasonalNaive",
                ],
                "mae": [1.2, 1.0, 1.0, 0.8],
                "rmse": [1.3, 1.1, 1.1, 0.9],
                "rmae": [0.12, 0.10, 0.20, 0.16],
            }
        )
        dataset_summary = {
            "dataset_name": "custom",
            "freq": "MS",
            "horizon": 12,
            "n_series": 2,
            "n_rows": 24,
            "min_length": 12,
            "median_length": 12,
            "max_length": 12,
        }
        html = build_html_report(
            aggregate_results=aggregate,
            per_series_results=per_series,
            dataset_summary=dataset_summary,
            representative_plot_sections=[
                "<figure>Representative series</figure>",
            ],
            mae_distribution_section=(
                "<section>Distribution of relative MAE across series</section>"
            ),
            model_comparison_section="<section>Model comparison details</section>",
            neural_diagnostics_section="<section>Neural model diagnostics</section>",
        )
        assert "tab-button" in html
        assert "General" in html
        assert "Representative series" in html
        assert "Distribution of relative MAE across series" in html
        assert "Neural model diagnostics" in html
        assert "<details>" in html
