"""Tests for benchmark reporting helpers."""

from __future__ import annotations

import pandas as pd

from scripts.benchmark_long_horizon import (
    build_benchmark_summary,
    format_markdown_report,
    resolve_html_report_output,
    should_include_arima,
)
from scripts.reporting import (
    build_best_by_slice_summary,
    build_html_benchmark_report,
    build_representative_forecast_sections,
)


class TestBenchmarkReporting:
    """Validate benchmark reporting and ARIMA selection helpers."""

    def test_should_include_arima(self) -> None:
        """ARIMA inclusion should invert the skip flag."""
        assert should_include_arima(skip_arima=False) is True
        assert should_include_arima(skip_arima=True) is False

    def test_build_benchmark_summary(self) -> None:
        """Summary should pick the best MAE row per dataset/frequency slice."""
        frame = pd.DataFrame(
            {
                "dataset": ["ECL", "ECL", "TrafficL"],
                "frequency": ["D", "D", "ME"],
                "model_name": ["A", "B", "C"],
                "mae": [0.4, 0.2, 0.1],
                "rmse": [0.5, 0.3, 0.2],
                "params": [10, 20, 0],
                "train_time": [1.0, 2.0, 0.1],
                "inference_time": [0.2, 0.3, 0.01],
            }
        )

        summary = build_benchmark_summary(frame)

        assert summary[0]["dataset"] == "ECL"
        assert summary[0]["frequency"] == "D"
        assert summary[0]["best_model"] == "B"
        assert summary[1]["best_model"] == "C"

    def test_format_markdown_report(self) -> None:
        """Markdown report should include the benchmark table and summary."""
        frame = pd.DataFrame(
            {
                "dataset": ["ECL"],
                "frequency": ["D"],
                "model_name": ["TimeBase"],
                "mae": [0.2],
                "rmse": [0.3],
                "params": [31],
                "train_time": [0.1],
                "inference_time": [0.01],
            }
        )

        report = format_markdown_report(frame, source_csv="logs/example.csv")

        assert "# Benchmark report" in report
        assert "logs/example.csv" in report
        assert "TimeBase" in report
        assert "Best MAE by slice" in report

    def test_build_best_by_slice_summary_supports_custom_slice_columns(self) -> None:
        """HTML summary helper should support non-long-horizon slice columns."""
        frame = pd.DataFrame(
            {
                "scenario": ["easy", "easy", "hard"],
                "model_name": ["A", "B", "C"],
                "mae": [0.4, 0.2, 0.3],
            }
        )

        summary = build_best_by_slice_summary(frame, ["scenario"])

        assert summary[0]["scenario"] == "easy"
        assert summary[0]["best_model"] == "B"
        assert summary[1]["scenario"] == "hard"

    def test_build_html_benchmark_report_renders_embedded_png_html(self) -> None:
        """Reusable HTML report should include plots, source, and tables."""
        frame = pd.DataFrame(
            {
                "dataset": ["ECL", "ECL", "TrafficL"],
                "frequency": ["D", "ME", "D"],
                "model_name": ["TimeBase", "NLinear", "MFLES"],
                "mae": [0.2, 0.3, 0.1],
                "rmse": [0.3, 0.4, 0.2],
                "params": [31, 300, 0],
                "train_time": [0.1, 0.2, 0.0],
                "inference_time": [0.01, 0.02, 0.0],
            }
        )

        report = build_html_benchmark_report(
            frame,
            title="Example benchmark",
            source_label="logs/example.csv",
            slice_columns=["dataset", "frequency"],
            description="Reusable benchmark report.",
            representative_sections=["<figure>Representative series</figure>"],
        )

        assert "Example benchmark" in report
        assert "logs/example.csv" in report
        assert "data:image/png;base64," in report
        assert "Leaderboard" in report
        assert "Best by slice" in report
        assert "Representative series" in report
        assert "tab-button" in report

    def test_resolve_html_report_output_uses_csv_stem_by_default(self) -> None:
        """Automatic HTML output should reuse the CSV stem with .html extension."""
        result = resolve_html_report_output(True, None, "logs/example.csv")

        assert str(result) == "logs/example.html"

    def test_resolve_html_report_output_prefers_explicit_path(self) -> None:
        """Explicit HTML path should override the automatic default."""
        result = resolve_html_report_output(
            True,
            "logs/custom-report.html",
            "logs/example.csv",
        )

        assert str(result) == "logs/custom-report.html"

    def test_representative_forecast_sections_handle_unaligned_indexes(self) -> None:
        """Representative forecast plots should not depend on aligned row indexes."""
        full_frame = pd.DataFrame(
            {
                "unique_id": ["a", "a", "a", "a"],
                "ds": pd.date_range("2024-01-01", periods=4, freq="D"),
                "y": [1.0, 2.0, 3.0, 4.0],
                "scenario": ["easy"] * 4,
            }
        )
        target_frame = pd.DataFrame(
            {
                "unique_id": ["a", "a"],
                "ds": pd.date_range("2024-01-03", periods=2, freq="D"),
                "y_true": [3.0, 4.0],
                "scenario": ["easy", "easy"],
            },
            index=pd.Index([10, 11]),
        )
        forecast_frames = {
            "Naive": pd.DataFrame(
                {
                    "unique_id": ["a", "a"],
                    "ds": pd.date_range("2024-01-03", periods=2, freq="D"),
                    "Naive": [2.5, 2.5],
                    "scenario": ["easy", "easy"],
                }
            )
        }

        sections = build_representative_forecast_sections(
            full_frame,
            target_frame,
            forecast_frames,
            slice_columns=["scenario"],
            n_examples=1,
            history_points=4,
        )

        assert any("Forecast plot for a" in section for section in sections)
