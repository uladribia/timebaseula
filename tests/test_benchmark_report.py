"""Tests for benchmark reporting helpers."""

from __future__ import annotations

import pandas as pd

from scripts.benchmark_long_horizon import (
    build_benchmark_summary,
    format_markdown_report,
    should_include_arima,
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
