"""Tests for shared benchmark helpers."""

from __future__ import annotations

import pandas as pd

from devtools.benchmark_common import build_markdown_report, evaluate_cv_results


class TestBenchmarkCommon:
    """Validate shared benchmark utilities."""

    def test_evaluate_cv_results_uses_native_utilsforecast_metrics(self) -> None:
        """Cross-validation summaries should expose MAE and RMSE per model."""
        frame = pd.DataFrame(
            {
                "unique_id": ["a", "a", "a", "a"],
                "ds": [1, 2, 3, 4],
                "cutoff": [0, 0, 1, 1],
                "y": [1.0, 2.0, 3.0, 4.0],
                "ModelA": [1.0, 2.0, 3.0, 4.0],
                "ModelB": [0.0, 2.0, 2.0, 5.0],
            }
        )

        result = evaluate_cv_results(frame, ["ModelA", "ModelB"])

        assert list(result.columns) == ["model_name", "mae", "rmse"]
        assert result.loc[result["model_name"] == "ModelA", "mae"].item() == 0.0
        assert result.loc[result["model_name"] == "ModelB", "mae"].item() > 0.0

    def test_build_markdown_report_summarizes_best_by_slice(self) -> None:
        """Markdown reports should include the source and best-by-slice section."""
        frame = pd.DataFrame(
            {
                "dataset": ["ECL", "ECL", "TrafficL"],
                "frequency": ["D", "D", "ME"],
                "model_name": ["A", "B", "C"],
                "mae": [0.4, 0.2, 0.1],
                "rmse": [0.5, 0.3, 0.2],
            }
        )

        report = build_markdown_report(
            title="Benchmark report",
            source_label="logs/example.csv",
            results_frame=frame,
            slice_columns=["dataset", "frequency"],
        )

        assert "# Benchmark report" in report
        assert "logs/example.csv" in report
        assert "Best by slice" in report
        assert "TrafficL" in report
