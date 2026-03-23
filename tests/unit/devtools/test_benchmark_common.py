"""Tests for shared benchmark helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from devtools.benchmark_common import (
    build_markdown_report,
    evaluate_cv_results,
    render_markdown_html,
    save_markdown_pdf,
    save_representative_forecast_plots,
    select_representative_series_ids,
)


class TestBenchmarkCommon:
    """Validate shared benchmark utilities."""

    def test_evaluate_cv_results_uses_native_utilsforecast_metrics(self) -> None:
        """Cross-validation summaries should expose MAE, RMSE, and RMAE."""
        frame = pd.DataFrame(
            {
                "unique_id": ["a", "a", "a", "a"],
                "ds": [1, 2, 3, 4],
                "cutoff": [0, 0, 1, 1],
                "y": [1.0, 2.0, 3.0, 4.0],
                "SeasonalNaive": [1.0, 1.0, 3.0, 3.0],
                "ModelA": [1.0, 2.0, 3.0, 4.0],
                "ModelB": [0.0, 2.0, 2.0, 5.0],
            }
        )

        result = evaluate_cv_results(
            frame,
            ["SeasonalNaive", "ModelA", "ModelB"],
            baseline_model="SeasonalNaive",
        )

        assert list(result.columns) == ["model_name", "mae", "rmse", "rmae"]
        assert result.loc[result["model_name"] == "ModelA", "mae"].item() == 0.0
        assert result.loc[result["model_name"] == "ModelA", "rmae"].item() == 0.0
        assert result.loc[result["model_name"] == "SeasonalNaive", "rmae"].item() == 1.0
        assert result.loc[result["model_name"] == "ModelB", "mae"].item() > 0.0

    def test_build_markdown_report_includes_extra_sections(self) -> None:
        """Markdown reports should include the source, summary, and extra sections."""
        frame = pd.DataFrame(
            {
                "dataset": ["ECL", "ECL", "TrafficL"],
                "frequency": ["D", "D", "ME"],
                "model_name": ["A", "B", "C"],
                "mae": [0.4, 0.2, 0.1],
                "rmse": [0.5, 0.3, 0.2],
                "rmae": [1.2, 0.8, 0.6],
                "execution_time": [1.0, 2.0, 3.0],
            }
        )

        report = build_markdown_report(
            title="Benchmark report",
            source_label="logs/example.csv",
            results_frame=frame,
            slice_columns=["dataset", "frequency"],
            extra_sections=[
                ("Metrics", "RMAE is relative to SeasonalNaive."),
                (
                    "Representative forecast plots",
                    "![Example forecast](plots/example.png)",
                ),
            ],
        )

        assert "# Benchmark report" in report
        assert "logs/example.csv" in report
        assert "Best by slice" in report
        assert "TrafficL" in report
        assert "Representative forecast plots" in report
        assert "plots/example.png" in report

    def test_select_representative_series_ids_prefers_extreme_series(self) -> None:
        """Series selection should cover distinct profile extremes."""
        frame = pd.concat(
            [
                pd.DataFrame(
                    {
                        "unique_id": ["var_high"] * 4,
                        "ds": pd.date_range("2024-01-01", periods=4, freq="D"),
                        "y": [0.0, 10.0, -10.0, 10.0],
                    }
                ),
                pd.DataFrame(
                    {
                        "unique_id": ["var_low"] * 4,
                        "ds": pd.date_range("2024-01-01", periods=4, freq="D"),
                        "y": [1.0, 1.0, 1.0, 1.0],
                    }
                ),
                pd.DataFrame(
                    {
                        "unique_id": ["long"] * 6,
                        "ds": pd.date_range("2024-01-01", periods=6, freq="D"),
                        "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    }
                ),
                pd.DataFrame(
                    {
                        "unique_id": ["short"] * 2,
                        "ds": pd.date_range("2024-01-01", periods=2, freq="D"),
                        "y": [3.0, 4.0],
                    }
                ),
                pd.DataFrame(
                    {
                        "unique_id": ["high_peak"] * 4,
                        "ds": pd.date_range("2024-01-01", periods=4, freq="D"),
                        "y": [1.0, 2.0, 30.0, 2.0],
                    }
                ),
                pd.DataFrame(
                    {
                        "unique_id": ["low_trough"] * 4,
                        "ds": pd.date_range("2024-01-01", periods=4, freq="D"),
                        "y": [-25.0, -2.0, -1.0, -2.0],
                    }
                ),
            ],
            ignore_index=True,
        )

        selected = select_representative_series_ids(frame, limit=5)

        assert len(selected) == 5
        assert {"var_high", "long", "short", "high_peak"}.issubset(set(selected))
        assert set(selected) & {"var_low", "low_trough"}

    def test_save_representative_forecast_plots_writes_pngs(
        self,
        tmp_path: Path,
    ) -> None:
        """Representative forecast plots should be persisted as PNG files."""
        frame = pd.DataFrame(
            {
                "unique_id": ["a"] * 4 + ["b"] * 4,
                "ds": list(pd.date_range("2024-01-01", periods=4, freq="D")) * 2,
                "y": [1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0],
            }
        )
        results_frame = pd.DataFrame(
            {
                "model_name": ["SeasonalNaive", "TimeBase"],
                "mae": [1.0, 0.5],
                "rmse": [1.0, 0.5],
                "rmae": [1.0, 0.5],
                "params": [0, 25],
                "execution_time": [0.0, 0.2],
            }
        )
        forecast_frames = {
            "SeasonalNaive": pd.DataFrame(
                {
                    "unique_id": ["a", "a", "b", "b"],
                    "ds": pd.to_datetime(
                        [
                            "2024-01-03",
                            "2024-01-04",
                            "2024-01-03",
                            "2024-01-04",
                        ]
                    ),
                    "y": [3.0, 4.0, 2.0, 1.0],
                    "SeasonalNaive": [2.0, 2.0, 3.0, 3.0],
                }
            ),
            "TimeBase": pd.DataFrame(
                {
                    "unique_id": ["a", "a", "b", "b"],
                    "ds": pd.to_datetime(
                        [
                            "2024-01-03",
                            "2024-01-04",
                            "2024-01-03",
                            "2024-01-04",
                        ]
                    ),
                    "y": [3.0, 4.0, 2.0, 1.0],
                    "TimeBase": [2.8, 3.7, 2.3, 1.4],
                }
            ),
        }

        saved_plots = save_representative_forecast_plots(
            frame=frame,
            forecast_frames=forecast_frames,
            results_frame=results_frame,
            output_dir=tmp_path,
            title_prefix="Example",
            limit=2,
        )

        assert len(saved_plots) == 2
        for saved_plot in saved_plots:
            assert saved_plot.path.exists()
            assert saved_plot.path.suffix == ".png"

    def test_render_markdown_html_converts_tables_and_embeds_images(
        self,
        tmp_path: Path,
    ) -> None:
        """Markdown HTML rendering should preserve tables and embed local images."""
        image_path = tmp_path / "plot.png"
        figure, axis = plt.subplots()
        axis.plot([0, 1], [0, 1])
        figure.savefig(image_path)
        plt.close(figure)

        markdown_text = (
            "# Report\n\n"
            "| model | mae |\n"
            "| --- | --- |\n"
            "| A | 0.1 |\n\n"
            "![Plot](plot.png)\n"
        )
        html_text = render_markdown_html(markdown_text=markdown_text, base_dir=tmp_path)

        assert "<h1>Report</h1>" in html_text
        assert "<table>" in html_text
        assert "data:image/png;base64," in html_text

    def test_save_markdown_pdf_writes_pdf(self, tmp_path: Path) -> None:
        """Markdown reports should be exportable as PDF files."""
        image_path = tmp_path / "plot.png"
        figure, axis = plt.subplots()
        axis.plot([0, 1], [0, 1])
        figure.savefig(image_path)
        plt.close(figure)
        output_pdf = tmp_path / "report.pdf"

        markdown_text = (
            "# Report\n\n## Summary\n\nA short paragraph.\n\n![Plot](plot.png)\n"
        )
        save_markdown_pdf(
            markdown_text=markdown_text,
            output_pdf=output_pdf,
            base_dir=tmp_path,
        )

        assert output_pdf.exists()
        assert output_pdf.read_bytes().startswith(b"%PDF")
