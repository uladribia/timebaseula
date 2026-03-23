"""Tests for the simplified custom benchmark CLI."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
from neuralforecast.models import DLinear, NLinear
from typer.testing import CliRunner

from devtools import benchmark_custom
from devtools.benchmark_common import BenchmarkArtifacts, SavedPlot
from devtools.benchmark_custom import (
    _build_neural_models,
    load_custom_dataset,
    validate_series_lengths,
)
from timebaseula import TimeBase, TimeBaseTrend


class TestBenchmarkCustom:
    """Validate helpers used by the custom benchmark."""

    def test_load_custom_dataset_keeps_core_columns(self) -> None:
        """The loader should keep the benchmark schema only."""
        frame = load_custom_dataset("datasets/custom/neuralforecast_monthly.csv")

        assert list(frame.columns) == ["unique_id", "ds", "y"]
        assert pd.api.types.is_datetime64_any_dtype(frame["ds"])
        assert frame["unique_id"].nunique() == 280

    def test_validate_series_lengths_rejects_short_series(self) -> None:
        """The custom benchmark should fail fast on very short series."""
        frame = pd.DataFrame(
            {
                "unique_id": ["a"] * 2 + ["b"] * 3,
                "ds": pd.date_range("2024-01-01", periods=5, freq="MS"),
                "y": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )

        with pytest.raises(ValueError, match="too short"):
            validate_series_lengths(frame, horizon=2)

    def test_build_neural_models_uses_default_models(self) -> None:
        """The benchmark should use the default model classes directly."""
        models = _build_neural_models(horizon=2, max_steps=7)

        assert any(isinstance(model, DLinear) for model in models)
        assert any(isinstance(model, NLinear) for model in models)
        assert any(isinstance(model, TimeBase) for model in models)
        assert any(isinstance(model, TimeBaseTrend) for model in models)

        dlinear = next(model for model in models if isinstance(model, DLinear))
        nlinear = next(model for model in models if isinstance(model, NLinear))
        timebase = next(model for model in models if isinstance(model, TimeBase))
        timebase_trend = next(
            model for model in models if isinstance(model, TimeBaseTrend)
        )

        assert dlinear.h == 2
        assert nlinear.h == 2
        assert timebase.h == 2
        assert timebase_trend.h == 2
        assert dlinear.max_steps == 7
        assert nlinear.max_steps == 7
        assert timebase.max_steps == 7
        assert timebase_trend.max_steps == 7

    def test_main_uses_default_max_steps(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The CLI should use the built-in benchmark defaults."""
        runner = CliRunner()
        benchmark_mock = Mock(
            return_value=BenchmarkArtifacts(
                results_frame=pd.DataFrame(
                    {
                        "model_name": ["TimeBase"],
                        "mae": [0.1],
                        "rmse": [0.2],
                        "rmae": [0.5],
                        "params": [31],
                        "execution_time": [1.0],
                    }
                ),
                forecast_frames={},
                source_frame=pd.DataFrame(
                    {
                        "unique_id": ["a"],
                        "ds": pd.to_datetime(["2024-01-01"]),
                        "y": [1.0],
                    }
                ),
            )
        )
        monkeypatch.setattr(benchmark_custom, "run_custom_benchmark", benchmark_mock)
        monkeypatch.setattr(
            benchmark_custom,
            "save_representative_forecast_plots",
            lambda *args, **kwargs: [],
        )

        result = runner.invoke(benchmark_custom.app, ["--quiet"])

        assert result.exit_code == 0
        assert benchmark_mock.call_args.kwargs["max_steps"] == 30

    def test_main_writes_csv_markdown_plots_and_pdf(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The CLI should persist the leaderboard, markdown report, plots, and PDF."""
        runner = CliRunner()
        benchmark_mock = Mock(
            return_value=BenchmarkArtifacts(
                results_frame=pd.DataFrame(
                    {
                        "model_name": ["TimeBase", "SeasonalNaive"],
                        "mae": [0.1, 0.2],
                        "rmse": [0.2, 0.3],
                        "rmae": [0.5, 1.0],
                        "params": [31, 0],
                        "execution_time": [1.0, 0.1],
                    }
                ),
                forecast_frames={},
                source_frame=pd.DataFrame(
                    {
                        "unique_id": ["a"],
                        "ds": pd.to_datetime(["2024-01-01"]),
                        "y": [1.0],
                    }
                ),
            )
        )
        monkeypatch.setattr(
            benchmark_custom,
            "run_custom_benchmark",
            benchmark_mock,
        )

        def fake_save_plots(*args: object, **kwargs: object) -> list[SavedPlot]:
            del args, kwargs
            plot_path = tmp_path / "custom" / "plots" / "series-a.png"
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            plot_path.write_text("placeholder", encoding="utf-8")
            return [SavedPlot(title="Series a", path=plot_path)]

        def fake_save_pdf(markdown_text: str, output_pdf: Path, base_dir: Path) -> None:
            del markdown_text, base_dir
            output_pdf.write_bytes(b"%PDF-1.4\n")

        monkeypatch.setattr(
            benchmark_custom,
            "save_representative_forecast_plots",
            fake_save_plots,
        )
        monkeypatch.setattr(benchmark_custom, "save_markdown_pdf", fake_save_pdf)

        output_dir = tmp_path / "custom"
        output_pdf = output_dir / "report.pdf"
        result = runner.invoke(
            benchmark_custom.app,
            [
                "--output-dir",
                str(output_dir),
                "--output-pdf",
                str(output_pdf),
                "--max-steps",
                "12",
            ],
        )

        assert result.exit_code == 0
        assert (output_dir / "leaderboard.csv").exists()
        assert (output_dir / "report.md").exists()
        assert output_pdf.exists()
        assert "Representative forecast plots" in (output_dir / "report.md").read_text(
            encoding="utf-8"
        )
        benchmark_mock.assert_called_once()
        assert benchmark_mock.call_args.kwargs["max_steps"] == 12

    def test_main_rejects_removed_auto_options(self) -> None:
        """The CLI should not expose auto-search tuning options anymore."""
        runner = CliRunner()

        result = runner.invoke(benchmark_custom.app, ["--auto-preset", "smoke"])

        assert result.exit_code == 2
        assert "No such option" in result.output
