"""Tests for the simplified long-horizon benchmark CLI."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
from typer.testing import CliRunner

from devtools import benchmark_long_horizon
from devtools.benchmark_common import BenchmarkArtifacts, SavedPlot
from devtools.benchmark_long_horizon import (
    _build_neural_models,
    aggregate_frame,
    ensure_aggregated_datasets,
    get_aggregated_dataset_path,
    resolve_auto_preset,
    resolve_dataset_group,
    resolve_mode_defaults,
)
from timebaseula import AutoTimeBase, AutoTimeBaseTrend


class TestBenchmarkLongHorizon:
    """Validate the long-horizon benchmark helpers."""

    def test_resolve_dataset_group_aliases(self) -> None:
        """Dataset aliases should map to datasetsforecast group names."""
        assert resolve_dataset_group("ecl") == "ECL"
        assert resolve_dataset_group("TrafficL") == "TrafficL"
        assert resolve_dataset_group("traffic") == "TrafficL"

    def test_get_aggregated_dataset_path(self) -> None:
        """Aggregated dataset files should live under the datasets directory."""
        assert get_aggregated_dataset_path(Path("datasets"), "ECL", "D") == Path(
            "datasets/ecl_daily.parquet"
        )
        assert get_aggregated_dataset_path(
            Path("datasets"),
            "TrafficL",
            "ME",
        ) == Path("datasets/trafficl_monthly.parquet")

    def test_resolve_mode_defaults(self) -> None:
        """Mode defaults should provide sensible daily and monthly settings."""
        assert resolve_mode_defaults("daily") == {
            "freq": "D",
            "horizon": 14,
            "max_steps": 50,
            "report_name": "daily",
        }
        assert resolve_mode_defaults("monthly") == {
            "freq": "ME",
            "horizon": 5,
            "max_steps": 30,
            "report_name": "monthly",
        }

    def test_aggregate_frame_daily(self) -> None:
        """Daily aggregation should preserve unique_id and mean by day."""
        frame = pd.DataFrame(
            {
                "unique_id": ["a", "a", "a", "a"],
                "ds": pd.to_datetime(
                    [
                        "2024-01-01 00:00:00",
                        "2024-01-01 12:00:00",
                        "2024-01-02 00:00:00",
                        "2024-01-02 12:00:00",
                    ]
                ),
                "y": [1.0, 3.0, 2.0, 6.0],
            }
        )

        result = aggregate_frame(frame, "D")

        assert list(result.columns) == ["unique_id", "ds", "y"]
        assert len(result) == 2
        assert result["y"].tolist() == [2.0, 4.0]

    def test_aggregate_frame_monthly(self) -> None:
        """Monthly aggregation should use month-end buckets."""
        frame = pd.DataFrame(
            {
                "unique_id": ["a", "a", "a"],
                "ds": pd.to_datetime(["2024-01-01", "2024-01-15", "2024-02-01"]),
                "y": [1.0, 3.0, 10.0],
            }
        )

        result = aggregate_frame(frame, "ME")

        assert len(result) == 2
        assert result["y"].tolist() == [2.0, 10.0]

    def test_ensure_aggregated_datasets_downloads_each_raw_dataset_once(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Dataset preparation should reuse one raw download per dataset."""
        download_calls: list[str] = []

        def fake_download(dataset: str) -> pd.DataFrame:
            download_calls.append(dataset)
            return pd.DataFrame(
                {
                    "unique_id": ["a", "a"],
                    "ds": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                    "y": [1.0, 2.0],
                }
            )

        monkeypatch.setattr(benchmark_long_horizon, "DATASETS_DIR", tmp_path)
        monkeypatch.setattr(
            benchmark_long_horizon,
            "download_raw_dataset",
            fake_download,
        )

        generated = ensure_aggregated_datasets(force_download=False)

        assert len(generated) == 4
        assert download_calls.count("ECL") == 1
        assert download_calls.count("TrafficL") == 1

    def test_resolve_auto_preset_returns_expected_values(self) -> None:
        """Preset names should map to benchmark-friendly auto-search settings."""
        assert resolve_auto_preset("smoke") == {"max_steps": 1, "auto_num_samples": 1}
        assert resolve_auto_preset("normal") == {"max_steps": 10, "auto_num_samples": 2}
        assert resolve_auto_preset("thorough") == {
            "max_steps": 20,
            "auto_num_samples": 4,
        }

    def test_build_neural_models_uses_auto_wrappers(self) -> None:
        """The benchmark should use searched auto wrappers instead of raw models."""
        models = _build_neural_models(
            horizon=2,
            max_steps=1,
            freq="D",
            auto_num_samples=3,
        )

        assert any(isinstance(model, AutoTimeBase) for model in models)
        assert any(isinstance(model, AutoTimeBaseTrend) for model in models)
        assert not any(type(model).__name__ == "TimeBase" for model in models)
        assert not any(type(model).__name__ == "TimeBaseTrend" for model in models)

        auto_timebase = next(
            model for model in models if isinstance(model, AutoTimeBase)
        )
        auto_timebase_trend = next(
            model for model in models if isinstance(model, AutoTimeBaseTrend)
        )
        assert auto_timebase.num_samples == 3
        assert auto_timebase_trend.num_samples == 3
        assert auto_timebase.config["max_steps"] == 1
        assert auto_timebase.config["accelerator"] == "cpu"
        assert auto_timebase.config["devices"] == 1
        assert not isinstance(auto_timebase.config["input_size"], int)
        assert not isinstance(auto_timebase.config["period_len"], int)
        assert not isinstance(auto_timebase_trend.config["moving_avg_window"], int)

    def test_run_command_uses_normal_auto_preset_by_default(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The CLI should default to the normal auto-search preset."""
        runner = CliRunner()
        benchmark_mock = Mock(
            return_value=BenchmarkArtifacts(
                results_frame=pd.DataFrame(
                    {
                        "dataset": ["ECL"],
                        "frequency": ["D"],
                        "model_name": ["AutoTimeBase"],
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
        monkeypatch.setattr(
            benchmark_long_horizon,
            "ensure_aggregated_datasets",
            Mock(return_value=[]),
        )
        monkeypatch.setattr(
            benchmark_long_horizon,
            "run_benchmark_block",
            benchmark_mock,
        )
        monkeypatch.setattr(
            benchmark_long_horizon,
            "save_representative_forecast_plots",
            lambda *args, **kwargs: [],
        )

        result = runner.invoke(benchmark_long_horizon.app, ["run", "--quiet"])

        assert result.exit_code == 0
        assert benchmark_mock.call_args.kwargs["max_steps"] == 10
        assert benchmark_mock.call_args.kwargs["auto_num_samples"] == 2

    def test_run_command_writes_csv_markdown_and_plots(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The CLI should persist CSV, markdown, and representative plots."""
        runner = CliRunner()
        benchmark_mock = Mock(
            return_value=BenchmarkArtifacts(
                results_frame=pd.DataFrame(
                    {
                        "dataset": ["ECL", "ECL"],
                        "frequency": ["D", "D"],
                        "model_name": ["AutoTimeBase", "SeasonalNaive"],
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
            benchmark_long_horizon,
            "ensure_aggregated_datasets",
            Mock(return_value=[]),
        )
        monkeypatch.setattr(
            benchmark_long_horizon,
            "run_benchmark_block",
            benchmark_mock,
        )

        def fake_save_plots(*args: object, **kwargs: object) -> list[SavedPlot]:
            del args, kwargs
            plot_path = tmp_path / "plots" / "series-a.png"
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            plot_path.write_text("placeholder", encoding="utf-8")
            return [SavedPlot(title="Series a", path=plot_path)]

        monkeypatch.setattr(
            benchmark_long_horizon,
            "save_representative_forecast_plots",
            fake_save_plots,
        )

        def fake_save_pdf(markdown_text: str, output_pdf: Path, base_dir: Path) -> None:
            del markdown_text, base_dir
            output_pdf.write_bytes(b"%PDF-1.4\n")

        monkeypatch.setattr(benchmark_long_horizon, "save_markdown_pdf", fake_save_pdf)

        output_csv = tmp_path / "benchmark.csv"
        output_md = tmp_path / "benchmark.md"
        output_pdf = tmp_path / "benchmark.pdf"
        result = runner.invoke(
            benchmark_long_horizon.app,
            [
                "run",
                "--dataset",
                "ECL",
                "--mode",
                "daily",
                "--output",
                str(output_csv),
                "--output-md",
                str(output_md),
                "--output-pdf",
                str(output_pdf),
                "--auto-preset",
                "thorough",
                "--auto-num-samples",
                "2",
            ],
        )

        assert result.exit_code == 0
        assert output_csv.exists()
        assert output_md.exists()
        assert output_pdf.exists()
        assert "Representative forecast plots" in output_md.read_text(encoding="utf-8")
        benchmark_mock.assert_called_once()
        assert benchmark_mock.call_args.kwargs["max_steps"] == 20
        assert benchmark_mock.call_args.kwargs["auto_num_samples"] == 2

    def test_run_command_rejects_removed_refit_option(self) -> None:
        """The CLI should not expose a refit toggle anymore."""
        runner = CliRunner()

        result = runner.invoke(benchmark_long_horizon.app, ["run", "--no-refit"])

        assert result.exit_code == 2
        assert "No such option" in result.output
