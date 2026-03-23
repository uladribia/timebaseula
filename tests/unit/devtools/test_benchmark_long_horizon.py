"""Tests for the simplified long-horizon benchmark CLI."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
from typer.testing import CliRunner

from devtools import benchmark_long_horizon
from devtools.benchmark_long_horizon import (
    aggregate_frame,
    ensure_aggregated_datasets,
    get_aggregated_dataset_path,
    resolve_dataset_group,
    resolve_mode_defaults,
)


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

    def test_run_command_writes_csv_and_markdown(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The CLI should persist the simplified CSV and markdown outputs."""
        runner = CliRunner()
        benchmark_mock = Mock(
            return_value=pd.DataFrame(
                {
                    "dataset": ["ECL"],
                    "frequency": ["D"],
                    "model_name": ["TimeBase"],
                    "mae": [0.1],
                    "rmse": [0.2],
                    "params": [31],
                    "train_time": [1.0],
                }
            )
        )
        monkeypatch.setattr(
            benchmark_long_horizon,
            "ensure_aggregated_datasets",
            Mock(return_value=[]),
        )
        monkeypatch.setattr(
            benchmark_long_horizon,
            "benchmark_dataset",
            benchmark_mock,
        )

        output_csv = tmp_path / "benchmark.csv"
        output_md = tmp_path / "benchmark.md"
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
            ],
        )

        assert result.exit_code == 0
        assert output_csv.exists()
        assert output_md.exists()
        benchmark_mock.assert_called_once()
