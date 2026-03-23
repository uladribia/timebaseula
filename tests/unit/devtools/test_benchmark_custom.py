"""Tests for the simplified custom benchmark CLI."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
from typer.testing import CliRunner

from devtools import benchmark_custom
from devtools.benchmark_custom import load_custom_dataset, validate_series_lengths


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

    def test_main_writes_csv_and_markdown(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The CLI should persist the simplified leaderboard and markdown report."""
        runner = CliRunner()
        benchmark_mock = Mock(
            return_value=pd.DataFrame(
                {
                    "model_name": ["TimeBase"],
                    "mae": [0.1],
                    "rmse": [0.2],
                    "params": [31],
                    "train_time": [1.0],
                }
            )
        )
        monkeypatch.setattr(
            benchmark_custom,
            "benchmark_custom_dataset",
            benchmark_mock,
        )

        output_dir = tmp_path / "custom"
        result = runner.invoke(
            benchmark_custom.app,
            [
                "--output-dir",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0
        assert (output_dir / "leaderboard.csv").exists()
        assert (output_dir / "report.md").exists()
        benchmark_mock.assert_called_once()
