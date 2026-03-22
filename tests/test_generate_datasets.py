"""Tests for the dataset generation CLI."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

from typer.testing import CliRunner

from scripts import generate_datasets


class TestGenerateDatasetsCli:
    """Validate the standardized dataset generation script."""

    def test_main_calls_dataset_builder_and_renders_paths(
        self, monkeypatch: Mock
    ) -> None:
        """The CLI should call the shared dataset preparation helper."""
        runner = CliRunner()
        mocked_paths = [
            Path("datasets/ecl_daily.parquet"),
            Path("datasets/ecl_monthly.parquet"),
        ]
        ensure_mock = Mock(return_value=mocked_paths)
        monkeypatch.setattr(
            generate_datasets, "ensure_aggregated_datasets", ensure_mock
        )

        result = runner.invoke(generate_datasets.app, [])

        assert result.exit_code == 0
        ensure_mock.assert_called_once_with(force_download=False)
        assert "datasets/ecl_daily.parquet" in result.stdout

    def test_force_download_flag_is_forwarded(self, monkeypatch: Mock) -> None:
        """The force-download flag should be passed through to the helper."""
        runner = CliRunner()
        ensure_mock = Mock(return_value=[])
        monkeypatch.setattr(
            generate_datasets, "ensure_aggregated_datasets", ensure_mock
        )

        result = runner.invoke(generate_datasets.app, ["--force-download"])

        assert result.exit_code == 0
        ensure_mock.assert_called_once_with(force_download=True)
