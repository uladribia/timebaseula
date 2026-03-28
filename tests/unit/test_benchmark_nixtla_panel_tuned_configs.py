"""Tests for optional tuned benchmark configuration loading."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.benchmark_nixtla_panel import load_tuned_model_configs


def test_load_tuned_model_configs_returns_empty_mapping_for_none() -> None:
    """Missing tuned configs should behave like no extra tuned models."""
    assert load_tuned_model_configs(None) == {}


def test_load_tuned_model_configs_reads_json_mapping(tmp_path: Path) -> None:
    """Tuned model configs should be loaded from a JSON artifact."""
    path = tmp_path / "best_configs.json"
    path.write_text(
        json.dumps({"AutoTimeBase": {"input_size": 84, "basis_num": 8}}),
        encoding="utf-8",
    )

    loaded = load_tuned_model_configs(path)

    assert loaded == {"AutoTimeBase": {"input_size": 84, "basis_num": 8}}
