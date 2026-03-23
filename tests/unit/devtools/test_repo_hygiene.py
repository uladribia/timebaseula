"""Tests for repository hygiene conventions."""

from __future__ import annotations

from pathlib import Path


class TestRepoHygiene:
    """Validate repository cleanup conventions."""

    def test_gitignore_keeps_generated_data_out_of_git(self) -> None:
        """Generated datasets and logs should remain ignored."""
        gitignore = Path(".gitignore").read_text(encoding="utf-8")

        assert "datasets/" in gitignore
        assert "logs/" in gitignore
        assert "/site" in gitignore

    def test_gitignore_does_not_ignore_tracked_benchmark_doc_stub(self) -> None:
        """The tracked benchmark docs placeholder should not be ignored."""
        gitignore = Path(".gitignore").read_text(encoding="utf-8")

        assert "/docs/benchmark.md" not in gitignore
