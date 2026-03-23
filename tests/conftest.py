"""Test suite configuration."""

from __future__ import annotations

import warnings

import pytest

warnings.filterwarnings(
    "ignore",
    category=SyntaxWarning,
    module=r"neuralforecast\..*",
)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register custom pytest command-line options."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests marked with 'integration'.",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Skip integration tests unless they were explicitly requested."""
    if config.getoption("--run-integration"):
        return

    skip_integration = pytest.mark.skip(
        reason="integration tests require --run-integration"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)
