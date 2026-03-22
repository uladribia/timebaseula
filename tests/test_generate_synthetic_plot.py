"""Tests for the synthetic Matplotlib plot script."""

from __future__ import annotations

import pandas as pd
from matplotlib import pyplot as plt

from scripts.generate_synthetic_plot import build_series_layer, render_chart_html


class TestGenerateSyntheticPlot:
    """Validate Matplotlib plotting helpers."""

    def test_build_series_layer_normalizes_plot_schema(self) -> None:
        """Series layers should expose ds, value, and series columns."""
        frame = pd.DataFrame(
            {
                "ds": pd.date_range("2024-01-01", periods=2, freq="D"),
                "y": [1.0, 2.0],
            }
        )

        result = build_series_layer(frame, label="observed", value_column="y")

        assert list(result.columns) == ["ds", "value", "series"]
        assert result["series"].tolist() == ["observed", "observed"]
        assert result["value"].tolist() == [1.0, 2.0]

    def test_render_chart_html_returns_embedded_png_document(self) -> None:
        """Rendered chart HTML should contain an embedded PNG image."""
        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4])

        html = render_chart_html(fig, "Example")

        assert "data:image/png;base64," in html
        assert "<html" in html.lower()
