"""Reusable Matplotlib HTML reporting helpers for benchmark result tables."""

from __future__ import annotations

import base64
from html import escape
from io import BytesIO
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODEL_COLORS = {
    "SeasonalNaive": "#7dd3fc",
    "AutoMFLES": "#34d399",
    "MFLES": "#34d399",
    "TimeBase": "#f59e0b",
    "TimeBaseTrend": "#f97316",
    "AutoTimeBase": "#f59e0b",
    "AutoTimeBaseTrend": "#f97316",
    "NLinear": "#a78bfa",
    "DLinear": "#f472b6",
    "AutoARIMA": "#cbd5e1",
    "Naive": "#7dd3fc",
    "Observed": "#e2e8f0",
    "Holdout": "#ffffff",
}


def render_matplotlib_figure(fig: plt.Figure, alt_text: str) -> str:
    """Render a Matplotlib figure as an embedded HTML image."""
    buffer = BytesIO()
    fig.tight_layout()
    fig.savefig(
        buffer,
        format="png",
        dpi=160,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f'<img class="plot-image" src="data:image/png;base64,{encoded}" alt="{escape(alt_text)}">'


def build_slice_label(frame: pd.DataFrame, slice_columns: list[str]) -> pd.Series:
    """Build a readable benchmark slice label from one or more columns."""
    return frame[slice_columns].astype(str).agg(" | ".join, axis=1)


def build_best_by_slice_summary(
    results_frame: pd.DataFrame,
    slice_columns: list[str],
    metric_column: str = "mae",
) -> list[dict[str, Any]]:
    """Summarize the best row for each benchmark slice."""
    if results_frame.empty:
        return []
    ordered = results_frame.sort_values(metric_column)
    summary: list[dict[str, Any]] = []
    for _, group in ordered.groupby(slice_columns, dropna=False):
        best_row = group.iloc[0]
        row = {column: best_row[column] for column in slice_columns}
        row.update(
            {
                "best_model": best_row["model_name"],
                f"best_{metric_column}": float(best_row[metric_column]),
            }
        )
        summary.append(row)
    return summary


def format_value(value: Any, column: str) -> str:
    """Format table values consistently for HTML output."""
    if pd.isna(value):
        return "-"
    if column in {"mae", "rmse", "train_time", "inference_time"}:
        return f"{float(value):.4f}"
    if column == "params":
        return f"{int(value)}"
    return escape(str(value))


def build_leaderboard_table(
    results_frame: pd.DataFrame,
    sort_columns: list[str],
) -> str:
    """Render a static HTML leaderboard table."""
    preferred_columns = [
        "model_name",
        "dataset",
        "frequency",
        "scenario",
        "mae",
        "rmse",
        "params",
        "train_time",
        "inference_time",
    ]
    columns = [
        column for column in preferred_columns if column in results_frame.columns
    ]
    rows = []
    for row in results_frame.sort_values(sort_columns).itertuples(index=False):
        values = []
        row_map = row._asdict()
        for column in columns:
            values.append(f"<td>{format_value(row_map[column], column)}</td>")
        rows.append("<tr>" + "".join(values) + "</tr>")
    header = "".join(
        f"<th>{escape(column.replace('_', ' ').title())}</th>" for column in columns
    )
    return (
        '<table class="leaderboard-table">'
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def plot_metric_by_slice(
    results_frame: pd.DataFrame,
    slice_columns: list[str],
    metric_column: str = "mae",
) -> str:
    """Plot the main metric by model for each benchmark slice."""
    frame = results_frame.copy()
    frame["slice_label"] = build_slice_label(frame, slice_columns)
    slices = frame["slice_label"].drop_duplicates().tolist()
    fig, axes = plt.subplots(
        len(slices), 1, figsize=(11, max(3.5, 2.8 * len(slices))), facecolor="#0f172a"
    )
    axes_list = [axes] if len(slices) == 1 else list(axes)
    for ax, slice_label in zip(axes_list, slices, strict=True):
        ax.set_facecolor("#0f172a")
        subset = frame[frame["slice_label"] == slice_label].sort_values(
            metric_column, ascending=True
        )
        ax.barh(
            subset["model_name"],
            subset[metric_column],
            color=[MODEL_COLORS.get(name, "#cbd5e1") for name in subset["model_name"]],
        )
        ax.set_title(
            f"{metric_column.upper()} by model — {slice_label}", color="#e2e8f0"
        )
        ax.set_xlabel(metric_column.upper(), color="#cbd5e1")
        ax.tick_params(colors="#cbd5e1")
        ax.grid(True, axis="x", alpha=0.18, color="#94a3b8")
    return render_matplotlib_figure(fig, f"{metric_column} by slice")


def choose_representative_series(
    frame: pd.DataFrame,
    n_examples: int = 5,
    random_seed: int = 42,
) -> list[str]:
    """Choose representative series by length, variance, trend, and random picks."""
    if frame.empty:
        return []

    stats_rows: list[dict[str, float | int | str]] = []
    for unique_id, series in frame.groupby("unique_id", sort=False):
        ordered = series.sort_values("ds")
        values = ordered["y"].to_numpy(dtype=float)
        if len(values) == 0:
            continue
        x_axis = np.arange(len(values), dtype=float)
        slope = float(np.polyfit(x_axis, values, 1)[0]) if len(values) >= 2 else 0.0
        stats_rows.append(
            {
                "unique_id": str(unique_id),
                "length": len(values),
                "variance": float(np.var(values)),
                "trend_strength": abs(slope),
            }
        )
    if not stats_rows:
        return []

    stats = pd.DataFrame(stats_rows)
    selected: list[str] = []

    def add_from_sorted(column: str) -> None:
        for unique_id in stats.sort_values(
            [column, "unique_id"], ascending=[False, True]
        )["unique_id"]:
            if unique_id not in selected:
                selected.append(str(unique_id))
                return

    add_from_sorted("length")
    add_from_sorted("variance")
    add_from_sorted("trend_strength")

    remaining = [
        unique_id
        for unique_id in stats.sort_values("unique_id")["unique_id"]
        if unique_id not in selected
    ]
    rng = np.random.default_rng(random_seed)
    random_target = min(max(0, n_examples - len(selected)), max(0, len(remaining)))
    if random_target > 0:
        selected.extend(
            rng.choice(remaining, size=random_target, replace=False).tolist()
        )

    for unique_id in remaining:
        if len(selected) >= min(n_examples, len(stats)):
            break
        if unique_id not in selected:
            selected.append(str(unique_id))
    return selected[: min(n_examples, len(stats))]


def build_representative_series_sections(
    full_frame: pd.DataFrame,
    slice_columns: list[str] | None = None,
    n_examples: int = 5,
    history_points: int = 120,
) -> list[str]:
    """Build representative observed-series sections for one or more slices."""
    if full_frame.empty:
        return []

    sections: list[str] = []
    grouped = (
        [((), full_frame)]
        if not slice_columns
        else list(full_frame.groupby(slice_columns, dropna=False, sort=False))
    )
    for slice_key, slice_frame in grouped:
        representative_ids = choose_representative_series(
            slice_frame, n_examples=n_examples
        )
        if not representative_ids:
            continue
        if slice_columns:
            labels = [
                f"{column}={value}"
                for column, value in zip(
                    slice_columns,
                    slice_key if isinstance(slice_key, tuple) else (slice_key,),
                    strict=True,
                )
            ]
            slice_title = " | ".join(labels)
            sections.append(f'<section class="card"><h2>{escape(slice_title)}</h2>')
        for unique_id in representative_ids:
            series_frame = (
                slice_frame[slice_frame["unique_id"] == unique_id]
                .sort_values("ds")
                .tail(history_points)
            )
            fig, ax = plt.subplots(figsize=(10.5, 3.8), facecolor="#0f172a")
            ax.set_facecolor("#0f172a")
            ax.plot(
                series_frame["ds"],
                series_frame["y"],
                color=MODEL_COLORS.get("Observed", "#e2e8f0"),
                linewidth=2.2,
            )
            ax.set_title(f"Representative series: {unique_id}", color="#e2e8f0")
            ax.set_xlabel("Date", color="#cbd5e1")
            ax.set_ylabel("Value", color="#cbd5e1")
            ax.tick_params(colors="#cbd5e1")
            ax.grid(True, alpha=0.18, color="#94a3b8")
            sections.append(
                f"<figure><h3>{escape(str(unique_id))}</h3>{render_matplotlib_figure(fig, f'Representative series {unique_id}')}</figure>"
            )
        if slice_columns:
            sections.append("</section>")
    return sections


def build_representative_forecast_sections(
    full_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    forecast_frames: dict[str, pd.DataFrame],
    slice_columns: list[str] | None = None,
    n_examples: int = 5,
    history_points: int = 120,
) -> list[str]:
    """Build representative sections with observed history, holdout, and forecasts."""
    if full_frame.empty:
        return []

    def filter_slice(frame: pd.DataFrame, slice_key: Any) -> pd.DataFrame:
        if not slice_columns or frame.empty:
            return frame.copy()
        values = slice_key if isinstance(slice_key, tuple) else (slice_key,)
        filtered = frame
        for column, value in zip(slice_columns, values, strict=True):
            filtered = filtered[filtered[column].eq(value)]
        return filtered.copy()

    grouped = (
        [((), full_frame, target_frame, forecast_frames)] if not slice_columns else []
    )
    if slice_columns:
        for slice_key, slice_frame in full_frame.groupby(
            slice_columns, dropna=False, sort=False
        ):
            slice_target = filter_slice(target_frame, slice_key)
            slice_forecasts = {
                model_name: filter_slice(frame, slice_key)
                for model_name, frame in forecast_frames.items()
            }
            grouped.append((slice_key, slice_frame, slice_target, slice_forecasts))

    sections: list[str] = []
    plot_order = ["Observed", "Holdout", *forecast_frames.keys()]
    for slice_key, slice_frame, slice_target, slice_forecasts in grouped:
        representative_ids = choose_representative_series(slice_frame, n_examples)
        if not representative_ids:
            continue
        if slice_columns:
            labels = [
                f"{column}={value}"
                for column, value in zip(
                    slice_columns,
                    slice_key if isinstance(slice_key, tuple) else (slice_key,),
                    strict=True,
                )
            ]
            sections.append(
                f'<section class="card"><h2>{escape(" | ".join(labels))}</h2>'
            )
        for unique_id in representative_ids:
            history = (
                slice_frame[slice_frame["unique_id"] == unique_id]
                .sort_values("ds")
                .tail(history_points)[["ds", "y"]]
                .assign(series="Observed", value=lambda df: df["y"])[
                    ["ds", "series", "value"]
                ]
            )
            holdout = (
                slice_target[slice_target["unique_id"] == unique_id]
                .sort_values("ds")[["ds", "y_true"]]
                .assign(series="Holdout", value=lambda df: df["y_true"])[
                    ["ds", "series", "value"]
                ]
            )
            forecast_parts = [history, holdout]
            for model_name, forecast_frame in slice_forecasts.items():
                if model_name not in forecast_frame.columns:
                    continue
                series_forecast = (
                    forecast_frame[forecast_frame["unique_id"] == unique_id]
                    .sort_values("ds")[["ds", model_name]]
                    .assign(series=model_name, value=lambda df, c=model_name: df[c])[
                        ["ds", "series", "value"]
                    ]
                )
                forecast_parts.append(series_forecast)
            plot_frame = pd.concat(forecast_parts, ignore_index=True)
            fig, ax = plt.subplots(figsize=(10.5, 3.8), facecolor="#0f172a")
            ax.set_facecolor("#0f172a")
            for series_name in plot_order:
                series_data = plot_frame[plot_frame["series"] == series_name]
                if series_data.empty:
                    continue
                ax.plot(
                    series_data["ds"],
                    series_data["value"],
                    label=series_name,
                    color=MODEL_COLORS.get(series_name, "#cbd5e1"),
                    linestyle="--" if series_name == "Holdout" else "-",
                    linewidth=2.2 if series_name in {"Observed", "Holdout"} else 1.8,
                )
            ax.set_title(f"Forecast plot for {unique_id}", color="#e2e8f0")
            ax.set_xlabel("Date", color="#cbd5e1")
            ax.set_ylabel("Value", color="#cbd5e1")
            ax.tick_params(colors="#cbd5e1")
            ax.grid(True, alpha=0.18, color="#94a3b8")
            legend = ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper left")
            for text in legend.get_texts():
                text.set_color("#e2e8f0")
            sections.append(
                f"<figure><h3>{escape(str(unique_id))}</h3>{render_matplotlib_figure(fig, f'Forecast plot for {unique_id}')}</figure>"
            )
        if slice_columns:
            sections.append("</section>")
    return sections


def plot_runtime_tradeoff(
    results_frame: pd.DataFrame, metric_column: str = "mae"
) -> str | None:
    """Plot train-time versus error when runtime data is available."""
    if (
        "train_time" not in results_frame.columns
        or results_frame["train_time"].fillna(0).eq(0).all()
    ):
        return None
    frame = results_frame.dropna(subset=[metric_column, "train_time"]).copy()
    fig, ax = plt.subplots(figsize=(8.5, 4.5), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")
    for row in frame.itertuples(index=False):
        color = MODEL_COLORS.get(row.model_name, "#cbd5e1")
        ax.scatter(
            row.train_time, getattr(row, metric_column), color=color, s=110, alpha=0.85
        )
        label_parts = [str(row.model_name)]
        for column in ["dataset", "frequency", "scenario"]:
            if hasattr(row, column):
                label_parts.append(str(getattr(row, column)))
        ax.annotate(
            " | ".join(label_parts),
            (row.train_time, getattr(row, metric_column)),
            color="#e2e8f0",
            fontsize=7,
        )
    ax.set_title(f"Training time vs {metric_column.upper()}", color="#e2e8f0")
    ax.set_xlabel("Train time (s)", color="#cbd5e1")
    ax.set_ylabel(metric_column.upper(), color="#cbd5e1")
    ax.tick_params(colors="#cbd5e1")
    ax.grid(True, alpha=0.18, color="#94a3b8")
    return render_matplotlib_figure(fig, f"Training time versus {metric_column}")


def build_html_benchmark_report(
    results_frame: pd.DataFrame,
    title: str,
    source_label: str,
    slice_columns: list[str],
    description: str,
    metric_column: str = "mae",
    representative_sections: list[str] | None = None,
) -> str:
    """Render a reusable benchmark HTML report from a result table."""
    sort_columns = [
        column
        for column in [metric_column, "rmse", "train_time"]
        if column in results_frame.columns
    ]
    summary = build_best_by_slice_summary(
        results_frame, slice_columns, metric_column=metric_column
    )
    summary_cards = []
    if not results_frame.empty:
        best_row = results_frame.sort_values(sort_columns).iloc[0]
        summary_cards.extend(
            [
                f'<div class="kpi"><strong>Best model:</strong> {escape(str(best_row["model_name"]))}</div>',
                f'<div class="kpi"><strong>Best {metric_column.upper()}:</strong> {float(best_row[metric_column]):.4f}</div>',
                f'<div class="kpi"><strong>Rows:</strong> {len(results_frame)}</div>',
                f'<div class="kpi"><strong>Slices:</strong> {len(summary)}</div>',
            ]
        )
    summary_table = pd.DataFrame(summary) if summary else pd.DataFrame()
    leaderboard_html = (
        build_leaderboard_table(results_frame, sort_columns)
        if not results_frame.empty
        else "<p>No results available.</p>"
    )
    slice_plot_html = (
        plot_metric_by_slice(results_frame, slice_columns, metric_column=metric_column)
        if not results_frame.empty
        else "<p>No plot available.</p>"
    )
    runtime_plot_html = plot_runtime_tradeoff(
        results_frame, metric_column=metric_column
    )
    summary_table_html = (
        dataframe_to_html(summary_table)
        if not summary_table.empty
        else "<p>No summary available.</p>"
    )
    runtime_section = (
        ""
        if runtime_plot_html is None
        else f'<section class="card"><h2>Runtime trade-off</h2>{runtime_plot_html}</section>'
    )
    representative_html = (
        "<p>No representative series available.</p>"
        if not representative_sections
        else "".join(representative_sections)
    )
    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>{escape(title)}</title>
  <style>
    :root {{ --bg: #07111f; --bg-soft: #0f172a; --panel: rgba(15, 23, 42, 0.92); --panel-soft: rgba(15, 23, 42, 0.72); --text: #e2e8f0; --muted: #94a3b8; --accent: #38bdf8; --accent-soft: rgba(56, 189, 248, 0.14); --border: rgba(148, 163, 184, 0.22); }}
    body {{ margin: 0; font-family: Roboto, Arial, sans-serif; background: linear-gradient(180deg, var(--bg) 0%, #111827 100%); color: var(--text); }}
    main {{ max-width: 1480px; margin: 0 auto; padding: 32px 24px 72px; }}
    .card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 18px; padding: 22px; margin: 24px 0; overflow-x: auto; }}
    .hero {{ padding: 28px; background: linear-gradient(135deg, rgba(56, 189, 248, 0.12), rgba(14, 165, 233, 0.02)); }}
    .kpi {{ display: inline-block; margin-right: 12px; margin-top: 10px; padding: 12px 16px; border-radius: 14px; background: var(--accent-soft); border: 1px solid rgba(56, 189, 248, 0.24); }}
    .plot-image {{ width: 100%; height: auto; display: block; border-radius: 12px; }}
    .leaderboard-table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    .leaderboard-table th, .leaderboard-table td {{ padding: 10px 12px; border-bottom: 1px solid rgba(148, 163, 184, 0.16); text-align: left; white-space: nowrap; }}
    .leaderboard-table th {{ color: #bae6fd; background: rgba(15, 23, 42, 0.98); }}
    .tab-buttons {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 24px 0 16px; }}
    .tab-button {{ border: 1px solid var(--border); background: var(--panel-soft); color: var(--text); border-radius: 999px; padding: 10px 14px; cursor: pointer; }}
    .tab-button.active {{ background: rgba(56, 189, 248, 0.16); border-color: rgba(56, 189, 248, 0.32); }}
    .tab-panel {{ display: none; }}
    .tab-panel.active {{ display: block; }}
    h1, h2, h3 {{ margin: 0 0 12px; }}
    figure {{ margin: 0 0 22px; }}
    p {{ color: var(--muted); line-height: 1.5; }}
  </style>
</head>
<body>
  <main>
    <section class=\"card hero\">
      <h1>{escape(title)}</h1>
      <p>{escape(description)}</p>
      <p><strong>Source:</strong> {escape(source_label)}</p>
      {"".join(summary_cards)}
    </section>
    <div class=\"tab-buttons\">
      <button class=\"tab-button active\" data-tab=\"general\">General</button>
      <button class=\"tab-button\" data-tab=\"representative\">Representative series</button>
      <button class=\"tab-button\" data-tab=\"comparisons\">Slice comparisons</button>
    </div>
    <section id=\"tab-general\" class=\"tab-panel active\">
      <section class=\"card\"><h2>Leaderboard</h2>{leaderboard_html}</section>
      <section class=\"card\"><h2>Best by slice</h2>{summary_table_html}</section>
    </section>
    <section id=\"tab-representative\" class=\"tab-panel\">
      <section class=\"card\"><h2>Representative series</h2><p>Each slice shows the longest history, the highest-variance series, the strongest-trend series, and up to two additional random series. Forecast reports also overlay model predictions and the holdout trajectory.</p>{representative_html}</section>
    </section>
    <section id=\"tab-comparisons\" class=\"tab-panel\">
      <section class=\"card\"><h2>{metric_column.upper()} by slice</h2>{slice_plot_html}</section>
      {runtime_section}
    </section>
  </main>
  <script>
    const buttons = Array.from(document.querySelectorAll('.tab-button'));
    const panels = Array.from(document.querySelectorAll('.tab-panel'));
    buttons.forEach((button) => {{
      button.addEventListener('click', () => {{
        const tab = button.dataset.tab;
        buttons.forEach((item) => item.classList.toggle('active', item === button));
        panels.forEach((panel) => panel.classList.toggle('active', panel.id === `tab-${{tab}}`));
      }});
    }});
  </script>
</body>
</html>
"""


def dataframe_to_html(frame: pd.DataFrame) -> str:
    """Convert a frame to a compact HTML table."""
    if frame.empty:
        return "<p>No data available.</p>"
    columns = list(frame.columns)
    header = "".join(
        f"<th>{escape(str(column).replace('_', ' ').title())}</th>"
        for column in columns
    )
    rows = []
    for _, row in frame.iterrows():
        cells = "".join(f"<td>{escape(str(row[column]))}</td>" for column in columns)
        rows.append(f"<tr>{cells}</tr>")
    return f'<table class="leaderboard-table"><thead><tr>{header}</tr></thead><tbody>{"".join(rows)}</tbody></table>'
