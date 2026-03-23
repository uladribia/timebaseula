"""Shared helpers for internal benchmark CLIs."""

from __future__ import annotations

import logging
import numbers
import os
import re
import textwrap
from dataclasses import dataclass
from functools import partial
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd
import torch
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae, rmae, rmse

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

BASELINE_MODEL_NAME = "SeasonalNaive"
FORECAST_JOIN_KEYS = ["unique_id", "ds", "cutoff", "y"]
REPRESENTATIVE_SERIES_RULES = (
    ("variance", False),
    ("variance", True),
    ("length", False),
    ("length", True),
    ("max_value", False),
    ("min_value", True),
)
MARKDOWN_IMAGE_PATTERN = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<path>[^)]+)\)")
PDF_PAGE_SIZE = (8.27, 11.69)
PDF_TEXT_WIDTH = 92
PDF_LINES_PER_PAGE = 48


@dataclass(frozen=True)
class BenchmarkArtifacts:
    """Bundle the outputs required by benchmark reports and plots."""

    results_frame: pd.DataFrame
    forecast_frames: dict[str, pd.DataFrame]
    source_frame: pd.DataFrame


@dataclass(frozen=True)
class SavedPlot:
    """Describe one plot persisted for a benchmark report."""

    title: str
    path: Path


def configure_logging(logger_name: str, log_path: Path) -> logging.Logger:
    """Configure structured rotating logs for benchmark execution."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(logger_name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=1)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def count_params(model: torch.nn.Module | None) -> int:
    """Count trainable parameters for a PyTorch model."""
    if model is None:
        return 0
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def normalize_forecast_frame(forecast_frame: pd.DataFrame) -> pd.DataFrame:
    """Ensure cross-validation forecasts expose `unique_id` as a column."""
    if "unique_id" in forecast_frame.columns:
        return forecast_frame.reset_index(drop=True)
    return forecast_frame.reset_index(drop=False)


def merge_baseline_forecast(
    forecast_frame: pd.DataFrame,
    baseline_frame: pd.DataFrame,
    baseline_model: str = BASELINE_MODEL_NAME,
) -> pd.DataFrame:
    """Attach the baseline predictions required by utilsforecast RMAE."""
    if baseline_model in forecast_frame.columns:
        return forecast_frame.copy()

    baseline_columns = [*FORECAST_JOIN_KEYS, baseline_model]
    return forecast_frame.merge(
        baseline_frame[baseline_columns],
        on=FORECAST_JOIN_KEYS,
        how="left",
        validate="one_to_one",
    )


def evaluate_cv_results(
    forecast_frame: pd.DataFrame,
    model_names: list[str],
    baseline_model: str = BASELINE_MODEL_NAME,
) -> pd.DataFrame:
    """Evaluate a cross-validation frame with native utilsforecast metrics."""
    metrics = [mae, rmse]
    if baseline_model in model_names:
        metrics.append(partial(rmae, baseline=baseline_model))

    metric_frame = evaluate(
        df=forecast_frame,
        metrics=metrics,
        models=model_names,
        agg_fn="mean",
    )
    rmae_metric_name = f"rmae_{baseline_model}"
    rows: list[dict[str, float | str]] = []
    for model_name in model_names:
        model_scores = (
            metric_frame[["metric", model_name]]
            .groupby("metric", as_index=True)[model_name]
            .mean()
        )
        rows.append(
            {
                "model_name": model_name,
                "mae": float(model_scores.loc["mae"]),
                "rmse": float(model_scores.loc["rmse"]),
                "rmae": (
                    float(model_scores.loc[rmae_metric_name])
                    if rmae_metric_name in model_scores.index
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["mae", "rmse"]).reset_index(drop=True)


def build_best_by_slice_summary(
    results_frame: pd.DataFrame,
    slice_columns: list[str],
) -> pd.DataFrame:
    """Build the best-MAE row for each requested benchmark slice."""
    if results_frame.empty or not slice_columns:
        return pd.DataFrame()

    summary_rows: list[dict[str, Any]] = []
    for _, group in results_frame.sort_values(["mae", "rmse"]).groupby(
        slice_columns,
        dropna=False,
    ):
        best_row = group.iloc[0]
        row = {column: best_row[column] for column in slice_columns}
        row.update(
            {
                "best_model": best_row["model_name"],
                "best_mae": float(best_row["mae"]),
            }
        )
        if "rmse" in best_row.index:
            row["best_rmse"] = float(best_row["rmse"])
        if "rmae" in best_row.index:
            row["best_rmae"] = float(best_row["rmae"])
        if "params" in best_row.index:
            row["best_params"] = int(best_row["params"])
        if "execution_time" in best_row.index:
            row["best_execution_time"] = float(best_row["execution_time"])
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def _format_markdown_value(value: Any) -> str:
    """Format one dataframe cell for markdown output."""
    if pd.isna(value):
        return "-"
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if isinstance(value, numbers.Real):
        return f"{float(value):.4f}"
    return str(value)


def dataframe_to_markdown_table(frame: pd.DataFrame) -> str:
    """Convert a frame to a simple markdown table without optional deps."""
    if frame.empty:
        return "No data available."
    columns = [str(column) for column in frame.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in frame.iterrows():
        values = [_format_markdown_value(row[column]) for column in frame.columns]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def build_markdown_report(
    title: str,
    source_label: str,
    results_frame: pd.DataFrame,
    slice_columns: list[str],
    extra_sections: list[tuple[str, str]] | None = None,
) -> str:
    """Render a benchmark report with results, summaries, and optional extras."""
    lines = [
        f"# {title}",
        "",
        f"Source: `{source_label}`",
        "",
        "## Results",
        "",
        dataframe_to_markdown_table(results_frame),
        "",
    ]
    summary = build_best_by_slice_summary(results_frame, slice_columns)
    if not summary.empty:
        lines.extend(
            [
                "## Best by slice",
                "",
                dataframe_to_markdown_table(summary),
                "",
            ]
        )
    for heading, content in extra_sections or []:
        lines.extend([f"## {heading}", "", content, ""])
    return "\n".join(lines)


def build_metrics_frame() -> pd.DataFrame:
    """Return a compact glossary for the benchmark metrics."""
    return pd.DataFrame(
        {
            "metric": ["mae", "rmse", "rmae", "execution_time"],
            "meaning": [
                "Mean absolute error on the holdout window.",
                "Root mean squared error on the holdout window.",
                "Relative MAE from utilsforecast using SeasonalNaive as baseline.",
                "Wall-clock time for one model cross-validation run.",
            ],
        }
    )


def build_dataset_summary(
    frame: pd.DataFrame,
    label: str,
    horizon: int,
    frequency: str | None = None,
) -> pd.DataFrame:
    """Summarize the dataset slice used by one benchmark run."""
    lengths = frame.groupby("unique_id").size()
    return pd.DataFrame(
        [
            {
                "dataset": label,
                "frequency": "-" if frequency is None else frequency,
                "rows": len(frame),
                "n_series": int(lengths.size),
                "horizon": horizon,
                "min_length": int(lengths.min()),
                "max_length": int(lengths.max()),
                "mean_length": float(lengths.mean()),
                "y_min": float(frame["y"].min()),
                "y_max": float(frame["y"].max()),
            }
        ]
    )


def _series_profile(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute simple per-series statistics for plot selection."""
    return (
        frame.groupby("unique_id", as_index=False)
        .agg(
            length=("y", "size"),
            variance=("y", "var"),
            max_value=("y", "max"),
            min_value=("y", "min"),
        )
        .fillna({"variance": 0.0})
    )


def select_representative_series_ids(
    frame: pd.DataFrame,
    limit: int = 5,
) -> list[str]:
    """Pick a small set of representative series from simple extremes."""
    if frame.empty or limit <= 0:
        return []

    profile = _series_profile(frame)
    selected_ids: list[str] = []
    for column_name, ascending in REPRESENTATIVE_SERIES_RULES:
        ordered_ids = profile.sort_values(
            [column_name, "unique_id"],
            ascending=[ascending, True],
            kind="stable",
        )["unique_id"]
        for unique_id in ordered_ids:
            candidate = str(unique_id)
            if candidate in selected_ids:
                continue
            selected_ids.append(candidate)
            if len(selected_ids) >= limit:
                return selected_ids
    return selected_ids[:limit]


def _format_rmae_label(rmae_value: float) -> str:
    """Format one RMAE value for plot legends."""
    if pd.isna(rmae_value):
        return "n/a"
    return f"{float(rmae_value):.3f}"


def _build_model_label(model_name: str, rmae_value: float, params: int) -> str:
    """Build one plot legend label for a model prediction line."""
    return f"{model_name} (RMAE={_format_rmae_label(rmae_value)}, params={params})"


def _slugify(text: str) -> str:
    """Convert a free-form string into a filesystem-safe slug."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return slug.lower() or "plot"


def save_representative_forecast_plots(
    frame: pd.DataFrame,
    forecast_frames: dict[str, pd.DataFrame],
    results_frame: pd.DataFrame,
    output_dir: Path,
    title_prefix: str,
    limit: int = 5,
) -> list[SavedPlot]:
    """Save representative train/test/prediction plots for a benchmark run."""
    selected_ids = select_representative_series_ids(frame, limit=limit)
    if not selected_ids or not forecast_frames:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    ordered_results = results_frame.sort_values(["mae", "rmse"]).reset_index(drop=True)
    reference_forecast = next(iter(forecast_frames.values()))
    saved_plots: list[SavedPlot] = []

    for unique_id in selected_ids:
        reference_series = reference_forecast[
            reference_forecast["unique_id"].astype(str) == unique_id
        ].sort_values("ds")
        if reference_series.empty:
            continue

        series_frame = frame[frame["unique_id"].astype(str) == unique_id].sort_values(
            "ds"
        )
        test_dates = reference_series["ds"].drop_duplicates()
        train_frame = series_frame[~series_frame["ds"].isin(test_dates)]

        figure, axis = plt.subplots(figsize=(12, 4))
        if not train_frame.empty:
            axis.plot(
                train_frame["ds"],
                train_frame["y"],
                color="black",
                linewidth=1.5,
                label="Train",
            )
        axis.plot(
            reference_series["ds"],
            reference_series["y"],
            color="tab:green",
            linewidth=2,
            marker="o",
            label="Test",
        )

        for row in ordered_results.itertuples():
            prediction_frame = forecast_frames.get(str(row.model_name))
            if (
                prediction_frame is None
                or row.model_name not in prediction_frame.columns
            ):
                continue
            prediction_series = prediction_frame[
                prediction_frame["unique_id"].astype(str) == unique_id
            ].sort_values("ds")
            if prediction_series.empty:
                continue
            axis.plot(
                prediction_series["ds"],
                prediction_series[row.model_name],
                linestyle="--",
                marker="o",
                label=_build_model_label(
                    str(row.model_name),
                    float(row.rmae),
                    int(row.params),
                ),
            )

        title = f"{title_prefix} - series {unique_id}"
        axis.set_title(title)
        axis.set_xlabel("ds")
        axis.set_ylabel("y")
        axis.grid(alpha=0.3)
        axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
        figure.tight_layout()

        output_path = output_dir / f"{_slugify(title)}.png"
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        saved_plots.append(SavedPlot(title=title, path=output_path))

    return saved_plots


def build_plot_markdown(saved_plots: list[SavedPlot], report_path: Path) -> str:
    """Render markdown image links for persisted plot files."""
    if not saved_plots:
        return "No plots were generated."

    lines: list[str] = []
    for saved_plot in saved_plots:
        relative_path = Path(
            os.path.relpath(saved_plot.path, start=report_path.parent)
        ).as_posix()
        lines.extend(
            [
                f"### {saved_plot.title}",
                "",
                f"![{saved_plot.title}]({relative_path})",
                "",
            ]
        )
    return "\n".join(lines)


def _wrap_markdown_lines(markdown_text: str) -> list[str]:
    """Wrap markdown text into fixed-width lines suitable for PDF pages."""
    wrapped_lines: list[str] = []
    for raw_line in markdown_text.splitlines():
        stripped_line = raw_line.strip()
        if MARKDOWN_IMAGE_PATTERN.search(stripped_line):
            continue
        if not stripped_line:
            wrapped_lines.append("")
            continue
        wrapped_lines.extend(
            textwrap.wrap(
                raw_line,
                width=PDF_TEXT_WIDTH,
                break_long_words=False,
                break_on_hyphens=False,
            )
            or [""]
        )
    return wrapped_lines


def _extract_markdown_images(
    markdown_text: str, base_dir: Path
) -> list[tuple[str, Path]]:
    """Extract markdown image references resolved against one base directory."""
    images: list[tuple[str, Path]] = []
    for match in MARKDOWN_IMAGE_PATTERN.finditer(markdown_text):
        title = match.group("alt") or Path(match.group("path")).stem
        resolved_path = Path(match.group("path"))
        if not resolved_path.is_absolute():
            resolved_path = base_dir / resolved_path
        images.append((title, resolved_path.resolve()))
    return images


def _save_pdf_text_pages(markdown_text: str, pdf_pages: PdfPages) -> None:
    """Save the markdown text as one or more PDF pages."""
    wrapped_lines = _wrap_markdown_lines(markdown_text)
    if not wrapped_lines:
        wrapped_lines = ["No report content available."]

    for start in range(0, len(wrapped_lines), PDF_LINES_PER_PAGE):
        page_lines = wrapped_lines[start : start + PDF_LINES_PER_PAGE]
        figure = plt.figure(figsize=PDF_PAGE_SIZE)
        figure.text(
            0.05,
            0.97,
            "\n".join(page_lines),
            va="top",
            ha="left",
            family="monospace",
            fontsize=9,
        )
        plt.axis("off")
        pdf_pages.savefig(figure, bbox_inches="tight")
        plt.close(figure)


def _save_pdf_image_pages(
    markdown_text: str,
    pdf_pages: PdfPages,
    base_dir: Path,
) -> None:
    """Append markdown-linked images as dedicated PDF pages."""
    for title, image_path in _extract_markdown_images(markdown_text, base_dir):
        if not image_path.exists():
            continue
        figure, axis = plt.subplots(figsize=PDF_PAGE_SIZE)
        axis.imshow(plt.imread(image_path))
        axis.set_title(title)
        axis.axis("off")
        figure.tight_layout()
        pdf_pages.savefig(figure, bbox_inches="tight")
        plt.close(figure)


def save_markdown_pdf(markdown_text: str, output_pdf: Path, base_dir: Path) -> None:
    """Render a markdown report and its linked images into a PDF file."""
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_pdf) as pdf_pages:
        _save_pdf_text_pages(markdown_text, pdf_pages)
        _save_pdf_image_pages(markdown_text, pdf_pages, base_dir)
