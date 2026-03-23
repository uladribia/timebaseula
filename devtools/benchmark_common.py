"""Shared helpers for internal benchmark CLIs."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae, rmse


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


def evaluate_cv_results(
    forecast_frame: pd.DataFrame,
    model_names: list[str],
) -> pd.DataFrame:
    """Evaluate a cross-validation frame with native utilsforecast metrics."""
    metric_frame = evaluate(
        df=forecast_frame,
        metrics=[mae, rmse],
        models=model_names,
        agg_fn="mean",
    )
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
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def dataframe_to_markdown_table(frame: pd.DataFrame) -> str:
    """Convert a frame to a simple markdown table without optional deps."""
    if frame.empty:
        return "No data available."
    columns = [str(column) for column in frame.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in frame.iterrows():
        values = [str(row[column]) for column in frame.columns]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def build_markdown_report(
    title: str,
    source_label: str,
    results_frame: pd.DataFrame,
    slice_columns: list[str],
) -> str:
    """Render a simple markdown benchmark report."""
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
    return "\n".join(lines)
