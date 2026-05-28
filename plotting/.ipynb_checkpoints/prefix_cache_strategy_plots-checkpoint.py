"""Notebook-friendly plots for prefix cache benchmark CSVs.

Example:
    from plotting.prefix_cache_strategy_plots import *

    df = load_prefix_cache_results("/Users/yifeifu/Downloads/sharegpt_all_prefix_cache_strategies.csv")
    fig, ax = plot_hit_rate_by_capacity(df)
    fig, ax = plot_marconi_vs_vllm_gain(df)
    fig, ax = plot_hit_rate_by_kv_fraction(df, capacity_gb=5)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


STRATEGY_LABELS = {
    "vllm_plus": "vLLM+",
    "marconi_v2": "Marconi",
    "hybrid_intersection": "Hybrid prefix cache",
}

STRATEGY_COLORS = {
    "vllm_plus": "#4C78A8",
    "marconi_v2": "#F58518",
    "hybrid_intersection": "#54A24B",
}

STRATEGY_MARKERS = {
    "vllm_plus": "o",
    "marconi_v2": "s",
    "hybrid_intersection": "^",
}


def load_prefix_cache_results(csv_path: str | Path) -> pd.DataFrame:
    """Load a prefix_cache_benchmark.py CSV with normalized strategy labels."""
    df = pd.read_csv(csv_path)
    df["cache_type"] = df["cache_type"].astype(str)
    df["strategy_label"] = df["cache_type"].map(STRATEGY_LABELS).fillna(df["cache_type"])
    return df


def strategy_subset(
    df: pd.DataFrame,
    include_simple: bool = False,
    simple_kv_fraction: float | None = None,
) -> pd.DataFrame:
    """Return rows for Marconi/vLLM+ and, optionally, simple hybrid prefix cache."""
    strategies = ["vllm_plus", "marconi_v2"]
    if include_simple:
        strategies.append("hybrid_intersection")

    out = df[df["cache_type"].isin(strategies)].copy()
    if include_simple and simple_kv_fraction is not None:
        simple_mask = out["cache_type"].eq("hybrid_intersection")
        keep_simple = out["kv_cache_fraction"].eq(simple_kv_fraction)
        out = out[~simple_mask | keep_simple]
    return out


def _plot_strategy_lines(
    ax: plt.Axes,
    df: pd.DataFrame,
    x: str,
    y: str,
    strategies: Iterable[str],
    y_scale: float = 100.0,
) -> None:
    for strategy in strategies:
        rows = df[df["cache_type"].eq(strategy)].sort_values(x)
        if rows.empty:
            continue
        ax.plot(
            rows[x],
            rows[y] * y_scale,
            marker=STRATEGY_MARKERS.get(strategy, "o"),
            linewidth=2,
            markersize=6,
            color=STRATEGY_COLORS.get(strategy),
            label=STRATEGY_LABELS.get(strategy, strategy),
        )


def plot_hit_rate_by_capacity(
    df: pd.DataFrame,
    metric: str = "token_hit_rate",
    include_simple: bool = False,
    simple_kv_fraction: float | None = 0.5,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (5.2, 3.2),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot token or request hit rate versus cache capacity."""
    data = strategy_subset(df, include_simple=include_simple, simple_kv_fraction=simple_kv_fraction)
    strategies = ["vllm_plus", "marconi_v2"] + (["hybrid_intersection"] if include_simple else [])

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    else:
        fig = ax.figure

    _plot_strategy_lines(ax, data, "capacity_gb", metric, strategies)
    ax.set_xlabel("Cache capacity (GB)")
    ax.set_ylabel("Token hit rate (%)" if metric == "token_hit_rate" else "Request hit rate (%)")
    ax.grid(color="lightgrey", linestyle="dashed", axis="y", linewidth=0.8)
    ax.legend(frameon=False)
    return fig, ax


def plot_marconi_vs_vllm_gain(
    df: pd.DataFrame,
    metric: str = "token_hit_rate",
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (5.2, 3.2),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot Marconi's relative gain over vLLM+ by capacity."""
    paired = (
        df[df["cache_type"].isin(["vllm_plus", "marconi_v2"])]
        .pivot_table(index="capacity_gb", columns="cache_type", values=metric, aggfunc="first")
        .dropna(subset=["vllm_plus", "marconi_v2"])
        .sort_index()
    )
    paired["relative_gain"] = (paired["marconi_v2"] / paired["vllm_plus"] - 1.0) * 100.0

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    else:
        fig = ax.figure

    ax.plot(
        paired.index,
        paired["relative_gain"],
        marker="D",
        linewidth=2,
        markersize=6,
        color=STRATEGY_COLORS["marconi_v2"],
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Cache capacity (GB)")
    ax.set_ylabel("Marconi gain over vLLM+ (%)")
    ax.grid(color="lightgrey", linestyle="dashed", axis="y", linewidth=0.8)
    return fig, ax


def plot_hit_rate_by_kv_fraction(
    df: pd.DataFrame,
    capacity_gb: float,
    metric: str = "token_hit_rate",
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (5.2, 3.2),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the simple hybrid-prefix-cache row versus KV cache fraction."""
    data = df[
        df["cache_type"].eq("hybrid_intersection")
        & df["capacity_gb"].eq(capacity_gb)
    ].sort_values("kv_cache_fraction")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    else:
        fig = ax.figure

    ax.plot(
        data["kv_cache_fraction"],
        data[metric] * 100.0,
        marker=STRATEGY_MARKERS["hybrid_intersection"],
        linewidth=2,
        markersize=6,
        color=STRATEGY_COLORS["hybrid_intersection"],
        label=STRATEGY_LABELS["hybrid_intersection"],
    )
    ax.set_xlabel("KV cache fraction")
    ax.set_ylabel("Token hit rate (%)" if metric == "token_hit_rate" else "Request hit rate (%)")
    ax.grid(color="lightgrey", linestyle="dashed", axis="y", linewidth=0.8)
    ax.legend(frameon=False)
    return fig, ax


def plot_cache_memory_usage(
    df: pd.DataFrame,
    include_simple: bool = False,
    simple_kv_fraction: float | None = 0.5,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (5.2, 3.2),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot total used cache memory for Marconi/vLLM+ and optional simple hybrid."""
    data = strategy_subset(df, include_simple=include_simple, simple_kv_fraction=simple_kv_fraction)
    data = data.copy()
    data["used_gb"] = data["kv_used_gb"].fillna(0) + data["ssm_used_gb"].fillna(0)
    strategies = ["vllm_plus", "marconi_v2"] + (["hybrid_intersection"] if include_simple else [])

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    else:
        fig = ax.figure

    _plot_strategy_lines(ax, data, "capacity_gb", "used_gb", strategies, y_scale=1.0)
    ax.set_xlabel("Cache capacity (GB)")
    ax.set_ylabel("Used cache memory (GB)")
    ax.grid(color="lightgrey", linestyle="dashed", axis="y", linewidth=0.8)
    ax.legend(frameon=False)
    return fig, ax

