"""Generate validated four-layer FlyWire DSK neuropil visualizations."""
from __future__ import annotations

import os
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

CONNECTIONS_FILE = "connections.csv"
DSK_FILE = "search_results_DSK.csv"
OUT_DIR = "figures"
TOP_N_NEUROPILS = 18
LAYER_NAMES = ["1° Upstream", "2° Upstream", "1° Downstream", "2° Downstream"]
EDGE_COLUMNS = ["pre_root_id", "post_root_id", "neuropil", "nt_type", "syn_count"]
EDGE_KEYS = ["pre_root_id", "post_root_id", "neuropil", "nt_type"]


def load_data() -> tuple[pd.DataFrame, set[str], dict[str, int]]:
    """Load CSVs, validate numeric counts and identifiers, and report exclusions."""
    raw = pd.read_csv(CONNECTIONS_FILE, dtype=str)
    required_connection_columns = {"From", "To", "Neuropil", "Synapses", "Neuro Transmitter"}
    missing_columns = required_connection_columns.difference(raw.columns)
    if missing_columns:
        raise ValueError(f"connections.csv is missing required columns: {sorted(missing_columns)}")

    connections = raw.rename(columns={
        "From": "pre_root_id",
        "To": "post_root_id",
        "Neuropil": "neuropil",
        "Synapses": "syn_count",
        "Neuro Transmitter": "nt_type",
    }).copy()
    # Convert immediately after loading/renaming; never aggregate string counts.
    connections["syn_count"] = pd.to_numeric(connections["syn_count"], errors="coerce")

    count_invalid = connections["syn_count"].isna() | ~np.isfinite(connections["syn_count"])
    count_invalid |= connections["syn_count"].lt(0)
    count_invalid |= connections["syn_count"].mod(1).ne(0)
    connections["pre_root_id"] = connections["pre_root_id"].str.strip()
    connections["post_root_id"] = connections["post_root_id"].str.strip()
    connections["neuropil"] = connections["neuropil"].str.strip()
    connections["nt_type"] = connections["nt_type"].str.strip()
    missing_endpoint = connections["pre_root_id"].isna() | connections["pre_root_id"].eq("")
    missing_endpoint |= connections["post_root_id"].isna() | connections["post_root_id"].eq("")
    missing_neuropil = connections["neuropil"].isna() | connections["neuropil"].eq("")

    exclusions = {
        "invalid/missing/negative/non-integer syn_count": int(count_invalid.sum()),
        "missing presynaptic or postsynaptic ID": int((~count_invalid & missing_endpoint).sum()),
        "missing neuropil": int((~count_invalid & ~missing_endpoint & missing_neuropil).sum()),
    }
    valid = ~(count_invalid | missing_endpoint | missing_neuropil)
    connections = connections.loc[valid, EDGE_COLUMNS].copy()
    connections["syn_count"] = connections["syn_count"].astype("int64")
    if connections.empty:
        raise ValueError("No valid connection rows remain after explicit data validation.")

    dsk = pd.read_csv(DSK_FILE, dtype=str)
    if "root_id" not in dsk.columns:
        raise ValueError("search_results_DSK.csv is missing the required root_id column.")
    main_ids = set(dsk["root_id"].dropna().str.strip())
    main_ids.discard("")
    if not main_ids:
        raise ValueError("No valid DSK root_id values were found.")
    return connections, main_ids, exclusions


def aggregate_edges(edges: pd.DataFrame) -> pd.DataFrame:
    """Sum numeric synapse counts while preserving both endpoint IDs."""
    if edges.empty:
        return pd.DataFrame(columns=EDGE_COLUMNS).astype({"syn_count": "int64"})
    result = (edges.groupby(EDGE_KEYS, as_index=False, dropna=False, sort=False)
              .agg(syn_count=("syn_count", "sum")))
    result["syn_count"] = result["syn_count"].astype("int64")
    return result[EDGE_COLUMNS]


def edges_from(connections: pd.DataFrame, src_ids: Iterable[str]) -> pd.DataFrame:
    """Get outgoing edges and retain source and target identity through grouping."""
    selected = connections.loc[connections["pre_root_id"].isin(src_ids), EDGE_COLUMNS]
    return aggregate_edges(selected)


def edges_to(connections: pd.DataFrame, target_ids: Iterable[str]) -> pd.DataFrame:
    """Get incoming edges and retain source and target identity through grouping."""
    selected = connections.loc[connections["post_root_id"].isin(target_ids), EDGE_COLUMNS]
    return aggregate_edges(selected)


def summarize_neuropils(edges: pd.DataFrame, layer: str) -> pd.DataFrame:
    """Compute synapse totals and percentages by neuropil for one layer."""
    if edges.empty:
        return pd.DataFrame({
            "neuropil": pd.Series(dtype="object"),
            "syn_count": pd.Series(dtype="int64"),
            "pct": pd.Series(dtype="float64"),
            "layer": pd.Series(dtype="object"),
        })
    summary = (edges.groupby("neuropil", as_index=False, dropna=False, sort=False)
               .agg(syn_count=("syn_count", "sum")))
    total = int(summary["syn_count"].sum())
    if total <= 0:
        raise ValueError(f"{layer} has no positive synapse total; percentages cannot be plotted honestly.")
    summary["pct"] = summary["syn_count"] / total * 100.0
    summary["layer"] = layer
    return summary[["neuropil", "syn_count", "pct", "layer"]]


def configure_plot_style() -> None:
    sns.set_theme(style="white")
    plt.rcParams.update({
        "axes.grid": True,
        "grid.linestyle": "--",
        "legend.framealpha": 1,
        "legend.facecolor": "white",
        "legend.shadow": False,
        "legend.fontsize": 10,
        "legend.title_fontsize": 11,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.labelsize": 11,
        "axes.titlesize": 15,
        "figure.dpi": 110,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    })


def draw_heatmap(ax: plt.Axes, pivot: pd.DataFrame, title: str) -> None:
    annotations = pivot.apply(lambda column: column.map(
        lambda value: "" if pd.isna(value) else f"{value:.1f}%"))
    sns.heatmap(
        pivot,
        mask=pivot.isna(),
        annot=annotations,
        fmt="",
        cmap="YlGnBu",
        vmin=0,
        vmax=100,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "% of layer synapses"},
        ax=ax,
    )
    ax.set_title(title, pad=12, weight="bold")
    ax.set_xlabel("Connectivity layer")
    ax.set_ylabel("Neuropil (ranked across all layers)")
    ax.tick_params(axis="x", rotation=20)
    ax.tick_params(axis="y", rotation=0)


def draw_stacked_bars(ax: plt.Axes, pivot: pd.DataFrame,
                      neuropil_order: list[str], colors: dict[str, tuple]) -> None:
    # Transpose: one row per layer, one column per neuropil, yielding four bars.
    bar_data = pivot.reindex(index=neuropil_order, columns=LAYER_NAMES).T.fillna(0.0)
    x_positions = np.arange(len(LAYER_NAMES))
    bottoms = np.zeros(len(LAYER_NAMES), dtype=float)
    for neuropil in neuropil_order:
        values = bar_data[neuropil].to_numpy(dtype=float)
        ax.bar(x_positions, values, bottom=bottoms, color=colors[neuropil],
               edgecolor="white", linewidth=0.25, width=0.72, label=neuropil)
        bottoms += values
    for index, total_pct in enumerate(bottoms):
        if np.isclose(total_pct, 0.0):
            ax.text(x_positions[index], 50, "No data", ha="center", va="center",
                    color="dimgray", rotation=90, fontsize=9)
    ax.set_xticks(x_positions, LAYER_NAMES, rotation=18, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Share of layer synapses (%)")
    ax.set_title("Neuropil composition by layer", pad=12, weight="bold")
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", alpha=0.45)
    ax.grid(axis="x", visible=False)
    ax.legend(title="Neuropil", bbox_to_anchor=(1.02, 1), loc="upper left", ncol=1,
              borderaxespad=0.0, fontsize=8, title_fontsize=9)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    connections, main_ids, exclusions = load_data()
    print("=== Input validation ===")
    print(f"Valid raw connection rows: {len(connections)}")
    print(f"DSK population: {len(main_ids)} unique neurons")
    for reason, count in exclusions.items():
        print(f"Excluded rows ({reason}): {count}")
    print("Missing neurotransmitter labels are retained as missing metadata; they are not used to fabricate categories.")

    # Direct layers include all valid edges, including recurrent DSK-to-DSK edges.
    up_1st = edges_to(connections, main_ids)
    down_1st = edges_from(connections, main_ids)
    up_1st_ids = set(up_1st["pre_root_id"])
    down_1st_ids = set(down_1st["post_root_id"])

    # Second-order expansion, excluding edges sourced by original DSK neurons.
    up_2nd = edges_to(connections, up_1st_ids)
    up_2nd = up_2nd.loc[~up_2nd["pre_root_id"].isin(main_ids)].reset_index(drop=True)
    down_2nd = edges_from(connections, down_1st_ids)
    down_2nd = down_2nd.loc[~down_2nd["pre_root_id"].isin(main_ids)].reset_index(drop=True)

    layer_edges = {
        "1° Upstream": up_1st,
        "2° Upstream": up_2nd,
        "1° Downstream": down_1st,
        "2° Downstream": down_2nd,
    }

    neuropil_df = pd.concat(
        [summarize_neuropils(layer_edges[name], name) for name in LAYER_NAMES],
        ignore_index=True,
    )
    # Build the pivot before it is consumed by figures or exports.
    pivot_counts = neuropil_df.pivot(index="neuropil", columns="layer", values="syn_count")
    pivot_counts = pivot_counts.reindex(columns=LAYER_NAMES)
    neuropil_order = pivot_counts.sum(axis=1).sort_values(ascending=False, kind="stable").index.tolist()
    top_neuropils = neuropil_order[:TOP_N_NEUROPILS]
    pivot = (neuropil_df.pivot(index="neuropil", columns="layer", values="pct")
             .reindex(index=top_neuropils, columns=LAYER_NAMES))

    configure_plot_style()
    all_colors = sns.color_palette("husl", n_colors=max(len(neuropil_order), 1))
    neuropil_colors = {neuropil: all_colors[index]
                       for index, neuropil in enumerate(neuropil_order)}

    heatmap_fig, heatmap_ax = plt.subplots(figsize=(12, max(6, 0.38 * len(top_neuropils))))
    draw_heatmap(heatmap_ax, pivot, "DSK connectivity by neuropil")
    heatmap_fig.tight_layout()
    heatmap_fig.savefig(os.path.join(OUT_DIR, "neuropil_heatmap.png"), bbox_inches="tight")
    plt.close(heatmap_fig)

    percentages = pivot_counts.div(
        pivot_counts.sum(axis=0).replace(0, np.nan), axis=1
    ).mul(100).reindex(index=neuropil_order)
    bar_fig, bar_ax = plt.subplots(figsize=(13, 8))
    draw_stacked_bars(bar_ax, percentages, neuropil_order, neuropil_colors)
    bar_fig.tight_layout()
    bar_fig.savefig(os.path.join(OUT_DIR, "neuropil_stacked.png"), bbox_inches="tight")
    plt.close(bar_fig)

    dashboard_fig, (dashboard_heatmap_ax, dashboard_bar_ax) = plt.subplots(
        1, 2, figsize=(24, max(9, 0.4 * len(top_neuropils))),
        gridspec_kw={"width_ratios": [1.05, 1.45]},
    )
    draw_heatmap(dashboard_heatmap_ax, pivot, "DSK connectivity by neuropil")
    draw_stacked_bars(dashboard_bar_ax, percentages, neuropil_order, neuropil_colors)
    dashboard_fig.suptitle("FlyWire DSK connectivity dashboard", fontsize=18, weight="bold", y=1.02)
    dashboard_fig.tight_layout()
    dashboard_fig.savefig(os.path.join(OUT_DIR, "flywire_dsk_dashboard.png"), bbox_inches="tight")
    plt.close(dashboard_fig)

    pivot.to_csv(os.path.join(OUT_DIR, "neuropil_pivot_pct.csv"))
    neuropil_df.to_csv(os.path.join(OUT_DIR, "neuropil_layer_summary.csv"), index=False)

    print("\n=== Strict layer validation ===")
    for name in LAYER_NAMES:
        edges = layer_edges[name]
        total = int(edges["syn_count"].sum())
        percentages_for_layer = neuropil_df.loc[neuropil_df["layer"].eq(name), "pct"]
        pct_sum = float(percentages_for_layer.sum())
        pct_display = f"{pct_sum:.6f}%" if total else "N/A (no synapses)"
        print(f"{name}: unique_pre={edges['pre_root_id'].nunique()}, "
              f"unique_post={edges['post_root_id'].nunique()}, edges={len(edges)}, "
              f"total_synapses={total}, percentage_sum={pct_display}, "
              f"unique_neuropils={edges['neuropil'].nunique()}")
        if total and not np.isclose(pct_sum, 100.0, atol=1e-8):
            raise AssertionError(f"{name} percentages sum to {pct_sum}, not 100%.")
        if not total and (not edges.empty or not percentages_for_layer.empty):
            raise AssertionError(f"{name} has no total synapses but contains data.")
        if not pd.api.types.is_numeric_dtype(edges["syn_count"]):
            raise AssertionError(f"{name} syn_count is not numeric.")

    second_order = pd.concat([up_2nd, down_2nd], ignore_index=True)
    dsk_source_leaks = int(second_order["pre_root_id"].isin(main_ids).sum())
    if dsk_source_leaks:
        raise AssertionError("Second-order edges include DSK presynaptic neurons.")
    recurrent_direct = down_1st.loc[down_1st["post_root_id"].isin(main_ids)]
    recurrent_synapses = int(recurrent_direct["syn_count"].sum())
    if not recurrent_direct.empty and not recurrent_direct["pre_root_id"].isin(main_ids).all():
        raise AssertionError("A recurrent direct downstream edge is not DSK-sourced.")
    print("Numeric aggregation check: PASS (syn_count numeric before every groupby sum; no string concatenation).")
    print(f"Direct recurrent DSK-to-DSK edges retained: {len(recurrent_direct)} "
          f"({recurrent_synapses} synapses).")
    print(f"Second-order DSK-output duplication check: PASS ({dsk_source_leaks} DSK-sourced second-order edges).")
    print("Saved: figures/neuropil_heatmap.png, figures/neuropil_stacked.png, "
          "figures/flywire_dsk_dashboard.png")


if __name__ == "__main__":
    main()