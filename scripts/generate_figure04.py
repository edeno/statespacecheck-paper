"""Real hippocampal data model comparison for state space model diagnostics.

This script generates Figure 4, which compares per-cell diagnostic metrics
between Continuous and Continuous-Fragmented decoder models on real neural
recording data from hippocampus.

Requires:
- non_local_detector package for decoder models
- Pre-exported neural recording data in data/real/
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from statespacecheck_paper.load_local_data import load_neural_recording_from_files
from statespacecheck_paper.real_data_analysis import (
    compute_model_diagnostics,
    create_decoder_environment,
    extract_place_fields,
    fit_decoder_models,
    get_spike_counts,
)
from statespacecheck_paper.real_data_plotting import (
    plot_metric_distributions,
    plot_metrics_time_vs_position_comparison,
    plot_model_comparison_with_posterior,
    plot_track_graph_2d,
)
from statespacecheck_paper.style import save_figure, set_figure_defaults

# Check for optional decoder dependencies
try:
    import non_local_detector  # noqa: F401

    DECODERS_AVAILABLE = True
except ImportError:
    DECODERS_AVAILABLE = False
    warnings.warn(
        "non_local_detector not available. Figure 4 requires fitted decoder models. "
        "Install with: pip install non_local_detector",
        stacklevel=2,
    )


# -----------------------------
# Configuration
# -----------------------------

DATA_PATH = Path(__file__).parent.parent / "data"
ANIMAL_DATE_EPOCH = "j1620210710_02_r1"

# Time window for Figure 4a (index into decoded results)
FIGURE_4A_WINDOW_CENTER = 177301  # Time index around which to show detail
FIGURE_4A_WINDOW_HALF_WIDTH = 50  # Half-width in time points


def run_demo() -> None:
    """Run the full Figure 4 generation pipeline.

    Loads data, fits models, computes diagnostics, and generates figures.
    Decodes all time points for summary statistics (Figure 4b), then shows
    a specific time window in detail (Figure 4a).
    """
    # Load data
    print("Loading data...")
    data = load_neural_recording_from_files(DATA_PATH, ANIMAL_DATE_EPOCH)
    print(f"  Loaded {len(data['spike_times'])} cells")

    # Create environment
    env = create_decoder_environment(
        track_graph=data["track_graph"],
        edge_order=data["linear_edge_order"],
        edge_spacing=data["linear_edge_spacing"],
    )

    # Prepare training data
    position_info = data["position_info"]
    time = position_info.index.values
    position = position_info[["head_position_x", "head_position_y"]].values
    linear_position = position_info["linear_position"].values
    spike_times_list: list[Any] = list(data["spike_times"])

    # Fit models
    print("Fitting models...")
    continuous_model, contfrag_model = fit_decoder_models(
        position=position,
        spike_times=spike_times_list,
        time=time,
        environment=env,
    )

    # Decode all time points
    print(f"Decoding {len(time)} time points...")

    continuous_results = continuous_model.predict(
        spike_times=spike_times_list,
        time=time,
        return_outputs=["filter", "predictive_posterior", "log_likelihood"],
    )
    contfrag_results = contfrag_model.predict(
        spike_times=spike_times_list,
        time=time,
        return_outputs=["filter", "predictive_posterior", "log_likelihood"],
    )

    # Get spike counts for all time
    spike_counts = get_spike_counts(spike_times_list, time)

    # Compute diagnostics for all time
    print("Computing diagnostics...")
    continuous_diagnostics = compute_model_diagnostics(
        continuous_model, continuous_results, spike_counts, time
    )
    contfrag_diagnostics = compute_model_diagnostics(
        contfrag_model, contfrag_results, spike_counts, time
    )

    # Extract place fields to get peak positions for sorting raster
    # Note: np.nanargmax returns 0 for all-NaN rows, which maps to first position bin
    place_fields, position_bins = extract_place_fields(continuous_model)
    if np.any(np.all(np.isnan(place_fields), axis=1)):
        warnings.warn(
            "Some cells have all-NaN place fields; peak positions may be incorrect",
            stacklevel=2,
        )
    place_field_peaks = position_bins[np.nanargmax(place_fields, axis=1)]

    # Print summary (all time points)
    print("\n=== Diagnostic Summary (all time points) ===")
    for metric in ["hpd_overlap", "kl_divergence", "spike_prob"]:
        cont_mean = np.nanmean(continuous_diagnostics[metric])
        frag_mean = np.nanmean(contfrag_diagnostics[metric])
        print(f"{metric}:")
        print(f"  Continuous: {cont_mean:.4f}")
        print(f"  ContFrag:   {frag_mean:.4f}")

    # Generate figures
    print("\nGenerating Figure 4...")
    set_figure_defaults()

    # Define time slice for Figure 4a (detail view)
    window_start = FIGURE_4A_WINDOW_CENTER - FIGURE_4A_WINDOW_HALF_WIDTH
    window_end = FIGURE_4A_WINDOW_CENTER + FIGURE_4A_WINDOW_HALF_WIDTH
    time_slice_ind = slice(window_start, window_end)

    # Figure 4a: Model comparison with filter, predictive, likelihood, raster, and diagnostics
    fig, axes = plot_model_comparison_with_posterior(
        time,
        linear_position,
        continuous_results,
        contfrag_results,
        continuous_diagnostics,
        contfrag_diagnostics,
        spike_times=spike_times_list,
        spike_counts=spike_counts,
        place_field_peaks=place_field_peaks,
        time_slice_ind=time_slice_ind,
        model_a_name="Continuous",
        model_b_name="Continuous-Fragmented",
        track_graph=data["track_graph"],
        edge_order=data["linear_edge_order"],
        edge_spacing=data["linear_edge_spacing"],
        show_running_average=False,
    )
    save_figure("figures/main/figure04a", close=True)
    print("Saved figures/main/figure04a.{pdf,png}")

    # Figure 4b: Metric distributions comparing all time points
    fig, axes = plot_metric_distributions(
        continuous_diagnostics,
        contfrag_diagnostics,
        model_a_name="Continuous",
        model_b_name="Continuous-Fragmented",
    )
    save_figure("figures/main/figure04b", close=True)
    print("Saved figures/main/figure04b.{pdf,png}")

    # Figure 4c: 2D track graph for reference
    fig, ax = plt.subplots(figsize=(2.5, 2.5), constrained_layout=True)
    plot_track_graph_2d(
        data["track_graph"],
        position_info,
        ax=ax,
        edge_order=data["linear_edge_order"],
        show_trajectory=True,
    )
    save_figure("figures/main/figure04c", close=True)
    print("Saved figures/main/figure04c.{pdf,png}")

    # Figure 4d: Metrics vs linear position for continuous-fragmented model
    fig, axes = plot_metrics_time_vs_position_comparison(
        linear_position,
        contfrag_diagnostics,
        model_name="Cont-Frag",
        track_graph=data["track_graph"],
        edge_order=data["linear_edge_order"],
        edge_spacing=data["linear_edge_spacing"],
    )
    save_figure("figures/main/figure04d", close=True)
    print("Saved figures/main/figure04d.{pdf,png}")

    # ---------------------------------------------------------------
    # Combined Figure 4: all panels in a single one-page figure
    # ---------------------------------------------------------------
    print("\nGenerating combined Figure 4...")

    # Two rows: (top) example window + track graph, (bottom) hexbin + diff
    fig = plt.figure(figsize=(7.0, 9.0), dpi=450, constrained_layout=True)
    subfigs = fig.subfigures(
        2,
        1,
        height_ratios=[5.5, 3.5],
        hspace=0.03,
    )

    # --- Row 0: (a) example time window + (b) track graph on right ---
    subfigs_top = subfigs[0].subfigures(
        1,
        2,
        width_ratios=[5.5, 1.5],
        wspace=0.01,
    )

    # Panel (a): example time window (6x2 grid)
    _, axes_a = plot_model_comparison_with_posterior(
        time,
        linear_position,
        continuous_results,
        contfrag_results,
        continuous_diagnostics,
        contfrag_diagnostics,
        spike_times=spike_times_list,
        spike_counts=spike_counts,
        place_field_peaks=place_field_peaks,
        time_slice_ind=time_slice_ind,
        model_a_name="Continuous",
        model_b_name="Cont-Frag",
        track_graph=data["track_graph"],
        edge_order=data["linear_edge_order"],
        edge_spacing=data["linear_edge_spacing"],
        show_running_average=False,
        fig=subfigs_top[0],
    )
    # Use blended transform: x in figure coords, y in axes coords
    # This ensures both panel labels share the same absolute x position
    from matplotlib.transforms import blended_transform_factory

    label_x = 0.01  # absolute figure x for both labels
    trans_a = blended_transform_factory(fig.transFigure, axes_a[0, 0].transAxes)
    axes_a[0, 0].text(
        label_x,
        1.15,
        "a",
        fontsize=8,
        fontweight="bold",
        transform=trans_a,
        va="top",
        ha="left",
    )

    # Panel (b): 2D track graph on the right, vertically centered
    ax_track = subfigs_top[1].subplots(1, 1)
    plot_track_graph_2d(
        data["track_graph"],
        position_info,
        ax=ax_track,
        edge_order=data["linear_edge_order"],
        show_trajectory=True,
    )
    ax_track.set_title("")
    ax_track.tick_params(labelsize=5)
    # --- Row 1: (b) model comparison hexbin + difference histograms ---
    _, axes_b = plot_metric_distributions(
        continuous_diagnostics,
        contfrag_diagnostics,
        model_a_name="Continuous",
        model_b_name="Cont-Frag",
        show_diff=True,
        fig=subfigs[1],
    )
    # Same blended transform: figure x, axes y — ensures vertical alignment with "a"
    trans_b = blended_transform_factory(fig.transFigure, axes_b[0, 0].transAxes)
    axes_b[0, 0].text(
        label_x,
        1.15,
        "b",
        fontsize=8,
        fontweight="bold",
        transform=trans_b,
        va="top",
        ha="left",
    )

    save_figure("figures/main/figure04", close=True)
    print("Saved figures/main/figure04.{pdf,png}")

    print("\nFigure 4 complete!")


if __name__ == "__main__":
    run_demo()
