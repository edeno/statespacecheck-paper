"""Sanity check for Figure 4b: sample time windows from different metric regions.

For each metric (HPD overlap, KL divergence, spike prob), finds time windows where
the model difference falls at different quantiles (Continuous better, similar,
Continuous-Fragmented better) and generates Figure 4a-style diagnostic panels.

Saves output to manuscript/figures/preview/sanity_check_4b/.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np

from statespacecheck_paper.analysis import PerCellDiagnostics
from statespacecheck_paper.load_local_data import load_neural_recording_from_files
from statespacecheck_paper.real_data_analysis import (
    compute_model_diagnostics,
    create_decoder_environment,
    extract_place_fields,
    fit_decoder_models,
    get_spike_counts,
)
from statespacecheck_paper.real_data_plotting import (
    plot_model_comparison_with_posterior,
)
from statespacecheck_paper.style import save_figure, set_figure_defaults

DATA_PATH = Path(__file__).parent.parent / "data"
ANIMAL_DATE_EPOCH = "j1620210710_02_r1"
OUTPUT_DIR = "manuscript/figures/preview/sanity_check_4b"
HALF_WIDTH = 50  # Same as Figure 4a


def find_representative_windows(
    continuous_diagnostics: PerCellDiagnostics,
    contfrag_diagnostics: PerCellDiagnostics,
    half_width: int = HALF_WIDTH,
    quantiles: tuple[float, ...] = (0.05, 0.50, 0.95),
) -> dict[str, list[tuple[str, int, slice]]]:
    """Find time windows at different quantiles of the metric difference.

    For each metric, computes the median-across-cells difference (B - A) at each
    time point, then finds time points near the target quantiles.

    Parameters
    ----------
    continuous_diagnostics : dict
        Model A diagnostics, each value shape (n_time, n_cells).
    contfrag_diagnostics : dict
        Model B diagnostics, each value shape (n_time, n_cells).
    half_width : int
        Half-width of the time window in indices.
    quantiles : tuple of float
        Quantiles to sample from.

    Returns
    -------
    windows : dict mapping metric name to list of (label, center_idx, slice).
    """
    metrics = ["hpd_overlap", "kl_divergence", "spike_prob"]
    # For labeling: which direction is "B better"
    # HPD: higher is better, so positive diff = B better
    # KL: lower is better, so negative diff = B better
    # spike_prob: lower is better, so negative diff = B better
    labels_by_quantile = {
        "hpd_overlap": {0.05: "Cont better", 0.50: "Similar", 0.95: "ContFrag better"},
        "kl_divergence": {0.05: "ContFrag better", 0.50: "Similar", 0.95: "Cont better"},
        "spike_prob": {0.05: "ContFrag better", 0.50: "Similar", 0.95: "Cont better"},
    }

    n_time = continuous_diagnostics.hpd_overlap.shape[0]
    windows: dict[str, list[tuple[str, int, slice]]] = {}

    for metric in metrics:
        diff = getattr(contfrag_diagnostics, metric) - getattr(continuous_diagnostics, metric)

        # Transform spike_prob to -log10 scale for consistency with 4b
        if metric == "spike_prob":
            a = -np.log10(np.maximum(getattr(continuous_diagnostics, metric), 1e-10))
            b = -np.log10(np.maximum(getattr(contfrag_diagnostics, metric), 1e-10))
            diff = b - a

        # Median across cells at each time point
        median_diff = np.nanmedian(diff, axis=1)

        # Remove edges where we can't center a window
        valid = np.ones(n_time, dtype=bool)
        valid[:half_width] = False
        valid[-half_width:] = False
        valid &= ~np.isnan(median_diff)

        metric_windows: list[tuple[str, int, slice]] = []
        for q in quantiles:
            target = float(np.nanquantile(median_diff[valid], q))
            # Find the time point closest to this quantile value
            distances = np.abs(median_diff - target)
            distances[~valid] = np.inf
            center_idx = int(np.argmin(distances))
            time_slice = slice(center_idx - half_width, center_idx + half_width)
            label = labels_by_quantile[metric][q]
            metric_windows.append((label, center_idx, time_slice))

        windows[metric] = metric_windows

    return windows


def run_sanity_check() -> None:
    """Generate sanity check panels for Figure 4b."""
    # --- Data loading and model fitting (same as generate_figure04.py) ---
    print("Loading data...")
    data = load_neural_recording_from_files(DATA_PATH, ANIMAL_DATE_EPOCH)
    print(f"  Loaded {len(data['spike_times'])} cells")

    env = create_decoder_environment(
        track_graph=data["track_graph"],
        edge_order=data["linear_edge_order"],
        edge_spacing=data["linear_edge_spacing"],
    )

    position_info = data["position_info"]
    time = position_info.index.values
    position = position_info[["head_position_x", "head_position_y"]].values
    spike_times_list: list[Any] = list(data["spike_times"])

    print("Fitting models...")
    continuous_model, contfrag_model = fit_decoder_models(
        position=position,
        spike_times=spike_times_list,
        time=time,
        environment=env,
    )

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

    spike_counts = get_spike_counts(spike_times_list, time)

    print("Computing diagnostics...")
    continuous_diagnostics = compute_model_diagnostics(
        continuous_model, continuous_results, spike_counts, time, spike_times=spike_times_list
    )
    contfrag_diagnostics = compute_model_diagnostics(
        contfrag_model, contfrag_results, spike_counts, time, spike_times=spike_times_list
    )

    place_fields, position_bins = extract_place_fields(continuous_model)
    if np.any(np.all(np.isnan(place_fields), axis=1)):
        warnings.warn(
            "Some cells have all-NaN place fields; peak positions may be incorrect",
            stacklevel=2,
        )
    place_field_peaks = position_bins[np.nanargmax(place_fields, axis=1)]

    # --- Find representative windows ---
    print("\nFinding representative time windows...")
    windows = find_representative_windows(
        continuous_diagnostics,
        contfrag_diagnostics,
    )

    # --- Generate panels ---
    set_figure_defaults()
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    linear_position = position_info["linear_position"].values

    for metric, metric_windows in windows.items():
        print(f"\n=== {metric} ===")
        for label, center_idx, time_slice_ind in metric_windows:
            # Print summary for this window
            diff_hpd = (
                contfrag_diagnostics.hpd_overlap[time_slice_ind]
                - continuous_diagnostics.hpd_overlap[time_slice_ind]
            )
            diff_kl = (
                contfrag_diagnostics.kl_divergence[time_slice_ind]
                - continuous_diagnostics.kl_divergence[time_slice_ind]
            )
            print(
                f"  {label} (idx={center_idx}): "
                f"median ΔHPD={np.nanmedian(diff_hpd):.3f}, "
                f"median ΔKL={np.nanmedian(diff_kl):.3f}"
            )

            fig, _axes = plot_model_comparison_with_posterior(
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

            # Add suptitle with context
            fig.suptitle(
                f"{metric}: {label} (center idx={center_idx})",
                fontsize=11,
                y=1.01,
            )

            safe_label = label.replace(" ", "_").lower()
            basename = f"{output_dir}/{metric}_{safe_label}_idx{center_idx}"
            save_figure(basename, close=True)
            print(f"    Saved {basename}.png")

    print("\nSanity check complete!")


if __name__ == "__main__":
    run_sanity_check()
