"""Find immobile high-activity windows where cont-frag model outperforms continuous.

This script identifies time windows where:
1. Animal is immobile (speed < threshold)
2. Multiunit firing rate is high (z-score > threshold)
3. HPD overlap is poor in Continuous model but good in Continuous-Fragmented model

These criteria target potential "replay" events where the continuous movement
assumption fails but the fragmented model succeeds.

Usage:
    python scripts/find_immobile_replay_windows.py
    python scripts/find_immobile_replay_windows.py --speed-threshold 3.0 --z-threshold 1.5
    python scripts/find_immobile_replay_windows.py --no-preview

Outputs:
    - Prints ranked list of candidate events
    - Optionally generates preview figures in manuscript/figures/preview/
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import label
from scipy.stats import zscore

from statespacecheck_paper.load_local_data import load_neural_recording_from_files
from statespacecheck_paper.real_data_analysis import (
    compute_model_diagnostics,
    compute_running_average,
    create_decoder_environment,
    extract_place_fields,
    fit_decoder_models,
    get_spike_counts,
)
from statespacecheck_paper.real_data_plotting import plot_model_comparison_with_posterior
from statespacecheck_paper.style import save_figure, set_figure_defaults

# -----------------------------
# Configuration
# -----------------------------

DATA_PATH = Path(__file__).parent.parent / "data"
ANIMAL_DATE_EPOCH = "j1620210710_02_r1"

# Event detection thresholds
SPEED_THRESHOLD = 4.0  # cm/s - below this is "immobile"
Z_MULTIUNIT_THRESHOLD = 2.0  # z-score threshold for high multiunit activity

# Window parameters
WINDOW_HALF_WIDTH = 100  # Half-width in time points for visualization
RUNNING_AVG_WINDOW = 0.020  # 20ms window for running average

# Diagnostic-driven detection defaults
DIAGNOSTIC_METRIC = "hpd_overlap"
CELL_THRESHOLD = 0.3  # Per-cell threshold for flagging disagreement
MIN_DISAGREEING_CELLS = 3  # Minimum cells that must disagree at a time point

# Filtering thresholds
MIN_DURATION_BINS = 5  # Minimum event duration in time bins (~10ms at 500Hz)
MIN_SPIKES = 10  # Minimum spikes in event

# Number of top candidates to report/visualize
N_TOP_CANDIDATES = 10


@dataclass
class ReplayCandidate:
    """A candidate event with diagnostic metrics."""

    start_idx: int
    end_idx: int
    center_idx: int
    time_start: float
    time_end: float
    duration_ms: float
    n_spikes: int
    # HPD overlap metrics (weighted averages)
    cont_hpd_overlap: float
    frag_hpd_overlap: float
    # Score: positive means frag is better than continuous
    hpd_difference: float
    # Behavioral characteristics (None in diagnostic mode)
    mean_speed: float | None = None
    mean_z_multiunit: float | None = None
    # Diagnostic-mode metrics
    mean_disagreeing_cells: float | None = None


def compute_multiunit_zscore(
    spike_counts: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Compute z-scored multiunit firing rate.

    Parameters
    ----------
    spike_counts : np.ndarray, shape (n_time, n_cells)
        Spike count matrix.

    Returns
    -------
    z_multiunit : np.ndarray, shape (n_time,)
        Z-scored multiunit firing rate.
    """
    multiunit_rate = spike_counts.sum(axis=1).astype(np.float64)
    # Handle case where std is 0
    z_multiunit: NDArray[np.float64] = zscore(multiunit_rate, nan_policy="omit")
    # Replace NaN with 0 (happens if all values are identical)
    z_multiunit = np.nan_to_num(z_multiunit, nan=0.0).astype(np.float64)
    return z_multiunit


def find_immobile_high_activity_periods(
    speed: NDArray[np.float64],
    z_multiunit: NDArray[np.float64],
    speed_threshold: float = SPEED_THRESHOLD,
    z_threshold: float = Z_MULTIUNIT_THRESHOLD,
    min_duration_bins: int = MIN_DURATION_BINS,
) -> list[tuple[int, int]]:
    """Find contiguous periods of immobility with high multiunit activity.

    Parameters
    ----------
    speed : np.ndarray, shape (n_time,)
        Animal speed in cm/s.
    z_multiunit : np.ndarray, shape (n_time,)
        Z-scored multiunit firing rate.
    speed_threshold : float
        Speed below which animal is considered immobile.
    z_threshold : float
        Z-score above which multiunit activity is considered high.
    min_duration_bins : int
        Minimum duration of event in time bins.

    Returns
    -------
    periods : list of (start_idx, end_idx) tuples
        List of contiguous periods meeting criteria.
    """
    # Create combined mask
    is_immobile = speed < speed_threshold
    is_high_activity = z_multiunit > z_threshold
    is_candidate = is_immobile & is_high_activity

    # Find contiguous regions
    labeled_array, n_labels = label(is_candidate)

    # Collect all periods and their durations for debugging
    all_durations = []
    periods = []
    for label_id in range(1, n_labels + 1):
        indices = np.where(labeled_array == label_id)[0]
        start_idx = int(indices[0])
        end_idx = int(indices[-1])

        # Check minimum duration
        duration = end_idx - start_idx + 1
        all_durations.append(duration)
        if duration >= min_duration_bins:
            periods.append((start_idx, end_idx))

    # Print debug info about period durations
    if all_durations:
        all_durations_arr = np.array(all_durations)
        print("  Period duration statistics (before filtering):")
        print(f"    Total periods found: {len(all_durations)}")
        print(f"    Duration range: {all_durations_arr.min()} - {all_durations_arr.max()} bins")
        print(f"    Mean duration: {all_durations_arr.mean():.1f} bins")
        print(f"    Median duration: {np.median(all_durations_arr):.1f} bins")
        print(f"    Periods >= {min_duration_bins} bins: {len(periods)}")

    return periods


def find_poor_diagnostic_periods(
    diagnostics: dict[str, NDArray[np.float64]],
    metric_name: str = DIAGNOSTIC_METRIC,
    cell_threshold: float = CELL_THRESHOLD,
    min_disagreeing_cells: int = MIN_DISAGREEING_CELLS,
    min_duration_bins: int = MIN_DURATION_BINS,
) -> list[tuple[int, int]]:
    """Find contiguous periods where multiple cells flag poor model fit.

    For each time bin, counts cells whose diagnostic metric crosses a threshold
    (below for HPD overlap/spike_prob, above for KL divergence). Periods where
    the count meets or exceeds ``min_disagreeing_cells`` are returned.

    Parameters
    ----------
    diagnostics : dict[str, np.ndarray]
        Diagnostics with per-cell metrics, each shape (n_time, n_cells).
    metric_name : str
        Which metric to threshold ('hpd_overlap', 'kl_divergence', 'spike_prob').
    cell_threshold : float
        Per-cell threshold for flagging disagreement.
    min_disagreeing_cells : int
        Minimum number of cells that must disagree at a time point.
    min_duration_bins : int
        Minimum contiguous duration in time bins.

    Returns
    -------
    periods : list of (start_idx, end_idx) tuples
        Contiguous periods meeting criteria.
    """
    metric = diagnostics[metric_name]  # (n_time, n_cells)

    # Flag per-cell disagreement: higher is worse for KL, lower is worse for others
    if metric_name == "kl_divergence":
        disagreeing = metric > cell_threshold
    else:
        disagreeing = metric < cell_threshold

    # NaN entries (no spike) are not disagreeing
    disagreeing = disagreeing & ~np.isnan(metric)

    # Count disagreeing cells per time bin
    n_disagreeing = disagreeing.sum(axis=1)  # (n_time,)

    # Find contiguous regions where enough cells disagree
    is_poor_fit = n_disagreeing >= min_disagreeing_cells
    labeled_array, n_labels = label(is_poor_fit)

    all_durations = []
    periods = []
    for label_id in range(1, n_labels + 1):
        indices = np.where(labeled_array == label_id)[0]
        start_idx = int(indices[0])
        end_idx = int(indices[-1])
        duration = end_idx - start_idx + 1
        all_durations.append(duration)
        if duration >= min_duration_bins:
            periods.append((start_idx, end_idx))

    if all_durations:
        all_durations_arr = np.array(all_durations)
        print("  Period duration statistics (before filtering):")
        print(f"    Total periods found: {len(all_durations)}")
        print(f"    Duration range: {all_durations_arr.min()} - {all_durations_arr.max()} bins")
        print(f"    Mean duration: {all_durations_arr.mean():.1f} bins")
        print(f"    Median duration: {np.median(all_durations_arr):.1f} bins")
        print(f"    Periods >= {min_duration_bins} bins: {len(periods)}")

    return periods


def compute_event_hpd_overlap(
    diagnostics: dict[str, NDArray[np.float64]],
    time: NDArray[np.float64],
    start_idx: int,
    end_idx: int,
    running_avg_window: float = RUNNING_AVG_WINDOW,
) -> float:
    """Compute weighted average HPD overlap for an event.

    Parameters
    ----------
    diagnostics : dict
        Diagnostics with 'hpd_overlap' key, shape (n_time, n_cells).
    time : np.ndarray
        Time values.
    start_idx, end_idx : int
        Event boundaries (inclusive).
    running_avg_window : float
        Window size for running average.

    Returns
    -------
    hpd_overlap : float
        Weighted average HPD overlap over the event.
    """
    window_slice = slice(start_idx, end_idx + 1)
    window_time = time[window_slice]
    metric_data = diagnostics["hpd_overlap"][window_slice]

    # Compute running average
    event_times = diagnostics.get("event_time")
    event_values = diagnostics.get("event_hpd_overlap")
    if event_times is not None and event_values is not None:
        event_times = np.asarray(event_times)
        time_start = time[start_idx]
        time_end = time[end_idx]
        event_mask = (event_times >= time_start) & (event_times <= time_end)
        running_avg, _ = compute_running_average(
            metric_data,
            window_time,
            window_size=running_avg_window,
            event_times=event_times[event_mask],
            event_values=np.asarray(event_values)[event_mask],
        )
    else:
        running_avg, _ = compute_running_average(
            metric_data, window_time, window_size=running_avg_window
        )

    # Return mean of running average (excludes NaN)
    return float(np.nanmean(running_avg))


def score_replay_candidates(
    periods: list[tuple[int, int]],
    time: NDArray[np.float64],
    spike_counts: NDArray[np.int64],
    continuous_diagnostics: dict[str, NDArray[np.float64]],
    contfrag_diagnostics: dict[str, NDArray[np.float64]],
    speed: NDArray[np.float64] | None = None,
    z_multiunit: NDArray[np.float64] | None = None,
    min_spikes: int = MIN_SPIKES,
    diagnostic_metric: str = DIAGNOSTIC_METRIC,
    cell_threshold: float = CELL_THRESHOLD,
) -> list[ReplayCandidate]:
    """Score candidate events by model difference in HPD overlap.

    Parameters
    ----------
    periods : list of (start_idx, end_idx) tuples
        Candidate periods.
    time : np.ndarray
        Time values.
    spike_counts : np.ndarray
        Spike count matrix.
    continuous_diagnostics : dict
        Diagnostics for continuous model.
    contfrag_diagnostics : dict
        Diagnostics for cont-frag model.
    speed : np.ndarray, optional
        Animal speed. If None, behavioral metrics are omitted.
    z_multiunit : np.ndarray, optional
        Z-scored multiunit rate. If None, behavioral metrics are omitted.
    min_spikes : int
        Minimum spikes required.
    diagnostic_metric : str
        Which metric to use for disagreeing cell count.
    cell_threshold : float
        Per-cell threshold for counting disagreeing cells.

    Returns
    -------
    candidates : list[ReplayCandidate]
        Scored candidates, sorted by hpd_difference (highest first).
    """
    candidates = []

    for start_idx, end_idx in periods:
        # Count spikes in event
        n_spikes = int(spike_counts[start_idx : end_idx + 1].sum())
        if n_spikes < min_spikes:
            continue

        # Behavioral characteristics (optional)
        mean_speed: float | None = None
        mean_z: float | None = None
        if speed is not None:
            mean_speed = float(np.mean(speed[start_idx : end_idx + 1]))
        if z_multiunit is not None:
            mean_z = float(np.mean(z_multiunit[start_idx : end_idx + 1]))

        # Compute HPD overlap for both models
        cont_hpd = compute_event_hpd_overlap(continuous_diagnostics, time, start_idx, end_idx)
        frag_hpd = compute_event_hpd_overlap(contfrag_diagnostics, time, start_idx, end_idx)

        # Score: positive means frag is better
        hpd_diff = frag_hpd - cont_hpd

        # Compute mean disagreeing cells in window (for the continuous model)
        metric_window = continuous_diagnostics[diagnostic_metric][start_idx : end_idx + 1]
        if diagnostic_metric == "kl_divergence":
            disagreeing_per_bin = (metric_window > cell_threshold) & ~np.isnan(metric_window)
        else:
            disagreeing_per_bin = (metric_window < cell_threshold) & ~np.isnan(metric_window)
        mean_disagreeing = float(np.mean(disagreeing_per_bin.sum(axis=1)))

        # Compute timing
        center_idx = (start_idx + end_idx) // 2
        time_start = float(time[start_idx])
        time_end = float(time[end_idx])
        duration_ms = (time_end - time_start) * 1000

        candidate = ReplayCandidate(
            start_idx=start_idx,
            end_idx=end_idx,
            center_idx=center_idx,
            time_start=time_start,
            time_end=time_end,
            duration_ms=duration_ms,
            n_spikes=n_spikes,
            cont_hpd_overlap=cont_hpd,
            frag_hpd_overlap=frag_hpd,
            hpd_difference=hpd_diff,
            mean_speed=mean_speed,
            mean_z_multiunit=mean_z,
            mean_disagreeing_cells=mean_disagreeing,
        )
        candidates.append(candidate)

    # Sort by HPD difference (highest = frag most better than continuous)
    candidates.sort(key=lambda c: c.hpd_difference, reverse=True)

    return candidates


def print_candidates(
    candidates: list[ReplayCandidate],
    n_top: int = N_TOP_CANDIDATES,
    diagnostic_mode: bool = False,
    speed_threshold: float = SPEED_THRESHOLD,
    z_threshold: float = Z_MULTIUNIT_THRESHOLD,
) -> None:
    """Print summary of top candidate events."""
    print(f"\n{'=' * 80}")
    if diagnostic_mode:
        print(f"Top {min(n_top, len(candidates))} Diagnostic-Flagged Events")
        print("  (Windows where per-cell diagnostics flag poor fit)")
    else:
        print(f"Top {min(n_top, len(candidates))} Immobile High-Activity Events")
        print("  (Cont-Frag HPD overlap better than Continuous)")
        print(f"Criteria: speed < {speed_threshold} cm/s, multiunit z-score > {z_threshold}")
    print(f"{'=' * 80}\n")

    for i, c in enumerate(candidates[:n_top]):
        print(f"Rank {i + 1}: idx={c.center_idx}")
        print(f"  Time: {c.time_start:.3f} - {c.time_end:.3f} s ({c.duration_ms:.0f} ms)")
        if c.mean_speed is not None and c.mean_z_multiunit is not None:
            print(f"  Speed: {c.mean_speed:.1f} cm/s, Z-multiunit: {c.mean_z_multiunit:.1f}")
        print(f"  N spikes: {c.n_spikes}")
        if c.mean_disagreeing_cells is not None:
            print(f"  Mean disagreeing cells: {c.mean_disagreeing_cells:.1f}")
        print(
            f"  HPD overlap: Continuous={c.cont_hpd_overlap:.3f}, "
            f"Cont-Frag={c.frag_hpd_overlap:.3f}"
        )
        print(f"  Difference (Frag - Cont): {c.hpd_difference:+.3f}")
        print()


def generate_preview_figures(
    candidates: list[ReplayCandidate],
    time: NDArray[np.float64],
    linear_position: NDArray[np.float64],
    continuous_results: Any,
    contfrag_results: Any,
    continuous_diagnostics: dict[str, NDArray[np.float64]],
    contfrag_diagnostics: dict[str, NDArray[np.float64]],
    spike_times: list[NDArray[np.float64]],
    spike_counts: NDArray[np.int64],
    place_field_peaks: NDArray[np.float64],
    data: dict[str, Any],
    n_top: int = N_TOP_CANDIDATES,
    output_dir: Path | None = None,
) -> None:
    """Generate preview figures for top candidates.

    Parameters
    ----------
    candidates : list[ReplayCandidate]
        Ranked candidate events.
    n_top : int
        Number of preview figures to generate.
    output_dir : Path, optional
        Output directory. If None, uses manuscript/figures/preview/.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "manuscript" / "figures" / "preview"

    output_dir.mkdir(parents=True, exist_ok=True)
    set_figure_defaults()

    for i, c in enumerate(candidates[:n_top]):
        print(f"Generating replay preview {i + 1}/{min(n_top, len(candidates))}...")

        # Use fixed window around center for consistent visualization
        window_start = max(0, c.center_idx - WINDOW_HALF_WIDTH)
        window_end = min(len(time), c.center_idx + WINDOW_HALF_WIDTH)
        time_slice_ind = slice(window_start, window_end)

        fig, axes = plot_model_comparison_with_posterior(
            time,
            linear_position,
            continuous_results,
            contfrag_results,
            continuous_diagnostics,
            contfrag_diagnostics,
            spike_times=spike_times,
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

        # Add title with event info
        speed_str = f", speed={c.mean_speed:.1f} cm/s" if c.mean_speed is not None else ""
        z_str = f", z={c.mean_z_multiunit:.1f}" if c.mean_z_multiunit is not None else ""
        disagree_str = (
            f", disagree={c.mean_disagreeing_cells:.1f} cells"
            if c.mean_disagreeing_cells is not None
            else ""
        )
        fig.suptitle(
            f"Rank {i + 1}: HPD diff={c.hpd_difference:+.3f}{disagree_str}{speed_str}{z_str}",
            fontsize=10,
        )

        save_figure(
            str(output_dir / f"replay_rank{i + 1:02d}_idx{c.center_idx}"),
            close=True,
        )

    print(f"\nPreview figures saved to {output_dir}/")


def main(
    generate_previews: bool = True,
    speed_threshold: float = SPEED_THRESHOLD,
    z_threshold: float = Z_MULTIUNIT_THRESHOLD,
    n_top: int = N_TOP_CANDIDATES,
    min_duration_bins: int = MIN_DURATION_BINS,
    diagnostic_mode: bool = False,
    diagnostic_metric: str = DIAGNOSTIC_METRIC,
    cell_threshold: float = CELL_THRESHOLD,
    min_disagreeing_cells: int = MIN_DISAGREEING_CELLS,
) -> list[ReplayCandidate]:
    """Run the window finding pipeline.

    Parameters
    ----------
    generate_previews : bool
        Whether to generate preview figures.
    speed_threshold : float
        Speed threshold for immobility (cm/s). Used in behavioral mode.
    z_threshold : float
        Z-score threshold for high multiunit activity. Used in behavioral mode.
    n_top : int
        Number of top candidates to report/visualize.
    min_duration_bins : int
        Minimum event duration in time bins.
    diagnostic_mode : bool
        If True, find windows using diagnostic metrics instead of behavioral filters.
    diagnostic_metric : str
        Which metric to use in diagnostic mode.
    cell_threshold : float
        Per-cell threshold for flagging disagreement in diagnostic mode.
    min_disagreeing_cells : int
        Minimum cells that must disagree at a time point in diagnostic mode.

    Returns
    -------
    candidates : list[ReplayCandidate]
        Ranked list of candidate events.
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

    # Get spike counts
    spike_counts = get_spike_counts(spike_times_list, time)

    # Compute diagnostics
    print("Computing diagnostics...")
    continuous_diagnostics = compute_model_diagnostics(
        continuous_model, continuous_results, spike_counts, time, spike_times=spike_times_list
    )
    contfrag_diagnostics = compute_model_diagnostics(
        contfrag_model, contfrag_results, spike_counts, time, spike_times=spike_times_list
    )

    # Find candidate periods
    speed_arr: NDArray[np.float64] | None = None
    z_multiunit: NDArray[np.float64] | None = None

    if diagnostic_mode:
        # Diagnostic-driven: threshold per-cell metrics, count disagreeing cells
        print(
            f"Finding periods where >= {min_disagreeing_cells} cells have "
            f"{diagnostic_metric} {'>' if diagnostic_metric == 'kl_divergence' else '<'} "
            f"{cell_threshold}..."
        )
        periods = find_poor_diagnostic_periods(
            continuous_diagnostics,
            metric_name=diagnostic_metric,
            cell_threshold=cell_threshold,
            min_disagreeing_cells=min_disagreeing_cells,
            min_duration_bins=min_duration_bins,
        )
    else:
        # Behavioral filtering: immobile + high multiunit activity
        speed_arr = position_info["head_speed"].values
        print("Computing multiunit z-score...")
        z_multiunit = compute_multiunit_zscore(spike_counts)
        print(
            f"Finding immobile (speed < {speed_threshold}) + "
            f"high-activity (z > {z_threshold}) periods..."
        )
        periods = find_immobile_high_activity_periods(
            speed_arr,
            z_multiunit,
            speed_threshold=speed_threshold,
            z_threshold=z_threshold,
            min_duration_bins=min_duration_bins,
        )

    print(f"  Found {len(periods)} candidate periods")

    # Score candidates by model difference
    print("Scoring candidates by HPD overlap difference...")
    candidates = score_replay_candidates(
        periods,
        time,
        spike_counts,
        continuous_diagnostics,
        contfrag_diagnostics,
        speed=speed_arr,
        z_multiunit=z_multiunit,
        diagnostic_metric=diagnostic_metric,
        cell_threshold=cell_threshold,
    )
    print(f"  {len(candidates)} candidates after filtering")

    # Print results
    print_candidates(
        candidates,
        n_top=n_top,
        diagnostic_mode=diagnostic_mode,
        speed_threshold=speed_threshold,
        z_threshold=z_threshold,
    )

    # Generate preview figures if requested
    if generate_previews and candidates:
        # Extract place fields for raster sorting
        place_fields, position_bins = extract_place_fields(continuous_model)
        if np.any(np.all(np.isnan(place_fields), axis=1)):
            warnings.warn(
                "Some cells have all-NaN place fields; peak positions may be incorrect",
                stacklevel=2,
            )
        place_field_peaks = position_bins[np.nanargmax(place_fields, axis=1)]

        generate_preview_figures(
            candidates,
            time,
            linear_position,
            continuous_results,
            contfrag_results,
            continuous_diagnostics,
            contfrag_diagnostics,
            spike_times_list,
            spike_counts,
            place_field_peaks,
            data,
            n_top=n_top,
        )

    return candidates


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Find immobile high-activity windows where cont-frag outperforms continuous"
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Skip generating preview figures",
    )
    parser.add_argument(
        "--speed-threshold",
        type=float,
        default=SPEED_THRESHOLD,
        help=f"Speed threshold for immobility (cm/s). Default: {SPEED_THRESHOLD}",
    )
    parser.add_argument(
        "--z-threshold",
        type=float,
        default=Z_MULTIUNIT_THRESHOLD,
        help=f"Z-score threshold for high multiunit activity. Default: {Z_MULTIUNIT_THRESHOLD}",
    )
    parser.add_argument(
        "--n-top",
        type=int,
        default=N_TOP_CANDIDATES,
        help=f"Number of top candidates to report/visualize. Default: {N_TOP_CANDIDATES}",
    )
    parser.add_argument(
        "--min-duration",
        type=int,
        default=MIN_DURATION_BINS,
        help=f"Minimum event duration in time bins. Default: {MIN_DURATION_BINS}",
    )
    parser.add_argument(
        "--diagnostic-mode",
        action="store_true",
        help="Find windows using diagnostic metrics instead of behavioral filters",
    )
    parser.add_argument(
        "--diagnostic-metric",
        type=str,
        default=DIAGNOSTIC_METRIC,
        choices=["hpd_overlap", "kl_divergence", "spike_prob"],
        help=f"Metric to use in diagnostic mode. Default: {DIAGNOSTIC_METRIC}",
    )
    parser.add_argument(
        "--cell-threshold",
        type=float,
        default=CELL_THRESHOLD,
        help=f"Per-cell threshold for flagging disagreement. Default: {CELL_THRESHOLD}",
    )
    parser.add_argument(
        "--min-disagreeing-cells",
        type=int,
        default=MIN_DISAGREEING_CELLS,
        help=f"Min cells that must disagree at a time point. Default: {MIN_DISAGREEING_CELLS}",
    )
    args = parser.parse_args()

    main(
        generate_previews=not args.no_preview,
        speed_threshold=args.speed_threshold,
        z_threshold=args.z_threshold,
        n_top=args.n_top,
        min_duration_bins=args.min_duration,
        diagnostic_mode=args.diagnostic_mode,
        diagnostic_metric=args.diagnostic_metric,
        cell_threshold=args.cell_threshold,
        min_disagreeing_cells=args.min_disagreeing_cells,
    )
