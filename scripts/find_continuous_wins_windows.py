"""Find windows where Continuous model clearly outperforms Continuous-Fragmented.

This script identifies time windows where the Continuous model has better
diagnostic metrics (higher HPD overlap) than the Continuous-Fragmented model,
regardless of movement state.

These cases demonstrate that the simpler continuous transition model can
provide a better fit in certain regimes.

Usage:
    python scripts/find_continuous_wins_windows.py
    python scripts/find_continuous_wins_windows.py --n-top 15
    python scripts/find_continuous_wins_windows.py --no-preview

Outputs:
    - Prints ranked list of candidate events
    - Optionally generates preview figures in manuscript/figures/preview/continuous_wins/
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import label

from statespacecheck_paper.analysis import PerCellDiagnostics
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

# Window parameters
WINDOW_HALF_WIDTH = 100  # Half-width in time points for visualization
RUNNING_AVG_WINDOW = 0.020  # 20ms window for running average

# Filtering: require HPD overlap difference to be meaningful
MIN_HPD_DIFF = 0.05  # Minimum HPD difference (cont better than frag)
MIN_DURATION_BINS = 5  # Minimum event duration in time bins
MIN_SPIKES = 10  # Minimum spikes in event

# Number of top candidates to report/visualize
N_TOP_CANDIDATES = 10


@dataclass
class ContinuousWinsCandidate:
    """A candidate event where Continuous model outperforms Cont-Frag."""

    start_idx: int
    end_idx: int
    center_idx: int
    time_start: float
    time_end: float
    duration_ms: float
    # Event characteristics
    mean_speed: float
    n_spikes: int
    # HPD overlap metrics (running averages over event)
    cont_hpd_overlap: float
    frag_hpd_overlap: float
    # Score: positive means continuous is better
    hpd_difference: float


def find_continuous_better_periods(
    continuous_diagnostics: PerCellDiagnostics,
    contfrag_diagnostics: PerCellDiagnostics,
    time: NDArray[np.float64],
    min_hpd_diff: float = MIN_HPD_DIFF,
    min_duration_bins: int = MIN_DURATION_BINS,
    running_avg_window: float = RUNNING_AVG_WINDOW,
) -> list[tuple[int, int]]:
    """Find contiguous periods where Continuous HPD overlap exceeds Cont-Frag.

    Uses running-averaged HPD overlap to find sustained periods where the
    Continuous model outperforms the Continuous-Fragmented model.

    Parameters
    ----------
    continuous_diagnostics : dict
        Diagnostics for continuous model with 'hpd_overlap' key.
    contfrag_diagnostics : dict
        Diagnostics for cont-frag model with 'hpd_overlap' key.
    time : np.ndarray, shape (n_time,)
        Time values.
    min_hpd_diff : float
        Minimum HPD overlap difference (cont - frag) to qualify.
    min_duration_bins : int
        Minimum duration of event in time bins.
    running_avg_window : float
        Window size for running average in seconds.

    Returns
    -------
    periods : list of (start_idx, end_idx) tuples
        List of contiguous periods meeting criteria.
    """
    # Compute running average of HPD overlap for both models
    # HPD overlap shape is (n_time, n_cells) — average across cells first
    cont_hpd = continuous_diagnostics.hpd_overlap  # (n_time, n_cells)
    frag_hpd = contfrag_diagnostics.hpd_overlap  # (n_time, n_cells)

    cont_avg, _ = compute_running_average(
        cont_hpd,
        time,
        window_size=running_avg_window,
        event_times=continuous_diagnostics.event_time,
        event_values=continuous_diagnostics.event_hpd_overlap,
    )
    frag_avg, _ = compute_running_average(
        frag_hpd,
        time,
        window_size=running_avg_window,
        event_times=contfrag_diagnostics.event_time,
        event_values=contfrag_diagnostics.event_hpd_overlap,
    )

    # Find where continuous is better by at least min_hpd_diff
    diff = cont_avg - frag_avg
    is_cont_better = diff > min_hpd_diff

    # Handle NaN values
    is_cont_better = is_cont_better & ~np.isnan(diff)

    # Find contiguous regions
    labeled_array, n_labels = label(is_cont_better)

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
    diagnostics: PerCellDiagnostics,
    time: NDArray[np.float64],
    start_idx: int,
    end_idx: int,
    running_avg_window: float = RUNNING_AVG_WINDOW,
) -> float:
    """Compute mean running-averaged HPD overlap for an event.

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
        Mean running-averaged HPD overlap over the event.
    """
    window_slice = slice(start_idx, end_idx + 1)
    window_time = time[window_slice]
    metric_data = diagnostics.hpd_overlap[window_slice]

    event_times = diagnostics.event_time
    event_values = diagnostics.event_hpd_overlap
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

    return float(np.nanmean(running_avg))


def score_continuous_wins(
    periods: list[tuple[int, int]],
    time: NDArray[np.float64],
    speed: NDArray[np.float64],
    spike_counts: NDArray[np.int64],
    continuous_diagnostics: PerCellDiagnostics,
    contfrag_diagnostics: PerCellDiagnostics,
    min_spikes: int = MIN_SPIKES,
) -> list[ContinuousWinsCandidate]:
    """Score candidate events by how much Continuous outperforms Cont-Frag.

    Parameters
    ----------
    periods : list of (start_idx, end_idx) tuples
        Candidate periods.
    time : np.ndarray
        Time values.
    speed : np.ndarray
        Animal speed.
    spike_counts : np.ndarray
        Spike count matrix.
    continuous_diagnostics : dict
        Diagnostics for continuous model.
    contfrag_diagnostics : dict
        Diagnostics for cont-frag model.
    min_spikes : int
        Minimum spikes required.

    Returns
    -------
    candidates : list[ContinuousWinsCandidate]
        Scored candidates, sorted by hpd_difference (highest first = continuous most better).
    """
    candidates = []

    for start_idx, end_idx in periods:
        n_spikes = int(spike_counts[start_idx : end_idx + 1].sum())
        if n_spikes < min_spikes:
            continue

        mean_speed = float(np.mean(speed[start_idx : end_idx + 1]))

        cont_hpd = compute_event_hpd_overlap(continuous_diagnostics, time, start_idx, end_idx)
        frag_hpd = compute_event_hpd_overlap(contfrag_diagnostics, time, start_idx, end_idx)

        # Score: positive means continuous is better
        hpd_diff = cont_hpd - frag_hpd

        center_idx = (start_idx + end_idx) // 2
        time_start = float(time[start_idx])
        time_end = float(time[end_idx])
        duration_ms = (time_end - time_start) * 1000

        candidate = ContinuousWinsCandidate(
            start_idx=start_idx,
            end_idx=end_idx,
            center_idx=center_idx,
            time_start=time_start,
            time_end=time_end,
            duration_ms=duration_ms,
            mean_speed=mean_speed,
            n_spikes=n_spikes,
            cont_hpd_overlap=cont_hpd,
            frag_hpd_overlap=frag_hpd,
            hpd_difference=hpd_diff,
        )
        candidates.append(candidate)

    # Sort by HPD difference (highest = continuous most better)
    candidates.sort(key=lambda c: c.hpd_difference, reverse=True)

    return candidates


def print_candidates(
    candidates: list[ContinuousWinsCandidate],
    n_top: int = N_TOP_CANDIDATES,
) -> None:
    """Print summary of top candidate events."""
    print(f"\n{'=' * 80}")
    print(f"Top {min(n_top, len(candidates))} Windows Where Continuous Outperforms Cont-Frag")
    print(f"{'=' * 80}\n")

    for i, c in enumerate(candidates[:n_top]):
        print(f"Rank {i + 1}: idx={c.center_idx}")
        print(f"  Time: {c.time_start:.3f} - {c.time_end:.3f} s ({c.duration_ms:.0f} ms)")
        print(f"  Speed: {c.mean_speed:.1f} cm/s, N spikes: {c.n_spikes}")
        print(
            f"  HPD overlap: Continuous={c.cont_hpd_overlap:.3f}, "
            f"Cont-Frag={c.frag_hpd_overlap:.3f}"
        )
        print(f"  Difference (Cont - Frag): {c.hpd_difference:+.3f}")
        print()


def generate_preview_figures(
    candidates: list[ContinuousWinsCandidate],
    time: NDArray[np.float64],
    linear_position: NDArray[np.float64],
    continuous_results: Any,
    contfrag_results: Any,
    continuous_diagnostics: PerCellDiagnostics,
    contfrag_diagnostics: PerCellDiagnostics,
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
    candidates : list[ContinuousWinsCandidate]
        Ranked candidate events.
    n_top : int
        Number of preview figures to generate.
    output_dir : Path, optional
        Output directory. If None, uses manuscript/figures/preview/continuous_wins/.
    """
    if output_dir is None:
        output_dir = (
            Path(__file__).parent.parent / "manuscript" / "figures" / "preview" / "continuous_wins"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    set_figure_defaults()

    for i, c in enumerate(candidates[:n_top]):
        print(f"Generating continuous-wins preview {i + 1}/{min(n_top, len(candidates))}...")

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
            show_running_average=True,
            running_average_window=0.020,
        )

        fig.suptitle(
            f"Continuous wins rank {i + 1}: HPD diff={c.hpd_difference:+.3f}, "
            f"speed={c.mean_speed:.1f} cm/s",
            fontsize=10,
        )

        save_figure(
            str(output_dir / f"cont_wins_rank{i + 1:02d}_idx{c.center_idx}"),
            close=True,
        )

    print(f"\nPreview figures saved to {output_dir}/")


def main(
    generate_previews: bool = True,
    min_hpd_diff: float = MIN_HPD_DIFF,
    n_top: int = N_TOP_CANDIDATES,
    min_duration_bins: int = MIN_DURATION_BINS,
) -> list[ContinuousWinsCandidate]:
    """Run the continuous-wins window finding pipeline.

    Parameters
    ----------
    generate_previews : bool
        Whether to generate preview figures.
    min_hpd_diff : float
        Minimum HPD overlap difference (cont - frag) to qualify.
    n_top : int
        Number of top candidates to report/visualize.
    min_duration_bins : int
        Minimum event duration in time bins.

    Returns
    -------
    candidates : list[ContinuousWinsCandidate]
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
    speed = position_info["head_speed"].values
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

    # Find periods where continuous is better
    print(f"Finding periods where Continuous HPD overlap exceeds Cont-Frag by > {min_hpd_diff}...")
    periods = find_continuous_better_periods(
        continuous_diagnostics,
        contfrag_diagnostics,
        time,
        min_hpd_diff=min_hpd_diff,
        min_duration_bins=min_duration_bins,
    )
    print(f"  Found {len(periods)} candidate periods")

    # Score candidates
    print("Scoring candidates...")
    candidates = score_continuous_wins(
        periods,
        time,
        speed,
        spike_counts,
        continuous_diagnostics,
        contfrag_diagnostics,
    )
    print(f"  {len(candidates)} candidates after filtering")

    # Print results
    print_candidates(candidates, n_top=n_top)

    # Generate preview figures if requested
    if generate_previews and candidates:
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
        description="Find windows where Continuous model outperforms Continuous-Fragmented"
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Skip generating preview figures",
    )
    parser.add_argument(
        "--min-hpd-diff",
        type=float,
        default=MIN_HPD_DIFF,
        help=f"Minimum HPD overlap difference (cont - frag). Default: {MIN_HPD_DIFF}",
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
    args = parser.parse_args()

    main(
        generate_previews=not args.no_preview,
        min_hpd_diff=args.min_hpd_diff,
        n_top=args.n_top,
        min_duration_bins=args.min_duration,
    )
