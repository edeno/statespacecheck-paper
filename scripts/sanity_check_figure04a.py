"""Sanity check: per-spike diagnostic visualizations for Figure 4a window.

For each spike event in the Figure 4a time window, plots:
- Predictive posterior with 95% HPD shading
- Normalized per-cell Poisson likelihood with 95% HPD shading
- Un-normalized place field rate curve

Annotated with true position, metric values (HPD overlap, KL divergence,
spike probability rank).  Produces plots for both Continuous and Cont-Frag
models, saved into separate folders under manuscript/figures/preview/sanity_check_4a/.

Spikes are sampled at quantiles of KL divergence so the output covers the
full range from best to worst model fit.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.stats import poisson

from statespacecheck_paper.analysis import PerCellDiagnostics
from statespacecheck_paper.load_local_data import load_neural_recording_from_files
from statespacecheck_paper.paths import ANIMAL_DATE_EPOCH, DATA_PATH
from statespacecheck_paper.plotting import compute_hpd_region
from statespacecheck_paper.real_data_analysis import (
    compute_model_diagnostics,
    create_decoder_environment,
    extract_place_fields,
    fit_decoder_models,
    get_spike_counts,
)
from statespacecheck_paper.simulation import normalize
from statespacecheck_paper.style import COLORS, set_figure_defaults

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
__all__ = ["ANIMAL_DATE_EPOCH", "DATA_PATH"]

# Time index for the Figure 4a replay event (matches generate_figure04.py)
WINDOW_CENTER = 177301
WINDOW_HALF_WIDTH = 50

# Number of spikes to sample at evenly-spaced KL quantiles
N_QUANTILE_SPIKES = 20

OUTPUT_ROOT = Path(__file__).parent.parent / "figures" / "preview" / "sanity_check_4a"


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _shade_hpd(
    ax: Axes,
    x: NDArray[np.float64],
    dist: NDArray[np.floating],
    color: str,
    coverage: float = 0.95,
) -> None:
    """Shade the HPD region of *dist* on *ax*."""
    mask = compute_hpd_region(x, dist, coverage=coverage)
    ax.fill_between(
        x,
        0,
        dist,
        where=mask.tolist(),
        color=color,
        alpha=0.25,
        label=f"{coverage:.0%} HPD",
    )


def _add_state_boundary(
    ax: Axes,
    n_bins_per_state: int,
) -> None:
    """Draw vertical separator between state halves for ContFrag models."""
    boundary = n_bins_per_state - 0.5
    ax.axvline(boundary, color="gray", ls=":", lw=0.8, alpha=0.6)


def _draw_track_on_axis(
    ax: Axes,
    track_graph: nx.Graph,
    edge_order: list[tuple[int, int]],
    edge_spacing: float | list[float],
    position_bins: NDArray[np.float64],
    n_bins_per_state: int | None,
) -> None:
    """Draw linearized track graph along the x-axis (bin-index space).

    For multi-state models, draws a copy of the track in each state half.
    The track is drawn at the bottom of the axes.
    """
    import matplotlib

    cmap = matplotlib.colormaps.get_cmap("tab10")

    n_edges = len(edge_order)
    if isinstance(edge_spacing, int | float):
        edge_spacing_list = [float(edge_spacing)] * (n_edges - 1)
    else:
        edge_spacing_list = list(edge_spacing)

    n_states = 1 if n_bins_per_state is None else len(position_bins) // n_bins_per_state
    bins_per_state = len(position_bins) if n_bins_per_state is None else n_bins_per_state

    y_bottom = ax.get_ylim()[0]

    for state in range(n_states):
        offset = state * bins_per_state
        state_pos_bins = position_bins[offset : offset + bins_per_state]

        # Build cm -> bin-index mapping for this state via interpolation
        cm_start = 0.0
        for edge_ind, edge in enumerate(edge_order):
            edge_color = cmap(edge_ind % 10)
            cm_end = cm_start + track_graph.edges[edge]["distance"]

            # Convert cm endpoints to bin indices within this state half
            bin_start = offset + float(np.argmin(np.abs(state_pos_bins - cm_start)))
            bin_end = offset + float(np.argmin(np.abs(state_pos_bins - cm_end)))

            ax.plot(
                (bin_start, bin_end),
                (y_bottom, y_bottom),
                color=edge_color,
                lw=3,
                solid_capstyle="butt",
                clip_on=False,
                zorder=7,
            )

            if edge_ind < len(edge_spacing_list):
                cm_start = cm_end + edge_spacing_list[edge_ind]
            else:
                cm_start = cm_end


def _find_true_pos_bin_indices(
    true_pos: float,
    position_bins: NDArray[np.float64],
    n_bins_per_state: int | None,
) -> list[int]:
    """Find bin index closest to true_pos in each state half.

    For single-state models, returns one index. For multi-state models,
    returns one index per state (e.g., [idx_continuous, idx_fragmented]).
    """
    if n_bins_per_state is None:
        # Single state: find closest bin
        return [int(np.argmin(np.abs(position_bins - true_pos)))]

    n_states = len(position_bins) // n_bins_per_state
    indices = []
    for s in range(n_states):
        start = s * n_bins_per_state
        end = start + n_bins_per_state
        state_bins = position_bins[start:end]
        local_idx = int(np.argmin(np.abs(state_bins - true_pos)))
        indices.append(start + local_idx)
    return indices


def plot_spike_sanity(
    bin_indices: NDArray[np.intp],
    predictive: NDArray[np.floating],
    likelihood_norm: NDArray[np.floating],
    place_field_rate: NDArray[np.floating],
    true_pos_indices: list[int],
    hpd_val: float,
    kl_val: float,
    sp_val: float,
    time_idx: int,
    cell_idx: int,
    model_name: str,
    n_bins_per_state: int | None = None,
    state_labels: list[str] | None = None,
    track_graph: nx.Graph | None = None,
    edge_order: list[tuple[int, int]] | None = None,
    edge_spacing: float | list[float] = 0.0,
    position_bins: NDArray[np.float64] | None = None,
) -> Figure:
    """Create a 3-panel sanity check figure for one spike event.

    Parameters
    ----------
    bin_indices : np.ndarray, shape (n_bins,)
        Integer bin indices (0 to n_total_bins-1). Used as x-axis.
    predictive : np.ndarray, shape (n_bins,)
        Predictive posterior distribution.
    likelihood_norm : np.ndarray, shape (n_bins,)
        Normalized Poisson likelihood P(k=1|x).
    place_field_rate : np.ndarray, shape (n_bins,)
        Un-normalized place field rate for this cell.
    true_pos_indices : list[int]
        Bin indices of the true position, one per state. For single-state
        models this has one entry; for ContFrag it has two.
    hpd_val : float
        HPD overlap metric value.
    kl_val : float
        KL divergence metric value.
    sp_val : float
        Spike probability rank value.
    time_idx : int
        Global time index for this spike.
    cell_idx : int
        Cell index for this spike.
    model_name : str
        Model label (e.g. "Continuous" or "Cont-Frag").
    n_bins_per_state : int or None
        Number of bins per state (for drawing ContFrag boundary). None for
        single-state models.
    state_labels : list[str] or None
        Labels for each state half (e.g. ["Continuous", "Fragmented"]).

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # Use float x-axis for HPD region computation (needs uniform spacing)
    x = bin_indices.astype(np.float64)

    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True, constrained_layout=True)

    panels = [
        (axes[0], predictive, COLORS.predictive, "Predictive posterior"),
        (axes[1], likelihood_norm, COLORS.likelihood, "Likelihood (norm)"),
        (axes[2], place_field_rate, "k", "Place field rate"),
    ]

    for i, (ax, data, color, ylabel) in enumerate(panels):
        ax.plot(x, data, color=color, lw=1.2)
        if i < 2:  # HPD shading for predictive and likelihood only
            _shade_hpd(ax, x, data, color)
        for tp_idx in true_pos_indices:
            ax.axvline(
                tp_idx,
                color=COLORS["ground_truth"],
                ls="--",
                lw=1.0,
                label="true pos" if (i == 0 and tp_idx == true_pos_indices[0]) else None,
            )
        if n_bins_per_state is not None:
            _add_state_boundary(ax, n_bins_per_state)
        ax.set_ylabel(ylabel)

        # Draw linearized track graph at bottom of each panel
        if track_graph is not None and edge_order is not None and position_bins is not None:
            _draw_track_on_axis(
                ax, track_graph, edge_order, edge_spacing, position_bins, n_bins_per_state
            )

    # Add state labels at top of first panel
    if state_labels is not None and n_bins_per_state is not None:
        ax_top = axes[0]
        for s, label in enumerate(state_labels):
            center = s * n_bins_per_state + n_bins_per_state / 2
            ax_top.text(
                center,
                ax_top.get_ylim()[1] * 0.95,
                label,
                ha="center",
                va="top",
                fontsize=7,
                fontstyle="italic",
                color="gray",
            )

    axes[0].legend(fontsize=7, loc="upper right")
    axes[2].set_xlabel("State bin index")

    fig.suptitle(
        f"{model_name}  |  t={time_idx}  cell={cell_idx}\n"
        f"HPD overlap={hpd_val:.4f}   KL={kl_val:.4f}   spike prob={sp_val:.4f}",
        fontsize=9,
    )

    return fig


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def _get_place_fields_posterior_and_bins(
    model: Any,
    results: Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], int | None]:
    """Extract place fields, predictive posterior, position bins, and state info.

    Returns
    -------
    place_fields : (n_cells, n_interior_bins)
    predictive_posterior : (n_time, n_interior_bins)
    position_bins : (n_interior_bins,)
        Physical position bin centers (cm). For multi-state models, position
        bins are repeated for each state.
    n_bins_per_state : int or None
        Number of interior bins per state. None for single-state models.
    """
    # Concatenate place fields across observation models, filter to interior
    all_pf = []
    all_pos_bins = []
    for obs in model.observation_models:
        pf, pos_bins = extract_place_fields(
            model,
            environment_name=obs.environment_name,
            encoding_group=obs.encoding_group,
        )
        all_pf.append(pf)
        all_pos_bins.append(pos_bins)

    place_fields = np.concatenate(all_pf, axis=1)
    position_bins = np.concatenate(all_pos_bins)

    interior_mask = model.is_track_interior_state_bins_
    place_fields = place_fields[:, interior_mask]
    position_bins = position_bins[interior_mask]

    predictive_posterior = results.predictive_posterior.dropna(dim="state_bins").values
    predictive_posterior = np.asarray(predictive_posterior)

    # Determine bins-per-state for multi-state models
    n_obs = len(model.observation_models)
    n_bins_per_state: int | None = None
    if n_obs > 1:
        assert place_fields.shape[1] % n_obs == 0, (
            f"n_bins {place_fields.shape[1]} not divisible by n_obs {n_obs}"
        )
        n_bins_per_state = place_fields.shape[1] // n_obs

    return place_fields, predictive_posterior, position_bins, n_bins_per_state


def _select_spikes(
    metric_values: NDArray[np.float64],
    spike_time_ind: NDArray[np.intp],
    spike_cell_ind: NDArray[np.intp],
    n_spikes: int = N_QUANTILE_SPIKES,
    mode: Literal["quantiles", "top", "bottom"] = "quantiles",
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Select spike events based on a diagnostic metric.

    Parameters
    ----------
    metric_values : (n_time, n_cells) diagnostic metric array.
    spike_time_ind, spike_cell_ind : indices of spike events.
    n_spikes : number of spikes to return.
    mode : selection strategy:
        - "quantiles": evenly spaced across the full range
        - "top": highest values
        - "bottom": lowest values
    """
    vals = metric_values[spike_time_ind, spike_cell_ind]

    valid = ~np.isnan(vals)
    valid_indices = np.where(valid)[0]
    vals_valid = vals[valid_indices]

    order = np.argsort(vals_valid)
    n_pick = min(n_spikes, len(order))

    if mode == "quantiles":
        pick = np.unique(np.linspace(0, len(order) - 1, n_pick, dtype=int))
    elif mode == "top":
        pick = order[-n_pick:]
    elif mode == "bottom":
        pick = order[:n_pick]

    selected = valid_indices[pick]
    return spike_time_ind[selected], spike_cell_ind[selected]


def _select_spikes_joint(
    diagnostics: PerCellDiagnostics,
    spike_time_ind: NDArray[np.intp],
    spike_cell_ind: NDArray[np.intp],
    primary_metric: str,
    primary_mode: Literal["top", "bottom"] = "bottom",
    filter_metric: str | None = None,
    filter_quantile_range: tuple[float, float] = (0.0, 1.0),
    n_spikes: int = N_QUANTILE_SPIKES,
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Select spikes by primary metric after filtering by another metric's range.

    Parameters
    ----------
    diagnostics : full diagnostics dict
    spike_time_ind, spike_cell_ind : spike event indices
    primary_metric : metric to rank/select by
    primary_mode : "top" or "bottom" of primary metric
    filter_metric : if set, restrict to spikes within quantile range of this metric
    filter_quantile_range : (lo, hi) quantiles for filter_metric (e.g., (0.0, 0.5))
    n_spikes : number to return
    """
    primary = getattr(diagnostics, primary_metric)[spike_time_ind, spike_cell_ind]
    valid = ~np.isnan(primary)

    if filter_metric is not None:
        filt = getattr(diagnostics, filter_metric)[spike_time_ind, spike_cell_ind]
        valid &= ~np.isnan(filt)
        filt_valid = filt[valid]
        lo = np.quantile(filt_valid, filter_quantile_range[0])
        hi = np.quantile(filt_valid, filter_quantile_range[1])
        valid &= (filt >= lo) & (filt <= hi)

    valid_indices = np.where(valid)[0]
    if len(valid_indices) == 0:
        return spike_time_ind[:0], spike_cell_ind[:0]

    primary_valid = primary[valid_indices]
    order = np.argsort(primary_valid)
    n_pick = min(n_spikes, len(order))

    if primary_mode == "bottom":
        pick = order[:n_pick]
    else:
        pick = order[-n_pick:]

    selected = valid_indices[pick]
    return spike_time_ind[selected], spike_cell_ind[selected]


def generate_sanity_plots(
    model: Any,
    results: Any,
    spike_counts: NDArray[np.int64],
    diagnostics: PerCellDiagnostics,
    linear_position: NDArray[np.float64],
    model_name: str,
    output_dir: Path,
    time_slice: slice,
    select_metric: str = "kl_divergence",
    select_mode: Literal["quantiles", "top", "bottom"] = "quantiles",
    filter_metric: str | None = None,
    filter_quantile_range: tuple[float, float] = (0.0, 1.0),
    track_graph: nx.Graph | None = None,
    edge_order: list[tuple[int, int]] | None = None,
    edge_spacing: float | list[float] = 0.0,
) -> None:
    """Generate per-spike sanity check plots for one model.

    Parameters
    ----------
    model : fitted decoder model
    results : xr.Dataset from model.predict()
    spike_counts : (n_time, n_cells)
    diagnostics : dict with hpd_overlap, kl_divergence, spike_prob
    linear_position : (n_time,) true position in physical units (cm)
    model_name : label for titles
    output_dir : where to save PNGs
    time_slice : slice selecting the Figure 4a window
    select_metric : diagnostic key to select spikes by
    select_mode : "quantiles", "top", or "bottom"
    filter_metric : if set, restrict to spikes within quantile range of this metric
    filter_quantile_range : (lo, hi) quantile range for filter_metric
    track_graph : networkx.Graph, optional
        Track graph for linearized track visualization on x-axis.
    edge_order : list[tuple[int, int]], optional
        Edge ordering for linearization.
    edge_spacing : float or list[float], default 0.0
        Spacing between track edges.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    place_fields, predictive_posterior, position_bins, n_bins_per_state = (
        _get_place_fields_posterior_and_bins(model, results)
    )

    n_total_bins = predictive_posterior.shape[1]
    bin_indices = np.arange(n_total_bins, dtype=np.intp)

    # State labels for multi-state models
    state_labels: list[str] | None = None
    if n_bins_per_state is not None:
        n_states = n_total_bins // n_bins_per_state
        state_labels = [
            getattr(obs, "state_name", f"State {i}")
            for i, obs in enumerate(model.observation_models[:n_states])
        ]

    # Restrict to window
    sc_window = spike_counts[time_slice]
    spike_t_local, spike_c = np.nonzero(sc_window)
    spike_t_global = spike_t_local + time_slice.start

    # Select spike subset
    if filter_metric is not None:
        sel_t, sel_c = _select_spikes_joint(
            diagnostics,
            spike_t_global.astype(np.intp),
            spike_c.astype(np.intp),
            primary_metric=select_metric,
            primary_mode="bottom" if select_mode == "bottom" else "top",
            filter_metric=filter_metric,
            filter_quantile_range=filter_quantile_range,
        )
    else:
        sel_t, sel_c = _select_spikes(
            getattr(diagnostics, select_metric),
            spike_t_global.astype(np.intp),
            spike_c.astype(np.intp),
            mode=select_mode,
        )

    print(f"  Generating {len(sel_t)} plots for {model_name}...")

    for t_idx, c_idx in zip(sel_t, sel_c, strict=True):
        pred = predictive_posterior[t_idx]
        rate = place_fields[c_idx]  # (n_bins,) un-normalized rate

        # Poisson likelihood P(k=1 | lambda), normalized over bins
        # (matches analysis.py:compute_per_cell_diagnostics_from_rates)
        lik_raw = poisson.pmf(k=1, mu=rate)  # (n_bins,)
        lik_norm = normalize(lik_raw, axis=0)  # normalize over bins

        hpd_val = float(diagnostics.hpd_overlap[t_idx, c_idx])
        kl_val = float(diagnostics.kl_divergence[t_idx, c_idx])
        sp_val = float(diagnostics.spike_prob[t_idx, c_idx])

        true_pos = float(linear_position[t_idx])
        true_pos_indices = _find_true_pos_bin_indices(true_pos, position_bins, n_bins_per_state)

        fig = plot_spike_sanity(
            bin_indices=bin_indices,
            predictive=pred,
            likelihood_norm=lik_norm,
            place_field_rate=rate,
            true_pos_indices=true_pos_indices,
            hpd_val=hpd_val,
            kl_val=kl_val,
            sp_val=sp_val,
            time_idx=int(t_idx),
            cell_idx=int(c_idx),
            model_name=model_name,
            n_bins_per_state=n_bins_per_state,
            state_labels=state_labels,
            track_graph=track_graph,
            edge_order=edge_order,
            edge_spacing=edge_spacing,
            position_bins=position_bins,
        )

        fname = f"t{t_idx:06d}_c{c_idx:03d}_kl{kl_val:.2f}.png"
        fig.savefig(output_dir / fname, dpi=150)  # preview only
        plt.close(fig)

    print(f"  Saved to {output_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full sanity check pipeline."""
    # Check dependencies
    try:
        import non_local_detector  # noqa: F401
    except ImportError:
        warnings.warn(
            "non_local_detector not available. Install to run this script.",
            stacklevel=2,
        )
        return

    set_figure_defaults(context="paper")

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

    position_info = data["position_info"]
    time = position_info.index.values
    position = position_info[["head_position_x", "head_position_y"]].values
    linear_position: NDArray[np.float64] = position_info["linear_position"].values
    spike_times_list: list[Any] = list(data["spike_times"])

    # Fit models
    print("Fitting models...")
    continuous_model, contfrag_model = fit_decoder_models(
        position=position,
        spike_times=spike_times_list,
        time=time,
        environment=env,
    )

    # Decode
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

    # Spike counts
    spike_counts = get_spike_counts(spike_times_list, time)

    # Diagnostics
    print("Computing diagnostics...")
    continuous_diag = compute_model_diagnostics(
        continuous_model, continuous_results, spike_counts, time, spike_times=spike_times_list
    )
    contfrag_diag = compute_model_diagnostics(
        contfrag_model, contfrag_results, spike_counts, time, spike_times=spike_times_list
    )

    # Use full session to find rare low-HPD/low-KL events
    n_time = len(time)
    time_slice = slice(0, n_time)

    # --- Selection config ---
    # HPD overlap near zero (lowest values, no KL filter)
    select_metric = "hpd_overlap"
    select_mode: Literal["quantiles", "top", "bottom"] = "bottom"
    filter_metric: str | None = None
    filter_quantile_range = (0.0, 1.0)

    suffix = "hpd_near_zero"

    # Joint distribution summary
    for name, diag in [("Continuous", continuous_diag), ("ContFrag", contfrag_diag)]:
        hpd = diag.hpd_overlap
        kl = diag.kl_divergence
        valid = ~np.isnan(hpd) & ~np.isnan(kl)
        hpd_v, kl_v = hpd[valid], kl[valid]

        print(f"\n{name} model:")
        print(
            f"  KL:  min={np.min(kl_v):.2f}  25th={np.quantile(kl_v, 0.25):.2f}  "
            f"median={np.median(kl_v):.2f}  75th={np.quantile(kl_v, 0.75):.2f}  "
            f"max={np.max(kl_v):.2f}"
        )
        print(
            f"  HPD: min={np.min(hpd_v):.4f}  5th={np.quantile(hpd_v, 0.05):.4f}  "
            f"median={np.median(hpd_v):.4f}  max={np.max(hpd_v):.4f}"
        )

        # Among HPD < 0.1, what's the KL range?
        low_hpd = hpd_v < 0.1
        n_low_hpd = int(np.sum(low_hpd))
        print(f"  Spikes with HPD < 0.1: {n_low_hpd} / {len(hpd_v)}")
        if n_low_hpd > 0:
            kl_at_low_hpd = kl_v[low_hpd]
            print(f"    KL range: {np.min(kl_at_low_hpd):.2f} — {np.max(kl_at_low_hpd):.2f}")
            print(
                f"    KL percentiles: 5th={np.quantile(kl_at_low_hpd, 0.05):.2f}  "
                f"25th={np.quantile(kl_at_low_hpd, 0.25):.2f}  "
                f"median={np.median(kl_at_low_hpd):.2f}"
            )

        # Among HPD == 0, what's the KL range?
        zero_hpd = hpd_v == 0.0
        n_zero = int(np.sum(zero_hpd))
        print(f"  Spikes with HPD == 0: {n_zero}")
        if n_zero > 0:
            kl_at_zero = kl_v[zero_hpd]
            print(f"    KL range: {np.min(kl_at_zero):.2f} — {np.max(kl_at_zero):.2f}")
            print(
                f"    KL percentiles: 5th={np.quantile(kl_at_zero, 0.05):.2f}  "
                f"25th={np.quantile(kl_at_zero, 0.25):.2f}  "
                f"median={np.median(kl_at_zero):.2f}"
            )

    # Generate plots for both models
    print(f"\nGenerating sanity check plots ({suffix})...")
    # Track graph info for x-axis visualization
    track_graph = data["track_graph"]
    edge_order = data["linear_edge_order"]
    edge_spacing = data["linear_edge_spacing"]

    generate_sanity_plots(
        model=continuous_model,
        results=continuous_results,
        spike_counts=spike_counts,
        diagnostics=continuous_diag,
        linear_position=linear_position,
        model_name="Continuous",
        output_dir=OUTPUT_ROOT / suffix / "continuous",
        time_slice=time_slice,
        select_metric=select_metric,
        select_mode=select_mode,
        filter_metric=filter_metric,
        filter_quantile_range=filter_quantile_range,
        track_graph=track_graph,
        edge_order=edge_order,
        edge_spacing=edge_spacing,
    )
    generate_sanity_plots(
        model=contfrag_model,
        results=contfrag_results,
        spike_counts=spike_counts,
        diagnostics=contfrag_diag,
        linear_position=linear_position,
        model_name="Cont-Frag",
        output_dir=OUTPUT_ROOT / suffix / "contfrag",
        time_slice=time_slice,
        select_metric=select_metric,
        select_mode=select_mode,
        filter_metric=filter_metric,
        filter_quantile_range=filter_quantile_range,
        track_graph=track_graph,
        edge_order=edge_order,
        edge_spacing=edge_spacing,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
