"""Real hippocampal data diagnostics for state space model.

This script generates Figure 4, which shows per-cell diagnostic metrics
for decoder models on real neural recording data from hippocampus.
Panel (a) shows a detail view with the Continuous decoder and panel (b)
shows the same detail view with the Continuous-Fragmented decoder.

Requires:
- non_local_detector package for decoder models
- Pre-exported neural recording data in data/real/
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import warnings
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

import joblib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Bbox
from numpy.typing import NDArray

from statespacecheck_paper.diagnostics import SpikeEventDiagnostics
from statespacecheck_paper.load_local_data import load_neural_recording_from_files
from statespacecheck_paper.paths import ANIMAL_DATE_EPOCH, DATA_PATH
from statespacecheck_paper.real_data_analysis import (
    Figure4Config,
    compute_flag_confusion,
    compute_model_diagnostics,
    create_decoder_environment,
    extract_place_fields,
    extract_shared_position_place_fields,
    fit_decoder_models,
    get_spike_counts,
)
from statespacecheck_paper.real_data_plotting import (
    ANIMAL_POSITION_LABEL_GID,
    THRESHOLD_LABEL_GID,
    WORSE_FIT_LABEL_GID,
    plot_per_spike_metric_hexbin_row,
    plot_single_model_diagnostics,
    plot_track_graph_2d,
)
from statespacecheck_paper.style import save_figure, set_figure_defaults

# -----------------------------
# Configuration
# -----------------------------

__all__ = ["ANIMAL_DATE_EPOCH", "DATA_PATH"]


# Time window for Figure 4a/b (detail view)
# Centered on a period of clear diagnostic activity at reward well
DETAIL_CENTER = 193069  # Time index with KL spike during immobility
DETAIL_HALF_WIDTH = 500  # Half-width in time points (~2 seconds at 500 Hz)
DIAGNOSTIC_ANNOTATION_GIDS = {THRESHOLD_LABEL_GID, WORSE_FIT_LABEL_GID}
FIG4_CACHE_SCHEMA_VERSION = 4

# --- Track-inset / hexbin pixel-nudge constants ---------------------------
# Empirically measured on the exported PNG at the current figure size (7.2 x
# 6.1 in) and DPI (450). They tune only artist placement, never any decoded or
# diagnostic value; changing the figure size or DPI would require re-measuring.
#
# ``add_scalebar`` appends the scale bar as the final line; SCALE_BAR_SHIFT /
# SCALE_BAR_DROP move the bar and its label together so the label clears the
# nearby reward-well marker.
SCALE_BAR_SHIFT = 22.0
SCALE_BAR_DROP = 5.0
# The trajectory line's vector bbox extends slightly farther left than the
# visually salient rendered diagram, so the track inset's left edge is nudged
# right by this many pixels when aligning it to the diagnostic annotations.
VISUAL_EDGE_CORRECTION_PX = 7.0
# Enlarge the track inset about its center for legibility.
TRACK_SIZE_SCALE = 1.10


@dataclasses.dataclass(frozen=True)
class Fig4Paths:
    """Injected data-location identifiers for the Figure-4 pipeline.

    Threaded into :func:`_load_or_compute_fig4_bundle` instead of reading the
    module-global ``DATA_PATH`` / ``ANIMAL_DATE_EPOCH`` so the compute/load
    layer is testable with synthetic inputs and a temporary cache directory.
    """

    data_path: Path
    animal_date_epoch: str

    @property
    def cache_path(self) -> Path:
        """Path for the cached Figure-4 decoder outputs (under data/intermediates).

        A single joblib bundle is used rather than netCDF because the decoder
        results carry a ``state_bins`` MultiIndex coordinate, which netCDF cannot
        serialize; joblib (pickle) preserves it exactly.
        """
        return self.data_path / "intermediates" / f"{self.animal_date_epoch}_fig4_cache.joblib"


@dataclasses.dataclass(frozen=True)
class Fig4Bundle:
    """Everything the Figure-4 render needs: fresh track data + decode payload.

    The decode payload (results / diagnostics / spike counts / place fields) is
    the expensive, cacheable content. The position/track data is always loaded
    fresh from :class:`Fig4Paths` (it is cheap and never cached), then combined
    with the decode payload here so the render reads a single object.
    """

    # Position / track data (always loaded fresh; not cached)
    position_info: Any
    time: NDArray[np.float64]
    position: NDArray[np.float64]
    linear_position: NDArray[np.float64]
    spike_times_list: list[Any]
    track_graph: Any
    edge_order: Any
    edge_spacing: Any
    # Decode payload (cached or recomputed)
    continuous_results: Any
    contfrag_results: Any
    continuous_diagnostics: SpikeEventDiagnostics
    contfrag_diagnostics: SpikeEventDiagnostics
    spike_counts: NDArray[np.int64]
    place_field_peaks: NDArray[np.float64]
    diagnostic_place_fields: NDArray[np.float64]
    diagnostic_position_bins: NDArray[np.float64]


def _installed_non_local_detector_version() -> str:
    """Return the installed ``non_local_detector`` version, or ``"unknown"``."""
    try:
        return version("non_local_detector")
    except PackageNotFoundError:
        return "unknown"


def fig4_cache_fingerprint(config: Figure4Config, paths: Fig4Paths) -> str:
    """Provenance fingerprint gating the Figure-4 cache.

    Hashes the schema version, the manuscript decoder parameters
    (:class:`Figure4Config`), the input-data identifier, and the *installed*
    ``non_local_detector`` revision. Any change forces a recompute; the cached
    bundle stores this fingerprint so a stale cache cannot silently produce a
    figure that no longer matches the current method or dependency.

    Bumping :data:`FIG4_CACHE_SCHEMA_VERSION` remains the manual override --- it
    is part of the hashed payload, so a bump invalidates every existing cache.
    """
    payload = {
        "schema_version": FIG4_CACHE_SCHEMA_VERSION,
        "config": dataclasses.asdict(config),
        "animal_date_epoch": paths.animal_date_epoch,
        "non_local_detector_version": _installed_non_local_detector_version(),
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def shift_diagnostic_event_times(
    diagnostics: SpikeEventDiagnostics,
    time_offset: float,
) -> SpikeEventDiagnostics:
    """Return diagnostics with event timestamps shifted by ``time_offset``.

    Returns the original instance unchanged when ``event_time`` is
    ``None`` (simulated data path) so callers don't need to branch.
    """
    if diagnostics.event_time is None:
        return diagnostics
    return dataclasses.replace(
        diagnostics,
        event_time=np.asarray(diagnostics.event_time, dtype=np.float64) - time_offset,
    )


def diagnostic_event_mean(diagnostics: SpikeEventDiagnostics, metric: str) -> float:
    """Return the per-spike mean for a diagnostic metric."""
    event_key = f"event_{metric}"
    if not hasattr(diagnostics, event_key):
        raise KeyError(f"Missing per-spike diagnostic array: {event_key}")
    return float(np.nanmean(getattr(diagnostics, event_key)))


def _visible_artist_bboxes(artists: list[Any], renderer: Any) -> list[Any]:
    """Return finite window extents for visible artists."""
    bboxes = []
    for artist in artists:
        if not artist.get_visible():
            continue
        bbox = artist.get_window_extent(renderer)
        if np.isfinite([bbox.x0, bbox.x1, bbox.y0, bbox.y1]).all():
            bboxes.append(bbox)
    return bboxes


def _axis_content_artists(ax: Any) -> list[Any]:
    """Return plotted artists that define an axis's visible content bounds."""
    return [*ax.lines, *ax.collections, *ax.texts]


def _shift_axis_to_artist_edge(
    ax: Any,
    artists: list[Any],
    renderer: Any,
    *,
    target_px: float,
    edge: Literal["left", "right"],
    correction_px: float = 0.0,
) -> None:
    """Shift an axis horizontally so its visible content edge matches a target."""
    bboxes = _visible_artist_bboxes(artists, renderer)
    if not bboxes:
        return

    content_edge_px = (
        min(bbox.x0 for bbox in bboxes) if edge == "left" else max(bbox.x1 for bbox in bboxes)
    )
    pos = ax.get_position()
    shift = (target_px - content_edge_px + correction_px) / ax.figure.bbox.width
    ax.set_position([pos.x0 + shift, pos.y0, pos.width, pos.height])


def _axes_tight_bbox_inches(fig: Any, *, pad_inches: float = 0.05) -> Bbox:
    """Return a figure bbox cropped to the union of visible axes."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bboxes = [
        bbox
        for ax in fig.axes
        if ax.get_visible()
        if (bbox := ax.get_tightbbox(renderer)) is not None and np.isfinite(bbox.extents).all()
    ]
    if not bboxes:
        return fig.bbox_inches

    bbox_inches = Bbox.union(bboxes).transformed(fig.dpi_scale_trans.inverted())
    return bbox_inches.padded(pad_inches)


def _compute_fig4_decode_payload(
    *,
    position: NDArray[np.float64],
    spike_times_list: list[Any],
    time: NDArray[np.float64],
    track_graph: Any,
    edge_order: Any,
    edge_spacing: Any,
) -> dict[str, Any]:
    """Fit both decoders, decode, and compute the cacheable decode payload.

    Returns exactly the keys stored in (and loaded from) the Figure-4 cache.
    """
    # Environment is only needed to fit the decoders.
    env = create_decoder_environment(
        track_graph=track_graph,
        edge_order=edge_order,
        edge_spacing=edge_spacing,
    )

    print("Fitting models...")
    continuous_model, contfrag_model = fit_decoder_models(
        position=position,
        spike_times=spike_times_list,
        time=time,
        environment=env,
    )

    print(f"Decoding {len(time)} time points...")
    decode_outputs = ["filter", "predictive_posterior", "log_likelihood"]
    continuous_results = continuous_model.predict(
        spike_times=spike_times_list,
        time=time,
        return_outputs=decode_outputs,
    )
    contfrag_results = contfrag_model.predict(
        spike_times=spike_times_list,
        time=time,
        return_outputs=decode_outputs,
    )

    spike_counts = get_spike_counts(spike_times_list, time)

    print("Computing diagnostics...")
    continuous_diagnostics = compute_model_diagnostics(
        continuous_model, continuous_results, spike_counts, time, spike_times=spike_times_list
    )
    contfrag_diagnostics = compute_model_diagnostics(
        contfrag_model, contfrag_results, spike_counts, time, spike_times=spike_times_list
    )

    # Extract place fields for raster sorting (use continuous model).
    place_fields, position_bins = extract_place_fields(continuous_model)
    if np.any(np.all(np.isnan(place_fields), axis=1)):
        warnings.warn(
            "Some cells have all-NaN place fields; peak positions may be incorrect",
            stacklevel=2,
        )
    place_field_peaks = position_bins[np.nanargmax(place_fields, axis=1)]

    # Shared interior place fields for the mean per-spike likelihood row.
    # The row is meant to be identical across decoders, so verify the two
    # models agree on both fields and grid before storing a single copy.
    diagnostic_place_fields, diagnostic_position_bins = extract_shared_position_place_fields(
        continuous_model
    )
    contfrag_place_fields, contfrag_position_bins = extract_shared_position_place_fields(
        contfrag_model
    )
    if not np.allclose(
        diagnostic_place_fields, contfrag_place_fields, equal_nan=True
    ) or not np.allclose(diagnostic_position_bins, contfrag_position_bins, equal_nan=True):
        raise ValueError(
            "Continuous and Continuous--Fragmented place fields or position "
            "grids differ; the shared likelihood row would misrepresent one "
            "of the decoders."
        )

    return {
        "continuous_results": continuous_results,
        "contfrag_results": contfrag_results,
        "continuous_diagnostics": continuous_diagnostics,
        "contfrag_diagnostics": contfrag_diagnostics,
        "spike_counts": spike_counts,
        "place_field_peaks": place_field_peaks,
        "diagnostic_place_fields": diagnostic_place_fields,
        "diagnostic_position_bins": diagnostic_position_bins,
    }


def _load_or_compute_fig4_bundle(
    config: Figure4Config,
    paths: Fig4Paths,
    *,
    use_cache: bool = True,
) -> Fig4Bundle:
    """Assemble the Figure-4 bundle: fresh track data + cached-or-computed decode.

    Reads only the injected ``config`` and ``paths`` (never the module-global
    ``DATA_PATH`` / ``ANIMAL_DATE_EPOCH``), so it is exercisable with synthetic
    inputs and a temporary cache directory. The cache is keyed on
    :func:`fig4_cache_fingerprint`; a config / data / dependency change forces a
    recompute. The position/track data is always loaded fresh (it is cheap and
    never cached).

    Parameters
    ----------
    config : Figure4Config
        Decoder configuration; hashed into the cache fingerprint.
    paths : Fig4Paths
        Injected data-location identifiers.
    use_cache : bool, default True
        When True and a fingerprint-matching cache exists, load it instead of
        recomputing. When False, always recompute and overwrite the cache.
    """
    print("Loading data...")
    data = load_neural_recording_from_files(paths.data_path, paths.animal_date_epoch)
    print(f"  Loaded {len(data['spike_times'])} cells")

    position_info = data["position_info"]
    render_data: dict[str, Any] = dict(
        position_info=position_info,
        time=position_info.index.values,
        position=position_info[["head_position_x", "head_position_y"]].values,
        linear_position=position_info["linear_position"].values,
        spike_times_list=list(data["spike_times"]),
        track_graph=data["track_graph"],
        edge_order=data["linear_edge_order"],
        edge_spacing=data["linear_edge_spacing"],
    )

    cache_path = paths.cache_path
    expected_fingerprint = fig4_cache_fingerprint(config, paths)
    if use_cache and cache_path.exists():
        print("Loading cached decoder outputs (use --force-recompute to rebuild)...")
        cached = joblib.load(cache_path)
        if cached.get("fingerprint") == expected_fingerprint:
            payload = {key: cached[key] for key in _FIG4_PAYLOAD_KEYS}
            return Fig4Bundle(**render_data, **payload)
        print(
            "  Cache fingerprint mismatch (config, data, or non_local_detector "
            "version changed, or a pre-fingerprint cache); recomputing."
        )

    payload = _compute_fig4_decode_payload(
        position=render_data["position"],
        spike_times_list=render_data["spike_times_list"],
        time=render_data["time"],
        track_graph=render_data["track_graph"],
        edge_order=render_data["edge_order"],
        edge_spacing=render_data["edge_spacing"],
    )

    print("Caching decoder outputs to data/intermediates ...")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "schema_version": FIG4_CACHE_SCHEMA_VERSION,
            "fingerprint": expected_fingerprint,
            **payload,
        },
        cache_path,
    )

    return Fig4Bundle(**render_data, **payload)


# The decode payload keys shared by the cache dict and :class:`Fig4Bundle`.
_FIG4_PAYLOAD_KEYS = (
    "continuous_results",
    "contfrag_results",
    "continuous_diagnostics",
    "contfrag_diagnostics",
    "spike_counts",
    "place_field_peaks",
    "diagnostic_place_fields",
    "diagnostic_position_bins",
)


def _print_fig4_summary(
    bundle: Fig4Bundle,
    thresholds: dict[str, float],
    metric_directions: dict[str, Literal["below", "above"]],
) -> None:
    """Print the whole-session diagnostic event-means and flag-agreement counts.

    These scalars can appear in the manuscript text, so their computation must
    stay identical to the cached decode; the render layer never alters them.
    """
    print("\n=== Diagnostic Summary (all time points) ===")
    for name, diag in [
        ("Continuous", bundle.continuous_diagnostics),
        ("ContFrag", bundle.contfrag_diagnostics),
    ]:
        print(f"\n{name}:")
        for metric in ["hpd_overlap", "kl_divergence", "predictive_pvalue"]:
            print(f"  {metric}: {diagnostic_event_mean(diag, metric):.4f}")

    # Per-spike flag agreement between the two decoders at these thresholds.
    # "Cont-only" is the rescue quadrant (flagged by Continuous but not by
    # Continuous-Fragmented); "rescue" is its fraction of all Continuous flags.
    print("\n=== Flag agreement: Continuous (A) vs Cont-Frag (B) ===")
    for metric, worse_when in metric_directions.items():
        conf = compute_flag_confusion(
            bundle.continuous_diagnostics,
            bundle.contfrag_diagnostics,
            metric,
            thresholds[metric],
            worse_when=worse_when,
        )
        print(
            f"  {metric}: n={conf.n:,} both={conf.both:,} cont-only={conf.a_only:,} "
            f"cf-only={conf.b_only:,} neither={conf.neither:,} "
            f"rescue={100 * conf.rescue_rate:.1f}%"
        )


def _place_track_inset(
    track_subfig: Any,
    fig: Any,
    bundle: Fig4Bundle,
    axes_b: Any,
) -> Any:
    """Draw the unlettered 2D track inset and align it to the diagnostic labels.

    Pixel-nudging is confined here and to the module-level ``SCALE_BAR_*`` /
    ``VISUAL_EDGE_CORRECTION_PX`` / ``TRACK_SIZE_SCALE`` constants (measured at
    the current figure size and DPI). Returns the inset axis so the hexbin
    layout can later align its right edge to the colorbar label.
    """
    # Unlettered track inset beside panels (a) and (b). Use the same row
    # rhythm as the detail stacks and place it beside the likelihood row.
    track_gs = track_subfig.add_gridspec(
        6,
        3,
        height_ratios=[2, 2, 1.5, 1, 1, 1],
        width_ratios=[0.01, 0.68, 0.31],
    )
    ax_track = track_subfig.add_subplot(track_gs[1, 1])
    # Reward wells sit at the arm tips, i.e. the degree-1 (leaf) nodes of the
    # track graph. Mark them so the 2D layout connects to the linearized axis
    # used in panels (a)-(b).
    track_graph = bundle.track_graph
    reward_well_nodes = [n for n in track_graph.nodes if track_graph.degree(n) == 1]
    plot_track_graph_2d(
        track_graph=track_graph,
        position_info=bundle.position_info,
        ax=ax_track,
        edge_order=bundle.edge_order,
        reward_well_nodes=reward_well_nodes,
        scalebar_length=20,
        scalebar_label="20 cm",
    )
    ax_track.set_anchor("W")
    # Move the scale bar and its label together so the label clears the nearby
    # reward-well marker.
    scale_bar_line = ax_track.lines[-1]
    scale_bar_line.set_xdata(np.asarray(scale_bar_line.get_xdata()) + SCALE_BAR_SHIFT)
    scale_bar_line.set_ydata(np.asarray(scale_bar_line.get_ydata()) - SCALE_BAR_DROP)
    scale_bar_line.set_linewidth(2.0)
    for text in ax_track.texts:
        if text.get_text() == "20 cm":
            x_pos, y_pos = text.get_position()
            text.set_position((x_pos + SCALE_BAR_SHIFT + 10, y_pos - 4 - SCALE_BAR_DROP))
            text.set_fontsize(8.5)
            text.set_clip_on(False)

    # Align the track diagram itself with the shared right-side diagnostic
    # annotations: the diagram's left edge should begin where the annotation
    # text ends.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    annotation_texts = [
        text
        for ax in axes_b[3:]
        for text in ax.texts
        if text.get_gid() in DIAGNOSTIC_ANNOTATION_GIDS
    ]
    annotation_bboxes = _visible_artist_bboxes(annotation_texts, renderer)
    if annotation_bboxes:
        annotation_right = max(bbox.x1 for bbox in annotation_bboxes)
        ax_track.set_in_layout(False)
        _shift_axis_to_artist_edge(
            ax_track,
            _axis_content_artists(ax_track),
            renderer,
            target_px=annotation_right,
            edge="left",
            correction_px=VISUAL_EDGE_CORRECTION_PX,
        )
    pos = ax_track.get_position()
    ax_track.set_position(
        [
            pos.x0,
            pos.y0 - pos.height * (TRACK_SIZE_SCALE - 1) / 2,
            pos.width * TRACK_SIZE_SCALE,
            pos.height * TRACK_SIZE_SCALE,
        ]
    )
    return ax_track


def _layout_hexbin_row(
    bottom_subfig: Any,
    fig: Any,
    bundle: Fig4Bundle,
    thresholds: dict[str, float],
    ax_track: Any,
) -> None:
    """Render and hand-place the whole-session metric hexbin row and colorbar.

    The subplot-grid rhythm and colorbar spacing are nudged in figure
    coordinates here; the final step aligns the track inset's right edge to the
    colorbar label so the two rows share a right margin.
    """
    subfigs_bot = bottom_subfig.subfigures(1, 3, width_ratios=[0.16, 7, 0.16], wspace=0.015)
    axes_hexbin = subfigs_bot[1].subplots(1, 3, gridspec_kw={"wspace": -0.02})
    axes_before_hexbin = tuple(fig.axes)
    plot_per_spike_metric_hexbin_row(
        bundle.continuous_diagnostics,
        bundle.contfrag_diagnostics,
        axes_hexbin,
        model_a_name="Continuous",
        model_b_name="Cont-Frag",
        thresholds=thresholds,
        colorbar_pad=0.006,
    )
    for ax, anchor in zip(axes_hexbin, ("E", "C", "W"), strict=True):
        ax.set_anchor(anchor)
    hexbin_colorbar_axes = [ax for ax in fig.axes if ax not in axes_before_hexbin]
    fig.canvas.draw()
    hexbin_positions = [ax.get_position() for ax in axes_hexbin]
    panel_width = min(pos.width for pos in hexbin_positions)
    panel_height = min(pos.height for pos in hexbin_positions)
    panel_gap = min(
        hexbin_positions[1].x0 - hexbin_positions[0].x1,
        hexbin_positions[2].x0 - hexbin_positions[1].x1,
    )
    panel_gap = max(panel_gap, 0.0)
    panel_left = hexbin_positions[0].x0
    panel_bottom = hexbin_positions[0].y0
    for panel_idx, ax in enumerate(axes_hexbin):
        ax.set_position(
            [
                panel_left + panel_idx * (panel_width + panel_gap),
                panel_bottom,
                panel_width,
                panel_height,
            ]
        )
    if hexbin_colorbar_axes:
        colorbar_ax = hexbin_colorbar_axes[-1]
        colorbar_pos = colorbar_ax.get_position()
        colorbar_gap = max(panel_gap * 0.5, 0.006)
        colorbar_ax.set_position(
            [
                axes_hexbin[-1].get_position().x1 + colorbar_gap,
                colorbar_pos.y0,
                colorbar_pos.width,
                colorbar_pos.height,
            ]
        )
    fig.set_constrained_layout(False)
    if hexbin_colorbar_axes:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        colorbar_label_bbox = hexbin_colorbar_axes[-1].yaxis.label.get_window_extent(renderer)
        _shift_axis_to_artist_edge(
            ax_track,
            _axis_content_artists(ax_track),
            renderer,
            target_px=colorbar_label_bbox.x1,
            edge="right",
        )
    axes_hexbin[0].text(
        -0.18,
        1.10,
        "c",
        fontsize=8,
        fontweight="bold",
        transform=axes_hexbin[0].transAxes,
        va="top",
        ha="right",
    )


def run_demo(*, use_cache: bool = True) -> None:
    """Run the full Figure 4 generation pipeline.

    A linear sequence: build the :class:`Figure4Config` and :class:`Fig4Paths`,
    assemble the :class:`Fig4Bundle` (cached-or-computed decode payload + fresh
    track data), print the summary scalars, render the detail stacks, place the
    track inset, lay out the hexbin row, and save.

    Parameters
    ----------
    use_cache : bool, default True
        When True and a fingerprint-matching cache of decoder outputs exists
        under ``data/intermediates``, load it and skip the expensive fit/decode
        step. When False (``--force-recompute``), always recompute and
        overwrite the cache. A config / data / ``non_local_detector`` change
        invalidates the cache automatically. Fitting + decoding both models
        takes several minutes; figure-only edits (styling, thresholds) reuse
        the cache.
    """
    # Compute or load the expensive decode payload and bundle it with the
    # freshly-loaded position/track data.
    config = Figure4Config()
    paths = Fig4Paths(data_path=DATA_PATH, animal_date_epoch=ANIMAL_DATE_EPOCH)
    bundle = _load_or_compute_fig4_bundle(config, paths, use_cache=use_cache)

    # Diagnostic thresholds. HPD overlap and the predictive p-value use fixed
    # cutoffs of 0.05. The KL divergence has no natural fixed cutoff, so it is
    # shown without a threshold line or a flagged-region callout.
    diagnostic_thresholds = {
        "hpd_overlap": 0.05,
        "predictive_pvalue": 0.05,
    }
    metric_directions: dict[str, Literal["below", "above"]] = {
        "hpd_overlap": "below",
        "predictive_pvalue": "below",
    }
    _print_fig4_summary(bundle, diagnostic_thresholds, metric_directions)

    # Generate Figure 4
    print("\nGenerating Figure 4...")
    set_figure_defaults(context="paper")

    # Define the detail-window time slice shared by both decoder panels.
    detail_slice = slice(
        DETAIL_CENTER - DETAIL_HALF_WIDTH,
        DETAIL_CENTER + DETAIL_HALF_WIDTH,
    )

    # Convert time to relative seconds from start of the detail window
    time_arr = np.asarray(bundle.time, dtype=np.float64)
    time_offset = time_arr[detail_slice.start]
    time_relative = time_arr - time_offset

    # Shift xarray time coordinates to relative seconds
    continuous_results = bundle.continuous_results.assign_coords(
        time=bundle.continuous_results.coords["time"].values - time_offset
    )
    contfrag_results = bundle.contfrag_results.assign_coords(
        time=bundle.contfrag_results.coords["time"].values - time_offset
    )

    # Shift spike times to relative seconds
    spike_times_relative: list[Any] = [st - time_offset for st in bundle.spike_times_list]
    continuous_diagnostics_relative = shift_diagnostic_event_times(
        bundle.continuous_diagnostics,
        time_offset,
    )
    contfrag_diagnostics_relative = shift_diagnostic_event_times(
        bundle.contfrag_diagnostics,
        time_offset,
    )

    # Two-row figure: (a)/(b) detail zooms with a track inset on top, and
    # (c) whole-session metric hexbins on the bottom.
    fig = plt.figure(figsize=(7.2, 6.1), dpi=450, constrained_layout=True)
    subfigs_rows = fig.subfigures(2, 1, height_ratios=[5.0, 2.6], hspace=0.02)

    # Shared plotting kwargs for detail panels
    detail_kwargs: dict[str, Any] = dict(
        spike_times=spike_times_relative,
        spike_counts=bundle.spike_counts,
        place_field_peaks=bundle.place_field_peaks,
        place_fields=bundle.diagnostic_place_fields,
        position_bins=bundle.diagnostic_position_bins,
        time_slice_ind=detail_slice,
        thresholds=diagnostic_thresholds,
        track_graph=bundle.track_graph,
        edge_order=bundle.edge_order,
        edge_spacing=bundle.edge_spacing,
    )

    # Top row: (a) Continuous and (b) ContFrag detail zooms, side by side,
    # with a small unlettered track inset on the right for spatial context.
    subfigs_top = subfigs_rows[0].subfigures(
        1,
        5,
        width_ratios=[0.055, 1.0, 1.07, 0.42, 0.04],
        wspace=0.005,
    )

    # Panel (a): Continuous detail view
    _, axes_a = plot_single_model_diagnostics(
        time_relative,
        bundle.linear_position,
        continuous_results,
        continuous_diagnostics_relative,
        model_name="Continuous Model",
        fig=subfigs_top[1],
        **detail_kwargs,
    )
    axes_a[3].set_ylabel("HPD\noverlap", fontsize=8, labelpad=7)

    # Panel (b): ContFrag detail view
    _, axes_b = plot_single_model_diagnostics(
        time_relative,
        bundle.linear_position,
        contfrag_results,
        contfrag_diagnostics_relative,
        model_name="Cont.-Frag. Model",
        fig=subfigs_top[2],
        **detail_kwargs,
    )

    # Match y-axis limits between detail panels for direct comparison
    for i in range(6):
        ylim_a = axes_a[i].get_ylim()
        ylim_b = axes_b[i].get_ylim()
        shared_ylim = (min(ylim_a[0], ylim_b[0]), max(ylim_a[1], ylim_b[1]))
        axes_a[i].set_ylim(shared_ylim)
        axes_b[i].set_ylim(shared_ylim)

    # Panel (b) repeats the row scales from panel (a), so keep only the
    # model-specific data and title on the right stack.
    for ax in axes_b:
        ax.set_ylabel("")
        ax.tick_params(axis="y", left=False, labelleft=False)
    for text in axes_b[0].texts:
        if text.get_gid() == ANIMAL_POSITION_LABEL_GID:
            text.set_visible(False)

    # Keep threshold / worse-fit row annotations only on panel (b), where they
    # read as shared labels for both model stacks.
    for ax in axes_a[3:]:
        for text in ax.texts:
            if text.get_gid() in DIAGNOSTIC_ANNOTATION_GIDS:
                text.set_visible(False)

    # Panel labels - place in axes coordinates on the predictive row of each.
    panel_label_x = {"a": -0.115, "b": -0.05}
    for axes, label in [(axes_a, "a"), (axes_b, "b")]:
        axes[0].text(
            panel_label_x[label],
            1.24,
            label,
            fontsize=8,
            fontweight="bold",
            transform=axes[0].transAxes,
            va="top",
            ha="right",
        )

    # Unlettered track inset beside panels (a) and (b), then the whole-session
    # metric hexbin row underneath.
    ax_track = _place_track_inset(subfigs_top[3], fig, bundle, axes_b)
    _layout_hexbin_row(subfigs_rows[1], fig, bundle, diagnostic_thresholds, ax_track)

    side_tight_bbox = _axes_tight_bbox_inches(fig, pad_inches=0.05)
    save_figure("manuscript/figures/main/figure04", close=True, bbox_inches=side_tight_bbox)
    print("Saved manuscript/figures/main/figure04.{pdf,png}")
    print("\nFigure 4 complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Figure 4.")
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help=(
            "Re-fit and re-decode both models instead of loading the cached "
            "decoder outputs under data/intermediates (overwrites the cache)."
        ),
    )
    args = parser.parse_args()
    run_demo(use_cache=not args.force_recompute)
