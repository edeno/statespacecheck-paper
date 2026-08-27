"""Figure-4 layout: artist arrangement and render-only transformations.

Owns everything about *how* Figure 4 looks — the validated detail-window
contract, pixel-nudge layout constants, bbox/edge-alignment helpers, track inset
and hexbin-row placement, and ``compose_figure04`` which assembles the two-row
figure and returns it with the tight bounding box to crop to. It reads a
:class:`Figure4RenderData` and imports only the render layers; it never loads
data, fits/decodes, reads the cache/config/paths, or saves.
"""

from __future__ import annotations

import dataclasses
import numbers
from collections.abc import Mapping
from typing import Any, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox
from numpy.typing import NDArray

from statespacecheck_paper.diagnostics import SpikeEventDiagnostics
from statespacecheck_paper.figure04_panels import (
    ModelDiagnosticPanelData,
    plot_per_spike_metric_hexbin_row,
    plot_single_model_diagnostics,
)
from statespacecheck_paper.figure04_plot_primitives import (
    ANIMAL_POSITION_LABEL_GID,
    THRESHOLD_LABEL_GID,
    WORSE_FIT_LABEL_GID,
)
from statespacecheck_paper.figure04_track_plots import plot_track_graph_2d
from statespacecheck_paper.figure04_workflow import Figure4RenderData

FIGURE4_DIAGNOSTIC_ANNOTATION_GIDS = {THRESHOLD_LABEL_GID, WORSE_FIT_LABEL_GID}

# --- Track-inset / hexbin pixel-nudge constants ---------------------------
# Empirically measured on the exported PNG at the current figure size (7.2 x
# 6.1 in) and DPI (450). They tune only artist placement, never any decoded or
# diagnostic value; changing the figure size or DPI would require re-measuring.
#
# ``add_scalebar`` appends the scale bar as the final line; FIGURE4_SCALE_BAR_HORIZONTAL_SHIFT_PX /
# FIGURE4_SCALE_BAR_VERTICAL_DROP_PX move the bar and its label together so the label clears the
# nearby reward-well marker.
FIGURE4_SCALE_BAR_HORIZONTAL_SHIFT_PX = 22.0
FIGURE4_SCALE_BAR_VERTICAL_DROP_PX = 5.0
# The trajectory line's vector bbox extends slightly farther left than the
# visually salient rendered diagram, so the track inset's left edge is nudged
# right by this many pixels when aligning it to the diagnostic annotations.
FIGURE4_TRACK_VISUAL_EDGE_CORRECTION_PX = 7.0
# Enlarge the track inset about its center for legibility.
FIGURE4_TRACK_SIZE_SCALE = 1.10


def _shift_diagnostic_event_times(
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
        return cast("Bbox", fig.bbox_inches)

    bbox_inches = Bbox.union(bboxes).transformed(fig.dpi_scale_trans.inverted())
    return bbox_inches.padded(pad_inches)


def _place_track_inset(
    track_subfig: Any,
    fig: Any,
    render_data: Figure4RenderData,
    axes_b: Any,
) -> Any:
    """Draw the unlettered 2D track inset and align it to the diagnostic labels.

    Pixel-nudging is confined here and to the module-level ``SCALE_BAR_*`` /
    ``FIGURE4_TRACK_VISUAL_EDGE_CORRECTION_PX`` / ``FIGURE4_TRACK_SIZE_SCALE`` constants
    (measured at
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
    track_graph = render_data.recording.track_graph
    reward_well_nodes = [n for n in track_graph.nodes if track_graph.degree(n) == 1]
    plot_track_graph_2d(
        track_graph=track_graph,
        position_info=render_data.recording.position_info,
        ax=ax_track,
        edge_order=render_data.recording.linear_edge_order,
        reward_well_nodes=reward_well_nodes,
        scalebar_length=20,
        scalebar_label="20 cm",
    )
    ax_track.set_anchor("W")
    # Move the scale bar and its label together so the label clears the nearby
    # reward-well marker.
    scale_bar_line = ax_track.lines[-1]
    scale_bar_line.set_xdata(
        np.asarray(scale_bar_line.get_xdata()) + FIGURE4_SCALE_BAR_HORIZONTAL_SHIFT_PX
    )
    scale_bar_line.set_ydata(
        np.asarray(scale_bar_line.get_ydata()) - FIGURE4_SCALE_BAR_VERTICAL_DROP_PX
    )
    scale_bar_line.set_linewidth(2.0)
    for text in ax_track.texts:
        if text.get_text() == "20 cm":
            x_pos, y_pos = text.get_position()
            text.set_position(
                (
                    x_pos + FIGURE4_SCALE_BAR_HORIZONTAL_SHIFT_PX + 10,
                    y_pos - 4 - FIGURE4_SCALE_BAR_VERTICAL_DROP_PX,
                )
            )
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
        if text.get_gid() in FIGURE4_DIAGNOSTIC_ANNOTATION_GIDS
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
            correction_px=FIGURE4_TRACK_VISUAL_EDGE_CORRECTION_PX,
        )
    pos = ax_track.get_position()
    ax_track.set_position(
        [
            pos.x0,
            pos.y0 - pos.height * (FIGURE4_TRACK_SIZE_SCALE - 1) / 2,
            pos.width * FIGURE4_TRACK_SIZE_SCALE,
            pos.height * FIGURE4_TRACK_SIZE_SCALE,
        ]
    )
    return ax_track


def _layout_hexbin_row(
    bottom_subfig: Any,
    fig: Any,
    render_data: Figure4RenderData,
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
        render_data.decode_results.continuous_diagnostics,
        render_data.decode_results.continuous_fragmented_diagnostics,
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


@dataclasses.dataclass(frozen=True)
class Figure4Composition:
    """The composed Figure-4 figure and the tight bbox to crop it to on save."""

    figure: Figure
    bbox_inches: Bbox


def _is_integer(value: object) -> bool:
    """True for Python and NumPy integers, excluding ``bool`` (an ``int`` subclass).

    Using :class:`numbers.Integral` accepts ``np.int64`` etc. (common when an
    index is derived from an array), which a strict ``type(x) is int`` check
    would spuriously reject in this NumPy-heavy codebase.
    """
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


@dataclasses.dataclass(frozen=True)
class Figure4DetailWindow:
    """Index window used for the side-by-side Figure-4 detail panels.

    The canonical values live in :mod:`figure04_generation`; layout receives
    them explicitly so tests and alternate recipes can select a scientifically
    meaningful window without mutating module globals.
    """

    center_index: int
    half_width_samples: int

    def __post_init__(self) -> None:
        if not _is_integer(self.center_index) or self.center_index < 0:
            raise ValueError("center_index must be a non-negative integer")
        if not _is_integer(self.half_width_samples) or self.half_width_samples <= 0:
            raise ValueError("half_width_samples must be a positive integer")

    def to_slice(self, n_time_samples: int) -> slice:
        """Return the validated half-open slice for a recording timeline."""
        if not _is_integer(n_time_samples) or n_time_samples <= 0:
            raise ValueError("n_time_samples must be a positive integer")
        start = self.center_index - self.half_width_samples
        stop = self.center_index + self.half_width_samples
        if start < 0 or stop > n_time_samples:
            raise ValueError(
                "detail window falls outside the recording timeline: "
                f"slice({start}, {stop}) for {n_time_samples} samples"
            )
        return slice(start, stop)


def compose_figure04(
    render_data: Figure4RenderData,
    *,
    diagnostic_thresholds: Mapping[str, float],
    detail_window: Figure4DetailWindow,
) -> Figure4Composition:
    """Arrange the Figure-4 artists and return the figure + its tight bbox.

    Render-only: relative-time copies, panel composition, track inset and
    hexbin placement, shared y-limits, and the tight-bbox crop. It loads no
    data, fits/decodes nothing, and reads no cache/config/paths.
    """
    thresholds_dict = dict(diagnostic_thresholds)
    # Define the explicit detail-window time slice shared by both decoder panels.
    detail_slice = detail_window.to_slice(render_data.time.size)

    # Convert time to relative seconds from start of the detail window
    time_arr = np.asarray(render_data.time, dtype=np.float64)
    time_offset = time_arr[detail_slice.start]
    time_relative = time_arr - time_offset

    # Shift xarray time coordinates to relative seconds
    continuous_results = render_data.decode_results.continuous_results.assign_coords(
        time=render_data.decode_results.continuous_results.coords["time"].values - time_offset
    )
    contfrag_results = render_data.decode_results.continuous_fragmented_results.assign_coords(
        time=render_data.decode_results.continuous_fragmented_results.coords["time"].values
        - time_offset
    )

    # Shift spike times to relative seconds
    spike_times_relative: list[NDArray[np.float64]] = [
        np.asarray(st - time_offset, dtype=np.float64) for st in render_data.recording.spike_times
    ]
    continuous_diagnostics_relative = _shift_diagnostic_event_times(
        render_data.decode_results.continuous_diagnostics,
        time_offset,
    )
    contfrag_diagnostics_relative = _shift_diagnostic_event_times(
        render_data.decode_results.continuous_fragmented_diagnostics,
        time_offset,
    )

    # Two-row figure: (a)/(b) detail zooms with a track inset on top, and
    # (c) whole-session metric hexbins on the bottom.
    fig = plt.figure(figsize=(7.2, 6.1), dpi=450, constrained_layout=True)
    subfigs_rows = fig.subfigures(2, 1, height_ratios=[5.0, 2.6], hspace=0.02)

    continuous_panel_data = ModelDiagnosticPanelData(
        time=time_relative,
        position=render_data.linear_position,
        results=continuous_results,
        diagnostics=continuous_diagnostics_relative,
        spike_times=spike_times_relative,
        spike_counts=render_data.decode_results.spike_counts,
        place_field_peaks=render_data.decode_results.place_field_peaks,
        place_fields=render_data.decode_results.diagnostic_place_fields,
        position_bins=render_data.decode_results.diagnostic_position_bins,
        track_graph=render_data.recording.track_graph,
        edge_order=render_data.recording.linear_edge_order,
        edge_spacing=render_data.recording.linear_edge_spacing,
    )
    continuous_fragmented_panel_data = ModelDiagnosticPanelData(
        time=time_relative,
        position=render_data.linear_position,
        results=contfrag_results,
        diagnostics=contfrag_diagnostics_relative,
        spike_times=spike_times_relative,
        spike_counts=render_data.decode_results.spike_counts,
        place_field_peaks=render_data.decode_results.place_field_peaks,
        place_fields=render_data.decode_results.diagnostic_place_fields,
        position_bins=render_data.decode_results.diagnostic_position_bins,
        track_graph=render_data.recording.track_graph,
        edge_order=render_data.recording.linear_edge_order,
        edge_spacing=render_data.recording.linear_edge_spacing,
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
        continuous_panel_data,
        time_slice_ind=detail_slice,
        thresholds=thresholds_dict,
        model_name="Continuous Model",
        fig=subfigs_top[1],
    )
    axes_a[3].set_ylabel("HPD\noverlap", fontsize=8, labelpad=7)

    # Panel (b): ContFrag detail view
    _, axes_b = plot_single_model_diagnostics(
        continuous_fragmented_panel_data,
        time_slice_ind=detail_slice,
        thresholds=thresholds_dict,
        model_name="Cont.-Frag. Model",
        fig=subfigs_top[2],
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
            if text.get_gid() in FIGURE4_DIAGNOSTIC_ANNOTATION_GIDS:
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
    ax_track = _place_track_inset(subfigs_top[3], fig, render_data, axes_b)
    _layout_hexbin_row(subfigs_rows[1], fig, render_data, thresholds_dict, ax_track)

    side_tight_bbox = _axes_tight_bbox_inches(fig, pad_inches=0.05)
    return Figure4Composition(figure=fig, bbox_inches=side_tight_bbox)
