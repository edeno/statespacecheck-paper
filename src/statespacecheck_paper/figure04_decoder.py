"""Figure-4 decoder configuration and construction.

Configuration: :class:`Figure4Config` and its four parts,
:class:`Figure4DecoderConfig` (injected decode parameters),
:class:`Figure4PackageDefaults` (recorded ``non_local_detector`` defaults),
:class:`Figure4ExecutionConfig` (performance-only settings), and
:class:`Figure4DiagnosticsConfig` (per-spike diagnostic settings).

Construction, using ``non_local_detector``: :func:`create_decoder_environment`
builds the track environment, :func:`build_decoder_models` the unfitted
Continuous and Continuous-Fragmented models, and :func:`fit_decoder_models` fits
them. :func:`validate_package_defaults` checks the built models against
:class:`Figure4PackageDefaults`, and :func:`get_spike_counts` bins spike times onto
the decode time grid.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
from numpy.typing import NDArray

from statespacecheck_paper.diagnostics import HPD_COVERAGE


def create_decoder_environment(
    track_graph: Any,
    edge_order: list[tuple[Any, Any]],
    edge_spacing: float | list[float],
    place_bin_size: float = 2.0,
) -> Any:
    """Create track environment for decoder models.

    Parameters
    ----------
    track_graph : networkx.Graph
        Track structure graph.
    edge_order : list[tuple]
        Edge ordering for linearization.
    edge_spacing : float or list[float]
        Gap inserted between consecutive edges of the linearized track.
    place_bin_size : float, default 2.0
        Spatial bin size in cm (Environment ``place_bin_size``). The default
        equals ``non_local_detector``'s own default and
        :attr:`Figure4DecoderConfig.position_bin_size_cm`.

    Returns
    -------
    env : Environment
        Track environment object.

    Raises
    ------
    ImportError
        If non_local_detector package is not available.

    Examples
    --------
    >>> # Requires non_local_detector package
    >>> # env = create_decoder_environment(track_graph, edge_order, edge_spacing)
    """
    try:
        from non_local_detector.environment import Environment
    except ImportError as e:
        raise ImportError(
            "non_local_detector package required. Install the project's locked "
            "environment with: make sync"
        ) from e

    return Environment(
        track_graph=track_graph,
        edge_order=edge_order,
        edge_spacing=edge_spacing,
        place_bin_size=place_bin_size,
    )


@dataclasses.dataclass(frozen=True)
class Figure4DecoderConfig:
    """Figure-4 decoder parameters that the construction code actually injects.

    Every field here is threaded into :func:`create_decoder_environment` /
    :func:`build_decoder_models` and genuinely controls the *scientific result*:
    changing one changes the decode. Contrast :class:`Figure4ExecutionConfig`
    (performance-only knobs that do not change the result) and
    :class:`Figure4PackageDefaults` (``non_local_detector`` defaults that are
    *recorded* and drift-guard pinned but deliberately **not** injected).

    Attributes
    ----------
    position_std : float
        Sorted-spikes KDE positional bandwidth, ``sqrt(12.5) ~= 3.54 cm``
        (``sorted_spikes_algorithm_params["position_std"]``).
    position_bin_size_cm : float
        Environment ``place_bin_size``, ``2 cm``.
    sampling_frequency_hz : float
        Decoder ``sampling_frequency`` ``500 Hz`` (i.e. ``2 ms`` spike bins).
    """

    position_std: float = float(np.sqrt(12.5))
    position_bin_size_cm: float = 2.0
    sampling_frequency_hz: float = 500.0

    def __post_init__(self) -> None:
        for name, value in (
            ("position_std", self.position_std),
            ("position_bin_size_cm", self.position_bin_size_cm),
            ("sampling_frequency_hz", self.sampling_frequency_hz),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"Figure4DecoderConfig.{name} must be finite and positive; got {value!r}"
                )


@dataclasses.dataclass(frozen=True)
class Figure4ExecutionConfig:
    """Figure-4 decoder execution knobs that do **not** change the decode result.

    These tune memory/performance only, so they are deliberately **not** hashed
    into the cache fingerprint: changing one must not invalidate a cached decode.

    Attributes
    ----------
    block_size : int
        KDE evaluation batch size (``sorted_spikes_algorithm_params["block_size"]``).
        ``non_local_detector`` partitions the KDE evaluation points into blocks of
        this size purely to bound memory; the density it returns is identical for
        any ``block_size`` (verified byte-identical for ``7`` vs ``100000``).
        Larger uses more memory and can be faster; smaller uses less.
    """

    block_size: int = 10000

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ValueError(
                f"Figure4ExecutionConfig.block_size must be a positive integer; "
                f"got {self.block_size!r}"
            )


@dataclasses.dataclass(frozen=True)
class Figure4PackageDefaults:
    """``non_local_detector`` defaults the Figure-4 decode relies on, recorded, not injected.

    These are ``non_local_detector`` class defaults that shape the decode. The
    code deliberately relies on those defaults rather than passing them
    explicitly: faithfully injecting them would require rebuilding the nested
    ``continuous_transition_types`` grid (a mix of ``RandomWalk`` and ``Uniform``),
    which risks silently changing the published decode. Instead they are pinned
    two ways:
    ``tests/test_figure04_decoder.py::TestFigure4ConfigMatchesManuscript`` asserts
    the *resolved* model attributes equal these values, and
    :func:`validate_package_defaults` re-checks them at decode time (runtime),
    so a dependency bump that moves a default fails loudly either way. They are
    also hashed into the cache fingerprint, so a recorded value changing
    invalidates the cache.

    The discrete-transition concentration and regularization are not recorded:
    ``non_local_detector`` reads them only when re-estimating the discrete
    transitions in ``estimate_parameters`` (EM), and Figure 4 only fits and
    predicts, so they cannot affect the decode.

    Attributes
    ----------
    movement_var : float
        Random-walk position-transition variance, ``6.0 cm^2`` (``RandomWalk``
        default).
    continuous_fragmented_diagonal_values : tuple[float, float]
        Continuous-Fragmented ``DiscreteStationaryDiagonal`` diagonal ``(0.98, 0.98)``
        (mode-transition matrix ``[[0.98, 0.02], [0.02, 0.98]]``).
    continuous_fragmented_discrete_initial_conditions : tuple[float, float]
        Continuous-Fragmented mode initial conditions ``(0.5, 0.5)``.
    non_local_detector_version : str
        Manuscript-stated ``non_local_detector`` version whose defaults these are.
    """

    movement_var: float = 6.0
    continuous_fragmented_diagonal_values: tuple[float, float] = (0.98, 0.98)
    continuous_fragmented_discrete_initial_conditions: tuple[float, float] = (0.5, 0.5)
    non_local_detector_version: str = "0.6.10.dev214+g956fdccaf"

    def __post_init__(self) -> None:
        if not np.isfinite(self.movement_var) or self.movement_var <= 0.0:
            raise ValueError(
                "Figure4PackageDefaults.movement_var must be finite and positive; "
                f"got {self.movement_var!r}"
            )
        for name, pair in (
            ("continuous_fragmented_diagonal_values", self.continuous_fragmented_diagonal_values),
            (
                "continuous_fragmented_discrete_initial_conditions",
                self.continuous_fragmented_discrete_initial_conditions,
            ),
        ):
            arr = np.asarray(pair, dtype=float)
            if (
                arr.shape != (2,)
                or not np.all(np.isfinite(arr))
                or np.any((arr < 0.0) | (arr > 1.0))
            ):
                raise ValueError(
                    f"Figure4PackageDefaults.{name} must be two finite probabilities in [0, 1]; "
                    f"got {pair!r}"
                )
        if not self.non_local_detector_version:
            raise ValueError("Figure4PackageDefaults.non_local_detector_version must be non-empty")


@dataclasses.dataclass(frozen=True)
class Figure4DiagnosticsConfig:
    """Figure-4 per-spike diagnostic settings, applied to the cached decode.

    These do not change the fitted models or their predictive distributions;
    they control how the per-spike diagnostics are computed from them. They are
    hashed into the *diagnostics* cache fingerprint (not the decode
    fingerprint), so changing one recomputes the diagnostics from the cached
    predictions without refitting.

    Attributes
    ----------
    hpd_coverage : float
        Coverage probability of the HPD regions compared by the HPD-overlap
        diagnostic; defaults to the paper's :data:`~statespacecheck_paper.diagnostics.HPD_COVERAGE`.
    event_selection : str
        Which spike events are diagnosed. ``"all_spikes_in_recording"`` means
        every spike of every unit whose timestamp lies within the decoded
        time grid; no behavioral, unit-quality, or training/validation mask is
        applied. Recorded so the cache identity states the selection rule.
    """

    hpd_coverage: float = HPD_COVERAGE
    event_selection: str = "all_spikes_in_recording"

    def __post_init__(self) -> None:
        if not (0.0 < self.hpd_coverage < 1.0):
            raise ValueError(
                f"Figure4DiagnosticsConfig.hpd_coverage must lie in (0, 1); "
                f"got {self.hpd_coverage!r}"
            )
        if self.event_selection != "all_spikes_in_recording":
            raise ValueError(
                "Figure4DiagnosticsConfig.event_selection must be "
                f"'all_spikes_in_recording'; got {self.event_selection!r}"
            )


@dataclasses.dataclass(frozen=True)
class Figure4Config:
    """Full Figure-4 configuration: the two decoders and their per-spike diagnostics.

    Both decoders (Continuous and Continuous-Fragmented) share one sorted-spikes
    kernel-density observation model. The parts say where each setting comes
    from: :attr:`decoder` holds the values this code passes to
    ``non_local_detector`` (KDE bandwidth, position bin size, time-bin rate);
    :attr:`package_defaults` records the ``non_local_detector`` defaults the
    decode relies on without passing them (the random-walk variance and the
    Continuous-Fragmented mode transitions and initial conditions);
    :attr:`diagnostics` sets how the per-spike diagnostics are computed from the
    predictions (HPD coverage, which spikes); and :attr:`execution` holds
    performance settings that do not change any result.

    Cache behavior: the decode cache fingerprint hashes :attr:`decoder` and
    :attr:`package_defaults`, so changing either refits; :attr:`diagnostics` is
    hashed into the separate diagnostics fingerprint, so changing it recomputes
    only the diagnostics; :attr:`execution` is hashed into neither (see
    :mod:`statespacecheck_paper.figure04_cache`).

    Attributes
    ----------
    decoder : Figure4DecoderConfig
        Parameters the construction code injects that control the decode result.
    package_defaults : Figure4PackageDefaults
        ``non_local_detector`` defaults recorded and drift-guard pinned, but not
        injected.
    execution : Figure4ExecutionConfig
        Performance/memory knobs that do not change the decode result and are not
        hashed into the fingerprint.
    diagnostics : Figure4DiagnosticsConfig
        Per-spike diagnostic settings applied to the cached decode.
    """

    decoder: Figure4DecoderConfig = dataclasses.field(default_factory=Figure4DecoderConfig)
    package_defaults: Figure4PackageDefaults = dataclasses.field(
        default_factory=Figure4PackageDefaults
    )
    execution: Figure4ExecutionConfig = dataclasses.field(default_factory=Figure4ExecutionConfig)
    diagnostics: Figure4DiagnosticsConfig = dataclasses.field(
        default_factory=Figure4DiagnosticsConfig
    )


def build_decoder_models(
    environment: Any,
    decoder_config: Figure4DecoderConfig | None = None,
    execution_config: Figure4ExecutionConfig | None = None,
) -> tuple[Any, Any]:
    """Construct the (unfitted) Continuous and Continuous-Fragmented decoder models.

    This holds the single source of decoder *construction* used by both
    :func:`fit_decoder_models` and the config drift guard. The
    :class:`Figure4DecoderConfig` values (``position_std``,
    ``sampling_frequency_hz``) and the :class:`Figure4ExecutionConfig`
    ``block_size`` are injected here; ``movement_var``, the mode-transition
    matrix, and the mode initial conditions come from ``non_local_detector``
    class defaults (see :class:`Figure4PackageDefaults` for why they are pinned rather than
    injected). The drift guard inspects the resolved attributes of these objects,
    so it never needs real data or a fit.

    Parameters
    ----------
    environment : Environment
        Track environment object. Its ``place_bin_size`` is set by
        :func:`create_decoder_environment` from the same config.
    decoder_config : Figure4DecoderConfig, optional
        Injected decoder parameters. Defaults to :class:`Figure4DecoderConfig`,
        the manuscript values. Its ``sampling_frequency_hz`` (500 Hz) equals
        the ``non_local_detector`` default, but its ``position_std``
        (``sqrt(12.5) ~= 3.54 cm``) replaces that package's default of 6.0 cm.
    execution_config : Figure4ExecutionConfig, optional
        Performance-only parameters (``block_size``). Defaults to
        :class:`Figure4ExecutionConfig`, whose ``block_size`` (10000) equals the
        ``non_local_detector`` default. Does not change the decode result.

    Returns
    -------
    continuous_model : SortedSpikesDecoder
        Unfitted continuous decoder model.
    continuous_fragmented_model : ContFragSortedSpikesClassifier
        Unfitted continuous-fragmented decoder model.

    Raises
    ------
    ImportError
        If non_local_detector package is not available.
    """
    try:
        from non_local_detector import (
            ContFragSortedSpikesClassifier,
            SortedSpikesDecoder,
        )
    except ImportError as e:
        raise ImportError(
            "non_local_detector package required. Install the project's locked "
            "environment with: make sync"
        ) from e

    if decoder_config is None:
        decoder_config = Figure4DecoderConfig()
    if execution_config is None:
        execution_config = Figure4ExecutionConfig()

    sorted_spikes_algorithm_params = {
        "block_size": execution_config.block_size,
        "position_std": decoder_config.position_std,
    }
    continuous_model = SortedSpikesDecoder(
        environments=[environment],
        sorted_spikes_algorithm_params=sorted_spikes_algorithm_params,
        sampling_frequency=decoder_config.sampling_frequency_hz,
    )
    continuous_fragmented_model = ContFragSortedSpikesClassifier(
        environments=[environment],
        sorted_spikes_algorithm_params=sorted_spikes_algorithm_params,
        sampling_frequency=decoder_config.sampling_frequency_hz,
    )
    return continuous_model, continuous_fragmented_model


def validate_package_defaults(
    continuous_model: Any,
    continuous_fragmented_model: Any,
    package_defaults: Figure4PackageDefaults | None = None,
) -> None:
    """Assert the built models still carry the recorded ``non_local_detector`` defaults.

    :class:`Figure4PackageDefaults` records nld class defaults that shape the decode but
    are deliberately not injected (see its docstring). A dependency bump could
    silently change one, producing a different published figure. This checks the
    *resolved* model attributes against the recorded values and raises at decode
    time (runtime), rather than relying only on the drift-guard test, so an
    unintended dependency change fails loudly instead of silently.

    Raises
    ------
    ValueError
        If any resolved model attribute diverges from the recorded default.
    """
    if package_defaults is None:
        package_defaults = Figure4PackageDefaults()

    scalar_checks: tuple[tuple[str, Any, float], ...] = (
        (
            "continuous movement_var",
            continuous_model.continuous_transition_types[0][0].movement_var,
            package_defaults.movement_var,
        ),
        (
            "continuous_fragmented movement_var",
            continuous_fragmented_model.continuous_transition_types[0][0].movement_var,
            package_defaults.movement_var,
        ),
    )
    for label, resolved, expected in scalar_checks:
        if not np.isclose(float(resolved), float(expected)):
            raise ValueError(
                f"non_local_detector default drift: {label} resolved to {resolved!r} but "
                f"Figure4PackageDefaults records {expected!r}. A dependency change moved a "
                "decode-shaping default; update Figure4PackageDefaults (and re-verify Figure 4) "
                "if this is intentional."
            )

    array_checks: tuple[tuple[str, Any, tuple[float, float]], ...] = (
        (
            "continuous_fragmented discrete_transition_type.diagonal_values",
            continuous_fragmented_model.discrete_transition_type.diagonal_values,
            package_defaults.continuous_fragmented_diagonal_values,
        ),
        (
            "continuous_fragmented discrete_initial_conditions",
            continuous_fragmented_model.discrete_initial_conditions,
            package_defaults.continuous_fragmented_discrete_initial_conditions,
        ),
    )
    for label, resolved, expected_pair in array_checks:
        if not np.allclose(
            np.asarray(resolved, dtype=float), np.asarray(expected_pair, dtype=float)
        ):
            raise ValueError(
                f"non_local_detector default drift: {label} resolved to "
                f"{np.asarray(resolved)!r} but Figure4PackageDefaults records {expected_pair!r}. "
                "A dependency change moved a decode-shaping default; update "
                "Figure4PackageDefaults (and re-verify Figure 4) if this is intentional."
            )


def fit_decoder_models(
    position: NDArray[np.float64],
    spike_times: list[NDArray[np.float64]],
    time: NDArray[np.float64],
    environment: Any,
    decoder_config: Figure4DecoderConfig | None = None,
    execution_config: Figure4ExecutionConfig | None = None,
) -> tuple[Any, Any]:
    """Fit Continuous and Continuous-Fragmented decoder models.

    Parameters
    ----------
    position : np.ndarray, shape (n_time,) or (n_time, n_dims)
        Position values. 1D arrays (linear position) are reshaped
        to (n_time, 1) before being passed to the model.
    spike_times : list[np.ndarray]
        List of spike time arrays, one per cell.
    time : np.ndarray, shape (n_time,)
        Time values corresponding to position.
    environment : Environment
        Track environment object.
    decoder_config : Figure4DecoderConfig, optional
        Injected decoder parameters passed to :func:`build_decoder_models`.
        Defaults to :class:`Figure4DecoderConfig`.
    execution_config : Figure4ExecutionConfig, optional
        Performance-only parameters passed to :func:`build_decoder_models`.
        Defaults to :class:`Figure4ExecutionConfig`.

    Returns
    -------
    continuous_model : SortedSpikesDecoder
        Fitted continuous decoder model.
    continuous_fragmented_model : ContFragSortedSpikesClassifier
        Fitted continuous-fragmented decoder model.

    Raises
    ------
    ImportError
        If non_local_detector package is not available.

    Examples
    --------
    >>> # Requires non_local_detector package and fitted environment
    >>> # continuous_model, continuous_fragmented_model = fit_decoder_models(
    >>> #     position, spike_times, time, environment
    >>> # )
    """
    continuous_model, continuous_fragmented_model = build_decoder_models(
        environment, decoder_config, execution_config
    )

    # Ensure position is 2D (n_time, 1) for the decoder
    position_2d = position.reshape(-1, 1) if position.ndim == 1 else position

    continuous_model.fit(position=position_2d, spike_times=spike_times, position_time=time)
    continuous_fragmented_model.fit(
        position=position_2d, spike_times=spike_times, position_time=time
    )

    return continuous_model, continuous_fragmented_model


def get_spike_counts(
    spike_times: list[NDArray[np.float64]],
    time: NDArray[np.float64],
) -> NDArray[np.int64]:
    """Get spike count matrix aligned to time bins.

    Parameters
    ----------
    spike_times : list[np.ndarray]
        List of spike time arrays, one per cell.
    time : np.ndarray, shape (n_time,)
        Decode time grid. Row ``i`` counts the spikes in
        ``[time[i], time[i + 1])``; the last of these intervals also includes
        ``time[-1]``.

    Returns
    -------
    spike_counts : np.ndarray, shape (n_time, n_cells)
        Spike count for each cell in each time bin. Spikes outside
        ``[time[0], time[-1]]`` are dropped, and the final row is always zero
        (``non_local_detector.likelihoods.common.get_spikecount_per_time_bin``).

    Raises
    ------
    ImportError
        If non_local_detector package is not available.

    Examples
    --------
    >>> # Requires non_local_detector package
    >>> # spike_counts = get_spike_counts(spike_times, time)
    >>> # spike_counts.shape  # (n_time, n_cells)
    """
    try:
        from non_local_detector.likelihoods.common import get_spikecount_per_time_bin
    except ImportError as e:
        raise ImportError(
            "non_local_detector package required. Install the project's locked "
            "environment with: make sync"
        ) from e

    counts_per_cell = [get_spikecount_per_time_bin(spike_times=st, time=time) for st in spike_times]
    spike_counts = np.stack(counts_per_cell, axis=1).astype(np.int64)

    return spike_counts
