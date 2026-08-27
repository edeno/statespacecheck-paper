"""Figure-4 decoder configuration and construction.

Holds the Figure-4 decode configuration objects (:class:`Figure4Config`,
:class:`Figure4DecoderConfig`, :class:`Figure4Provenance`) and the functions that
build and fit the Continuous / Continuous-Fragmented decoder models from
``non_local_detector``, plus the spike-count helper.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
from numpy.typing import NDArray


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
        Spacing between nodes.
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
            "non_local_detector package required. Install with: pip install non_local_detector"
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
    :class:`Figure4Provenance` (``non_local_detector`` defaults that are
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

    @property
    def time_bin_size_ms(self) -> float:
        """Spike time-bin size in milliseconds (``1000 / sampling_frequency``)."""
        return 1000.0 / self.sampling_frequency_hz


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


@dataclasses.dataclass(frozen=True)
class Figure4Provenance:
    """Figure-4 decode parameters recorded for provenance but **not** injected.

    These are ``non_local_detector`` class defaults that shape the decode. The
    code deliberately relies on those defaults rather than passing them
    explicitly: faithfully injecting them would require rebuilding the nested
    ``continuous_transition_types`` grid (a mix of ``RandomWalk`` and ``Uniform``)
    and would hit the concentration-default split (``1.0`` for the continuous
    decoder, ``1.1`` for the ContFrag classifier) -- either of which risks
    silently changing the published decode. Instead they are pinned by
    ``tests/test_figure04_decoder.py::TestFigure4ConfigMatchesManuscript``,
    which asserts the *resolved* model attributes still equal these values, so a
    dependency bump that moves a default fails loudly. They are also hashed into
    the cache fingerprint, so a recorded value changing invalidates the cache.

    Attributes
    ----------
    movement_var : float
        Random-walk position-transition variance, ``6.0 cm^2`` (``RandomWalk``
        default).
    contfrag_diagonal_values : tuple[float, float]
        ContFrag ``DiscreteStationaryDiagonal`` diagonal ``(0.98, 0.98)``
        (mode-transition matrix ``[[0.98, 0.02], [0.02, 0.98]]``).
    contfrag_discrete_initial_conditions : tuple[float, float]
        ContFrag mode initial conditions ``(0.5, 0.5)``.
    discrete_transition_concentration : float
        ContFrag Dirichlet concentration (unprinted effective default ``1.1``;
        the continuous decoder's own default is ``1.0``).
    discrete_transition_regularization : float
        Discrete-transition regularization (unprinted default ``1e-10``).
    non_local_detector_version : str
        Manuscript-stated ``non_local_detector`` version, for provenance.
    """

    movement_var: float = 6.0
    contfrag_diagonal_values: tuple[float, float] = (0.98, 0.98)
    contfrag_discrete_initial_conditions: tuple[float, float] = (0.5, 0.5)
    discrete_transition_concentration: float = 1.1
    discrete_transition_regularization: float = 1e-10
    non_local_detector_version: str = "0.6.10.dev214+g956fdccaf"


@dataclasses.dataclass(frozen=True)
class Figure4Config:
    """Full Figure-4 decode configuration: injected knobs + recorded provenance.

    Split into clearly-scoped parts so a reader can tell which parameters drive
    the scientific result (:attr:`decoder`), which are recorded-but-not-injected
    provenance (:attr:`provenance`), and which are performance-only
    (:attr:`execution`). The cache fingerprint hashes :attr:`decoder` and
    :attr:`provenance` -- changing either invalidates the cache -- but **not**
    :attr:`execution`, whose values do not change the decode result (see
    :func:`figure04_cache.compute_figure04_cache_fingerprint`).

    Attributes
    ----------
    decoder : Figure4DecoderConfig
        Parameters the construction code injects that control the decode result.
    provenance : Figure4Provenance
        ``non_local_detector`` defaults recorded and drift-guard pinned, but not
        injected.
    execution : Figure4ExecutionConfig
        Performance/memory knobs that do not change the decode result and are not
        hashed into the fingerprint.
    """

    decoder: Figure4DecoderConfig = dataclasses.field(default_factory=Figure4DecoderConfig)
    provenance: Figure4Provenance = dataclasses.field(default_factory=Figure4Provenance)
    execution: Figure4ExecutionConfig = dataclasses.field(default_factory=Figure4ExecutionConfig)


def build_decoder_models(
    environment: Any,
    decoder_config: Figure4DecoderConfig | None = None,
    execution_config: Figure4ExecutionConfig | None = None,
) -> tuple[Any, Any]:
    """Construct the (unfitted) Continuous and ContFrag decoder models.

    This holds the single source of decoder *construction* used by both
    :func:`fit_decoder_models` and the config drift guard. The
    :class:`Figure4DecoderConfig` values (``position_std``,
    ``sampling_frequency_hz``) and the :class:`Figure4ExecutionConfig`
    ``block_size`` are injected here; ``movement_var``, the mode-transition
    matrix, the mode initial conditions, and the discrete-transition
    concentration / regularization all come from ``non_local_detector`` class
    defaults (see :class:`Figure4Provenance` for why they are pinned rather than
    injected). The drift guard inspects the resolved attributes of these objects,
    so it never needs real data or a fit.

    Parameters
    ----------
    environment : Environment
        Track environment object. Its ``place_bin_size`` is set by
        :func:`create_decoder_environment` from the same config.
    decoder_config : Figure4DecoderConfig, optional
        Injected decoder parameters. Defaults to :class:`Figure4DecoderConfig`
        (the manuscript values, which equal the ``non_local_detector`` defaults).
    execution_config : Figure4ExecutionConfig, optional
        Performance-only parameters (``block_size``). Defaults to
        :class:`Figure4ExecutionConfig`. Does not change the decode result.

    Returns
    -------
    continuous_model : SortedSpikesDecoder
        Unfitted continuous decoder model.
    contfrag_model : ContFragSortedSpikesClassifier
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
            "non_local_detector package required. Install with: pip install non_local_detector"
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
    contfrag_model = ContFragSortedSpikesClassifier(
        environments=[environment],
        sorted_spikes_algorithm_params=sorted_spikes_algorithm_params,
        sampling_frequency=decoder_config.sampling_frequency_hz,
    )
    return continuous_model, contfrag_model


def fit_decoder_models(
    position: NDArray[np.float64],
    spike_times: list[NDArray[np.float64]],
    time: NDArray[np.float64],
    environment: Any,
    decoder_config: Figure4DecoderConfig | None = None,
    execution_config: Figure4ExecutionConfig | None = None,
) -> tuple[Any, Any]:
    """Fit Continuous and ContFrag decoder models.

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
    contfrag_model : ContFragSortedSpikesClassifier
        Fitted continuous-fragmented decoder model.

    Raises
    ------
    ImportError
        If non_local_detector package is not available.

    Examples
    --------
    >>> # Requires non_local_detector package and fitted environment
    >>> # continuous_model, contfrag_model = fit_decoder_models(
    >>> #     position, spike_times, time, environment
    >>> # )
    """
    continuous_model, contfrag_model = build_decoder_models(
        environment, decoder_config, execution_config
    )

    # Ensure position is 2D (n_time, 1) for the decoder
    position_2d = position.reshape(-1, 1) if position.ndim == 1 else position

    continuous_model.fit(position=position_2d, spike_times=spike_times, position_time=time)
    contfrag_model.fit(position=position_2d, spike_times=spike_times, position_time=time)

    return continuous_model, contfrag_model


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
        Time bin centers.

    Returns
    -------
    spike_counts : np.ndarray, shape (n_time, n_cells)
        Spike count for each cell at each time bin.

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
            "non_local_detector package required. Install with: pip install non_local_detector"
        ) from e

    counts_per_cell = [get_spikecount_per_time_bin(spike_times=st, time=time) for st in spike_times]
    spike_counts = np.stack(counts_per_cell, axis=1).astype(np.int64)

    return spike_counts
