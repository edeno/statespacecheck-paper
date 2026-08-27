"""Integration tests for figure generation scripts."""

from __future__ import annotations

import dataclasses
import importlib
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest

from statespacecheck_paper.diagnostics import SpikeEventDiagnostics

# Add scripts directory to path so we can import the figure scripts.
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


@pytest.fixture(autouse=True, scope="module")
def cleanup_sys_path() -> Iterator[None]:
    """Remove scripts directory from sys.path after the module's tests run."""
    yield
    if str(SCRIPTS_DIR) in sys.path:
        sys.path.remove(str(SCRIPTS_DIR))


# ---------------------------------------------------------------------------
# Per-figure script contract: each script defines an entry point and pulls
# from the shared style module. One parameterized test replaces four near-
# identical TestFigure*Integration classes.
# ---------------------------------------------------------------------------


_FIGURE_CONTRACT = [
    ("generate_figure01", "create_figure", ["COLORS", "save_figure"]),
    ("generate_figure02", "create_figure", ["save_figure"]),
    ("generate_figure03", "main", ["generate_figure03"]),
    (
        "generate_figure04",
        "run_demo",
        [
            "DATA_PATH",
            "ANIMAL_DATE_EPOCH",
            "DETAIL_CENTER",
            "DETAIL_HALF_WIDTH",
            "create_decoder_environment",
            "fit_decoder_models",
            "get_spike_counts",
            "compute_model_diagnostics",
            "plot_single_model_diagnostics",
        ],
    ),
]


@pytest.mark.parametrize(
    ("module_name", "entry_point", "required_attrs"),
    _FIGURE_CONTRACT,
    ids=[contract[0] for contract in _FIGURE_CONTRACT],
)
def test_figure_script_exports_expected_api(
    module_name: str, entry_point: str, required_attrs: list[str]
) -> None:
    """Each figure script must import cleanly, expose its entry point, and
    pull required utilities from shared modules — anything missing breaks
    ``generate_all_figures.py``."""
    module = importlib.import_module(module_name)
    assert callable(getattr(module, entry_point, None)), (
        f"{module_name}.{entry_point} must be callable"
    )
    missing = [name for name in required_attrs if not hasattr(module, name)]
    assert not missing, f"{module_name} missing attributes: {missing}"


# ---------------------------------------------------------------------------
# generate_figure04 helper functions: small focused logic that is hard to
# regression-test through the figure pipeline.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def figure04() -> ModuleType:
    return importlib.import_module("generate_figure04")


def _make_per_cell_diagnostics(
    *,
    event_time: np.ndarray | None,
    event_hpd_overlap: np.ndarray,
) -> SpikeEventDiagnostics:
    n_spikes = event_hpd_overlap.shape[0]
    return SpikeEventDiagnostics(
        event_time_ind=np.zeros(n_spikes, dtype=np.intp),
        event_cell_ind=np.zeros(n_spikes, dtype=np.intp),
        event_hpd_overlap=event_hpd_overlap,
        event_kl_divergence=np.zeros(n_spikes),
        event_predictive_pvalue=np.zeros(n_spikes),
        hpd_overlap=None,
        kl_divergence=None,
        predictive_pvalue=None,
        per_spike_likelihood=None,
        event_time=event_time,
    )


class TestFigure04Helpers:
    def test_shift_diagnostic_event_times_subtracts_offset(self, figure04: ModuleType) -> None:
        """Per-spike event times must be relative to the same time base as
        the figure axis — otherwise scatter points slide off the panels."""
        diagnostics = _make_per_cell_diagnostics(
            event_time=np.array([101.0, 101.5]),
            event_hpd_overlap=np.array([0.25, 0.75]),
        )
        shifted = figure04.shift_diagnostic_event_times(diagnostics, 100.0)
        np.testing.assert_allclose(shifted.event_time, [1.0, 1.5])
        # Original instance not mutated (frozen + write-protected).
        np.testing.assert_allclose(diagnostics.event_time, [101.0, 101.5])
        # Non-time arrays passed through by reference (zero-copy).
        assert shifted.event_hpd_overlap is diagnostics.event_hpd_overlap

    def test_shift_diagnostic_event_times_passthrough_when_none(self, figure04: ModuleType) -> None:
        """Simulated-data path leaves ``event_time`` as ``None``; the
        shift must be a no-op there, not raise."""
        diagnostics = _make_per_cell_diagnostics(event_time=None, event_hpd_overlap=np.array([0.5]))
        shifted = figure04.shift_diagnostic_event_times(diagnostics, 100.0)
        assert shifted is diagnostics

    def test_diagnostic_event_mean_uses_per_spike_array(self, figure04: ModuleType) -> None:
        """Summary mean must use per-spike values, not the (n_time, n_cells)
        matrix collapsed by nanmean — those answers differ when multiple
        spikes share a (time, cell)."""
        diagnostics = _make_per_cell_diagnostics(
            event_time=None,
            event_hpd_overlap=np.array([0.0, 1.0, 1.0]),
        )
        result = figure04.diagnostic_event_mean(diagnostics, "hpd_overlap")
        assert result == pytest.approx(2.0 / 3.0)

    def test_diagnostic_event_mean_raises_when_event_array_missing(
        self, figure04: ModuleType
    ) -> None:
        """Silently falling back to bin values would re-introduce the bug
        the per-spike array was created to fix; raise loudly instead."""
        diagnostics = _make_per_cell_diagnostics(event_time=None, event_hpd_overlap=np.array([0.5]))
        with pytest.raises(KeyError, match="event_made_up_metric"):
            figure04.diagnostic_event_mean(diagnostics, "made_up_metric")


def _synthetic_recording() -> dict[str, Any]:
    """A tiny in-memory stand-in for load_neural_recording_from_files output.

    Only the fields ``_load_or_compute_fig4_bundle`` reads are populated, so the
    cache / paths tests never touch the unpublished real-data export.
    """
    n_time = 8
    position_info = pd.DataFrame(
        {
            "head_position_x": np.linspace(0.0, 1.0, n_time),
            "head_position_y": np.linspace(1.0, 0.0, n_time),
            "linear_position": np.linspace(0.0, 2.0, n_time),
        },
        index=np.linspace(0.0, 0.014, n_time),
    )
    return {
        "position_info": position_info,
        "spike_times": [np.array([0.001, 0.005]), np.array([0.010])],
        "track_graph": object(),
        "linear_edge_order": [(0, 1)],
        "linear_edge_spacing": 0.0,
    }


def _synthetic_payload() -> dict[str, Any]:
    """A joblib-serializable decode payload matching the cache/bundle keys."""
    return {
        "continuous_results": np.zeros(3),
        "contfrag_results": np.ones(3),
        "continuous_diagnostics": {"tag": "cont"},
        "contfrag_diagnostics": {"tag": "cf"},
        "spike_counts": np.zeros((8, 2), dtype=np.int64),
        "place_field_peaks": np.zeros(2),
        "diagnostic_place_fields": np.zeros((2, 4)),
        "diagnostic_position_bins": np.arange(4.0),
    }


class TestFig4BundleCacheAndPaths:
    """Provenance cache + path injection for ``_load_or_compute_fig4_bundle``,
    exercised entirely on synthetic inputs (no real data, no decoder)."""

    def test_fig4_cache_invalidates_on_config_change(
        self,
        figure04: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """A changed config fingerprint forces recompute; an identical one hits
        the cache. Guards against a stale cache silently outliving a decoder
        parameter (or dependency) change."""
        monkeypatch.setattr(
            figure04, "load_neural_recording_from_files", lambda *a, **k: _synthetic_recording()
        )
        compute_calls = {"n": 0}

        def fake_compute(**kwargs: Any) -> dict[str, Any]:
            compute_calls["n"] += 1
            return _synthetic_payload()

        monkeypatch.setattr(figure04, "_compute_fig4_decode_payload", fake_compute)

        paths = figure04.Fig4Paths(data_path=tmp_path, animal_date_epoch="synthetic_epoch")
        config = figure04.Figure4Config()

        # First call: no cache file yet -> compute + write.
        figure04._load_or_compute_fig4_bundle(config, paths, use_cache=True)
        assert compute_calls["n"] == 1
        assert paths.cache_path.exists()

        # Second call, identical config -> fingerprint match -> cache hit.
        figure04._load_or_compute_fig4_bundle(config, paths, use_cache=True)
        assert compute_calls["n"] == 1

        # Changed config -> fingerprint mismatch -> recompute.
        changed = dataclasses.replace(config, movement_var=config.movement_var + 1.0)
        figure04._load_or_compute_fig4_bundle(changed, paths, use_cache=True)
        assert compute_calls["n"] == 2

    def test_fig4_cache_invalidates_on_dependency_change(
        self,
        figure04: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """A changed installed ``non_local_detector`` revision invalidates the
        cache even when the config is unchanged."""
        monkeypatch.setattr(
            figure04, "load_neural_recording_from_files", lambda *a, **k: _synthetic_recording()
        )
        compute_calls = {"n": 0}

        def fake_compute(**kwargs: Any) -> dict[str, Any]:
            compute_calls["n"] += 1
            return _synthetic_payload()

        monkeypatch.setattr(figure04, "_compute_fig4_decode_payload", fake_compute)
        monkeypatch.setattr(figure04, "_installed_non_local_detector_version", lambda: "1.0.0")

        paths = figure04.Fig4Paths(data_path=tmp_path, animal_date_epoch="synthetic_epoch")
        config = figure04.Figure4Config()

        figure04._load_or_compute_fig4_bundle(config, paths, use_cache=True)
        assert compute_calls["n"] == 1

        # Simulate a dependency bump: same config, different installed version.
        monkeypatch.setattr(figure04, "_installed_non_local_detector_version", lambda: "2.0.0")
        figure04._load_or_compute_fig4_bundle(config, paths, use_cache=True)
        assert compute_calls["n"] == 2

    def test_fig4_bundle_uses_injected_paths(
        self,
        figure04: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """The compute/load layer must read only the injected paths, never the
        module-global DATA_PATH / ANIMAL_DATE_EPOCH."""
        seen: dict[str, Any] = {}

        def spy_load(data_path: Any, animal_date_epoch: Any) -> dict[str, Any]:
            seen["data_path"] = data_path
            seen["animal_date_epoch"] = animal_date_epoch
            return _synthetic_recording()

        monkeypatch.setattr(figure04, "load_neural_recording_from_files", spy_load)
        monkeypatch.setattr(
            figure04, "_compute_fig4_decode_payload", lambda **k: _synthetic_payload()
        )
        # Poison the module globals: if the function reads them instead of the
        # injected paths, the assertions below fail.
        monkeypatch.setattr(figure04, "DATA_PATH", Path("/nonexistent/global/path"))
        monkeypatch.setattr(figure04, "ANIMAL_DATE_EPOCH", "WRONG_GLOBAL_EPOCH")

        injected = figure04.Fig4Paths(data_path=tmp_path, animal_date_epoch="injected_epoch")
        bundle = figure04._load_or_compute_fig4_bundle(
            figure04.Figure4Config(), injected, use_cache=False
        )

        assert seen == {"data_path": tmp_path, "animal_date_epoch": "injected_epoch"}
        assert (
            injected.cache_path == tmp_path / "intermediates" / "injected_epoch_fig4_cache.joblib"
        )
        assert injected.cache_path.exists()  # cache written under the injected path
        # Bundle carries the freshly-loaded render data.
        assert list(bundle.spike_times_list)  # non-empty
        assert bundle.spike_counts.shape == (8, 2)


def test_figure02_create_shared_example_samples_y_tilde_with_noise() -> None:
    """The Figure 2 predictive-check MC loop must draw y_tilde from
    N(x_s, like_std), not use x_s as the observation. That step is the
    only thing distinguishing the corrected schematic from the previous
    mean-prediction shortcut, so a regression that quietly reverted it
    would land silently.
    """
    import generate_figure02

    rng = np.random.default_rng(42)
    data = generate_figure02.create_shared_example(rng)

    p_value = data["p_value"]
    assert 0.0 <= p_value <= 1.0, f"p_value out of [0, 1]: {p_value}"

    observed = data["observed_log_pred"]
    simulated = data["simulated_log_pred"]
    assert np.isfinite(observed), f"observed_log_pred is not finite: {observed}"
    assert np.all(np.isfinite(simulated)), (
        f"simulated_log_pred contains non-finite values: "
        f"{np.sum(~np.isfinite(simulated))} of {simulated.size}"
    )

    positions = np.asarray(data["showcase_positions"])
    y_tildes = np.asarray(data["showcase_y_tildes"])
    assert positions.shape == y_tildes.shape, (
        "showcase_positions and showcase_y_tildes must have the same shape"
    )
    # Load-bearing assertion: y_tilde must differ from its originating
    # state position by more than rounding (~1 bin width = 0.5). If every
    # y_tilde sits exactly on its sample position, the MC loop has been
    # reverted to the deterministic y_tilde = x_s shortcut and the
    # manuscript's predictive-check definition is no longer depicted.
    deltas = np.abs(y_tildes - positions)
    assert np.any(deltas > 0.5), (
        f"showcase_y_tildes equal showcase_positions (max |Δ| = {deltas.max():.3f}); "
        f"the y_tilde ~ N(x_s, like_std) draw step was skipped or shortcut."
    )


def test_figure02_panels_module_is_load_bearing() -> None:
    """After the figure-02 extraction, the script must import its panel
    renderers from ``statespacecheck_paper.figure02_panels``. A revert
    that inlined the panels back into the script would silently pass
    every other check; this test pins the architectural decision."""
    import generate_figure02

    panel_module = "statespacecheck_paper.figure02_panels"
    assert panel_module in sys.modules, (
        f"generate_figure02 did not import {panel_module}; "
        f"the figure-02 extraction may have been undone."
    )
    # And the script must re-export at least one panel symbol pulled
    # from that module, so callers (e.g. notebook code in the repo)
    # importing the script keep working.
    assert hasattr(generate_figure02, "plot_kl_distributions"), (
        "generate_figure02 must re-export plot_kl_distributions from figure02_panels"
    )


def test_figure02_create_figure_invokes_all_panels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end smoke test: ``create_figure`` must run the 9 panel
    renderers + the 2 shared helpers without raising. Without this,
    any panel could ``raise`` on every invocation and only manual
    figure regeneration would notice — the existing tests don't
    invoke the entry point.

    Redirect the figure write to a tmp_path so we don't touch the
    real ``manuscript/figures/main/`` artifacts. The actual byte-
    identical check lives in the figure-3 SHA workflow.
    """
    import generate_figure02

    # Redirect ``save_figure`` to write into a tmp directory.
    out_dir = tmp_path / "fig02"

    def _save(name: str | Path, **kwargs: object) -> None:
        target = out_dir / Path(name).name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.with_suffix(".pdf").touch()
        target.with_suffix(".png").touch()

    monkeypatch.setattr(generate_figure02, "save_figure", _save)
    generate_figure02.create_figure()  # does not raise
    # The redirected ``save_figure`` is called once; the smoke test's
    # job is to surface a panel-renderer regression, not to verify
    # disk-writing semantics.
    assert (out_dir / "figure02.png").exists()
