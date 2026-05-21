"""Integration tests for figure generation scripts."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

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
    ("generate_figure02", "create_figure", ["COLORS", "save_figure"]),
    (
        "generate_figure03",
        "run_demo",
        [
            "DecodeParams",
            "simulate_walk",
            "decode_and_diagnostics",
            "plot_combined_diagnostics",
        ],
    ),
    (
        "generate_figure04",
        "run_demo",
        [
            "DATA_PATH",
            "ANIMAL_DATE_EPOCH",
            "FIGURE_4A_CONTEXT_CENTER",
            "FIGURE_4A_CONTEXT_HALF_WIDTH",
            "FIGURE_4B_DETAIL_CENTER",
            "FIGURE_4B_DETAIL_HALF_WIDTH",
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


class TestFigure04Helpers:
    def test_shift_diagnostic_event_times_subtracts_offset(self, figure04: ModuleType) -> None:
        """Per-spike event times must be relative to the same time base as
        the figure axis — otherwise scatter points slide off the panels."""
        diagnostics = {
            "event_time": np.array([101.0, 101.5]),
            "event_hpd_overlap": np.array([0.25, 0.75]),
        }
        shifted = figure04.shift_diagnostic_event_times(diagnostics, 100.0)
        np.testing.assert_allclose(shifted["event_time"], [1.0, 1.5])
        # Original dict not mutated.
        np.testing.assert_allclose(diagnostics["event_time"], [101.0, 101.5])
        # Non-time arrays passed through by reference (zero-copy).
        assert shifted["event_hpd_overlap"] is diagnostics["event_hpd_overlap"]

    def test_diagnostic_event_mean_uses_per_spike_array(self, figure04: ModuleType) -> None:
        """Summary mean must use per-spike values, not the (n_time, n_cells)
        matrix collapsed by nanmean — those answers differ when multiple
        spikes share a (time, cell)."""
        diagnostics = {
            "hpd_overlap": np.array([[0.0, np.nan], [1.0, np.nan]]),
            "event_hpd_overlap": np.array([0.0, 1.0, 1.0]),
        }
        result = figure04.diagnostic_event_mean(diagnostics, "hpd_overlap")
        assert result == pytest.approx(2.0 / 3.0)

    def test_diagnostic_event_mean_raises_when_event_array_missing(
        self, figure04: ModuleType
    ) -> None:
        """Silently falling back to bin values would re-introduce the bug
        the per-spike array was created to fix; raise loudly instead."""
        with pytest.raises(KeyError, match="event_hpd_overlap"):
            figure04.diagnostic_event_mean({"hpd_overlap": np.array([[0.0, 1.0]])}, "hpd_overlap")


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
