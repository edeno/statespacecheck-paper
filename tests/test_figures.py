"""Integration tests for figure generation scripts."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator
from pathlib import Path

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
    ("generate_figure02", "create_figure", ["save_figure"]),
    ("generate_figure03", "main", ["generate_figure03"]),
    ("generate_figure04", "main", ["generate_figure04"]),
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
