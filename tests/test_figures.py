"""Integration tests for figure generation scripts."""

from __future__ import annotations

import importlib
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm

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
    ("generate_figure01", "main", ["generate_figure01"]),
    ("generate_figure02", "main", ["generate_figure02"]),
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
    from statespacecheck_paper.figure02_panels import create_shared_example

    rng = np.random.default_rng(42)
    data = create_shared_example(rng)

    p_value = data.p_value
    assert 0.0 <= p_value <= 1.0, f"p_value out of [0, 1]: {p_value}"

    observed = data.observed_log_pred
    simulated = data.simulated_log_pred
    assert np.isfinite(observed), f"observed_log_pred is not finite: {observed}"
    assert np.all(np.isfinite(simulated)), (
        f"simulated_log_pred contains non-finite values: "
        f"{np.sum(~np.isfinite(simulated))} of {simulated.size}"
    )

    # The observed predictive density is the raw observation-model mixture
    # Σ_x P(x) p(y=60 | x), not an inner product with a likelihood normalized
    # across x. This differs near a finite grid boundary and guards the units of
    # the predictive check even though the plotted HPD/KL likelihood is normalized.
    raw_observation_density = norm.pdf(
        data.position_bins,
        loc=60.0,
        scale=12.0,
    )
    expected_observed = np.log(np.sum(data.predictive * raw_observation_density))
    assert observed == pytest.approx(expected_observed, rel=1e-13)

    positions = np.asarray(data.showcase_positions)
    y_tildes = np.asarray(data.showcase_y_tildes)
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
    assert not data.predictive.flags.writeable
    assert not data.showcase_likelihoods.flags.writeable
    with pytest.raises(ValueError, match="p_value must lie"):
        replace(data, p_value=float("nan"))


def test_figure02_panels_module_is_load_bearing() -> None:
    """After the figure-02 extraction, the generation recipe imports its panel
    renderers from ``statespacecheck_paper.figure02_panels``. A revert
    that inlined the panels back into the script would silently pass
    every other check; this test pins the architectural decision."""
    import statespacecheck_paper.figure02_generation as figure02_generation

    panel_module = "statespacecheck_paper.figure02_panels"
    assert panel_module in sys.modules, (
        f"figure02_generation did not import {panel_module}; "
        f"the figure-02 extraction may have been undone."
    )
    assert hasattr(figure02_generation, "plot_kl_distributions"), (
        "figure02_generation must compose renderers from figure02_panels"
    )


def test_figure02_generation_invokes_all_panels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end smoke test: ``generate_figure02`` runs all nine panel
    renderers + the 2 shared helpers without raising. Without this,
    any panel could ``raise`` on every invocation and only manual
    figure regeneration would notice — the existing tests don't
    invoke the entry point.

    Redirect the figure write to a tmp_path so we don't touch the
    real ``manuscript/figures/main/`` artifacts. This is a run-without-raising
    smoke test; it does not assert byte-identical figure output.
    """
    import statespacecheck_paper.figure02_generation as figure02_generation

    # Redirect ``save_figure`` to write into a tmp directory.
    out_dir = tmp_path / "fig02"

    def _save(name: str | Path, **kwargs: object) -> None:
        target = out_dir / Path(name).name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.with_suffix(".pdf").touch()
        target.with_suffix(".png").touch()

    monkeypatch.setattr(figure02_generation, "save_figure", _save)
    figure02_generation.generate_figure02()  # does not raise
    # The redirected ``save_figure`` is called once; the smoke test's
    # job is to surface a panel-renderer regression, not to verify
    # disk-writing semantics.
    assert (out_dir / "figure02.png").exists()


def test_figure01_composition_renders_content_in_every_panel() -> None:
    """The Figure-1 recipe exposes its in-memory composition separately, and
    every panel must actually draw something. A bare axis-count check would pass
    even if the schematic, equation boxes, or a distribution panel silently
    rendered blank (e.g. a renderer no-op'ing after a refactor)."""
    import matplotlib.pyplot as plt

    from statespacecheck_paper.figure01_generation import compose_figure01

    fig = compose_figure01()
    try:
        axes = fig.axes
        assert len(axes) >= 7

        def _has_content(ax: object) -> bool:
            return bool(ax.lines or ax.patches or ax.collections or ax.images or ax.texts)

        blank = [i for i, ax in enumerate(axes) if not _has_content(ax)]
        # At most one axis may be an empty layout container (the distribution-panel
        # parent that only holds inset sub-axes); every actual panel must draw.
        assert len(blank) <= 1, f"Figure-1 axes with no drawn content: {blank}"
    finally:
        plt.close(fig)


def test_generate_all_runs_each_cli_in_a_normal_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The all-figures command executes CLI files normally, never via ``exec``."""
    import generate_all_figures

    calls: list[list[str]] = []

    def _run(command: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        assert check is False
        return subprocess.CompletedProcess(command, returncode=0)

    monkeypatch.setattr(generate_all_figures.subprocess, "run", _run)
    assert generate_all_figures.main() == 0
    assert [Path(command[1]).name for command in calls] == [
        "generate_figure01.py",
        "generate_figure02.py",
        "generate_figure03.py",
        "generate_figure04.py",
    ]
