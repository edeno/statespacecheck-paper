"""Integration tests for figure generation scripts."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

# Add figures directory to path to import figure scripts
FIGURES_DIR = Path(__file__).parent.parent / "figures"
sys.path.insert(0, str(FIGURES_DIR))


class TestFigure01Integration:
    """Integration tests for figure01.py script."""

    def test_imports_work(self) -> None:
        """Test that figure01.py imports successfully."""
        # This will raise ImportError if imports fail
        import figure01  # noqa: F401

    def test_create_figure_runs(self) -> None:
        """Test that create_figure() runs without errors."""
        from figure01 import create_figure

        # Note: create_figure() saves and closes the figure internally,
        # so it returns None. We just verify it runs without errors.
        result = create_figure()

        # Assert - function completes successfully
        assert result is None  # Expected behavior

    def test_creates_expected_output_files(self) -> None:
        """Test that figure01.py creates expected output files."""
        from figure01 import create_figure

        # Get expected output paths
        expected_pdf = FIGURES_DIR.parent / "figures" / "figure01.pdf"
        expected_png = FIGURES_DIR.parent / "figures" / "figure01.png"

        # Remove old files if they exist
        expected_pdf.unlink(missing_ok=True)
        expected_png.unlink(missing_ok=True)

        # Create figure (saves files internally)
        create_figure()

        # Assert files were created
        assert expected_pdf.exists(), f"Expected {expected_pdf} to exist"
        assert expected_png.exists(), f"Expected {expected_png} to exist"


class TestFigure02Integration:
    """Integration tests for figure02.py script."""

    def test_imports_work(self) -> None:
        """Test that figure02.py imports successfully."""
        # This will raise ImportError if imports fail
        import figure02  # noqa: F401

    def test_run_demo_with_small_params(self) -> None:
        """Test that run_demo() runs with small parameters."""
        from figure02 import DecodeParams, run_demo

        # Create minimal parameters for fast test
        # Keep spatial resolution same as defaults to avoid remap index issues
        # But reduce timeline for speed
        params = DecodeParams(
            T_remap_start=600,
            T_remap_end=1000,
            T_recovery1_end=1400,
            T_flat_end=1600,
            T_recovery2_end=2000,
            T_fast_end=2400,
            T_recovery3_end=2800,
            T_slow_end=3200,
            # Keep default spatial params (xs_min=0, xs_max=100, xs_step=1)
            # to avoid remapping index issues
        )

        # Execute - should not raise any errors
        # Note: This will create actual figure files and returns None
        result = run_demo(params)

        # Assert function completes successfully (returns None)
        assert result is None  # Expected behavior

    def test_imports_required_modules(self) -> None:
        """Test that figure02.py can import all required modules."""
        # Import the main module
        import figure02

        # Check that key functions are available
        assert hasattr(figure02, "run_demo")
        assert hasattr(figure02, "DecodeParams")

        # Check that imported utilities are accessible
        assert hasattr(figure02, "simulate_walk")
        assert hasattr(figure02, "decode_and_diagnostics")
        assert hasattr(figure02, "plot_combined_diagnostics")


class TestFiguresModuleStructure:
    """Tests for overall figure module structure and consistency."""

    def test_both_figures_use_shared_style(self) -> None:
        """Test that both figures import from shared style module."""
        import figure01
        import figure02

        # Both should import WONG colors from style module
        assert hasattr(figure01, "WONG") or hasattr(figure01, "wong")
        # figure02 uses functions that internally use WONG

        # Both should use save_figure function
        assert hasattr(figure01, "save_figure")
        assert hasattr(figure02, "save_figure")

    def test_both_figures_are_executable(self) -> None:
        """Test that both figure scripts have proper structure."""
        import figure01
        import figure02

        # figure01 should have create_figure function
        assert callable(getattr(figure01, "create_figure", None))

        # figure02 should have run_demo function
        assert callable(getattr(figure02, "run_demo", None))


# Cleanup sys.path after tests
@pytest.fixture(autouse=True, scope="module")
def cleanup_sys_path() -> None:
    """Remove figures directory from sys.path after tests."""
    yield
    if str(FIGURES_DIR) in sys.path:
        sys.path.remove(str(FIGURES_DIR))
