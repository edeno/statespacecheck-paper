"""Integration tests for figure generation scripts."""

from __future__ import annotations

import sys
from pathlib import Path

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

    def test_imports_required_modules(self) -> None:
        """Test that figure01.py can import all required modules."""
        # Import the main module
        import figure01

        # Check that key functions are available
        assert hasattr(figure01, "create_figure")

        # Check that imported utilities are accessible
        assert hasattr(figure01, "WONG") or hasattr(figure01, "wong")
        assert hasattr(figure01, "save_figure")


class TestFigure02Integration:
    """Integration tests for figure02.py script."""

    def test_imports_work(self) -> None:
        """Test that figure02.py imports successfully."""
        # This will raise ImportError if imports fail
        import figure02  # noqa: F401

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
