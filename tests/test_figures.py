"""Integration tests for figure generation scripts."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add scripts directory to path to import figure scripts
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


class TestFigure01Integration:
    """Integration tests for generate_figure01.py script."""

    def test_imports_work(self) -> None:
        """Test that generate_figure01.py imports successfully."""
        # This will raise ImportError if imports fail
        import generate_figure01  # noqa: F401

    def test_imports_required_modules(self) -> None:
        """Test that generate_figure01.py can import all required modules."""
        # Import the main module
        import generate_figure01

        # Check that key functions are available
        assert hasattr(generate_figure01, "create_figure")

        # Check that imported utilities are accessible
        assert hasattr(generate_figure01, "WONG") or hasattr(generate_figure01, "wong")
        assert hasattr(generate_figure01, "save_figure")


class TestFigure02Integration:
    """Integration tests for generate_figure02.py script."""

    def test_imports_work(self) -> None:
        """Test that generate_figure02.py imports successfully."""
        # This will raise ImportError if imports fail
        import generate_figure02  # noqa: F401

    def test_imports_required_modules(self) -> None:
        """Test that generate_figure02.py can import all required modules."""
        # Import the main module
        import generate_figure02

        # Check that key functions are available
        assert hasattr(generate_figure02, "run_demo")
        assert hasattr(generate_figure02, "DecodeParams")

        # Check that imported utilities are accessible
        assert hasattr(generate_figure02, "simulate_walk")
        assert hasattr(generate_figure02, "decode_and_diagnostics")
        assert hasattr(generate_figure02, "plot_combined_diagnostics")


class TestFigure03Integration:
    """Integration tests for generate_figure03_candidates.py script."""

    def test_imports_work(self) -> None:
        """Test that generate_figure03_candidates.py imports successfully."""
        # This will raise ImportError if imports fail
        import generate_figure03_candidates  # noqa: F401

    def test_imports_required_modules(self) -> None:
        """Test that generate_figure03_candidates.py can import all required modules."""
        # Import the main module
        import generate_figure03_candidates

        # Check that key functions are available
        assert hasattr(generate_figure03_candidates, "main")
        assert hasattr(generate_figure03_candidates, "fit_and_predict_models")
        assert hasattr(generate_figure03_candidates, "save_intermediate_results")
        assert hasattr(generate_figure03_candidates, "load_intermediate_results")

        # Check that constants are defined
        assert hasattr(generate_figure03_candidates, "EXAMPLE_CONTEXT_SAMPLES")
        assert hasattr(generate_figure03_candidates, "LOW_OVERLAP_THRESHOLD")
        assert hasattr(generate_figure03_candidates, "MODERATE_OVERLAP_THRESHOLD")
        assert hasattr(generate_figure03_candidates, "SAMPLING_FREQUENCY")

    def test_has_all_figure_generation_functions(self) -> None:
        """Test that all figure generation functions are present."""
        import generate_figure03_candidates

        # Check all 7 figure generation functions exist
        assert hasattr(generate_figure03_candidates, "generate_lowest_overlap_examples")
        assert hasattr(generate_figure03_candidates, "generate_highest_difference_examples")
        assert hasattr(generate_figure03_candidates, "generate_sustained_region_figures")
        assert hasattr(generate_figure03_candidates, "generate_covariate_figures")
        assert hasattr(generate_figure03_candidates, "generate_mechanics_figures")
        assert hasattr(generate_figure03_candidates, "generate_ahead_behind_figures")
        assert hasattr(generate_figure03_candidates, "generate_fragmented_state_figures")


class TestFiguresModuleStructure:
    """Tests for overall figure module structure and consistency."""

    def test_all_figures_use_shared_style(self) -> None:
        """Test that all figures import from shared style module."""
        import generate_figure01
        import generate_figure02

        # Both should import WONG colors from style module
        assert hasattr(generate_figure01, "WONG") or hasattr(generate_figure01, "wong")
        # generate_figure02 uses functions that internally use WONG

        # Both should use save_figure function
        assert hasattr(generate_figure01, "save_figure")
        assert hasattr(generate_figure02, "save_figure")

    def test_all_figures_are_executable(self) -> None:
        """Test that all figure scripts have proper structure."""
        import generate_figure01
        import generate_figure02
        import generate_figure03_candidates

        # generate_figure01 should have create_figure function
        assert callable(getattr(generate_figure01, "create_figure", None))

        # generate_figure02 should have run_demo function
        assert callable(getattr(generate_figure02, "run_demo", None))

        # generate_figure03_candidates should have main function
        assert callable(getattr(generate_figure03_candidates, "main", None))


# Cleanup sys.path after tests
@pytest.fixture(autouse=True, scope="module")
def cleanup_sys_path() -> None:
    """Remove scripts directory from sys.path after tests."""
    yield
    if str(SCRIPTS_DIR) in sys.path:
        sys.path.remove(str(SCRIPTS_DIR))
