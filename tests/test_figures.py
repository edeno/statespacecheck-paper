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
        assert hasattr(generate_figure01, "COLORS")
        assert hasattr(generate_figure01, "save_figure")


class TestFigure02Integration:
    """Integration tests for generate_figure02.py script (diagnostic metrics)."""

    def test_imports_work(self) -> None:
        """Test that generate_figure02.py imports successfully."""
        # This will raise ImportError if imports fail
        import generate_figure02  # noqa: F401

    def test_imports_required_modules(self) -> None:
        """Test that generate_figure02.py can import all required modules."""
        # Import the main module
        import generate_figure02

        # Check that key functions are available
        assert hasattr(generate_figure02, "create_figure")

        # Check that imported utilities are accessible
        assert hasattr(generate_figure02, "COLORS")
        assert hasattr(generate_figure02, "save_figure")


class TestFigure03Integration:
    """Integration tests for generate_figure03.py script (simulation demo)."""

    def test_imports_work(self) -> None:
        """Test that generate_figure03.py imports successfully."""
        # This will raise ImportError if imports fail
        import generate_figure03  # noqa: F401

    def test_imports_required_modules(self) -> None:
        """Test that generate_figure03.py can import all required modules."""
        # Import the main module
        import generate_figure03

        # Check that key functions are available
        assert hasattr(generate_figure03, "run_demo")
        assert hasattr(generate_figure03, "DecodeParams")

        # Check that imported utilities are accessible
        assert hasattr(generate_figure03, "simulate_walk")
        assert hasattr(generate_figure03, "decode_and_diagnostics")
        assert hasattr(generate_figure03, "plot_combined_diagnostics")


class TestFigure04Integration:
    """Integration tests for generate_figure04.py script (real data model comparison)."""

    def test_imports_work(self) -> None:
        """Test that generate_figure04.py imports successfully."""
        # This will raise ImportError if imports fail
        import generate_figure04  # noqa: F401

    def test_imports_required_modules(self) -> None:
        """Test that generate_figure04.py can import all required modules."""
        # Import the main module
        import generate_figure04

        # Check that key functions are available
        assert hasattr(generate_figure04, "run_demo")

        # Check that configuration constants are defined
        assert hasattr(generate_figure04, "DATA_PATH")
        assert hasattr(generate_figure04, "ANIMAL_DATE_EPOCH")
        assert hasattr(generate_figure04, "FIGURE_4A_WINDOW_CENTER")
        assert hasattr(generate_figure04, "FIGURE_4A_WINDOW_HALF_WIDTH")

    def test_imports_analysis_utilities(self) -> None:
        """Test that generate_figure04.py imports required analysis utilities."""
        import generate_figure04

        # Check imports from real_data_analysis module
        assert hasattr(generate_figure04, "create_decoder_environment")
        assert hasattr(generate_figure04, "fit_decoder_models")
        assert hasattr(generate_figure04, "get_spike_counts")
        assert hasattr(generate_figure04, "compute_model_diagnostics")

    def test_imports_plotting_utilities(self) -> None:
        """Test that generate_figure04.py imports required plotting utilities."""
        import generate_figure04

        # Check imports from real_data_plotting module
        assert hasattr(generate_figure04, "plot_model_comparison_with_posterior")
        assert hasattr(generate_figure04, "plot_metric_distributions")
        assert hasattr(generate_figure04, "plot_track_graph_2d")


class TestFiguresModuleStructure:
    """Tests for overall figure module structure and consistency."""

    def test_all_figures_use_shared_style(self) -> None:
        """Test that all figures import from shared style module."""
        import generate_figure01
        import generate_figure02

        # Both should import COLORS from style module
        assert hasattr(generate_figure01, "COLORS")
        # generate_figure02 uses functions that internally use COLORS

        # Both should use save_figure function
        assert hasattr(generate_figure01, "save_figure")
        assert hasattr(generate_figure02, "save_figure")

    def test_all_figures_are_executable(self) -> None:
        """Test that all figure scripts have proper structure."""
        import generate_figure01
        import generate_figure02
        import generate_figure03
        import generate_figure04

        # generate_figure01 should have create_figure function
        assert callable(getattr(generate_figure01, "create_figure", None))

        # generate_figure02 should have create_figure function (diagnostic metrics)
        assert callable(getattr(generate_figure02, "create_figure", None))

        # generate_figure03 should have run_demo function (simulation demo)
        assert callable(getattr(generate_figure03, "run_demo", None))

        # generate_figure04 should have run_demo function (real data model comparison)
        assert callable(getattr(generate_figure04, "run_demo", None))


# Cleanup sys.path after tests
@pytest.fixture(autouse=True, scope="module")
def cleanup_sys_path() -> None:
    """Remove scripts directory from sys.path after tests."""
    yield
    if str(SCRIPTS_DIR) in sys.path:
        sys.path.remove(str(SCRIPTS_DIR))
