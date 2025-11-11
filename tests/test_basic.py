"""Basic tests for statespacecheck-paper package."""

import statespacecheck_paper


def test_version() -> None:
    """Test that version is defined."""
    assert hasattr(statespacecheck_paper, "__version__")
    assert isinstance(statespacecheck_paper.__version__, str)
    assert statespacecheck_paper.__version__ == "0.1.0"
