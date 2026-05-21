"""Generate all figures for the paper.

This script runs all figure generation scripts in sequence and reports results.
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path


def main() -> int:
    """Run all figure generation scripts.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure)
    """
    # Get scripts directory
    scripts_dir = Path(__file__).parent

    # Define figure scripts in order
    figure_scripts = [
        "generate_figure01.py",
        "generate_figure02.py",
        "generate_figure03.py",
        "generate_figure04.py",
    ]

    print("=" * 70)
    print("Generating all figures for statespacecheck-paper")
    print("=" * 70)

    results: list[tuple[str, bool, float]] = []

    for script_name in figure_scripts:
        script_path = scripts_dir / script_name
        if not script_path.exists():
            print(f"\n⚠️  Warning: {script_name} not found, skipping...")
            results.append((script_name, False, 0.0))
            continue

        print(f"\n{'─' * 70}")
        print(f"Running {script_name}...")
        print(f"{'─' * 70}")

        start_time = time.time()
        try:
            # Import and run the figure script
            # Using exec to run the script in its own namespace
            with open(script_path) as f:
                code = compile(f.read(), script_path, "exec")
                exec(code, {"__name__": "__main__"})
            elapsed = time.time() - start_time
            results.append((script_name, True, elapsed))
            print(f"✅ {script_name} completed in {elapsed:.1f}s")
        except Exception:
            elapsed = time.time() - start_time
            results.append((script_name, False, elapsed))
            print(f"❌ {script_name} failed after {elapsed:.1f}s:")
            traceback.print_exc()

    # Print summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    total_time = sum(elapsed for _, _, elapsed in results)
    successes = sum(1 for _, success, _ in results if success)
    failures = len(results) - successes

    for script_name, success, elapsed in results:
        status = "✅" if success else "❌"
        print(f"{status} {script_name:30s} {elapsed:6.1f}s")

    print(f"{'─' * 70}")
    print(f"Total: {successes} succeeded, {failures} failed")
    print(f"Time: {total_time:.1f}s")
    print("=" * 70)

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
