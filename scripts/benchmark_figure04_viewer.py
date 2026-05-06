"""Performance benchmark for the Figure 4 interactive viewer skeleton.

Plan gate (see ``docs/figure04_interactive_viewer_plan.md``):

- median frame time <= 16 ms (60 Hz) for the UI path
  (slice + view-range update);
- window-load p95 <= 50 ms for 20 s windows;
- no memory growth over 60 s of scrubbing.

This run sweeps the center-time slider across the requested duration,
waiting for each load to commit before advancing (paced) so the
"load latency" number is the realistic time-to-redraw after the user
pauses on a new window.

The script reports both ``UI median`` (the cheap per-tick path) and
``Load p95`` (the full dispatch -> commit time including main-thread
panel rendering). Two notes:

1. The UI median is the metric the user perceives during scrubbing
   because the slice-panel animation runs on this path; it should
   stay well under one 60 Hz frame.
2. ``Load p95`` includes Qt main-thread heatmap upload. With the
   offscreen platform plugin (``--offscreen``, the default) Qt does
   software rasterization, which inflates this number considerably
   versus a real display with hardware acceleration. For a true
   gate measurement, run with ``--no-offscreen`` on a real display.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), pct))


def _setup_qt(offscreen: bool) -> Any:
    if offscreen:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets  # noqa: PLC0415

    # Import from the ``app`` module where ``configure_qt_application``
    # is defined; ``viewer`` only re-exports it for back-compat and
    # mypy doesn't see the re-export as an explicit ``__all__`` entry.
    from statespacecheck_paper.interactive.app import configure_qt_application  # noqa: PLC0415

    existing = QtWidgets.QApplication.instance()
    app = (
        existing
        if isinstance(existing, QtWidgets.QApplication)
        else QtWidgets.QApplication(sys.argv)
    )
    configure_qt_application(app)
    return app


def run_benchmark(
    *,
    cache_dir: Path,
    model: str,
    window_seconds: float = 20.0,
    sweep_seconds: float = 60.0,
    tick_hz: float = 60.0,
    offscreen: bool = True,
) -> dict[str, float]:
    """Sweep center-time and report frame timings."""
    app = _setup_qt(offscreen)

    from PySide6 import QtCore  # noqa: PLC0415

    from statespacecheck_paper.interactive.data_source import Figure4DataSource  # noqa: PLC0415
    from statespacecheck_paper.interactive.viewer import Figure4Viewer  # noqa: PLC0415

    ds = Figure4DataSource(cache_dir, model=model)  # type: ignore[arg-type]
    viewer = Figure4Viewer(ds)
    # Match the requested window size.
    viewer._window_seconds = window_seconds  # noqa: SLF001
    viewer.show()
    app.processEvents()

    # Warm-up: dispatch one synchronous load so the first measured
    # frame doesn't dominate the cold-cache cost.
    viewer.set_center_time(ds.time[0] + window_seconds)
    viewer.force_reload_now()
    deadline = time.perf_counter() + 5.0
    while (
        viewer._latest_committed_request_id < viewer._next_request_id
        and time.perf_counter() < deadline
    ):  # noqa: SLF001
        app.processEvents()

    n_ticks = int(round(sweep_seconds * tick_hz))
    t_start = ds.time[0] + window_seconds
    centers = np.linspace(t_start, t_start + sweep_seconds, n_ticks)

    ui_frame_ms: list[float] = []
    load_latencies_ms: list[float] = []

    # Hook into the load worker to time per-load end-to-end latency.
    request_dispatch_times: dict[int, float] = {}
    original_dispatch = viewer._dispatch_load  # noqa: SLF001

    def timed_dispatch() -> None:
        # Capture the request id this dispatch will produce. The
        # dispatch is a no-op if a worker is already in flight (the
        # viewer holds a single in-flight slot), so only count actual
        # dispatches.
        was_inflight = viewer._inflight_request_id is not None  # noqa: SLF001
        original_dispatch()
        if not was_inflight and viewer._inflight_request_id is not None:  # noqa: SLF001
            request_dispatch_times[viewer._inflight_request_id] = time.perf_counter()  # noqa: SLF001

    viewer._dispatch_load = timed_dispatch  # type: ignore[method-assign]  # noqa: SLF001

    def on_loaded(request_id: int, *_: object) -> None:
        sent = request_dispatch_times.pop(request_id, None)
        if sent is not None:
            load_latencies_ms.append((time.perf_counter() - sent) * 1000.0)

    viewer._load_signals.finished.connect(on_loaded)  # noqa: SLF001

    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()

    # Paced sweep: at each tick, set the center, dispatch a load, and
    # wait for that load to commit before moving on. This measures
    # the realistic "user paused, viewer must redraw" latency rather
    # than queue throughput under continuous scrubbing.
    for center in centers:
        t_tick = time.perf_counter()
        viewer.set_center_time(float(center))
        viewer.force_reload_now()
        target = viewer._next_request_id  # noqa: SLF001
        ui_frame_ms.append((time.perf_counter() - t_tick) * 1000.0)

        wait_deadline = time.perf_counter() + 2.0
        while (
            viewer._latest_committed_request_id < target  # noqa: SLF001
            and time.perf_counter() < wait_deadline
        ):
            app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 5)

    # Drain any pending loads.
    drain_deadline = time.perf_counter() + 5.0
    while request_dispatch_times and time.perf_counter() < drain_deadline:
        app.processEvents()

    snapshot_after = tracemalloc.take_snapshot()
    diff = snapshot_after.compare_to(snapshot_before, "filename")
    top_growth_kb = sum(stat.size_diff for stat in diff[:10]) / 1024.0
    tracemalloc.stop()

    ds.close()
    viewer.close()

    return {
        "ticks": float(n_ticks),
        "ui_median_ms": float(np.median(ui_frame_ms)) if ui_frame_ms else float("nan"),
        "ui_p95_ms": _percentile(ui_frame_ms, 95.0),
        "ui_max_ms": float(np.max(ui_frame_ms)) if ui_frame_ms else float("nan"),
        "load_median_ms": (
            float(np.median(load_latencies_ms)) if load_latencies_ms else float("nan")
        ),
        "load_p95_ms": _percentile(load_latencies_ms, 95.0),
        "load_max_ms": (float(np.max(load_latencies_ms)) if load_latencies_ms else float("nan")),
        "loads_completed": float(len(load_latencies_ms)),
        "top_growth_kb": float(top_growth_kb),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="benchmark_figure04_viewer")
    parser.add_argument(
        "--cache-dir",
        default=str(REPO_ROOT / "data" / "cache"),
        help="Cache directory (default: data/cache).",
    )
    parser.add_argument(
        "--model",
        choices=("continuous", "contfrag"),
        default="continuous",
    )
    parser.add_argument("--window-seconds", type=float, default=20.0)
    parser.add_argument("--sweep-seconds", type=float, default=60.0)
    parser.add_argument("--tick-hz", type=float, default=60.0)
    parser.add_argument("--offscreen", action="store_true", default=True)
    parser.add_argument("--no-offscreen", dest="offscreen", action="store_false")
    args = parser.parse_args(argv)

    metrics = run_benchmark(
        cache_dir=Path(args.cache_dir),
        model=args.model,
        window_seconds=args.window_seconds,
        sweep_seconds=args.sweep_seconds,
        tick_hz=args.tick_hz,
        offscreen=args.offscreen,
    )

    print("=== Figure 4 viewer skeleton benchmark ===")
    print(f"Model              : {args.model}")
    print(f"Window             : {args.window_seconds:.1f} s")
    print(f"Sweep              : {args.sweep_seconds:.1f} s @ {args.tick_hz:.1f} Hz")
    print(f"Ticks              : {int(metrics['ticks'])}")
    print(f"UI frame median    : {metrics['ui_median_ms']:.2f} ms")
    print(f"UI frame p95       : {metrics['ui_p95_ms']:.2f} ms")
    print(f"UI frame max       : {metrics['ui_max_ms']:.2f} ms")
    print(f"Load median        : {metrics['load_median_ms']:.2f} ms")
    print(f"Load p95           : {metrics['load_p95_ms']:.2f} ms")
    print(f"Load max           : {metrics['load_max_ms']:.2f} ms")
    print(f"Loads completed    : {int(metrics['loads_completed'])}")
    print(f"Top-10 mem growth  : {metrics['top_growth_kb']:.0f} KB")

    print("\nPlan gate")
    ui_pass = metrics["ui_median_ms"] <= 16.0
    load_pass = metrics["load_p95_ms"] <= 50.0
    ui_status = "PASS" if ui_pass else "FAIL"
    load_status = "PASS" if load_pass else "FAIL"
    print(f"  UI median <= 16 ms : {ui_status} ({metrics['ui_median_ms']:.2f} ms)")
    print(f"  Load p95  <= 50 ms : {load_status} ({metrics['load_p95_ms']:.2f} ms)")

    return 0 if (ui_pass and load_pass) else 1


if __name__ == "__main__":
    raise SystemExit(main())
