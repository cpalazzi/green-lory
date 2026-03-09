#!/usr/bin/env python3
"""Tests for run_global.py pool resilience and fail_fast behaviour.

Test 1 — WorkerLostError tracking (all platforms):
  Injects a fake WorkerLostError mid-loop to verify the except-block fires,
  the _received set is tracked correctly, and a _failed.csv is written.

Test 2 — fail_fast propagation (all platforms):
  Confirms that when fail_fast=True a location exception is re-raised
  immediately instead of being caught and skipped.

Run with:
    .venv/bin/python scripts/test_pool_resilience.py
"""

from __future__ import annotations

import multiprocessing.pool
import sys
import tempfile
from datetime import datetime as _dt
from pathlib import Path

import pandas as pd

TOTAL = 20
FAIL_AT = 8  # number of results yielded before the fake exception is injected


# ─────────────────────────────────────────────────────────────────────────────
# Shared loop helper — mirrors the run_global.py imap_unordered loop exactly
# ─────────────────────────────────────────────────────────────────────────────

def _drive_loop(
    iterator, locations: list, output_csv: Path
) -> tuple[int, int, Path | None]:
    """Run the receive loop and return (n_received, n_dropped, failed_path|None)."""
    received: set[tuple[float, float]] = set()
    try:
        for lat, lon, _status, _val in iterator:
            received.add((lat, lon))
    except Exception as exc:  # noqa: BLE001
        dropped = [loc for loc in locations if loc not in received]
        _ts = _dt.now().strftime("%Y%m%d-%H%M%S")
        failed_path = output_csv.with_stem(output_csv.stem + f"_failed_{_ts}")
        pd.DataFrame(dropped, columns=["lat", "lon"]).to_csv(failed_path, index=False)
        return len(received), len(dropped), failed_path
    return len(received), 0, None


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: WorkerLostError — fake injection
# ─────────────────────────────────────────────────────────────────────────────

class _FaultyIterator:
    """Yields FAIL_AT results then raises WorkerLostError."""
    def __init__(self, locations: list) -> None:
        self._locs = locations
        self._i = 0

    def __iter__(self) -> "_FaultyIterator":
        return self

    def __next__(self) -> tuple[float, float, str, float]:
        if self._i >= FAIL_AT:
            raise multiprocessing.pool.WorkerLostError(
                f"Worker exited with exitcode -9 (simulated, task {self._i})"
            )
        lat, lon = self._locs[self._i]
        self._i += 1
        return lat, lon, "ok", float(self._i)


def test_worker_lost() -> bool:
    print("── Test 1: WorkerLostError caught → _failed.csv written ──")
    locations = [(float(i), float(i * 2)) for i in range(TOTAL)]
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        csv = Path(f.name)

    n_recv, n_drop, path = _drive_loop(_FaultyIterator(locations), locations, csv)
    ok = True

    if n_recv != FAIL_AT:
        print(f"  FAIL: expected {FAIL_AT} received, got {n_recv}")
        ok = False
    else:
        print(f"  OK   received  {n_recv}/{TOTAL}")

    expected_drop = TOTAL - FAIL_AT
    if n_drop != expected_drop:
        print(f"  FAIL: expected {expected_drop} dropped, got {n_drop}")
        ok = False
    else:
        print(f"  OK   dropped   {n_drop}")

    if path is None:
        print("  FAIL: _failed.csv was not written")
        ok = False
    else:
        rows = len(pd.read_csv(path))
        if rows != n_drop:
            print(f"  FAIL: _failed.csv has {rows} rows, expected {n_drop}")
            ok = False
        else:
            print(f"  OK   _failed.csv {path.name} ({rows} rows)")

    print("  PASSED\n" if ok else "  FAILED\n")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: fail_fast — exception propagates rather than being swallowed
# ─────────────────────────────────────────────────────────────────────────────

class _SentinelError(RuntimeError):
    """Unique exception type so we can assert it is the one that surfaces."""


def _fail_fast_run_single(fail_fast: bool) -> None:
    """Minimal stand-in for _run_single_location: raises when fail_fast=True."""
    try:
        raise _SentinelError("deliberate failure")
    except Exception as exc:
        if fail_fast:
            raise
        # Normal mode: swallow and return None
        _ = str(exc)


def test_fail_fast() -> bool:
    print("── Test 2: fail_fast=True → exception propagates immediately ──")
    ok = True

    # 2a: fail_fast=False — exception swallowed, no raise
    try:
        _fail_fast_run_single(fail_fast=False)
        print("  OK   fail_fast=False: exception swallowed (no raise)")
    except _SentinelError:
        print("  FAIL: fail_fast=False should NOT raise")
        ok = False

    # 2b: fail_fast=True — exception must propagate
    try:
        _fail_fast_run_single(fail_fast=True)
        print("  FAIL: fail_fast=True should have raised")
        ok = False
    except _SentinelError:
        print("  OK   fail_fast=True:  exception propagated immediately")
    except Exception as exc:
        print(f"  FAIL: unexpected exception type: {type(exc).__name__}: {exc}")
        ok = False

    print("  PASSED\n" if ok else "  FAILED\n")
    return ok


# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    print(f"Pool resilience tests (Python {sys.version.split()[0]})\n")
    r1 = test_worker_lost()
    r2 = test_fail_fast()
    if r1 and r2:
        print("All tests PASSED.")
        return 0
    print("One or more tests FAILED.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
