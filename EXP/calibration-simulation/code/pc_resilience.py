#!/usr/bin/env python3
"""
pc_resilience.py

Crash-safe replacement of the two CSV cache helpers of photometric_calibration.py
(``_read_cache``, ``_append_rows``). Import it once, before calibration runs:

    import pc_resilience  # noqa: F401  (patches photometric_calibration in place)

Author: Rodrigo Sá
"""

from __future__ import annotations

import csv
import io
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import photometric_calibration as pc
from colab_state import write_bytes_atomic

_NUMERIC = set(pc.STAT_KEYS) | {"n_tiles", "n_tiles_valid", "raw_over_4095", "raw_over_16383", "fg_fraction"}
_VALID = {"True", "False"}


def _row_ok(r: Dict[str, Any], cols: Sequence[str]) -> bool:
    if None in r or any(r.get(c) is None for c in cols):
        return False
    if r.get("valid") not in _VALID:
        return False
    for c in cols:
        if c in _NUMERIC:
            try:
                v = float(r[c])
            except (TypeError, ValueError):
                return False
            if r["valid"] == "True" and c in pc.STAT_KEYS and not math.isfinite(v):
                return False
    key = "base" if "base" in cols else "relative_path"
    return bool(r.get(key))


def repair_csv(path: Path) -> Dict[str, int]:
    """Truncate to the last complete line; keep only complete, parsable rows (last occurrence per key)."""
    path = Path(path)
    if not path.exists():
        return {"kept": 0, "dropped": 0}
    raw = path.read_bytes()
    end = raw.rfind(b"\n")
    complete = raw[: end + 1] if end >= 0 else b""
    text = complete.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    cols = reader.fieldnames or []
    rows = list(reader)
    key = "base" if "base" in cols else "relative_path"
    good: Dict[str, Dict[str, str]] = {}
    dropped = 0
    for r in rows:
        if cols and _row_ok(r, cols):
            good[r[key]] = r
        else:
            dropped += 1
    duplicates = len(rows) - dropped - len(good)
    if dropped or duplicates or len(complete) != len(raw) or not cols:
        if not cols:
            path.unlink()
            return {"kept": 0, "dropped": dropped, "removed_empty_file": 1}
        buf = io.StringIO(newline="")
        w = csv.DictWriter(buf, fieldnames=cols)
        w.writeheader()
        w.writerows(good.values())
        write_bytes_atomic(path, buf.getvalue().encode("utf-8"))
    return {"kept": len(good), "dropped": dropped, "duplicates": duplicates, "truncated_bytes": len(raw) - len(complete)}


def _read_cache(path: Path) -> Dict[str, Dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return {}
    rep = repair_csv(path)
    if rep.get("dropped") or rep.get("truncated_bytes"):
        print(f"  [pc_resilience] {path.name}: repaired {rep}", flush=True)
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    key = "base" if rows and "base" in rows[0] else "relative_path"
    return {r[key]: r for r in rows}


def _append_rows(path: Path, cols: Sequence[str], rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path = Path(path)
    new = not path.exists()
    if not new:
        with path.open(newline="") as f:
            header = next(csv.reader(f), None)
        if header != list(cols):
            raise RuntimeError(f"{path}: header {header} != expected {list(cols)}")
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cols), extrasaction="ignore")
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)
        f.flush()
        os.fsync(f.fileno())


pc._read_cache = _read_cache
pc._append_rows = _append_rows
PATCHED = True
