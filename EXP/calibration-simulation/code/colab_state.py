#!/usr/bin/env python3
"""
colab_state.py

Fault-tolerant persistence on Google Colab (one T4, 5-hour session limit).

Author: Rodrigo Sá
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import socket
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

EXIT_BUDGET = 75          # "temporary failure": rerun to resume
SESSION_ENV = "PARTAB_SESSION_START"


class BudgetExhausted(RuntimeError):
    """Raised before starting a unit that would not finish inside the session budget."""


def _json_default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (Path, bytes)):
        return str(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, indent=1, default=_json_default, allow_nan=True).encode("utf-8")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _fsync_dir(d: Path) -> None:
    try:
        fd = os.open(str(d), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def write_bytes_atomic(path: Path, data: bytes) -> str:
    """tmp in the same directory -> write -> fsync -> rename -> fsync(dir). Returns SHA-256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)
    return sha256_bytes(data)


def npz_bytes(**arrays: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def write_verified_files(directory: Path, payloads: Dict[str, bytes], marker_name: str, marker: Dict[str, Any]) -> None:
    """Write several payloads, then a marker (last) that records each file's SHA-256 and size."""
    files = {}
    for name, data in payloads.items():
        files[name] = {"sha256": write_bytes_atomic(Path(directory) / name, data), "bytes": len(data)}
    write_bytes_atomic(Path(directory) / marker_name, json_bytes({**marker, "files": files}))


def verify_files(directory: Path, marker: Dict[str, Any]) -> bool:
    files = marker.get("files")
    if not isinstance(files, dict) or not files:
        return False
    for name, rec in files.items():
        p = Path(directory) / name
        if not p.exists() or p.stat().st_size != rec.get("bytes") or sha256_file(p) != rec.get("sha256"):
            return False
    return True


class UnitStore:
    """Content-verified units: <root>/<unit>.payload + <root>/<unit>.done.json (marker written last)."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _paths(self, unit: str, kind: str):
        base = self.root / unit
        return base.parent / f"{base.name}.{kind}", base.parent / f"{base.name}.done.json"

    def _load(self, unit: str, digest: str, kind: str) -> Optional[bytes]:
        p, m = self._paths(unit, kind)
        if not m.exists():
            return None
        try:
            rec = json.loads(m.read_text())
            data = p.read_bytes() if p.exists() else None
        except (OSError, ValueError):
            self.discard(unit, kind)
            return None
        if (data is None or rec.get("digest") != digest or len(data) != rec.get("bytes")
                or sha256_bytes(data) != rec.get("sha256")):
            self.discard(unit, kind)
            return None
        return data

    def _commit(self, unit: str, digest: str, data: bytes, kind: str, meta: Optional[Dict[str, Any]]) -> None:
        p, m = self._paths(unit, kind)
        sha = write_bytes_atomic(p, data)
        write_bytes_atomic(m, json_bytes({"digest": digest, "sha256": sha, "bytes": len(data), "meta": meta or {},
                                          "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                          "host": host_id()}))

    def discard(self, unit: str, kind: str = "npz") -> None:
        for q in self._paths(unit, kind):
            try:
                q.unlink()
            except FileNotFoundError:
                pass

    def load_npz(self, unit: str, digest: str) -> Optional[Dict[str, np.ndarray]]:
        data = self._load(unit, digest, "npz")
        if data is None:
            return None
        with np.load(io.BytesIO(data), allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    def commit_npz(self, unit: str, digest: str, arrays: Dict[str, np.ndarray], meta: Optional[Dict[str, Any]] = None) -> None:
        self._commit(unit, digest, npz_bytes(**arrays), "npz", meta)

    def load_json(self, unit: str, digest: str) -> Optional[Any]:
        data = self._load(unit, digest, "json")
        return None if data is None else json.loads(data.decode("utf-8"))

    def commit_json(self, unit: str, digest: str, obj: Any, meta: Optional[Dict[str, Any]] = None) -> Any:
        data = json_bytes(obj)
        self._commit(unit, digest, data, "json", meta)
        return json.loads(data.decode("utf-8"))   # callers always receive the JSON round trip


# -----------------------------------------------------------------------------
# Session budget, timings, run lock
# -----------------------------------------------------------------------------

def host_id() -> str:
    try:
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()[:8]
    except OSError:
        boot = "noboot"
    return f"{socket.gethostname()}:{boot}"


def session_start_epoch() -> float:
    """Explicit env value (set by the orchestrator), else VM boot time (a Colab session is one VM)."""
    env = os.environ.get(SESSION_ENV)
    if env:
        return float(env)
    try:
        up = float(Path("/proc/uptime").read_text().split()[0])
        return time.time() - up
    except OSError:
        return time.time()


class SessionBudget:
    def __init__(self, hours: float = 5.0, margin_minutes: float = 20.0, start_epoch: Optional[float] = None) -> None:
        self.start = float(start_epoch) if start_epoch is not None else session_start_epoch()
        self.deadline = self.start + hours * 3600.0 - margin_minutes * 60.0

    def remaining(self) -> float:
        return self.deadline - time.time()

    def require(self, seconds: float, what: str) -> None:
        if self.remaining() < seconds:
            raise BudgetExhausted(f"{what}: needs about {seconds / 60:.1f} min, {max(0.0, self.remaining()) / 60:.1f} min "
                                  "left in this session's budget. Rerun in a new session to resume.")


class Timings:
    """Observed seconds per unit kind (kept on Drive); estimate = 1.25 x max of the last 5 observations."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        try:
            self.data: Dict[str, list] = json.loads(self.path.read_text())
        except (OSError, ValueError):
            self.data = {}

    def estimate(self, kind: str, default: float) -> float:
        v = self.data.get(kind) or []
        return 1.25 * max(v[-5:]) if v else default

    def record(self, kind: str, seconds: float) -> None:
        self.data.setdefault(kind, []).append(round(float(seconds), 2))
        self.data[kind] = self.data[kind][-20:]
        write_bytes_atomic(self.path, json_bytes(self.data))


class RunLock:
    def __init__(self, path: Path, stale_seconds: float = 1200.0) -> None:
        self.path, self.stale, self.me, self._last = Path(path), stale_seconds, host_id(), 0.0

    def acquire(self) -> None:
        if self.path.exists():
            try:
                rec = json.loads(self.path.read_text())
            except (OSError, ValueError):
                rec = {}
            age = time.time() - float(rec.get("heartbeat", 0))
            if rec.get("host") not in (None, self.me) and age < self.stale:
                raise RuntimeError(f"{self.path} is held by {rec.get('host')} (heartbeat {age:.0f} s ago). "
                                   "Stop the other session, or wait until the lock is older than "
                                   f"{self.stale / 60:.0f} min.")
        self.beat(force=True)

    def beat(self, force: bool = False) -> None:
        if force or time.time() - self._last > 60:
            write_bytes_atomic(self.path, json_bytes({"host": self.me, "pid": os.getpid(), "heartbeat": time.time()}))
            self._last = time.time()

    def release(self) -> None:
        try:
            if json.loads(self.path.read_text()).get("host") == self.me:
                self.path.unlink()
        except (OSError, ValueError):
            pass


def sync_filesystems() -> None:
    """Best effort from a subprocess; the notebook should still call drive.flush_and_unmount()."""
    try:
        os.sync()
    except AttributeError:
        pass


def fmt_minutes(seconds: float) -> str:
    return "n/a" if not math.isfinite(seconds) else f"{seconds / 60:.1f} min"
