#!/usr/bin/env python3
"""
colab_setup_env.py

Author: Rodrigo Sá
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

BASE_PACKAGES = ("numpy", "scipy", "scikit-learn", "tifffile", "pyyaml", "matplotlib")
PROBE = r"""
import importlib.metadata as md, json, re
def norm(n): return re.sub(r"[-_.]+", "-", n).lower()
v = {}
for d in md.distributions():
    n = norm(d.metadata.get("Name") or "")
    if n in %s or n.startswith("opencv"):
        v[n] = d.version
out = {"versions": v}
try:
    import numpy, scipy, sklearn, tifffile, yaml, matplotlib, cv2
    out["imports_ok"] = True
    out["cv2_imported"] = cv2.__version__
except Exception as exc:
    out["imports_ok"] = False
    out["import_error"] = f"{type(exc).__name__}: {exc}"
try:
    import torch
    out["torch"] = torch.__version__
    out["torch_cuda"] = torch.version.cuda
    out["cuda_available"] = bool(torch.cuda.is_available())
    out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
except Exception as exc:
    out["torch_error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(out))
""" % repr(tuple(BASE_PACKAGES))


def norm(n: str) -> str:
    return re.sub(r"[-_.]+", "-", n).lower()


def probe() -> Dict:
    out = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True, timeout=600)
    if out.returncode != 0:
        raise RuntimeError(f"environment probe failed: {out.stderr[-2000:]}")
    return json.loads(out.stdout.strip().splitlines()[-1])


def parse_lock(path: Path) -> Dict[str, str]:
    pins: Dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.count("==") != 1:
            raise ValueError(f"{path}: bad line {raw!r}")
        name, ver = (s.strip() for s in line.split("=="))
        pins[norm(name)] = ver
    return pins


def pip(args: List[str]) -> None:
    cmd = [sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check", *args]
    print("$ " + " ".join(cmd), flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        raise RuntimeError(f"pip install failed (exit {rc}); see the output above")


def main() -> None:
    p = argparse.ArgumentParser(description="Install or verify the Part A/B environment lock.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--no-densenet-lock", action="store_true")
    a = p.parse_args()
    cfg = json.loads(a.config.read_text())
    state = Path(cfg["drive_root"]) / "state"
    state.mkdir(parents=True, exist_ok=True)
    lock = state / "partab_env.lock"
    dn_lock = Path(cfg["paths"].get("densenet_env_lock", ""))

    if a.fresh or not lock.exists():
        if a.fresh and (Path(cfg["drive_root"]) / "unit_checks" / "unit_check_report.json").exists():
            print("WARNING: unit checks already ran; a new lock with another OpenCV version will stop Part A/B.")
        if dn_lock.is_file() and not a.no_densenet_lock:
            print(f"first session: installing the DenseNet-121 frozen versions from {dn_lock}")
            pip(["-r", str(dn_lock)])
        else:
            print("first session: DenseNet lock not used; keeping the Colab image versions")
        have = probe()["versions"]
        missing = [n for n in BASE_PACKAGES if norm(n) not in have]
        if missing:
            pip(missing)
        if not any(n.startswith("opencv") for n in have):
            pip(["opencv-python-headless"])
        res = probe()
        if not res.get("imports_ok"):
            raise RuntimeError(f"imports fail after installation: {res.get('import_error')}")
        lines = ["# partab_env.lock: frozen by colab_setup_env.py on the first session",
                 f"# created_utc: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
                 f"# python: {sys.version.split()[0]}; torch {res.get('torch')} (CUDA {res.get('torch_cuda')}) "
                 "from the Colab image, recorded only"]
        lines += [f"{n}=={v}" for n, v in sorted(res["versions"].items())]
        lock.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {lock}")
    pins = parse_lock(lock)
    have = probe()["versions"]
    wrong = {n: (v, have.get(n)) for n, v in pins.items() if have.get(n) != v}
    if wrong:
        print(f"installing locked versions: {wrong}")
        pip([f"{n}=={v}" for n, (v, _) in wrong.items()])
    res = probe()
    still = {n: (v, res["versions"].get(n)) for n, v in pins.items() if res["versions"].get(n) != v}
    if still or not res.get("imports_ok"):
        raise RuntimeError(f"environment does not match {lock}: {still}; imports_ok={res.get('imports_ok')} "
                           f"{res.get('import_error', '')}")
    rec = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "python": sys.version.split()[0], **res}
    with (state / "env_sessions.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps({k: rec[k] for k in ("torch", "torch_cuda", "cuda_available", "gpu", "cv2_imported") if k in rec}))
    if not rec.get("cuda_available"):
        print("WARNING: CUDA not available. Runtime > Change runtime type > T4 GPU, then rerun cells 1 to 3.")
    elif "T4" not in str(rec.get("gpu")):
        print(f"NOTE: GPU is {rec.get('gpu')}, not a Tesla T4.")
    print(f"ENV OK: {len(pins)} locked distributions match {lock}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
