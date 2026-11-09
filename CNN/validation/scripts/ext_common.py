#!/usr/bin/env python3
"""
ext_common.py

Shared infrastructure for the DenseNet-121 external validation

Responsibilities
----------------
- Deterministic, FP32-only inference configuration (no TF32, no cuDNN
  autotuning, deterministic algorithms requested in warn-only mode).
- Pipeline scope: the two analysed pipelines (locked-pixel, photometric).
  The exclusion rationale of the scale-matched pipeline is generated from
  optical_geometry.py (single source of truth for pixel spacing, binning,
  field of view and every derived spatial figure).
- Integrity: SHA-256 pins of the legacy scripts, densenet_model.py and the two
  configs. The configs' optical geometry is additionally verified field by
  field against optical_geometry.py at run time
  (audit_config_geometry -> provenance/optical_geometry.json).
- Environment lock: requirements_external.lock frozen from the runtime of the
  clean run and verified at run time; OpenCV provenance.
- Exact-artifact return bundle, reproduction check against the archived frozen
  run, hashing helpers, collision-safe module import, JSON/CSV writers.

Author: Rodrigo Sá
Date: 2026
"""

from __future__ import annotations

import os

# Must be set before the CUDA context exists
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import random
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from optical_geometry import excluded_scale_pipeline_rationale

EXPECTED_CLASS_TO_IDX: Dict[str, int] = {"Candida albicans": 0, "Candida glabrata": 1}
CLASS_NAMES: List[str] = ["Candida albicans", "Candida glabrata"]

# Paths are relative to the validation folder; any edit to a pinned file is refused.
# - scripts/legacy/*.py and scripts/densenet_model.py: byte-identical copies of the
#   published pipeline (repository commit 490673e, ViT/HybridViT/validation/scripts
#   and CNN/scripts/densenet_model.py); they carry the protocol parity with the
#   published Hybrid_43 external run.
# - configs/*: pinned at the revision whose optical geometry equals optical_geometry.py
#   (0.08698 um/px external, 2x2 binning; 0.1746 um/px development). The analysed
#   pipelines do not consume these spacings, so pixel-domain parity is unaffected.
PINNED_SHA256: Dict[str, str] = {
    "scripts/legacy/external_manifest.py": "911598f8c576bf80783728915bac3fa655c575a7324b5ce2f94b86308f96ef38",
    "scripts/legacy/preprocess_external.py": "3df9a093ced600e5407b32bd07f05a77712725e2fc87faa27b63483d3f96e4c0",
    "scripts/legacy/evaluate_external_generalization.py": "dd49c82483448ffd1fae2dd95f1d85251b1aff35effa58af458204e699c4ada2",
    "scripts/legacy/load_frozen_hybrid.py": "9eb98fd9cf292094e3f22655d5693f12c30d88ef07c1a52bafe669f03ff54962",
    "scripts/legacy/external_statistics.py": "deee3b514706f9820d7bb60cb29337330fa773801eb253f12ac6001ba0a449e1",
    "scripts/legacy/compare_external_pipelines.py": "efae60db1a012bb319cb2f3083ed47b39a13e6f48c5e9f0077cc05244cc1477f",
    "configs/external_validation_config.yaml": "d777ffc1e333617daf309ec9058cb2902d5ccc9b0f6a541cf71802290a4703de",
    "configs/external_validation_metadata.json": "973bad95f5840b40358959f6d6625ba3b6ee5137377aa5270a99e9628884d251",
    "scripts/densenet_model.py": "8cf07088fac6397187139e31ff5af4920ea26d941e6a7afc43e671ac45688585",
}

# -----------------------------------------------------------------------------
# Pipeline scope (Phase 1 decision: Option B)
# -----------------------------------------------------------------------------

PIPELINE_LOCKED = "locked_pixel_pipeline"
PIPELINE_PHOTOMETRIC = "photometric_mean_std_matched_pipeline"
ANALYSED_PIPELINES: Tuple[str, str] = (PIPELINE_LOCKED, PIPELINE_PHOTOMETRIC)
PIPELINE_ROLES: Dict[str, str] = {
    PIPELINE_LOCKED: "primary unadapted estimate",
    PIPELINE_PHOTOMETRIC: "post-hoc, label-free radiometric probe (per-image mean/SD harmonization)",
}
# The rationale is generated from optical_geometry.py (0.08698 um/px external, 2x2 binning;
# 0.1746 um/px development), so its figures cannot drift from the optics.
EXCLUDED_PIPELINES: Dict[str, str] = {
    "scale_matched_physical_pipeline": excluded_scale_pipeline_rationale(),
}

MODEL_FAMILY = "densenet121"
MODEL_DISPLAY = "DenseNet-121"
REFERENCE_LABEL = "hybrid43_archived"
REFERENCE_DISPLAY = "Hybrid$_{43}$ (archived predictions)"

# slide level (n = 24). The archived Hybrid_43 slide CSVs must reproduce every
# value below to 4 decimals (point estimates, class-stratified slide bootstrap
# B = 10,000 seed 42 percentile intervals, Clopper-Pearson recall intervals).
REFERENCE_HYBRID43_TABLE10: Dict[str, Dict[str, Any]] = {
    PIPELINE_LOCKED: {
        "point": {"balanced_accuracy": 0.5417, "matthews_corrcoef": 0.2085, "auroc": 0.7153,
                  "recall_albicans": 1.0000, "recall_glabrata": 0.0833,
                  "brier": 0.4298, "ece": 0.4447},
        "bootstrap_ci": {"balanced_accuracy": (0.5000, 0.6250), "matthews_corrcoef": (0.0000, 0.3780),
                         "auroc": (0.4653, 0.9306)},
        "clopper_pearson": {"recall_albicans": (0.7354, 1.0000), "recall_glabrata": (0.0021, 0.3848)},
    },
    PIPELINE_PHOTOMETRIC: {
        "point": {"balanced_accuracy": 0.8333, "matthews_corrcoef": 0.6761, "auroc": 0.9306,
                  "recall_albicans": 0.7500, "recall_glabrata": 0.9167},
        "bootstrap_ci": {"balanced_accuracy": (0.6667, 0.9583), "matthews_corrcoef": (0.3774, 0.9199),
                         "auroc": (0.8056, 1.0000)},
        "clopper_pearson": {"recall_albicans": (0.4281, 0.9451), "recall_glabrata": (0.6152, 0.9979)},
    },
}
REFERENCE_PINS_FILENAME = "reference_pins.json"

# -----------------------------------------------------------------------------
# Out-of-scope code and outputs (batch-normalization adaptation probe, removed)
# -----------------------------------------------------------------------------

FORBIDDEN_CODE_PATHS: Tuple[str, ...] = ("scripts/bn_adaptation.py",)
FORBIDDEN_PATH_REGEX = re.compile(r"(?i)(bn_probe|bn_adaptation|per_image_prior|cohort_reestimate|_bn_)")
OPTICAL_GEOMETRY_PROVENANCE = "provenance/optical_geometry.json"
PREEXISTING_OUTPUT_ALLOWLIST = frozenset({"pytest_log.txt", "provenance/checkpoint_identity_precheck.json"})
EXPECTED_TEST_COUNT = 48

# -----------------------------------------------------------------------------
# Environment lock
# -----------------------------------------------------------------------------

LOCK_PACKAGES: Tuple[str, ...] = ("numpy", "scipy", "scikit-learn", "tifffile", "pyyaml", "pytest")
LOCK_FILENAME = "requirements_external.lock"

BUNDLE_ZIP_NAME = "phase3_return_bundle.zip"
BUNDLE_MANIFEST_NAME = "return_bundle_manifest.json"
BUNDLE_CONTENTS_NAME = "BUNDLE_CONTENTS.json"


# -----------------------------------------------------------------------------
# Determinism / precision
# -----------------------------------------------------------------------------

def configure_deterministic_fp32(seed: int = 42) -> Dict[str, Any]:
    """
    Configure PyTorch for deterministic FP32 inference and return a record of
    the settings actually in force.

    TF32 is disabled explicitly. On an NVIDIA T4 (Turing, sm_75) TF32 does not
    exist, so this is a no-op there; on Ampere or newer GPUs PyTorch enables
    TF32 for cuDNN convolutions by default, which would silently lower the
    precision of every DenseNet convolution.
    """
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:  # pragma: no cover - very old torch
        pass
    torch.use_deterministic_algorithms(True, warn_only=True)

    return {
        "seed": int(seed),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "float32_matmul_precision": (
            torch.get_float32_matmul_precision()
            if hasattr(torch, "get_float32_matmul_precision") else None
        ),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "deterministic_algorithms_warn_only": bool(
            torch.is_deterministic_algorithms_warn_only_enabled()
        ) if hasattr(torch, "is_deterministic_algorithms_warn_only_enabled") else None,
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def assert_fp32_inference_context() -> None:
    """Raise if an autocast region is active (FP32 inference is mandatory)."""
    import torch

    active = False
    for dev in ("cuda", "cpu"):
        try:
            active = active or bool(torch.is_autocast_enabled(dev))
        except TypeError:  # torch < 2.4: device-less API
            active = active or bool(torch.is_autocast_enabled())
            if hasattr(torch, "is_autocast_cpu_enabled"):
                active = active or bool(torch.is_autocast_cpu_enabled())
    if active:
        raise RuntimeError(
            "An autocast region is active. External classification must run in "
            "FP32 (main.tex Table A.1: FP32 for classification)."
        )


def resolve_device(device: str = "auto"):
    import torch

    d = str(device).lower()
    if d == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if d.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    return torch.device(device)


# -----------------------------------------------------------------------------
# Environment snapshot and lock
# -----------------------------------------------------------------------------

def _version(module_name: str) -> Optional[str]:
    try:
        mod = __import__(module_name)
        return str(getattr(mod, "__version__", "unknown"))
    except Exception:
        return None


def normalize_dist_name(name: str) -> str:
    """PEP 503 normalization: 'PyYAML' -> 'pyyaml', 'scikit_learn' -> 'scikit-learn'."""
    return re.sub(r"[-_.]+", "-", str(name)).lower()


def installed_version(dist_name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def opencv_provenance() -> Dict[str, Any]:
    """Every installed opencv-* distribution, the cv2 that is actually imported, and its build hash."""
    dists: Dict[str, str] = {}
    for d in importlib.metadata.distributions():
        name = normalize_dist_name(d.metadata.get("Name") or "")
        if name.startswith("opencv"):
            dists[name] = d.version
    out: Dict[str, Any] = {"distributions": dict(sorted(dists.items()))}
    try:
        import cv2  # noqa: WPS433

        out["cv2_version"] = str(cv2.__version__)
        out["cv2_file"] = str(getattr(cv2, "__file__", None))
        out["build_information_sha256"] = hashlib.sha256(
            cv2.getBuildInformation().encode("utf-8")).hexdigest()
    except Exception as exc:  # pragma: no cover - depends on the runtime
        out["cv2_import_error"] = f"{type(exc).__name__}: {exc}"
    return out


def environment_snapshot(include_pip_freeze: bool = False) -> Dict[str, Any]:
    import torch

    snap: Dict[str, Any] = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cuda_available": bool(torch.cuda.is_available()),
        "numpy": _version("numpy"),
        "scipy": _version("scipy"),
        "sklearn": _version("sklearn"),
        "cv2": _version("cv2"),
        "tifffile": _version("tifffile"),
        "torchvision": _version("torchvision"),
        "yaml": _version("yaml"),
        "lock_package_versions": {p: installed_version(p) for p in LOCK_PACKAGES},
        "opencv": opencv_provenance(),
        "env": {
            k: os.environ.get(k)
            for k in [
                "CUBLAS_WORKSPACE_CONFIG",
                "OMP_NUM_THREADS",
                "PYTHONHASHSEED",
                "CUDA_VISIBLE_DEVICES",
            ]
        },
    }
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        snap["gpu"] = {
            "name": props.name,
            "capability": f"{props.major}.{props.minor}",
            "total_memory_gb": round(props.total_memory / 1024**3, 2),
            "multi_processor_count": props.multi_processor_count,
        }
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=20,
            )
            snap["nvidia_smi"] = out.stdout.strip()
        except Exception as exc:  # pragma: no cover
            snap["nvidia_smi"] = f"unavailable: {exc}"
    if include_pip_freeze:
        try:
            out = subprocess.run(
                [sys.executable, "-m", "pip", "freeze"],
                capture_output=True, text=True, timeout=120,
            )
            snap["pip_freeze"] = out.stdout.splitlines()
        except Exception as exc:  # pragma: no cover
            snap["pip_freeze"] = [f"unavailable: {exc}"]
    return snap


def parse_lock(path: Path) -> Dict[str, str]:
    """Parse 'name==version' lines; comments and blank lines are ignored; anything else is an error."""
    pins: Dict[str, str] = {}
    for lineno, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.count("==") != 1:
            raise ValueError(f"{path}:{lineno}: expected 'name==version', got {raw!r}")
        name, version = (s.strip() for s in line.split("=="))
        if not name or not version:
            raise ValueError(f"{path}:{lineno}: empty name or version in {raw!r}")
        key = normalize_dist_name(name)
        if key in pins:
            raise ValueError(f"{path}:{lineno}: duplicate pin for {name}")
        pins[key] = version
    return pins


def write_lock_from_environment(lock_path: Path) -> Dict[str, str]:
    """
    Freeze the versions installed in THIS runtime into requirements_external.lock:
    LOCK_PACKAGES plus every installed opencv-* distribution (so that the cv2
    actually imported is fully determined). Run once, after the missing packages
    have been installed and before the first evaluation; later runs install from it.
    """
    pins: Dict[str, Tuple[str, str]] = {}
    missing = []
    for name in LOCK_PACKAGES:
        v = installed_version(name)
        if v is None:
            missing.append(name)
        else:
            pins[normalize_dist_name(name)] = (name, v)
    if missing:
        raise RuntimeError(f"Cannot write the lock: not installed: {missing}")
    for name, v in opencv_provenance()["distributions"].items():
        pins[normalize_dist_name(name)] = (name, v)
    lines = [
        "# requirements_external.lock -- frozen from the runtime that executed the clean run",
        f"# created_utc: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        f"# python: {sys.version.split()[0]}",
        "# torch / torchvision / CUDA / cuDNN are provided by Colab and recorded, not installed.",
    ]
    lines += [f"{name}=={v}" for _, (name, v) in sorted(pins.items())]
    Path(lock_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {k: v for k, (_, v) in pins.items()}


def check_environment(lock_path: Path, strict: bool = True) -> Dict[str, Any]:
    """Compare installed distribution versions with the lock file."""
    lock_path = Path(lock_path)
    if not lock_path.exists():
        raise FileNotFoundError(f"Environment lock not found: {lock_path}")
    expected = parse_lock(lock_path)
    installed = {name: installed_version(name) for name in expected}
    mismatches = {name: {"expected": expected[name], "installed": installed[name]}
                  for name in expected if installed[name] != expected[name]}
    result = {
        "lock_path": str(lock_path),
        "lock_sha256": sha256_file(lock_path),
        "expected": expected,
        "installed": installed,
        "mismatches": mismatches,
        "ok": not mismatches,
        "opencv": opencv_provenance(),
    }
    if mismatches and strict:
        raise RuntimeError(f"Installed versions differ from {lock_path.name}: {mismatches}. "
                           "Run the pip install cell and restart the session.")
    return result


# -----------------------------------------------------------------------------
# Hashing
# -----------------------------------------------------------------------------

def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_array_tree(root: Path, suffix: str = ".npy") -> Dict[str, Any]:
    """
    Order-independent fingerprint of a directory of arrays: SHA-256 over the
    sorted list of (relative path, file SHA-256). Two runs that produced
    byte-identical arrays give the same digest.
    """
    root = Path(root)
    files = sorted(p for p in root.rglob(f"*{suffix}") if p.is_file())
    h = hashlib.sha256()
    for p in files:
        rel = p.relative_to(root).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(sha256_file(p).encode("ascii"))
        h.update(b"\n")
    return {"root": str(root), "n_files": len(files), "tree_sha256": h.hexdigest()}


def file_tree_manifest(root: Path) -> List[Dict[str, Any]]:
    """Sorted (relative path, bytes, sha256) of every file under root."""
    root = Path(root)
    out = []
    for p in sorted(x for x in root.rglob("*") if x.is_file()):
        out.append({"path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size,
                    "sha256": sha256_file(p)})
    return out


def tree_digest(entries: Sequence[Mapping[str, Any]]) -> str:
    h = hashlib.sha256()
    for e in sorted(entries, key=lambda e: e["path"]):
        h.update(f"{e['path']}\0{e['sha256']}\n".encode("utf-8"))
    return h.hexdigest()


# -----------------------------------------------------------------------------
# Module import by path
# -----------------------------------------------------------------------------

def import_module_from_path(unique_name: str, path: Path) -> ModuleType:
    """
    Import ``path`` as module ``unique_name``. Refuses to reuse an existing
    ``sys.modules`` entry that points to a different file.
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Module file not found: {path}")
    existing = sys.modules.get(unique_name)
    if existing is not None:
        existing_file = Path(getattr(existing, "__file__", "") or "").resolve()
        if existing_file != path:
            raise ImportError(
                f"Module name collision: {unique_name!r} already loaded from "
                f"{existing_file}, refusing to load {path}."
            )
        return existing
    spec = importlib.util.spec_from_file_location(unique_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(unique_name, None)
        raise
    return module


# -----------------------------------------------------------------------------
# IO
# -----------------------------------------------------------------------------

def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, tuple)):
        return list(obj)
    return str(obj)


def write_json(data: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=True, default=_json_default)
    tmp.replace(path)


def read_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(rows: Sequence[Dict[str, Any]], path: Path, columns: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def read_csv(path: Path) -> List[Dict[str, str]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"CSV is empty: {path}")
    return rows


def slide_date(slide_id: str) -> str:
    """'20260507_gla_A' -> '20260507'."""
    return str(slide_id).split("_")[0]


def floats_equal_exact(a: Sequence[float], b: Sequence[float]) -> bool:
    a = [float(x) for x in a]
    b = [float(x) for x in b]
    return len(a) == len(b) and all(x == y for x, y in zip(a, b))


# -----------------------------------------------------------------------------
# Integrity gates
# -----------------------------------------------------------------------------

def verify_pinned_files(validation_dir: Path) -> Dict[str, Any]:
    """Check every pinned file exists and is byte-identical to the published version."""
    validation_dir = Path(validation_dir)
    report: Dict[str, Any] = {}
    bad: List[str] = []
    for rel, expected in PINNED_SHA256.items():
        p = validation_dir / rel
        if not p.exists():
            report[rel] = {"status": "MISSING"}
            bad.append(rel)
            continue
        got = sha256_file(p)
        ok = got == expected
        report[rel] = {"status": "OK" if ok else "MODIFIED", "sha256": got}
        if not ok:
            bad.append(rel)
    if bad:
        raise RuntimeError(f"Pinned files missing or modified: {bad}. Re-copy them from the bundle.")
    return report


def verify_forbidden_absent(validation_dir: Path) -> Dict[str, Any]:
    """Out-of-scope code must not exist anywhere under scripts/ (by path or by file name)."""
    v = Path(validation_dir)
    present = [r for r in FORBIDDEN_CODE_PATHS if (v / r).exists()]
    named = sorted(p.relative_to(v).as_posix() for p in (v / "scripts").rglob("*")
                   if p.is_file() and FORBIDDEN_PATH_REGEX.search(p.name))
    if present or named:
        raise RuntimeError(
            f"Out-of-scope files present: {sorted(set(present) | set(named))}. "
            "Delete them (or move them outside scripts/) before the frozen evaluation.")
    return {"forbidden_code_paths": list(FORBIDDEN_CODE_PATHS), "absent": True}


def list_output_files(out_dir: Path) -> List[str]:
    out = Path(out_dir)
    if not out.exists():
        return []
    return sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())


def assert_outputs_preflight(out_dir: Path) -> List[str]:
    """outputs/ must be absent or contain only the two pre-run records."""
    files = list_output_files(out_dir)
    extra = [f for f in files if f not in PREEXISTING_OUTPUT_ALLOWLIST]
    if extra:
        raise RuntimeError(
            f"outputs/ contains {len(extra)} file(s) from an earlier run (first: {extra[:5]}). "
            "Delete outputs/ (or move it into archive/<run>/outputs/ to keep it as an earlier run) before the frozen evaluation.")
    return files


def verify_pre_run_records(out_dir: Path, expected_tests: int = EXPECTED_TEST_COUNT) -> Dict[str, Any]:
    """The synthetic test log (all tests passed, none skipped) and the checkpoint precheck must exist."""
    out = Path(out_dir)
    log = out / "pytest_log.txt"
    pre = out / "provenance" / "checkpoint_identity_precheck.json"
    if not log.exists():
        raise RuntimeError("outputs/pytest_log.txt missing: run the synthetic test cell first.")
    text = log.read_text(encoding="utf-8", errors="replace")
    m = re.findall(r"\b(\d+) passed\b", text)
    bad = re.findall(r"\b\d+ (failed|errors?|skipped|xfailed|xpassed)\b", text)
    if not m or int(m[-1]) != int(expected_tests) or bad:
        raise RuntimeError(f"pytest_log.txt does not show exactly {expected_tests} passed and nothing else "
                           f"(passed={m[-1] if m else None}, other={bad}).")
    if not pre.exists():
        raise RuntimeError("outputs/provenance/checkpoint_identity_precheck.json missing: run the precheck cell.")
    return {"pytest_passed": int(m[-1]), "pytest_log_sha256": sha256_file(log),
            "precheck_sha256": sha256_file(pre)}


# -----------------------------------------------------------------------------
# Exact-artifact return bundle
# -----------------------------------------------------------------------------

def expected_artifacts(pipelines: Sequence[str], *, with_reference: bool, amended: bool) -> List[str]:
    """The exact list of files, relative to outputs/, that a successful run produces."""
    a: List[str] = [
        "pytest_log.txt", "run_log.txt", "analysis_plan_frozen.json",
        "results_summary.json", "inferential_summary.json", "latex_external_table.tex",
        "provenance/forbidden_absent.json", "provenance/pre_run_records.json",
        "provenance/pinned_files.json", "provenance/environment_check.json",
        "provenance/env_snapshot.json", "provenance/checkpoint_identity_precheck.json",
        "provenance/checkpoint_identity.json", "provenance/preprocessing_array_hashes.json",
        "provenance/reproduction_check.json", OPTICAL_GEOMETRY_PROVENANCE,
        "manifest/external_manifest.csv", "manifest/external_manifest_summary.json",
        "manifest/external_validation_metadata_snapshot.json",
        "manifest/external_validation_config_snapshot.yaml",
    ]
    if amended:
        a.append("analysis_plan_amendment.json")
    for pl in pipelines:
        a += [f"preprocessing_sidecars/{pl}/preprocessing_metadata.json",
              f"preprocessing_sidecars/{pl}/preprocessing_failures.csv"]
        ev = f"{MODEL_FAMILY}/{pl}/evaluation"
        a += [f"{ev}/{n}" for n in ("image_predictions.csv", "slide_predictions.csv",
                                     "metrics_image_level.json", "metrics_slide_level.json",
                                     "confusion_matrix_image.csv", "confusion_matrix_slide.csv",
                                     "external_validation_metadata.json")]
        st = f"{MODEL_FAMILY}/{pl}/statistics"
        a += [f"{st}/{n}" for n in ("external_statistics_summary.json", "slide_metrics_with_ci.json",
                                     "image_metrics_descriptive.json", "bootstrap_distributions.csv",
                                     "subgroup_statistics.csv", "calibration_statistics.json",
                                     "confidence_by_slide.csv", "error_analysis_slide_level.csv",
                                     "optional_domain_shift_statistics.json")]
        a += [f"{MODEL_FAMILY}/{pl}/supplement/supplement_summary.json",
              f"{MODEL_FAMILY}/{pl}/supplement/latex_row.tex"]
    comparison_arms = [p for p in pipelines if p != PIPELINE_LOCKED]
    if PIPELINE_LOCKED in pipelines and comparison_arms:
        pc = f"{MODEL_FAMILY}/pipeline_comparison"
        a += [f"{pc}/multi_pipeline_slide_comparison.csv", f"{pc}/multi_pipeline_summary.json"]
        for c in comparison_arms:
            a += [f"{pc}/pairwise_comparison__{c}.csv", f"{pc}/pairwise_summary__{c}.json"]
    if with_reference:
        a += ["latex_model_comparison.tex", "provenance/reference_hybrid43_status.json"]
        for pl in pipelines:
            mc = f"model_comparison/{pl}"
            a += [f"{mc}/model_comparison_summary.json", f"{mc}/paired_slide_table.csv",
                  f"{mc}/bootstrap_deltas.csv", f"{mc}/latex_comparison_rows.tex"]
    if len(set(a)) != len(a):
        raise AssertionError("expected_artifacts produced duplicates")
    return sorted(a)


def _zip_add(zf: zipfile.ZipFile, arcname: str, data: bytes) -> None:
    info = zipfile.ZipInfo(arcname, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    zf.writestr(info, data)


def build_return_bundle(out_dir: Path, expected: Sequence[str], *,
                        zip_name: str = BUNDLE_ZIP_NAME,
                        manifest_name: str = BUNDLE_MANIFEST_NAME) -> Dict[str, Any]:
    """
    Bundle exactly ``expected``. Missing, extra or out-of-scope files are errors.
    The zip is deterministic (sorted members, fixed timestamps) and contains a
    BUNDLE_CONTENTS.json; the external manifest additionally records the zip hash.
    """
    out = Path(out_dir)
    present = set(list_output_files(out)) - {zip_name, manifest_name}
    expected_set = set(expected)
    missing = sorted(expected_set - present)
    extra = sorted(present - expected_set)
    forbidden = sorted(f for f in present | expected_set if FORBIDDEN_PATH_REGEX.search(f))
    if missing or extra or forbidden:
        raise RuntimeError(f"Return bundle refused: missing={missing}, extra={extra}, forbidden={forbidden}")
    entries = [{"path": p, "bytes": (out / p).stat().st_size, "sha256": sha256_file(out / p)}
               for p in sorted(expected_set)]
    contents = {"n_files": len(entries), "total_bytes": int(sum(e["bytes"] for e in entries)),
                "tree_sha256": tree_digest(entries), "files": entries}
    zip_path = out / zip_name
    tmp = zip_path.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp, "w") as zf:
        for e in entries:
            _zip_add(zf, e["path"], (out / e["path"]).read_bytes())
        _zip_add(zf, BUNDLE_CONTENTS_NAME,
                 json.dumps(contents, indent=2, sort_keys=True).encode("utf-8"))
    tmp.replace(zip_path)
    manifest = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                **contents, "zip": zip_name, "zip_sha256": sha256_file(zip_path)}
    write_json(manifest, out / manifest_name)
    return manifest


# -----------------------------------------------------------------------------
# Archive lookup and reproduction check
# -----------------------------------------------------------------------------

def archived_runs(archive_dir: Path) -> List[Path]:
    """archive/<utc>/outputs directories, oldest first (utc names sort chronologically)."""
    a = Path(archive_dir)
    if not a.exists():
        return []
    return sorted(p / "outputs" for p in a.iterdir() if p.is_dir() and (p / "outputs").is_dir())


def baseline_evaluation_dir(archive_dir: Path, pipeline: str) -> Optional[Path]:
    """The OLDEST archived frozen evaluation of ``pipeline`` (the original pre-specified run)."""
    for run in archived_runs(archive_dir):
        d = run / MODEL_FAMILY / pipeline / "evaluation"
        if (d / "slide_predictions.csv").exists() and (d / "image_predictions.csv").exists():
            return d
    return None


def reproduction_check(new_eval_dir: Path, archived_eval_dir: Optional[Path], *,
                       p_tol: float = 1e-6, flip_band: float = 1e-4) -> Dict[str, Any]:
    """
    Slide labels must be identical; image labels may differ only where both runs
    have |p_glabrata - 0.5| < flip_band. Max |delta p| is reported at both levels.
    """
    if archived_eval_dir is None:
        return {"status": "no_archived_baseline"}
    new_eval_dir, archived_eval_dir = Path(new_eval_dir), Path(archived_eval_dir)

    def by_key(path: Path, key: str) -> Dict[str, Dict[str, str]]:
        rows = read_csv(path)
        out = {r[key]: r for r in rows}
        if len(out) != len(rows):
            raise RuntimeError(f"Duplicate {key} in {path}")
        return out

    s_new = by_key(new_eval_dir / "slide_predictions.csv", "slide_id")
    s_old = by_key(archived_eval_dir / "slide_predictions.csv", "slide_id")
    if set(s_new) != set(s_old):
        raise RuntimeError("Reproduction check: slide sets differ from the archived run.")
    slide_flips = sorted(s for s in s_new if s_new[s]["pred_label"] != s_old[s]["pred_label"])
    if slide_flips:
        raise RuntimeError(f"Reproduction check failed: slide labels changed for {slide_flips}.")
    dp_slide = max(abs(float(s_new[s]["mean_prob_glabrata"]) - float(s_old[s]["mean_prob_glabrata"]))
                   for s in s_new)

    i_new = by_key(new_eval_dir / "image_predictions.csv", "relative_path")
    i_old = by_key(archived_eval_dir / "image_predictions.csv", "relative_path")
    if set(i_new) != set(i_old):
        raise RuntimeError("Reproduction check: image sets differ from the archived run.")
    near, far = [], []
    for k in i_new:
        if i_new[k]["pred_label"] != i_old[k]["pred_label"]:
            pn, po = float(i_new[k]["prob_glabrata"]), float(i_old[k]["prob_glabrata"])
            (near if abs(pn - 0.5) < flip_band and abs(po - 0.5) < flip_band else far).append(k)
    if far:
        raise RuntimeError(f"Reproduction check failed: {len(far)} image label(s) changed away from "
                           f"the decision boundary (first: {far[:5]}).")
    dp_image = max(abs(float(i_new[k]["prob_glabrata"]) - float(i_old[k]["prob_glabrata"])) for k in i_new)
    return {
        "status": "reproduced",
        "archived_evaluation_dir": str(archived_eval_dir),
        "archived_slide_predictions_sha256": sha256_file(archived_eval_dir / "slide_predictions.csv"),
        "n_slides": len(s_new), "n_images": len(i_new),
        "slide_label_changes": 0,
        "image_label_changes_near_boundary": len(near),
        "max_abs_delta_p_slide": dp_slide, "max_abs_delta_p_image": dp_image,
        "within_p_tol": bool(dp_slide <= p_tol and dp_image <= p_tol), "p_tol": p_tol,
        "flip_band": flip_band,
    }
