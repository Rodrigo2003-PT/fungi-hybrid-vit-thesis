#!/usr/bin/env python3
"""
verify_refactored_pipeline.py

Standalone, author-run verification of the external-validation
pipeline. Uses ONLY synthetic data created inside a temporary directory:

    * 72 synthetic 14-bit-in-16-bit TIFFs (24 slides x 3 images, 64 x 64 px)
      with ImageJ unit + XResolution/YResolution calibration tags;
    * synthetic metadata JSON (schema 2.0 optics block) and config YAML;
    * a stub ``training_logic.py`` exposing a 4-parameter model
      (logit_c = w_c * mean(x) + b_c) and five tiny dummy checkpoints.

Dependencies of this script: Python standard library, numpy, tifffile, torch,
scikit-learn. The pipeline scripts under test additionally need opencv-python,
PyYAML and matplotlib (already project requirements).

Test groups
-----------
    A  optics derivation, validation rules and TIFF calibration parsing
    B  complete removal of the retired physical-scale experiment
    C  frozen-checkpoint loader provenance (architecture, mean/std, safe load)
    D  end-to-end synthetic run: manifest -> audit -> preprocess (both modes)
       -> evaluate -> statistics -> comparison, with hand-derived answers
    E  statistical unit tests (McNemar, sign-flip, Clopper-Pearson, DeLong)
    F  CSV/JSON schema integrity and optics provenance consistency

Exit codes: 0 all tests passed; 1 at least one test failed; 2 environment error.

Author: Rodrigo Sá
Date: 2026
"""

from __future__ import annotations

import argparse
import ast
import csv
import datetime
import hashlib
import importlib
import itertools
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

REQUIRED_MODULES = ("numpy", "tifffile", "torch", "sklearn")
PIPELINE_MODULES = ("cv2", "yaml", "matplotlib")

try:
    import numpy as np
    import tifffile
    import torch
    from sklearn.metrics import roc_auc_score
except ImportError as _exc:  # pragma: no cover - environment guard
    print(f"ENVIRONMENT ERROR: missing dependency: {_exc}", file=sys.stderr)
    sys.exit(2)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

PRIMARY = "locked_pixel_pipeline"
PHOTOMETRIC = "photometric_mean_std_matched_pipeline"

FORBIDDEN_TOKENS = (
    "scale_matched", "scale-match", "scale_match", "SCALE_LABEL", "run-scale-sensitivity",
    "pixel-spacing-um", "0.043", "4.069767", "0.245714", "16.562", "97.008", "0.213136",
    "0.786864", "geometric scale", "geometric-scale", "canvas", "0.175",
)
FORBIDDEN_FUNCTIONS = ("center_crop_or_pad_to_canvas", "scale_match_to_training_physical_canvas")
EXPECTED_MODE_CHOICES = {PRIMARY, PHOTOMETRIC}
EXEMPT_FROM_SCAN = {"verify_refactored_pipeline.py"}
PIPELINE_SCRIPT_FILES = (
    "external_manifest.py", "audit_external_tiffs.py", "preprocess_external.py",
    "evaluate_external_generalization.py", "external_statistics.py", "compare_external_pipelines.py",
    "load_frozen_hybrid.py", "run_external_validation_colab_local.py", "optics_provenance.py",
    "pipeline_common.py", "validation_stats.py",
)

# Clopper-Pearson references (computed independently with scipy.stats.binomtest, exact method).
CLOPPER_PEARSON_REFERENCE = {
    (22, 24): (0.7300271984423964, 0.9897436587187498),
    (12, 24): (0.29124177983621197, 0.7087582201637879),
    (0, 24): (0.0, 0.14247359772252563),
    (24, 24): (0.8575264022774745, 1.0),
    (10, 24): (0.22109690534667853, 0.6335693543436376),
}

# Synthetic acquisition: real external sensor primitives, scaled-down image size.
SYN_DX_EXT = Fraction(274, 100) * 2 / 63      # 137/1575 um/px
SYN_DX_TRN = Fraction(11, 63)
SYN_OPTICS: Dict[str, Any] = {
    "schema_version": "2.0",
    "external": {
        "camera": "synthetic external camera",
        "sensor_pixel_pitch_um": 2.74,
        "sensor_native_px": [128, 128],
        "binning_factor": 2,
        "binning_mode": "unknown",
        "objective_magnification": 63.0,
        "objective_na": 1.4,
        "camera_adapter_magnification": 1.0,
        "stored_image_px": [64, 64],
        "header_pixel_spacing_um": float(SYN_DX_EXT),
        "header_source": "synthetic imagej_unit_XResolution",
    },
    "training": {
        "camera": "synthetic training camera",
        "sensor_pixel_pitch_um": 11.0,
        "sensor_native_px": [32, 32],
        "binning_factor": 1,
        "binning_mode": "none",
        "objective_magnification": 63.0,
        "objective_na": 1.4,
        "camera_adapter_magnification": 1.0,
        "stored_image_px": [32, 32],
        "header_pixel_spacing_um": float(SYN_DX_TRN),
        "header_source": "synthetic imagej_unit_XResolution",
    },
    "network_input_px": 16,
    "tolerances": {"spacing_rel": 0.005},
}

IMAGE_SIZE = 16
STORED = 64
IMAGES_PER_SLIDE = 3
NOISE_SD = 20.0
ALB_LEVEL = 5000
GLA_LEVEL = 12000
PLANTED_GLA_AS_ALB = ("K", "L")
CKPT_MEAN = 0.5
CKPT_STD = 0.1
STUB_W = (-4.0, 4.0)
STUB_B = (0.1, 0.0)
N_BOOT = 500
SEED = 42
LETTERS = "ABCDEFGHIJKL"
HYPHAE_LETTERS = set("HIJKL")
DATE_GLA = "20990101"
DATE_ALB = "20990102"
DATE_ALB_B = "20990101"
DATE_ALB_A = "20200101"   # outside the synthetic acquisition window (mirrors the real 2020 slide)

STUB_TRAINING_LOGIC = '''\
"""Stub training_logic for verify_refactored_pipeline.py (synthetic tests only)."""
import hashlib
import json
import warnings

import torch
import torch.nn as nn

STUB_TRAINING_LOGIC = True


def compute_arch_hash(config):
    payload = {k: config.get(k) for k in ("experiment_name", "image_size", "channels")}
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class StubModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(%(w)r, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(%(b)r, dtype=torch.float32))

    def forward(self, x):
        m = x.mean(dim=(1, 2, 3))
        return m[:, None] * self.w[None, :] + self.b[None, :]


def build_model(config, num_classes, device):
    if not str(config.get("experiment_name", "")).startswith("stub"):
        raise ValueError("stub build_model: unrecognised experiment_name")
    return StubModel(num_classes).to(device)


def validate_checkpoint_arch(checkpoint, current_config, strict=True):
    stored_meta = checkpoint.get("arch_metadata")
    if stored_meta is None:
        warnings.warn("Checkpoint has no 'arch_metadata'. Skipping architecture validation.")
        return True
    if compute_arch_hash(current_config) != stored_meta.get("arch_hash"):
        if strict:
            raise RuntimeError("Architecture mismatch between checkpoint and current config!")
        return False
    return True
''' % {"w": list(STUB_W), "b": list(STUB_B)}

DECOY_TRAINING_LOGIC = 'raise ImportError("DECOY training_logic imported: import precedence is wrong (F-23)")\n'


# -----------------------------------------------------------------------------
# Test framework
# -----------------------------------------------------------------------------

class CheckFailed(AssertionError):
    pass


def check(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailed(message)


def close(a: float, b: float, tol: float, label: str) -> None:
    check(a is not None and b is not None and abs(float(a) - float(b)) <= tol,
          f"{label}: {a!r} != {b!r} (tol {tol})")


@dataclass
class Result:
    group: str
    test_id: str
    description: str
    status: str
    detail: str = ""
    seconds: float = 0.0


@dataclass
class Ctx:
    args: argparse.Namespace
    scripts_dir: Path
    tmp: Path
    env: Dict[str, str] = field(default_factory=dict)
    store: Dict[str, Any] = field(default_factory=dict)
    pending_author_input: List[str] = field(default_factory=list)

    def need(self, key: str) -> Any:
        if key not in self.store:
            raise CheckFailed(f"prerequisite '{key}' unavailable (an earlier test failed)")
        return self.store[key]


TESTS: List[Tuple[str, str, str, Callable[[Ctx], Optional[str]]]] = []


def test(group: str, test_id: str, description: str):
    def deco(fn: Callable[[Ctx], Optional[str]]):
        TESTS.append((group, test_id, description, fn))
        return fn
    return deco


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def canonical_sha256(data: Any) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def stub_arch_hash(config: Dict[str, Any]) -> str:
    payload = {k: config.get(k) for k in ("experiment_name", "image_size", "channels")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_script(ctx: Ctx, script: str, args: Sequence[Any], *, expect_rc: Optional[int] = 0,
               timeout: int = 600) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(ctx.scripts_dir / script)] + [str(a) for a in args]
    for a in args:
        p = str(a)
        if p.startswith("/") and not p.startswith(str(ctx.tmp)) and not p.startswith(str(ctx.scripts_dir)):
            raise CheckFailed(f"safety guard: refusing to pass a path outside the temporary directory: {p}")
    proc = subprocess.run(cmd, cwd=str(ctx.tmp / "cwd"), env=ctx.env, capture_output=True, text=True, timeout=timeout)
    if expect_rc is not None and proc.returncode != expect_rc:
        tail = (proc.stderr or proc.stdout)[-2500:]
        raise CheckFailed(f"{script} exited {proc.returncode} (expected {expect_rc}).\n{tail}")
    return proc


def import_pipeline_module(ctx: Ctx, name: str):
    if str(ctx.scripts_dir) not in sys.path:
        sys.path.insert(0, str(ctx.scripts_dir))
    return importlib.import_module(name)


def walk_json(obj: Any, path: Tuple[str, ...] = ()):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield path + (str(k),), k, v
            yield from walk_json(v, path + (str(k),))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk_json(v, path + (str(i),))


def slide_ids() -> List[Tuple[str, str, str, int, str]]:
    """(slide_id, species_code, species_name, label, hyphae_status) for the 24 synthetic slides."""
    out = []
    for letter in LETTERS:
        date = DATE_ALB_A if letter == "A" else (DATE_ALB_B if letter == "B" else DATE_ALB)
        out.append((f"{date}_alb_{letter}", "alb", "Candida albicans", 0,
                    "hyphae" if letter in HYPHAE_LETTERS else "no_hyphae"))
    for letter in LETTERS:
        out.append((f"{DATE_GLA}_gla_{letter}", "gla", "Candida glabrata", 1, "not_applicable"))
    return out


def slide_level(slide_id: str) -> int:
    if "_alb_" in slide_id:
        return ALB_LEVEL
    return ALB_LEVEL if slide_id.rsplit("_", 1)[-1] in PLANTED_GLA_AS_ALB else GLA_LEVEL


def write_calibrated_tiff(path: Path, array: np.ndarray, dx_um: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), array, imagej=True, resolution=(1.0 / dx_um, 1.0 / dx_um), metadata={"unit": "um"})


def stub_probabilities(arr: np.ndarray) -> Tuple[float, float]:
    """Independent float64 re-implementation of dataset standardization + stub model + softmax."""
    m = float(np.mean((arr.astype(np.float64) - CKPT_MEAN) / CKPT_STD))
    logits = np.array([STUB_W[0] * m + STUB_B[0], STUB_W[1] * m + STUB_B[1]])
    e = np.exp(logits - logits.max())
    p = e / e.sum()
    return float(p[0]), float(p[1])


def independent_stratified_bootstrap(y: np.ndarray, stat: Callable[[np.ndarray], float], n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    classes = sorted(set(int(v) for v in y))
    out = np.empty(n)
    for b in range(n):
        idx = np.concatenate([rng.choice(np.where(y == c)[0], size=int((y == c).sum()), replace=True) for c in classes])
        out[b] = stat(idx)
    return out


# -----------------------------------------------------------------------------
# Fixture construction
# -----------------------------------------------------------------------------

def build_fixtures(ctx: Ctx) -> None:
    tmp = ctx.tmp
    (tmp / "cwd" / "source").mkdir(parents=True)
    (tmp / "cwd" / "source" / "training_logic.py").write_text(DECOY_TRAINING_LOGIC, encoding="utf-8")
    stub_dir = tmp / "stub_src"
    stub_dir.mkdir()
    (stub_dir / "training_logic.py").write_text(STUB_TRAINING_LOGIC, encoding="utf-8")

    ctx.env = {
        **{k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "EXTERNAL_VALIDATION_SOURCE_CODE_DIR")},
        "PYTHONPATH": str(ctx.scripts_dir),
        "EXTERNAL_VALIDATION_SOURCE_CODE_DIR": str(stub_dir),
        "CUDA_VISIBLE_DEVICES": "",
        "MPLBACKEND": "Agg",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
    }

    slides = slide_ids()
    metadata = {
        "schema_version": "2.0",
        "dataset_role": "external_validation",
        "domain": "synthetic_external",
        "label_mapping": {"Candida albicans": 0, "Candida glabrata": 1},
        "species_slide_sets": {
            "Candida albicans": [s[0] for s in slides if s[1] == "alb"],
            "Candida glabrata": [s[0] for s in slides if s[1] == "gla"],
        },
        "albicans_morphology_subgroups": {
            "hyphae": [s[0] for s in slides if s[4] == "hyphae"],
            "no_hyphae": [s[0] for s in slides if s[4] == "no_hyphae"],
            "not_applicable": [s[0] for s in slides if s[4] == "not_applicable"],
        },
        "expected_slide_counts": {
            "total_slides": 24, "Candida albicans": 12, "Candida glabrata": 12,
            "Candida albicans with hyphae": 5, "Candida albicans without hyphae": 7,
        },
        "optics": SYN_OPTICS,
        "external_acquisition_metadata": {"camera": "synthetic", "bit_depth": 14},
        "training_domain_reference": {"bit_depth": 12},
    }
    meta_path = tmp / "inputs" / "metadata.json"
    meta_path.parent.mkdir(parents=True)
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    arch_cfg = {"experiment_name": "stub_test", "image_size": IMAGE_SIZE, "channels": 1}
    expected_hash = stub_arch_hash(arch_cfg)
    config = {
        "schema_version": "2.0",
        "checkpoints": {"expected_model": {
            "model_variant": "StubModel", "experiment_name": "stub_test", "architecture_hash": expected_hash,
            "image_size": IMAGE_SIZE, "channels": 1, "num_classes": 2,
            "checkpoint_mean": [CKPT_MEAN], "checkpoint_std": [CKPT_STD],
        }},
        "labels": {"class_names": ["Candida albicans", "Candida glabrata"]},
        "expected_external_cohort": {
            "total_slides": 24,
            "slides_per_species": {"Candida albicans": 12, "Candida glabrata": 12},
            "expected_images_per_slide": {"acceptable_minimum_for_audit_warning": 1, "acceptable_maximum_for_audit_warning": 10},
            "expected_acquisition_window": {"start": "20990101", "end": "20990131"},
        },
        "tiff_audit": {"expected_external_effective_bit_depth": 14, "expected_container_bit_depths": [14, 16]},
        "evaluation": {"bootstrap_confidence_intervals": {"n_bootstrap": N_BOOT, "seed": SEED}},
    }
    cfg_path = tmp / "inputs" / "config.yaml"   # JSON is valid YAML
    cfg_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    rng = np.random.default_rng(20261108)
    data_root = tmp / "data" / "fungi"
    for sid, code, species, _label, _h in slides:
        level = slide_level(sid)
        for i in range(1, IMAGES_PER_SLIDE + 1):
            arr = np.clip(np.rint(level + rng.normal(0.0, NOISE_SD, (STORED, STORED))), 0, 16383).astype(np.uint16)
            write_calibrated_tiff(data_root / species / f"{sid}_{i}.tif", arr, float(SYN_DX_EXT))

    ckpt_dir = tmp / "checkpoints"
    ckpt_dir.mkdir()
    state = {"w": torch.tensor(STUB_W, dtype=torch.float32), "b": torch.tensor(STUB_B, dtype=torch.float32)}
    base = {
        "model_state_dict": state,
        "config": dict(arch_cfg),
        "arch_metadata": {"arch_hash": expected_hash, "model_class": "StubModel"},
        "mean": [CKPT_MEAN],
        "std": [CKPT_STD],
    }
    variants = {
        "A_valid": base,
        "B_no_arch_metadata": {k: v for k, v in base.items() if k != "arch_metadata"},
        "C_wrong_arch": {**{k: v for k, v in base.items() if k != "arch_metadata"},
                         "config": {**arch_cfg, "experiment_name": "stub_other"}},
        "D_no_mean_std": {k: v for k, v in base.items() if k not in ("mean", "std")},
        "E_provenance_changed": {**base, "model_state_dict": {"w": torch.tensor([-4.0, 4.5]), "b": state["b"].clone()}},
        "F_unsafe_pickle": {**base, "created": datetime.datetime(2026, 1, 1)},
        "A_provenance_copy": base,
    }
    for name, payload in variants.items():
        torch.save(payload, str(ckpt_dir / f"{name}.pth"))

    ctx.store.update(
        metadata_path=meta_path, config_path=cfg_path, metadata=metadata, config=config,
        data_dir=tmp / "data", ckpt_dir=ckpt_dir, expected_hash=expected_hash, stub_dir=stub_dir,
        out=tmp / "out", pre=tmp / "pre",
    )


# -----------------------------------------------------------------------------
# A  Optics
# -----------------------------------------------------------------------------

def real_optics() -> Dict[str, Any]:
    return {
        "schema_version": "2.0",
        "external": {
            "camera": "ZEISS Axiocam 820 mono", "sensor_pixel_pitch_um": 2.74, "sensor_native_px": [4512, 4512],
            "binning_factor": 2, "binning_mode": "unknown", "objective_magnification": 63.0, "objective_na": 1.4,
            "camera_adapter_magnification": 1.0, "stored_image_px": [2256, 2256],
            "header_pixel_spacing_um": 0.08698412974552792, "header_source": "imagej_unit_XResolution",
        },
        "training": {
            "camera": "Teledyne Photometrics Prime 95B", "sensor_pixel_pitch_um": 11.0, "sensor_native_px": [1200, 1200],
            "binning_factor": 1, "binning_mode": "none", "objective_magnification": 63.0, "objective_na": 1.4,
            "camera_adapter_magnification": 1.0, "stored_image_px": [1200, 1200],
            "header_pixel_spacing_um": 0.1746031967750091, "header_source": "imagej_unit_XResolution",
        },
        "network_input_px": 384,
        "tolerances": {"spacing_rel": 0.005},
    }


@test("A", "A1", "Exact optical geometry for the real acquisition primitives")
def t_a1(ctx: Ctx) -> str:
    op = import_pipeline_module(ctx, "optics_provenance")
    rec = op.optics_provenance_record(real_optics())
    d = rec["derived"]
    dx_e, dx_t = Fraction(274, 100) * 2 / 63, Fraction(11, 63)
    exact = {
        ("external", "pixel_spacing_um"): dx_e,
        ("training", "pixel_spacing_um"): dx_t,
    }
    for (dom, key), val in exact.items():
        close(d[dom][key], float(val), 1e-15, f"{dom}.{key}")
    close(d["external"]["pixel_spacing_um"], 137 / 1575, 1e-15, "external dx == 137/1575")
    close(d["external"]["fov_um"][0], float(2256 * dx_e), 1e-12, "external FOV")
    close(d["training"]["fov_um"][0], float(1200 * dx_t), 1e-12, "training FOV")
    close(d["external"]["network_input_spacing_um"][0], float(2256 * dx_e / 384), 1e-15, "external dx_in")
    close(d["training"]["network_input_spacing_um"][0], float(1200 * dx_t / 384), 1e-15, "training dx_in")
    close(d["external"]["decimation_factor"][0], 5.875, 0, "external decimation")
    close(d["training"]["decimation_factor"][0], 3.125, 0, "training decimation")
    cross = d["cross_domain"]
    close(cross["training_over_external_native_spacing"], float(dx_t / dx_e), 1e-14, "training/external spacing")
    close(cross["external_over_training_fov_linear"], float((2256 * dx_e) / (1200 * dx_t)), 1e-14, "FOV ratio")
    close(cross["apparent_object_scale_external_vs_training_in_network_input"],
          float((1200 * dx_t) / (2256 * dx_e)), 1e-14, "apparent object scale")
    close(d["external"]["fov_um"][0], 196.2361904761905, 1e-9, "external FOV numeric")
    close(cross["apparent_object_scale_external_vs_training_in_network_input"], 1.0677123776984, 1e-12, "1.0677 scale")
    check(rec["validation"]["ok"], f"real primitives should validate: {rec['validation']['errors']}")
    check(any("O-6" in g for g in rec["provenance_gaps"]), "binning_mode unknown must be an O-6 provenance gap")
    return "dx_ext=137/1575, dx_trn=11/63, FOV 196.236/209.524 um, apparent scale 1.06771"


@test("A", "A2", "O-2 rejects stored size inconsistent with sensor size and binning")
def t_a2(ctx: Ctx) -> None:
    op = import_pipeline_module(ctx, "optics_provenance")
    bad = json.loads(json.dumps(real_optics()))
    bad["external"]["binning_factor"] = 1
    bad["external"]["header_pixel_spacing_um"] = 2.74 / 63
    res, _ = op.validate_optics(bad)
    check(any(e.startswith("O-2") for e in res.errors), f"expected O-2 error, got {res.errors}")


@test("A", "A3", "O-3 rejects an unbinned header spacing on binned images (~x2.02)")
def t_a3(ctx: Ctx) -> None:
    op = import_pipeline_module(ctx, "optics_provenance")
    bad = json.loads(json.dumps(real_optics()))
    bad["external"]["header_pixel_spacing_um"] = 0.043
    res, _ = op.validate_optics(bad)
    msgs = [e for e in res.errors if e.startswith("O-3")]
    check(bool(msgs) and "2.02" in msgs[0], f"expected O-3 error with ratio 2.02, got {res.errors}")
    good = real_optics()
    res_good, _ = op.validate_optics(good)
    check(res_good.ok, f"verified header 0.0869841297 must pass O-3: {res_good.errors}")


@test("A", "A4", "Placeholders are pending author input, not silent passes")
def t_a4(ctx: Ctx) -> None:
    op = import_pipeline_module(ctx, "optics_provenance")
    ph = json.loads(json.dumps(real_optics()))
    ph["external"]["header_pixel_spacing_um"] = "AUTHOR_TO_ENTER"
    ph["external"]["binning_mode"] = "AUTHOR_TO_CONFIRM"
    res, domains = op.validate_optics(ph)
    check(len(res.pending_author_input) == 2, f"expected 2 pending items, got {res.pending_author_input}")
    check("external" in domains, "geometry should still be derivable with a pending header value")


@test("A", "A5", "Project metadata/config (text only): optics valid, no stale optical keys")
def t_a5(ctx: Ctx) -> str:
    if not ctx.args.project_metadata and not ctx.args.project_config:
        return "SKIPPED: --project-metadata/--project-config not given"
    notes = []
    if ctx.args.project_metadata:
        op = import_pipeline_module(ctx, "optics_provenance")
        meta = load_json(Path(ctx.args.project_metadata))
        check(str(meta.get("schema_version")) == "2.0", "project metadata schema_version must be 2.0")
        check(isinstance(meta.get("optics"), dict), "project metadata has no optics block")
        res, _ = op.validate_optics(meta["optics"])
        check(res.ok, f"project optics invalid: {res.errors}")
        if res.pending_author_input:
            ctx.pending_author_input.extend(res.pending_author_input)
            check(not ctx.args.require_final_optics, f"pending author input: {res.pending_author_input}")
        for path, key, _ in walk_json(meta):
            if "pixel_spacing_um" in str(key) and "optics" not in path:
                raise CheckFailed(f"stale spacing key outside optics: {'.'.join(path)}")
        check("domain_shift_metadata" not in meta, "domain_shift_metadata must be removed")
        notes.append(f"metadata optics ok; gaps={len(res.provenance_gaps)}")
    if ctx.args.project_config:
        text = Path(ctx.args.project_config).read_text(encoding="utf-8")
        check("schema_version: \"2.0\"" in text, "project config schema_version must be 2.0")
        for key in ("pixel_spacing_um", "physical_field_of_view_um", "external_plotting", "use_amp"):
            check(key not in text, f"project config still contains '{key}'")
        notes.append("config clean")
    return "; ".join(notes)


@test("A", "A6", "optics_sha256 is stable and sensitive to primitive changes")
def t_a6(ctx: Ctx) -> None:
    op = import_pipeline_module(ctx, "optics_provenance")
    a, b = op.optics_sha256(real_optics()), op.optics_sha256(real_optics())
    check(a == b == canonical_sha256(real_optics()), "hash must be deterministic canonical-JSON sha256")
    changed = real_optics()
    changed["external"]["camera_adapter_magnification"] = 1.6
    check(op.optics_sha256(changed) != a, "hash must change when a primitive changes")


@test("A", "A7", "TIFF calibration parser: ImageJ > OME > TIFF ResolutionUnit, absent recorded")
def t_a7(ctx: Ctx) -> None:
    op = import_pipeline_module(ctx, "optics_provenance")
    d = ctx.tmp / "calibration"
    arr = np.full((8, 8), 5000, dtype=np.uint16)
    dx = float(SYN_DX_EXT)
    write_calibrated_tiff(d / "imagej.tif", arr, dx)
    tifffile.imwrite(str(d / "ome.ome.tif"), arr, ome=True,
                     metadata={"PhysicalSizeX": dx, "PhysicalSizeY": dx, "PhysicalSizeXUnit": "µm", "PhysicalSizeYUnit": "µm"})
    tifffile.imwrite(str(d / "resunit.tif"), arr, resolution=(1e4 / dx, 1e4 / dx), resolutionunit="CENTIMETER")
    tifffile.imwrite(str(d / "none.tif"), arr)
    expect = {"imagej.tif": "imagej_unit_XResolution", "ome.ome.tif": "ome_PhysicalSize",
              "resunit.tif": "tiff_ResolutionUnit_XResolution", "none.tif": "calibration_tag_absent"}
    for name, source in expect.items():
        with tifffile.TiffFile(str(d / name)) as tif:
            cal = op.read_tiff_calibration(tif)
        check(cal["source"] == source, f"{name}: source {cal['source']} != {source}")
        if source != "calibration_tag_absent":
            close(cal["x_um"], dx, dx * 1e-6, f"{name} x_um")
            close(cal["y_um"], dx, dx * 1e-6, f"{name} y_um")
        else:
            check(cal["x_um"] is None, "absent calibration must not yield a value")


# -----------------------------------------------------------------------------
# B  Removal of the retired experiment
# -----------------------------------------------------------------------------

def scan_text_for_tokens(text: str) -> List[str]:
    return [t for t in FORBIDDEN_TOKENS if t in text]


@test("B", "B1", "Forbidden-token scan of scripts, config, metadata and notebook")
def t_b1(ctx: Ctx) -> str:
    hits: List[str] = []
    files = sorted(p for p in ctx.scripts_dir.glob("*.py") if p.name not in EXEMPT_FROM_SCAN)
    check(all((ctx.scripts_dir / f).exists() for f in PIPELINE_SCRIPT_FILES),
          f"missing pipeline scripts: {[f for f in PIPELINE_SCRIPT_FILES if not (ctx.scripts_dir / f).exists()]}")
    for p in files:
        for tok in scan_text_for_tokens(p.read_text(encoding="utf-8")):
            hits.append(f"{p.name}: {tok}")
    for opt in (ctx.args.project_config, ctx.args.project_metadata):
        if opt:
            for tok in scan_text_for_tokens(Path(opt).read_text(encoding="utf-8")):
                hits.append(f"{Path(opt).name}: {tok}")
    if ctx.args.project_notebook:
        nb = load_json(Path(ctx.args.project_notebook))
        for i, cell in enumerate(nb.get("cells", [])):
            src = "".join(cell.get("source", []))
            for tok in scan_text_for_tokens(src):
                hits.append(f"notebook cell {i}: {tok}")
            if cell.get("cell_type") == "code" and cell.get("outputs"):
                hits.append(f"notebook cell {i}: stored outputs present (clear all outputs)")
    check(not hits, "residual references:\n  " + "\n  ".join(hits))
    return f"{len(files)} scripts scanned"


@test("B", "B2", "AST: retired functions and constants are absent")
def t_b2(ctx: Ctx) -> None:
    for p in ctx.scripts_dir.glob("*.py"):
        if p.name in EXEMPT_FROM_SCAN:
            continue
        tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                check(node.name not in FORBIDDEN_FUNCTIONS, f"{p.name}: retired function {node.name} still defined")
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    check(not (isinstance(tgt, ast.Name) and tgt.id == "SCALE_LABEL"), f"{p.name}: SCALE_LABEL assigned")


@test("B", "B3", "CLI rejects every retired option and mode (argparse exit 2)")
def t_b3(ctx: Ctx) -> None:
    t = ctx.tmp
    cases = [
        ("preprocess_external.py", ["--manifest", t / "x", "--checkpoint", t / "x", "--output-dir", t / "x",
                                    "--mode", "scale_matched_physical_pipeline"], "invalid choice"),
        ("preprocess_external.py", ["--manifest", t / "x", "--checkpoint", t / "x", "--output-dir", t / "x",
                                    "--training-pixel-spacing-um", "0.1"], "unrecognized arguments"),
        ("run_external_validation_colab_local.py", ["--drive-project-dir", t, "--raw-zip", t / "x",
                                                    "--primary-checkpoint", t / "x", "--metadata-json", t / "x",
                                                    "--config-yaml", t / "x", "--run-scale-sensitivity"],
         "unrecognized arguments"),
        ("compare_external_pipelines.py", ["--primary-slide-predictions", t / "x", "--output-dir", t / "x",
                                           "--sensitivity-slide-predictions", t / "x"], "unrecognized arguments"),
    ]
    for script, args, needle in cases:
        proc = run_script(ctx, script, args, expect_rc=2)
        check(needle in proc.stderr, f"{script}: expected '{needle}' in stderr, got {proc.stderr[-400:]}")


@test("B", "B4", "Parsers: mode choices exactly {locked, photometric}; no scale/spacing options anywhere")
def t_b4(ctx: Ctx) -> None:
    modules = ["external_manifest", "audit_external_tiffs", "preprocess_external", "evaluate_external_generalization",
               "external_statistics", "compare_external_pipelines", "run_external_validation_colab_local",
               "load_frozen_hybrid"]
    for name in modules:
        parser = import_pipeline_module(ctx, name).build_arg_parser()
        for action in parser._actions:
            for opt in action.option_strings:
                check(not re.search(r"scale|spacing|canvas|sensitivity-label|sensitivity-slide", opt),
                      f"{name}: retired option {opt}")
            if action.dest == "mode" and name == "preprocess_external":
                check(set(action.choices) == EXPECTED_MODE_CHOICES, f"mode choices {action.choices}")


@test("B", "B5", "Runner --dry-run plans exactly the 9 remaining steps (and captures environment)")
def t_b5(ctx: Ctx) -> str:
    t = ctx.tmp
    proc = run_script(ctx, "run_external_validation_colab_local.py", [
        "--drive-project-dir", t / "project", "--raw-zip", t / "project" / "fungi.zip",
        "--primary-checkpoint", t / "project" / "m.pth", "--metadata-json", ctx.need("metadata_path"),
        "--config-yaml", ctx.need("config_path"), "--drive-output-dir", t / "project" / "outputs_external_v2",
        "--local-work-dir", t / "project" / "work", "--dry-run",
    ])
    plan = json.loads(proc.stdout)
    names = [s["name"] for s in plan["steps"]]
    expected = ["manifest", "audit", f"preprocess:{PRIMARY}", f"evaluate:{PRIMARY}", f"statistics:{PRIMARY}",
                f"preprocess:{PHOTOMETRIC}", f"evaluate:{PHOTOMETRIC}", f"statistics:{PHOTOMETRIC}", "compare"]
    check(names == expected, f"planned steps {names}")
    for s in plan["steps"]:
        for tok in scan_text_for_tokens(" ".join(s["command"])):
            raise CheckFailed(f"step {s['name']} contains retired token {tok}")
    check(plan["settings_resolution"]["n_bootstrap"]["value"] == N_BOOT, "n_bootstrap not resolved from config")
    check(plan["settings_resolution"]["n_bootstrap"]["resolved_from"].startswith("config:"), "n_bootstrap source")
    env = plan["environment"]
    check("packages" in env and "file_sha256" in env and "git_commit" in env, "environment capture incomplete (G2)")
    check(env["file_sha256"].get("preprocess_external.py"), "script hashes missing")
    return "9 steps"


# -----------------------------------------------------------------------------
# C  Loader provenance
# -----------------------------------------------------------------------------

def run_loader(ctx: Ctx, ckpt: str, *extra: Any, expect_rc: int = 0) -> Tuple[subprocess.CompletedProcess, Optional[Dict[str, Any]]]:
    out = ctx.tmp / "loader" / f"{ckpt}_{len(extra)}.json"
    out.parent.mkdir(exist_ok=True)
    proc = run_script(ctx, "load_frozen_hybrid.py", [
        "--checkpoint", ctx.need("ckpt_dir") / f"{ckpt}.pth", "--config", ctx.need("config_path"),
        "--device", "cpu", "--metadata-output", out, *extra,
    ], expect_rc=expect_rc)
    return proc, (load_json(out) if expect_rc == 0 and out.exists() else None)


@test("C", "C1", "Valid checkpoint: architecture validated from stored arch_metadata; explicit mean/std source")
def t_c1(ctx: Ctx) -> None:
    _, meta = run_loader(ctx, "A_valid", "--provenance-checkpoint", ctx.need("ckpt_dir") / "A_provenance_copy.pth")
    arch = meta["architecture_identity"]
    check(arch["architecture_hash_validated"] is True, "architecture must validate")
    check(arch["architecture_validation_method"] == "stored_arch_metadata", arch["architecture_validation_method"])
    check(meta["scientific_guardrails"]["architecture_hash_validated"] is True, "guardrail must be computed True")
    check(meta["mean_std_source"] == "mean/std", meta["mean_std_source"])
    check(meta["mean_std_matches_config"] is True, "mean/std must match config")
    check(meta["mean_std_matches_provenance"] is True, "mean/std must match provenance")
    check(meta["checkpoint_load_mode"] == "weights_only", meta["checkpoint_load_mode"])
    check(meta["provenance_comparison"]["equivalence_status"] == "equivalent", "provenance must be equivalent")
    check(str(ctx.need("stub_dir")) in str(meta["training_logic_file"]), f"wrong training_logic: {meta['training_logic_file']}")
    check(meta["schema_version"] == "2.0", "loader schema 2.0")


@test("C", "C2", "No arch_metadata: hash recomputed from embedded config (no silent pass)")
def t_c2(ctx: Ctx) -> None:
    _, meta = run_loader(ctx, "B_no_arch_metadata")
    arch = meta["architecture_identity"]
    check(arch["architecture_validation_method"] == "recomputed_from_embedded_config", arch["architecture_validation_method"])
    check(arch["architecture_hash_observed"] == ctx.need("expected_hash"), "recomputed hash")
    check(any("arch_metadata" in w for w in arch["training_logic_validator_warnings"]), "validator warning not captured")


@test("C", "C3", "Wrong architecture: strict run aborts")
def t_c3(ctx: Ctx) -> None:
    proc, _ = run_loader(ctx, "C_wrong_arch", expect_rc=1)
    check("Architecture hash mismatch" in proc.stderr, proc.stderr[-400:])


@test("C", "C4", "Missing mean/std: strict aborts; non-strict records built_in_default")
def t_c4(ctx: Ctx) -> None:
    proc, _ = run_loader(ctx, "D_no_mean_std", expect_rc=1)
    check("Refusing mean/std fallback" in proc.stderr, proc.stderr[-400:])
    _, meta = run_loader(ctx, "D_no_mean_std", "--no-strict")
    check(meta["mean_std_source"] == "built_in_default", meta["mean_std_source"])


@test("C", "C5", "Provenance checkpoint with different weights: strict aborts")
def t_c5(ctx: Ctx) -> None:
    proc, _ = run_loader(ctx, "A_valid", "--provenance-checkpoint", ctx.need("ckpt_dir") / "E_provenance_changed.pth",
                         expect_rc=1)
    check("not equivalent" in proc.stderr, proc.stderr[-400:])


@test("C", "C6", "Checkpoint needing full unpickling is loaded and flagged")
def t_c6(ctx: Ctx) -> None:
    _, meta = run_loader(ctx, "F_unsafe_pickle")
    check(meta["checkpoint_load_mode"] == "unsafe_pickle_trusted_artifact", meta["checkpoint_load_mode"])


# -----------------------------------------------------------------------------
# D  End-to-end synthetic pipeline
# -----------------------------------------------------------------------------

@test("D", "D1", "Manifest: 72 rows, schema 2.0, optics record, date x species confounding")
def t_d1(ctx: Ctx) -> None:
    out = ctx.need("out") / "manifest" / "external_manifest.csv"
    run_script(ctx, "external_manifest.py", ["--input", ctx.need("data_dir"), "--metadata", ctx.need("metadata_path"),
                                             "--config", ctx.need("config_path"), "--output", out, "--strict"])
    cols, rows = read_csv(out)
    mod = import_pipeline_module(ctx, "external_manifest")
    check(cols == list(mod.MANIFEST_COLUMNS), "manifest header != MANIFEST_COLUMNS")
    check(not any("spacing" in c or "scale" in c for c in cols), f"spacing/scale columns remain: {cols}")
    check(len(rows) == 72 and len({r['slide_id'] for r in rows}) == 24, "72 images / 24 slides expected")
    summary = load_json(out.with_name("external_manifest_summary.json"))
    check(summary["schema_version"] == "2.0", "manifest summary schema 2.0")
    check(summary["optics"]["optics_sha256"] == canonical_sha256(SYN_OPTICS), "optics hash")
    close(summary["optics"]["derived"]["external"]["pixel_spacing_um"], float(SYN_DX_EXT), 1e-15, "derived dx")
    tab = summary["acquisition_date_by_species"]
    check(tab == {DATE_ALB_A: {"Candida albicans": 1},
                  DATE_GLA: {"Candida albicans": 1, "Candida glabrata": 12},
                  DATE_ALB: {"Candida albicans": 10}}, f"date x species table {tab}")
    flags = summary["confounding_flags"]
    check(flags["species_perfectly_separated_by_date"] is False and flags["dates_with_both_species"] == [DATE_GLA], str(flags))
    check(flags["slides_outside_expected_acquisition_window"] == [f"{DATE_ALB_A}_alb_A"], str(flags))
    close(flags["date_predicts_species_accuracy"], 23 / 24, 1e-15, "date predicts species accuracy")
    check(any("confounded with species" in w for w in summary["warnings"]), "confounding warning missing")
    ctx.store["manifest_csv"] = out
    ctx.store["manifest_rows"] = rows


@test("D", "D2", "Audit: calibration O-4 and shape O-5 pass; expected 14-bit condition is info, not warning")
def t_d2(ctx: Ctx) -> None:
    audit_dir = ctx.need("out") / "audit"
    run_script(ctx, "audit_external_tiffs.py", ["--manifest", ctx.need("manifest_csv"), "--metadata", ctx.need("metadata_path"),
                                                "--config", ctx.need("config_path"), "--output-dir", audit_dir, "--strict"])
    cols, rows = read_csv(audit_dir / "tiff_audit_per_image.csv")
    mod = import_pipeline_module(ctx, "audit_external_tiffs")
    check(cols == list(mod.PER_IMAGE_COLUMNS), "audit per-image header")
    check(all(r["audit_status"] == "passed" for r in rows), f"statuses {set(r['audit_status'] for r in rows)}")
    check(all("4095" in r["audit_info"] for r in rows), "values>4095 must be recorded in audit_info")
    check(all(r["header_spacing_source"] == "imagej_unit_XResolution" for r in rows), "calibration source")
    check(all(float(r["spacing_rel_error"]) < 1e-6 for r in rows), "spacing relative error")
    check(all(r["shape_matches_optics"] == "True" for r in rows), "shape_matches_optics")
    check(all(r["inferred_effective_bit_depth"] == "14" and r["container_bit_depth"] == "16" for r in rows), "bit depths")
    summary = load_json(audit_dir / "tiff_audit_summary.json")
    check(summary["validation_status"] == "passed", summary["validation_status"])
    check(summary["schema_version"] == "2.0" and summary["optics_sha256"] == canonical_sha256(SYN_OPTICS), "audit summary")
    check(summary["settings_resolution"]["expected_effective_bit_depth"]["resolved_from"].startswith("config:"), "precedence")
    ctx.store["audit_dir"] = audit_dir


@test("D", "D3", "Audit negative controls: unbinned-spacing header (O-4) and wrong shape (O-5) fail")
def t_d3(ctx: Ctx) -> None:
    manifest_rows = ctx.need("manifest_rows")
    bad_dir = ctx.tmp / "bad_tiffs"
    arr = np.full((STORED, STORED), ALB_LEVEL, dtype=np.uint16)
    write_calibrated_tiff(bad_dir / "bad_spacing.tif", arr, 0.043)
    write_calibrated_tiff(bad_dir / "bad_shape.tif", np.full((STORED, 60), ALB_LEVEL, dtype=np.uint16), float(SYN_DX_EXT))
    rows = []
    for name in ("bad_spacing.tif", "bad_shape.tif"):
        r = dict(manifest_rows[0])
        r.update(image_path=str(bad_dir / name), relative_path=f"bad/{name}", filename=name)
        rows.append(r)
    man = bad_dir / "manifest.csv"
    with man.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    out = bad_dir / "audit"
    run_script(ctx, "audit_external_tiffs.py", ["--manifest", man, "--metadata", ctx.need("metadata_path"),
                                                "--config", ctx.need("config_path"), "--output-dir", out, "--strict"],
               expect_rc=2)
    _, res = read_csv(out / "tiff_audit_per_image.csv")
    by = {r["filename"]: r for r in res}
    check(by["bad_spacing.tif"]["audit_status"] == "failed" and "O-4" in by["bad_spacing.tif"]["audit_errors"],
          by["bad_spacing.tif"]["audit_errors"])
    check(by["bad_shape.tif"]["audit_status"] == "failed" and "O-5" in by["bad_shape.tif"]["audit_errors"],
          by["bad_shape.tif"]["audit_errors"])


def preprocess_args(ctx: Ctx, mode: str, *extra: Any) -> List[Any]:
    return ["--manifest", ctx.need("manifest_csv"), "--checkpoint", ctx.need("ckpt_dir") / "A_valid.pth",
            "--metadata", ctx.need("metadata_path"), "--config", ctx.need("config_path"),
            "--output-dir", ctx.need("pre") / mode, "--mode", mode, "--image-size", IMAGE_SIZE,
            "--num-workers", 1, "--progress-every", 0, "--strict", *extra]


def load_arrays(ctx: Ctx, mode: str) -> Dict[str, np.ndarray]:
    out = {}
    for r in ctx.need("manifest_rows"):
        out[r["relative_path"]] = np.load(ctx.need("pre") / mode / Path(r["relative_path"]).with_suffix(".npy"))
    return out


@test("D", "D4", "Locked preprocessing: shape, dtype, range, 14-bit normalization, metadata 2.0")
def t_d4(ctx: Ctx) -> str:
    run_script(ctx, "preprocess_external.py", preprocess_args(ctx, PRIMARY))
    arrays = load_arrays(ctx, PRIMARY)
    check(len(arrays) == 72, "72 arrays expected")
    worst = 0.0
    for rel, a in arrays.items():
        check(a.shape == (IMAGE_SIZE, IMAGE_SIZE) and a.dtype == np.float32, f"{rel}: {a.shape} {a.dtype}")
        check(float(a.min()) >= 0.0 and float(a.max()) <= 1.0, f"{rel}: outside [0,1]")
        sid = rel.split("/")[-1].rsplit("_", 1)[0]
        expected = slide_level(sid) / 16383.0
        worst = max(worst, abs(float(a.mean()) - expected))
    check(worst < 2e-3, f"mean normalized intensity deviates from level/16383 by {worst}")
    meta = load_json(ctx.need("pre") / PRIMARY / "preprocessing_metadata.json")
    check(meta["schema_version"] == "2.0", "preprocessing schema 2.0")
    check(meta["counts"]["normalization_denominator_counts_processed"] == {"16383": 72}, "denominator 16383")
    sp = meta["spatial_policy"]
    check(sp["resize_interpolation"] == "cv2.INTER_LANCZOS4" and sp["interpolation_kernel_matches_training"] is True, str(sp))
    check(sp["physical_scale_resampling_applied"] is False, "no physical resampling")
    close(sp["decimation_factor"]["external"][0], STORED / IMAGE_SIZE, 0, "decimation")
    ref = meta["optics_reference"]
    check(ref["role"] == "descriptive_only_not_used_in_computation" and ref["optics_sha256"] == canonical_sha256(SYN_OPTICS), str(ref))
    check((ctx.need("pre") / PRIMARY / "preprocessing_fingerprint.json").exists(), "fingerprint missing")
    ctx.store["locked_arrays"] = arrays
    return f"max |mean - level/16383| = {worst:.2e}"


@test("D", "D5", "Fingerprint guard: reuse on identical parameters, refuse on changed parameters")
def t_d5(ctx: Ctx) -> None:
    ctx.need("locked_arrays")
    run_script(ctx, "preprocess_external.py", preprocess_args(ctx, PRIMARY))
    meta = load_json(ctx.need("pre") / PRIMARY / "preprocessing_metadata.json")
    check(meta["reuse_decision"] == "reuse" and meta["counts"]["n_skipped_existing"] == 72, "identical rerun must reuse")
    proc = run_script(ctx, "preprocess_external.py", preprocess_args(ctx, PRIMARY, "--no-post-resize-clip"), expect_rc=1)
    check("different preprocessing parameters" in proc.stderr, proc.stderr[-400:])


@test("D", "D6", "Photometric preprocessing: per-image mean/std matched to checkpoint statistics")
def t_d6(ctx: Ctx) -> None:
    run_script(ctx, "preprocess_external.py", preprocess_args(ctx, PHOTOMETRIC))
    arrays = load_arrays(ctx, PHOTOMETRIC)
    for rel, a in arrays.items():
        check(a.shape == (IMAGE_SIZE, IMAGE_SIZE) and a.dtype == np.float32, rel)
        close(float(a.mean()), CKPT_MEAN, 1e-4, f"{rel} mean")
        close(float(a.std()), CKPT_STD, 2e-3, f"{rel} std")
    meta = load_json(ctx.need("pre") / PHOTOMETRIC / "preprocessing_metadata.json")
    check(meta["photometric_harmonization"]["n_images_harmonized"] == 72, "harmonized count")
    ctx.store["photometric_arrays"] = arrays


def evaluate_args(ctx: Ctx, mode: str, pre_mode: Optional[str] = None, *extra: Any) -> List[Any]:
    return ["--checkpoint", ctx.need("ckpt_dir") / "A_valid.pth",
            "--provenance-checkpoint", ctx.need("ckpt_dir") / "A_provenance_copy.pth",
            "--manifest", ctx.need("manifest_csv"), "--preprocessed-dir", ctx.need("pre") / (pre_mode or mode),
            "--config", ctx.need("config_path"), "--metadata", ctx.need("metadata_path"),
            "--output-dir", ctx.need("out") / "evaluation" / mode, "--batch-size", 8, "--num-workers", 0,
            "--input-distribution-max-images", 24, "--device", "cpu", "--strict", "--preprocessing-mode", mode, *extra]


def check_evaluation(ctx: Ctx, mode: str, arrays: Dict[str, np.ndarray]) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    eval_dir = ctx.need("out") / "evaluation" / mode
    icols, irows = read_csv(eval_dir / "image_predictions.csv")
    scols, srows = read_csv(eval_dir / "slide_predictions.csv")
    mod = import_pipeline_module(ctx, "evaluate_external_generalization")
    check(icols == list(mod.IMAGE_PREDICTION_COLUMNS) and scols == list(mod.SLIDE_PREDICTION_COLUMNS), "prediction headers")
    check(len(irows) == 72 and len(srows) == 24, "row counts")
    by_slide: Dict[str, List[Dict[str, str]]] = {}
    for r in irows:
        pa, pg = stub_probabilities(arrays[r["relative_path"]])
        close(float(r["prob_albicans"]), pa, 2e-5, f"{r['relative_path']} prob_albicans")
        close(float(r["prob_glabrata"]), pg, 2e-5, f"{r['relative_path']} prob_glabrata")
        check(int(r["pred_label"]) == int(np.argmax([float(r["prob_albicans"]), float(r["prob_glabrata"])])), "image argmax")
        by_slide.setdefault(r["slide_id"], []).append(r)
    for s in srows:
        imgs = by_slide[s["slide_id"]]
        ma = float(np.mean([float(r["prob_albicans"]) for r in imgs]))
        mg = float(np.mean([float(r["prob_glabrata"]) for r in imgs]))
        close(float(s["mean_prob_albicans"]), ma, 1e-12, f"{s['slide_id']} slide mean albicans")
        close(float(s["mean_prob_glabrata"]), mg, 1e-12, f"{s['slide_id']} slide mean glabrata")
        check(int(s["n_images"]) == IMAGES_PER_SLIDE, "n_images per slide")
        check(int(s["pred_label"]) == int(np.argmax([ma, mg])), f"{s['slide_id']} slide argmax")
        check(int(s["correct"]) == int(int(s["pred_label"]) == int(s["true_label"])), "correct flag")
        close(float(s["confidence"]), max(ma, mg), 1e-12, "slide confidence")
    meta = load_json(eval_dir / "external_validation_metadata.json")
    return srows, meta


@test("D", "D7", "Locked evaluation: exact image probabilities, mean-probability slide aggregation, 22/24 correct")
def t_d7(ctx: Ctx) -> str:
    run_script(ctx, "evaluate_external_generalization.py", evaluate_args(ctx, PRIMARY))
    srows, meta = check_evaluation(ctx, PRIMARY, ctx.need("locked_arrays"))
    wrong = sorted(s["slide_id"] for s in srows if s["correct"] == "0")
    check(wrong == [f"{DATE_GLA}_gla_K", f"{DATE_GLA}_gla_L"], f"misclassified {wrong}")
    metrics = load_json(ctx.need("out") / "evaluation" / PRIMARY / "metrics_slide_level.json")
    close(metrics["accuracy"], 22 / 24, 1e-15, "slide accuracy")
    check(meta["schema_version"] == "2.0", "evaluation schema 2.0")
    check(meta["scientific_guardrails"]["architecture_hash_validated"] is True, "architecture guardrail")
    check(meta["preprocessing_consistency"]["consistent"] is True, "preprocessing consistency")
    check(meta["optics_sha256"] == canonical_sha256(SYN_OPTICS), "evaluation optics hash")
    check(meta["preprocessing"]["input_distribution_summary"]["n_slides_covered"] == 24, "balanced sample (F-09)")
    check(str(ctx.need("stub_dir")) in str(meta["model_loader_metadata"]["training_logic_file"]), "stub training_logic")
    ctx.store["locked_slides"] = sorted(srows, key=lambda r: r["slide_id"])
    return "22/24 correct; misclassified gla_K, gla_L"


@test("D", "D8", "Evaluation guards: standardization and mode mismatches abort (F-17)")
def t_d8(ctx: Ctx) -> None:
    ctx.need("locked_slides")
    proc = run_script(ctx, "evaluate_external_generalization.py",
                      evaluate_args(ctx, PRIMARY, None, "--arrays-already-standardized"), expect_rc=1)
    check("Standardization mismatch" in proc.stderr, proc.stderr[-400:])
    proc = run_script(ctx, "evaluate_external_generalization.py", evaluate_args(ctx, PHOTOMETRIC, PRIMARY), expect_rc=1)
    check("preprocessing_mode mismatch" in proc.stderr, proc.stderr[-400:])


@test("D", "D9", "Photometric evaluation: every image albicans (bias tie-break), accuracy exactly 0.5")
def t_d9(ctx: Ctx) -> None:
    run_script(ctx, "evaluate_external_generalization.py", evaluate_args(ctx, PHOTOMETRIC))
    srows, _ = check_evaluation(ctx, PHOTOMETRIC, ctx.need("photometric_arrays"))
    check(all(s["pred_label"] == "0" for s in srows), "all slides predicted albicans")
    for s in srows:
        close(float(s["mean_prob_albicans"]), 1 / (1 + np.exp(-0.1)), 1e-4, "sigmoid(0.1)")
    ctx.store["photometric_slides"] = sorted(srows, key=lambda r: r["slide_id"])


def statistics_args(ctx: Ctx, mode: str) -> List[Any]:
    ev = ctx.need("out") / "evaluation" / mode
    return ["--image-predictions", ev / "image_predictions.csv", "--slide-predictions", ev / "slide_predictions.csv",
            "--manifest", ctx.need("manifest_csv"), "--config", ctx.need("config_path"),
            "--metadata", ctx.need("metadata_path"), "--output-dir", ctx.need("out") / "statistics" / mode,
            "--n-bootstrap", N_BOOT, "--seed", SEED, "--strict",
            "--tiff-audit-per-image", ctx.need("audit_dir") / "tiff_audit_per_image.csv",
            "--tiff-audit-per-slide", ctx.need("audit_dir") / "tiff_audit_per_slide.csv"]


@test("D", "D10", "Locked statistics: exact CI, DeLong AUROC, bootstrap CI reproduced independently")
def t_d10(ctx: Ctx) -> str:
    slides = ctx.need("locked_slides")
    run_script(ctx, "external_statistics.py", statistics_args(ctx, PRIMARY))
    sdir = ctx.need("out") / "statistics" / PRIMARY
    summary = load_json(sdir / "external_statistics_summary.json")
    check(summary["schema_version"] == "2.0", "statistics schema 2.0")
    exact = summary["primary_slide_metrics_exact_intervals"]
    lo, hi = CLOPPER_PEARSON_REFERENCE[(22, 24)]
    close(exact["accuracy"]["ci_lower"], lo, 1e-9, "CP lower 22/24")
    close(exact["accuracy"]["ci_upper"], hi, 1e-9, "CP upper 22/24")
    y = np.array([int(s["true_label"]) for s in slides])
    pred = np.array([int(s["pred_label"]) for s in slides])
    pg = np.array([float(s["mean_prob_glabrata"]) for s in slides])
    close(exact["auroc"]["auc"], float(roc_auc_score(y, pg)), 1e-12, "DeLong AUC == sklearn AUC")
    boot = independent_stratified_bootstrap(y, lambda idx: float(np.mean(y[idx] == pred[idx])), N_BOOT, SEED)
    acc = summary["primary_slide_metrics_with_ci"]["accuracy"]
    close(acc["ci_lower"], float(np.percentile(boot, 2.5)), 1e-12, "bootstrap accuracy lower")
    close(acc["ci_upper"], float(np.percentile(boot, 97.5)), 1e-12, "bootstrap accuracy upper")
    check(acc["ci_degenerate"] is False, "locked accuracy bootstrap should not be degenerate")
    cols, dates = read_csv(sdir / "subgroup_by_acquisition_date.csv")
    mod = import_pipeline_module(ctx, "external_statistics")
    check(cols == list(mod.DATE_SUBGROUP_COLUMNS), "date subgroup header")
    by = {r["acquisition_date"]: r for r in dates}
    close(float(by[DATE_GLA]["slide_accuracy"]), 11 / 13, 1e-12, "date subgroup accuracy")
    check(by[DATE_ALB_A]["n_slides"] == "1" and by[DATE_ALB]["n_slides"] == "10", "date subgroup counts")
    check(summary["calibration_statistics"]["ece_slide"]["ece_reliable"] is False, "ECE reliability flag")
    return f"bootstrap accuracy CI [{acc['ci_lower']:.4f}, {acc['ci_upper']:.4f}] reproduced"


@test("D", "D11", "Photometric statistics: degenerate bootstrap flagged, exact CI for 12/24 and 0/12")
def t_d11(ctx: Ctx) -> None:
    ctx.need("photometric_slides")
    run_script(ctx, "external_statistics.py", statistics_args(ctx, PHOTOMETRIC))
    summary = load_json(ctx.need("out") / "statistics" / PHOTOMETRIC / "external_statistics_summary.json")
    exact = summary["primary_slide_metrics_exact_intervals"]
    lo, hi = CLOPPER_PEARSON_REFERENCE[(12, 24)]
    close(exact["accuracy"]["ci_lower"], lo, 1e-9, "CP lower 12/24")
    close(exact["accuracy"]["ci_upper"], hi, 1e-9, "CP upper 12/24")
    check(exact["sensitivity_recall_Candida_glabrata"]["ci_lower"] == 0.0, "recall 0/12 lower bound")
    ba = summary["primary_slide_metrics_with_ci"]["balanced_accuracy"]
    check(ba["ci_degenerate"] is True and ba["ci_lower"] is None and ba["ci_upper"] is None, f"degenerate flag {ba}")
    close(ba["point_estimate"], 0.5, 1e-15, "balanced accuracy 0.5")


@test("D", "D12", "Paired comparison: change counts, exact McNemar, sign-flip, paired bootstrap, schema 3.0")
def t_d12(ctx: Ctx) -> str:
    ctx.need("photometric_slides")
    out = ctx.need("out") / "statistics" / "multi_pipeline_comparison"
    ev = ctx.need("out") / "evaluation"
    run_script(ctx, "compare_external_pipelines.py", [
        "--primary-slide-predictions", ev / PRIMARY / "slide_predictions.csv",
        "--comparison", f"{PHOTOMETRIC}={ev / PHOTOMETRIC / 'slide_predictions.csv'}",
        "--output-dir", out, "--n-bootstrap", N_BOOT, "--seed", SEED,
    ])
    multi = load_json(out / "multi_pipeline_summary.json")
    pair = load_json(out / f"pairwise_summary__{PHOTOMETRIC}.json")
    check(multi["schema_version"] == "3.0" and pair["schema_version"] == "3.0", "comparison schema 3.0")
    check([c["label"] for c in multi["comparison_pipelines"]] == [PHOTOMETRIC], "exactly one comparison")
    check(pair["correctness_change_counts"] == {"both_correct": 12, "both_wrong": 2, "worsened": 10}, str(pair["correctness_change_counts"]))
    close(pair["mcnemar_exact_pvalue_descriptive"], 2 / 1024, 1e-15, "McNemar 0 vs 10")
    close(pair["metric_deltas_comparison_minus_primary"]["accuracy"], -10 / 24, 1e-12, "delta accuracy")
    cols, prow = read_csv(out / f"pairwise_comparison__{PHOTOMETRIC}.csv")
    mod = import_pipeline_module(ctx, "compare_external_pipelines")
    check(cols == list(mod.PAIRWISE_BASE_COLUMNS), "pairwise header")
    vs = import_pipeline_module(ctx, "validation_stats")
    deltas = [float(r["delta_true_class_probability"]) for r in prow]
    sf = pair["sign_flip_test_mean_delta_true_class_probability_descriptive"]
    ref = vs.sign_flip_pvalue(deltas, seed=SEED)
    check(sf == ref and sf["method"] == "exact" and sf["n_nonzero"] == 24, f"sign flip {sf} vs {ref}")
    y = np.array([int(r["true_label"]) for r in prow])
    pc = np.array([int(r["primary_correct"]) for r in prow], dtype=float)
    cc = np.array([int(r["comparison_correct"]) for r in prow], dtype=float)
    boot = independent_stratified_bootstrap(y, lambda idx: float(cc[idx].mean() - pc[idx].mean()), N_BOOT, SEED)
    b = pair["bootstrap_delta_accuracy"]
    close(b["ci_lower"], float(np.percentile(boot, 2.5)), 1e-12, "paired bootstrap lower")
    close(b["ci_upper"], float(np.percentile(boot, 97.5)), 1e-12, "paired bootstrap upper")
    check(set(multi["recommended_thesis_language"].keys()) == {"general", PHOTOMETRIC}, "thesis language keys")
    return f"McNemar p = 2/1024; sign-flip p = {sf['p_value']:.3g} (exact)"


# -----------------------------------------------------------------------------
# E  Statistical unit tests
# -----------------------------------------------------------------------------

@test("E", "E1", "Exact McNemar")
def t_e1(ctx: Ctx) -> None:
    vs = import_pipeline_module(ctx, "validation_stats")
    close(vs.exact_mcnemar_pvalue(5, 1), 14 / 64, 1e-15, "McNemar 5/1")
    close(vs.exact_mcnemar_pvalue(0, 10), 2 / 1024, 1e-15, "McNemar 0/10")
    check(vs.exact_mcnemar_pvalue(0, 0) is None, "no discordant pairs -> None")


@test("E", "E2", "Sign-flip exact p on [1, 2, 3] equals 0.25")
def t_e2(ctx: Ctx) -> None:
    vs = import_pipeline_module(ctx, "validation_stats")
    r = vs.sign_flip_pvalue([1.0, 2.0, 3.0, 0.0])
    check(r["method"] == "exact" and r["n_nonzero"] == 3, str(r))
    close(r["p_value"], 0.25, 1e-15, "sign-flip p")


@test("E", "E3", "Chunked exact sign-flip equals naive enumeration (n = 12)")
def t_e3(ctx: Ctx) -> None:
    vs = import_pipeline_module(ctx, "validation_stats")
    vals = np.random.default_rng(7).normal(0.05, 0.2, 12)
    obs = abs(vals.mean())
    naive = sum(abs(np.mean(vals * np.array(s))) >= obs - 1e-12 * max(1, obs)
                for s in itertools.product([-1.0, 1.0], repeat=12)) / 2 ** 12
    close(vs.sign_flip_pvalue(vals)["p_value"], naive, 1e-15, "chunked vs naive")


@test("E", "E4", "Sign-flip switches to seeded Monte Carlo above 24 differences")
def t_e4(ctx: Ctx) -> None:
    vs = import_pipeline_module(ctx, "validation_stats")
    vals = np.random.default_rng(11).normal(0.03, 0.2, 30)
    r = vs.sign_flip_pvalue(vals, seed=3)
    check(r["method"] == "monte_carlo", str(r))
    signs = np.random.default_rng(99).choice([-1.0, 1.0], size=(1_000_000, 30))
    ref = float(np.mean(np.abs(signs @ vals) / 30 >= abs(vals.mean()) - 1e-12))
    close(r["p_value"], ref, 0.01, "Monte Carlo vs 1e6-draw reference")
    check(vs.sign_flip_pvalue(vals, seed=3) == r, "Monte Carlo must be deterministic for a fixed seed")


@test("E", "E5", "Clopper-Pearson matches independent exact references")
def t_e5(ctx: Ctx) -> None:
    vs = import_pipeline_module(ctx, "validation_stats")
    for (k, n), (lo, hi) in CLOPPER_PEARSON_REFERENCE.items():
        r = vs.clopper_pearson_ci(k, n)
        close(r["ci_lower"], lo, 1e-9, f"CP lower {k}/{n}")
        close(r["ci_upper"], hi, 1e-9, f"CP upper {k}/{n}")


@test("E", "E6", "DeLong AUROC and variance match an independent implementation")
def t_e6(ctx: Ctx) -> None:
    vs = import_pipeline_module(ctx, "validation_stats")
    rng = np.random.default_rng(5)
    y = np.array([0] * 12 + [1] * 12)
    s = np.concatenate([rng.normal(0, 1, 12), rng.normal(1, 1, 12)])
    s[3] = s[15]  # one tie
    r = vs.delong_auc_ci(y, s)
    close(r["auc"], float(roc_auc_score(y, s)), 1e-12, "AUC")
    pos, neg = s[y == 1], s[y == 0]
    v10 = np.array([np.mean([1.0 if p > q else 0.5 if p == q else 0.0 for q in neg]) for p in pos])
    v01 = np.array([np.mean([1.0 if p > q else 0.5 if p == q else 0.0 for p in pos]) for q in neg])
    var = v10.var(ddof=1) / len(pos) + v01.var(ddof=1) / len(neg)
    close(r["variance"], float(var), 1e-14, "DeLong variance")
    check(r["ci_lower"] <= r["auc"] <= r["ci_upper"], "CI contains AUC")
    perfect = vs.delong_auc_ci([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    check(perfect["auc"] == 1.0 and perfect["ci_degenerate"] is True, "perfect separation is degenerate")


# -----------------------------------------------------------------------------
# F  Schema integrity and provenance consistency
# -----------------------------------------------------------------------------

FORBIDDEN_KEY_RE = re.compile(r"scale_match|scale_factor|canvas|padding_fraction|content_fraction")


@test("F", "F1", "Every generated CSV header equals its module column constant")
def t_f1(ctx: Ctx) -> str:
    out = ctx.need("out")
    stats_mod = import_pipeline_module(ctx, "external_statistics")
    audit_mod = import_pipeline_module(ctx, "audit_external_tiffs")
    pre_mod = import_pipeline_module(ctx, "preprocess_external")
    expectations = {
        out / "audit" / "tiff_audit_per_slide.csv": audit_mod.PER_SLIDE_COLUMNS,
        ctx.need("pre") / PRIMARY / "preprocessing_failures.csv": pre_mod.FAILURE_COLUMNS,
    }
    for mode in (PRIMARY, PHOTOMETRIC):
        sdir = out / "statistics" / mode
        expectations[sdir / "subgroup_statistics.csv"] = stats_mod.SUBGROUP_COLUMNS
        expectations[sdir / "confidence_by_slide.csv"] = stats_mod.CONFIDENCE_BY_SLIDE_COLUMNS
        expectations[sdir / "error_analysis_slide_level.csv"] = stats_mod.ERROR_ANALYSIS_COLUMNS
        expectations[sdir / "subgroup_by_acquisition_date.csv"] = stats_mod.DATE_SUBGROUP_COLUMNS
    for path, cols in expectations.items():
        check(path.exists(), f"missing {path.name}")
        header, _ = read_csv(path)
        check(header == list(cols), f"{path}: header mismatch")
    return f"{len(expectations)} extra CSV headers checked (others in D1/D2/D7/D10/D12)"


@test("F", "F2", "No retired keys or values in any generated JSON")
def t_f2(ctx: Ctx) -> str:
    files = sorted(list(ctx.need("out").rglob("*.json")) + list(ctx.need("pre").rglob("*.json")))
    check(len(files) >= 15, f"too few JSON outputs ({len(files)})")
    for f in files:
        data = load_json(f)
        for path, key, value in walk_json(data):
            check(not FORBIDDEN_KEY_RE.search(str(key)), f"{f.name}: retired key {'.'.join(path)}")
            if "pixel_spacing_um" in str(key):
                check(any("optics" in p for p in path), f"{f.name}: spacing key outside optics: {'.'.join(path)}")
            if isinstance(value, str):
                for tok in ("scale_matched", "scale_match", "canvas"):
                    check(tok not in value, f"{f.name}: retired token '{tok}' in {'.'.join(path)}")
    return f"{len(files)} JSON files"


@test("F", "F3", "Schema versions of every artifact")
def t_f3(ctx: Ctx) -> None:
    out, pre = ctx.need("out"), ctx.need("pre")
    expected = {
        out / "manifest" / "external_manifest_summary.json": "2.0",
        out / "audit" / "tiff_audit_summary.json": "2.0",
        pre / PRIMARY / "preprocessing_metadata.json": "2.0",
        pre / PHOTOMETRIC / "preprocessing_metadata.json": "2.0",
        out / "evaluation" / PRIMARY / "external_validation_metadata.json": "2.0",
        out / "evaluation" / PHOTOMETRIC / "external_validation_metadata.json": "2.0",
        out / "statistics" / PRIMARY / "external_statistics_summary.json": "2.0",
        out / "statistics" / PHOTOMETRIC / "external_statistics_summary.json": "2.0",
        out / "statistics" / "multi_pipeline_comparison" / "multi_pipeline_summary.json": "3.0",
    }
    for path, version in expected.items():
        check(load_json(path).get("schema_version") == version, f"{path.name}: schema_version != {version}")
    loader = load_json(out / "evaluation" / PRIMARY / "external_validation_metadata.json")["model_loader_metadata"]
    check(loader["schema_version"] == "2.0", "loader schema 2.0")


@test("F", "F4", "optics_sha256 identical across manifest, audit, preprocessing and evaluation")
def t_f4(ctx: Ctx) -> None:
    out, pre = ctx.need("out"), ctx.need("pre")
    h = canonical_sha256(SYN_OPTICS)
    found = {
        "manifest": load_json(out / "manifest" / "external_manifest_summary.json")["optics"]["optics_sha256"],
        "audit": load_json(out / "audit" / "tiff_audit_summary.json")["optics_sha256"],
        "preprocess_locked": load_json(pre / PRIMARY / "preprocessing_metadata.json")["optics_reference"]["optics_sha256"],
        "preprocess_photometric": load_json(pre / PHOTOMETRIC / "preprocessing_metadata.json")["optics_reference"]["optics_sha256"],
        "evaluate_locked": load_json(out / "evaluation" / PRIMARY / "external_validation_metadata.json")["optics_sha256"],
        "evaluate_photometric": load_json(out / "evaluation" / PHOTOMETRIC / "external_validation_metadata.json")["optics_sha256"],
    }
    check(all(v == h for v in found.values()), f"optics hashes differ: {found}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic-only verification of the refactored external-validation pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scripts-dir", type=Path, default=Path(__file__).resolve().parent,
                        help="Directory containing the refactored pipeline scripts.")
    parser.add_argument("--project-config", type=Path, default=None,
                        help="Real external_validation_config.yaml (read as text only).")
    parser.add_argument("--project-metadata", type=Path, default=None,
                        help="Real external_validation_metadata.json (read as text only).")
    parser.add_argument("--project-notebook", type=Path, default=None,
                        help="Real Run.ipynb (source cells scanned; outputs must be cleared).")
    parser.add_argument("--require-final-optics", action="store_true",
                        help="Treat AUTHOR_TO_* placeholders in the project optics as failures.")
    parser.add_argument("--only", type=str, default=None, help="Comma-separated test groups to run, e.g. A,B,E.")
    parser.add_argument("--report", type=Path, default=None, help="Optional path for verification_report.json.")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    scripts_dir = args.scripts_dir.resolve()
    missing_env = []
    for name in PIPELINE_MODULES:
        try:
            importlib.import_module(name)
        except ImportError:
            missing_env.append(name)
    if missing_env:
        print(f"ENVIRONMENT ERROR: pipeline scripts need {missing_env}", file=sys.stderr)
        return 2
    if not scripts_dir.is_dir():
        print(f"ENVIRONMENT ERROR: scripts dir not found: {scripts_dir}", file=sys.stderr)
        return 2

    groups = set(args.only.split(",")) if args.only else None
    results: List[Result] = []
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="verify_ev_") as tmpdir:
        ctx = Ctx(args=args, scripts_dir=scripts_dir, tmp=Path(tmpdir).resolve())
        build_fixtures(ctx)
        for group, test_id, description, fn in TESTS:
            if groups and group not in groups:
                continue
            t0 = time.time()
            try:
                detail = fn(ctx)
                status = "skipped" if isinstance(detail, str) and detail.startswith("SKIPPED") else "passed"
                results.append(Result(group, test_id, description, status, str(detail or ""), time.time() - t0))
            except CheckFailed as exc:
                results.append(Result(group, test_id, description, "failed", str(exc), time.time() - t0))
            except Exception as exc:
                results.append(Result(group, test_id, description, "failed",
                                      f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}", time.time() - t0))
            r = results[-1]
            print(f"[{r.status.upper():7s}] {r.test_id:4s} {r.description} ({r.seconds:.1f}s)")
            if r.status == "failed":
                print("         " + r.detail.replace("\n", "\n         ")[:3000])
        pending = list(ctx.pending_author_input)

    n_failed = sum(r.status == "failed" for r in results)
    n_passed = sum(r.status == "passed" for r in results)
    n_skipped = sum(r.status == "skipped" for r in results)
    summary = {"passed": n_passed, "failed": n_failed, "skipped": n_skipped, "pending_author_input": pending}
    print("\n" + "=" * 80)
    print(f"VERIFICATION {'PASSED' if n_failed == 0 else 'FAILED'}: {n_passed} passed, {n_failed} failed, "
          f"{n_skipped} skipped, {len(pending)} pending author inputs ({time.time() - started:.1f}s)")
    print("=" * 80)

    if args.report:
        report = {
            "schema_version": "1.0",
            "started_utc": datetime.datetime.fromtimestamp(started, datetime.timezone.utc).isoformat(),
            "python": sys.version,
            "packages": {m: str(getattr(importlib.import_module(m), "__version__", "?"))
                         for m in REQUIRED_MODULES + PIPELINE_MODULES},
            "scripts_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(scripts_dir.glob("*.py"))},
            "results": [r.__dict__ for r in results],
            "summary": summary,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Report written to {args.report}")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
