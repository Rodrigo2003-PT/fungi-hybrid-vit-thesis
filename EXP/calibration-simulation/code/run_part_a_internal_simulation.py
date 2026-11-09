#!/usr/bin/env python3
"""
run_part_a_internal_simulation.py

Forward simulation of the external acquisition shift on the
internal held-out cohort (12 slides, 821 fields) with both 
frozen checkpoints (ViT+CCT+LSA_43 and DenseNet-121), inference only,

What this script does
---------------------
1. Gate. Refuses to start unless ``unit_check_report.json`` (verify_unit_checks.py)
   is PASS (or CONDITIONAL with --accept-conditional), the C0 model check ran,
   the imported OpenCV equals the version the unit checks used, and the
   OpenCV boundary rule identified there is re-identified on a held-out frame.
2. Calibration. Loads, or fits through ``photometric_calibration.fit_calibration``,
   the label-free affine map x' = a x + b (48 development slides, 24 unlabeled
   external slides, equal slide weights, B = 10,000 slide bootstrap, seed 42).
   Bootstrap extremes are the joint replicates at the 2.5th / 97.5th percentile
   of s* = a* M_int + b*. Marginal percentiles are reported alongside.
3. Frozen plan. Writes ``part_a_plan_frozen.json`` (conditions, responses, tests,
   Holm families, reproduction thresholds, model / data / calibration hashes)
4. Conditions (34 per model, plus alpha = 0 at any supplementary level), in this order:
      C0 (T=1200, alpha=0), C-S (external-matched T, alpha=0), C-I (T=1200, alpha=1),
      C-SI (external-matched T, alpha=1), the remaining 26 cells of the 6 x 5 surface
      (levels of geometry_spec.json, alpha in ALPHAS), C-I and C-SI at the two bootstrap extremes.
   Levels are identified by their tile side T (geometry_spec.json, derived from the exact optics):
   nominal m = 1, 1.25, 1.5, 1.75, 2 (T = 1200, 960, 800, 686, 600) and the external-matched
   level (T = 1124, m = 1200 / T* = 1.0677, 0.511 um/px at 384 px). Condition identifiers are
   T{T}_a{alpha}_{source}, with a calibration tag when alpha > 0.
   Each field: raw -> /4095 -> affine (a(alpha), b(alpha)) -> margin-aware
   OpenCV-exact Lanczos-4 tiles -> internal clip rule
   (photometric_calibration.internal_condition) -> frozen standardization ->
   FP32 logits. C0 is gated against (tab:test_metrics, tab:strata)
   and, when present, against the unit-check C0 logits (|dz| <= 1e-4, no flip)
5. Cache and resume (colab_state.py). Each condition runs in shards of 2,304
   tiles (whole fields; batch-aligned, so the 32-tile batch partition is that
   of an uninterrupted run). Each shard is committed to <out>/shards with a
   marker holding its SHA-256 and parameter digest; a shard counts as done only
   if it re-hashes correctly. Completed shards are assembled into the condition
   directory (``preproc.npz``, ``<model>.npz``) with ``DONE.json`` written last,
   listing every file's SHA-256. Before each shard the session budget (5 h minus
   a margin) is checked; when it would be exceeded the script exits with code 75
   and the next session resumes at the first missing shard. Inference runs in a
   dedicated thread so CPU preprocessing overlaps the GPU.
6. Statistics (slide level, all tiles of all fields of a slide):
      y   = slide mean of the true-class log-odds z_true - z_other (primary response;
            the raw true-class logit is not identified under a 2-logit softmax,
            Phase 1 audit P7; its slide mean is reported descriptively);
      g   = slide mean of z_glabrata - z_albicans (direction response, secondary);
      p   = slide mean of tile probabilities; predicted species = argmax.
   Contrasts d_S = y_S - y_0, d_I = y_I - y_0, d_SxI = (y_SI - y_S) - (y_I - y_0);
   exact two-sided sign-flip test over all 2^12 = 4,096 sign patterns;
   Holm over 6 tests (3 contrasts x 2 models). The same tests on g form a
   separate, secondary Holm family of 6. Contrasts at the bootstrap extremes are
   reported unadjusted (uncertainty propagation, outside both families).
   Pre-specified reproduction criteria R1 to R3 are evaluated for every condition.
7. Output: ``part_a_summary.json`` and ``part_a_slide_level.csv`` in --out.

Author: Rodrigo Sá
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import queue                                  # noqa: E402
import threading                              # noqa: E402

import colab_state as cs                      # noqa: E402
import geometry_harmonization as geo          # noqa: E402
import optics                                 # noqa: E402
import photometric_calibration as pc          # noqa: E402
import pc_resilience                          # noqa: E402,F401  (crash-safe calibration CSV cache)
import verify_unit_checks as vuc              # noqa: E402

# -----------------------------------------------------------------------------
# Constants fixed
# -----------------------------------------------------------------------------

SCHEMA = "part_a_internal_simulation/2.0"
DIGEST_VERSION = 2
OPERATOR_VERSION = "internal_condition/1"   # bump by hand only when the numerical operator changes
N_FIELDS = 821
N_SLIDES = 12
N_DEV_SLIDES = 48
N_EXT_SLIDES = 24
ALPHA_TEST = 0.05
SIGNFLIP_REL_TOL = 1e-12

MODEL_KEYS: Tuple[str, ...] = ("hybrid43", "densenet121")
MODEL_DISPLAY = {"hybrid43": "ViT+CCT+LSA_43", "densenet121": "DenseNet-121"}
# External locked-pixel collapse: DenseNet-121 -> C. glabrata, ViT+CCT+LSA_43 -> C. albicans
DEFAULT_SPECIES = {"densenet121": 1, "hybrid43": 0}

EXTREMES = ("extreme_low", "extreme_high")
PRIMARY_NAMES = ("C0", "C-S", "C-I", "C-SI")

# Pre-specified reproduction thresholds.
CRITERIA = {
    "R1_direction": "mean slide change in g relative to C0 has the sign of the default species "
                    "(> 0 for C. glabrata, < 0 for C. albicans) and a strict majority (>= 7 of 12) "
                    "of slides is predicted as the default species",
    "R1_min_default_slides": 7,
    "R2_min_default_slides": 10,
    "R3_p_default_threshold": 0.95,
    "R3_min_slides_above_threshold": 10,
    "R3_max_iqr_p_default": 0.02,
    "reproduces": "R1 and R2 and R3",
    "external_reference": "DenseNet-121: 22 of 24 external slides within 3e-6 of p_glabrata = 1; "
                          "ViT+CCT+LSA_43: 22 of 24 with p_glabrata in [0.0080, 0.0085]",
}
CLIP_FLAG_FRACTION = 0.01   # flag a condition whose mean pre-clip out-of-range fraction exceeds 1%

PREPROC_FLOAT_KEYS = ("frac_clipped", "frac_outside_unit", "frac_below_0", "frac_above_1",
                      "frac_outside_tolerance", "min", "max")


def cid_of(T: int, alpha: float, src: str, cal_tag: Optional[str] = None) -> str:
    """T{T}_a{alpha}_{src}; conditions with alpha > 0 depend on (a, b) and carry the calibration tag."""
    base = f"T{int(T)}_a{alpha:g}_{src}"
    if alpha > 0:
        if not cal_tag:
            raise ValueError(f"{base}: alpha > 0 needs a calibration tag")
        return f"{base}_cal{cal_tag}"
    return base


@dataclass(frozen=True)
class Condition:
    cid: str
    T: int
    m: float            # physical scale N_INT / T (label only)
    alpha: float
    ab_source: str      # point | extreme_low | extreme_high
    role: str           # primary | surface | uncertainty | supplementary
    level_role: str     # nominal | external_matched | supplementary


def build_conditions(levels: Sequence[geo.Level], cal_tag: str) -> Tuple[List[Condition], Dict[str, str]]:
    """Conditions in execution order and the alias table (C0, C-S, C-I, C-SI and the extreme variants)."""
    full = next(lv for lv in levels if lv.T == geo.N_INT)
    ext = geo.external_level(levels)

    def mk(lv: geo.Level, al: float, src: str, role: str) -> Condition:
        return Condition(cid_of(lv.T, al, src, cal_tag), lv.T, lv.m, float(al), src, role, lv.role)

    prim = {"C0": (full, 0.0), "C-S": (ext, 0.0), "C-I": (full, 1.0), "C-SI": (ext, 1.0)}
    out = [mk(lv, al, "point", "primary") for lv, al in prim.values()]
    aliases = {n: c.cid for n, c in zip(prim, out)}
    seen = {c.cid for c in out}
    for lv in sorted(levels, key=lambda v: v.m):
        if lv.role == geo.ROLE_SUPPLEMENTARY:
            continue
        for al in pc.ALPHAS:
            c = mk(lv, al, "point", "surface")
            if c.cid not in seen:
                out.append(c)
                seen.add(c.cid)
    for src in EXTREMES:
        for name, lv in (("C-I", full), ("C-SI", ext)):
            c = mk(lv, 1.0, src, "uncertainty")
            out.append(c)
            aliases[f"{name}@{src}"] = c.cid
    n_supp = 0
    for lv in levels:
        if lv.role == geo.ROLE_SUPPLEMENTARY:
            out.append(mk(lv, 0.0, "point", "supplementary"))
            n_supp += 1
    if len(out) != 34 + n_supp or len({c.cid for c in out}) != len(out):
        raise AssertionError(f"expected {34 + n_supp} unique conditions, built {len(out)}")
    return out, aliases


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def _json_default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def write_json_atomic(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=_json_default, allow_nan=True))
    os.replace(tmp, path)


def save_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def digest_of(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=_json_default).encode()).hexdigest()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# -----------------------------------------------------------------------------
# Statistics (pure NumPy; no data access)
# -----------------------------------------------------------------------------

_SIGN_CACHE: Dict[int, np.ndarray] = {}


def sign_matrix(n: int) -> np.ndarray:
    if n not in _SIGN_CACHE:
        k = np.arange(2 ** n, dtype=np.int64)[:, None]
        _SIGN_CACHE[n] = (((k >> np.arange(n, dtype=np.int64)) & 1) * 2 - 1).astype(np.float64)
    return _SIGN_CACHE[n]


def signflip_exact(d: np.ndarray) -> Dict[str, Any]:
    """Exact two-sided sign-flip test of mean(d) = 0 over all 2^n sign patterns (n = 12 -> 4,096)."""
    d = np.asarray(d, dtype=np.float64)
    if d.ndim != 1 or not np.isfinite(d).all():
        raise ValueError("d must be a finite 1-D vector")
    n = d.size
    E = sign_matrix(n)
    t = np.abs(E @ d) / n
    t_obs = abs(float(d.mean()))
    tol = SIGNFLIP_REL_TOL * max(1.0, t_obs)
    count = int((t >= t_obs - tol).sum())
    return {"n": n, "n_patterns": int(E.shape[0]), "mean": float(d.mean()), "median": float(np.median(d)),
            "t_obs_abs_mean": t_obs, "count_ge": count, "p_two_sided": count / E.shape[0],
            "p_floor": 2.0 / E.shape[0], "n_positive": int((d > 0).sum()), "n_negative": int((d < 0).sum()),
            "values": d.tolist()}


def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: (kv[1], kv[0]))
    m = len(items)
    out: Dict[str, float] = {}
    running = 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * float(p)))
        out[k] = running
    return out


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> List[float]:
    from scipy.stats import beta

    if n == 0:
        return [float("nan"), float("nan")]
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return [lo, hi]


def softmax64(z: np.ndarray) -> np.ndarray:
    zz = z.astype(np.float64)
    e = np.exp(zz - zz.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def classification_summary(labels: np.ndarray, pred: np.ndarray) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    recalls = []
    for c, name in enumerate(pc.CLASS_NAMES):
        idx = labels == c
        k, n = int((pred[idx] == c).sum()), int(idx.sum())
        r = k / n if n else float("nan")
        recalls.append(r)
        out[f"recall_{name.split()[-1]}"] = {"k": k, "n": n, "recall": r, "clopper_pearson_95": clopper_pearson(k, n)}
    out["balanced_accuracy"] = float(np.mean(recalls))
    out["n_pred_albicans"] = int((pred == 0).sum())
    out["n_pred_glabrata"] = int((pred == 1).sum())
    return out


def reproduction_criteria(model: str, cond: Dict[str, np.ndarray], c0: Dict[str, np.ndarray]) -> Dict[str, Any]:
    dsp = DEFAULT_SPECIES[model]
    pred = cond["pred"]
    n_def = int((pred == dsp).sum())
    dg = cond["g"] - c0["g"]
    mean_dg = float(dg.mean())
    sign_ok = mean_dg > 0 if dsp == 1 else mean_dg < 0
    p_def = cond["p_glabrata"] if dsp == 1 else 1.0 - cond["p_glabrata"]
    q25, q75 = (float(v) for v in np.percentile(p_def, [25, 75]))
    n_hi = int((p_def >= CRITERIA["R3_p_default_threshold"]).sum())
    r1 = bool(sign_ok and n_def >= CRITERIA["R1_min_default_slides"])
    r2 = bool(n_def >= CRITERIA["R2_min_default_slides"])
    r3 = bool(n_hi >= CRITERIA["R3_min_slides_above_threshold"] and (q75 - q25) <= CRITERIA["R3_max_iqr_p_default"])
    return {"default_species": pc.CLASS_NAMES[dsp], "n_slides_pred_default": n_def, "mean_delta_g_vs_C0": mean_dg,
            "direction_sign_matches": bool(sign_ok), "n_slides_p_default_ge_threshold": n_hi,
            "iqr_p_default": q75 - q25, "p_default_q25_q75": [q25, q75],
            "p_default_min_max": [float(p_def.min()), float(p_def.max())],
            "R1": r1, "R2": r2, "R3": r3, "reproduces": bool(r1 and r2 and r3)}


def interpretation(rep: Dict[str, bool], m_ext: float) -> List[str]:
    """Reduced Part A reading (Phase 1 audit, 3c). Necessity claims wait for Part C."""
    s, i, si = rep.get("C-S", False), rep.get("C-I", False), rep.get("C-SI", False)
    out = []
    if s and not i:
        out.append(f"C-S reproduces and C-I does not: the external-matched geometric scale (m = {m_ext:.4f}) alone "
                   "is sufficient to produce the collapse; scale-specific behaviour does not discriminate "
                   "morphology learning from shortcuts.")
    if i:
        out.append("C-I reproduces: the intensity offset alone is sufficient; Part B decides instrument versus session.")
    if si and not s and not i:
        out.append("C-SI reproduces and neither C-S nor C-I alone does: scale and intensity are jointly sufficient "
                   "(d_SxI expected non-zero).")
    if not (s or i or si):
        out.append("No primary condition reproduces: scale and intensity are insufficient on internal images; the "
                   "failure lies in residual factors (noise, illumination field, session biology).")
    if si and (s or i) and not out:
        out.append("C-SI reproduces together with a single-factor condition; see the single-factor rows.")
    out.append("Sufficiency only: necessity and the faithfulness-transfer reading wait for Part C.")
    return out


# -----------------------------------------------------------------------------
# Gates and inputs
# -----------------------------------------------------------------------------

def unit_check_gate(a: argparse.Namespace) -> Dict[str, Any]:
    path = Path(a.unit_check_dir) / "unit_check_report.json"
    if not path.exists():
        raise RuntimeError(f"{path} not found: run verify_unit_checks.py (with the model branch) first.")
    rep = json.loads(path.read_text())
    status = rep.get("overall_status")
    if status == vuc.FAIL or status is None:
        raise RuntimeError(f"unit_check_report overall_status = {status}; Part A refuses to run.")
    if status == vuc.COND and not a.accept_conditional:
        raise RuntimeError(f"unit_check_report is CONDITIONAL ({rep.get('conditions')}); rerun with "
                           "--accept-conditional to proceed and disclose the conditions.")
    checks = rep.get("checks", {})
    a2 = checks.get("a2_C0_frozen_predictions", {}).get("status")
    if a2 not in (vuc.PASS, vuc.COND):
        raise RuntimeError(f"a2_C0_frozen_predictions status {a2}: the model branch of the unit checks must run.")
    for name in ("d0_border_rule_and_cv2_parity", "d_tile_coordinates_all_levels"):
        if checks.get(name, {}).get("status") != vuc.PASS:
            raise RuntimeError(f"unit check {name} is {checks.get(name, {}).get('status')}, PASS required.")
    # (c) may be CONDITIONAL only for its external E-cal half, which belongs to Part B
    c_st = checks.get("c_affine_resample_commutation", {})
    if c_st.get("status") not in (vuc.PASS, vuc.COND) or c_st.get("internal_worst", 1.0) > vuc.COMMUTE_TOL:
        raise RuntimeError(f"unit check c_affine_resample_commutation failed on internal tiles: {c_st.get('status')}")
    rule = checks["d0_border_rule_and_cv2_parity"]["selected_border_rule"]
    cv2_used = checks.get("environment", {}).get("opencv", {}).get("cv2_version")
    cv2_now = geo.opencv_provenance()["cv2_version"]
    if cv2_used != cv2_now:
        raise RuntimeError(f"OpenCV {cv2_now} differs from the unit-check run ({cv2_used}); install that version.")
    return {"path": str(path), "sha256": pc.sha256_file(path), "overall_status": status,
            "conditions": rep.get("conditions"), "border_rule": rule, "cv2_version": cv2_used,
            "a2_status": a2, "T_external_matched": rep.get("inputs", {}).get("T_external_matched")}


class HeldoutRaw:
    """Raw 12-bit held-out frames, cached once as a uint16 memmap on local disk (resumable)."""

    def __init__(self, bases: Sequence[str], raw_root: Path, work_dir: Path, use_cache: bool) -> None:
        self.bases = list(bases)
        self.root = Path(raw_root)
        self.use_cache = use_cache
        self.path = Path(work_dir) / "heldout_raw_u16.npy"
        self.done_path = Path(work_dir) / "heldout_raw_done.npy"
        self.meta_path = Path(work_dir) / "heldout_raw_meta.json"
        self.mm: Optional[np.ndarray] = None

    def _read(self, base: str) -> np.ndarray:
        raw = pc.read_internal_raw(pc.internal_raw_path(self.root, base))
        if raw.dtype.kind not in "ui":
            raise ValueError(f"{base}: raw dtype {raw.dtype} is not integer")
        if int(raw.min()) < 0 or int(raw.max()) > 65535:
            raise ValueError(f"{base}: raw values outside uint16")
        return raw

    def build(self) -> Dict[str, Any]:
        if not self.use_cache:
            return {"cache": False}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"bases_sha256": digest_of(self.bases), "shape": [len(self.bases), geo.N_INT, geo.N_INT]}
        fresh = not (self.path.exists() and self.meta_path.exists() and json.loads(self.meta_path.read_text()) == meta)
        if fresh:
            mm = np.lib.format.open_memmap(self.path, mode="w+", dtype=np.uint16, shape=tuple(meta["shape"]))
            done = np.zeros(len(self.bases), dtype=bool)
            self.meta_path.write_text(json.dumps(meta))
        else:
            mm = np.load(self.path, mmap_mode="r+")
            done = np.load(self.done_path) if self.done_path.exists() else np.zeros(len(self.bases), dtype=bool)
        todo = np.flatnonzero(~done)
        t0 = time.time()
        for n, i in enumerate(todo, 1):
            mm[i] = self._read(self.bases[i]).astype(np.uint16)
            done[i] = True
            if n % 50 == 0 or n == todo.size:
                mm.flush()
                np.save(self.done_path, done)
                log(f"raw cache {int(done.sum())}/{done.size} ({time.time() - t0:.0f}s)")
        del mm
        self.mm = np.load(self.path, mmap_mode="r")
        return {"cache": True, "path": str(self.path), "n_cached": int(done.sum())}

    def get(self, i: int) -> np.ndarray:
        if self.mm is not None:
            return np.asarray(self.mm[i])
        return self._read(self.bases[i])


class Context:
    def __init__(self, a: argparse.Namespace) -> None:
        self.a = a
        self.out = Path(a.out)
        self.cond_root = self.out / "conditions_v2"          # v1 directories (m-named) are never read
        split = json.loads(Path(a.split).read_text())
        test = list(split["test_base_paths"])
        dev = list(split["train_base_paths"])
        self.bases = sorted(test, key=lambda b: (vuc.slide_of(b), b))
        self.slides = sorted({vuc.slide_of(b) for b in self.bases})
        if len(self.bases) != N_FIELDS or len(self.slides) != N_SLIDES:
            raise RuntimeError(f"held-out set is {len(self.bases)} fields / {len(self.slides)} slides, "
                               f"expected {N_FIELDS} / {N_SLIDES}")
        dev_slides = {vuc.slide_of(b) for b in dev}
        if dev_slides & set(self.slides):
            raise RuntimeError(f"held-out slides overlap development slides: {sorted(dev_slides & set(self.slides))}")
        sidx = {s: k for k, s in enumerate(self.slides)}
        self.field_slide = np.array([sidx[vuc.slide_of(b)] for b in self.bases], dtype=np.int64)
        self.field_label = np.array([vuc.label_of(b) for b in self.bases], dtype=np.int64)
        self.slide_label = np.array([self.field_label[self.field_slide == k][0] for k in range(N_SLIDES)])
        for k in range(N_SLIDES):
            if len(set(self.field_label[self.field_slide == k].tolist())) != 1:
                raise RuntimeError(f"slide {self.slides[k]} mixes labels")
        if set(self.slide_label.tolist()) != {0, 1}:
            raise RuntimeError("held-out slides do not contain both species")
        self.split_sha = pc.sha256_file(Path(a.split))
        self.bases_sha = digest_of(self.bases)
        self.spec = json.loads(Path(a.geometry_spec).read_text())
        self.spec_sha = pc.sha256_file(Path(a.geometry_spec))
        level_list = geo.levels_from_spec(self.spec)            # schema 2.0, strictly ordered, fails otherwise
        self.geometry_digest = str(self.spec["geometry_digest"])
        self.T_ext = int(self.spec["external_matched_T_used"])
        self.ext_level = geo.external_level(level_list)
        self.levels = {lv.T: lv for lv in level_list}
        self.p_int_um = float(self.spec["external_matched"]["p_int_um"])
        for lv in level_list:
            geo.make_axis_plan(lv.T)            # warm the plan cache before threading
        self.aliases: Dict[str, str] = {}
        self.cal_tag = ""
        self.cv2_version = ""
        self.models: Dict[str, Dict[str, Any]] = {}
        self.cal: Dict[str, Any] = {}
        self.cal_sha = ""
        self.border_rule = ""
        self.raw: Optional[HeldoutRaw] = None
        self.store = cs.UnitStore(self.out / "shards_v2")
        self.budget = cs.SessionBudget(a.budget_hours, a.margin_minutes, a.session_start)
        self.timings = cs.Timings(self.out / "timings.json")
        self.lock = cs.RunLock(self.out / "RUN_LOCK.json")


# -----------------------------------------------------------------------------
# Calibration
# -----------------------------------------------------------------------------

def obtain_calibration(ctx: Context, border_rule: str) -> None:
    a = ctx.a
    cdir = Path(a.calibration_dir) if a.calibration_dir else ctx.out / "calibration"
    path = cdir / "calibration.json"
    if not path.exists():
        miss = [n for n in ("external_manifest", "pinned_scripts") if getattr(a, n) is None]
        if miss:
            raise RuntimeError(f"{path} absent and fitting needs {miss}")
        log(f"fitting label-free calibration into {cdir} (cached and resumable) ...")
        pc.fit_calibration(split_json=Path(a.split), internal_raw_root=Path(a.internal_raw_root),
                           external_manifest=Path(a.external_manifest),
                           external_root=Path(a.external_root) if a.external_root else None,
                           pinned_scripts=Path(a.pinned_scripts), geometry_spec=Path(a.geometry_spec),
                           out_dir=cdir, params=pc.MaskParams(), n_boot=a.n_boot, seed=a.seed,
                           workers=a.calib_workers, border_rule=border_rule)
    cal = json.loads(path.read_text())
    if cal.get("schema") != pc.CALIBRATION_SCHEMA:
        raise RuntimeError(f"{path}: schema {cal.get('schema')!r} != {pc.CALIBRATION_SCHEMA}; legacy calibration, "
                           "archive it (migrate stage) and refit")
    if cal["geometry"].get("geometry_digest") != ctx.geometry_digest:
        raise RuntimeError(f"{path}: geometry digest differs from geometry_spec.json; refit the calibration")
    cnt = cal["counts"]
    if cnt["internal_slides"] != N_DEV_SLIDES or cnt["external_slides"] != N_EXT_SLIDES:
        raise RuntimeError(f"calibration counts {cnt} != 48 internal / 24 external slides")
    if int(cal["geometry"]["T_external_matched"]) != ctx.T_ext:
        raise RuntimeError(f"calibration T {cal['geometry']['T_external_matched']} != geometry spec T {ctx.T_ext}")
    if cal["geometry"]["border_rule"] != border_rule:
        raise RuntimeError(f"calibration border rule {cal['geometry']['border_rule']} != {border_rule}")
    if set(cal["slide_medians"]["internal"]) & set(ctx.slides):
        raise RuntimeError("calibration used held-out slides")
    for key in ("point", "bootstrap"):
        if key not in cal:
            raise RuntimeError(f"calibration.json lacks '{key}'")
    for src in ("point", *EXTREMES):
        aa, _ = ab_for(cal, src)
        if not (aa > 0 and math.isfinite(aa)):
            raise RuntimeError(f"calibration {src}: a = {aa}")
    ctx.cal, ctx.cal_sha = cal, pc.sha256_file(path)
    ctx.cal_tag = pc.calibration_digest(cal)[:8]
    pt, bt = cal["point"], cal["bootstrap"]
    log(f"a = {pt['a']:.6f} (95% {bt['a_percentile_95']}), b = {pt['b']:.6f} (95% {bt['b_percentile_95']}); "
        f"extremes low ({bt['extreme_low']['a']:.6f}, {bt['extreme_low']['b']:.6f}), "
        f"high ({bt['extreme_high']['a']:.6f}, {bt['extreme_high']['b']:.6f})")


def ab_for(cal: Dict[str, Any], src: str) -> Tuple[float, float]:
    if src == "point":
        return float(cal["point"]["a"]), float(cal["point"]["b"])
    e = cal["bootstrap"][src]
    return float(e["a"]), float(e["b"])


def condition_params(ctx: Context, c: Condition) -> Dict[str, Any]:
    a_src, b_src = ab_for(ctx.cal, c.ab_source)
    a_al, b_al = pc.dose_params(a_src, b_src, c.alpha)
    lv = ctx.levels[c.T]
    return {"cid": c.cid, "T": lv.T, "m": lv.m, "m_exact": optics.fstr(lv.m_exact), "m_label": lv.label,
            "level_role": lv.role, "alpha": c.alpha, "ab_source": c.ab_source, "role": c.role,
            "a_source": a_src, "b_source": b_src, "a_alpha": a_al, "b_alpha": b_al,
            "identity_intensity": bool(a_al == 1.0 and b_al == 0.0),
            "n_tiles_per_field": lv.n_tiles, "offsets": list(lv.offsets),
            "frame_edge_share_of_tile_sides": lv.frame_edge_share_of_tile_sides,
            "fov_um": lv.T * ctx.p_int_um, "output_spacing_um": lv.T * ctx.p_int_um / geo.D}


def condition_digest_v2(*, bases_sha: str, T: int, offsets: Sequence[int], margin: int, border_rule: str,
                        a_alpha: float, b_alpha: float, models: Dict[str, Dict[str, Any]], batch_size: int,
                        chunk_tiles: int, shard_tiles: int, cv2_version: str,
                        operator_version: str = OPERATOR_VERSION, int_denom: int = pc.INT_DENOM) -> str:
    """
    Digest of exactly the inputs that fix a condition's floating-point arrays. Labels (cid, m, aliases),
    the geometry_spec file bytes and, for identity-intensity cells, the calibration are excluded: they do
    not change any number. Pure function (also used by migrate_part_ab.py).
    """
    return digest_of({"digest_version": DIGEST_VERSION, "operator_version": operator_version,
                      "int_denom": int(int_denom), "bases": bases_sha, "T": int(T), "offsets": [int(o) for o in offsets],
                      "margin": int(margin), "border_rule": border_rule,
                      "a_alpha": float(a_alpha).hex(), "b_alpha": float(b_alpha).hex(),
                      "models": {k: {"sha256": v["sha256"], "mu": float(v["mu"]).hex(), "sigma": float(v["sigma"]).hex()}
                                 for k, v in sorted(models.items())},
                      "batch_size": int(batch_size), "chunk_tiles": int(chunk_tiles), "shard_tiles": int(shard_tiles),
                      "cv2_version": str(cv2_version)})


def condition_digest(ctx: Context, p: Dict[str, Any]) -> str:
    return condition_digest_v2(bases_sha=ctx.bases_sha, T=p["T"], offsets=p["offsets"], margin=geo.MARGIN_PX,
                               border_rule=ctx.border_rule, a_alpha=p["a_alpha"], b_alpha=p["b_alpha"],
                               models=ctx.models, batch_size=ctx.a.batch_size, chunk_tiles=ctx.a.chunk_tiles,
                               shard_tiles=SHARD_TILES, cv2_version=ctx.cv2_version)


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

def load_models(ctx: Context) -> Dict[str, Any]:
    det = vuc._configure_torch(ctx.a.seed)
    import torch

    if not torch.cuda.is_available():
        log("WARNING: CUDA unavailable; running on CPU (slow, and not the target T4 environment).")
    ctx.models = vuc.load_models(ctx.a)
    for k, v in ctx.models.items():
        if not (v["sigma"] > 0 and math.isfinite(v["mu"])):
            raise RuntimeError(f"{k}: invalid frozen statistics mu={v['mu']} sigma={v['sigma']}")
    return det


# -----------------------------------------------------------------------------
# Plan
# -----------------------------------------------------------------------------

def build_plan(ctx: Context, conds: Sequence[Condition], gate: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "schema": SCHEMA,
        "protocol_scope": "Protocol.md Part A only (Parts B and C are separate runs)",
        "conditions": [condition_params(ctx, c) for c in conds],
        "aliases": ctx.aliases,
        "aggregation": "slide = mean over all tiles of all fields of the slide (equal tiles per field)",
        "responses": {
            "primary_y": "slide mean of true-class log-odds z_true - z_other",
            "secondary_g": "slide mean of glabrata log-odds z_glabrata - z_albicans",
            "descriptive": ["slide mean raw true-class logit z_true", "slide mean probabilities",
                            "predicted species = argmax of slide mean probabilities (tie -> C. albicans)"],
        },
        "contrasts": {"d_S": "y[C-S] - y[C0]", "d_I": "y[C-I] - y[C0]",
                      "d_SxI": "(y[C-SI] - y[C-S]) - (y[C-I] - y[C0])"},
        "test": {"name": "exact two-sided sign-flip test of the mean slide contrast",
                 "statistic": "|sum_s e_s d_s| / 12", "n_patterns": 4096,
                 "tie_tolerance": f"{SIGNFLIP_REL_TOL} * max(1, T_obs)"},
        "families": {"primary": "Holm over {d_S, d_I, d_SxI} on y x 2 models (m = 6)",
                     "secondary": "Holm over {d_S, d_I, d_SxI} on g x 2 models (m = 6), reported separately",
                     "uncertainty": "d_I and d_SxI at the bootstrap extremes, unadjusted"},
        "alpha": ALPHA_TEST,
        "reproduction_criteria": CRITERIA,
        "default_species": {k: pc.CLASS_NAMES[v] for k, v in DEFAULT_SPECIES.items()},
        "bootstrap_extremes": "joint replicates at the 2.5th / 97.5th nearest-rank percentile of "
                              "s* = a* M_int + b* (calibration.json bootstrap.extreme_low / extreme_high)",
        "clip_flag_fraction": CLIP_FLAG_FRACTION,
        "models": {k: {"display": MODEL_DISPLAY[k], "sha256": v["sha256"], "mu": v["mu"], "sigma": v["sigma"],
                       "optimal_epochs": v.get("optimal_epochs")} for k, v in sorted(ctx.models.items())},
        "data": {"split_sha256": ctx.split_sha, "heldout_bases_digest": ctx.bases_sha, "n_fields": N_FIELDS,
                 "slides": ctx.slides, "slide_labels": [pc.CLASS_NAMES[int(v)] for v in ctx.slide_label]},
        "geometry_digest": ctx.geometry_digest, "T_external_matched": ctx.T_ext,
        "optics": ctx.spec["optics"],
        "levels": [geo.level_summary(lv, ctx.p_int_um) for lv in ctx.levels.values()],
        "external_matched": {k: ctx.spec["external_matched"][k] for k in
                             ("T_star_exact", "T", "m_external", "fov_int_um", "fov_ext_um", "relative_residual")},
        "digest_version": DIGEST_VERSION, "operator_version": OPERATOR_VERSION,
        "protocol_deviation": {"text": ctx.a.protocol_deviation or "",
                               "status": "recorded" if ctx.a.protocol_deviation else
                               "AUTHOR TEXT REQUIRED: Protocol.md specified m = 2.165 (legacy 0.043 um/px external "
                               "spacing, 2 x 2 binning omitted); the external-matched level now follows the optics"},
        "migration_manifest_sha256": pc.sha256_file(Path(ctx.a.migration_manifest)) if ctx.a.migration_manifest else None,
        "border_rule": ctx.border_rule, "margin_px": geo.MARGIN_PX,
        "calibration_sha256": ctx.cal_sha,
        "calibration_digest": pc.calibration_digest(ctx.cal),
        "calibration_point": ctx.cal["point"],
        "calibration_extremes": {s: ctx.cal["bootstrap"][s] for s in EXTREMES},
        "unit_check_report_sha256": gate["sha256"], "unit_check_status": gate["overall_status"],
        "script_sha256": pc.sha256_file(Path(__file__)),
        "seed": ctx.a.seed, "batch_size": ctx.a.batch_size, "precision": "FP32 (autocast forbidden, TF32 off)",
    }


def freeze_plan(ctx: Context, plan: Dict[str, Any]) -> Dict[str, Any]:
    path = ctx.out / "part_a_plan_frozen.json"
    body_digest = digest_of(plan)
    history: List[Dict[str, Any]] = []
    if path.exists():
        old = json.loads(path.read_text())
        history = old.get("plan_history", [])
        if old.get("plan_digest") == body_digest:
            log(f"frozen plan matches ({body_digest[:12]})")
            return old
        if not ctx.a.replan:
            diff = sorted(k for k in set(plan) | set(old.get("plan", {}))
                          if plan.get(k) != old.get("plan", {}).get(k))
            raise RuntimeError(f"plan differs from {path} in {diff}; rerun with --replan to archive and replace it.")
        arch = path.with_name(f"part_a_plan_frozen.superseded_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
        shutil.copy2(path, arch)
        history.append({"superseded_digest": old.get("plan_digest"), "archived_as": arch.name,
                        "replaced_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        log(f"plan replaced; previous archived as {arch.name}")
    rec = {"frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "plan_digest": body_digest,
           "plan_history": history, "plan": plan}
    write_json_atomic(rec, path)
    log(f"wrote frozen plan {path} ({body_digest[:12]})")
    return rec


# -----------------------------------------------------------------------------
# Condition execution with caching
# -----------------------------------------------------------------------------

DEFAULT_BATCH_SIZE, DEFAULT_CHUNK_TILES = 32, 288
SHARD_TILES = 2304   # 8 x 288: whole fields at 1, 4 and 9 tiles per field (checked against the actual levels),
                     # and a multiple of the 32-tile batch


def shard_ranges(nt: int) -> List[Tuple[int, int]]:
    """Field ranges of a condition's shards. Each shard starts at a global tile index that is a multiple of
    SHARD_TILES, hence of the batch size, so the batch partition equals the unsharded one (32-tile batches
    over the condition's global tile order, last one partial)."""
    if SHARD_TILES % nt:
        raise ValueError(f"SHARD_TILES {SHARD_TILES} is not a multiple of {nt} tiles per field")
    fps = SHARD_TILES // nt
    return [(f0, min(N_FIELDS, f0 + fps)) for f0 in range(0, N_FIELDS, fps)]


def _done_record(cdir: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads((cdir / "DONE.json").read_text())
    except (OSError, ValueError):
        return None


def condition_complete(ctx: Context, cdir: Path, dg: str) -> bool:
    """Digest matches AND every file re-hashes to the SHA-256 recorded in DONE.json (written last)."""
    rec = _done_record(cdir)
    return bool(rec and rec.get("digest") == dg and cs.verify_files(cdir, rec)
                and set(rec["files"]) >= {"preproc.npz", *(f"{k}.npz" for k in MODEL_KEYS)})


def _prep_field(ctx: Context, i: int, level: geo.Level, a_al: float, b_al: float):
    tiles, clips, norm = pc.internal_condition(ctx.raw.get(i), level, a_al, b_al, border_rule=ctx.border_rule)
    return i, tiles, clips, norm


class _InferenceWorker:
    """One thread owns every CUDA call; it consumes chunks in submission order, so outputs and the batch
    partition are identical to calling vuc.logits_for inline, while CPU preprocessing keeps running."""

    def __init__(self, ctx: Context, logits: Dict[str, np.ndarray]) -> None:
        self.ctx, self.logits, self.err = ctx, logits, None
        self.q: "queue.Queue" = queue.Queue(maxsize=max(1, ctx.a.inference_queue))
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def _run(self) -> None:
        while True:
            item = self.q.get()
            if item is None:
                return
            if self.err is not None:
                continue
            start, arrs = item
            try:
                for k in MODEL_KEYS:
                    m = self.ctx.models[k]
                    self.logits[k][start:start + len(arrs)] = vuc.logits_for(m["model"], arrs, m["mu"], m["sigma"],
                                                                             self.ctx.a.batch_size)
            except BaseException as exc:  # surfaced in the main thread
                self.err = exc

    def put(self, start: int, arrs: List[np.ndarray]) -> None:
        if self.err is not None:
            raise self.err
        self.q.put((start, arrs))

    def close(self) -> None:
        self.q.put(None)
        self.th.join()
        if self.err is not None:
            raise self.err


def run_shard(ctx: Context, c: Condition, p: Dict[str, Any], level: geo.Level, f0: int, f1: int) -> Dict[str, np.ndarray]:
    nt = level.n_tiles
    n = (f1 - f0) * nt
    stats = {k: np.full(n, np.nan, dtype=np.float32) for k in PREPROC_FLOAT_KEYS}
    trig = np.zeros(n, dtype=bool)
    norm_clip = np.zeros(f1 - f0, dtype=bool)
    raw_over = np.zeros(f1 - f0, dtype=np.int64)
    logits = {k: np.full((n, 2), np.nan, dtype=np.float32) for k in MODEL_KEYS}
    worker = _InferenceWorker(ctx, logits)
    a_al, b_al = p["a_alpha"], p["b_alpha"]
    buf: List[np.ndarray] = []
    buf_start = 0
    try:
        with ThreadPoolExecutor(max_workers=max(1, ctx.a.prep_workers)) as ex:
            pending: deque = deque()
            nxt = f0
            while nxt < f1 and len(pending) < ctx.a.prefetch_fields:
                pending.append(ex.submit(_prep_field, ctx, nxt, level, a_al, b_al))
                nxt += 1
            while pending:
                i, tiles, clips, norm = pending.popleft().result()
                if nxt < f1:
                    pending.append(ex.submit(_prep_field, ctx, nxt, level, a_al, b_al))
                    nxt += 1
                if tiles.shape != (nt, geo.D, geo.D):
                    raise RuntimeError(f"{c.cid} field {i}: tiles {tiles.shape}")
                pos = (i - f0) * nt
                if pos != buf_start + len(buf):
                    raise RuntimeError("field order broken")
                for j, rec in enumerate(clips):
                    for k in PREPROC_FLOAT_KEYS:
                        stats[k][pos + j] = rec[k]
                    trig[pos + j] = bool(rec["clip_triggered"])
                norm_clip[i - f0] = bool(norm["normalization_clip_applied"])
                raw_over[i - f0] = int(norm["raw_over_4095"])
                buf.extend(tiles[j] for j in range(nt))
                if len(buf) >= ctx.a.chunk_tiles:
                    worker.put(buf_start, buf)
                    buf_start += len(buf)
                    buf = []
            if buf:
                worker.put(buf_start, buf)
                buf_start += len(buf)
                buf = []
    finally:
        worker.close()
    if buf_start != n:
        raise RuntimeError(f"{c.cid} shard [{f0}, {f1}): inferred {buf_start} tiles, expected {n}")
    for k in MODEL_KEYS:
        if not np.isfinite(logits[k]).all():
            raise RuntimeError(f"{c.cid}/{k}: non-finite logits in shard [{f0}, {f1})")
    return {"trig": trig, "norm_clip": norm_clip, "raw_over": raw_over, **{f"stat_{k}": v for k, v in stats.items()},
            **{f"logits_{k}": v for k, v in logits.items()}}


def run_condition(ctx: Context, c: Condition) -> None:
    p = condition_params(ctx, c)
    dg = condition_digest(ctx, p)
    cdir = ctx.cond_root / c.cid
    if condition_complete(ctx, cdir, dg):
        rec = _done_record(cdir) or {}
        if rec.get("migrated_from") and not spot_check_migrated(ctx, c, p, cdir):
            log(f"{c.cid}: migrated arrays failed the spot recomputation; recomputing")
        else:
            log(f"{c.cid}: cached (verified)")
            return
    level = ctx.levels[c.T]
    nt = level.n_tiles
    ranges = shard_ranges(nt)
    parts: List[Dict[str, np.ndarray]] = []
    for s, (f0, f1) in enumerate(ranges):
        unit = f"{c.cid}/shard_{s:03d}"
        sdg = digest_of({"condition": dg, "shard": s, "fields": [f0, f1], "shard_tiles": SHARD_TILES,
                         "chunk_tiles": ctx.a.chunk_tiles})
        got = ctx.store.load_npz(unit, sdg)
        if got is None:
            kind = f"partA_shard_nt{nt}"
            ctx.budget.require(ctx.timings.estimate(kind, 240.0 * (f1 - f0) * nt / SHARD_TILES), unit)
            t0 = time.time()
            got = run_shard(ctx, c, p, level, f0, f1)
            dt = time.time() - t0
            ctx.store.commit_npz(unit, sdg, got, meta={"seconds": round(dt, 1), "fields": [f0, f1]})
            ctx.timings.record(kind, dt * SHARD_TILES / ((f1 - f0) * nt))
            ctx.lock.beat()
            log(f"  {unit}: {(f1 - f0) * nt} tiles in {dt:.0f}s; {ctx.budget.remaining() / 60:.0f} min left")
            got = ctx.store.load_npz(unit, sdg)   # use exactly what is on disk
            if got is None:
                raise RuntimeError(f"{unit}: committed shard failed verification")
        parts.append(got)

    cat = lambda key: np.concatenate([q[key] for q in parts])  # noqa: E731
    field_idx = np.repeat(np.arange(N_FIELDS, dtype=np.int32), nt)
    tile_idx = np.tile(np.arange(nt, dtype=np.int16), N_FIELDS)
    stats = {k: cat(f"stat_{k}") for k in PREPROC_FLOAT_KEYS}
    payloads = {"preproc.npz": cs.npz_bytes(field_idx=field_idx, tile_idx=tile_idx, clip_triggered=cat("trig"),
                                            normalization_clip_applied=cat("norm_clip"), raw_over_4095=cat("raw_over"),
                                            bases=np.array(ctx.bases), **stats)}
    for k in MODEL_KEYS:
        z = cat(f"logits_{k}")
        if z.shape != (N_FIELDS * nt, 2):
            raise RuntimeError(f"{c.cid}/{k}: assembled logits {z.shape}")
        payloads[f"{k}.npz"] = cs.npz_bytes(logits=z, probs=softmax64(z), field_idx=field_idx, tile_idx=tile_idx)
    cdir.mkdir(parents=True, exist_ok=True)
    cs.write_verified_files(cdir, payloads, "DONE.json",
                            {"digest": dg, "params": p, "n_tiles": N_FIELDS * nt, "n_shards": len(ranges),
                             "shard_tiles": SHARD_TILES,
                             "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    if not condition_complete(ctx, cdir, dg):
        raise RuntimeError(f"{c.cid}: assembled condition failed verification")
    log(f"{c.cid}: assembled {N_FIELDS * nt} tiles from {len(ranges)} shard(s); mean clipped fraction "
        f"{float(stats['frac_clipped'].mean()):.2e}")


def spot_check_migrated(ctx: Context, c: Condition, p: Dict[str, Any], cdir: Path) -> bool:
    """
    Controlled recomputation for a condition re-signed by migrate_part_ab.py: the first --chunk-tiles tiles
    (whole fields, batch-aligned, so the 32-tile batches equal those of the original run) are recomputed and
    compared with the migrated arrays. Bitwise equality passes; |dz| <= LOGIT_TOL with no label flip passes
    as a disclosed exception; anything else moves the directory aside and returns False. The verdict is kept
    in SPOTCHECK.json (outside the files hashed by DONE.json).
    """
    sc = cdir / "SPOTCHECK.json"
    if sc.exists():
        try:
            if json.loads(sc.read_text()).get("status") in ("bitwise", "within_tolerance"):
                return True
        except ValueError:
            pass
    level = ctx.levels[c.T]
    nf = ctx.a.chunk_tiles // level.n_tiles
    got = run_shard(ctx, c, p, level, 0, nf)
    pre = dict(np.load(cdir / "preproc.npz", allow_pickle=False))
    n = nf * level.n_tiles
    same_pre = all(np.array_equal(pre[k][:n].view(np.uint32), got[f"stat_{k}"].view(np.uint32))
                   for k in PREPROC_FLOAT_KEYS) and np.array_equal(pre["clip_triggered"][:n], got["trig"])
    worst, flips, bitwise = 0.0, 0, same_pre
    for k in MODEL_KEYS:
        z_old = np.load(cdir / f"{k}.npz", allow_pickle=False)["logits"][:n]
        z_new = got[f"logits_{k}"]
        bitwise &= bool(np.array_equal(z_old.view(np.uint32), z_new.view(np.uint32)))
        worst = max(worst, float(np.max(np.abs(z_old.astype(np.float64) - z_new))))
        flips += int((z_old.argmax(1) != z_new.argmax(1)).sum())
    status = ("bitwise" if bitwise else
              "within_tolerance" if same_pre and worst <= vuc.LOGIT_TOL and flips == 0 else "failed")
    write_json_atomic({"status": status, "tiles": n, "preproc_bitwise": bool(same_pre), "max_abs_logit_diff": worst,
                       "label_flips": flips, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, sc)
    log(f"{c.cid}: migrated spot check {status} ({n} tiles, max |dz| {worst:.2e})")
    if status == "failed":
        aside = ctx.cond_root / f"_rejected_{c.cid}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
        os.replace(cdir, aside)
        return False
    return True


def load_condition(ctx: Context, cid: str) -> Optional[Dict[str, Any]]:
    cdir = ctx.cond_root / cid
    rec = _done_record(cdir)
    if rec is None:
        return None
    if not cs.verify_files(cdir, rec):
        raise RuntimeError(f"{cid}: files do not match the SHA-256 recorded in DONE.json; delete {cdir} and rerun")
    pre = dict(np.load(cdir / "preproc.npz", allow_pickle=False))
    if list(pre["bases"]) != ctx.bases:
        raise RuntimeError(f"{cid}: cached field order differs from the current held-out list")
    out = {"done": rec, "pre": pre, "logits": {}}
    for k in MODEL_KEYS:
        d = np.load(cdir / f"{k}.npz", allow_pickle=False)
        if not np.array_equal(d["field_idx"], pre["field_idx"]):
            raise RuntimeError(f"{cid}/{k}: tile index mismatch with preproc.npz")
        out["logits"][k] = d["logits"]
    return out


# -----------------------------------------------------------------------------
# C0 regression gate
# -----------------------------------------------------------------------------

def c0_gate(ctx: Context) -> Dict[str, Any]:
    data = load_condition(ctx, ctx.aliases["C0"])
    if data is None:
        raise RuntimeError("C0 not cached")
    if not np.array_equal(data["pre"]["field_idx"], np.arange(N_FIELDS)):
        raise RuntimeError("C0 must hold exactly one tile per field")
    res: Dict[str, Any] = {}
    ok = True
    for k in MODEL_KEYS:
        z = data["logits"][k]
        summ = vuc.heldout_summary(ctx.bases, z)
        exp = vuc.EXPECTED[k]
        mism = [f"{blk}.{key}: got {summ[blk][key]}, main.tex {v}" for blk in ("counts", "metrics_4dp")
                for key, v in exp[blk].items() if summ[blk][key] != v]
        rec: Dict[str, Any] = {"summary": summ, "mismatches_vs_main_tex": mism}
        model_ok = not mism
        cache = Path(ctx.a.unit_check_dir) / "logit_cache" / f"C0_recomputed_{k}.npz"
        if cache.exists():
            ref = np.load(cache, allow_pickle=False)
            row = {b: r for r, b in enumerate(ref["bases"].tolist())}
            if set(row) != set(ctx.bases):
                raise RuntimeError(f"{cache}: patch set differs")
            zr = ref["logits"][[row[b] for b in ctx.bases]]
            d = float(np.max(np.abs(zr.astype(np.float64) - z)))
            flips = int((zr.argmax(1) != z.argmax(1)).sum())
            rec["vs_unit_check_C0_logits"] = {"max_abs_logit_diff": d, "label_flips": flips,
                                              "tolerance": vuc.LOGIT_TOL}
            model_ok &= d <= vuc.LOGIT_TOL and flips == 0
        else:
            rec["vs_unit_check_C0_logits"] = "not available"
        rec["PASS"] = bool(model_ok)
        ok &= model_ok
        res[k] = rec
    res["PASS"] = bool(ok)
    write_json_atomic(res, ctx.out / "c0_regression_gate.json")
    if not ok:
        raise RuntimeError(f"C0 regression gate FAILED (see {ctx.out / 'c0_regression_gate.json'}); "
                           "no other condition is trusted.")
    log("C0 regression gate PASS for both models")
    return res


# -----------------------------------------------------------------------------
# Aggregation and summary
# -----------------------------------------------------------------------------

def slide_stats(ctx: Context, data: Dict[str, Any], k: str) -> Dict[str, np.ndarray]:
    fi = data["pre"]["field_idx"].astype(np.int64)
    z = data["logits"][k].astype(np.float64)
    s = ctx.field_slide[fi]
    y = ctx.field_label[fi]
    ar = np.arange(z.shape[0])
    l_true = z[ar, y] - z[ar, 1 - y]
    g = z[:, 1] - z[:, 0]
    z_true = z[ar, y]
    p = softmax64(z)
    n = np.bincount(s, minlength=N_SLIDES).astype(np.float64)

    def mean(v: np.ndarray) -> np.ndarray:
        return np.bincount(s, weights=v, minlength=N_SLIDES) / n

    p0, p1 = mean(p[:, 0]), mean(p[:, 1])
    tile_pred = (p[:, 1] > p[:, 0]).astype(np.float64)
    return {"n_tiles": n, "y": mean(l_true), "g": mean(g), "z_true": mean(z_true),
            "p_albicans": p0, "p_glabrata": p1, "pred": (p1 > p0).astype(np.int64),
            "tile_frac_pred_glabrata": mean(tile_pred),
            "clip_frac_mean": mean(data["pre"]["frac_clipped"].astype(np.float64)),
            "outside_unit_frac_mean": mean(data["pre"]["frac_outside_unit"].astype(np.float64))}


def clip_summary(data: Dict[str, Any]) -> Dict[str, Any]:
    pre = data["pre"]
    fo = pre["frac_outside_unit"].astype(np.float64)
    fc = pre["frac_clipped"].astype(np.float64)
    return {"mean_frac_pixels_outside_unit_preclip": float(fo.mean()), "max_frac_outside_unit_tile": float(fo.max()),
            "mean_frac_pixels_clipped": float(fc.mean()), "frac_tiles_clip_triggered": float(pre["clip_triggered"].mean()),
            "frac_tiles_outside_tolerance": float((pre["frac_outside_tolerance"] > 0).mean()),
            "n_fields_normalization_clip": int(pre["normalization_clip_applied"].sum()),
            "flag_heavy_clipping": bool(fo.mean() > CLIP_FLAG_FRACTION)}


def per_slide_records(ctx: Context, st: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
    return [{"slide_id": ctx.slides[i], "label": pc.CLASS_NAMES[int(ctx.slide_label[i])],
             "n_tiles": int(st["n_tiles"][i]), "y_true_logodds": float(st["y"][i]), "g_glabrata_logodds": float(st["g"][i]),
             "z_true_raw_logit": float(st["z_true"][i]), "p_albicans": float(st["p_albicans"][i]),
             "p_glabrata": float(st["p_glabrata"][i]), "pred": pc.CLASS_NAMES[int(st["pred"][i])],
             "correct": bool(st["pred"][i] == ctx.slide_label[i]),
             "tile_frac_pred_glabrata": float(st["tile_frac_pred_glabrata"][i]),
             "clip_frac_mean": float(st["clip_frac_mean"][i]),
             "outside_unit_frac_mean": float(st["outside_unit_frac_mean"][i])} for i in range(N_SLIDES)]


def contrasts(st: Dict[str, Dict[str, np.ndarray]], key: str, ci: str = "C-I", csi: str = "C-SI") -> Dict[str, np.ndarray]:
    y0, ys, yi, ysi = (st[n][key] for n in ("C0", "C-S", ci, csi))
    return {"d_S": ys - y0, "d_I": yi - y0, "d_SxI": (ysi - ys) - (yi - y0)}


def summarize(ctx: Context, conds: Sequence[Condition], plan_rec: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    plan = plan_rec["plan"]
    ALIASES = ctx.aliases
    stats: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {k: {} for k in MODEL_KEYS}
    cond_out: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []
    missing = []
    inv_alias = {v: k for k, v in ALIASES.items()}
    for c in conds:
        data = load_condition(ctx, c.cid)
        if data is None:
            missing.append(c.cid)
            continue
        rec = {"params": data["done"]["params"], "alias": inv_alias.get(c.cid), "clip": clip_summary(data),
               "seconds": data["done"].get("seconds"), "models": {}}
        for k in MODEL_KEYS:
            st = slide_stats(ctx, data, k)
            stats[k][c.cid] = st
            rec["models"][k] = {"classification": classification_summary(ctx.slide_label, st["pred"]),
                                "mean_y": float(st["y"].mean()), "mean_g": float(st["g"].mean()),
                                "mean_z_true_raw": float(st["z_true"].mean()),
                                "slides": per_slide_records(ctx, st)}
            for r in rec["models"][k]["slides"]:
                rows.append({"condition": c.cid, "alias": inv_alias.get(c.cid, ""), "T": c.T, "m": c.m,
                             "level_role": c.level_role, "alpha": c.alpha,
                             "ab_source": c.ab_source, "model": MODEL_DISPLAY[k], **r})
        cond_out[c.cid] = rec

    c0_id = ALIASES["C0"]
    for c in conds:
        if c.cid in cond_out and c0_id in cond_out:
            for k in MODEL_KEYS:
                cond_out[c.cid]["models"][k]["reproduction"] = reproduction_criteria(k, stats[k][c.cid], stats[k][c0_id])

    summary: Dict[str, Any] = {
        "schema": SCHEMA, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "plan_digest": plan_rec["plan_digest"], "plan_history": plan_rec.get("plan_history", []),
        "missing_conditions": missing, "complete": not missing, **extra,
    }

    cal = ctx.cal or {"point": plan["calibration_point"], "bootstrap": {**plan["calibration_extremes"]}}
    shift = {}
    for k, m in plan["models"].items():
        aa, bb = float(cal["point"]["a"]), float(cal["point"]["b"])
        shift[k] = {"mu": m["mu"], "sigma": m["sigma"],
                    "delta_training_mean_in_training_sd": (aa * m["mu"] + bb - m["mu"]) / m["sigma"]}
    summary["calibration"] = {
        "sha256": plan["calibration_sha256"], "point": cal["point"],
        "a_percentile_95": cal["bootstrap"].get("a_percentile_95"), "b_percentile_95": cal["bootstrap"].get("b_percentile_95"),
        "extreme_low": cal["bootstrap"].get("extreme_low"), "extreme_high": cal["bootstrap"].get("extreme_high"),
        "n_boot": cal["bootstrap"].get("n_boot"), "seed": cal["bootstrap"].get("seed"),
        "sensitivity_contrast_definition": cal.get("sensitivity_contrast_definition"),
        "dose_response_point": cal.get("dose_response_point"),
        "factor_N_noise": cal.get("factor_N_noise"),
        "training_mean_shift_under_forward_map": shift,
    }
    em = ctx.spec["external_matched"]
    summary["geometry"] = {"geometry_digest": ctx.geometry_digest, "geometry_spec_sha256": ctx.spec_sha,
                           "T_external_matched": ctx.T_ext, "m_external": em["m_external"],
                           "T_star_exact": em["T_star_exact"], "relative_residual": em["relative_residual"],
                           "fov_um": {"internal": em["fov_int_um"], "external": em["fov_ext_um"]},
                           "spacing_at_384_um": {"external": em["spacing_external_at_384_um"],
                                                 "external_matched_tile": em["spacing_tile_at_384_um"],
                                                 "internal_full_frame": em["spacing_internal_full_frame_at_384_um"]},
                           "metadata_status": em["metadata_status"], "optics": ctx.spec["optics"],
                           "levels": [geo.level_summary(lv, ctx.p_int_um)
                                      for lv in sorted(ctx.levels.values(), key=lambda v: v.m)]}
    summary["conditions"] = cond_out

    prim_ids = [ALIASES[n] for n in ("C0", "C-S", "C-I", "C-SI")]
    if all(i in cond_out for i in prim_ids):
        named = {k: {n: stats[k][ALIASES[n]] for n in ALIASES if ALIASES[n] in stats[k]} for k in MODEL_KEYS}
        fam: Dict[str, Dict[str, Any]] = {"primary_y": {}, "secondary_g": {}}
        for fam_key, resp in (("primary_y", "y"), ("secondary_g", "g")):
            raw_p = {}
            for k in MODEL_KEYS:
                for cname, d in contrasts(named[k], resp).items():
                    t = signflip_exact(d)
                    fam[fam_key][f"{k}:{cname}"] = t
                    raw_p[f"{k}:{cname}"] = t["p_two_sided"]
            adj = holm(raw_p)
            for key, t in fam[fam_key].items():
                t["p_holm"] = adj[key]
                t["reject_at_0.05"] = bool(adj[key] <= ALPHA_TEST)
                t["model"], t["contrast"] = key.split(":")
        summary["contrasts"] = {"response_primary": plan["responses"]["primary_y"],
                                "family_primary_holm_m6": fam["primary_y"],
                                "family_secondary_direction_holm_m6": fam["secondary_g"]}
        unc: Dict[str, Any] = {}
        for src in EXTREMES:
            ci, csi = f"C-I@{src}", f"C-SI@{src}"
            for k in MODEL_KEYS:
                if ci in named[k] and csi in named[k]:
                    cc = contrasts(named[k], "y", ci, csi)
                    unc[f"{k}:{src}"] = {
                        "a": cal["bootstrap"][src]["a"], "b": cal["bootstrap"][src]["b"],
                        "d_I": signflip_exact(cc["d_I"]), "d_SxI": signflip_exact(cc["d_SxI"]),
                        "C-I_reproduction": cond_out[ALIASES[ci]]["models"][k]["reproduction"],
                        "C-SI_reproduction": cond_out[ALIASES[csi]]["models"][k]["reproduction"],
                        "note": "p unadjusted; outside the primary family"}
        summary["uncertainty_propagation"] = unc

        verdicts: Dict[str, Any] = {}
        for k in MODEL_KEYS:
            rep = {n: cond_out[ALIASES[n]]["models"][k]["reproduction"]["reproduces"] for n in ("C-S", "C-I", "C-SI")}
            rob = {}
            for n in ("C-I", "C-SI"):
                vals = [cond_out[ALIASES[f"{n}@{s}"]]["models"][k]["reproduction"]["reproduces"]
                        for s in EXTREMES if ALIASES[f"{n}@{s}"] in cond_out]
                rob[n] = {"point": rep[n], "extremes": vals,
                          "stable_across_calibration_interval": bool(len(vals) == 2 and all(v == rep[n] for v in vals))}
            verdicts[k] = {"display": MODEL_DISPLAY[k], "default_species": pc.CLASS_NAMES[DEFAULT_SPECIES[k]],
                           "reproduces": rep, "calibration_robustness": rob,
                           "reading": interpretation(rep, ctx.ext_level.m),
                           "factorial": {n: {"slide_balanced_accuracy":
                                             cond_out[ALIASES[n]]["models"][k]["classification"]["balanced_accuracy"],
                                             "mean_y": cond_out[ALIASES[n]]["models"][k]["mean_y"],
                                             "mean_g": cond_out[ALIASES[n]]["models"][k]["mean_g"],
                                             "n_pred_default": cond_out[ALIASES[n]]["models"][k]["reproduction"]["n_slides_pred_default"],
                                             "mean_frac_pixels_clipped": cond_out[ALIASES[n]]["clip"]["mean_frac_pixels_clipped"]}
                                         for n in ("C0", "C-S", "C-I", "C-SI")}}
        summary["reproduction_verdicts"] = verdicts

    surf: Dict[str, Any] = {}
    surf_levels = sorted((lv for lv in ctx.levels.values() if lv.role != geo.ROLE_SUPPLEMENTARY), key=lambda v: v.m)
    geo.assert_monotonic(surf_levels)
    for k in MODEL_KEYS:
        grid = []
        for lv in surf_levels:
            row = []
            for al in pc.ALPHAS:
                cid = cid_of(lv.T, al, "point", ctx.cal_tag)
                if cid not in cond_out:
                    row.append(None)
                    continue
                mk = cond_out[cid]["models"][k]
                row.append({"T": lv.T, "m": lv.m, "alpha": al,
                            "slide_balanced_accuracy": mk["classification"]["balanced_accuracy"],
                            "mean_y": mk["mean_y"], "mean_g": mk["mean_g"],
                            "n_pred_glabrata": mk["classification"]["n_pred_glabrata"],
                            "reproduces": mk.get("reproduction", {}).get("reproduces"),
                            "mean_frac_pixels_clipped": cond_out[cid]["clip"]["mean_frac_pixels_clipped"],
                            "frame_edge_share_of_tile_sides": cond_out[cid]["params"]["frame_edge_share_of_tile_sides"]})
            grid.append(row)
        surf[k] = {"rows_T": [lv.T for lv in surf_levels], "rows_m": [lv.m for lv in surf_levels],
                   "rows_role": [lv.role for lv in surf_levels], "rows_label": [lv.label for lv in surf_levels],
                   "order": "m ascending (T descending)", "cols_alpha": list(pc.ALPHAS), "grid": grid,
                   "supplementary": {cid_of(lv.T, 0.0, "point"): {"T": lv.T, "m": lv.m, "label": lv.label}
                                     for lv in ctx.levels.values() if lv.role == geo.ROLE_SUPPLEMENTARY}}
    summary["sensitivity_surface"] = surf

    cols = ["condition", "alias", "T", "m", "level_role", "alpha", "ab_source", "model", "slide_id", "label", "n_tiles", "y_true_logodds",
            "g_glabrata_logodds", "z_true_raw_logit", "p_albicans", "p_glabrata", "pred", "correct",
            "tile_frac_pred_glabrata", "clip_frac_mean", "outside_unit_frac_mean"]
    tmp = ctx.out / "part_a_slide_level.csv.tmp"
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, ctx.out / "part_a_slide_level.csv")
    write_json_atomic(summary, ctx.out / "part_a_summary.json")
    return summary


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Protocol Part A: forward simulation on the internal held-out cohort.")
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--internal-raw-root", type=Path, required=True)
    p.add_argument("--geometry-spec", type=Path, required=True)
    p.add_argument("--unit-check-dir", type=Path, required=True, help="Folder with unit_check_report.json")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--work-dir", type=Path, default=Path("/content/part_a_work"), help="Local disk for the raw cache")
    p.add_argument("--no-raw-cache", action="store_true")
    p.add_argument("--calibration-dir", type=Path, default=None, help="Default: <out>/calibration")
    p.add_argument("--external-manifest", type=Path, default=None)
    p.add_argument("--external-root", type=Path, default=None)
    p.add_argument("--pinned-scripts", type=Path, default=None)
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--calib-workers", type=int, default=3,
                   help="Calibration is I/O-bound on Colab; 3 workers on 2 vCPUs overlap reads and statistics")
    p.add_argument("--densenet-scripts", type=Path)
    p.add_argument("--densenet-ckpt", type=Path)
    p.add_argument("--hybrid-validation-scripts", type=Path)
    p.add_argument("--hybrid-source", type=Path)
    p.add_argument("--hybrid-ckpt", type=Path)
    p.add_argument("--hybrid-config", type=Path)
    p.add_argument("--allow-unsafe-densenet-load", action="store_true")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--chunk-tiles", type=int, default=DEFAULT_CHUNK_TILES,
                   help="Tiles per inference flush (host RAM ~0.6 MB each)")
    p.add_argument("--prep-workers", type=int, default=2)
    p.add_argument("--prefetch-fields", type=int, default=40,
                   help="Fields prepared ahead of the inference thread (40 fields x tiles per field x 0.6 MB)")
    p.add_argument("--inference-queue", type=int, default=2, help="Ready chunks waiting for the GPU")
    p.add_argument("--budget-hours", type=float, default=5.0)
    p.add_argument("--margin-minutes", type=float, default=20.0)
    p.add_argument("--session-start", type=float, default=None, help="Epoch seconds; default: env or VM boot")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--accept-conditional", action="store_true")
    p.add_argument("--plan-only", action="store_true", help="Write the frozen plan and stop")
    p.add_argument("--replan", action="store_true", help="Archive a differing frozen plan and replace it")
    p.add_argument("--summarize-only", action="store_true", help="Rebuild the summary from caches (no inference)")
    p.add_argument("--protocol-deviation", default=None,
                   help="Author text recorded in the frozen plan (the external-matched level replaces the legacy Protocol m = 2.165)")
    p.add_argument("--migration-manifest", type=Path, default=None,
                   help="migration/part_a_plan.json from migrate_part_ab.py (its SHA-256 enters the plan)")
    a = p.parse_args(argv)
    if not a.summarize_only:
        miss = [n for n in ("densenet_scripts", "densenet_ckpt", "hybrid_validation_scripts", "hybrid_source",
                            "hybrid_ckpt", "hybrid_config") if getattr(a, n) is None]
        if miss:
            p.error(f"model arguments missing: {miss}")
    # Batch-partition invariance: shards and flushes start on multiples of the batch size, so every
    # 32-tile batch is the same as in an uninterrupted run (and as in the unsharded implementation).
    if a.chunk_tiles % a.batch_size or SHARD_TILES % a.chunk_tiles:
        p.error(f"--chunk-tiles must divide {SHARD_TILES} and be a multiple of --batch-size")
    return a


def main(argv: Optional[Sequence[str]] = None) -> None:
    a = parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    ctx = Context(a)
    bad = [lv.T for lv in ctx.levels.values() if a.chunk_tiles % lv.n_tiles or SHARD_TILES % lv.n_tiles]
    if bad:
        raise RuntimeError(f"--chunk-tiles {a.chunk_tiles} / SHARD_TILES {SHARD_TILES} not whole fields at T = {bad}")
    gate = unit_check_gate(a)
    ctx.border_rule = gate["border_rule"]
    ctx.cv2_version = str(gate["cv2_version"])
    if gate.get("T_external_matched") not in (None, ctx.T_ext):
        raise RuntimeError(f"unit checks used T = {gate['T_external_matched']}, geometry spec says {ctx.T_ext}")

    if a.summarize_only:
        plan_rec = json.loads((a.out / "part_a_plan_frozen.json").read_text())
        if plan_rec["plan"].get("geometry_digest") != ctx.geometry_digest:
            raise RuntimeError("frozen plan geometry digest differs from geometry_spec.json")
        ctx.cal_sha = plan_rec["plan"]["calibration_sha256"]
        ctx.cal_tag = plan_rec["plan"]["calibration_digest"][:8]
        cpath = (Path(a.calibration_dir) if a.calibration_dir else a.out / "calibration") / "calibration.json"
        if cpath.exists() and pc.sha256_file(cpath) == ctx.cal_sha:
            ctx.cal = json.loads(cpath.read_text())
        conds, ctx.aliases = build_conditions(list(ctx.levels.values()), ctx.cal_tag)
        s = summarize(ctx, conds, plan_rec, {"gate": gate, "mode": "summarize-only"})
        log(f"summary rebuilt ({'complete' if s['complete'] else 'missing ' + str(s['missing_conditions'])})")
        return

    obtain_calibration(ctx, ctx.border_rule)
    conds, ctx.aliases = build_conditions(list(ctx.levels.values()), ctx.cal_tag)
    env = load_models(ctx)
    plan_rec = freeze_plan(ctx, build_plan(ctx, conds, gate))
    if a.plan_only:
        log("--plan-only: stopping before held-out inference")
        return

    ctx.lock.acquire()
    try:
        pending = [c for c in conds
                   if not condition_complete(ctx, ctx.cond_root / c.cid, condition_digest(ctx, condition_params(ctx, c)))]
        log(f"{len(conds) - len(pending)}/{len(conds)} conditions already complete and verified; "
            f"budget left {ctx.budget.remaining() / 60:.0f} min")
        raw_info: Dict[str, Any] = {"cache": "not needed (all conditions complete)"}
        rule_now: Dict[str, Any] = {"note": "not re-identified (no inference this session)"}
        if pending:
            ctx.budget.require(ctx.timings.estimate("partA_raw_cache", 300.0), "held-out raw cache")
            t0 = time.time()
            ctx.raw = HeldoutRaw(ctx.bases, a.internal_raw_root, a.work_dir, not a.no_raw_cache)
            raw_info = ctx.raw.build()
            ctx.timings.record("partA_raw_cache", time.time() - t0)
            x0, _ = pc.normalize_internal(ctx.raw.get(0))
            rule_now = geo.identify_cv2_border_rule(x0, tol=vuc.ENGINE_TOL)
            if not rule_now["PASS"] or rule_now["selected"] != ctx.border_rule:
                raise RuntimeError(f"border rule re-identification {rule_now} disagrees with the unit checks "
                                   f"({ctx.border_rule})")
        run_condition(ctx, conds[0])
        gate_c0 = c0_gate(ctx)
        for c in conds[1:]:
            run_condition(ctx, c)
    finally:
        ctx.lock.release()
        cs.sync_filesystems()

    import torch

    extra = {
        "gate": gate, "c0_regression_gate": {k: v["PASS"] if isinstance(v, dict) else v for k, v in gate_c0.items()},
        "border_rule_runtime": rule_now, "raw_cache": raw_info,
        "environment": {"torch_config": env, "opencv": geo.opencv_provenance(),
                        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                        "python": sys.version.split()[0]},
        "runtime_seconds": round(time.time() - t_start, 1), "geometry_spec_sha256": ctx.spec_sha,
        "args": {k: str(v) for k, v in vars(a).items()},
    }
    s = summarize(ctx, conds, plan_rec, extra)
    log(f"wrote {a.out / 'part_a_summary.json'} ({'complete' if s['complete'] else 'INCOMPLETE'})")
    for k, v in s.get("reproduction_verdicts", {}).items():
        log(f"  {v['display']}: reproduces {v['reproduces']}")
    for key, t in s.get("contrasts", {}).get("family_primary_holm_m6", {}).items():
        log(f"  {key}: mean d = {t['mean']:+.3f}, p = {t['p_two_sided']:.4f}, Holm p = {t['p_holm']:.4f}")


if __name__ == "__main__":
    try:
        main()
    except cs.BudgetExhausted as exc:
        cs.sync_filesystems()
        print(f"PAUSED (session budget): {exc}", file=sys.stderr)
        sys.exit(cs.EXIT_BUDGET)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
