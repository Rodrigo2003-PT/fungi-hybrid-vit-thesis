#!/usr/bin/env python3
"""
run_part_b_and_export.py

Instrument calibration versus per-image standardization on
the external cohort, frozen models, inference only

Pipelines (24 slides, 1,253 images, both frozen checkpoints)
------------------------------------------------------------
  E-locked              pinned locked-pixel operator          (re-run, gated on the archive)
  E-z                   pinned photometric mean/SD operator   (re-run, gated on the archive)
  E-cal                 /16383 -> (x - b) / a -> pinned Lanczos-4 resize -> clip [0, 1]
  E-cal@extreme_low     the same at the joint bootstrap replicate at the 2.5th percentile
  E-cal@extreme_high    the same at the 97.5th percentile
Every array comes from photometric_calibration.external_condition (pinned
preprocess_external.py imported by SHA-256), is standardized with the
checkpoint's own mu and sigma (pc.standardize, identical to ExternalNPYDataset),
and is classified in FP32 (verify_unit_checks.logits_for).

Reused, not rewritten
---------------------
  model loading            verify_unit_checks.load_models (load_frozen_model, load_frozen_hybrid)
  slide aggregation, CSVs  pinned evaluate_external_generalization (aggregate_to_slides, schemas)
  archive gate             ext_common.reproduction_check
  statistics               ext_stats (binary_metrics incl. Brier and ECE, marginal_bootstrap,
                           clopper_pearson, mcnemar_exact, holm_adjust, stratified_bootstrap_indices),
                           compare_external_models.paired_contrast (paired class-stratified slide
                           bootstrap with identical indices, exact McNemar, DeLong cross-check)
New here: the exact sign-flip test over all 2^n patterns (meet-in-the-middle,
same zero-removal rule as compare_external_pipelines.exact_sign_flip_pvalue,
optional cross-check against that function), the AUROC interpretation rule, 
the recovery ratio, and the exports.

Multiplicity (frozen in part_b_plan_frozen.json before inference)
-----------------------------------------------------------------
  pipeline family   {E-cal - E-locked, E-z - E-cal, E-z - E-locked} x {McNemar, sign-flip}
                    x 2 models; --holm-scheme protocol_m12 (default, E-cal joins the
                    pipeline family) or separate_m4_m8 (published m = 4 family kept; new m = 8).
  model family      DenseNet-121 versus ViT+CCT+LSA_43 McNemar on E-locked, E-cal, E-z (m = 3).
  reproduction      the published m = 4 and m = 2 families are recomputed and reported next to
                    the printed values (0.37, 0.40, 0.38; 0.68, 0.53).

Exports
-------
  external_slides_extended.csv / .tex     24-slide probabilities, 3 pipelines x 2 models, E-alb_B marked
  figs/part_a_sensitivity_surface.{pdf,png}   6 x 5 slide-BA surface per model from part_a_summary.json
                                              (rows: physical scale m = 1200 / T ascending; the
                                              external-matched row, m = 1.0677, is outlined)
  experimental_summary_for_manuscript.json    

Author: Rodrigo Sá
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import hashlib
import json
import math
import queue
import shutil
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import colab_state as cs                            # noqa: E402
import geometry_harmonization as geo                 # noqa: E402
import photometric_calibration as pc                 # noqa: E402
import verify_unit_checks as vuc                     # noqa: E402
import run_part_a_internal_simulation as pa          # noqa: E402

SCHEMA = "part_b_and_export/1.0"
N_IMAGES, N_SLIDES = 1253, 24
MODELS = pa.MODEL_KEYS                               # ("hybrid43", "densenet121")
DISPLAY = pa.MODEL_DISPLAY
FAMILY_NAME = {"hybrid43": "hybrid43", "densenet121": "densenet121"}
WITHIN_SESSION_DATE = "20260507"                     # E-alb_B: the only C. albicans slide of this date
AUROC_PAIR_GAP, AUROC_PAIR_SHARE = 1e-3, 0.5         # interpretation rule (144 cross-class pairs)
ZERO_DIFF = 1e-12                                    # compare_external_pipelines.exact_sign_flip_pvalue

PIPELINES = {   # name -> (pc mode, calibration source, preprocessing_mode label)
    "E-locked": ("E-locked", None, "locked_pixel_pipeline"),
    "E-z": ("E-z", None, "photometric_mean_std_matched_pipeline"),
    "E-cal": ("E-cal", "point", "instrument_calibrated_pipeline"),
    "E-cal@extreme_low": ("E-cal", "extreme_low", "instrument_calibrated_pipeline_extreme_low"),
    "E-cal@extreme_high": ("E-cal", "extreme_high", "instrument_calibrated_pipeline_extreme_high"),
}
PRIMARY_PIPES = ("E-locked", "E-cal", "E-z")
PIPE_CONTRASTS = (("E-cal", "E-locked"), ("E-z", "E-cal"), ("E-z", "E-locked"))   # (B, A): delta = B - A

# tab:external_locked (4 decimals). Fresh E-locked and E-z must reproduce them.
PUBLISHED = {
    ("hybrid43", "E-locked"): {"balanced_accuracy": 0.5417, "matthews_corrcoef": 0.2085, "auroc": 0.7153,
                               "recall_albicans": 1.0, "recall_glabrata": 0.0833, "brier": 0.4298},
    ("densenet121", "E-locked"): {"balanced_accuracy": 0.4167, "matthews_corrcoef": -0.3015, "auroc": 0.0833,
                                  "recall_albicans": 0.0, "recall_glabrata": 0.8333, "brier": 0.5545},
    ("hybrid43", "E-z"): {"balanced_accuracy": 0.8333, "matthews_corrcoef": 0.6761, "auroc": 0.9306,
                          "recall_albicans": 0.75, "recall_glabrata": 0.9167, "brier": 0.1182},
    ("densenet121", "E-z"): {"balanced_accuracy": 0.625, "matthews_corrcoef": 0.378, "auroc": 1.0,
                             "recall_albicans": 1.0, "recall_glabrata": 0.25, "brier": 0.2448},
}
PUBLISHED_HOLM = {"model_family_m2": {"E-locked": 0.68, "E-z": 0.53},
                  "pipeline_family_m4_printed": [0.37, 0.40, 0.38]}

log, write_json, save_npz, digest_of = pa.log, pa.write_json_atomic, pa.save_npz_atomic, pa.digest_of


# -----------------------------------------------------------------------------
# Statistics added in this script (pure NumPy)
# -----------------------------------------------------------------------------

def _all_sign_sums(v: np.ndarray) -> np.ndarray:
    k = v.size
    E = ((np.arange(2 ** k, dtype=np.int64)[:, None] >> np.arange(k, dtype=np.int64)) & 1) * 2 - 1
    return E.astype(np.float64) @ v


def signflip_exact(values: Sequence[float]) -> Dict[str, Any]:
    """
    Exact two-sided sign-flip test of the paired mean, all 2^n patterns, meet-in-the-middle
    (two halves of 2^(n/2) sums, sorted, searchsorted). Zero differences (|d| <= 1e-12) are removed
    as in the legacy function. Tie tolerance 1e-12 x max(1, |mean|) on the mean scale.
    """
    d_all = np.asarray(values, dtype=np.float64)
    v = d_all[np.abs(d_all) > ZERO_DIFF]
    n = int(v.size)
    out: Dict[str, Any] = {"n_slides": int(d_all.size), "n_nonzero": n, "mean_all": float(d_all.mean()),
                           "values": d_all.tolist()}
    if n == 0:
        return {**out, "p_two_sided": None, "note": "all differences zero"}
    obs = abs(float(v.mean()))
    thr = n * (obs - 1e-12 * max(1.0, obs))
    A = _all_sign_sums(v[: n // 2])
    B = np.sort(_all_sign_sums(v[n // 2:]))
    if thr <= 0:
        count = A.size * B.size
    else:
        ge = B.size - np.searchsorted(B, thr - A, side="left")
        le = np.searchsorted(B, -thr - A, side="right")
        count = int((ge + le).sum())
    return {**out, "mean_nonzero": float(v.mean()), "n_patterns": 2 ** n, "count_ge": int(count),
            "p_two_sided": count / 2 ** n, "p_floor": 2.0 / 2 ** n}


def auroc_rule(y: np.ndarray, p: np.ndarray) -> Dict[str, Any]:
    """main.tex: interpret a slide AUROC only if >= half of the cross-class pairs differ by >= 1e-3."""
    from sklearn.metrics import roc_auc_score

    pos, neg = p[y == 1], p[y == 0]
    gaps = np.abs(pos[:, None] - neg[None, :])
    share = float((gaps >= AUROC_PAIR_GAP).mean())
    return {"n_cross_class_pairs": int(gaps.size), "n_pairs_gap_ge_1e-3": int((gaps >= AUROC_PAIR_GAP).sum()),
            "share_pairs_gap_ge_1e-3": share, "interpretable": bool(share >= AUROC_PAIR_SHARE),
            "auroc": float(roc_auc_score(y, p)), "auroc_from_4dp_probabilities": float(roc_auc_score(y, np.round(p, 4)))}


def _ba(y: np.ndarray, pred: np.ndarray) -> float:
    return 0.5 * (float((pred[y == 0] == 0).mean()) + float((pred[y == 1] == 1).mean()))


def recovery_ratio(y: np.ndarray, pl: np.ndarray, pc_: np.ndarray, pz: np.ndarray, *, B: int, seed: int,
                   ext_stats: Any) -> Dict[str, Any]:
    """rho = (BA_cal - BA_locked) / (BA_z - BA_locked); paired class-stratified slide bootstrap."""
    def rho(idx: np.ndarray) -> float:
        yl = y[idx]
        den = _ba(yl, pz[idx]) - _ba(yl, pl[idx])
        return float("nan") if abs(den) < 1e-12 else (_ba(yl, pc_[idx]) - _ba(yl, pl[idx])) / den
    point = rho(np.arange(y.size))
    reps = np.array([rho(idx) for idx in ext_stats.stratified_bootstrap_indices(y, B, seed)])
    lo, hi, nv = ext_stats.percentile_interval(reps)
    return {"point": point, "ci_lower": lo, "ci_upper": hi, "n_valid_bootstrap": nv,
            "n_undefined_replicates": int(np.isnan(reps).sum()),
            "definition": "(BA_E-cal - BA_E-locked) / (BA_E-z - BA_E-locked)", "descriptive": True}


# -----------------------------------------------------------------------------
# Context
# -----------------------------------------------------------------------------

class Ctx:
    def __init__(self, a: argparse.Namespace) -> None:
        self.a, self.out = a, Path(a.out)
        self.cond_root = self.out / "conditions"
        self.models: Dict[str, Dict[str, Any]] = {}
        self.geometry_digest = ""
        self.model_ids: Dict[str, List[Any]] = {}       # {model: [sha256, mu, sigma]} from the models or the frozen plan
        self.store = cs.UnitStore(self.out / "shards")
        self.memo = cs.UnitStore(self.out / "memo")
        self.budget = cs.SessionBudget(a.budget_hours, a.margin_minutes, a.session_start)
        self.timings = cs.Timings(self.out / "timings.json")
        self.lock = cs.RunLock(self.out / "RUN_LOCK.json")

    def setup_imports(self) -> None:
        """After vuc.load_models put the DenseNet validation folder on sys.path."""
        sys.path.insert(0, str(self.a.densenet_scripts))
        sys.path.insert(0, str(self.a.pinned_scripts))
        import ext_common, ext_stats, compare_external_models  # noqa: E401
        self.ec, self.es, self.cm = ext_common, ext_stats, compare_external_models
        pins = ext_common.PINNED_SHA256
        legacy = {}
        for name in ("evaluate_external_generalization.py", "compare_external_pipelines.py"):
            path = Path(self.a.pinned_scripts) / name
            got, exp = pc.sha256_file(path), pins[f"scripts/legacy/{name}"]
            if got != exp:
                raise RuntimeError(f"{path} sha256 {got} != pin {exp}")
            legacy[name] = ext_common.import_module_from_path(f"_partb_legacy_{path.stem}", path)
        self.lg = legacy["evaluate_external_generalization.py"]
        self.lcp = legacy["compare_external_pipelines.py"]
        self.pinned = pc.load_pinned_preprocess(Path(self.a.pinned_scripts))

    def load_manifest(self) -> None:
        rows = self.lg.read_manifest(Path(self.a.manifest))
        self.manifest_warnings = self.lg.validate_manifest_cohort(rows, strict=True)
        self.rows = sorted(rows, key=lambda r: r["relative_path"])
        self.slides = sorted({r["slide_id"] for r in self.rows})
        if len(self.rows) != N_IMAGES or len(self.slides) != N_SLIDES:
            raise RuntimeError(f"manifest has {len(self.rows)} images / {len(self.slides)} slides")
        self.rel = [r["relative_path"] for r in self.rows]
        lab = {r["slide_id"]: int(float(r["label"])) for r in self.rows}
        alb_b = [s for s in self.slides if lab[s] == 0 and self.ec.slide_date(s) == WITHIN_SESSION_DATE]
        if len(alb_b) != 1:
            raise RuntimeError(f"expected one C. albicans slide dated {WITHIN_SESSION_DATE}, found {alb_b}")
        self.alb_b = alb_b[0]

    def path(self, i: int) -> Path:
        r = self.rows[i]
        return Path(self.a.external_root) / r["relative_path"] if self.a.external_root else Path(r["image_path"])


def display_id(slide_id: str) -> str:
    """'20260507_alb_B' -> 'E-alb_B' (main.tex convention); the date prefix is dropped."""
    parts = slide_id.split("_")
    return "E-" + "_".join(parts[1:]) if len(parts) >= 3 else slide_id


# -----------------------------------------------------------------------------
# Inference with caching
# -----------------------------------------------------------------------------

def pipe_params(ctx: Ctx, name: str) -> Dict[str, Any]:
    mode, src, label = PIPELINES[name]
    p: Dict[str, Any] = {"pipeline": name, "mode": mode, "ab_source": src, "preprocessing_mode": label}
    if src is not None:
        p["a"], p["b"] = pa.ab_for(ctx.cal, src)
        p["calibration_digest"] = pc.calibration_digest(ctx.cal)
    return p


SHARD_IMAGES = 256      # multiple of --chunk (128) and --batch-size (32): batch partition unchanged by sharding
GATE_IMAGES = (0, 417, 835, 1252)


def pipe_digest_from(params: Dict[str, Any], images_digest: str, model_ids: Dict[str, List[Any]], bs: int) -> str:
    """Pure form. E-locked and E-z parameters carry no geometry or calibration field, so their digests (and
    DONE.json files) are unchanged by the optical refactor; E-cal parameters carry (a, b) and the calibration
    digest. SCHEMA is intentionally unchanged."""
    return digest_of({"schema": SCHEMA, "p": params, "images": images_digest,
                      "pinned": pc.PINNED_PREPROCESS_SHA256, "models": dict(sorted(model_ids.items())), "bs": bs})


def pipe_digest(ctx: Ctx, name: str) -> str:
    """Identical to the digest of the unsharded implementation (same keys, same values)."""
    return pipe_digest_from(pipe_params(ctx, name), digest_of(ctx.rel), ctx.model_ids, ctx.a.batch_size)


def pipeline_complete(ctx: Ctx, name: str) -> bool:
    cdir = ctx.cond_root / name
    try:
        rec = json.loads((cdir / "DONE.json").read_text())
    except (OSError, ValueError):
        return False
    return rec.get("digest") == pipe_digest(ctx, name) and cs.verify_files(cdir, rec)


def _write_pipeline(ctx: Ctx, name: str, logits: Dict[str, np.ndarray], clip: Dict[str, np.ndarray], seconds: float) -> None:
    for k in MODELS:
        if logits[k].shape != (N_IMAGES, 2) or not np.isfinite(logits[k]).all():
            raise RuntimeError(f"{name}/{k}: logits {logits[k].shape} incomplete or non-finite")
    rel = np.array(ctx.rel)
    payloads = {f"{k}.npz": cs.npz_bytes(logits=logits[k], relative_path=rel) for k in MODELS}
    payloads["clip.npz"] = cs.npz_bytes(relative_path=rel, **clip)
    cdir = ctx.cond_root / name
    cdir.mkdir(parents=True, exist_ok=True)
    cs.write_verified_files(cdir, payloads, "DONE.json",
                            {"digest": pipe_digest(ctx, name), "params": pipe_params(ctx, name),
                             "seconds": round(seconds, 1)})
    if not pipeline_complete(ctx, name):
        raise RuntimeError(f"{name}: written pipeline failed verification")
    log(f"{name}: verified; mean clipped fraction {np.nanmean(clip['frac_clipped']):.2e}")


# ----- single-read preprocessing: one TIFF decode feeds every pipeline -------------------------------------

class Stream:
    """One input stream: a pipeline, and for E-z one (mu, sigma) group. Mirrors the grouping of the
    unsharded implementation (E-z groups models by their frozen statistics; clip statistics from group 0)."""

    def __init__(self, key: str, pipeline: str, mode: str, a: Optional[float], b: Optional[float],
                 mu: Optional[float], sigma: Optional[float], models: Tuple[str, ...], records_clip: bool) -> None:
        self.key, self.pipeline, self.mode, self.a, self.b = key, pipeline, mode, a, b
        self.mu, self.sigma, self.models, self.records_clip = mu, sigma, models, records_clip


def build_streams(ctx: Ctx, names: Optional[Sequence[str]] = None) -> List[Stream]:
    """Streams for the given pipelines only (default: all), so cached pipelines are never recomputed."""
    out: List[Stream] = []
    for name in (PIPELINES if names is None else names):
        p = pipe_params(ctx, name)
        if p["mode"] == "E-z":
            groups: Dict[Tuple[float, float], List[str]] = {}
            for k in MODELS:
                groups.setdefault((ctx.model_ids[k][1], ctx.model_ids[k][2]), []).append(k)
            for gi, ((mu, sg), mem) in enumerate(groups.items()):
                out.append(Stream(f"{name}#g{gi}", name, "E-z", None, None, mu, sg, tuple(mem), gi == 0))
        else:
            if p["mode"] == "E-cal" and not (p["a"] > 0):
                raise ValueError(f"{name}: a = {p['a']}")
            out.append(Stream(name, name, p["mode"], p.get("a"), p.get("b"), None, None, MODELS, True))
    return out


def prep_image_set(ctx: Ctx, i: int, streams: Sequence[Stream]) -> Tuple[int, Dict[str, Tuple[np.ndarray, float, float, float]]]:
    """
    The operator sequence of photometric_calibration.external_condition, once per stream, after a single
    read and normalization. E-locked and E-z share one resize of the same normalized image (pc applies the
    identical resize to the identical input in both). Every consumer gets a private copy, so no pinned
    function can alter another stream's input. Bitwise identity with external_condition is gated at run time.
    """
    x, _ = pc.read_external_normalized(ctx.path(i), ctx.pinned)
    base: Optional[np.ndarray] = None
    out: Dict[str, Tuple[np.ndarray, float, float, float]] = {}
    for s in streams:
        if s.mode == "E-cal":
            r = ctx.pinned.resize_to_square(pc.inverse_map(x.copy(), s.a, s.b), geo.D, post_resize_clip=False)
        else:
            if base is None:
                base = ctx.pinned.resize_to_square(x.copy(), geo.D, post_resize_clip=False)
            r = base.copy()
        y, clip = pc.apply_external_clip(r, ctx.pinned)
        if s.mode == "E-z":
            y, _meta = ctx.pinned.photometric_mean_std_match_to_checkpoint(y, [float(s.mu)], [float(s.sigma)],
                                                                           min_source_std=1e-6,
                                                                           post_harmonization_clip=True)
        y = y.astype(np.float32, copy=False)
        out[s.key] = (y, float(clip["frac_clipped"]), float(clip["frac_outside_unit"]),
                      float(((y <= 0.0) | (y >= 1.0)).mean()))
    return i, out


def prep_gate(ctx: Ctx, streams: Sequence[Stream]) -> Dict[str, Any]:
    """Bitwise comparison of the single-read path against pc.external_condition on GATE_IMAGES."""
    rows, ok = [], True
    for i in GATE_IMAGES:
        _, got = prep_image_set(ctx, i, streams)
        for s in streams:
            ref, info = pc.external_condition(ctx.path(i), s.mode, ctx.pinned, a=s.a, b=s.b, mu=s.mu, sigma=s.sigma)
            y = got[s.key][0]
            same = (ref.dtype == y.dtype == np.float32 and ref.shape == y.shape
                    and np.array_equal(np.ascontiguousarray(ref).view(np.uint32), np.ascontiguousarray(y).view(np.uint32))
                    and info["clip"]["frac_clipped"] == got[s.key][1]
                    and info["clip"]["frac_outside_unit"] == got[s.key][2])
            ok &= bool(same)
            rows.append({"image": i, "stream": s.key, "bitwise_equal": bool(same),
                         "max_abs_diff": float(np.max(np.abs(ref.astype(np.float64) - y))) if ref.shape == y.shape else None})
    return {"PASS": bool(ok), "images": list(GATE_IMAGES), "checks": rows}


class _StreamInference:
    """Single thread owning CUDA; per stream, chunks arrive in order, so each stream's 32-image batches are the
    batches of the unsharded run."""

    def __init__(self, ctx: Ctx, logits: Dict[Tuple[str, str], np.ndarray], queue_size: int) -> None:
        self.ctx, self.logits, self.err = ctx, logits, None
        self.q: "queue.Queue" = queue.Queue(maxsize=max(1, queue_size))
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def _run(self) -> None:
        while True:
            item = self.q.get()
            if item is None:
                return
            if self.err is not None:
                continue
            s, start, arrs = item
            try:
                for k in s.models:
                    m = self.ctx.models[k]
                    self.logits[(s.pipeline, k)][start:start + len(arrs)] = vuc.logits_for(
                        m["model"], arrs, m["mu"], m["sigma"], self.ctx.a.batch_size)
            except BaseException as exc:
                self.err = exc

    def put(self, s: Stream, start: int, arrs: List[np.ndarray]) -> None:
        if self.err is not None:
            raise self.err
        self.q.put((s, start, arrs))

    def close(self) -> None:
        self.q.put(None)
        self.th.join()
        if self.err is not None:
            raise self.err


def run_infer_shard(ctx: Ctx, streams: Sequence[Stream], i0: int, i1: int) -> Dict[str, np.ndarray]:
    n = i1 - i0
    pipes = list(dict.fromkeys(s.pipeline for s in streams))
    logits = {(p, k): np.full((n, 2), np.nan, dtype=np.float32) for p in pipes for k in MODELS}
    clip = {(p, f): np.full(n, np.nan) for p in pipes for f in ("frac_clipped", "frac_outside_unit", "frac_at_bounds_final")}
    worker = _StreamInference(ctx, logits, ctx.a.inference_queue)
    bufs: Dict[str, List[np.ndarray]] = {s.key: [] for s in streams}
    starts: Dict[str, int] = {s.key: 0 for s in streams}
    try:
        with ThreadPoolExecutor(max_workers=max(1, ctx.a.prep_workers)) as ex:
            q: deque = deque(ex.submit(prep_image_set, ctx, i, streams) for i in range(i0, min(i1, i0 + ctx.a.prefetch)))
            nxt = i0 + len(q)
            while q:
                i, got = q.popleft().result()
                if nxt < i1:
                    q.append(ex.submit(prep_image_set, ctx, nxt, streams))
                    nxt += 1
                for s in streams:
                    y, fc, fo, fb = got[s.key]
                    if i - i0 != starts[s.key] + len(bufs[s.key]):
                        raise RuntimeError("image order broken")
                    if s.records_clip:
                        clip[(s.pipeline, "frac_clipped")][i - i0] = fc
                        clip[(s.pipeline, "frac_outside_unit")][i - i0] = fo
                        clip[(s.pipeline, "frac_at_bounds_final")][i - i0] = fb
                    bufs[s.key].append(y)
                    if len(bufs[s.key]) >= ctx.a.chunk:
                        worker.put(s, starts[s.key], bufs[s.key])
                        starts[s.key] += len(bufs[s.key])
                        bufs[s.key] = []
            for s in streams:
                if bufs[s.key]:
                    worker.put(s, starts[s.key], bufs[s.key])
                    starts[s.key] += len(bufs[s.key])
                    bufs[s.key] = []
    finally:
        worker.close()
    for s in streams:
        if starts[s.key] != n:
            raise RuntimeError(f"{s.key}: inferred {starts[s.key]} of {n} images in shard [{i0}, {i1})")
    out = {f"logits__{p}__{k}": v for (p, k), v in logits.items()}
    out.update({f"clip__{p}__{f}": v for (p, f), v in clip.items()})
    return out


def run_inference(ctx: Ctx) -> Dict[str, Any]:
    todo = [n for n in PIPELINES if not pipeline_complete(ctx, n)]
    if not todo:
        log("all pipelines complete and verified")
        return {"status": "cached"}
    streams = build_streams(ctx, todo)
    gate = prep_gate(ctx, streams) if ctx.a.prep == "single-read" else {"PASS": False, "note": "--prep per-pipeline"}
    write_json(gate, ctx.out / "prep_equivalence_gate.json")
    if not gate["PASS"]:
        if ctx.a.prep == "single-read" and not ctx.a.fallback_per_pipeline:
            raise RuntimeError("single-read preprocessing is not bitwise identical to external_condition here "
                               "(see prep_equivalence_gate.json); rerun with --prep per-pipeline")
        log("using the per-pipeline path (one read per pipeline)")
        for n in todo:
            run_pipeline(ctx, n)
        return {"status": "per-pipeline", "gate": gate}
    log(f"single-read gate PASS ({len(streams)} streams x {len(GATE_IMAGES)} images bitwise identical)")
    sdg = digest_of({"pipelines": {n: pipe_digest(ctx, n) for n in todo}, "shard_images": SHARD_IMAGES,
                     "chunk": ctx.a.chunk, "streams": [[s.key, s.models, s.records_clip] for s in streams]})
    parts, seconds = [], 0.0
    for si, i0 in enumerate(range(0, N_IMAGES, SHARD_IMAGES)):
        i1 = min(N_IMAGES, i0 + SHARD_IMAGES)
        unit = f"infer/shard_{si:02d}"
        got = ctx.store.load_npz(unit, sdg)
        if got is None:
            ctx.budget.require(ctx.timings.estimate("partB_shard", 600.0) * (i1 - i0) / SHARD_IMAGES, unit)
            t0 = time.time()
            arrays = run_infer_shard(ctx, streams, i0, i1)
            dt = time.time() - t0
            ctx.store.commit_npz(unit, sdg, arrays, meta={"seconds": round(dt, 1), "images": [i0, i1]})
            ctx.timings.record("partB_shard", dt * SHARD_IMAGES / (i1 - i0))
            ctx.lock.beat()
            seconds += dt
            log(f"  {unit}: {i1 - i0} images x {len(streams)} streams in {dt:.0f}s; "
                f"{ctx.budget.remaining() / 60:.0f} min left")
            got = ctx.store.load_npz(unit, sdg)
            if got is None:
                raise RuntimeError(f"{unit}: committed shard failed verification")
        parts.append(got)
    for name in todo:
        cat = lambda key: np.concatenate([q[key] for q in parts])  # noqa: E731
        _write_pipeline(ctx, name, {k: cat(f"logits__{name}__{k}") for k in MODELS},
                        {f: cat(f"clip__{name}__{f}") for f in ("frac_clipped", "frac_outside_unit", "frac_at_bounds_final")},
                        seconds)
    return {"status": "single-read", "gate": gate}


def run_pipeline(ctx: Ctx, name: str) -> None:
    """Per-pipeline fallback (one read per pipeline), the original implementation with verified output."""
    if pipeline_complete(ctx, name):
        log(f"{name}: cached (verified)")
        return
    p = pipe_params(ctx, name)
    ctx.budget.require(ctx.timings.estimate("partB_pipeline", 1200.0), f"pipeline {name}")
    groups: Dict[Tuple[float, float], List[str]] = {}
    for k in MODELS:
        key = (ctx.models[k]["mu"], ctx.models[k]["sigma"]) if p["mode"] == "E-z" else (0.0, 0.0)
        groups.setdefault(key, []).append(k)
    clip = {k: np.full(N_IMAGES, np.nan) for k in ("frac_clipped", "frac_outside_unit", "frac_at_bounds_final")}
    logits = {k: np.full((N_IMAGES, 2), np.nan, dtype=np.float32) for k in MODELS}
    t0 = time.time()
    for gi, ((mu, sg), members) in enumerate(groups.items()):
        kw = {"a": p.get("a"), "b": p.get("b")}
        if p["mode"] == "E-z":
            kw.update(mu=mu, sigma=sg)

        def prep(i: int):
            y, info = pc.external_condition(ctx.path(i), p["mode"], ctx.pinned, **kw)
            return i, y, info

        buf: List[np.ndarray] = []
        start = 0

        def flush() -> None:
            nonlocal buf, start
            for k in members:
                m = ctx.models[k]
                logits[k][start:start + len(buf)] = vuc.logits_for(m["model"], buf, m["mu"], m["sigma"], ctx.a.batch_size)
            start += len(buf)
            buf = []

        with ThreadPoolExecutor(max_workers=max(1, ctx.a.prep_workers)) as ex:
            q: deque = deque(ex.submit(prep, i) for i in range(min(ctx.a.prefetch, N_IMAGES)))
            nxt = len(q)
            while q:
                i, y, info = q.popleft().result()
                if nxt < N_IMAGES:
                    q.append(ex.submit(prep, nxt))
                    nxt += 1
                if i != start + len(buf):
                    raise RuntimeError("image order broken")
                if gi == 0:
                    clip["frac_clipped"][i] = info["clip"]["frac_clipped"]
                    clip["frac_outside_unit"][i] = info["clip"]["frac_outside_unit"]
                    clip["frac_at_bounds_final"][i] = float(((y <= 0.0) | (y >= 1.0)).mean())
                buf.append(y)
                if len(buf) >= ctx.a.chunk:
                    flush()
            flush()
    ctx.timings.record("partB_pipeline", time.time() - t0)
    _write_pipeline(ctx, name, logits, clip, time.time() - t0)
    ctx.lock.beat()


# ----- statistics acceleration and memoization -------------------------------------------------------------

class _CachedBinaryMetrics:
    """
    ext_stats.binary_metrics is a pure function of its converted inputs. All bootstrap procedures draw the
    same class-stratified indices (same seed), so the metrics of one arm on one replicate are recomputed
    across its marginal bootstrap and every paired contrast it enters. This cache returns the stored result
    for byte-identical inputs; values are therefore identical to recomputation, and the work drops from about
    240,000 to about 60,000 evaluations.
    """

    def __init__(self, fn: Any) -> None:
        self.fn, self.cache, self.hits, self.misses = fn, {}, 0, 0

    def __call__(self, y_true: Any, y_pred: Any, p_glabrata: Any, confidence: Any = None) -> Dict[str, Any]:
        y = np.asarray(y_true, dtype=int)
        yp = np.asarray(y_pred, dtype=int)
        p = np.asarray(p_glabrata, dtype=float)
        c = None if confidence is None else np.asarray(confidence, dtype=float)
        h = hashlib.sha256()
        for arr in (y, yp, p) + (() if c is None else (c,)):
            h.update(str(arr.shape).encode())
            h.update(np.ascontiguousarray(arr).tobytes())
        h.update(b"conf-none" if c is None else b"conf")
        key = h.digest()
        if key not in self.cache:
            self.misses += 1
            self.cache[key] = self.fn(y_true, y_pred, p_glabrata, confidence)
        else:
            self.hits += 1
        return dict(self.cache[key])


def install_metric_cache(ctx: Ctx) -> _CachedBinaryMetrics:
    original = getattr(ctx.es.binary_metrics, "fn", ctx.es.binary_metrics)
    cached = _CachedBinaryMetrics(original)
    ctx.es.binary_metrics = cached        # used by ext_stats.marginal_bootstrap
    ctx.cm.binary_metrics = cached        # imported by name into compare_external_models
    return cached


def memo(ctx: Ctx, name: str, payload: Dict[str, Any], fn: Any, est_default: float) -> Any:
    """Content-addressed, verified result cache for one statistics task (JSON round trip on every path)."""
    key = digest_of({"schema": SCHEMA, "task": name, "payload": payload, "B": ctx.a.n_bootstrap, "seed": ctx.a.seed})
    unit = f"{name.split(':')[0]}/{key[:24]}"
    got = ctx.memo.load_json(unit, key)
    if got is not None:
        return got
    kind = f"partB_stat_{name.split(':')[0]}"
    ctx.budget.require(ctx.timings.estimate(kind, est_default), f"statistics task {name}")
    t0 = time.time()
    res = fn()
    ctx.timings.record(kind, time.time() - t0)
    ctx.lock.beat()
    return ctx.memo.commit_json(unit, key, res, meta={"task": name})


def write_predictions(ctx: Ctx, name: str, k: str) -> Path:
    """Legacy-schema image and slide CSVs, built with the pinned aggregate_to_slides."""
    d = np.load(ctx.cond_root / name / f"{k}.npz", allow_pickle=False)
    if list(d["relative_path"]) != ctx.rel:
        raise RuntimeError(f"{name}/{k}: cached image order differs")
    z = d["logits"].astype(np.float64)
    prob = pa.softmax64(z)
    rows = []
    for i, r in enumerate(ctx.rows):
        pa_, pg = float(prob[i, 0]), float(prob[i, 1])
        pred, true = int(pg > pa_), int(float(r["label"]))
        rows.append({"relative_path": r["relative_path"], "filename": r["filename"], "slide_id": r["slide_id"],
                     "species_name": r["species_name"], "true_label": true, "pred_label": pred,
                     "pred_species": pc.CLASS_NAMES[pred], "prob_albicans": pa_, "prob_glabrata": pg,
                     "confidence": max(pa_, pg), "correct": int(pred == true), "hyphae_status": r["hyphae_status"],
                     "model_family": FAMILY_NAME[k], "logit_albicans": float(z[i, 0]), "logit_glabrata": float(z[i, 1])})
    mode = PIPELINES[name][2]
    ctx.lg.add_preprocessing_mode_to_image_rows(rows, mode)
    slides = ctx.lg.aggregate_to_slides(rows, preprocessing_mode=mode)
    for s in slides:
        s["model_family"] = FAMILY_NAME[k]
    edir = ctx.out / "predictions" / k / name
    ctx.ec.write_csv(rows, edir / "image_predictions.csv",
                     list(ctx.lg.IMAGE_PREDICTION_COLUMNS) + ["model_family", "logit_albicans", "logit_glabrata"])
    ctx.ec.write_csv(slides, edir / "slide_predictions.csv", list(ctx.lg.SLIDE_PREDICTION_COLUMNS) + ["model_family"])
    return edir


def archive_gate(ctx: Ctx, pred_dirs: Dict[Tuple[str, str], Path]) -> Dict[str, Any]:
    arch: Dict[Tuple[str, str], Path] = {}
    for spec in ctx.a.archive:
        key, path = spec.split("=", 1)
        m, pipe = key.split(":")
        if m not in MODELS or pipe not in ("E-locked", "E-z"):
            raise ValueError(f"--archive {spec}: expected <hybrid43|densenet121>:<E-locked|E-z>=DIR")
        arch[(m, pipe)] = Path(path)
    out: Dict[str, Any] = {}
    for m in MODELS:
        for pipe in ("E-locked", "E-z"):
            key = f"{m}:{pipe}"
            if (m, pipe) in arch:
                out[key] = ctx.ec.reproduction_check(pred_dirs[(m, pipe)], arch[(m, pipe)])   # raises on failure
            elif ctx.a.allow_ungated:
                out[key] = {"status": "not_gated (--allow-ungated)"}
            else:
                raise RuntimeError(f"no --archive for {key}; supply it or pass --allow-ungated (disclosed)")
    return out


# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------

def slide_arrays(ctx: Ctx, edir: Path) -> Dict[str, Any]:
    rows = sorted(ctx.ec.read_csv(edir / "slide_predictions.csv"), key=lambda r: r["slide_id"])
    if [r["slide_id"] for r in rows] != ctx.slides:
        raise RuntimeError(f"{edir}: slide set differs")
    f = lambda k: np.array([float(r[k]) for r in rows])  # noqa: E731
    y = f("true_label").astype(int)
    arr = {"pred": f("pred_label").astype(int), "p": f("mean_prob_glabrata"), "c": f("confidence"),
           "correct": f("correct").astype(int)}
    arr["p_true"] = np.where(y == 1, arr["p"], 1.0 - arr["p"])
    return {"y": y, "arr": arr, "rows": rows}


def clip_by_slide(ctx: Ctx, name: str) -> Dict[str, Any]:
    c = np.load(ctx.cond_root / name / "clip.npz", allow_pickle=False)
    sid = np.array([r["slide_id"] for r in ctx.rows])
    per = {s: float(np.mean(c["frac_clipped"][sid == s])) for s in ctx.slides}
    return {"mean_frac_pixels_clipped": float(np.mean(c["frac_clipped"])),
            "max_frac_pixels_clipped_image": float(np.max(c["frac_clipped"])),
            "mean_frac_at_bounds_final": float(np.mean(c["frac_at_bounds_final"])),
            "flag_heavy_clipping_gt_1pct": bool(np.mean(c["frac_clipped"]) > pa.CLIP_FLAG_FRACTION),
            "per_slide_mean_frac_clipped": per}


def per_condition_stats(ctx: Ctx, S: Dict[str, Any], img_dir: Path, bootstrap: bool) -> Dict[str, Any]:
    es, y, a = ctx.es, S["y"], S["arr"]
    point = es.binary_metrics(y, a["pred"], a["p"], a["c"])
    out = {"point": point, "correct_slides": f"{int(a['correct'].sum())}/{N_SLIDES}",
           "clopper_pearson": {"recall_albicans": es.clopper_pearson(point["k_correct_albicans"], point["n_albicans"]),
                               "recall_glabrata": es.clopper_pearson(point["k_correct_glabrata"], point["n_glabrata"])},
           "auroc_interpretation_rule": auroc_rule(y, a["p"]),
           "n_pred_albicans": int((a["pred"] == 0).sum()), "n_pred_glabrata": int((a["pred"] == 1).sum())}
    if bootstrap:
        out["bootstrap_ci"] = memo(ctx, "marginal", {"y": y, "pred": a["pred"], "p": a["p"], "c": a["c"]},
                                   lambda: es.marginal_bootstrap(y, a["pred"], a["p"], a["c"],
                                                                 n_bootstrap=ctx.a.n_bootstrap, seed=ctx.a.seed), 120.0)
    img = ctx.ec.read_csv(img_dir / "image_predictions.csv")
    out["image_level_descriptive"] = es.binary_metrics([int(r["true_label"]) for r in img],
                                                       [int(r["pred_label"]) for r in img],
                                                       [float(r["prob_glabrata"]) for r in img])
    return out


def _paired(ctx: Ctx, y: np.ndarray, A: Dict[str, np.ndarray], B: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """compare_external_models.paired_contrast (delta = B - A), memoized; replicate vectors are not kept."""
    def fn() -> Dict[str, Any]:
        r = ctx.cm.paired_contrast(y, A, B, n_bootstrap=ctx.a.n_bootstrap, seed=ctx.a.seed)
        r.pop("replicate_deltas", None)
        return r
    payload = {"y": y, "a": {k: A[k] for k in ("pred", "p", "c", "correct")},
               "b": {k: B[k] for k in ("pred", "p", "c", "correct")}}
    return memo(ctx, "paired", payload, fn, 240.0)


def compute_statistics(ctx: Ctx, dirs: Dict[Tuple[str, str], Path]) -> Dict[str, Any]:
    es, cm = ctx.es, ctx.cm
    S = {key: slide_arrays(ctx, d) for key, d in dirs.items()}
    y = S[(MODELS[0], "E-locked")]["y"]
    for key, s in S.items():
        if not np.array_equal(s["y"], y):
            raise RuntimeError(f"labels differ in {key}")
    res: Dict[str, Any] = {"per_condition": {}, "clip": {n: clip_by_slide(ctx, n) for n in PIPELINES}}
    for (m, n), s in S.items():
        res["per_condition"][f"{m}:{n}"] = per_condition_stats(ctx, s, dirs[(m, n)], bootstrap=n in PRIMARY_PIPES)

    published_check = {}
    for (m, n), exp in PUBLISHED.items():
        got = res["per_condition"][f"{m}:{n}"]["point"]
        mism = {k: [round(got[k], 4) if got[k] is not None else None, v] for k, v in exp.items()
                if got[k] is None or round(float(got[k]), 4) != round(v, 4)}
        published_check[f"{m}:{n}"] = {"reproduced": not mism, "mismatches_observed_vs_main_tex": mism}
    res["published_metrics_check"] = published_check
    if not all(v["reproduced"] for v in published_check.values()) and not ctx.a.allow_ungated:
        raise RuntimeError(f"fresh E-locked/E-z do not reproduce tab:external_locked: {published_check}")

    # Pipeline contrasts within each model
    pipe: Dict[str, Any] = {}
    raw: Dict[str, float] = {}
    for m in MODELS:
        for b, a_ in PIPE_CONTRASTS:
            pcres = _paired(ctx, y, S[(m, a_)]["arr"], S[(m, b)]["arr"])
            sf = signflip_exact(S[(m, b)]["arr"]["p_true"] - S[(m, a_)]["arr"]["p_true"])
            key = f"{m}:{b}-minus-{a_}"
            pipe[key] = {"paired": pcres, "signflip_delta_true_class_probability": sf}
            raw[f"{key}:mcnemar"] = pcres["mcnemar"]["p_value_two_sided_exact"]
            raw[f"{key}:signflip"] = sf["p_two_sided"]
    if ctx.a.holm_scheme == "protocol_m12":
        fams = {"pipeline_family_m12": raw}
    else:
        old = {k: v for k, v in raw.items() if ":E-z-minus-E-locked:" in k}
        fams = {"published_pipeline_family_m4": old,
                "new_pipeline_family_m8": {k: v for k, v in raw.items() if k not in old}}
    res["pipeline_contrasts"] = pipe
    res["holm"] = {fam: {k: {"p_raw": p, "p_holm": h} for k, h in es.holm_adjust(ps).items() for p in [ps[k]]}
                   for fam, ps in fams.items()}
    res["holm"]["published_pipeline_family_m4_recomputed"] = {
        k: {"p_raw": raw[k], "p_holm": h} for k, h in
        es.holm_adjust({k: v for k, v in raw.items() if ":E-z-minus-E-locked:" in k}).items()}
    res["holm"]["published_pipeline_family_m4_printed_values"] = PUBLISHED_HOLM["pipeline_family_m4_printed"]

    # Model contrasts within each pipeline (delta = DenseNet-121 minus ViT+CCT+LSA_43)
    mc, mraw = {}, {}
    for n in PRIMARY_PIPES:
        r = _paired(ctx, y, S[("hybrid43", n)]["arr"], S[("densenet121", n)]["arr"])
        mc[n] = r
        mraw[n] = r["mcnemar"]["p_value_two_sided_exact"]
    res["model_contrasts"] = {"delta_convention": "DenseNet-121 minus ViT+CCT+LSA_43", "by_pipeline": mc,
                              "holm_model_family_m3": {k: {"p_raw": mraw[k], "p_holm": v}
                                                       for k, v in es.holm_adjust(mraw).items()},
                              "holm_published_m2_recomputed": es.holm_adjust({k: mraw[k] for k in ("E-locked", "E-z")}),
                              "holm_published_m2_printed": PUBLISHED_HOLM["model_family_m2"]}

    # Recovery ratio, uncertainty propagation, E-alb_B
    res["recovery_ratio"] = {m: memo(ctx, "recovery", {"y": y, "preds": [S[(m, n)]["arr"]["pred"] for n in PRIMARY_PIPES]},
                                     lambda m=m: recovery_ratio(y, *(S[(m, n)]["arr"]["pred"] for n in PRIMARY_PIPES),
                                                                B=ctx.a.n_bootstrap, seed=ctx.a.seed, ext_stats=es), 60.0)
                             for m in MODELS}
    unc = {}
    for m in MODELS:
        for src in ("extreme_low", "extreme_high"):
            n = f"E-cal@{src}"
            unc[f"{m}:{src}"] = {
                "a": ctx.cal["bootstrap"][src]["a"], "b": ctx.cal["bootstrap"][src]["b"],
                "point": res["per_condition"][f"{m}:{n}"]["point"],
                "vs_E-locked_mcnemar": es.mcnemar_exact(S[(m, "E-locked")]["arr"]["correct"], S[(m, n)]["arr"]["correct"]),
                "vs_E-locked_signflip": signflip_exact(S[(m, n)]["arr"]["p_true"] - S[(m, "E-locked")]["arr"]["p_true"]),
                "same_slide_predictions_as_point_E-cal": bool(np.array_equal(S[(m, n)]["arr"]["pred"],
                                                                             S[(m, "E-cal")]["arr"]["pred"])),
                "note": "unadjusted; outside every Holm family"}
    res["uncertainty_propagation"] = unc
    i_b = ctx.slides.index(ctx.alb_b)
    res["E-alb_B"] = {"slide_id": ctx.alb_b, "display": display_id(ctx.alb_b), **{
        f"{m}:{n}": {"p_glabrata": float(S[(m, n)]["arr"]["p"][i_b]), "pred": pc.CLASS_NAMES[S[(m, n)]["arr"]["pred"][i_b]],
                     "correct": bool(S[(m, n)]["arr"]["correct"][i_b])} for (m, n) in S}}

    if ctx.a.legacy_signflip_crosscheck:
        cc = {}
        for key, v in pipe.items():
            vals = v["signflip_delta_true_class_probability"]["values"]
            cc[key] = {"new": v["signflip_delta_true_class_probability"]["p_two_sided"],
                       "legacy": ctx.lcp.exact_sign_flip_pvalue(vals)}
        res["signflip_legacy_crosscheck"] = cc
    res["_S"] = S
    return res


# -----------------------------------------------------------------------------
# Exports
# -----------------------------------------------------------------------------

def export_slide_table(ctx: Ctx, S: Dict[Tuple[str, str], Dict[str, Any]]) -> Dict[str, Any]:
    first = {r["slide_id"]: r for r in S[(MODELS[0], "E-locked")]["rows"]}
    order = sorted(ctx.slides, key=lambda s: (int(float(first[s]["true_label"])), display_id(s)))
    cols = [(m, n) for m in MODELS for n in PRIMARY_PIPES]
    rows = []
    for s in order:
        i = ctx.slides.index(s)
        r = {"slide_id": s, "display": display_id(s), "species": first[s]["species_name"],
             "hyphae_status": first[s]["hyphae_status"], "date": ctx.ec.slide_date(s),
             "within_session_contrast": s == ctx.alb_b}
        for m, n in cols:
            r[f"{m}:{n}:p_glabrata"] = float(S[(m, n)]["arr"]["p"][i])
            r[f"{m}:{n}:correct"] = int(S[(m, n)]["arr"]["correct"][i])
        rows.append(r)
    keys = list(rows[0])
    ctx.ec.write_csv(rows, ctx.out / "external_slides_extended.csv", keys)

    def cell(r: Dict[str, Any], m: str, n: str) -> str:
        v = f"{r[f'{m}:{n}:p_glabrata']:.4f}"
        return v if r[f"{m}:{n}:correct"] else f"\\underline{{{v}}}"

    tex = [r"\begin{table}[tbp]", r"\centering",
           r"\caption{Slide-mean probability of \textit{C. glabrata} for the 24 external slides under the "
           r"locked-pixel (L), instrument-calibrated (C) and photometric (Z) pipelines. Underlined values are "
           r"misclassified at the 0.5 threshold. \texttt{E-alb\_B} (bold) is the only \textit{C. albicans} slide "
           r"acquired on \texttt{20260507}.}",
           r"\label{tab:external_slides_extended}", r"\setlength{\tabcolsep}{4pt}",
           r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{llccc ccc}", r"\hline",
           r" & & \multicolumn{3}{c}{\textbf{ViT+CCT+LSA$_{43}$}} & \multicolumn{3}{c}{\textbf{DenseNet-121}} \\",
           r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
           r"\textbf{Slide} & \textbf{Hyphae} & L & C & Z & L & C & Z \\", r"\hline"]
    for r in rows:
        name = r["display"].replace("_", r"\_")
        name = rf"\textbf{{\texttt{{{name}}}}}" if r["within_session_contrast"] else rf"\texttt{{{name}}}"
        hy = {"hyphae": "yes", "no_hyphae": "no"}.get(r["hyphae_status"], "n/a")
        tex.append(f"{name} & {hy} & " + " & ".join(cell(r, m, n) for m, n in cols) + r" \\")
    tex += [r"\hline", r"\end{tabular}%", "}", r"\end{table}"]
    (ctx.out / "external_slides_extended.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    return {"csv": "external_slides_extended.csv", "tex": "external_slides_extended.tex", "rows": rows}


def export_surface(part_a: Dict[str, Any], figdir: Path) -> Dict[str, Any]:
    """6 x 5 slide-BA surface per model, diverging around chance (0.5); hatched cells meet R1-R3.
    Rows are physical scale m = 1200 / T in ascending order (verified strictly increasing); the
    external-matched row is outlined and labelled with its tile size and spacing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    figdir.mkdir(parents=True, exist_ok=True)
    cmap = LinearSegmentedColormap.from_list("ba", ["#e34948", "#f0efec", "#256abf"])   # red - gray - blue
    norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
    surf = part_a["sensitivity_surface"]
    geom = part_a.get("geometry", {})
    sp = {lv["T"]: lv["output_spacing_um"] for lv in geom.get("levels", [])}
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), constrained_layout=True)
    grids = {}
    for ax, m in zip(axes, MODELS):
        s = surf[m]
        ms = list(s["rows_m"])
        if any(a_ >= b_ for a_, b_ in zip(ms, ms[1:])):
            raise RuntimeError(f"surface rows not strictly increasing in m: {ms}")
        G = np.array([[c["slide_balanced_accuracy"] if c else np.nan for c in row] for row in s["grid"]])
        R_ = np.array([[bool(c and c.get("reproduces")) for c in row] for row in s["grid"]])
        grids[m] = {"rows_T": s["rows_T"], "rows_m": ms, "balanced_accuracy": G.tolist(), "reproduces": R_.tolist()}
        im = ax.imshow(G, cmap=cmap, norm=norm, origin="lower", aspect="auto")
        for i in range(G.shape[0]):
            for j in range(G.shape[1]):
                if R_[i, j]:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="//",
                                               edgecolor="#55555580", linewidth=0))
                if np.isfinite(G[i, j]):
                    dark = abs(G[i, j] - 0.5) > 0.35 and not R_[i, j]
                    ax.text(j, i, f"{G[i, j]:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if dark else "#1a1a19",
                            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85) if R_[i, j] else None)
        for i, role in enumerate(s["rows_role"]):
            if role == "external_matched":
                ax.add_patch(plt.Rectangle((-0.5, i - 0.5), G.shape[1], 1, fill=False, edgecolor="#1a1a19",
                                           linewidth=1.4))
        ax.set_xticks(range(len(s["cols_alpha"])), [f"{a:g}" for a in s["cols_alpha"]])
        ax.set_yticks(range(len(ms)), [f"{m_:.3f}{'*' if r == 'external_matched' else ''}  (T {t})"
                                       for m_, r, t in zip(ms, s["rows_role"], s["rows_T"])], fontsize=7.5)
        ax.set_xlabel(r"intensity level $\alpha$")
        ax.set_ylabel("physical scale $m = 1200/T$")
        ax.set_title(DISPLAY[m].replace("_43", r"$_{43}$"), fontsize=10)
        for spn in ax.spines.values():
            spn.set_visible(False)
    cb = fig.colorbar(im, ax=axes, shrink=0.85)
    cb.set_label("slide balanced accuracy (12 held-out slides)")
    t_ext = geom.get("T_external_matched")
    note = (f"* external-matched: m = {geom.get('m_external', float('nan')):.4f}, T = {t_ext} px, "
            f"{sp.get(t_ext, float('nan')):.3f} um/px at 384 px.  Hatched cells meet reproduction criteria R1 to R3")
    fig.suptitle(note, fontsize=8, color="#555555")
    for ext in ("pdf", "png"):
        fig.savefig(figdir / f"part_a_sensitivity_surface.{ext}", dpi=300)
    plt.close(fig)
    return {"files": ["figs/part_a_sensitivity_surface.pdf", "figs/part_a_sensitivity_surface.png"], "grids": grids}


def manuscript_summary(ctx: Ctx, part_a: Dict[str, Any], st: Dict[str, Any], table: Dict[str, Any],
                       surface: Dict[str, Any], gate: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
    def bt(m: str, n: str) -> Dict[str, Any]:
        c = st["per_condition"][f"{m}:{n}"]
        return {"correct_slides": c["correct_slides"], "point": {k: c["point"][k] for k in
                ("balanced_accuracy", "matthews_corrcoef", "auroc", "recall_albicans", "recall_glabrata", "brier", "ece")},
                "bootstrap_ci": c.get("bootstrap_ci"), "clopper_pearson": c["clopper_pearson"],
                "auroc_interpretation_rule": c["auroc_interpretation_rule"],
                "image_level": {k: c["image_level_descriptive"][k] for k in ("balanced_accuracy", "auroc")}}
    return {
        "schema": SCHEMA, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "plan_digest": plan["plan_digest"], "holm_scheme": ctx.a.holm_scheme,
        "targets": {
            "sec:results-domain": "calibration (a, b, intervals, training-mean shift, noise)",
            "new Methods subsections": "geometry and calibration constants",
            "new Results subsection + factorial table": "part_a.factorial, part_a.contrasts, part_a.verdicts",
            "new figure": "figs/part_a_sensitivity_surface.pdf",
            "tab:external_locked": "part_b.metrics (add E-cal columns)",
            "tab:external_contrast": "part_b.model_contrasts (m = 3 Holm)",
            "sec:results-sensitivity": "part_b.pipeline_contrasts, holm, recovery_ratio, E-alb_B",
            "tab:external_slides (Appendix)": "external_slides_extended.tex",
            "sec:limitations": "clip fractions, calibration intervals, uncertainty propagation"},
        "part_a": {"complete": part_a.get("complete"), "calibration": part_a.get("calibration"),
                   "geometry_digest": (part_a.get("geometry") or {}).get("geometry_digest"),
                   "geometry": part_a.get("geometry"), "c0_regression_gate": part_a.get("c0_regression_gate"),
                   "factorial": {m: v.get("factorial") for m, v in part_a.get("reproduction_verdicts", {}).items()},
                   "verdicts": part_a.get("reproduction_verdicts"), "contrasts": part_a.get("contrasts"),
                   "uncertainty_propagation": part_a.get("uncertainty_propagation"), "surface": surface["grids"]},
        "part_b": {
            "calibration_used": {"sha256": plan["plan"]["calibration_sha256"], "point": ctx.cal["point"],
                                 "extreme_low": ctx.cal["bootstrap"]["extreme_low"],
                                 "extreme_high": ctx.cal["bootstrap"]["extreme_high"],
                                 "inverse_map": "(x - b) / a on /16383-normalized external images"},
            "archive_gate": gate, "published_metrics_check": st["published_metrics_check"],
            "metrics": {f"{m}:{n}": bt(m, n) for m in MODELS for n in PRIMARY_PIPES},
            "clip": {n: {k: v for k, v in c.items() if k != "per_slide_mean_frac_clipped"} for n, c in st["clip"].items()},
            "pipeline_contrasts": {k: {"delta_balanced_accuracy": v["paired"]["deltas"]["balanced_accuracy"],
                                       "delta_auroc": v["paired"]["deltas"]["auroc"],
                                       "delta_brier": v["paired"]["deltas"]["brier"],
                                       "mcnemar": v["paired"]["mcnemar"],
                                       "signflip": {x: v["signflip_delta_true_class_probability"].get(x) for x in
                                                    ("mean_all", "n_nonzero", "p_two_sided")}}
                                   for k, v in st["pipeline_contrasts"].items()},
            "holm": st["holm"],
            "model_contrasts": {"delta_convention": st["model_contrasts"]["delta_convention"],
                                "holm_model_family_m3": st["model_contrasts"]["holm_model_family_m3"],
                                "holm_published_m2_recomputed": st["model_contrasts"]["holm_published_m2_recomputed"],
                                "holm_published_m2_printed": st["model_contrasts"]["holm_published_m2_printed"],
                                "by_pipeline": {n: {"deltas": r["deltas"], "mcnemar": r["mcnemar"],
                                                    "prediction_agreement_rate": r["prediction_agreement_rate"],
                                                    "delong_crosscheck": r["delong"]}
                                                for n, r in st["model_contrasts"]["by_pipeline"].items()}},
            "recovery_ratio": st["recovery_ratio"], "uncertainty_propagation": st["uncertainty_propagation"],
            "E-alb_B": st["E-alb_B"], "signflip_legacy_crosscheck": st.get("signflip_legacy_crosscheck"),
            "slide_table": table["rows"]},
        "provenance": {"unit_check": plan["plan"]["unit_check"], "models": plan["plan"]["models"],
                       "pinned_preprocess_sha256": pc.PINNED_PREPROCESS_SHA256,
                       "part_a_plan_digest": part_a.get("plan_digest"),
                       "script_sha256": pc.sha256_file(Path(__file__))},
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Protocol Part B and manuscript export.")
    p.add_argument("--stage", choices=("all", "infer", "stats"), default="all",
                   help="infer: GPU inference only; stats: statistics and exports (no CUDA, run in a fresh process)")
    p.add_argument("--part-a-dir", type=Path, required=True)
    p.add_argument("--calibration", type=Path, default=None, help="Default: <part-a-dir>/calibration/calibration.json")
    p.add_argument("--unit-check-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--external-root", type=Path, default=None)
    p.add_argument("--pinned-scripts", type=Path, required=True, help="ViT/HybridViT/validation/scripts")
    for n in ("densenet-scripts", "densenet-ckpt", "hybrid-validation-scripts", "hybrid-source", "hybrid-ckpt",
              "hybrid-config"):
        p.add_argument(f"--{n}", type=Path, required=True)
    p.add_argument("--allow-unsafe-densenet-load", action="store_true")
    p.add_argument("--archive", action="append", default=[], help="MODEL:PIPELINE=evaluation dir (E-locked, E-z)")
    p.add_argument("--allow-ungated", action="store_true", help="Proceed without archives / published match (disclosed)")
    p.add_argument("--accept-conditional", action="store_true")
    p.add_argument("--holm-scheme", choices=("protocol_m12", "separate_m4_m8"), default="protocol_m12")
    p.add_argument("--n-bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--chunk", type=int, default=128)
    p.add_argument("--prep", choices=("single-read", "per-pipeline"), default="single-read")
    p.add_argument("--fallback-per-pipeline", action="store_true",
                   help="If the single-read equivalence gate fails, use the per-pipeline path instead of stopping")
    p.add_argument("--prep-workers", type=int, default=2)
    p.add_argument("--prefetch", type=int, default=8)
    p.add_argument("--inference-queue", type=int, default=3)
    p.add_argument("--legacy-signflip-crosscheck", action="store_true", help="Slow (itertools over 2^24)")
    p.add_argument("--budget-hours", type=float, default=5.0)
    p.add_argument("--margin-minutes", type=float, default=20.0)
    p.add_argument("--session-start", type=float, default=None)
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--replan", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    if a.chunk % a.batch_size or SHARD_IMAGES % a.chunk:
        p.error(f"--chunk must be a multiple of --batch-size and divide {SHARD_IMAGES}")
    return a


def _common_inputs(ctx: Ctx) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    a = ctx.a
    gate_uc = pa.unit_check_gate(a)
    part_a = json.loads((a.part_a_dir / "part_a_summary.json").read_text())
    plan_a = json.loads((a.part_a_dir / "part_a_plan_frozen.json").read_text())
    if not part_a.get("complete"):
        raise RuntimeError(f"Part A incomplete: {part_a.get('missing_conditions')}")
    cpath = a.calibration or a.part_a_dir / "calibration" / "calibration.json"
    if pc.sha256_file(cpath) != plan_a["plan"]["calibration_sha256"]:
        raise RuntimeError("calibration.json differs from the one frozen in the Part A plan")
    ctx.cal = json.loads(cpath.read_text())
    gd = {"part_a_plan": plan_a["plan"].get("geometry_digest"),
          "part_a_summary": (part_a.get("geometry") or {}).get("geometry_digest"),
          "calibration": (ctx.cal.get("geometry") or {}).get("geometry_digest")}
    if None in gd.values() or len(set(gd.values())) != 1:
        raise RuntimeError(f"geometry digests disagree or are missing (legacy outputs?): {gd}")
    ctx.geometry_digest = gd["part_a_plan"]
    return gate_uc, part_a, plan_a


def _freeze_plan(ctx: Ctx, gate_uc: Dict[str, Any], plan_a: Dict[str, Any]) -> Dict[str, Any]:
    a = ctx.a
    plan = {"schema": SCHEMA, "pipelines": {n: pipe_params(ctx, n) for n in PIPELINES},
            "pipeline_contrasts": [f"{b} minus {a_}" for b, a_ in PIPE_CONTRASTS],
            "tests": {"mcnemar": "ext_stats.mcnemar_exact on slide correctness",
                      "signflip": "exact, all 2^n patterns, paired change in slide true-class probability, zeros removed"},
            "holm_scheme": a.holm_scheme, "model_family": "McNemar DenseNet-121 vs ViT+CCT+LSA_43 on E-locked, E-cal, E-z (m = 3)",
            "bootstrap": {"B": a.n_bootstrap, "seed": a.seed, "scheme": "class-stratified slide, paired"},
            "auroc_rule": {"gap": AUROC_PAIR_GAP, "share": AUROC_PAIR_SHARE},
            "calibration_sha256": plan_a["plan"]["calibration_sha256"], "part_a_plan_digest": plan_a["plan_digest"],
            "calibration_digest": pc.calibration_digest(ctx.cal), "geometry_digest": ctx.geometry_digest,
            "models": {k: {"sha256": m["sha256"], "mu": m["mu"], "sigma": m["sigma"]} for k, m in sorted(ctx.models.items())},
            "unit_check": {"sha256": gate_uc["sha256"], "status": gate_uc["overall_status"]},
            "manifest_sha256": pc.sha256_file(a.manifest), "pinned_preprocess_sha256": pc.PINNED_PREPROCESS_SHA256}
    ppath = a.out / "part_b_plan_frozen.json"
    dg = digest_of(plan)
    if ppath.exists():
        old = json.loads(ppath.read_text())
        if old["plan_digest"] != dg:
            if not a.replan:
                raise RuntimeError(f"{ppath} differs; use --replan to archive and replace it")
            arch = ppath.with_name(f"part_b_plan_frozen.superseded_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
            shutil.copy2(ppath, arch)
            hist = list(old.get("plan_history", [])) + [{"superseded_digest": old["plan_digest"],
                                                         "archived_as": arch.name,
                                                         "replaced_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}]
            write_json({"frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "plan_digest": dg,
                        "replaced": old["plan_digest"], "plan_history": hist, "plan": plan}, ppath)
    else:
        write_json({"frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "plan_digest": dg, "plan": plan}, ppath)
    return json.loads(ppath.read_text())


def main(argv: Optional[Sequence[str]] = None) -> None:
    a = parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ctx = Ctx(a)
    gate_uc, part_a, plan_a = _common_inputs(ctx)
    torch_cfg: Dict[str, Any] = {}

    if a.stage in ("all", "infer"):
        torch_cfg = vuc._configure_torch(a.seed)
        ctx.models = vuc.load_models(a)
        for k in MODELS:
            if ctx.models[k]["sha256"] != plan_a["plan"]["models"][k]["sha256"]:
                raise RuntimeError(f"{k}: checkpoint differs from Part A")
        ctx.model_ids = {k: [m["sha256"], m["mu"], m["sigma"]] for k, m in ctx.models.items()}
        ctx.setup_imports()
        ctx.load_manifest()
        plan_rec = _freeze_plan(ctx, gate_uc, plan_a)
        if a.plan_only:
            log(f"--plan-only: wrote {a.out / 'part_b_plan_frozen.json'}")
            return
        ctx.lock.acquire()
        try:
            infer_info = run_inference(ctx)
        finally:
            ctx.lock.release()
            cs.sync_filesystems()
        write_json({"infer": infer_info, "torch": torch_cfg, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                   a.out / "inference_record.json")
        if a.stage == "infer":
            return

    # ---- statistics and exports (no CUDA needed) ----
    plan_rec = json.loads((a.out / "part_b_plan_frozen.json").read_text())
    if not ctx.model_ids:
        ctx.model_ids = {k: [v["sha256"], v["mu"], v["sigma"]] for k, v in plan_rec["plan"]["models"].items()}
        ctx.setup_imports()
        ctx.load_manifest()
    if plan_rec["plan"]["pipelines"] != json.loads(json.dumps({n: pipe_params(ctx, n) for n in PIPELINES})):
        raise RuntimeError("pipeline parameters differ from part_b_plan_frozen.json")
    missing = [n for n in PIPELINES if not pipeline_complete(ctx, n)]
    if missing:
        raise RuntimeError(f"pipelines not complete or failing verification: {missing}; run --stage infer")
    cache = install_metric_cache(ctx)
    ctx.lock.acquire()
    try:
        dirs = {(m, n): write_predictions(ctx, n, m) for m in MODELS for n in PIPELINES}
        gate = archive_gate(ctx, dirs)
        log("archive gate: " + ", ".join(f"{k}={v.get('status')}" for k, v in gate.items()))
        st = compute_statistics(ctx, dirs)
    finally:
        ctx.lock.release()
        cs.sync_filesystems()
    S = st.pop("_S")
    table = export_slide_table(ctx, S)
    surface = export_surface(part_a, a.out / "figs")
    write_json({k: v for k, v in st.items()}, a.out / "part_b_statistics_full.json")
    summ = manuscript_summary(ctx, part_a, st, table, surface, gate, plan_rec)
    summ["environment"] = {"torch": torch_cfg or "statistics-only process", "runtime_seconds": round(time.time() - t0, 1),
                           "manifest_warnings": ctx.manifest_warnings,
                           "binary_metrics_cache": {"hits": cache.hits, "misses": cache.misses}}
    write_json(summ, a.out / "experimental_summary_for_manuscript.json")
    for m in MODELS:
        ba = {n: st["per_condition"][f"{m}:{n}"]["point"]["balanced_accuracy"] for n in PRIMARY_PIPES}
        log(f"{DISPLAY[m]}: BA {ba}; rho = {st['recovery_ratio'][m]['point']}")
    log(f"wrote {a.out / 'experimental_summary_for_manuscript.json'} (metric cache {cache.hits} hits / {cache.misses} misses)")


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
