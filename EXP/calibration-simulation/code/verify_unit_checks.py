#!/usr/bin/env python3
"""
verify_unit_checks.py

Checks
------
(a) C0 reproduction. m = 1, alpha = 0 arrays recomputed from the raw 12-bit
    TIFFs equal the committed internal arrays, and both frozen checkpoints, 
    fed the recomputed arrays, reproduce the held-out results published
    (tab:test_metrics, tab:strata) exactly in counts and to 4 decimals.
(b) Physical spacing. The geometry spec is recomputed from the exact optics
    (optics.py: pitch x binning / magnification); its geometry digest, T and
    0.1% residual must match, and TIFF metadata must agree with the optics.
    FAIL on any disagreement; CONDITIONAL only when internal metadata is absent.
(c) Commutation. Affine-then-resample equals resample-then-affine on pixels
    that neither order clips (internal tiles at all levels; external E-cal).
(d) Coordinates. All levels of the spec (strictly ordered by T and m): full-frame coverage, cuts
    inside the frame, interior taps inside the 8-pixel margin, border-rule taps
    only on frame-boundary sides, 384 x 384 output; plus run-time
    identification of OpenCV's boundary rule and crop parity with cv2.resize
    at every tile size.
Supporting: mask gain-equivariance; E-cal at (a, b) = (1, 0) equals E-locked
bitwise; E-locked equals the pinned locked-pixel operator bitwise.

Author: Rodrigo Sá
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import hashlib
import json
import math
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import geometry_harmonization as geo          # noqa: E402
import photometric_calibration as pc          # noqa: E402

PASS, FAIL, COND, SKIP = "PASS", "FAIL", "CONDITIONAL", "SKIPPED"
ARRAY_TOL = 1e-6
ENGINE_TOL = 1e-5
COMMUTE_TOL = 1e-5
LOGIT_TOL = 1e-4
MASK_TOL = 1e-4               # mask change under a float64 affine map (code equivariance)
STAT_TOL = 1e-3               # calibration statistics error, as a fraction of the mapped contrast a * c
MASK_ROUNDING_BOUND = 1e-2    # mask change from float32 rounding of a x + b (sanity bound)
SYNTHETIC_AB = (1.31, 0.137)
DISCORDANT_SLIDE = "20250918_alb_A"

# tab:test_metrics and tab:strata (held-out cohort, 821 patches, 12 slides)
EXPECTED: Dict[str, Dict[str, Any]] = {
    "densenet121": {
        "counts": {"patch_errors": 3, "fp": 2, "fn": 1, "slide_errors": 0, "slide_error_ids": [],
                   "errors_concordant": 3, "errors_discordant": 0},
        "metrics_4dp": {"patch_ba": 0.9964, "patch_mcc": 0.9927, "patch_f1": 0.9963, "patch_auroc": 0.9999,
                        "slide_ba": 1.0000, "slide_mcc": 1.0000, "slide_f1": 1.0000, "slide_auroc": 1.0000},
    },
    "hybrid43": {
        "counts": {"patch_errors": 37, "fp": 35, "fn": 2, "slide_errors": 1, "slide_error_ids": [DISCORDANT_SLIDE],
                   "errors_concordant": 5, "errors_discordant": 32},
        "metrics_4dp": {"patch_ba": 0.9576, "patch_mcc": 0.9129, "patch_f1": 0.9550, "patch_auroc": 0.9832,
                        "slide_ba": 0.9286, "slide_mcc": 0.8452, "slide_f1": 0.9172, "slide_auroc": 1.0000},
    },
}


# -----------------------------------------------------------------------------
# Report plumbing
# -----------------------------------------------------------------------------

class Report:
    def __init__(self, out: Path) -> None:
        self.out = out
        self.data: Dict[str, Any] = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                     "checks": {}}

    def run(self, name: str, fn: Callable[[], Dict[str, Any]], required: bool = True) -> Dict[str, Any]:
        t0 = time.time()
        print(f"[{name}] ...", flush=True)
        try:
            res = fn()
        except Exception as exc:  # recorded, never swallowed silently
            res = {"status": FAIL, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        res["required"] = required
        res["seconds"] = round(time.time() - t0, 2)
        self.data["checks"][name] = res
        print(f"[{name}] {res['status']} ({res['seconds']}s)" + (f" -> {res.get('error')}" if "error" in res else ""),
              flush=True)
        self.save()
        return res

    def finish(self) -> str:
        st = [c["status"] for c in self.data["checks"].values() if c["required"]]
        overall = FAIL if FAIL in st else (COND if COND in st or SKIP in st else PASS)
        self.data["overall_status"] = overall
        self.data["conditions"] = {k: c.get("condition") for k, c in self.data["checks"].items()
                                   if c["status"] in (COND, SKIP) and c["required"]}
        self.save()
        return overall

    def save(self) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "unit_check_report.json").write_text(json.dumps(self.data, indent=2, default=_json))


def _json(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def slide_of(base: str) -> str:
    return Path(base).name.rsplit("_", 1)[0]


def label_of(base: str) -> int:
    return pc.CLASS_NAMES.index(Path(base).parts[0])


def maxdiff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------

class Inputs:
    def __init__(self, a: argparse.Namespace) -> None:
        self.a = a
        split = json.loads(Path(a.split).read_text())
        self.train = list(split["train_base_paths"])
        self.test = list(split["test_base_paths"])
        if len(self.test) != 821 or len({slide_of(b) for b in self.test}) != 12:
            raise RuntimeError("split_indices.json is not the canonical 821-patch / 12-slide held-out set")
        by: Dict[str, List[str]] = {}
        for b in sorted(self.test):
            by.setdefault(slide_of(b), []).append(b)
        per = max(1, math.ceil(a.n_sample / len(by)))
        self.sample = [b for s in sorted(by) for b in by[s][:per]][: a.n_sample]
        self._frames: Dict[str, np.ndarray] = {}
        self.pinned = pc.load_pinned_preprocess(a.pinned_scripts) if a.pinned_scripts else None
        self.ext_rows = pc.read_external_manifest(a.external_manifest) if a.external_manifest else []
        self.ext_sample = sorted(self.ext_rows, key=lambda r: r["relative_path"])[:: max(1, len(self.ext_rows) // max(1, a.n_ext_sample))][: a.n_ext_sample] if self.ext_rows else []
        self.spec = json.loads(Path(a.geometry_spec).read_text()) if a.geometry_spec else None

    def frame(self, base: str) -> np.ndarray:
        if base not in self._frames:
            raw = pc.read_internal_raw(pc.internal_raw_path(self.a.internal_raw_root, base))
            self._frames[base], _ = pc.normalize_internal(raw)
        return self._frames[base]

    def raw(self, base: str) -> np.ndarray:
        return pc.read_internal_raw(pc.internal_raw_path(self.a.internal_raw_root, base))

    def ext_path(self, row: Dict[str, str]) -> Path:
        return Path(self.a.external_root) / row["relative_path"] if self.a.external_root else Path(row["image_path"])

    def T_external(self) -> int:
        if not self.spec:
            raise RuntimeError("--geometry-spec is required: there is no default external-matched tile size")
        return int(self.spec["external_matched_T_used"])

    def levels(self) -> List[Any]:
        if not self.spec:
            raise RuntimeError("--geometry-spec is required")
        return geo.levels_from_spec(self.spec)


# -----------------------------------------------------------------------------
# Checks
# -----------------------------------------------------------------------------

def check_environment(inp: Inputs) -> Dict[str, Any]:
    import platform

    prov = geo.assert_opencv_version(inp.a.expected_opencv)
    out = {"status": PASS, "python": sys.version.split()[0], "platform": platform.platform(), "opencv": prov}
    try:
        import tifffile
        out["tifffile"] = tifffile.__version__
    except Exception as exc:
        out["tifffile"] = f"unavailable: {exc}"
    out["condition"] = ("Original internal OpenCV version is not recorded in simpleViT_fungi; check (a1) is the "
                        "operative equivalence test.")
    return out


def check_border_and_parity(inp: Inputs) -> Dict[str, Any]:
    rules, parity = [], {}
    for base in inp.sample[: max(2, inp.a.n_sample // 4)]:
        rules.append(geo.identify_cv2_border_rule(inp.frame(base), tol=ENGINE_TOL))
    selected = {r["selected"] for r in rules}
    worst_full = max(min(r["max_abs_diff"].values()) for r in rules)
    rule = rules[0]["selected"]
    for lv in inp.levels():
        if lv.T == geo.N_INT:
            continue
        d = 0.0
        for base in inp.sample[:3]:
            f = inp.frame(base)
            for t in (lv.tiles[0], lv.tiles[-1]):
                d = max(d, geo.crop_parity_vs_cv2(f, lv.T, t.y0, t.x0, rule))
        parity[str(lv.T)] = d
    ok = len(selected) == 1 and worst_full <= ENGINE_TOL and all(v <= ENGINE_TOL for v in parity.values())
    inp.border_rule = rule
    return {"status": PASS if ok else FAIL, "selected_border_rule": rule, "rules_per_frame": rules,
            "full_frame_worst_max_abs_diff": worst_full, "crop_parity_max_abs_diff_by_T": parity,
            "tolerance": ENGINE_TOL}


def check_d_coordinates(inp: Inputs) -> Dict[str, Any]:
    rule = getattr(inp, "border_rule", "replicate")
    levels = inp.levels()
    geo.assert_monotonic(levels)
    per_level = []
    ok = True
    for lv in levels:
        plan = geo.make_axis_plan(lv.T)
        need_l, need_r = -plan.tap_min, plan.tap_max_over
        cov = np.zeros((geo.N_INT, geo.N_INT), dtype=np.int32)
        for t in lv.tiles:
            cov[t.y0:t.y0 + t.T, t.x0:t.x0 + t.T] += 1
        rec = {"T": lv.T, "m": lv.m, "role": lv.role, "n_tiles": lv.n_tiles, "offsets": list(lv.offsets),
               "stride_is_integer": float(lv.stride).is_integer(), "full_coverage": bool((cov > 0).all()),
               "required_raw_margin_left": need_l, "required_raw_margin_right": need_r,
               "margin_sufficient": max(need_l, need_r) <= geo.MARGIN_PX,
               "edge_share": lv.frame_edge_share_of_tile_sides, "guard_hits": plan.n_guard,
               "frames_checked": 0, "border_rule_taps_interior_sides": 0, "border_rule_taps_frame_sides": 0}
        for base in inp.sample:
            tiles, prov = geo.extract_level_tiles(inp.frame(base), lv, border_rule=rule)
            if tiles.shape != (lv.n_tiles, geo.D, geo.D) or not np.isfinite(tiles).all():
                raise geo.CoordinateError(f"T={lv.T}: bad output {tiles.shape}")
            for p in prov:
                if p["engine"] != "opencv_exact_numpy":
                    continue
                t = p["tile"]
                for ax, lo_side, hi_side in (("y", "top_on_frame", "bottom_on_frame"),
                                             ("x", "left_on_frame", "right_on_frame")):
                    lo, hi = p[ax]["taps_border_rule_low"], p[ax]["taps_border_rule_high"]
                    rec["border_rule_taps_frame_sides"] += lo * int(t[lo_side]) + hi * int(t[hi_side])
                    rec["border_rule_taps_interior_sides"] += lo * int(not t[lo_side]) + hi * int(not t[hi_side])
                cy0, cy1, cx0, cx1 = p["cut"]
                if not (0 <= cy0 < cy1 <= geo.N_INT and 0 <= cx0 < cx1 <= geo.N_INT):
                    raise geo.CoordinateError(f"cut {p['cut']} outside frame")
            rec["frames_checked"] += 1
        level_ok = (rec["full_coverage"] and rec["margin_sufficient"] and rec["guard_hits"] == 0
                    and rec["border_rule_taps_interior_sides"] == 0)
        rec["PASS"] = level_ok
        ok &= level_ok
        per_level.append(rec)
    ex = geo.external_level(levels)
    diag = geo.naive_vs_margin_difference(inp.frame(inp.sample[0]), ex, ex.n_tiles // 2, rule)
    return {"status": PASS if ok else FAIL, "border_rule": rule, "levels": per_level,
            "ordering": {"T": [lv.T for lv in levels], "m": [lv.m for lv in levels], "strict": True},
            "naive_crop_vs_margin_center_tile": diag,
            "assertion": "every tap reads a frame pixel; interior sides never use the boundary rule"}


def check_b_spacing(inp: Inputs) -> Dict[str, Any]:
    """Physical check: the spec must equal what the exact optics imply, and metadata must agree with the optics."""
    if inp.spec is None:
        return {"status": FAIL, "error": "no --geometry-spec supplied"}
    spec = inp.spec
    if spec.get("schema") != geo.SPEC_SCHEMA:
        return {"status": FAIL, "error": f"legacy geometry spec schema {spec.get('schema')!r}"}
    cfg = {"optics": {d: {k: spec["optics"][d][k] for k in ("camera", "sensor_pitch_um", "binning",
                                                           "total_magnification", "n_px", "native_n_px")}
                      for d in ("internal", "external")},
           "metadata_policy": spec["metadata_policy"], "geometry": {"supplementary_T": spec.get("supplementary_T", [])}}
    for d in cfg["optics"].values():
        if d["native_n_px"] is None:
            d.pop("native_n_px")
    exp = geo.expected_geometry(cfg)
    em, g = exp["external_matched"], spec["external_matched"]
    T_used = int(spec["external_matched_T_used"])
    out = {"T_used": T_used, "T_from_optics": em["T"], "T_star_exact": em["T_star_exact"],
           "p_int_exact": em["p_int_exact"], "p_ext_exact": em["p_ext_exact"],
           "fov_um": [em["fov_int_um"], em["fov_ext_um"]], "m_external": em["m_external"],
           "relative_residual": em["relative_residual"], "within_0.1pct": em["residual_within_0.1pct"],
           "geometry_digest_spec": spec.get("geometry_digest"), "geometry_digest_recomputed": exp["geometry_digest"],
           "metadata_status": g.get("metadata_status"), "metadata_value_um": g.get("metadata_value_um")}
    problems = []
    if T_used != em["T"]:
        problems.append(f"spec T {T_used} != optics T {em['T']}")
    if spec.get("geometry_digest") != exp["geometry_digest"]:
        problems.append("geometry digest differs from the optics recomputation")
    if not em["residual_within_0.1pct"]:
        problems.append(f"residual {em['relative_residual']:+.4%} not below 0.1%")
    for dom in ("internal", "external"):
        st = (g.get("metadata_status") or {}).get(dom)
        if st not in ("AGREES", "ABSENT"):
            problems.append(f"{dom} metadata status {st!r}")
        if st == "ABSENT" and spec["metadata_policy"][dom] == "required":
            problems.append(f"{dom} metadata required but absent")
    if problems:
        out.update(status=FAIL, error="; ".join(problems))
    elif (g.get("metadata_status") or {}).get("internal") == "ABSENT" or \
            (g.get("metadata_status") or {}).get("external") == "ABSENT":
        out.update(status=COND, condition="pixel-size metadata absent in at least one domain; the spacing rests "
                                          "on the exact optics parameters alone (disclose)")
    else:
        out["status"] = PASS
    return out


def _ab(inp: Inputs) -> Tuple[float, float, str]:
    if inp.a.calibration:
        c = json.loads(Path(inp.a.calibration).read_text())
        return float(c["point"]["a"]), float(c["point"]["b"]), "calibration.json point estimate"
    return SYNTHETIC_AB[0], SYNTHETIC_AB[1], "synthetic test values (no --calibration given)"


def check_c_commutation(inp: Inputs) -> Dict[str, Any]:
    rule = getattr(inp, "border_rule", "replicate")
    a, b, src = _ab(inp)
    worst, excluded, n_px = 0.0, 0, 0
    by_level: Dict[str, float] = {}
    for lv in inp.levels():
        d_lv = 0.0
        for base in inp.sample[: max(2, inp.a.n_sample // 3)]:
            x = inp.frame(base)
            for alpha in (0.5, 1.0):
                aa, bb = pc.dose_params(a, b, alpha)
                first, _ = geo.extract_level_tiles(pc.forward_map(x, aa, bb), lv, border_rule=rule)
                tiles, _ = geo.extract_level_tiles(x, lv, border_rule=rule)
                second = pc.forward_map(tiles, aa, bb)
                keep = (first >= 0) & (first <= 1) & (second >= 0) & (second <= 1)
                excluded += int((~keep).sum())
                n_px += keep.size
                if keep.any():
                    d_lv = max(d_lv, float(np.max(np.abs(first[keep].astype(np.float64) - second[keep]))))
        by_level[str(lv.T)] = d_lv
        worst = max(worst, d_lv)
    ext_worst = None
    if inp.pinned is not None and inp.ext_sample:
        ext_worst = 0.0
        for row in inp.ext_sample:
            x, _ = pc.read_external_normalized(inp.ext_path(row), inp.pinned)
            first = inp.pinned.resize_to_square(pc.inverse_map(x, a, b), geo.D, post_resize_clip=False)
            second = pc.inverse_map(inp.pinned.resize_to_square(x, geo.D, post_resize_clip=False), a, b)
            keep = (first >= 0) & (first <= 1) & (second >= 0) & (second <= 1)
            if keep.any():
                ext_worst = max(ext_worst, float(np.max(np.abs(first[keep].astype(np.float64) - second[keep]))))
    ok = worst <= COMMUTE_TOL and (ext_worst is None or ext_worst <= COMMUTE_TOL)
    out = {"status": PASS if ok else FAIL, "a": a, "b": b, "ab_source": src, "tolerance": COMMUTE_TOL,
           "internal_max_abs_diff_by_T": by_level, "internal_worst": worst,
           "fraction_pixels_excluded_as_clipped": excluded / max(n_px, 1), "external_E_cal_worst": ext_worst}
    if ok and ext_worst is None:
        out["status"] = COND
        out["condition"] = "external E-cal commutation not tested (no external manifest)"
    return out


def check_mask_equivariance(inp: Inputs) -> Dict[str, Any]:
    """
    The calibration needs m(a x + b) = a m(x) + b and c(a x + b) = a c(x) for a > 0, so that the fitted
    (a, b) do not depend on which detector's gain the mask happens to see. Three criteria:
      (1) code equivariance: with the affine map applied in float64, the mask is unchanged
          (fraction of changed pixels <= MASK_TOL; any change would be an operator, not a rounding, effect);
      (2) statistics equivariance on float32 data (the real data path): |m1 - (a m0 + b)| and |c1 - a c0|
          are at most STAT_TOL x a c0, i.e. 0.1% of the contrast that sets a;
      (3) sanity bound: the float32 mask change, caused only by rounding of a x + b near the threshold,
          stays below MASK_ROUNDING_BOUND (a non-equivariant threshold would change tens of percent).
    """
    rule = getattr(inp, "border_rule", "replicate")
    p = pc.MaskParams()
    lv = geo.make_level(inp.T_external(), role=geo.ROLE_EXTERNAL)
    worst64, worst32, worst_stat, n_pairs, n_skipped = 0.0, 0.0, 0.0, 0, 0
    rng = np.random.default_rng(42)
    for base in inp.sample[:3]:
        tiles, _ = geo.extract_level_tiles(inp.frame(base), lv, border_rule=rule)
        for t in tiles[:3]:
            m0, _ = pc.foreground_mask(t, p)
            s0 = pc.image_statistics(t, p)
            for _ in range(3):
                a, b = float(rng.uniform(0.5, 2.0)), float(rng.uniform(-0.2, 0.2))
                x64 = a * t.astype(np.float64) + b
                worst64 = max(worst64, float((pc.foreground_mask(x64, p)[0] != m0).mean()))
                x32 = (np.float32(a) * t + np.float32(b)).astype(np.float32)
                worst32 = max(worst32, float((pc.foreground_mask(x32, p)[0] != m0).mean()))
                s1 = pc.image_statistics(x32, p)
                if not (s0["valid"] and s1["valid"]):
                    n_skipped += 1
                    continue
                scale = a * s0["c"]
                worst_stat = max(worst_stat, abs(s1["m"] - (a * s0["m"] + b)) / scale, abs(s1["c"] - scale) / scale)
                n_pairs += 1
    ok = (worst64 <= MASK_TOL and n_pairs > 0 and worst_stat <= STAT_TOL and worst32 <= MASK_ROUNDING_BOUND)
    return {"status": PASS if ok else FAIL,
            "code_equivariance_float64_max_fraction_mask_changed": worst64, "tolerance_float64": MASK_TOL,
            "statistics_max_error_over_contrast": worst_stat, "tolerance_statistics": STAT_TOL,
            "float32_max_fraction_mask_changed": worst32, "bound_float32_rounding": MASK_ROUNDING_BOUND,
            "n_tile_perturbation_pairs": n_pairs, "n_pairs_skipped_invalid_mask": n_skipped,
            "mask_params": p.__dict__}


def check_external_identities(inp: Inputs) -> Dict[str, Any]:
    if inp.pinned is None or not inp.ext_sample:
        return {"status": SKIP, "condition": "no --external-manifest / --pinned-scripts supplied"}
    worst_cal, worst_pinned = 0.0, 0.0
    clipped = []
    for row in inp.ext_sample:
        path = inp.ext_path(row)
        locked, info = pc.external_condition(path, "E-locked", inp.pinned)
        cal, _ = pc.external_condition(path, "E-cal", inp.pinned, a=1.0, b=0.0)
        image, meta = inp.pinned.get_tiff_page_metadata(path)
        image = inp.pinned.squeeze_singleton_channel(image)
        x, _ = inp.pinned.normalize_linear_raw(image, 14)
        ref = inp.pinned.resize_to_square(x, geo.D, post_resize_clip=True)
        worst_cal = max(worst_cal, maxdiff(locked, cal))
        worst_pinned = max(worst_pinned, maxdiff(locked, ref))
        clipped.append(info["clip"]["frac_clipped"])
    ok = worst_cal == 0.0 and worst_pinned == 0.0
    return {"status": PASS if ok else FAIL, "E_cal_identity_vs_E_locked_max_abs_diff": worst_cal,
            "E_locked_vs_pinned_locked_pixel_max_abs_diff": worst_pinned, "n_images": len(inp.ext_sample),
            "E_locked_clipped_fraction_mean": float(np.mean(clipped))}


# ----- (a) C0 reproduction ---------------------------------------------------

def c0_arrays(inp: Inputs) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    rule = getattr(inp, "border_rule", "replicate")
    lv1 = geo.make_level(geo.N_INT)
    arrs, clips = [], []
    for i, base in enumerate(inp.test, 1):
        t, cl, _ = pc.internal_condition(inp.raw(base), lv1, 1.0, 0.0, border_rule=rule)
        arrs.append(t[0])
        clips.append(cl[0])
        if i % 200 == 0:
            print(f"  C0 arrays {i}/{len(inp.test)}", flush=True)
    return arrs, clips


def check_a1_arrays(inp: Inputs) -> Dict[str, Any]:
    arrs, clips = c0_arrays(inp)
    inp.c0 = arrs
    out: Dict[str, Any] = {"n": len(arrs), "n_clip_triggered": int(sum(c["clip_triggered"] for c in clips))}
    if not inp.a.npy_root:
        out.update(status=SKIP, condition="no --npy-root: array identity with the committed arrays not tested")
        return out
    diffs, exact = [], 0
    for base, arr in zip(inp.test, arrs):
        ref = np.load(Path(inp.a.npy_root) / f"{base}.npy")
        d = maxdiff(arr, ref)
        diffs.append(d)
        exact += int(d == 0.0)
    worst = max(diffs)
    out.update(n_bitwise_identical=exact, max_abs_diff=worst, median_max_abs_diff=float(np.median(diffs)),
               tolerance=ARRAY_TOL, status=PASS if worst <= ARRAY_TOL else FAIL)
    if worst > 1e-3:
        out["diagnosis"] = ("Large differences: the committed arrays were not produced by the 'baseline' path "
                            "(check for the BaSiC 'optimized' method, the CLI default in simpleViT_fungi).")
    return out


def _configure_torch(seed: int = 42) -> Dict[str, Any]:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    return {"torch": torch.__version__, "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def _purge(names: Sequence[str]) -> None:
    for n in names:
        sys.modules.pop(n, None)


def load_models(a: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sys.path.insert(0, str(a.densenet_scripts))
    from load_frozen_model import load_frozen_model

    dn = load_frozen_model(a.densenet_ckpt, device=dev, densenet_source=Path(a.densenet_scripts) / "densenet_model.py",
                           allow_unsafe_checkpoint_load=a.allow_unsafe_densenet_load)
    _purge(["config", "training_logic", "training_utils", "lsa_convtok_vit", "lsa_attention", "lsa_logger"])
    os.environ["EXTERNAL_VALIDATION_SOURCE_CODE_DIR"] = str(a.hybrid_source)
    sys.path.insert(0, str(a.hybrid_source))
    sys.path.insert(0, str(a.hybrid_validation_scripts))
    from load_frozen_hybrid import load_frozen_hybrid

    hy = load_frozen_hybrid(checkpoint_path=a.hybrid_ckpt, config_path=a.hybrid_config, device=dev, strict=True)
    raw = torch.load(str(a.hybrid_ckpt), map_location="cpu", weights_only=False)
    top_mean, top_std = raw.get("mean"), raw.get("std")
    if top_mean is None or top_std is None:
        raise RuntimeError("Hybrid checkpoint lacks top-level mean/std: the published loader would use defaults")
    if [float(v) for v in top_mean] != hy["mean"] or [float(v) for v in top_std] != hy["std"]:
        raise RuntimeError("Hybrid loader mean/std differ from the checkpoint's own values")
    if list(hy.get("class_names", [])) != list(pc.CLASS_NAMES):
        raise RuntimeError(f"Hybrid class order {hy.get('class_names')} != {list(pc.CLASS_NAMES)}")
    return {
        "densenet121": {"model": dn["model"].eval(), "mu": float(dn["mean"][0]), "sigma": float(dn["std"][0]),
                        "sha256": dn["metadata"]["checkpoint_sha256"], "optimal_epochs": dn["metadata"]["optimal_epochs"]},
        "hybrid43": {"model": hy["model"].eval(), "mu": float(hy["mean"][0]), "sigma": float(hy["std"][0]),
                     "sha256": pc.sha256_file(Path(a.hybrid_ckpt)), "optimal_epochs": raw.get("optimal_epochs")},
    }


def logits_for(model: Any, arrays: Sequence[np.ndarray], mu: float, sigma: float, bs: int) -> np.ndarray:
    import torch

    device = next(model.parameters()).device
    out = []
    with torch.inference_mode():
        for i in range(0, len(arrays), bs):
            x = np.stack([pc.standardize(t, mu, sigma) for t in arrays[i:i + bs]])[:, None]
            for d in ("cuda", "cpu"):
                try:
                    if torch.is_autocast_enabled(d):
                        raise RuntimeError("autocast active; FP32 required")
                except TypeError:
                    if torch.is_autocast_enabled():
                        raise RuntimeError("autocast active; FP32 required")
            z = model(torch.from_numpy(x).to(device))
            if z.dtype != torch.float32 or tuple(z.shape[1:]) != (2,):
                raise RuntimeError(f"unexpected logits {z.dtype} {tuple(z.shape)}")
            out.append(z.float().cpu().numpy())
    return np.concatenate(out)


def heldout_summary(bases: Sequence[str], z: np.ndarray) -> Dict[str, Any]:
    from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef, roc_auc_score

    y = np.array([label_of(b) for b in bases])
    zz = z.astype(np.float64)
    p = np.exp(zz - zz.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    pred = p.argmax(1)
    slides = sorted({slide_of(b) for b in bases})
    sidx = {s: [i for i, b in enumerate(bases) if slide_of(b) == s] for s in slides}
    sp = np.array([p[sidx[s]].mean(0) for s in slides])
    sy = np.array([y[sidx[s][0]] for s in slides])
    spred = sp.argmax(1)
    disc = np.array([slide_of(b) == DISCORDANT_SLIDE for b in bases])
    err = pred != y
    return {
        "counts": {"patch_errors": int(err.sum()), "fp": int(((y == 0) & (pred == 1)).sum()),
                   "fn": int(((y == 1) & (pred == 0)).sum()), "slide_errors": int((spred != sy).sum()),
                   "slide_error_ids": [s for s, a_, b_ in zip(slides, spred, sy) if a_ != b_],
                   "errors_concordant": int(err[~disc].sum()), "errors_discordant": int(err[disc].sum())},
        "metrics_4dp": {"patch_ba": round(balanced_accuracy_score(y, pred), 4),
                        "patch_mcc": round(matthews_corrcoef(y, pred), 4),
                        "patch_f1": round(f1_score(y, pred, average="weighted", zero_division=0), 4),
                        "patch_auroc": round(roc_auc_score(y, p[:, 1]), 4),
                        "slide_ba": round(balanced_accuracy_score(sy, spred), 4),
                        "slide_mcc": round(matthews_corrcoef(sy, spred), 4),
                        "slide_f1": round(f1_score(sy, spred, average="weighted", zero_division=0), 4),
                        "slide_auroc": round(roc_auc_score(sy, sp[:, 1]), 4)},
    }


def _cached_logits(path: Path, bases: Sequence[str], compute: Callable[[], np.ndarray], reuse: bool,
                   meta: Dict[str, Any]) -> np.ndarray:
    """Reuse only when the sidecar provenance (checkpoint, mu, sigma, batch size, patch list) matches exactly;
    a cache without a sidecar is recomputed."""
    side = path.with_name(path.name + ".meta.json")
    meta = {**meta, "bases_sha256": hashlib.sha256(json.dumps(list(bases)).encode()).hexdigest()}
    if reuse and path.exists() and side.exists():
        try:
            same = json.loads(side.read_text()) == json.loads(json.dumps(meta))
        except ValueError:
            same = False
        if same:
            d = np.load(path, allow_pickle=False)
            if list(d["bases"]) == list(bases):
                return d["logits"]
    z = compute()
    np.savez_compressed(path, bases=np.array(bases), logits=z)
    side.write_text(json.dumps(meta, indent=1))
    return z


def check_a2_models(inp: Inputs) -> Dict[str, Any]:
    a = inp.a
    det = _configure_torch(a.seed)
    models = load_models(a)
    if not hasattr(inp, "c0"):
        inp.c0, _ = c0_arrays(inp)
    cache = Path(a.out) / "logit_cache"
    cache.mkdir(parents=True, exist_ok=True)
    res: Dict[str, Any] = {"torch": det, "constants": {k: {"mu": v["mu"], "sigma": v["sigma"], "sha256": v["sha256"],
                                                           "optimal_epochs": v["optimal_epochs"]}
                                                       for k, v in models.items()}}
    res["constants"]["identical_between_models"] = (models["densenet121"]["mu"] == models["hybrid43"]["mu"]
                                                    and models["densenet121"]["sigma"] == models["hybrid43"]["sigma"])
    ok = True
    for name, m in models.items():
        lmeta = {"model": name, "checkpoint_sha256": m["sha256"], "mu": m["mu"], "sigma": m["sigma"],
                 "batch_size": a.batch_size}
        z_c0 = _cached_logits(cache / f"C0_recomputed_{name}.npz", inp.test,
                              lambda: logits_for(m["model"], inp.c0, m["mu"], m["sigma"], a.batch_size), a.reuse_cache,
                              {**lmeta, "arrays": "recomputed"})
        summ = heldout_summary(inp.test, z_c0)
        exp = EXPECTED[name]
        mism = [f"{blk}.{k}: got {summ[blk][k]}, main.tex {v}" for blk in ("counts", "metrics_4dp")
                for k, v in exp[blk].items() if summ[blk][k] != v]
        rec: Dict[str, Any] = {"recomputed_C0": summ, "mismatches_vs_main_tex": mism}
        model_ok = not mism
        if a.npy_root:
            committed = [pc.apply_internal_clip_rule(np.load(Path(a.npy_root) / f"{b}.npy").astype(np.float32))[0]
                         for b in inp.test]
            z_ref = _cached_logits(cache / f"C0_committed_{name}.npz", inp.test,
                                   lambda: logits_for(m["model"], committed, m["mu"], m["sigma"], a.batch_size),
                                   a.reuse_cache, {**lmeta, "arrays": "committed"})
            flips = int((z_ref.argmax(1) != z_c0.argmax(1)).sum())
            rec["vs_committed"] = {"max_abs_logit_diff": maxdiff(z_c0, z_ref), "label_flips": flips,
                                   "tolerance": LOGIT_TOL}
            model_ok &= rec["vs_committed"]["max_abs_logit_diff"] <= LOGIT_TOL and flips == 0
        rec["PASS"] = model_ok
        ok &= model_ok
        res[name] = rec
    res["status"] = PASS if ok else FAIL
    if ok and not a.npy_root:
        res["status"] = COND
        res["condition"] = "logit-level identity with the committed arrays not tested (no --npy-root)"
    return res


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--internal-raw-root", type=Path, required=True, help="Folder holding the two class folders of raw TIFFs")
    p.add_argument("--npy-root", type=Path, default=None, help="Extracted committed internal arrays (fungi.zip)")
    p.add_argument("--external-manifest", type=Path, default=None)
    p.add_argument("--external-root", type=Path, default=None)
    p.add_argument("--pinned-scripts", type=Path, default=None, help="ViT/HybridViT/validation/scripts")
    p.add_argument("--geometry-spec", type=Path, required=True, help="Output of geometry_harmonization.py (schema 2.0)")
    p.add_argument("--calibration", type=Path, default=None, help="calibration.json (else synthetic a, b for (c))")
    p.add_argument("--expected-opencv", default=None)
    p.add_argument("--n-sample", type=int, default=12, help="Held-out frames for (c), (d) and parity checks")
    p.add_argument("--n-ext-sample", type=int, default=6)
    p.add_argument("--skip-models", action="store_true")
    p.add_argument("--densenet-scripts", type=Path)
    p.add_argument("--densenet-ckpt", type=Path)
    p.add_argument("--hybrid-validation-scripts", type=Path)
    p.add_argument("--hybrid-source", type=Path)
    p.add_argument("--hybrid-ckpt", type=Path)
    p.add_argument("--hybrid-config", type=Path)
    p.add_argument("--allow-unsafe-densenet-load", action="store_true",
                   help="Pass through to load_frozen_model only if the archived checkpoint_identity.json "
                        "shows the frozen run used the same fallback")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reuse-cache", action="store_true", help="Reuse cached C0 logits when the patch list matches")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    if not a.skip_models:
        miss = [n for n in ("densenet_scripts", "densenet_ckpt", "hybrid_validation_scripts", "hybrid_source",
                            "hybrid_ckpt", "hybrid_config") if getattr(a, n) is None]
        if miss:
            p.error(f"model checks need {miss} (or pass --skip-models)")
    return a


def main() -> None:
    a = parse_args()
    rep = Report(a.out)
    rep.data["args"] = {k: str(v) for k, v in vars(a).items()}
    inp = Inputs(a)
    rep.data["inputs"] = {"split_sha256": pc.sha256_file(a.split), "n_heldout": len(inp.test),
                          "sample": inp.sample, "external_sample": [r["relative_path"] for r in inp.ext_sample],
                          "T_external_matched": inp.T_external()}
    rep.run("environment", lambda: check_environment(inp))
    rep.run("d0_border_rule_and_cv2_parity", lambda: check_border_and_parity(inp))
    rep.run("d_tile_coordinates_all_levels", lambda: check_d_coordinates(inp))
    rep.run("b_spacing_physical", lambda: check_b_spacing(inp))
    rep.run("c_affine_resample_commutation", lambda: check_c_commutation(inp))
    rep.run("mask_gain_equivariance", lambda: check_mask_equivariance(inp))
    rep.run("external_operator_identities", lambda: check_external_identities(inp))
    rep.run("a1_C0_arrays_vs_committed", lambda: check_a1_arrays(inp))
    if a.skip_models:
        rep.data["checks"]["a2_C0_frozen_predictions"] = {"status": SKIP, "required": True,
                                                         "condition": "--skip-models"}
    else:
        rep.run("a2_C0_frozen_predictions", lambda: check_a2_models(inp))
    overall = rep.finish()
    print(f"\nOVERALL: {overall}  ->  {a.out / 'unit_check_report.json'}")
    for k, c in rep.data["checks"].items():
        print(f"  {c['status']:<12} {k}")


if __name__ == "__main__":
    main()
