#!/usr/bin/env python3
"""
photometric_calibration.py

Intensity factor I: label-free affine calibration between the
internal (12-bit, Prime 95B) and external (14-bit, Axiocam 820)
detectors, (internal, forward map) and (external: E-locked, E-cal,
E-z), clip accounting, and the noise-power diagnostic (factor N).

Operators reused verbatim
-------------------------
Spatial scale. All mask and noise parameters below are in pixels of the 384 x 384 statistics
image. They are physically commensurate between domains only because both are sampled at the
external-matched spacing (about 0.511 um/px, from geometry_spec.json); fit_calibration refuses to
run when the two spacings differ by 0.1% or more, and records every kernel in micrometres.

Internal raw scaling replicates simpleViT_fungi ``data/preprocessing.py``
(``MicroscopyPreprocessor._global_linear_normalize``, method 'baseline'):
float32 / 4095, clip to [0, 1] only when a value leaves [0, 1]; the post-resize
rule clips the whole image only when a value leaves [-0.01, 1.01].
External operators are imported, unmodified, from the SHA-256-pinned
``preprocess_external.py`` (normalize_linear_raw, resize_to_square,
photometric_mean_std_match_to_checkpoint, TIFF reader and bit-depth inference).

Author: Rodrigo Sá
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import geometry_harmonization as geo

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

INT_DENOM = 4095        # 12-bit, simpleViT_fungi bit_depth_max
EXT_DENOM = 16383       # 14-bit, pinned normalize_linear_raw
INTERNAL_CLIP_TOL = 0.01
ALPHAS: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
PINNED_PREPROCESS_SHA256 = "3df9a093ced600e5407b32bd07f05a77712725e2fc87faa27b63483d3f96e4c0"  # ext_common.PINNED_SHA256
TIFF_SUFFIXES = (".tif", ".tiff", ".TIF", ".TIFF")
CLASS_NAMES = ("Candida albicans", "Candida glabrata")


@dataclass(frozen=True)
class MaskParams:
    gauss_sigma: float = 1.0          # px at the matched spacing (um value: calibration.json kernel_scales_um)
    sobel_ksize: int = 3
    mad_k: float = 3.0
    dilate_radius: int = 3            # px at the matched spacing (um value: calibration.json kernel_scales_um)
    min_fraction: float = 0.01        # minimum share of pixels in each of foreground and background
    percentile: float = 5.0           # Protocol.md: 5th percentile of foreground pixels
    noise_highpass_sigma: float = 2.0 # px at the matched spacing
    mask_dtype: str = "float64"       # blur, Sobel and threshold in float64: gain-equivariant up to input rounding

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]


# -----------------------------------------------------------------------------
# Pinned external module
# -----------------------------------------------------------------------------

_PINNED: Dict[str, Any] = {}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def load_pinned_preprocess(scripts_dir: Path) -> Any:
    """Import the pinned preprocess_external.py by path after verifying its SHA-256."""
    path = Path(scripts_dir) / "preprocess_external.py"
    key = str(path.resolve())
    if key in _PINNED:
        return _PINNED[key]
    digest = sha256_file(path)
    if digest != PINNED_PREPROCESS_SHA256:
        raise RuntimeError(f"{path} sha256 {digest} != pinned {PINNED_PREPROCESS_SHA256}")
    spec = importlib.util.spec_from_file_location("_pinned_preprocess_external", str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    _PINNED[key] = mod
    return mod


# -----------------------------------------------------------------------------
# Raw readers and normalization
# -----------------------------------------------------------------------------

def internal_raw_path(raw_root: Path, base: str) -> Path:
    """split_indices base path 'Candida albicans/20250813_alb_B_1' -> raw TIFF under raw_root."""
    for suf in TIFF_SUFFIXES:
        p = Path(raw_root) / f"{base}{suf}"
        if p.exists():
            return p
    raise FileNotFoundError(f"No raw TIFF for {base} under {raw_root}")


def read_internal_raw(path: Path) -> np.ndarray:
    """simpleViT_fungi load_12bit_tif: TiffFile.asarray(); single 2-D 1200 x 1200 page."""
    import tifffile

    with tifffile.TiffFile(str(path)) as tf:
        img = tf.asarray()
    if img.ndim != 2:
        raise ValueError(f"{path.name}: expected a 2-D image, got {img.shape}")
    if img.shape != (geo.N_INT, geo.N_INT):
        raise ValueError(f"{path.name}: shape {img.shape} != {(geo.N_INT, geo.N_INT)}")
    return img


def normalize_internal(raw: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """MicroscopyPreprocessor._global_linear_normalize (bit_depth 12, method 'baseline')."""
    x = raw.astype(np.float32) / INT_DENOM
    info = {"raw_max": int(raw.max()), "raw_min": int(raw.min()), "raw_over_4095": int((raw > INT_DENOM).sum())}
    if x.max() > 1.0 or x.min() < 0.0:
        x = np.clip(x, 0.0, 1.0)
        info["normalization_clip_applied"] = True
    else:
        info["normalization_clip_applied"] = False
    return x.astype(np.float32, copy=False), info


def read_external_normalized(path: Path, pinned: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    """preprocess_one (pinned) up to normalize_linear_raw: 14-bit expected, strict."""
    image, meta = pinned.get_tiff_page_metadata(Path(path))
    if not pinned.is_single_channel_image(image, meta.get("samples_per_pixel")):
        raise ValueError(f"{path}: not single channel {image.shape}")
    image = pinned.squeeze_singleton_channel(image)
    if image.ndim != 2 or image.size == 0:
        raise ValueError(f"{path}: invalid image {image.shape}")
    inferred = pinned.infer_effective_bit_depth(max_pixel=int(np.max(image)),
                                                container_bit_depth=meta.get("container_bit_depth"),
                                                expected_effective_bit_depth=14)
    if inferred != 14:
        raise ValueError(f"{path}: inferred bit depth {inferred} != 14")
    x, denom = pinned.normalize_linear_raw(image, inferred)
    if denom != EXT_DENOM:
        raise ValueError(f"{path}: denominator {denom} != {EXT_DENOM}")
    if tuple(image.shape) != (geo.N_EXT, geo.N_EXT):
        raise ValueError(f"{path}: shape {tuple(image.shape)} != {(geo.N_EXT, geo.N_EXT)} (2 x 2 binned Axiocam 820)")
    return x, {"raw_max": int(image.max()), "raw_over_16383": int((image > EXT_DENOM).sum()),
               "shape": list(image.shape)}


# -----------------------------------------------------------------------------
# Affine maps, clip rules, standardization
# -----------------------------------------------------------------------------

def dose_params(a: float, b: float, alpha: float) -> Tuple[float, float]:
    """a(alpha) = 1 + alpha (a - 1), b(alpha) = alpha b."""
    return 1.0 + float(alpha) * (float(a) - 1.0), float(alpha) * float(b)


def forward_map(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Internal -> external units. Exact identity for (1, 0)."""
    if a == 1.0 and b == 0.0:
        return x
    return (np.float32(a) * x + np.float32(b)).astype(np.float32, copy=False)


def inverse_map(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """External -> internal units (E-cal). Exact identity for (1, 0)."""
    if a == 1.0 and b == 0.0:
        return x
    return ((x - np.float32(b)) / np.float32(a)).astype(np.float32, copy=False)


def _range_stats(x: np.ndarray) -> Dict[str, float]:
    n = float(x.size)
    return {"min": float(x.min()), "max": float(x.max()),
            "frac_below_0": float((x < 0.0).sum() / n), "frac_above_1": float((x > 1.0).sum() / n),
            "frac_outside_unit": float(((x < 0.0) | (x > 1.0)).sum() / n),
            "frac_outside_tolerance": float(((x < -INTERNAL_CLIP_TOL) | (x > 1.0 + INTERNAL_CLIP_TOL)).sum() / n)}


def apply_internal_clip_rule(x: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """simpleViT_fungi post-resize rule (and NPYImageFolder loader): clip the whole image only beyond +/-0.01."""
    s = _range_stats(x)
    triggered = bool(x.min() < -INTERNAL_CLIP_TOL or x.max() > 1.0 + INTERNAL_CLIP_TOL)
    s["clip_rule"] = "internal_conditional"
    s["clip_triggered"] = triggered
    s["frac_clipped"] = s["frac_outside_unit"] if triggered else 0.0
    return (np.clip(x, 0.0, 1.0).astype(np.float32) if triggered else x.astype(np.float32, copy=False)), s


def apply_external_clip(x: np.ndarray, pinned: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Pinned clip_unit_interval (unconditional)."""
    s = _range_stats(x)
    s["clip_rule"] = "external_unconditional"
    s["clip_triggered"] = bool(s["frac_outside_unit"] > 0)
    s["frac_clipped"] = s["frac_outside_unit"]
    return pinned.clip_unit_interval(x), s


def standardize(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """(x - mu) / sigma in float32: identical to torchvision Normalize and to ExternalNPYDataset."""
    if not (sigma > 0 and math.isfinite(sigma) and math.isfinite(mu)):
        raise ValueError(f"Invalid frozen statistics mu={mu}, sigma={sigma}")
    return ((x.astype(np.float32, copy=False) - np.float32(mu)) / np.float32(sigma)).astype(np.float32, copy=False)


# -----------------------------------------------------------------------------
# Condition pipelines (exact Protocol order)
# -----------------------------------------------------------------------------

def internal_condition(raw: np.ndarray, level: geo.Level, a_alpha: float, b_alpha: float, *, border_rule: str,
                       margin: int = geo.MARGIN_PX, engine: str = "auto"
                       ) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Part A: raw -> /4095 -> affine (a(alpha), b(alpha)) -> tile cut with margin ->
    Lanczos-4 to 384 -> trim -> internal clip rule. Returns unit-range tiles
    (n_tiles, 384, 384) float32 before standardization, per-tile clip records,
    and normalization info. (m = 1, alpha = 0) reproduces the original array.
    """
    x, norm_info = normalize_internal(raw)
    x = forward_map(x, a_alpha, b_alpha)
    tiles, prov = geo.extract_level_tiles(x, level, border_rule=border_rule, margin=margin, engine=engine)
    out = np.empty_like(tiles)
    clip_records = []
    for i in range(tiles.shape[0]):
        out[i], s = apply_internal_clip_rule(tiles[i])
        s["tile_index"] = i
        s["cut"] = prov[i]["cut"]
        clip_records.append(s)
    return out, clip_records, norm_info


EXTERNAL_MODES = ("E-locked", "E-cal", "E-z")


def external_condition(path: Path, mode: str, pinned: Any, *, a: Optional[float] = None, b: Optional[float] = None,
                       mu: Optional[float] = None, sigma: Optional[float] = None
                       ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Part B, unit-range 384 x 384 array before standardization.
      E-locked: normalize -> resize -> clip                     (= pinned locked_pixel_pipeline)
      E-cal   : normalize -> inverse affine -> resize -> clip    (one instrument-level transform)
      E-z     : normalize -> resize -> clip -> per-image mean/std to (mu, sigma) -> clip
                                                               (= pinned photometric_mean_std_matched_pipeline)
    """
    if mode not in EXTERNAL_MODES:
        raise ValueError(f"mode must be one of {EXTERNAL_MODES}")
    x, info = read_external_normalized(path, pinned)
    if mode == "E-cal":
        if a is None or b is None or not a > 0:
            raise ValueError("E-cal requires a > 0 and b")
        x = inverse_map(x, a, b)
        info["after_inverse_map"] = _range_stats(x)
    resized = pinned.resize_to_square(x, geo.D, post_resize_clip=False)
    y, clip = apply_external_clip(resized, pinned)
    info["clip"] = clip
    if mode == "E-z":
        if mu is None or sigma is None:
            raise ValueError("E-z requires the frozen mu and sigma")
        y, meta = pinned.photometric_mean_std_match_to_checkpoint(y, [float(mu)], [float(sigma)], min_source_std=1e-6,
                                                                  post_harmonization_clip=True)
        info["photometric"] = {k: v for k, v in meta.items() if k != "interpretation_guardrail"}
    info["mode"] = mode
    return y.astype(np.float32, copy=False), info


# -----------------------------------------------------------------------------
# Mask, per-image statistics, noise power
# -----------------------------------------------------------------------------

def foreground_mask(x: np.ndarray, p: MaskParams) -> Tuple[np.ndarray, Dict[str, float]]:
    """Gain-equivariant mask: for x' = a x + b (a > 0), g and tau both scale by a, so the mask is unchanged."""
    import cv2

    if p.mask_dtype not in ("float64", "float32"):
        raise ValueError(f"mask_dtype must be float64 or float32, got {p.mask_dtype!r}")
    f64 = p.mask_dtype == "float64"
    xf = np.ascontiguousarray(x, dtype=np.float64 if f64 else np.float32)
    depth = cv2.CV_64F if f64 else cv2.CV_32F
    xs = cv2.GaussianBlur(xf, (0, 0), sigmaX=p.gauss_sigma, sigmaY=p.gauss_sigma, borderType=cv2.BORDER_REFLECT_101)
    gx = cv2.Sobel(xs, depth, 1, 0, ksize=p.sobel_ksize, borderType=cv2.BORDER_REFLECT_101)
    gy = cv2.Sobel(xs, depth, 0, 1, ksize=p.sobel_ksize, borderType=cv2.BORDER_REFLECT_101)
    g = np.sqrt(gx.astype(np.float64) ** 2 + gy.astype(np.float64) ** 2)
    med = float(np.median(g))
    mad = float(np.median(np.abs(g - med)))
    tau = med + p.mad_k * 1.4826 * mad
    seed = (g > tau).astype(np.uint8)
    k = 2 * p.dilate_radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    fg = cv2.dilate(seed, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    return fg, {"grad_median": med, "grad_mad": mad, "tau": tau, "fg_fraction": float(fg.mean())}


def image_statistics(x: np.ndarray, p: MaskParams) -> Dict[str, Any]:
    """Background median m, foreground contrast c = m - P5(fg), sensitivity variants, noise power."""
    import cv2

    fg, mi = foreground_mask(x, p)
    bg = ~fg
    rec: Dict[str, Any] = {**mi, "n_fg": int(fg.sum()), "n_bg": int(bg.sum())}
    ok = fg.mean() >= p.min_fraction and bg.mean() >= p.min_fraction
    rec["valid"] = bool(ok)
    if not ok:
        return rec
    xd = x.astype(np.float64)
    xs = cv2.GaussianBlur(np.ascontiguousarray(x, dtype=np.float32), (0, 0), sigmaX=p.gauss_sigma,
                          sigmaY=p.gauss_sigma, borderType=cv2.BORDER_REFLECT_101).astype(np.float64)
    m = float(np.median(xd[bg]))
    fgv = xd[fg]
    rec["m"] = m
    rec["q5"] = float(np.percentile(fgv, p.percentile, method="linear"))
    rec["c"] = m - rec["q5"]
    rec["c_p1"] = m - float(np.percentile(fgv, 1.0, method="linear"))
    rec["c_p10"] = m - float(np.percentile(fgv, 10.0, method="linear"))
    ms = float(np.median(xs[bg]))
    rec["m_smoothed"] = ms
    rec["c_smoothed"] = ms - float(np.percentile(xs[fg], p.percentile, method="linear"))
    hp = xd - cv2.GaussianBlur(np.ascontiguousarray(x, dtype=np.float32), (0, 0), sigmaX=p.noise_highpass_sigma,
                               sigmaY=p.noise_highpass_sigma, borderType=cv2.BORDER_REFLECT_101).astype(np.float64)
    r = hp[bg]
    rec["noise_var"] = float(np.var(r))
    rec["noise_mad2"] = float((1.4826 * np.median(np.abs(r - np.median(r)))) ** 2)
    return rec


STAT_KEYS = ("m", "c", "q5", "c_p1", "c_p10", "m_smoothed", "c_smoothed", "noise_var", "noise_mad2")


def internal_field_statistics(raw: np.ndarray, level: geo.Level, p: MaskParams, border_rule: str) -> Dict[str, Any]:
    """External-matched tiles of one development field (no affine, no clip); field value = median over tiles."""
    x, norm = normalize_internal(raw)
    tiles, _ = geo.extract_level_tiles(x, level, border_rule=border_rule)
    recs = [image_statistics(t, p) for t in tiles]
    valid = [r for r in recs if r["valid"]]
    out: Dict[str, Any] = {"n_tiles": len(recs), "n_tiles_valid": len(valid), "raw_over_4095": norm["raw_over_4095"]}
    for k in STAT_KEYS:
        out[k] = float(np.median([r[k] for r in valid])) if valid else float("nan")
    out["valid"] = bool(valid)
    return out


def external_frame_statistics(path: Path, pinned: Any, p: MaskParams) -> Dict[str, Any]:
    """Pinned locked-pixel geometry (2256 -> 384), statistics before any clip."""
    x, info = read_external_normalized(path, pinned)
    resized = pinned.resize_to_square(x, geo.D, post_resize_clip=False)
    rec = image_statistics(resized, p)
    out = {k: rec.get(k, float("nan")) for k in STAT_KEYS}
    out.update({"valid": rec["valid"], "raw_over_16383": info["raw_over_16383"], "fg_fraction": rec["fg_fraction"]})
    return out


# -----------------------------------------------------------------------------
# Aggregation, fit, bootstrap
# -----------------------------------------------------------------------------

def slide_medians(rows: Sequence[Dict[str, Any]], keys: Sequence[str] = STAT_KEYS) -> Dict[str, Dict[str, float]]:
    """Median over valid images within each slide (equal slide weights downstream)."""
    by: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        if r.get("valid") in (True, "True", "true", 1, "1"):
            by.setdefault(str(r["slide_id"]), []).append(r)
    out: Dict[str, Dict[str, float]] = {}
    for s, rs in sorted(by.items()):
        out[s] = {k: float(np.median([float(r[k]) for r in rs])) for k in keys}
        out[s]["n_images"] = float(len(rs))
    return out


def fit_affine(int_slides: Dict[str, Dict[str, float]], ext_slides: Dict[str, Dict[str, float]],
               m_key: str = "m", c_key: str = "c") -> Dict[str, float]:
    M_int = float(np.median([v[m_key] for v in int_slides.values()]))
    C_int = float(np.median([v[c_key] for v in int_slides.values()]))
    M_ext = float(np.median([v[m_key] for v in ext_slides.values()]))
    C_ext = float(np.median([v[c_key] for v in ext_slides.values()]))
    if not (C_int > 0 and C_ext > 0):
        raise RuntimeError(f"Non-positive contrast: C_int={C_int}, C_ext={C_ext}")
    a = C_ext / C_int
    b = M_ext - a * M_int
    return {"M_int": M_int, "C_int": C_int, "M_ext": M_ext, "C_ext": C_ext, "a": a, "b": b,
            "mapped_internal_background": a * M_int + b, "inverse_mapped_external_background": (M_ext - b) / a}


def bootstrap_affine(int_slides: Dict[str, Dict[str, float]], ext_slides: Dict[str, Dict[str, float]], *,
                     n_boot: int = 10000, seed: int = 42, m_key: str = "m", c_key: str = "c") -> Dict[str, Any]:
    """
    Slides resampled with replacement independently within each domain (no
    species stratification: labels are not used). Ranking statistic for the
    extreme pairs: s* = a* M_int + b* (internal background mapped by the
    replicate), with M_int the point estimate; each extreme keeps its own (a*, b*).
    """
    mi = np.array([v[m_key] for v in int_slides.values()])
    ci = np.array([v[c_key] for v in int_slides.values()])
    me = np.array([v[m_key] for v in ext_slides.values()])
    ce = np.array([v[c_key] for v in ext_slides.values()])
    rng = np.random.default_rng(seed)
    I = rng.integers(0, mi.size, size=(n_boot, mi.size))
    E = rng.integers(0, me.size, size=(n_boot, me.size))
    Mi, Ci = np.median(mi[I], axis=1), np.median(ci[I], axis=1)
    Me, Ce = np.median(me[E], axis=1), np.median(ce[E], axis=1)
    a = Ce / Ci
    b = Me - a * Mi
    point = fit_affine(int_slides, ext_slides, m_key, c_key)
    s = a * point["M_int"] + b
    order = np.argsort(s, kind="stable")
    lo_i = int(order[max(math.ceil(0.025 * n_boot) - 1, 0)])
    hi_i = int(order[min(math.ceil(0.975 * n_boot) - 1, n_boot - 1)])
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
    return {
        "n_boot": int(n_boot), "seed": int(seed), "n_internal_slides": int(mi.size), "n_external_slides": int(me.size),
        "a_percentile_95": q(a), "b_percentile_95": q(b), "s_percentile_95": q(s),
        "extreme_low": {"replicate": lo_i, "a": float(a[lo_i]), "b": float(b[lo_i]), "s": float(s[lo_i])},
        "extreme_high": {"replicate": hi_i, "a": float(a[hi_i]), "b": float(b[hi_i]), "s": float(s[hi_i])},
        "ranking_statistic": "s* = a* * M_int(point) + b*",
        "a_replicates_sha256": hashlib.sha256(a.tobytes()).hexdigest(),
    }


CALIBRATION_SCHEMA = "calibration/2.0"


def calibration_digest(cal: Dict[str, Any]) -> str:
    """Digest of the numbers a downstream array depends on: point and joint-extreme (a, b) as exact float hex.
    Rewriting calibration.json with identical constants leaves it unchanged."""
    vals = {"point": [float(cal["point"]["a"]).hex(), float(cal["point"]["b"]).hex()]}
    for src in ("extreme_low", "extreme_high"):
        e = cal["bootstrap"][src]
        vals[src] = [float(e["a"]).hex(), float(e["b"]).hex()]
    return hashlib.sha256(json.dumps(vals, sort_keys=True).encode()).hexdigest()


def kernel_scales_um(params: "MaskParams", spacing_int_um: float, spacing_ext_um: float) -> Dict[str, Any]:
    """Physical size of every pixel-defined statistic kernel in each domain."""
    def one(sp: float) -> Dict[str, float]:
        return {"spacing_um_per_px": sp, "gauss_sigma_um": params.gauss_sigma * sp,
                "dilate_radius_um": params.dilate_radius * sp, "noise_highpass_sigma_um": params.noise_highpass_sigma * sp,
                "sobel_support_um": params.sobel_ksize * sp}
    return {"internal_tiles": one(spacing_int_um), "external_frames": one(spacing_ext_um),
            "relative_spacing_difference": (spacing_int_um - spacing_ext_um) / spacing_ext_um}


def dose_response_table(a: float, b: float, alphas: Sequence[float] = ALPHAS) -> List[Dict[str, float]]:
    out = []
    for al in alphas:
        aa, bb = dose_params(a, b, al)
        out.append({"alpha": float(al), "a_alpha": aa, "b_alpha": bb, "inverse_a": aa, "inverse_b": bb})
    return out


def domain_noise_summary(int_slides: Dict[str, Dict[str, float]], ext_slides: Dict[str, Dict[str, float]],
                         a: float) -> Dict[str, Any]:
    """Factor N: equal-slide median noise power at the matched 384-px spacing (calibration.json
    geometry.spacing_tile_um); external also in internal units (divided by a^2)."""
    out: Dict[str, Any] = {}
    for key in ("noise_var", "noise_mad2"):
        ni = float(np.median([v[key] for v in int_slides.values()]))
        ne = float(np.median([v[key] for v in ext_slides.values()]))
        out[key] = {"internal": ni, "external_raw_units": ne, "external_in_internal_units": ne / (a * a),
                    "ratio_external_over_internal_internal_units": (ne / (a * a)) / ni if ni > 0 else float("nan")}
    return out


# -----------------------------------------------------------------------------
# Fitting workflow with on-disk caching
# -----------------------------------------------------------------------------

INT_COLS = ["base", "slide_id", "valid", "n_tiles", "n_tiles_valid", "raw_over_4095", *STAT_KEYS]
EXT_COLS = ["relative_path", "slide_id", "valid", "raw_over_16383", "fg_fraction", *STAT_KEYS]


def _slide_of_base(base: str) -> str:
    return Path(base).name.rsplit("_", 1)[0]


def _read_cache(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    key = "base" if rows and "base" in rows[0] else "relative_path"
    return {r[key]: r for r in rows}


def _append_rows(path: Path, cols: Sequence[str], rows: Iterable[Dict[str, Any]]) -> None:
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cols), extrasaction="ignore")
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def _int_task(args: Tuple[str, str, int, str, Dict[str, Any]]) -> Dict[str, Any]:
    raw_root, base, T, border, pd = args
    level = geo.make_level(T, role=geo.ROLE_EXTERNAL)
    rec = internal_field_statistics(read_internal_raw(internal_raw_path(Path(raw_root), base)), level,
                                    MaskParams(**pd), border)
    return {"base": base, "slide_id": _slide_of_base(base), **rec}


def _ext_task(args: Tuple[str, str, str, str, Dict[str, Any]]) -> Dict[str, Any]:
    path, rel, slide, pinned_dir, pd = args
    rec = external_frame_statistics(Path(path), load_pinned_preprocess(Path(pinned_dir)), MaskParams(**pd))
    return {"relative_path": rel, "slide_id": slide, **rec}


def _run(tasks: List[Any], fn, workers: int, cache_path: Path, cols: Sequence[str], label: str) -> None:
    if not tasks:
        return
    t0 = time.time()
    buf: List[Dict[str, Any]] = []
    if workers <= 1:
        it = map(fn, tasks)
        for i, r in enumerate(it, 1):
            buf.append(r)
            if len(buf) >= 50:
                _append_rows(cache_path, cols, buf)
                buf = []
            if i % 200 == 0:
                print(f"  {label}: {i}/{len(tasks)} ({time.time() - t0:.0f}s)", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for i, r in enumerate(ex.map(fn, tasks, chunksize=8), 1):
                buf.append(r)
                if len(buf) >= 50:
                    _append_rows(cache_path, cols, buf)
                    buf = []
                if i % 200 == 0:
                    print(f"  {label}: {i}/{len(tasks)} ({time.time() - t0:.0f}s)", flush=True)
    _append_rows(cache_path, cols, buf)


def read_external_manifest(path: Path) -> List[Dict[str, str]]:
    with Path(path).open(newline="") as f:
        rows = list(csv.DictReader(f))
    need = {"image_path", "relative_path", "slide_id"}
    if not rows or not need <= set(rows[0]):
        raise ValueError(f"{path}: manifest needs columns {sorted(need)}")
    return rows


def fit_calibration(*, split_json: Path, internal_raw_root: Path, external_manifest: Path,
                    external_root: Optional[Path], pinned_scripts: Path, geometry_spec: Path, out_dir: Path,
                    params: MaskParams, n_boot: int, seed: int, workers: int, border_rule: str) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = json.loads(Path(geometry_spec).read_text())
    if spec.get("schema") != geo.SPEC_SCHEMA:
        raise RuntimeError(f"{geometry_spec}: schema {spec.get('schema')!r} is not {geo.SPEC_SCHEMA}; rebuild the spec")
    T = int(spec["external_matched_T_used"])
    gdig = str(spec["geometry_digest"])
    em = spec["external_matched"]
    scales = kernel_scales_um(params, float(em["spacing_tile_at_384_um"]), float(em["spacing_external_at_384_um"]))
    if abs(scales["relative_spacing_difference"]) >= geo.SPACING_TOLERANCE:
        raise RuntimeError(f"internal tile spacing {em['spacing_tile_at_384_um']} and external spacing "
                           f"{em['spacing_external_at_384_um']} um/px differ by >= 0.1%: the pixel-defined mask "
                           "and noise statistics would compare different physical scales")
    split = json.loads(Path(split_json).read_text())
    dev = list(split["train_base_paths"])
    n_dev_slides = len({_slide_of_base(b) for b in dev})
    if n_dev_slides != 48 or len(dev) != 2910:
        raise RuntimeError(f"Expected 48 development slides / 2,910 fields, got {n_dev_slides} / {len(dev)}")
    pinned = load_pinned_preprocess(pinned_scripts)

    if border_rule == "auto":
        x0, _ = normalize_internal(read_internal_raw(internal_raw_path(internal_raw_root, dev[0])))
        bi = geo.identify_cv2_border_rule(x0)
        if not bi["PASS"]:
            raise RuntimeError(f"No border rule reproduces cv2.resize: {bi}")
        border_rule = bi["selected"]

    int_cache = out_dir / internal_cache_name(T, gdig, params, border_rule)
    ext_cache = out_dir / external_cache_name(params)
    done_i = _read_cache(int_cache)
    pd = asdict(params)
    _run([(str(internal_raw_root), b, T, border_rule, pd) for b in dev if b not in done_i], _int_task, workers,
         int_cache, INT_COLS, "internal fields")

    manifest = read_external_manifest(external_manifest)
    if len(manifest) != 1253 or len({r["slide_id"] for r in manifest}) != 24:
        raise RuntimeError(f"Expected 1,253 external images / 24 slides, got {len(manifest)} / "
                           f"{len({r['slide_id'] for r in manifest})}")
    done_e = _read_cache(ext_cache)
    ext_tasks = []
    for r in manifest:
        if r["relative_path"] in done_e:
            continue
        path = Path(external_root) / r["relative_path"] if external_root else Path(r["image_path"])
        ext_tasks.append((str(path), r["relative_path"], r["slide_id"], str(pinned_scripts), pd))
    _run(ext_tasks, _ext_task, workers, ext_cache, EXT_COLS, "external frames")

    int_rows = list(_read_cache(int_cache).values())
    ext_rows = list(_read_cache(ext_cache).values())
    if len(int_rows) != len(dev) or len(ext_rows) != len(manifest):
        raise RuntimeError(f"Cache incomplete: {len(int_rows)}/{len(dev)} internal, {len(ext_rows)}/{len(manifest)} external")
    si, se = slide_medians(int_rows), slide_medians(ext_rows)
    if len(si) != 48 or len(se) != 24:
        raise RuntimeError(f"Slides with valid images: internal {len(si)}/48, external {len(se)}/24")

    point = fit_affine(si, se)
    boot = bootstrap_affine(si, se, n_boot=n_boot, seed=seed)
    sens = {name: fit_affine(si, se, mk, ck) for name, mk, ck in
            (("P1", "m", "c_p1"), ("P10", "m", "c_p10"), ("smoothed_P5", "m_smoothed", "c_smoothed"))}
    result = {
        "schema": CALIBRATION_SCHEMA,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "module": "photometric_calibration.py",
        "label_free": True,
        "transductive_note": "external statistics use unlabeled external images from all 24 slides",
        "geometry": {"T_external_matched": T, "border_rule": border_rule, "geometry_digest": gdig,
                     "spacing_tile_um": em["spacing_tile_at_384_um"],
                     "spacing_external_um": em["spacing_external_at_384_um"],
                     "m_external": em["m_external"], "metadata_status": em["metadata_status"]},
        "optics": spec["optics"],
        "kernel_scales_um": scales,
        "mask_params": asdict(params), "mask_params_digest": params.digest(),
        "counts": {"internal_fields": len(int_rows), "internal_fields_valid": int(sum(r["valid"] == "True" for r in int_rows)),
                   "internal_slides": len(si), "external_images": len(ext_rows),
                   "external_images_valid": int(sum(r["valid"] == "True" for r in ext_rows)), "external_slides": len(se)},
        "slide_medians": {"internal": si, "external": se},
        "point": point,
        "bootstrap": boot,
        "sensitivity_contrast_definition": sens,
        "dose_response_point": dose_response_table(point["a"], point["b"]),
        "dose_response_extreme_low": dose_response_table(boot["extreme_low"]["a"], boot["extreme_low"]["b"]),
        "dose_response_extreme_high": dose_response_table(boot["extreme_high"]["a"], boot["extreme_high"]["b"]),
        "inverse_map_E_cal": {"formula": "(x - b) / a", "a": point["a"], "b": point["b"]},
        "factor_N_noise": domain_noise_summary(si, se, point["a"]),
        "provenance": {"pinned_preprocess_sha256": sha256_file(Path(pinned_scripts) / "preprocess_external.py"),
                       "split_sha256": sha256_file(Path(split_json)),
                       "geometry_spec_sha256": sha256_file(Path(geometry_spec)),
                       "internal_cache": int_cache.name, "external_cache": ext_cache.name,
                       "opencv": geo.opencv_provenance()},
    }
    result["calibration_digest"] = calibration_digest(result)
    tmp = out_dir / "calibration.json.tmp"
    tmp.write_text(json.dumps(result, indent=2, default=str))
    os.replace(tmp, out_dir / "calibration.json")
    return result


def internal_cache_name(T: int, geometry_digest: str, params: "MaskParams", border_rule: str) -> str:
    return f"internal_field_stats_T{int(T)}_{geometry_digest[:12]}_{params.digest()}_{border_rule}.csv"


def external_cache_name(params: "MaskParams") -> str:
    """External statistics depend on the mask, the pinned operator and D only (2256 -> 384 is spacing-free)."""
    return f"external_frame_stats_{params.digest()}_{PINNED_PREPROCESS_SHA256[:12]}_D{geo.D}.csv"


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Label-free photometric calibration (Protocol Parts A/B).")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--split", type=Path, required=True)
    f.add_argument("--internal-raw-root", type=Path, required=True)
    f.add_argument("--external-manifest", type=Path, required=True)
    f.add_argument("--external-root", type=Path, default=None,
                   help="Directory holding the class folders; paths = root / relative_path")
    f.add_argument("--pinned-scripts", type=Path, required=True)
    f.add_argument("--geometry-spec", type=Path, required=True)
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--border-rule", default="auto", choices=["auto", *geo.BORDER_RULES])
    f.add_argument("--bootstrap", type=int, default=10000)
    f.add_argument("--seed", type=int, default=42)
    f.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    f.add_argument("--gauss-sigma", type=float, default=MaskParams.gauss_sigma)
    f.add_argument("--mad-k", type=float, default=MaskParams.mad_k)
    f.add_argument("--dilate-radius", type=int, default=MaskParams.dilate_radius)
    a = p.parse_args(argv)
    params = MaskParams(gauss_sigma=a.gauss_sigma, mad_k=a.mad_k, dilate_radius=a.dilate_radius)
    res = fit_calibration(split_json=a.split, internal_raw_root=a.internal_raw_root,
                          external_manifest=a.external_manifest, external_root=a.external_root,
                          pinned_scripts=a.pinned_scripts, geometry_spec=a.geometry_spec, out_dir=a.out,
                          params=params, n_boot=a.bootstrap, seed=a.seed, workers=a.workers,
                          border_rule=a.border_rule)
    pt, bt = res["point"], res["bootstrap"]
    print(f"a = {pt['a']:.6f} [{bt['a_percentile_95'][0]:.6f}, {bt['a_percentile_95'][1]:.6f}]  "
          f"b = {pt['b']:.6f} [{bt['b_percentile_95'][0]:.6f}, {bt['b_percentile_95'][1]:.6f}]")
    print(f"Wrote {Path(a.out) / 'calibration.json'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
