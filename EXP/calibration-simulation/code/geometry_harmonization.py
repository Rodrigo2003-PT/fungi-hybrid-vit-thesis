#!/usr/bin/env python3
"""
geometry_harmonization.py

Geometry factor S: exact-spacing tiling of the internal 1200 x 1200
frames so that each tile, resampled to 384 x 384, has the pixel
spacing of an external input, with every input pixel computed from real image content.

Physical scale (optics.py is the only source; TIFF metadata is checked against it)
-------------------------------------------------------------------------------
internal  11.0 um / 63 = 0.174603 um/px, FOV 209.52 um; full frame at 384 px: 0.5456 um/px
external  2 x 2.74 um / 63 = 0.086984 um/px (2 x 2 binning, 4512 -> 2256), FOV 196.24 um;
          at 384 px: 0.5110 um/px
external-matched tile T = round(1123.90) = 1124 px, physical scale m = 1200 / T* = 1.0677.
Levels are identified by their integer tile side T; m = 1200 / T is a derived label.
The sweep (nominal m = 1, 1.25, 1.5, 1.75, 2 plus the external-matched level) is kept
strictly ordered: T decreasing, m increasing.

Provenance of the operator
--------------------------------------------------
The committed internal arrays were produced by simpleViT_fungi
``data/preprocessing.py::resize_and_preprocess_image``:

    x = tifffile.TiffFile(path).asarray().astype(np.float32) / 4095
    x = clip(x, 0, 1)                      only if a value leaves [0, 1]
    y = cv2.resize(x, (384, 384), interpolation=cv2.INTER_LANCZOS4)
    y = clip(y, 0, 1)                      only if y.min() < -0.01 or y.max() > 1.01

``cv2.INTER_LANCZOS4`` is a fixed 8-tap separable kernel
whose support does not grow when downsampling.
This module reproduces it in two ways:

* full frame (m = 1): ``cv2.resize`` itself, i.e. the original call;
* tiles (m > 1): an exact reimplementation of OpenCV's resize arithmetic
  (float32 source coordinate, ``interpolateLanczos4`` weights, sequential
  float32 accumulation, horizontal pass then vertical pass) applied to a tile
  cut with an 8-pixel margin on every interior side. The 384 target samples
  are addressed directly; this equals resampling an extended grid and trimming
  it, without requiring the trimmed margin to be an integer. Parity with 
  ``cv2.resize`` is tested on crops at every tile size (``crop_parity_vs_cv2``).

Author: Rodrigo Sá
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import optics

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

SPEC_SCHEMA = "geometry_spec/2.0"
N_INT = 1200                     # internal frame side contract; must equal optics.internal.n_px
N_EXT = 2256                     # external frame side contract (2 x 2 binned 4512); must equal optics.external.n_px
D = optics.D                     # network input side
MARGIN_PX = 8                    # Protocol.md: raw margin on interior tile edges
NOMINAL_M: Tuple[float, ...] = (1.0, 1.25, 1.5, 1.75, 2.0)   # nominal sweep, T = round(1200 / m)
ROLE_NOMINAL, ROLE_EXTERNAL, ROLE_SUPPLEMENTARY = "nominal", "external_matched", "supplementary"
SPACING_TOLERANCE = float(optics.SPACING_TOLERANCE)          # Protocol.md: mismatch below 0.1%
FLT_EPSILON = float(np.finfo(np.float32).eps)
BORDER_RULES = ("replicate", "reflect101")


class CoordinateError(AssertionError):
    """A resampling tap would read outside real image content."""


def round_half_up(x: float) -> int:
    """Rounding used for every tile size (Python's round() is half-to-even)."""
    return int(math.floor(float(x) + 0.5))


# -----------------------------------------------------------------------------
# Pixel spacing from metadata
# -----------------------------------------------------------------------------

_MICRON_UNITS = {"micron", "microns", "um", "µm", "μm", "\\u00b5m", "micrometer", "micrometre"}
_OME_UNIT_TO_UM = {"um": 1.0, "µm": 1.0, "μm": 1.0, "nm": 1e-3, "mm": 1e3, "cm": 1e4, "m": 1e6}


SPACING_KEYS = ("ome_PhysicalSizeX", "micromanager_PixelSizeUm", "micromanager_summary_PixelSize_um",
                "imagej_unit_XResolution", "tiff_XResolution_cm", "tiff_XResolution_inch")


def _ppu(v: Any) -> Optional[float]:
    if isinstance(v, tuple) and len(v) == 2 and v[0] and v[1]:
        return float(v[0]) / float(v[1])
    return None


def read_spacing_candidates(path: Path) -> Dict[str, Any]:
    """
    Every pixel-size statement found in one TIFF (um/px), keyed by source, read
    from the header only. Sources: OME-XML PhysicalSizeX, Micro-Manager
    PixelSizeUm, ImageJ unit + XResolution, TIFF XResolution with
    ResolutionUnit inch/cm. Statements are *rejected* (listed with a reason,
    never used) when ResolutionUnit is absent, when the value lies outside
    the plausible object-plane range (e.g. a 72 dpi writer default), and
    X/Y resolutions that differ are flagged as anisotropic.
    """
    import tifffile

    lo, hi = (float(v) for v in optics.PLAUSIBLE_UM_PER_PX)
    out: Dict[str, Any] = {"file": str(path), "candidates": {}, "rejected": {}, "anisotropic": None}

    def add(key: str, v: float) -> None:
        if lo <= v <= hi and math.isfinite(v):
            out["candidates"][key] = v
        else:
            out["rejected"][key] = f"{v:.6g} um/px outside plausible [{lo}, {hi}]"

    with tifffile.TiffFile(str(path)) as tf:
        page = tf.pages[0]
        tags = {t.name: t.value for t in page.tags.values()}
        out["shape"] = list(page.shape)
        if tf.is_ome and tf.ome_metadata:
            m = re.search(r'PhysicalSizeX="([0-9.eE+-]+)"', tf.ome_metadata)
            u = re.search(r'PhysicalSizeXUnit="([^"]+)"', tf.ome_metadata)
            if m:
                unit = u.group(1) if u else "µm"
                if unit in _OME_UNIT_TO_UM:
                    add("ome_PhysicalSizeX", float(m.group(1)) * _OME_UNIT_TO_UM[unit])
                else:
                    out["rejected"]["ome_PhysicalSizeX"] = f"unknown unit {unit!r}"
        mm = tags.get("MicroManagerMetadata")
        if isinstance(mm, dict) and mm.get("PixelSizeUm") not in (None, 0, 0.0):
            add("micromanager_PixelSizeUm", float(mm["PixelSizeUm"]))
        mmeta = getattr(tf, "micromanager_metadata", None)
        if isinstance(mmeta, dict):
            summ = mmeta.get("Summary") or mmeta.get("summary") or {}
            if isinstance(summ, dict) and summ.get("PixelSize_um"):
                add("micromanager_summary_PixelSize_um", float(summ["PixelSize_um"]))
        xppu, yppu, unit = _ppu(tags.get("XResolution")), _ppu(tags.get("YResolution")), tags.get("ResolutionUnit")
        if xppu is not None:
            if yppu is not None and abs(yppu - xppu) > 1e-9 * xppu:
                out["anisotropic"] = {"x_pixels_per_unit": xppu, "y_pixels_per_unit": yppu}
            if unit is None:
                out["rejected"]["tiff_XResolution"] = "ResolutionUnit tag absent; unit unknown"
            else:
                u = int(unit)
                ij = tf.imagej_metadata or {}
                ij_unit = str(ij.get("unit", "")).strip().lower()
                if u == 1 and ij_unit in _MICRON_UNITS and xppu > 0:
                    add("imagej_unit_XResolution", 1.0 / xppu)
                elif u == 3 and xppu > 0:
                    add("tiff_XResolution_cm", 1e4 / xppu)
                elif u == 2 and xppu > 0:
                    add("tiff_XResolution_inch", 25400.0 / xppu)
                else:
                    out["rejected"]["tiff_XResolution"] = f"ResolutionUnit {u} without a micron unit"
    return out


@dataclass
class SpacingRecord:
    domain: str
    um_per_px: float                 # float(optics pixel_um): the value used downstream
    exact_fraction: str              # the same value as an exact rational
    source: str                      # always "optics"
    metadata_status: str             # AGREES | ABSENT
    metadata_value_um: Optional[float]
    metadata_source: Optional[str]
    n_files_checked: int
    policy: str
    per_file: List[Dict[str, Any]] = field(default_factory=list)


def resolve_domain_spacing(files: Sequence[Path], acq: "optics.Acquisition", *, policy: str,
                           rel_tol: Any = optics.DEFAULT_REL_TOL) -> SpacingRecord:
    """
    Spacing of one domain = the exact optics value. TIFF headers are evidence:
    every file must have the expected shape; every plausible statement found in
    every file must agree with the optics within ``rel_tol``; statements must
    agree between files; anisotropic pixels fail. With policy 'required', a
    domain with no usable statement fails. There is no fallback value.
    """
    if policy not in optics.POLICIES:
        raise optics.OpticsError(f"unknown metadata policy {policy!r}")
    if not files:
        raise optics.OpticsError(f"{acq.domain}: no files given for the metadata check")
    tol = optics.exact(rel_tol, "rel_tol") if not isinstance(rel_tol, float) else optics.exact(repr(rel_tol))
    per_file = [read_spacing_candidates(Path(f)) for f in files]
    for pf in per_file:
        if pf["shape"][:2] != [acq.n_px, acq.n_px] and pf["shape"][-2:] != [acq.n_px, acq.n_px]:
            raise optics.OpticsError(f"{acq.domain}: {pf['file']} has shape {pf['shape']}, optics n_px = {acq.n_px}")
        if pf["anisotropic"]:
            raise optics.OpticsError(f"{acq.domain}: {pf['file']} has anisotropic pixels {pf['anisotropic']}")
    checks: List[Dict[str, Any]] = []
    for pf in per_file:
        for key, v in pf["candidates"].items():
            rec = optics.compare_metadata(acq, v, key, tol)
            rec["file"] = pf["file"]
            checks.append(rec)
            if not rec["agrees"]:
                raise optics.OpticsError(rec["message"] + f" [{pf['file']}]")
    status, mval, msrc = "ABSENT", None, None
    for key in SPACING_KEYS:
        vals = [pf["candidates"].get(key) for pf in per_file]
        present = [v for v in vals if v is not None]
        if not present:
            continue
        if len(present) != len(vals):
            raise optics.OpticsError(f"{acq.domain}: {key} present in {len(present)} of {len(vals)} files: "
                                     "metadata is inconsistent across the sampled files")
        v0 = float(present[0])
        if any(abs(float(v) - v0) > 1e-6 * v0 for v in present):
            raise optics.OpticsError(f"{acq.domain}: {key} disagrees between files: {sorted(set(present))}")
        if status == "ABSENT":
            status, mval, msrc = "AGREES", v0, key
    if status == "ABSENT" and policy == "required":
        raise optics.OpticsError(f"{acq.domain}: no usable pixel-size metadata in {len(per_file)} file(s) and "
                                 "metadata_policy is 'required' (rejected statements: "
                                 f"{[pf['rejected'] for pf in per_file if pf['rejected']]})")
    return SpacingRecord(acq.domain, float(acq.pixel_um), optics.fstr(acq.pixel_um), "optics", status, mval, msrc,
                         len(per_file), policy, per_file)


def external_matched_geometry(acq_i: "optics.Acquisition", acq_e: "optics.Acquisition",
                              sp_int: SpacingRecord, sp_ext: SpacingRecord) -> Dict[str, Any]:
    """Exact T*, T, m_ext and residual (optics.external_matched) plus the metadata status of both domains.
    Fails when the residual at 384 x 384 is not below 0.1%."""
    if (acq_i.n_px, acq_e.n_px) != (N_INT, N_EXT):
        raise optics.OpticsError(f"optics n_px ({acq_i.n_px}, {acq_e.n_px}) != frame contracts ({N_INT}, {N_EXT})")
    out = optics.external_matched(acq_i, acq_e, D)
    out["p_int_source"] = out["p_ext_source"] = "optics"
    out["metadata_status"] = {"internal": sp_int.metadata_status, "external": sp_ext.metadata_status}
    out["metadata_value_um"] = {"internal": sp_int.metadata_value_um, "external": sp_ext.metadata_value_um}
    out["metadata_source"] = {"internal": sp_int.metadata_source, "external": sp_ext.metadata_source}
    if not out["residual_within_0.1pct"]:
        raise optics.OpticsError(f"tile/external spacing residual {out['relative_residual']:+.4%} is not below 0.1%")
    return out


# -----------------------------------------------------------------------------
# Tile grids
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Tile:
    index: int
    y0: int
    x0: int
    T: int
    top_on_frame: bool
    bottom_on_frame: bool
    left_on_frame: bool
    right_on_frame: bool


@dataclass(frozen=True)
class Level:
    m: float                         # physical scale relative to the internal full frame: n_src / T (derived)
    T: int                           # tile side in internal raw pixels: the identity of the level
    n_per_axis: int
    stride: float
    offsets: Tuple[int, ...]
    tiles: Tuple[Tile, ...]
    frame_edge_share_of_tile_sides: float
    role: str = ROLE_NOMINAL

    @property
    def n_tiles(self) -> int:
        return len(self.tiles)

    @property
    def m_exact(self) -> "optics.Fraction":
        return optics.Fraction(N_INT, self.T)

    @property
    def label(self) -> str:
        tag = {ROLE_EXTERNAL: " (external-matched)", ROLE_SUPPLEMENTARY: " (supplementary)"}.get(self.role, "")
        return f"m = {self.m:.4f}{tag}"


def tile_size_for_m(m: float) -> int:
    """Tile side of a nominal level m (exact rational rounding of 1200 / m)."""
    return optics.round_half_up(optics.Fraction(N_INT) / optics.exact(repr(float(m))))


def make_level(T: int, role: str = ROLE_NOMINAL, n_src: int = N_INT) -> Level:
    """Level of tile side T: ceil(n_src / T) tiles per axis at uniform stride, first at 0, last ending at n_src."""
    if isinstance(T, bool) or not isinstance(T, (int, np.integer)):
        raise TypeError(f"make_level takes the integer tile side T, got {T!r}")
    T = int(T)
    m = n_src / T
    if role not in (ROLE_NOMINAL, ROLE_EXTERNAL, ROLE_SUPPLEMENTARY):
        raise ValueError(f"unknown level role {role!r}")
    if not (D <= T <= n_src):
        raise ValueError(f"Tile size {T} outside [{D}, {n_src}]")
    n = math.ceil(n_src / T)
    if n == 1:
        offsets: Tuple[int, ...] = (0,)
        stride = 0.0
    else:
        stride = (n_src - T) / (n - 1)
        offsets = tuple(round_half_up(k * stride) for k in range(n))
    if offsets[0] != 0 or offsets[-1] + T != n_src:
        raise CoordinateError(f"T={T}: offsets {offsets} do not span [0, {n_src})")
    covered = np.zeros(n_src, dtype=bool)
    for o in offsets:
        covered[o:o + T] = True
    if not covered.all():
        raise CoordinateError(f"T={T}: offsets {offsets} leave uncovered pixels")
    tiles = []
    for iy, y0 in enumerate(offsets):
        for ix, x0 in enumerate(offsets):
            tiles.append(Tile(index=iy * n + ix, y0=y0, x0=x0, T=T,
                              top_on_frame=y0 == 0, bottom_on_frame=y0 + T == n_src,
                              left_on_frame=x0 == 0, right_on_frame=x0 + T == n_src))
    sides = 4 * len(tiles)
    on_edge = sum(int(t.top_on_frame) + int(t.bottom_on_frame) + int(t.left_on_frame) + int(t.right_on_frame)
                  for t in tiles)
    return Level(m=float(m), T=T, n_per_axis=n, stride=stride, offsets=offsets, tiles=tuple(tiles),
                 frame_edge_share_of_tile_sides=on_edge / sides, role=role)


def build_levels(T_external: int, supplementary: Sequence[int] = ()) -> List[Level]:
    """
    Nominal levels (NOMINAL_M), the external-matched level T_external, and optional
    supplementary levels, sorted by T strictly decreasing (m strictly increasing).
    A T shared by a nominal and the external-matched level is one level with the
    external-matched role (never duplicated). Raises on any ordering violation.
    """
    if T_external is None:
        raise ValueError("T_external is required (geometry_spec.json external_matched_T_used)")
    roles: Dict[int, str] = {tile_size_for_m(m): ROLE_NOMINAL for m in NOMINAL_M}
    for t in supplementary:
        if int(t) in roles:
            raise ValueError(f"supplementary T={t} duplicates a nominal level")
        roles[int(t)] = ROLE_SUPPLEMENTARY
    if roles.get(int(T_external)) == ROLE_SUPPLEMENTARY:
        raise ValueError(f"supplementary T={T_external} equals the external-matched T")
    roles[int(T_external)] = ROLE_EXTERNAL
    levels = [make_level(t, role=r) for t, r in sorted(roles.items(), key=lambda kv: -kv[0])]
    assert_monotonic(levels)
    return levels


def assert_monotonic(levels: Sequence[Level]) -> None:
    Ts = [lv.T for lv in levels]
    ms = [lv.m for lv in levels]
    if len(set(Ts)) != len(Ts) or any(a <= b for a, b in zip(Ts, Ts[1:])) or any(a >= b for a, b in zip(ms, ms[1:])):
        raise ValueError(f"levels not strictly ordered: T {Ts}, m {ms}")
    if sum(lv.role == ROLE_EXTERNAL for lv in levels) != 1:
        raise ValueError("exactly one external-matched level is required")


def levels_from_spec(spec: Dict[str, Any]) -> List[Level]:
    """Levels of a geometry_spec.json (schema 2.0 only)."""
    if spec.get("schema") != SPEC_SCHEMA:
        raise ValueError(f"geometry spec schema {spec.get('schema')!r} != {SPEC_SCHEMA!r}: legacy spec, "
                         "rebuild it (run_all_colab.py migrate_snapshot then geometry)")
    return build_levels(int(spec["external_matched_T_used"]), spec.get("supplementary_T", []))


def external_level(levels: Sequence[Level]) -> Level:
    return next(lv for lv in levels if lv.role == ROLE_EXTERNAL)


def level_table(levels: Sequence[Level]) -> List[List[Any]]:
    return [[lv.T, lv.role, list(lv.offsets)] for lv in levels]


def level_summary(level: Level, p_int_um: float) -> Dict[str, Any]:
    return {"T": level.T, "m": level.m, "m_exact": optics.fstr(level.m_exact), "role": level.role,
            "label": level.label, "n_per_axis": level.n_per_axis, "tiles_per_field": level.n_tiles,
            "stride": level.stride, "stride_is_integer": float(level.stride).is_integer(),
            "offsets": list(level.offsets), "fov_um": level.T * p_int_um,
            "output_spacing_um": level.T * p_int_um / D,
            "frame_edge_share_of_tile_sides": level.frame_edge_share_of_tile_sides}


# -----------------------------------------------------------------------------
# OpenCV-exact Lanczos-4 resampling
# -----------------------------------------------------------------------------

_S45 = 0.70710678118654752440084436210485
_CS = ((1.0, 0.0), (-_S45, -_S45), (0.0, 1.0), (_S45, -_S45),
       (-1.0, 0.0), (_S45, _S45), (0.0, -1.0), (-_S45, _S45))


def opencv_lanczos4_coeffs(frac: float) -> Tuple[np.ndarray, bool]:
    """
    OpenCV ``interpolateLanczos4(float x, float* coeffs)``: double-precision
    trigonometry, float32 coefficients, float32 running sum, multiplication by
    float32(1 / sum). Returns (coeffs[8] float32, guard_used).
    """
    x = np.float32(frac)
    if float(x) < FLT_EPSILON:
        c = np.zeros(8, dtype=np.float32)
        c[3] = np.float32(1.0)
        return c, True
    xd = float(x)
    y0 = -(xd + 3.0) * math.pi * 0.25
    s0, c0 = math.sin(y0), math.cos(y0)
    coeffs = np.empty(8, dtype=np.float32)
    total = np.float32(0.0)
    for i in range(8):
        y = -(xd + 3.0 - i) * math.pi * 0.25
        coeffs[i] = np.float32((_CS[i][0] * s0 + _CS[i][1] * c0) / (y * y))
        total = np.float32(total + coeffs[i])
    inv = np.float32(np.float32(1.0) / total)
    return (coeffs * inv).astype(np.float32), False


@dataclass(frozen=True)
class AxisPlan:
    """Sampling plan for one axis of a T -> D resize (indices relative to the tile start)."""
    T: int
    n_out: int
    idx: np.ndarray          # (n_out, 8) int64
    w: np.ndarray            # (n_out, 8) float32
    tap_min: int             # most negative tap index (relative to tile start)
    tap_max_over: int        # largest tap index minus (T - 1)
    n_guard: int             # samples whose fractional part fell below FLT_EPSILON


_PLAN_CACHE: Dict[Tuple[int, int], AxisPlan] = {}


def make_axis_plan(T: int, n_out: int = D) -> AxisPlan:
    """OpenCV resize geometry: inv = n_out/T, scale = 1/inv, fx = float32((j+0.5)*scale - 0.5)."""
    key = (int(T), int(n_out))
    if key in _PLAN_CACHE:
        return _PLAN_CACHE[key]
    inv_scale = float(n_out) / float(T)
    scale = 1.0 / inv_scale
    idx = np.empty((n_out, 8), dtype=np.int64)
    w = np.empty((n_out, 8), dtype=np.float32)
    n_guard = 0
    for j in range(n_out):
        fx = np.float32((j + 0.5) * scale - 0.5)
        sx = int(math.floor(float(fx)))
        frac = np.float32(fx - np.float32(sx))
        c, guard = opencv_lanczos4_coeffs(float(frac))
        n_guard += int(guard)
        idx[j] = sx - 3 + np.arange(8)
        w[j] = c
    plan = AxisPlan(T=int(T), n_out=int(n_out), idx=idx, w=w, tap_min=int(idx.min()),
                    tap_max_over=int(idx.max() - (T - 1)), n_guard=n_guard)
    _PLAN_CACHE[key] = plan
    return plan


def _border_index(k: np.ndarray, n: int, rule: str) -> np.ndarray:
    if rule == "replicate":
        return np.clip(k, 0, n - 1)
    if rule == "reflect101":
        k = np.where(k < 0, -k, k)
        k = np.where(k > n - 1, 2 * (n - 1) - k, k)
        return np.clip(k, 0, n - 1)
    raise ValueError(f"Unknown border rule {rule!r}")


def _axis_indices(plan: AxisPlan, offset_in_cut: int, cut_len: int, lo_on_frame: bool, hi_on_frame: bool,
                  rule: str, axis: str) -> Tuple[np.ndarray, Dict[str, int]]:
    k = plan.idx + int(offset_in_cut)
    below = k < 0
    above = k >= cut_len
    if below.any() and not lo_on_frame:
        raise CoordinateError(f"{axis}: {int(below.sum())} taps below an interior cut edge "
                              f"(min tap {int(k.min())}); margin too small")
    if above.any() and not hi_on_frame:
        raise CoordinateError(f"{axis}: {int(above.sum())} taps above an interior cut edge "
                              f"(max tap {int(k.max())} >= {cut_len}); margin too small")
    resolved = _border_index(k, cut_len, rule)
    return resolved, {"taps_border_rule_low": int(below.sum()), "taps_border_rule_high": int(above.sum()),
                      "min_tap_in_cut": int(k.min()), "max_tap_in_cut": int(k.max())}


def _separable_apply(cut: np.ndarray, iy: np.ndarray, wy: np.ndarray, ix: np.ndarray, wx: np.ndarray) -> np.ndarray:
    """Horizontal then vertical pass, sequential float32 accumulation over the 8 taps (OpenCV order)."""
    src = cut.astype(np.float32, copy=False)
    tmp = np.zeros((src.shape[0], ix.shape[0]), dtype=np.float32)
    for i in range(8):
        tmp += src[:, ix[:, i]] * wx[None, :, i]
    out = tmp[iy[:, 0], :] * wy[:, 0][:, None]
    for i in range(1, 8):
        out += tmp[iy[:, i], :] * wy[:, i][:, None]
    return out.astype(np.float32, copy=False)


@dataclass
class CutTile:
    array: np.ndarray
    cy0: int
    cy1: int
    cx0: int
    cx1: int
    off_y: int
    off_x: int


def cut_with_margin(frame: np.ndarray, tile: Tile, margin: int = MARGIN_PX) -> CutTile:
    """Tile plus ``margin`` raw pixels on every interior side; frame-boundary sides get none (no padding ever)."""
    H, W = frame.shape
    cy0 = tile.y0 - (0 if tile.top_on_frame else margin)
    cy1 = tile.y0 + tile.T + (0 if tile.bottom_on_frame else margin)
    cx0 = tile.x0 - (0 if tile.left_on_frame else margin)
    cx1 = tile.x0 + tile.T + (0 if tile.right_on_frame else margin)
    if cy0 < 0 or cx0 < 0 or cy1 > H or cx1 > W:
        raise CoordinateError(f"Cut [{cy0}:{cy1}, {cx0}:{cx1}] leaves the {H}x{W} frame")
    return CutTile(array=frame[cy0:cy1, cx0:cx1], cy0=cy0, cy1=cy1, cx0=cx0, cx1=cx1,
                   off_y=tile.y0 - cy0, off_x=tile.x0 - cx0)


def resample_tile(frame: np.ndarray, tile: Tile, *, border_rule: str, margin: int = MARGIN_PX,
                  engine: str = "auto") -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Margin-aware cut -> OpenCV-exact Lanczos-4 -> 384 x 384 output (the trim is
    the restriction to the 384 target samples). ``engine='auto'`` uses
    cv2.resize for a full-frame tile (the original call) and the exact
    reimplementation otherwise; ``engine='numpy'`` forces the reimplementation.
    """
    import cv2

    H, W = frame.shape
    if engine not in ("auto", "numpy"):
        raise ValueError(f"engine must be 'auto' or 'numpy', got {engine!r}")
    full_frame = tile.T == H == W and tile.y0 == 0 and tile.x0 == 0
    if full_frame and engine == "auto":
        out = cv2.resize(np.ascontiguousarray(frame, dtype=np.float32), (D, D),
                         interpolation=cv2.INTER_LANCZOS4).astype(np.float32, copy=False)
        return out, {"tile": asdict(tile), "engine": "cv2.resize(INTER_LANCZOS4)", "cut": [0, H, 0, W],
                     "border_rule": "opencv_internal"}
    cut = cut_with_margin(frame, tile, margin)
    plan = make_axis_plan(tile.T)
    iy, info_y = _axis_indices(plan, cut.off_y, cut.array.shape[0], tile.top_on_frame, tile.bottom_on_frame,
                               border_rule, "y")
    ix, info_x = _axis_indices(plan, cut.off_x, cut.array.shape[1], tile.left_on_frame, tile.right_on_frame,
                               border_rule, "x")
    out = _separable_apply(cut.array, iy, plan.w, ix, plan.w)
    if out.shape != (D, D):
        raise CoordinateError(f"Output shape {out.shape} != {(D, D)}")
    return out, {"tile": asdict(tile), "engine": "opencv_exact_numpy", "cut": [cut.cy0, cut.cy1, cut.cx0, cut.cx1],
                 "border_rule": border_rule, "y": info_y, "x": info_x, "plan_guard_hits": plan.n_guard}


def extract_level_tiles(frame: np.ndarray, level: Level, *, border_rule: str, margin: int = MARGIN_PX,
                        engine: str = "auto") -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    if frame.shape != (N_INT, N_INT):
        raise ValueError(f"Internal frame must be {N_INT}x{N_INT}, got {frame.shape}")
    outs, prov = [], []
    for t in level.tiles:
        o, p = resample_tile(frame, t, border_rule=border_rule, margin=margin, engine=engine)
        outs.append(o)
        prov.append(p)
    return np.stack(outs).astype(np.float32, copy=False), prov


# -----------------------------------------------------------------------------
# Run-time identification and parity checks against cv2.resize
# -----------------------------------------------------------------------------

def _resize_whole_with_engine(img: np.ndarray, border_rule: str) -> np.ndarray:
    """Engine applied to a whole array treated as a frame (all four sides on the boundary)."""
    n = img.shape[0]
    if img.shape[0] != img.shape[1]:
        raise ValueError("square input required")
    t = Tile(index=0, y0=0, x0=0, T=n, top_on_frame=True, bottom_on_frame=True, left_on_frame=True,
             right_on_frame=True)
    out, _ = resample_tile(img, t, border_rule=border_rule, margin=0, engine="numpy")
    return out


def identify_cv2_border_rule(frame: np.ndarray, tol: float = 1e-5) -> Dict[str, Any]:
    """Which boundary rule makes the engine equal cv2.resize on a full frame."""
    import cv2

    ref = cv2.resize(np.ascontiguousarray(frame, dtype=np.float32), (D, D), interpolation=cv2.INTER_LANCZOS4)
    diffs = {r: float(np.max(np.abs(_resize_whole_with_engine(frame, r).astype(np.float64) - ref)))
             for r in BORDER_RULES}
    best = min(diffs, key=diffs.get)
    return {"max_abs_diff": diffs, "selected": best, "tolerance": tol, "PASS": diffs[best] <= tol,
            "bitwise_equal_selected": diffs[best] == 0.0}


def crop_parity_vs_cv2(frame: np.ndarray, T: int, y0: int, x0: int, border_rule: str) -> float:
    """Engine on an isolated crop (all sides treated as boundaries) versus cv2.resize of the same crop."""
    import cv2

    crop = np.ascontiguousarray(frame[y0:y0 + T, x0:x0 + T], dtype=np.float32)
    ref = cv2.resize(crop, (D, D), interpolation=cv2.INTER_LANCZOS4)
    return float(np.max(np.abs(_resize_whole_with_engine(crop, border_rule).astype(np.float64) - ref)))


def naive_vs_margin_difference(frame: np.ndarray, level: Level, tile_index: int, border_rule: str) -> Dict[str, float]:
    """Diagnostic: interior-edge error of a naive crop-then-resize relative to the margin-aware tile."""
    import cv2

    t = level.tiles[tile_index]
    good, _ = resample_tile(frame, t, border_rule=border_rule)
    crop = np.ascontiguousarray(frame[t.y0:t.y0 + t.T, t.x0:t.x0 + t.T], dtype=np.float32)
    naive = cv2.resize(crop, (D, D), interpolation=cv2.INTER_LANCZOS4)
    band = np.zeros((D, D), dtype=bool)
    band[:4, :] = band[-4:, :] = band[:, :4] = band[:, -4:] = True
    d = np.abs(good.astype(np.float64) - naive)
    return {"max_abs_diff_outer_4px": float(d[band].max()), "max_abs_diff_interior": float(d[~band].max())}


def opencv_provenance() -> Dict[str, Any]:
    import cv2

    return {"cv2_version": cv2.__version__, "numpy_version": np.__version__,
            "cv2_build_information_sha256": hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest(),
            "resize_call": "cv2.resize(float32, (384, 384), interpolation=cv2.INTER_LANCZOS4)",
            "internal_pipeline_source": "simpleViT_fungi data/preprocessing.py::resize_and_preprocess_image"}


def assert_opencv_version(expected: Optional[str]) -> Dict[str, Any]:
    prov = opencv_provenance()
    if expected is not None and prov["cv2_version"] != expected:
        raise RuntimeError(f"OpenCV {prov['cv2_version']} != required {expected}")
    prov["required_version"] = expected
    return prov


# -----------------------------------------------------------------------------
# CLI: geometry specification
# -----------------------------------------------------------------------------

def expected_geometry(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Geometry implied by a config alone (no files): optics, T_ext, levels and the geometry digest.
    The run driver compares an existing spec against this before reusing it."""
    acq_i, acq_e, policy = optics.load_optics(cfg)
    if (acq_i.n_px, acq_e.n_px) != (N_INT, N_EXT):
        raise optics.OpticsError(f"optics n_px ({acq_i.n_px}, {acq_e.n_px}) != frame contracts ({N_INT}, {N_EXT})")
    em = optics.external_matched(acq_i, acq_e, D)
    supp = [int(t) for t in (cfg.get("geometry", {}) or {}).get("supplementary_T", []) or []]
    levels = build_levels(em["T"], supp)
    dg = optics.geometry_digest(acq_i, acq_e, em["T"], level_table(levels), d=D, margin=MARGIN_PX)
    return {"acq_i": acq_i, "acq_e": acq_e, "policy": policy, "external_matched": em, "levels": levels,
            "supplementary_T": supp, "geometry_digest": dg}


def build_geometry_spec(internal_files: Sequence[Path], external_files: Sequence[Path], *, cfg: Dict[str, Any],
                        expected_opencv: Optional[str], record_opencv: bool = True) -> Dict[str, Any]:
    exp = expected_geometry(cfg)
    acq_i, acq_e, policy = exp["acq_i"], exp["acq_e"], exp["policy"]
    sp_i = resolve_domain_spacing(internal_files, acq_i, policy=policy["internal"], rel_tol=policy["rel_tol"])
    sp_e = resolve_domain_spacing(external_files, acq_e, policy=policy["external"], rel_tol=policy["rel_tol"])
    em = external_matched_geometry(acq_i, acq_e, sp_i, sp_e)
    levels = exp["levels"]
    return {
        "schema": SPEC_SCHEMA,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "module": "geometry_harmonization.py",
        "optics": {"internal": acq_i.to_json(), "external": acq_e.to_json()},
        "metadata_policy": policy,
        "frame": {"N_int": N_INT, "N_ext": N_EXT, "D": D, "margin_px": MARGIN_PX},
        "spacing": {"internal": asdict(sp_i), "external": asdict(sp_e)},
        "external_matched": em,
        "levels": [level_summary(lv, sp_i.um_per_px) for lv in levels],
        "external_matched_T_used": external_level(levels).T,
        "supplementary_T": exp["supplementary_T"],
        "geometry_digest": exp["geometry_digest"],
        "opencv": assert_opencv_version(expected_opencv) if record_opencv else {"recorded": False},
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Geometry specification for Protocol Part A (exact optics, "
                                            "metadata-verified; no fallback spacing exists).")
    p.add_argument("--config", type=Path, required=True, help="colab_config.json with 'optics' and 'metadata_policy'")
    p.add_argument("--internal-files", nargs="+", type=Path, required=True)
    p.add_argument("--external-files", nargs="+", type=Path, required=True)
    p.add_argument("--expected-opencv", default=None, help="Fail unless cv2.__version__ equals this")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    cfg = json.loads(a.config.read_text())
    spec = build_geometry_spec(a.internal_files, a.external_files, cfg=cfg, expected_opencv=a.expected_opencv)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_name(a.out.name + ".tmp")
    tmp.write_text(json.dumps(spec, indent=2, default=str))
    tmp.replace(a.out)
    g = spec["external_matched"]
    print(f"p_int = {g['p_int_exact']} = {g['p_int_um']:.6f} um/px (FOV {g['fov_int_um']:.2f} um); "
          f"p_ext = {g['p_ext_exact']} = {g['p_ext_um']:.6f} um/px (FOV {g['fov_ext_um']:.2f} um)")
    print(f"metadata: {g['metadata_status']}; T* = {g['T_star']:.3f} -> T = {g['T']}; m_ext = {g['m_external']:.5f}; "
          f"residual {100 * g['relative_residual']:+.4f}%")
    for lv in spec["levels"]:
        print(f"  T={lv['T']:<5} {lv['label']:<32} n={lv['n_per_axis']} stride={lv['stride']:<6} "
              f"spacing={lv['output_spacing_um']:.5f} um/px FOV={lv['fov_um']:.1f} um")
    print(f"geometry_digest {spec['geometry_digest'][:16]}; wrote {a.out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
