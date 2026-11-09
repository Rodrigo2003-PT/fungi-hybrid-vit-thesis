#!/usr/bin/env python3
"""
optics.py

Single source of truth for the acquisition geometry of Parts A and B.

Object-plane pixel spacing is derived exactly from explicit physical parameters,
    p = sensor_pitch_um * binning / total_magnification          (um/px)
with every quantity held as a ``fractions.Fraction`` built from decimal strings
(never from binary floats). TIFF metadata is evidence that is checked against
this value; it never replaces it, and there is no rounded fallback.

Reference values (colab_config.json):
    internal  Photometrics Prime 95B, 11.0 um pitch, 1 x 1, 63x, 1200 px
              p_int = 11/63 = 0.174603 um/px, FOV 209.524 um
    external  ZEISS Axiocam 820 mono, 2.74 um pitch, 2 x 2 binning (4512 -> 2256), 63x
              p_ext = 5.48/63 = 137/1575 = 0.086984 um/px, FOV 196.236 um
    T* = N_ext p_ext / p_int = 1123.898 -> T = 1124; m_ext = N_int / T* = 1.06771

Author: Rodrigo Sá
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

OPTICS_SCHEMA = "optics/1.0"
D = 384                                            # network input side (px)
PLAUSIBLE_UM_PER_PX = (Fraction(1, 50), Fraction(2))   # [0.02, 2.0] um/px at the object plane
DEFAULT_REL_TOL = Fraction(5, 1000)                # metadata vs optics agreement (0.5%)
SPACING_TOLERANCE = Fraction(1, 1000)              # Protocol.md: tile vs external spacing below 0.1%
POLICIES = ("required", "verify_if_present")
DOMAINS = ("internal", "external")


class OpticsError(ValueError):
    """Inconsistent, missing or implausible acquisition geometry."""


def exact(x: Any, what: str = "value") -> Fraction:
    """Fraction from a decimal string, int or Fraction. Binary floats are refused (they are not exact)."""
    if isinstance(x, bool):
        raise OpticsError(f"{what}: boolean is not a number")
    if isinstance(x, Fraction):
        return x
    if isinstance(x, int):
        return Fraction(x)
    if isinstance(x, str):
        try:
            return Fraction(x.strip())
        except (ValueError, ZeroDivisionError) as exc:
            raise OpticsError(f"{what}: cannot parse {x!r} as an exact decimal") from exc
    raise OpticsError(f"{what}: give an exact decimal string (e.g. \"2.74\"), not {type(x).__name__} {x!r}")


def round_half_up(q: Fraction) -> int:
    """Exact round-half-up of a rational (no binary rounding)."""
    return math.floor(q + Fraction(1, 2))


def fstr(q: Fraction) -> str:
    return f"{q.numerator}/{q.denominator}" if q.denominator != 1 else str(q.numerator)


@dataclass(frozen=True)
class Acquisition:
    domain: str
    camera: str
    sensor_pitch_um: Fraction
    binning: int
    total_magnification: Fraction
    n_px: int
    native_n_px: Optional[int] = None

    @classmethod
    def from_config(cls, domain: str, d: Mapping[str, Any]) -> "Acquisition":
        if domain not in DOMAINS:
            raise OpticsError(f"unknown domain {domain!r}")
        need = {"camera", "sensor_pitch_um", "binning", "total_magnification", "n_px"}
        miss = sorted(need - set(d))
        if miss:
            raise OpticsError(f"optics.{domain}: missing {miss}")
        extra = sorted(set(d) - need - {"native_n_px"})
        if extra:
            raise OpticsError(f"optics.{domain}: unknown keys {extra}")
        for k in ("binning", "n_px", "native_n_px"):
            v = d.get(k)
            if v is not None and (isinstance(v, bool) or not isinstance(v, int)):
                raise OpticsError(f"optics.{domain}.{k} must be an integer, got {v!r}")
        acq = cls(domain=domain, camera=str(d["camera"]),
                  sensor_pitch_um=exact(d["sensor_pitch_um"], f"optics.{domain}.sensor_pitch_um"),
                  binning=int(d["binning"]),
                  total_magnification=exact(d["total_magnification"], f"optics.{domain}.total_magnification"),
                  n_px=int(d["n_px"]), native_n_px=d.get("native_n_px"))
        acq.validate()
        return acq

    def validate(self) -> None:
        if self.sensor_pitch_um <= 0 or self.total_magnification <= 0 or self.binning < 1 or self.n_px < 1:
            raise OpticsError(f"{self.domain}: pitch, magnification, binning and n_px must be positive ({self})")
        if self.native_n_px is not None:
            if self.native_n_px % self.binning or self.native_n_px // self.binning != self.n_px:
                raise OpticsError(f"{self.domain}: native {self.native_n_px} px / binning {self.binning} "
                                  f"!= acquired {self.n_px} px")
        lo, hi = PLAUSIBLE_UM_PER_PX
        if not lo <= self.pixel_um <= hi:
            raise OpticsError(f"{self.domain}: object-plane spacing {float(self.pixel_um):.6g} um/px outside "
                              f"[{float(lo)}, {float(hi)}]")

    @property
    def pixel_um(self) -> Fraction:
        return self.sensor_pitch_um * self.binning / self.total_magnification

    @property
    def fov_um(self) -> Fraction:
        return self.n_px * self.pixel_um

    def to_json(self) -> Dict[str, Any]:
        return {"domain": self.domain, "camera": self.camera, "sensor_pitch_um": fstr(self.sensor_pitch_um),
                "binning": self.binning, "total_magnification": fstr(self.total_magnification),
                "n_px": self.n_px, "native_n_px": self.native_n_px,
                "pixel_um_exact": fstr(self.pixel_um), "pixel_um": float(self.pixel_um),
                "fov_um_exact": fstr(self.fov_um), "fov_um": float(self.fov_um)}


def load_optics(cfg: Mapping[str, Any]) -> Tuple[Acquisition, Acquisition, Dict[str, Any]]:
    """(internal, external, metadata policy) from a colab_config.json dictionary. Fails on anything missing."""
    o = cfg.get("optics")
    if not isinstance(o, Mapping):
        raise OpticsError("config has no 'optics' block: the geometry is not defined (no fallback exists)")
    acq_i = Acquisition.from_config("internal", o.get("internal") or {})
    acq_e = Acquisition.from_config("external", o.get("external") or {})
    return acq_i, acq_e, load_policy(cfg.get("metadata_policy"))


def load_policy(p: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not isinstance(p, Mapping):
        raise OpticsError("config has no 'metadata_policy' block")
    out: Dict[str, Any] = {}
    for d in DOMAINS:
        v = p.get(d)
        if v not in POLICIES:
            raise OpticsError(f"metadata_policy.{d} must be one of {POLICIES}, got {v!r}")
        out[d] = v
    out["rel_tol"] = fstr(exact(p.get("rel_tol", "0.005"), "metadata_policy.rel_tol"))
    if not 0 < exact(out["rel_tol"]) < Fraction(1, 10):
        raise OpticsError("metadata_policy.rel_tol must lie in (0, 0.1)")
    return out


def compare_metadata(acq: Acquisition, value_um: float, source: str,
                     rel_tol: Fraction = DEFAULT_REL_TOL) -> Dict[str, Any]:
    """One metadata statement against the derived spacing. A ratio near 1/2 or 2 is named as a binning mismatch."""
    p = float(acq.pixel_um)
    ratio = float(value_um) / p
    agrees = abs(ratio - 1.0) <= float(rel_tol)
    rec = {"source": source, "metadata_um": float(value_um), "derived_um": p, "ratio_metadata_over_derived": ratio,
           "agrees": bool(agrees)}
    if not agrees:
        hint = ""
        for k in (2, 3, 4):
            if abs(ratio * k - 1) < 0.02:
                hint = f"; ratio ~ 1/{k}: the metadata looks like the unbinned pitch (binning factor {k} missing)"
            elif abs(ratio / k - 1) < 0.02:
                hint = f"; ratio ~ {k}: binning factor {k} may be applied twice or the optics binning is wrong"
        rec["message"] = (f"{acq.domain}: {source} = {value_um:.6g} um/px disagrees with optics "
                          f"{p:.6g} um/px (ratio {ratio:.4f}, tolerance {float(rel_tol):.2%}){hint}")
    return rec


def external_matched(acq_i: Acquisition, acq_e: Acquisition, d: int = D) -> Dict[str, Any]:
    """T* = N_ext p_ext / p_int, T = round_half_up(T*), and the exact residual at d x d."""
    p_i, p_e = acq_i.pixel_um, acq_e.pixel_um
    t_star = acq_e.n_px * p_e / p_i
    T = round_half_up(t_star)
    if not d <= T <= acq_i.n_px:
        raise OpticsError(f"external-matched tile {T} px outside [{d}, {acq_i.n_px}]: the domains cannot be "
                          "matched by cropping the internal frame")
    sp_tile = T * p_i / d
    sp_ext = acq_e.n_px * p_e / d
    resid = (sp_tile - sp_ext) / sp_ext
    return {"p_int_um": float(p_i), "p_ext_um": float(p_e), "p_int_exact": fstr(p_i), "p_ext_exact": fstr(p_e),
            "fov_int_um": float(acq_i.fov_um), "fov_ext_um": float(acq_e.fov_um),
            "T_star": float(t_star), "T_star_exact": fstr(t_star), "T": T,
            "m_external": float(acq_i.n_px / t_star), "m_external_exact": fstr(acq_i.n_px / t_star),
            "m_tile": float(Fraction(acq_i.n_px, T)),
            "spacing_tile_at_384_um": float(sp_tile), "spacing_external_at_384_um": float(sp_ext),
            "spacing_internal_full_frame_at_384_um": float(acq_i.n_px * p_i / d),
            "relative_residual": float(resid), "relative_residual_exact": fstr(resid),
            "residual_within_0.1pct": bool(abs(resid) < SPACING_TOLERANCE)}


def canonical_digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def geometry_digest(acq_i: Acquisition, acq_e: Acquisition, T_ext: int, level_table: Sequence[Sequence[Any]],
                    *, d: int = D, margin: int) -> str:
    """Digest of everything that fixes a resampled pixel: optics, matched T, the level table, D and the margin.
    Creation time, metadata evidence and library versions are deliberately excluded."""
    return canonical_digest({"schema": "geometry_digest/1", "optics": [acq_i.to_json(), acq_e.to_json()],
                             "T_external": int(T_ext), "levels": [list(x) for x in level_table],
                             "D": int(d), "margin_px": int(margin)})
