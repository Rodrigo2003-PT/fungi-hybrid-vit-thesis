#!/usr/bin/env python3
"""
optics_provenance.py

Single source of truth for optical acquisition geometry in the external
validation pipeline.

Scientific role
---------------
Only physical primitives are stored by hand (sensor pixel pitch, native sensor
size, binning factor, objective and camera-adapter magnification, stored image
size, header-recorded pixel spacing). Every derived quantity - pixel spacing,
field of view, network-input spacing, decimation factor, cross-domain ratios -
is computed here and cross-checked against the TIFF header calibration. No
derived ratio may be stored in configuration or metadata files, because stored
ratios can silently diverge from the primitives they summarise.

Geometry
--------
    pixel spacing         dx    = pitch * binning / (M_objective * M_adapter)
    field of view         FOV   = N_stored * dx
    network-input spacing dx_in = FOV / N_input
    decimation factor     D     = N_stored / N_input

These quantities are *descriptive provenance*. They are never used to transform
pixels: both remaining preprocessing pipelines are pixel-locked resizes to the
frozen model input size.

Validation rules (IDs used throughout the pipeline)
---------------------------------------------------
    O-1  all primitives present, finite, positive; no AUTHOR_TO_* placeholders
    O-2  stored_image_px == sensor_native_px / binning_factor (per axis)
    O-3  |derived dx - header dx| / header dx <= tolerances.spacing_rel
    O-4  per-image TIFF calibration agrees with derived dx (audit stage)
    O-5  per-image TIFF shape equals stored_image_px (audit/preprocess stages)
    O-6  binning_mode == "unknown" is logged as a provenance gap

Author: Rodrigo Sá
Date: 2026
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

OPTICS_SCHEMA_VERSION = "2.0"
PLACEHOLDER_PREFIX = "AUTHOR_TO_"
DEFAULT_SPACING_REL_TOLERANCE = 0.005
VALID_BINNING_MODES = {
    "hardware_binning",
    "digital_sum",
    "digital_average",
    "subsampling",
    "none",
    "unknown",
}
DOMAIN_NAMES = ("external", "training")

_NUMERIC_KEYS = (
    "sensor_pixel_pitch_um",
    "objective_magnification",
    "camera_adapter_magnification",
)
_PAIR_KEYS = ("sensor_native_px", "stored_image_px")

# Length units in micrometres for TIFF/ImageJ/OME calibration parsing.
_UNIT_TO_UM = {
    "um": 1.0,
    "µm": 1.0,
    "μm": 1.0,
    "micron": 1.0,
    "microns": 1.0,
    "micrometer": 1.0,
    "micrometre": 1.0,
    "\\u00b5m": 1.0,
    "nm": 1e-3,
    "nanometer": 1e-3,
    "mm": 1e3,
    "millimeter": 1e3,
    "cm": 1e4,
    "centimeter": 1e4,
    "m": 1e6,
    "meter": 1e6,
    "inch": 25400.0,
    "in": 25400.0,
}


class OpticsError(ValueError):
    """Raised when optics metadata cannot be parsed into a usable domain."""


@dataclass(frozen=True)
class DomainOptics:
    """Physical acquisition primitives for one microscope domain."""

    name: str
    camera: str
    sensor_pixel_pitch_um: float
    sensor_native_px: Tuple[int, int]
    binning_factor: int
    binning_mode: str
    objective_magnification: float
    objective_na: Optional[float]
    camera_adapter_magnification: float
    stored_image_px: Tuple[int, int]
    header_pixel_spacing_um: Optional[float]
    header_source: str


@dataclass
class OpticsValidation:
    """Outcome of validate_optics. ``ok`` is False when any error exists."""

    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    pending_author_input: List[str] = field(default_factory=list)
    provenance_gaps: List[str] = field(default_factory=list)
    checks: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "pending_author_input": list(self.pending_author_input),
            "provenance_gaps": list(self.provenance_gaps),
            "checks": dict(self.checks),
        }


# -----------------------------------------------------------------------------
# Parsing
# -----------------------------------------------------------------------------

def is_placeholder(value: Any) -> bool:
    return isinstance(value, str) and value.strip().upper().startswith(PLACEHOLDER_PREFIX)


def _positive_float(value: Any, label: str) -> float:
    if is_placeholder(value):
        raise OpticsError(f"{label} is an unfilled placeholder: {value!r}")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise OpticsError(f"{label} must be numeric, got {value!r}") from exc
    if not math.isfinite(out) or out <= 0:
        raise OpticsError(f"{label} must be finite and > 0, got {value!r}")
    return out


def _positive_int_pair(value: Any, label: str) -> Tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise OpticsError(f"{label} must be a [height, width] pair, got {value!r}")
    pair = []
    for v in value:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or int(v) != v or int(v) <= 0:
            raise OpticsError(f"{label} must contain positive integers, got {value!r}")
        pair.append(int(v))
    return int(pair[0]), int(pair[1])


def parse_domain(name: str, raw: Mapping[str, Any]) -> Tuple[Optional[DomainOptics], List[str], List[str]]:
    """
    Parse one optics domain block.

    Returns (domain_or_None, errors, pending_author_input). A placeholder header
    spacing is reported as pending rather than as a parse error, so that the
    rest of the geometry can still be derived and inspected.
    """
    errors: List[str] = []
    pending: List[str] = []
    prefix = f"optics.{name}"

    if not isinstance(raw, Mapping):
        return None, [f"{prefix} must be an object."], pending

    values: Dict[str, Any] = {}
    for key in _NUMERIC_KEYS:
        try:
            values[key] = _positive_float(raw.get(key), f"{prefix}.{key}")
        except OpticsError as exc:
            (pending if is_placeholder(raw.get(key)) else errors).append(str(exc))

    for key in _PAIR_KEYS:
        try:
            values[key] = _positive_int_pair(raw.get(key), f"{prefix}.{key}")
        except OpticsError as exc:
            errors.append(str(exc))

    binning = raw.get("binning_factor")
    if isinstance(binning, bool) or not isinstance(binning, int) or binning < 1:
        errors.append(f"{prefix}.binning_factor must be an integer >= 1, got {binning!r}")
    else:
        values["binning_factor"] = int(binning)

    mode = raw.get("binning_mode")
    if is_placeholder(mode):
        pending.append(f"{prefix}.binning_mode is an unfilled placeholder: {mode!r}")
        mode = "unknown"
    elif mode not in VALID_BINNING_MODES:
        errors.append(
            f"{prefix}.binning_mode must be one of {sorted(VALID_BINNING_MODES)}, got {mode!r}"
        )

    na = raw.get("objective_na")
    objective_na: Optional[float] = None
    if na is not None:
        try:
            objective_na = _positive_float(na, f"{prefix}.objective_na")
        except OpticsError as exc:
            errors.append(str(exc))

    header = raw.get("header_pixel_spacing_um")
    header_value: Optional[float] = None
    if header is None or is_placeholder(header):
        pending.append(f"{prefix}.header_pixel_spacing_um has not been entered ({header!r}).")
    else:
        try:
            header_value = _positive_float(header, f"{prefix}.header_pixel_spacing_um")
        except OpticsError as exc:
            errors.append(str(exc))

    header_source = raw.get("header_source", "")
    if is_placeholder(header_source):
        pending.append(f"{prefix}.header_source is an unfilled placeholder.")

    if errors or any(k not in values for k in (*_NUMERIC_KEYS, *_PAIR_KEYS, "binning_factor")):
        return None, errors, pending

    return (
        DomainOptics(
            name=name,
            camera=str(raw.get("camera", "")),
            sensor_pixel_pitch_um=values["sensor_pixel_pitch_um"],
            sensor_native_px=values["sensor_native_px"],
            binning_factor=values["binning_factor"],
            binning_mode=str(mode),
            objective_magnification=values["objective_magnification"],
            objective_na=objective_na,
            camera_adapter_magnification=values["camera_adapter_magnification"],
            stored_image_px=values["stored_image_px"],
            header_pixel_spacing_um=header_value,
            header_source=str(header_source),
        ),
        errors,
        pending,
    )


# -----------------------------------------------------------------------------
# Derivation
# -----------------------------------------------------------------------------

def derive_pixel_spacing_um(domain: DomainOptics) -> float:
    """dx = pitch * binning / (M_objective * M_adapter), in µm per stored pixel."""
    return (
        domain.sensor_pixel_pitch_um
        * domain.binning_factor
        / (domain.objective_magnification * domain.camera_adapter_magnification)
    )


def derive_fov_um(domain: DomainOptics) -> Tuple[float, float]:
    """Field of view (height, width) in µm of a stored image."""
    dx = derive_pixel_spacing_um(domain)
    return domain.stored_image_px[0] * dx, domain.stored_image_px[1] * dx


def derive_network_input_spacing_um(domain: DomainOptics, network_input_px: int) -> Tuple[float, float]:
    """µm per network-input pixel after a pixel-locked resize to network_input_px."""
    if int(network_input_px) <= 0:
        raise OpticsError(f"network_input_px must be positive, got {network_input_px!r}")
    fov_h, fov_w = derive_fov_um(domain)
    return fov_h / int(network_input_px), fov_w / int(network_input_px)


def derive_decimation_factor(domain: DomainOptics, network_input_px: int) -> Tuple[float, float]:
    """Stored-to-network linear decimation factor (height, width)."""
    n = int(network_input_px)
    return domain.stored_image_px[0] / n, domain.stored_image_px[1] / n


def derive_domain(domain: DomainOptics, network_input_px: int) -> Dict[str, Any]:
    dx = derive_pixel_spacing_um(domain)
    fov_h, fov_w = derive_fov_um(domain)
    din_h, din_w = derive_network_input_spacing_um(domain, network_input_px)
    dec_h, dec_w = derive_decimation_factor(domain, network_input_px)
    return {
        "pixel_spacing_um": dx,
        "fov_um": [fov_h, fov_w],
        "network_input_spacing_um": [din_h, din_w],
        "decimation_factor": [dec_h, dec_w],
    }


def derive_cross_domain(
    training: DomainOptics,
    external: DomainOptics,
    network_input_px: int,
) -> Dict[str, float]:
    """
    Cross-domain ratios, always named <numerator>_over_<denominator>.

    apparent_object_scale_external_vs_training_in_network_input > 1 means an
    object of fixed physical size spans more network-input pixels in the
    external domain than in the training domain.
    """
    dx_t = derive_pixel_spacing_um(training)
    dx_e = derive_pixel_spacing_um(external)
    fov_t = derive_fov_um(training)[0]
    fov_e = derive_fov_um(external)[0]
    din_t = derive_network_input_spacing_um(training, network_input_px)[0]
    din_e = derive_network_input_spacing_um(external, network_input_px)[0]
    return {
        "training_over_external_native_spacing": dx_t / dx_e,
        "external_over_training_fov_linear": fov_e / fov_t,
        "external_over_training_fov_area": (fov_e / fov_t) ** 2,
        "apparent_object_scale_external_vs_training_in_network_input": din_t / din_e,
    }


# -----------------------------------------------------------------------------
# Validation and provenance
# -----------------------------------------------------------------------------

def relative_error(observed: float, reference: float) -> float:
    return abs(float(observed) - float(reference)) / abs(float(reference))


def validate_optics(optics: Mapping[str, Any]) -> Tuple[OpticsValidation, Dict[str, DomainOptics]]:
    """
    Apply O-1, O-2, O-3 and O-6 to an ``optics`` block.

    Placeholders are reported in ``pending_author_input``; callers decide
    whether pending input is fatal (strict runs) or a logged gap.
    """
    result = OpticsValidation()
    domains: Dict[str, DomainOptics] = {}

    if not isinstance(optics, Mapping):
        result.errors.append("O-1: metadata 'optics' block is missing or not an object.")
        return result, domains

    schema = optics.get("schema_version")
    if str(schema) != OPTICS_SCHEMA_VERSION:
        result.errors.append(
            f"O-1: optics.schema_version must be {OPTICS_SCHEMA_VERSION!r}, got {schema!r}."
        )

    tolerances = optics.get("tolerances", {}) or {}
    try:
        tol = float(tolerances.get("spacing_rel", DEFAULT_SPACING_REL_TOLERANCE))
        if not (0 < tol < 1):
            raise ValueError
    except (TypeError, ValueError):
        result.errors.append("O-1: optics.tolerances.spacing_rel must be in (0, 1).")
        tol = DEFAULT_SPACING_REL_TOLERANCE
    result.checks["spacing_rel_tolerance"] = tol

    n_input = optics.get("network_input_px")
    if isinstance(n_input, bool) or not isinstance(n_input, int) or n_input <= 0:
        result.errors.append(f"O-1: optics.network_input_px must be a positive integer, got {n_input!r}.")

    for name in DOMAIN_NAMES:
        dom, errs, pend = parse_domain(name, optics.get(name))
        result.errors.extend(f"O-1: {e}" for e in errs)
        result.pending_author_input.extend(pend)
        if dom is None:
            continue
        domains[name] = dom

        # O-2: stored image size consistent with sensor size and binning.
        expected = (
            dom.sensor_native_px[0] / dom.binning_factor,
            dom.sensor_native_px[1] / dom.binning_factor,
        )
        o2_ok = expected == (float(dom.stored_image_px[0]), float(dom.stored_image_px[1]))
        result.checks[f"O-2_{name}"] = {
            "stored_image_px": list(dom.stored_image_px),
            "expected_from_native_and_binning": [expected[0], expected[1]],
            "passed": o2_ok,
        }
        if not o2_ok:
            result.errors.append(
                f"O-2: optics.{name}: stored_image_px {list(dom.stored_image_px)} != "
                f"sensor_native_px {list(dom.sensor_native_px)} / binning_factor {dom.binning_factor}."
            )

        # O-3: derived spacing vs header spacing.
        derived = derive_pixel_spacing_um(dom)
        if dom.header_pixel_spacing_um is not None:
            rel = relative_error(derived, dom.header_pixel_spacing_um)
            passed = rel <= tol
            result.checks[f"O-3_{name}"] = {
                "derived_pixel_spacing_um": derived,
                "header_pixel_spacing_um": dom.header_pixel_spacing_um,
                "relative_error": rel,
                "header_over_derived": dom.header_pixel_spacing_um / derived,
                "passed": passed,
            }
            if not passed:
                result.errors.append(
                    f"O-3: optics.{name}: header spacing {dom.header_pixel_spacing_um:.8g} µm/px "
                    f"disagrees with derived {derived:.8g} µm/px (ratio "
                    f"{derived / dom.header_pixel_spacing_um:.4f}, relative error {rel:.4%} > {tol:.2%})."
                )
        else:
            result.checks[f"O-3_{name}"] = {"derived_pixel_spacing_um": derived, "passed": None}

        # O-6: unknown binning mode is a provenance gap, not an error.
        if dom.binning_mode == "unknown":
            result.provenance_gaps.append(
                f"O-6: optics.{name}.binning_mode is 'unknown' (factor {dom.binning_factor}); "
                "hardware binning vs digital subsampling/averaging is not recorded in the TIFF header."
            )

    return result, domains


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def optics_sha256(optics: Mapping[str, Any]) -> str:
    """Deterministic hash of the stored optics primitives (not of derived values)."""
    return hashlib.sha256(canonical_json(optics).encode("utf-8")).hexdigest()


def optics_provenance_record(optics: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Full provenance record: stored primitives, derived geometry, validation
    verdict and a hash identifying the primitives. Safe to embed in any output.
    """
    validation, domains = validate_optics(optics)
    derived: Dict[str, Any] = {}
    n_input = optics.get("network_input_px") if isinstance(optics, Mapping) else None
    if isinstance(n_input, int) and n_input > 0:
        for name, dom in domains.items():
            derived[name] = derive_domain(dom, n_input)
        if "training" in domains and "external" in domains:
            derived["cross_domain"] = derive_cross_domain(domains["training"], domains["external"], n_input)
    return {
        "schema_version": OPTICS_SCHEMA_VERSION,
        "role": "descriptive_provenance_not_used_to_transform_pixels",
        "optics_sha256": optics_sha256(optics) if isinstance(optics, Mapping) else None,
        "primitives": json.loads(canonical_json(optics)) if isinstance(optics, Mapping) else None,
        "derived": derived,
        "validation": validation.as_dict(),
        "provenance_gaps": list(validation.provenance_gaps),
    }


def optics_reference_summary(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Compact descriptive subset for preprocessing/evaluation metadata."""
    derived = record.get("derived", {}) or {}
    ext = derived.get("external", {}) or {}
    trn = derived.get("training", {}) or {}
    cross = derived.get("cross_domain", {}) or {}
    return {
        "role": "descriptive_only_not_used_in_computation",
        "optics_sha256": record.get("optics_sha256"),
        "external_pixel_spacing_um": ext.get("pixel_spacing_um"),
        "training_pixel_spacing_um": trn.get("pixel_spacing_um"),
        "external_network_input_spacing_um": ext.get("network_input_spacing_um"),
        "training_network_input_spacing_um": trn.get("network_input_spacing_um"),
        "external_decimation_factor": ext.get("decimation_factor"),
        "training_decimation_factor": trn.get("decimation_factor"),
        "apparent_object_scale_external_vs_training_in_network_input": cross.get(
            "apparent_object_scale_external_vs_training_in_network_input"
        ),
        "provenance_gaps": list(record.get("provenance_gaps", [])),
        "validation_ok": (record.get("validation", {}) or {}).get("ok"),
    }


def load_optics_from_metadata(metadata_path: Path) -> Dict[str, Any]:
    with Path(metadata_path).open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    optics = metadata.get("optics") if isinstance(metadata, dict) else None
    if not isinstance(optics, dict):
        raise OpticsError(f"Metadata JSON has no 'optics' object: {metadata_path}")
    return optics


# -----------------------------------------------------------------------------
# TIFF calibration parsing (O-4)
# -----------------------------------------------------------------------------

def _unit_to_um(unit: Any) -> Optional[float]:
    if unit is None:
        return None
    key = str(unit).strip().lower()
    key = key.replace("\\u00b5", "µ").replace("μ", "µ")
    if key in _UNIT_TO_UM:
        return _UNIT_TO_UM[key]
    if key.replace("µ", "u") in _UNIT_TO_UM:
        return _UNIT_TO_UM[key.replace("µ", "u")]
    return None


def _rational_to_float(value: Any) -> Optional[float]:
    try:
        if isinstance(value, (tuple, list)) and len(value) == 2:
            num, den = float(value[0]), float(value[1])
            return num / den if den != 0 else None
        return float(value)
    except (TypeError, ValueError):
        return None


def read_tiff_calibration(tif: Any) -> Dict[str, Any]:
    """
    Read pixel calibration from an open ``tifffile.TiffFile`` without reading
    pixel data. Priority:

        1. imagej_unit_XResolution  - ImageJ ``unit`` + XResolution/YResolution
                                      (format used by both the internal and
                                      external acquisitions, author-verified)
        2. ome_PhysicalSize         - OME-XML PhysicalSizeX/Y (+ unit)
        3. tiff_ResolutionUnit      - TIFF ResolutionUnit INCH/CENTIMETER

    Returns {x_um, y_um, source, unit}. ``source`` is "calibration_tag_absent"
    when no usable calibration exists; this is never treated as a pass.
    """
    out: Dict[str, Any] = {"x_um": None, "y_um": None, "source": "calibration_tag_absent", "unit": None}
    try:
        page = tif.pages[0]
        tags = page.tags
    except Exception:
        return out

    xres = _rational_to_float(tags["XResolution"].value) if "XResolution" in tags else None
    yres = _rational_to_float(tags["YResolution"].value) if "YResolution" in tags else None

    # 1. ImageJ unit + resolution (pixels per unit).
    ij = getattr(tif, "imagej_metadata", None) or {}
    ij_unit = ij.get("unit") if isinstance(ij, dict) else None
    ij_scale = _unit_to_um(ij_unit)
    if ij_scale is not None and xres and yres:
        out.update(
            x_um=ij_scale / xres,
            y_um=ij_scale / yres,
            source="imagej_unit_XResolution",
            unit=str(ij_unit),
        )
        return out

    # 2. OME-XML physical size.
    ome = getattr(tif, "ome_metadata", None)
    if ome:
        try:
            root = ET.fromstring(ome)
            pixels = next((el for el in root.iter() if el.tag.endswith("Pixels")), None)
            if pixels is not None and pixels.get("PhysicalSizeX"):
                ux = _unit_to_um(pixels.get("PhysicalSizeXUnit", "µm")) or 1.0
                uy = _unit_to_um(pixels.get("PhysicalSizeYUnit", "µm")) or 1.0
                px = float(pixels.get("PhysicalSizeX"))
                py = float(pixels.get("PhysicalSizeY", pixels.get("PhysicalSizeX")))
                out.update(x_um=px * ux, y_um=py * uy, source="ome_PhysicalSize", unit="um")
                return out
        except (ET.ParseError, ValueError, TypeError):
            pass

    # 3. Standard TIFF resolution unit.
    if "ResolutionUnit" in tags and xres and yres:
        try:
            unit_code = int(tags["ResolutionUnit"].value)
        except (TypeError, ValueError):
            unit_code = 1
        unit_um = {2: 25400.0, 3: 10000.0}.get(unit_code)
        if unit_um is not None:
            out.update(
                x_um=unit_um / xres,
                y_um=unit_um / yres,
                source="tiff_ResolutionUnit_XResolution",
                unit={2: "inch", 3: "centimeter"}[unit_code],
            )
    return out


_SLIDE_DATE_RE = re.compile(r"^(\d{8})_")


def acquisition_date_from_slide_id(slide_id: str) -> Optional[str]:
    match = _SLIDE_DATE_RE.match(str(slide_id))
    return match.group(1) if match else None
