#!/usr/bin/env python3
"""
optical_geometry.py

Single source of truth for the optical constants of the two microscopes and
for every spatial figure derived from them (pixel pitch, field of view,
network-input sampling pitch, scale-matched tile, canvas padding).

Optics
------
Object-space pixel pitch = binning x photosite pitch / (objective x adapter magnification).

* Development cohort: Photometrics Prime 95B, 1200 x 1200 px, 11.0 um photosites,
  no binning, 63x objective, 1.0x adapter -> 11.0 / 63 = 0.1746 um/px;
  field of view 1200 x 0.1746 = 209.52 um.
* External cohort: ZEISS Axiocam 820 mono, 4512 x 4512 px sensor, 2.74 um
  photosites, read out with 2 x 2 binning (5.48 um effective pixel) into
  2256 x 2256 px images, 63x objective, 1.0x adapter -> 5.48 / 63 = 0.08698 um/px,
  as recorded by ZEISS ZEN in the TIFF headers; field of view 196.23 um.

Consequences used throughout the pipeline
-----------------------------------------
* Both cameras image nearly the same specimen area (196.2 vs 209.5 um).
* The external images sample the specimen ~2.0x more finely per axis.
* After the pixel-locked resize of the full frame to 384 px, one network-input
  pixel spans 0.511 um (external) vs 0.546 um (development): external cells
  appear 1.068x larger (6.8 % linear).
* The analysed pipelines (locked_pixel_pipeline, photometric_mean_std_matched_pipeline)
  work in the pixel domain and do not consume a physical spacing.

The configs (configs/external_validation_config.yaml,
configs/external_validation_metadata.json) carry these spacings and the
geometry derived from them; audit_config_geometry() verifies at run time that
every spacing and every derived figure in them equals what this module computes,
and fails closed otherwise.

Importing this module performs arithmetic only (stdlib), plus a consistency
self-check that raises if the constants below are edited inconsistently.

Author: Rodrigo Sá
Date: 2026
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

# -----------------------------------------------------------------------------
# Hardware constants
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraGeometry:
    """One microscope/camera combination. All lengths in micrometres."""

    camera: str
    native_px: int                    # sensor side length (square sensors)
    sensor_pitch_um: float            # physical photosite pitch
    binning: int                      # on-camera binning factor per axis
    acquired_px: int                  # image side length as stored in the TIFFs
    objective_magnification: float
    adapter_magnification: float      # camera adapter (C-mount) magnification
    recorded_spacing_um: float        # spacing written at acquisition
    recorded_spacing_source: str

    @property
    def effective_pixel_pitch_um(self) -> float:
        """Sensor-plane size of one stored pixel (binning x photosite pitch)."""
        return self.binning * self.sensor_pitch_um

    @property
    def derived_object_pitch_um(self) -> float:
        """Object-space pitch of one stored pixel."""
        return self.effective_pixel_pitch_um / (self.objective_magnification * self.adapter_magnification)

    @property
    def derived_fov_um(self) -> float:
        return self.acquired_px * self.derived_object_pitch_um

    @property
    def recorded_fov_um(self) -> float:
        return self.acquired_px * self.recorded_spacing_um


INTERNAL = CameraGeometry(
    camera="Photometrics Prime 95B", native_px=1200, sensor_pitch_um=11.0, binning=1,
    acquired_px=1200, objective_magnification=63.0, adapter_magnification=1.0,
    recorded_spacing_um=0.1746, recorded_spacing_source="development-cohort acquisition records",
)
EXTERNAL = CameraGeometry(
    camera="ZEISS Axiocam 820 mono", native_px=4512, sensor_pitch_um=2.74, binning=2,
    acquired_px=2256, objective_magnification=63.0, adapter_magnification=1.0,
    recorded_spacing_um=0.08698, recorded_spacing_source="ZEISS ZEN TIFF headers (external cohort)",
)

MODEL_INPUT_PX = 384                 # frozen network input side
TRAINING_CANVAS_PX = 1200            # scale-matching target canvas (development raw size)

DERIVED_VS_RECORDED_REL_TOL = 1e-3   # derived optics vs recorded spacing
CONFIG_SPACING_REL_TOL = 1e-3        # a config/header spacing matches the recorded value within this
CONFIG_DERIVED_DECIMALS = 6          # derived figures in the configs must match to this many decimals


def _self_check() -> None:
    problems: List[str] = []
    for g in (INTERNAL, EXTERNAL):
        if g.native_px != g.binning * g.acquired_px:
            problems.append(f"{g.camera}: native_px != binning x acquired_px")
        rel = abs(g.derived_object_pitch_um - g.recorded_spacing_um) / g.recorded_spacing_um
        if rel > DERIVED_VS_RECORDED_REL_TOL:
            problems.append(f"{g.camera}: derived {g.derived_object_pitch_um:.6f} vs recorded "
                            f"{g.recorded_spacing_um} um/px (rel {rel:.2e})")
    if problems:
        raise RuntimeError("optical_geometry constants are inconsistent: " + "; ".join(problems))


_self_check()

# -----------------------------------------------------------------------------
# Derived geometry
# -----------------------------------------------------------------------------


def round_half_up(x: float) -> int:
    """Deterministic rounding (no banker's rounding)."""
    return int(math.floor(float(x) + 0.5))


def _positive(name: str, v: float) -> float:
    v = float(v)
    if not math.isfinite(v) or v <= 0:
        raise ValueError(f"{name} must be finite and > 0, got {v!r}")
    return v


def scale_geometry(
    external_spacing_um: float = EXTERNAL.recorded_spacing_um,
    training_spacing_um: float = INTERNAL.recorded_spacing_um,
    *,
    external_px: int = EXTERNAL.acquired_px,
    training_px: int = INTERNAL.acquired_px,
    canvas_px: int = TRAINING_CANVAS_PX,
    model_px: int = MODEL_INPUT_PX,
) -> Dict[str, Any]:
    """
    Every spatial quantity that depends on the two pixel spacings.

    Locked-pixel pipeline: the whole frame (side N px) is resized to model_px,
    so one network-input pixel spans FOV / model_px um.

    Scale-matched pipeline (config ``scale_matching_sensitivity``): content is
    resampled by s = external_spacing / training_spacing, giving a tile of
    round(external_px * s) px at the training spacing, centre-padded (or
    cropped) into a canvas_px x canvas_px canvas, then resized to model_px.
    """
    e = _positive("external_spacing_um", external_spacing_um)
    t = _positive("training_spacing_um", training_spacing_um)
    for name, v in (("external_px", external_px), ("training_px", training_px),
                    ("canvas_px", canvas_px), ("model_px", model_px)):
        if int(v) != v or v <= 0:
            raise ValueError(f"{name} must be a positive integer, got {v!r}")

    ext_fov = external_px * e
    trn_fov = training_px * t
    s = e / t
    tile_exact = external_px * s
    tile_px = round_half_up(tile_exact)
    content_side = min(tile_px, canvas_px)
    content_fraction = (content_side / canvas_px) ** 2
    ext_input_pitch = ext_fov / model_px
    trn_input_pitch = trn_fov / model_px
    return {
        "external_spacing_um": e,
        "training_spacing_um": t,
        "external_fov_um": ext_fov,
        "training_fov_um": trn_fov,
        "fov_ratio_external_over_training_linear": ext_fov / trn_fov,
        "fov_ratio_external_over_training_area": (ext_fov / trn_fov) ** 2,
        "training_over_external_linear_sampling_density_ratio": t / e,
        "approximate_area_sampling_density_ratio": (t / e) ** 2,
        # locked-pixel / photometric pipelines (pixel domain)
        "locked_resize_factor_external": external_px / model_px,
        "locked_resize_factor_training": training_px / model_px,
        "locked_input_pitch_external_um": ext_input_pitch,
        "locked_input_pitch_training_um": trn_input_pitch,
        "locked_apparent_magnification_external_vs_training": trn_input_pitch / ext_input_pitch,
        # scale-matched pipeline (excluded)
        "content_resampling_factor_external_to_training": s,
        "scale_matched_tile_px_exact": tile_exact,
        "scale_matched_tile_px": tile_px,
        "scale_matched_tile_cropped": tile_px > canvas_px,
        "scale_matched_canvas_px": canvas_px,
        "scale_matched_content_fraction": content_fraction,
        "scale_matched_padding_fraction": 1.0 - content_fraction,
        "scale_matched_border_px_per_side": (canvas_px - content_side) / 2.0,
        "scale_matched_tile_in_model_px": round_half_up(content_side * model_px / canvas_px),
        "scale_matched_input_pitch_um": canvas_px * t / model_px,
        "scale_matched_tile_fov_um": tile_px * t,
    }


GEOMETRY: Dict[str, Any] = scale_geometry()

# -----------------------------------------------------------------------------
# Config audit (text files only; no data, no images)
# -----------------------------------------------------------------------------

#: Spacing fields: dotted path -> expected value.
CONFIG_SPACING_FIELDS: Dict[str, float] = {
    "training_domain_reference.pixel_spacing_um.x": INTERNAL.recorded_spacing_um,
    "training_domain_reference.pixel_spacing_um.y": INTERNAL.recorded_spacing_um,
    "external_domain.pixel_spacing_um.x": EXTERNAL.recorded_spacing_um,
    "external_domain.pixel_spacing_um.y": EXTERNAL.recorded_spacing_um,
    "domain_shift.pixel_spacing.training_um_per_px": INTERNAL.recorded_spacing_um,
    "domain_shift.pixel_spacing.external_um_per_px": EXTERNAL.recorded_spacing_um,
    "scale_matching_sensitivity.training_pixel_spacing_um": INTERNAL.recorded_spacing_um,
    "scale_matching_sensitivity.external_pixel_spacing_um": EXTERNAL.recorded_spacing_um,
    "preprocessing.available_modes.scale_matched_physical_pipeline.training_pixel_spacing_um":
        INTERNAL.recorded_spacing_um,
    "preprocessing.available_modes.scale_matched_physical_pipeline.external_pixel_spacing_um":
        EXTERNAL.recorded_spacing_um,
}
METADATA_SPACING_FIELDS: Dict[str, float] = {
    "external_acquisition_metadata.pixel_spacing_um.x": EXTERNAL.recorded_spacing_um,
    "external_acquisition_metadata.pixel_spacing_um.y": EXTERNAL.recorded_spacing_um,
    "training_domain_reference.pixel_spacing_um.x": INTERNAL.recorded_spacing_um,
    "training_domain_reference.pixel_spacing_um.y": INTERNAL.recorded_spacing_um,
}

_SM = "scale_matching_sensitivity"
_SMP = "preprocessing.available_modes.scale_matched_physical_pipeline"
#: Derived-figure fields: dotted path (list fields get "[i]") -> GEOMETRY key.
CONFIG_DERIVED_FIELDS: Dict[str, str] = {
    "training_domain_reference.physical_field_of_view_um.height": "training_fov_um",
    "training_domain_reference.physical_field_of_view_um.width": "training_fov_um",
    "external_domain.physical_field_of_view_um.height": "external_fov_um",
    "external_domain.physical_field_of_view_um.width": "external_fov_um",
    "domain_shift.pixel_spacing.external_to_training_linear_sampling_density_ratio":
        "training_over_external_linear_sampling_density_ratio",
    "domain_shift.pixel_spacing.external_content_resampling_factor_to_training_spacing":
        "content_resampling_factor_external_to_training",
    "domain_shift.pixel_spacing.training_physical_fov_um[0]": "training_fov_um",
    "domain_shift.pixel_spacing.training_physical_fov_um[1]": "training_fov_um",
    "domain_shift.pixel_spacing.external_physical_fov_um[0]": "external_fov_um",
    "domain_shift.pixel_spacing.external_physical_fov_um[1]": "external_fov_um",
    f"{_SM}.content_resampling_factor_external_to_training": "content_resampling_factor_external_to_training",
    f"{_SM}.training_over_external_linear_sampling_density_ratio":
        "training_over_external_linear_sampling_density_ratio",
    f"{_SM}.training_physical_fov_um.height": "training_fov_um",
    f"{_SM}.training_physical_fov_um.width": "training_fov_um",
    f"{_SM}.external_physical_fov_um.height": "external_fov_um",
    f"{_SM}.external_physical_fov_um.width": "external_fov_um",
    f"{_SM}.expected_resampled_external_content_px_at_training_spacing.height": "scale_matched_tile_px",
    f"{_SM}.expected_resampled_external_content_px_at_training_spacing.width": "scale_matched_tile_px",
    f"{_SM}.expected_content_fraction_in_training_canvas": "scale_matched_content_fraction",
    f"{_SM}.expected_padding_fraction_in_training_canvas": "scale_matched_padding_fraction",
    f"{_SM}.expected_effective_content_size_in_384_px.height": "scale_matched_tile_in_model_px",
    f"{_SM}.expected_effective_content_size_in_384_px.width": "scale_matched_tile_in_model_px",
    f"{_SMP}.content_scale_external_to_training": "content_resampling_factor_external_to_training",
    f"{_SMP}.training_over_external_linear_sampling_density_ratio":
        "training_over_external_linear_sampling_density_ratio",
    f"{_SMP}.approximate_area_sampling_density_ratio": "approximate_area_sampling_density_ratio",
    f"{_SMP}.expected_resampled_external_content_px_at_training_spacing[0]": "scale_matched_tile_px",
    f"{_SMP}.expected_resampled_external_content_px_at_training_spacing[1]": "scale_matched_tile_px",
    f"{_SMP}.expected_content_fraction_in_training_canvas": "scale_matched_content_fraction",
    f"{_SMP}.expected_padding_fraction_in_training_canvas": "scale_matched_padding_fraction",
    f"{_SMP}.expected_effective_content_size_in_384_px[0]": "scale_matched_tile_in_model_px",
    f"{_SMP}.expected_effective_content_size_in_384_px[1]": "scale_matched_tile_in_model_px",
}
METADATA_DERIVED_FIELDS: Dict[str, str] = {
    "domain_shift_metadata.external_to_training_pixel_spacing_ratio":
        "training_over_external_linear_sampling_density_ratio",
}
#: Hardware fields documented in the configs: dotted path -> expected value.
CONFIG_HARDWARE_FIELDS: Dict[str, Any] = {
    "training_domain_reference.raw_image_size_px.height": INTERNAL.acquired_px,
    "training_domain_reference.raw_image_size_px.width": INTERNAL.acquired_px,
    "external_domain.raw_image_size_px.height": EXTERNAL.acquired_px,
    "external_domain.raw_image_size_px.width": EXTERNAL.acquired_px,
    "external_domain.native_sensor_size_px.height": EXTERNAL.native_px,
    "external_domain.native_sensor_size_px.width": EXTERNAL.native_px,
    "external_domain.sensor_pixel_pitch_um": EXTERNAL.sensor_pitch_um,
    "external_domain.binning": EXTERNAL.binning,
    f"{_SM}.target_canvas_height_px": TRAINING_CANVAS_PX,
    f"{_SM}.target_canvas_width_px": TRAINING_CANVAS_PX,
}
METADATA_HARDWARE_FIELDS: Dict[str, Any] = {
    "external_acquisition_metadata.native_sensor_size_px": EXTERNAL.native_px,
    "external_acquisition_metadata.sensor_pixel_pitch_um": EXTERNAL.sensor_pitch_um,
    "external_acquisition_metadata.binning": EXTERNAL.binning,
    "external_acquisition_metadata.acquired_image_size_px": EXTERNAL.acquired_px,
}


def _get(d: Any, dotted: str) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        idx: Optional[int] = None
        if part.endswith("]") and "[" in part:
            part, raw = part[:-1].split("[", 1)
            idx = int(raw)
        if not isinstance(cur, Mapping) or part not in cur:
            raise KeyError(dotted)
        cur = cur[part]
        if idx is not None:
            if not isinstance(cur, (list, tuple)) or idx >= len(cur):
                raise KeyError(dotted)
            cur = cur[idx]
    return cur


def spacing_matches(value: Any, expected: float) -> bool:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v) and abs(v - expected) / expected <= CONFIG_SPACING_REL_TOL


def _check(doc: Any, label: str, fields: Mapping[str, Any], match) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for path, expected in fields.items():
        try:
            v = _get(doc, path)
        except KeyError:
            raise RuntimeError(f"[optical_geometry] {label}: field {path!r} not found") from None
        ok = match(v, expected)
        out[f"{label}:{path}"] = {"value": v, "expected": expected, "ok": bool(ok)}
    return out


def _derived_match(v: Any, key: str) -> bool:
    try:
        return round(float(v), CONFIG_DERIVED_DECIMALS) == round(float(GEOMETRY[key]), CONFIG_DERIVED_DECIMALS)
    except (TypeError, ValueError):
        return False


def _exact_match(v: Any, expected: Any) -> bool:
    try:
        return float(v) == float(expected)
    except (TypeError, ValueError):
        return False


def audit_config_geometry(config_yaml: Path, metadata_json: Path) -> Dict[str, Any]:
    """
    Verify that every spacing, hardware and derived-geometry field of the two
    configs equals the values of this module. Any missing or differing field raises.
    """
    import json

    import yaml

    cfg = yaml.safe_load(Path(config_yaml).read_text(encoding="utf-8"))
    md = json.loads(Path(metadata_json).read_text(encoding="utf-8"))
    checks: Dict[str, Dict[str, Any]] = {}
    checks.update(_check(cfg, "config", CONFIG_SPACING_FIELDS, spacing_matches))
    checks.update(_check(md, "metadata", METADATA_SPACING_FIELDS, spacing_matches))
    checks.update(_check(cfg, "config", CONFIG_HARDWARE_FIELDS, _exact_match))
    checks.update(_check(md, "metadata", METADATA_HARDWARE_FIELDS, _exact_match))
    checks.update(_check(cfg, "config", CONFIG_DERIVED_FIELDS, _derived_match))
    checks.update(_check(md, "metadata", METADATA_DERIVED_FIELDS, _derived_match))
    bad = {k: v for k, v in checks.items() if not v["ok"]}
    if bad:
        raise RuntimeError(f"[optical_geometry] config geometry does not match optical_geometry.py: {bad}")
    return {
        "schema_version": "2.0",
        "script": "optical_geometry.py",
        "summary": optics_statement(),
        "n_fields_checked": len(checks),
        "all_fields_match": True,
        "fields": checks,
        "hardware": {g.camera: hardware_record(g) for g in (INTERNAL, EXTERNAL)},
        "geometry": dict(GEOMETRY),
        "analysed_pipelines_consume_spacing": False,
        "analysed_pipelines_note": ("locked_pixel_pipeline and photometric_mean_std_matched_pipeline resize the "
                                    "2256 px frame to 384 px in the pixel domain; no physical spacing enters "
                                    "their arrays, predictions or statistics."),
        "external_spacing_um": EXTERNAL.recorded_spacing_um,
        "training_spacing_um": INTERNAL.recorded_spacing_um,
    }


def hardware_record(g: CameraGeometry) -> Dict[str, Any]:
    return {
        "camera": g.camera, "native_px": g.native_px, "sensor_pitch_um": g.sensor_pitch_um,
        "binning": g.binning, "acquired_px": g.acquired_px,
        "effective_pixel_pitch_um": g.effective_pixel_pitch_um,
        "objective_magnification": g.objective_magnification, "adapter_magnification": g.adapter_magnification,
        "derived_object_pitch_um": g.derived_object_pitch_um,
        "recorded_spacing_um": g.recorded_spacing_um, "recorded_spacing_source": g.recorded_spacing_source,
        "derived_fov_um": g.derived_fov_um, "recorded_fov_um": g.recorded_fov_um,
    }


def optics_statement() -> str:
    g = GEOMETRY
    return (f"External: {EXTERNAL.camera}, {EXTERNAL.native_px} px sensor at {EXTERNAL.sensor_pitch_um} um, "
            f"{EXTERNAL.binning}x{EXTERNAL.binning} binning -> {EXTERNAL.acquired_px} px at "
            f"{EXTERNAL.recorded_spacing_um} um/px (FOV {g['external_fov_um']:.2f} um). "
            f"Development: {INTERNAL.camera}, {INTERNAL.acquired_px} px at {INTERNAL.recorded_spacing_um} um/px "
            f"(FOV {g['training_fov_um']:.2f} um). Network input ({MODEL_INPUT_PX} px): "
            f"{g['locked_input_pitch_external_um']:.3f} vs {g['locked_input_pitch_training_um']:.3f} um/px, "
            f"apparent magnification {g['locked_apparent_magnification_external_vs_training']:.3f}x.")


def plan_block() -> Dict[str, Any]:
    """Compact record for analysis_plan_frozen.json and results_summary.json."""
    g = GEOMETRY
    return {
        "external_pixel_spacing_um": EXTERNAL.recorded_spacing_um,
        "external_pixel_spacing_source": EXTERNAL.recorded_spacing_source,
        "external_binning": EXTERNAL.binning,
        "training_pixel_spacing_um": INTERNAL.recorded_spacing_um,
        "external_fov_um": round(g["external_fov_um"], 3),
        "training_fov_um": round(g["training_fov_um"], 3),
        "locked_input_pitch_external_um": round(g["locked_input_pitch_external_um"], 4),
        "locked_input_pitch_training_um": round(g["locked_input_pitch_training_um"], 4),
        "locked_apparent_magnification_external_vs_training": round(
            g["locked_apparent_magnification_external_vs_training"], 4),
        "analysed_pipelines_consume_spacing": False,
    }


def excluded_scale_pipeline_rationale() -> str:
    g = GEOMETRY
    mag = g["locked_apparent_magnification_external_vs_training"]
    return (
        "Not analysed. The locked-pixel input samples the specimen at "
        f"{g['locked_input_pitch_external_um']:.3f} um/px versus {g['locked_input_pitch_training_um']:.3f} um/px "
        f"for the development images (FOV {g['external_fov_um']:.1f} vs {g['training_fov_um']:.1f} um; apparent "
        f"magnification {mag:.3f}x, i.e. {100 * (mag - 1):.1f}% linear). Scale matching "
        f"(s = {g['content_resampling_factor_external_to_training']:.3f}, {g['scale_matched_tile_px']}-px tile in a "
        f"{g['scale_matched_canvas_px']}-px canvas) would correct that small residual while adding a "
        f"{100 * g['scale_matched_padding_fraction']:.1f}% median canvas, ~2x2 area averaging and a "
        f"{g['scale_matched_border_px_per_side']:.0f}-px frame border, and it keeps the +26.2 sigma_train "
        "radiometric offset, so the scale effect is neither material nor identified at n = 24. The mode remains "
        "in the pinned legacy code and is never called."
    )


# -----------------------------------------------------------------------------
# Header-only TIFF spacing (verification helper; reads tags, not pixels)
# -----------------------------------------------------------------------------

_UNIT_UM = {2: 25400.0, 3: 10000.0}   # TIFF ResolutionUnit: 2 = inch, 3 = centimetre


def spacing_um_from_resolution(resolution: Any, unit: int) -> Optional[float]:
    """TIFF X/YResolution (pixels per unit, rational or float) -> um per pixel; None if unit-less/absent."""
    if resolution is None or int(unit) not in _UNIT_UM:
        return None
    if isinstance(resolution, (tuple, list)):
        num, den = resolution
        px_per_unit = float(num) / float(den) if float(den) else 0.0
    else:
        px_per_unit = float(resolution)
    if not math.isfinite(px_per_unit) or px_per_unit <= 0:
        return None
    return _UNIT_UM[int(unit)] / px_per_unit


def read_tiff_pixel_spacing_um(path: Path) -> Dict[str, Any]:
    """Spacing from the baseline TIFF resolution tags of page 0; raises if it differs from the
    recorded external spacing. Pixel data are not read."""
    import tifffile

    with tifffile.TiffFile(str(path)) as tf:
        tags = tf.pages[0].tags
        unit = int(tags["ResolutionUnit"].value) if "ResolutionUnit" in tags else 2
        xr = tags["XResolution"].value if "XResolution" in tags else None
        yr = tags["YResolution"].value if "YResolution" in tags else None
        shape = tuple(tf.pages[0].shape)
    sx, sy = spacing_um_from_resolution(xr, unit), spacing_um_from_resolution(yr, unit)
    for axis, v in (("x", sx), ("y", sy)):
        if v is not None and not spacing_matches(v, EXTERNAL.recorded_spacing_um):
            raise ValueError(f"{path}: TIFF {axis} spacing {v} um/px differs from {EXTERNAL.recorded_spacing_um}")
    return {"path": str(path), "shape": shape, "resolution_unit": unit, "spacing_x_um": sx,
            "spacing_y_um": sy, "matches_recorded_spacing": sx is not None and sy is not None}


if __name__ == "__main__":  # pragma: no cover - convenience printout, arithmetic only
    import json as _json

    print(optics_statement())
    print(_json.dumps(GEOMETRY, indent=2))
