#!/usr/bin/env python3
"""
verify_optical_configuration.py

Standalone, author-run verification of the optical-configuration.
No GPU, no torch, no dataset access: every TIFF is synthetic and header-only
(written with ``shape=`` and no pixel data) in a temporary directory, and every
cache, plan and summary is a small synthetic JSON/NPZ.

    python verify_optical_configuration.py [--json report.json]
    python verify_optical_configuration.py --header FILE [FILE ...]   # read tags of real TIFFs only

Groups
  A  optical formula exactness (Fractions) and end-to-end spec from synthetic metadata
  B  fail-loud behaviour (no fallback, conflicting / implausible / missing metadata, legacy flags)
  C  strict monotonicity of the physical scale levels, tile coverage, margins, condition grid
  D  consistency of physical constants in docstrings, schemas, config, notebook, export checks
  E  deterministic digests, cache invariance and the migration of legacy caches

Exit code 0 only when every check passes.

Author: Rodrigo Sá
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import re
import sys
import tempfile
import traceback
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import optics                                   # noqa: E402
import geometry_harmonization as geo            # noqa: E402

CONFIG = HERE / "colab_config.json"
P_INT = Fraction(11, 63)                        # 11.0 um / 63
P_EXT = Fraction(548, 100) / 63                 # 2 x 2.74 um / 63 = 137/1575
RESULTS: List[Dict[str, Any]] = []


class CheckFailed(AssertionError):
    pass


def ensure(cond: bool, msg: str) -> None:
    if not cond:
        raise CheckFailed(msg)


def raises(fn: Callable[[], Any], exc: Any, contains: str = "") -> str:
    try:
        fn()
    except exc as e:  # noqa: PERF203
        ensure(contains.lower() in str(e).lower(), f"error raised but message lacks {contains!r}: {e}")
        return str(e)
    raise CheckFailed(f"expected {getattr(exc, '__name__', exc)} was not raised")


def check(cid: str, title: str) -> Callable[[Callable[[], Any]], Callable[[], Any]]:
    def deco(fn: Callable[[], Any]) -> Callable[[], Any]:
        fn._cid, fn._title = cid, title          # type: ignore[attr-defined]
        return fn
    return deco


def cfg() -> Dict[str, Any]:
    return json.loads(CONFIG.read_text())


# ---- synthetic TIFFs ---------------------------------------------------------------------------------------

def tif(path: Path, n: int, *, kind: str = "none", um: float = 0.0, um_y: float = 0.0) -> Path:
    import tifffile

    kw: Dict[str, Any] = {"shape": (n, n), "dtype": "uint16"}
    if kind == "ome":
        kw.update(ome=True, metadata={"PhysicalSizeX": um, "PhysicalSizeXUnit": "µm", "PhysicalSizeY": um})
    elif kind == "imagej":
        kw.update(imagej=True, resolution=(1 / um, 1 / um), metadata={"unit": "micron"})
    elif kind == "cm":
        kw.update(resolution=(1e4 / um, 1e4 / (um_y or um)), resolutionunit="CENTIMETER")
    elif kind == "inch72":
        kw.update(resolution=(72.0, 72.0), resolutionunit="INCH")
    tifffile.imwrite(str(path), **kw)
    return path


def strip_resolution_unit(path: Path) -> None:
    """Rename tag 296 (ResolutionUnit) to a private code, leaving X/YResolution: 'unit absent'."""
    import tifffile

    with tifffile.TiffFile(str(path)) as t:
        off = t.pages[0].tags["ResolutionUnit"].offset
    b = bytearray(path.read_bytes())
    ensure(b[off:off + 2] == (296).to_bytes(2, "little"), "ResolutionUnit tag not at the expected offset")
    b[off:off + 2] = (65000).to_bytes(2, "little")
    path.write_bytes(bytes(b))


def acqs() -> Any:
    return optics.load_optics(cfg())


# =============================================================================
# A. Optical formula exactness
# =============================================================================

@check("A1", "exact pitch, FOV, T*, T, m_ext and residual from the config optics")
def a1() -> Dict[str, Any]:
    ai, ae, _ = acqs()
    ensure(ai.pixel_um == P_INT and ae.pixel_um == P_EXT == Fraction(137, 1575), "pixel spacings not exact")
    ensure(ai.fov_um == 1200 * P_INT == Fraction(4400, 21), "internal FOV")
    ensure(ae.fov_um == 2256 * P_EXT == 4512 * Fraction(274, 100) / 63, "external FOV (binning preserves FOV)")
    em = optics.external_matched(ai, ae)
    t_star = 2256 * P_EXT / P_INT
    ensure(Fraction(em["T_star_exact"]) == t_star == Fraction(2256 * 548, 1100), "T* not exact")
    ensure(em["T"] == 1124 and optics.round_half_up(t_star) == 1124, "T != 1124")
    ensure(Fraction(em["m_external_exact"]) == 1200 / t_star, "m_ext not exact")
    resid = (1124 * P_INT / 384 - 2256 * P_EXT / 384) / (2256 * P_EXT / 384)
    ensure(Fraction(em["relative_residual_exact"]) == resid and abs(resid) < Fraction(1, 1000), "residual")
    ensure(abs(em["spacing_external_at_384_um"] - 0.5110317460317461) < 1e-15, "external spacing at 384")
    return {"p_int": str(ai.pixel_um), "p_ext": str(ae.pixel_um), "fov_um": [float(ai.fov_um), float(ae.fov_um)],
            "T_star": float(t_star), "T": em["T"], "m_ext": em["m_external"], "residual": float(resid)}


@check("A2", "geometry spec built end-to-end from synthetic headers (OME external, ImageJ internal)")
def a2(tmp: Path) -> Dict[str, Any]:
    ints = [tif(tmp / f"i{k}.tif", 1200, kind="imagej", um=float(P_INT)) for k in range(3)]
    exts = [tif(tmp / f"e{k}.tif", 2256, kind="ome", um=float(P_EXT)) for k in range(3)]
    spec = geo.build_geometry_spec(ints, exts, cfg=cfg(), expected_opencv=None, record_opencv=False)
    em = spec["external_matched"]
    ensure(spec["schema"] == "geometry_spec/2.0" and spec["external_matched_T_used"] == 1124, "spec T")
    ensure(em["metadata_status"] == {"internal": "AGREES", "external": "AGREES"}, f"metadata {em['metadata_status']}")
    ensure(em["metadata_source"]["external"] == "ome_PhysicalSizeX", "external source")
    ensure(spec["geometry_digest"] == geo.expected_geometry(cfg())["geometry_digest"], "digest != config-implied digest")
    return {"T": spec["external_matched_T_used"], "metadata": em["metadata_status"],
            "metadata_um": em["metadata_value_um"], "digest": spec["geometry_digest"][:16]}


@check("A3", "rounded real-world header (0.0870 um/px, TIFF cm) agrees; spacing still from the optics")
def a3(tmp: Path) -> Dict[str, Any]:
    exts = [tif(tmp / f"e{k}.tif", 2256, kind="cm", um=0.0870) for k in range(2)]
    ai, ae, pol = acqs()
    sp = geo.resolve_domain_spacing(exts, ae, policy="required", rel_tol=pol["rel_tol"])
    ensure(sp.metadata_status == "AGREES" and sp.metadata_source == "tiff_XResolution_cm", "cm header not used")
    ensure(sp.um_per_px == float(P_EXT) and sp.exact_fraction == "137/1575", "spacing must come from the optics")
    return {"metadata_um": sp.metadata_value_um, "used_um": sp.um_per_px,
            "relative_difference": sp.metadata_value_um / sp.um_per_px - 1}


@check("A4", "legacy reference: 0.175 / 0.043 gives T = 554 and m = 2.1648, 2.0275x the true scale")
def a4() -> Dict[str, Any]:
    legacy_p_int, legacy_p_ext = Fraction("0.175"), Fraction("0.043")      # legacy config values (reference only)
    t_legacy = 2256 * legacy_p_ext / legacy_p_int
    m_legacy = Fraction(1200) * legacy_p_int / (2256 * legacy_p_ext)
    m_true = Fraction(1200) * P_INT / (2256 * P_EXT)
    ensure(optics.round_half_up(t_legacy) == 554, "legacy T")
    ensure(abs(float(m_legacy / m_true) - 2.0275) < 1e-4, "legacy / true ratio")
    ensure(abs(float(Fraction("2.74") / 63) - 0.043492) < 1e-6, "0.043 is the unbinned pitch / 63")
    return {"T_legacy": 554, "m_legacy": float(m_legacy), "m_true": float(m_true), "ratio": float(m_legacy / m_true)}


# =============================================================================
# B. Fail-loud behaviour
# =============================================================================

@check("B1", "header at the unbinned value (0.043 um/px) fails and names the binning")
def b1(tmp: Path) -> Dict[str, Any]:
    _, ae, pol = acqs()
    f = [tif(tmp / "e.tif", 2256, kind="ome", um=0.043)]
    msg = raises(lambda: geo.resolve_domain_spacing(f, ae, policy="required", rel_tol=pol["rel_tol"]),
                 optics.OpticsError, "binning")
    return {"message": msg}


@check("B2", "required metadata absent -> fails (no fallback value exists)")
def b2(tmp: Path) -> Dict[str, Any]:
    _, ae, _ = acqs()
    f = [tif(tmp / "e.tif", 2256)]
    return {"message": raises(lambda: geo.resolve_domain_spacing(f, ae, policy="required"), optics.OpticsError,
                              "no usable pixel-size metadata")}


@check("B3", "internal metadata absent under verify_if_present -> ABSENT, spec builds, unit check (b) CONDITIONAL")
def b3(tmp: Path) -> Dict[str, Any]:
    import verify_unit_checks as vuc
    ints = [tif(tmp / "i.tif", 1200)]
    exts = [tif(tmp / "e.tif", 2256, kind="ome", um=float(P_EXT))]
    spec = geo.build_geometry_spec(ints, exts, cfg=cfg(), expected_opencv=None, record_opencv=False)
    ensure(spec["external_matched"]["metadata_status"]["internal"] == "ABSENT", "internal should be ABSENT")
    r = vuc.check_b_spacing(SimpleNamespace(spec=spec))
    ensure(r["status"] == vuc.COND, f"check (b) {r['status']}")
    return {"check_b": r["status"], "condition": r.get("condition")}


@check("B4", "72-dpi writer default and missing ResolutionUnit are rejected, never used")
def b4(tmp: Path) -> Dict[str, Any]:
    _, ae, _ = acqs()
    f72 = tif(tmp / "e72.tif", 2256, kind="inch72")
    c72 = geo.read_spacing_candidates(f72)
    ensure(not c72["candidates"] and "tiff_XResolution_inch" in c72["rejected"], f"72 dpi used: {c72}")
    fnu = tif(tmp / "enu.tif", 2256, kind="cm", um=float(P_EXT))
    strip_resolution_unit(fnu)
    cnu = geo.read_spacing_candidates(fnu)
    ensure(not cnu["candidates"] and "absent" in cnu["rejected"].get("tiff_XResolution", ""), f"unit-less used: {cnu}")
    raises(lambda: geo.resolve_domain_spacing([f72, fnu], ae, policy="required"), optics.OpticsError, "no usable")
    return {"rejected_72dpi": c72["rejected"], "rejected_no_unit": cnu["rejected"]}


@check("B5", "anisotropic pixels, inter-file disagreement, partial metadata and wrong frame shape fail")
def b5(tmp: Path) -> Dict[str, Any]:
    _, ae, _ = acqs()
    an = [tif(tmp / "an.tif", 2256, kind="cm", um=float(P_EXT), um_y=float(P_EXT) * 1.01)]
    raises(lambda: geo.resolve_domain_spacing(an, ae, policy="required"), optics.OpticsError, "anisotropic")
    d = [tif(tmp / "d1.tif", 2256, kind="ome", um=0.086984), tif(tmp / "d2.tif", 2256, kind="ome", um=0.0870)]
    raises(lambda: geo.resolve_domain_spacing(d, ae, policy="required"), optics.OpticsError, "disagrees between files")
    part = [tif(tmp / "p1.tif", 2256, kind="ome", um=float(P_EXT)), tif(tmp / "p2.tif", 2256)]
    raises(lambda: geo.resolve_domain_spacing(part, ae, policy="required"), optics.OpticsError, "inconsistent")
    wrong = [tif(tmp / "w.tif", 4512, kind="ome", um=float(P_EXT))]
    raises(lambda: geo.resolve_domain_spacing(wrong, ae, policy="required"), optics.OpticsError, "shape")
    return {"anisotropic": "fail", "disagreement": "fail", "partial": "fail", "unbinned_frame_shape": "fail"}


@check("B6", "removed CLI flags and legacy config keys are refused")
def b6(tmp: Path) -> Dict[str, Any]:
    import run_all_colab as ra
    out = {}
    for flag in (["--allow-config-spacing"], ["--p-ext", "0.043"], ["--p-int", "0.175"]):
        argv = ["--config", str(CONFIG), "--internal-files", "x", "--external-files", "y", "--out", str(tmp / "s"), *flag]
        try:
            import contextlib
            with contextlib.redirect_stderr(io.StringIO()):
                geo.main(argv)
            raise CheckFailed(f"{flag} accepted")
        except SystemExit as e:
            ensure(e.code == 2, f"{flag}: exit {e.code}")
            out[flag[0]] = "argparse error (exit 2)"
    ensure(set(ra.LEGACY_GEOMETRY_KEYS) >= {"p_int_um", "p_ext_um", "allow_config_spacing"}, "driver key list")
    ensure(not set(cfg()["geometry"]) & set(ra.LEGACY_GEOMETRY_KEYS), "config still has legacy geometry keys")
    return out


@check("B7", "optics config errors: missing block, binary float, binning/native mismatch, implausible spacing")
def b7() -> Dict[str, Any]:
    c = cfg()
    raises(lambda: optics.load_optics({k: v for k, v in c.items() if k != "optics"}), optics.OpticsError, "no 'optics'")
    bad = json.loads(json.dumps(c))
    bad["optics"]["external"]["sensor_pitch_um"] = 2.74
    raises(lambda: optics.load_optics(bad), optics.OpticsError, "exact decimal")
    bad = json.loads(json.dumps(c))
    bad["optics"]["external"]["binning"] = 1
    raises(lambda: optics.load_optics(bad), optics.OpticsError, "binning")
    bad = json.loads(json.dumps(c))
    bad["optics"]["external"]["total_magnification"] = "0.63"
    raises(lambda: optics.load_optics(bad), optics.OpticsError, "outside")
    bad = json.loads(json.dumps(c))
    bad["metadata_policy"]["external"] = "fallback"
    raises(lambda: optics.load_optics(bad), optics.OpticsError, "metadata_policy")
    return {"refused": 5}


@check("B8", "no fallback symbols remain (AST of geometry_harmonization.py)")
def b8() -> Dict[str, Any]:
    tree = ast.parse((HERE / "geometry_harmonization.py").read_text())
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | \
            {t.id for n in ast.walk(tree) if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
    args = {a.arg for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) for a in n.args.args + n.args.kwonlyargs}
    banned = {"CONFIG_SPACING_UM", "CONFIG_SPACING_HALF_UNIT", "EXTERNAL_MATCHED_M", "M_LEVELS"}
    ensure(not names & banned, f"banned names present: {names & banned}")
    ensure(not args & {"override_um", "allow_config_fallback", "allow_config_spacing", "p_int", "p_ext"},
           f"fallback parameters present: {args}")
    return {"banned_absent": sorted(banned)}


@check("B9", "calibration refuses mismatched spacings; kernel scales in micrometres")
def b9() -> Dict[str, Any]:
    import photometric_calibration as pc
    mp = pc.MaskParams()
    good = pc.kernel_scales_um(mp, 1124 * float(P_INT) / 384, 2256 * float(P_EXT) / 384)
    ensure(abs(good["relative_spacing_difference"]) < geo.SPACING_TOLERANCE, "true geometry should pass")
    ensure(abs(good["external_frames"]["dilate_radius_um"] - 3 * 0.5110317) < 1e-6, "dilation in um")
    legacy = pc.kernel_scales_um(mp, 554 * 0.175 / 384, 2256 * float(P_EXT) / 384)
    ensure(abs(legacy["relative_spacing_difference"]) >= geo.SPACING_TOLERANCE, "legacy tiles should be refused")
    src = (HERE / "photometric_calibration.py").read_text()
    ensure("refuses to\nrun when the two spacings differ" in src or "differ by >= 0.1%" in src, "refusal not in code")
    return {"true_relative_difference": good["relative_spacing_difference"],
            "legacy_relative_difference": legacy["relative_spacing_difference"],
            "kernel_um_external": good["external_frames"]}


# =============================================================================
# C. Monotonic physical scale grid
# =============================================================================

@check("C1", "levels strictly ordered (T down, m up); external-matched level in place; tiles and margins valid")
def c1() -> Dict[str, Any]:
    lv = geo.build_levels(1124)
    Ts, ms = [x.T for x in lv], [x.m for x in lv]
    ensure(Ts == [1200, 1124, 960, 800, 686, 600], f"T order {Ts}")
    ensure(all(a < b for a, b in zip(ms, ms[1:])), f"m order {ms}")
    ensure([x.role for x in lv].count("external_matched") == 1 and lv[1].role == "external_matched", "ext role")
    ensure([x.n_tiles for x in lv] == [1, 4, 4, 4, 4, 4], "tiles per field")
    for x in lv:
        plan = geo.make_axis_plan(x.T)
        ensure(-plan.tap_min <= geo.MARGIN_PX and plan.tap_max_over <= geo.MARGIN_PX, f"margin at T={x.T}")
        ensure(plan.n_guard == 0, f"guard hits at T={x.T}")
        cov = np.zeros(geo.N_INT, bool)
        for o in x.offsets:
            cov[o:o + x.T] = True
        ensure(cov.all() and x.offsets[0] == 0 and x.offsets[-1] + x.T == geo.N_INT, f"coverage T={x.T}")
    return {"T": Ts, "m": [round(m, 5) for m in ms], "labels": [x.label for x in lv]}


@check("C2", "supplementary legacy level sorts last; collisions merge or fail; disorder is caught")
def c2() -> Dict[str, Any]:
    s = geo.build_levels(1124, supplementary=(554,))
    ensure([x.T for x in s][-1] == 554 and s[-1].role == "supplementary" and s[-1].n_tiles == 9, "supplementary")
    merged = geo.build_levels(600)
    ensure(len(merged) == 5 and [x.T for x in merged if x.role == "external_matched"] == [600], "merge")
    raises(lambda: geo.build_levels(1124, supplementary=(960,)), ValueError, "duplicates")
    raises(lambda: geo.build_levels(554, supplementary=(554,)), ValueError, "equals")
    raises(lambda: geo.assert_monotonic([geo.make_level(600), geo.make_level(1200)]), ValueError, "strictly")
    raises(lambda: geo.make_level(2.165), TypeError, "integer")
    return {"with_supplementary": [x.T for x in s], "merged_at_600": [x.T for x in merged]}


@check("C3", "Part A condition grid: 34 unique T-keyed identifiers, aliases on the right levels, no legacy names")
def c3() -> Dict[str, Any]:
    import run_part_a_internal_simulation as pa
    lv = geo.build_levels(1124)
    conds, al = pa.build_conditions(lv, "deadbeef")
    cids = [c.cid for c in conds]
    ensure(len(cids) == 34 == len(set(cids)), "34 unique conditions")
    ensure(al["C0"] == "T1200_a0_point" and al["C-S"] == "T1124_a0_point", f"aliases {al}")
    ensure(al["C-SI"] == "T1124_a1_point_caldeadbeef" and al["C-SI@extreme_low"].startswith("T1124_a1_extreme_low"), "C-SI")
    ensure(all(re.fullmatch(r"T\d+_a[0-9.]+_(point|extreme_low|extreme_high)(_cal[0-9a-f]{8})?", c) for c in cids), "format")
    ensure(not any(c.startswith("m") for c in cids), "legacy m-names")
    ensure(all(("_cal" in c) == (cd.alpha > 0) for c, cd in zip(cids, conds)), "calibration tag rule")
    s_conds, _ = pa.build_conditions(geo.build_levels(1124, (554,)), "deadbeef")
    ensure(len(s_conds) == 35 and s_conds[-1].cid == "T554_a0_point", "supplementary condition")
    return {"n": len(cids), "aliases": {k: al[k] for k in ("C0", "C-S", "C-I", "C-SI")}}


@check("C4", "summary surface and figures refuse non-monotonic rows")
def c4() -> Dict[str, Any]:
    import export_manuscript_artifacts as ex
    surf = {"rows_m": [1.0, 2.165, 1.25], "rows_role": ["nominal"] * 3, "cols_alpha": [0, 1],
            "grid": [[None, None]] * 3}
    with tempfile.TemporaryDirectory() as d:
        raises(lambda: ex.fig_dose_response({"sensitivity_surface": {"hybrid43": surf, "densenet121": surf}}, Path(d)),
               ex.ExportConsistencyError, "strictly increasing")
    pa_src = (HERE / "run_part_a_internal_simulation.py").read_text()
    ensure("geo.assert_monotonic(surf_levels)" in pa_src, "summary does not assert ordering")
    return {"figure": "refuses", "summary": "asserts"}


# =============================================================================
# D. Physical constants in docstrings, schemas, config and exports
# =============================================================================

CODE = ("optics.py", "geometry_harmonization.py", "photometric_calibration.py", "verify_unit_checks.py",
        "run_part_a_internal_simulation.py", "run_part_b_and_export.py", "export_manuscript_artifacts.py",
        "run_all_colab.py", "profile_colab_t4.py", "migrate_part_ab.py", "partab_colab_t4.py", "colab_config.json",
        "pc_resilience.py", "colab_state.py", "colab_setup_env.py")
OBSOLETE = re.compile(r"2\.165|0\.043\b|0\.253|0\.25263|0\.76 ?um|\b2\.165:|: 9\}|\b9 \* R\[|40 x 9|554\b")


@check("D1", "no obsolete optical constant outside explicitly labelled legacy references")
def d1() -> Dict[str, Any]:
    hits = []
    for f in CODE:
        for i, line in enumerate((HERE / f).read_text().splitlines(), 1):
            if OBSOLETE.search(line) and "legacy" not in line.lower():
                hits.append(f"{f}:{i}: {line.strip()[:100]}")
    ensure(not hits, "obsolete constants:\n  " + "\n  ".join(hits))
    return {"files_scanned": len(CODE), "unlabelled_hits": 0}


@check("D2", "every physical number stated in docstrings equals the computed value")
def d2() -> Dict[str, Any]:
    ai, ae, _ = acqs()
    em = optics.external_matched(ai, ae)
    want = {"0.174603": float(ai.pixel_um), "0.086984": float(ae.pixel_um), "209.52": float(ai.fov_um),
            "196.24": float(ae.fov_um), "1123.90": em["T_star"], "1.0677": em["m_external"],
            "0.5110": em["spacing_external_at_384_um"], "0.5456": em["spacing_internal_full_frame_at_384_um"]}
    checked = {}
    for f in ("optics.py", "geometry_harmonization.py", "run_part_a_internal_simulation.py"):
        doc = ast.get_docstring(ast.parse((HERE / f).read_text())) or ""
        for txt, val in want.items():
            if txt in doc:
                dec = len(txt.split(".")[1])
                ensure(round(val, dec) == float(txt), f"{f}: docstring {txt} != computed {val}")
                checked[f"{f}:{txt}"] = val
    ensure(len(checked) >= 10, f"too few stated constants found: {checked}")
    mp = (HERE / "photometric_calibration.py").read_text()
    blk = mp[mp.index("class MaskParams"):mp.index("def digest")]
    ensure(not re.search(r"\d\.\d+ ?um", blk), "MaskParams still states a fixed micrometre scale")
    return {"constants_checked": len(checked)}


@check("D3", "spec / level / config / notebook schemas carry the physical fields and safe defaults")
def d3(tmp: Path) -> Dict[str, Any]:
    ints = [tif(tmp / "i.tif", 1200, kind="imagej", um=float(P_INT))]
    exts = [tif(tmp / "e.tif", 2256, kind="ome", um=float(P_EXT))]
    spec = geo.build_geometry_spec(ints, exts, cfg=cfg(), expected_opencv=None, record_opencv=False)
    for k in ("schema", "optics", "metadata_policy", "external_matched", "levels", "geometry_digest",
              "external_matched_T_used", "supplementary_T"):
        ensure(k in spec, f"spec lacks {k}")
    for k in ("p_int_exact", "p_ext_exact", "fov_int_um", "fov_ext_um", "T_star_exact", "m_external", "metadata_status"):
        ensure(k in spec["external_matched"], f"external_matched lacks {k}")
    for lv in spec["levels"]:
        ensure({"T", "m", "m_exact", "role", "label", "fov_um", "output_spacing_um"} <= set(lv), "level summary")
    c = cfg()
    ensure(c["options"]["accept_conditional"] is False, "accept_conditional must default to false")
    ensure(c["migration"]["accept_sha256"] is None, "migration must not be pre-accepted")
    nb = (HERE / "partab_colab_t4.py").read_text()
    ensure("ACCEPT_CONDITIONAL = False" in nb and "ACCEPT_MIGRATION = False" in nb, "notebook defaults")
    ensure("spacings_exact" not in nb, "notebook reads a removed key")
    return {"levels": [(lv["T"], lv["label"]) for lv in spec["levels"]]}


@check("D4", "unit check (b) is physical: tampered T or digest fails; agreement passes")
def d4(tmp: Path) -> Dict[str, Any]:
    import verify_unit_checks as vuc
    ints = [tif(tmp / "i.tif", 1200, kind="imagej", um=float(P_INT))]
    exts = [tif(tmp / "e.tif", 2256, kind="ome", um=float(P_EXT))]
    spec = geo.build_geometry_spec(ints, exts, cfg=cfg(), expected_opencv=None, record_opencv=False)
    ok = vuc.check_b_spacing(SimpleNamespace(spec=spec))
    ensure(ok["status"] == vuc.PASS, f"agreeing spec: {ok}")
    bad = json.loads(json.dumps(spec))
    bad["external_matched_T_used"] = 554
    r1 = vuc.check_b_spacing(SimpleNamespace(spec=bad))
    bad2 = json.loads(json.dumps(spec))
    bad2["geometry_digest"] = "0" * 64
    r2 = vuc.check_b_spacing(SimpleNamespace(spec=bad2))
    legacy = {k: v for k, v in spec.items() if k != "schema"}
    r3 = vuc.check_b_spacing(SimpleNamespace(spec=legacy))
    ensure(r1["status"] == r2["status"] == r3["status"] == vuc.FAIL, f"{r1['status']} {r2['status']} {r3['status']}")
    return {"agreeing": ok["status"], "tampered_T": r1["error"], "tampered_digest": r2["error"], "legacy_schema": r3["status"]}


@check("D5", "export refuses mixed geometry / calibration provenance")
def d5() -> Dict[str, Any]:
    import export_manuscript_artifacts as ex
    G = {"geometry_digest": "g1"}
    C = {"geometry": {"geometry_digest": "g1"}}
    A = {"geometry": {"geometry_digest": "g1"}, "calibration": {"sha256": "c1"}}
    PA = {"plan_digest": "pa", "plan": {"geometry_digest": "g1", "calibration_sha256": "c1"}}
    PB = {"plan_digest": "pb", "plan": {"geometry_digest": "g1", "calibration_sha256": "c1", "part_a_plan_digest": "pa"}}
    B = {"plan_digest": "pb"}
    ok = ex.check_consistency(G, A, C, PA, PB, B)
    raises(lambda: ex.check_consistency({"geometry_digest": "g2"}, A, C, PA, PB, B), ex.ExportConsistencyError, "geometry")
    raises(lambda: ex.check_consistency(G, A, {"geometry": {}}, PA, PB, B), ex.ExportConsistencyError, "missing")
    PB2 = json.loads(json.dumps(PB))
    PB2["plan"]["calibration_sha256"] = "c2"
    raises(lambda: ex.check_consistency(G, A, C, PA, PB2, B), ex.ExportConsistencyError, "calibration")
    PB3 = json.loads(json.dumps(PB))
    PB3["plan"]["part_a_plan_digest"] = "old"
    raises(lambda: ex.check_consistency(G, A, C, PA, PB3, B), ex.ExportConsistencyError, "different Part A")
    return {"consistent": ok, "refusals": 4}


# =============================================================================
# E. Digests, cache invariance, migration
# =============================================================================

@check("E1", "geometry digest: deterministic across rebuilds, sensitive to optics and levels only")
def e1(tmp: Path) -> Dict[str, Any]:
    d0 = geo.expected_geometry(cfg())["geometry_digest"]
    ensure(d0 == geo.expected_geometry(cfg())["geometry_digest"], "not deterministic")
    c = cfg()
    c["optics"]["external"]["sensor_pitch_um"] = "2.740"          # same rational: unchanged
    ensure(geo.expected_geometry(c)["geometry_digest"] == d0, "string spelling of the same value changed the digest")
    c["optics"]["external"]["sensor_pitch_um"] = "2.75"
    ensure(geo.expected_geometry(c)["geometry_digest"] != d0, "pitch change not detected")
    c = cfg()
    c["geometry"]["supplementary_T"] = [554]
    ensure(geo.expected_geometry(c)["geometry_digest"] != d0, "level change not detected")
    c = cfg()
    c["metadata_policy"]["internal"] = "required"
    ensure(geo.expected_geometry(c)["geometry_digest"] == d0, "policy (not numerical) changed the digest")
    return {"geometry_digest": d0}


def _dg(**over: Any) -> str:
    import run_part_a_internal_simulation as pa
    base = dict(bases_sha="b" * 64, T=1200, offsets=[0], margin=8, border_rule="reflect101", a_alpha=1.0, b_alpha=0.0,
                models={"hybrid43": {"sha256": "h", "mu": 0.24, "sigma": 0.0074},
                        "densenet121": {"sha256": "d", "mu": 0.24, "sigma": 0.0074}},
                batch_size=32, chunk_tiles=288, shard_tiles=2304, cv2_version="4.10.0")
    base.update(over)
    return pa.condition_digest_v2(**base)


@check("E2", "condition digest v2: invariant to labels, sensitive to every array-determining input")
def e2() -> Dict[str, Any]:
    import run_part_a_internal_simulation as pa
    d0 = _dg()
    ensure(d0 == _dg(), "not deterministic")
    sig = inspect_params(pa.condition_digest_v2)
    ensure(not {"cid", "m", "alias", "geometry_spec", "calibration_sha"} & set(sig), f"label inputs in digest: {sig}")
    changes = {"T": dict(T=1124, offsets=[0, 76]), "offsets": dict(offsets=[1]), "a_alpha": dict(a_alpha=1.0000001),
               "b_alpha": dict(b_alpha=-0.0), "model": dict(models={"hybrid43": {"sha256": "x", "mu": 0.24, "sigma": 0.0074}}),
               "batch": dict(batch_size=16), "chunk": dict(chunk_tiles=576), "cv2": dict(cv2_version="4.11.0"),
               "border": dict(border_rule="replicate"), "operator": dict(operator_version="internal_condition/2")}
    for name, kw in changes.items():
        ensure(_dg(**kw) != d0, f"digest insensitive to {name}")
    return {"insensitive_to": ["cid", "m", "alias", "spec file bytes", "calibration (identity cells)"],
            "sensitive_to": list(changes)}


def inspect_params(fn: Callable[..., Any]) -> List[str]:
    import inspect
    return list(inspect.signature(fn).parameters)


@check("E3", "Part B: E-locked / E-z digests unchanged by the refactor; E-cal tied to the calibration digest")
def e3() -> Dict[str, Any]:
    import photometric_calibration as pc
    import run_part_b_and_export as pb
    rel, ids = pb.digest_of(["a/1.tif", "b/2.tif"]), {"hybrid43": ["h", 0.24, 0.0074]}
    out = {}
    for name in ("E-locked", "E-z"):
        mode, src, label = pb.PIPELINES[name]
        params = {"pipeline": name, "mode": mode, "ab_source": src, "preprocessing_mode": label}
        v1 = pb.digest_of({"schema": "part_b_and_export/1.0", "p": params, "images": rel,
                           "pinned": pc.PINNED_PREPROCESS_SHA256, "models": dict(sorted(ids.items())), "bs": 32})
        ensure(pb.pipe_digest_from(params, rel, ids, 32) == v1 and pb.SCHEMA == "part_b_and_export/1.0", name)
        out[name] = "unchanged"
    cal = {"point": {"a": 1.3, "b": 0.1}, "bootstrap": {"extreme_low": {"a": 1.2, "b": 0.09},
                                                        "extreme_high": {"a": 1.4, "b": 0.11}}}
    ctx = SimpleNamespace(cal=cal)
    p1 = pb.pipe_params(ctx, "E-cal")
    cal2 = json.loads(json.dumps(cal))
    cal2["point"]["a"] = 1.3000001
    p2 = pb.pipe_params(SimpleNamespace(cal=cal2), "E-cal")
    ensure(p1["calibration_digest"] != p2["calibration_digest"], "E-cal digest insensitive to a")
    ensure("calibration_digest" not in pb.pipe_params(ctx, "E-locked"), "E-locked carries calibration")
    c3 = {**cal, "created_utc": "other", "counts": {}}
    ensure(pc.calibration_digest(c3) == pc.calibration_digest(cal), "calibration digest depends on metadata")
    return out


@check("E4", "driver stage digests follow file content, not paths")
def e4(tmp: Path) -> Dict[str, Any]:
    import run_all_colab as ra
    f = tmp / "spec.json"
    f.write_text('{"x": 1}')
    d1 = ra.content_digest(["--geometry-spec", str(f)], {"spec": f})
    f.write_text('{"x": 2}')
    d2 = ra.content_digest(["--geometry-spec", str(f)], {"spec": f})
    d3 = ra.content_digest(["--geometry-spec", str(f)], {"spec": tmp / "absent.json"})
    ensure(len({d1, d2, d3}) == 3, "content change not detected")
    return {"distinct_digests": 3}


@check("E5", "migration: snapshot -> plan -> apply on a synthetic legacy tree")
def e5(tmp: Path) -> Dict[str, Any]:
    import colab_state as cs
    import migrate_part_ab as mig
    import run_part_a_internal_simulation as pa
    import verify_unit_checks as vuc

    root = tmp / "drive"
    bases = ["Candida albicans/20250101_alb_A_1", "Candida albicans/20250101_alb_A_2",
             "Candida glabrata/20250102_gla_A_1"]
    split = tmp / "split.json"
    split.write_text(json.dumps({"train_base_paths": [], "test_base_paths": bases}))
    bsha = pa.digest_of(sorted(bases, key=lambda b: (vuc.slide_of(b), b)))
    models = {"hybrid43": {"sha256": "h", "mu": 0.24, "sigma": 0.0074},
              "densenet121": {"sha256": "d", "mu": 0.24, "sigma": 0.0074}}
    torch_env = {"torch": "2.9", "cuda": "12.6", "gpu": "Tesla T4"}

    def legacy_cond(cid: str, T: int, offsets: List[int], a: float, b: float, tamper: bool = False) -> None:
        d = root / "part_a" / "conditions" / cid
        nt = len(offsets) ** 2
        payload = {"preproc.npz": cs.npz_bytes(field_idx=np.repeat(np.arange(3), nt), frac_clipped=np.zeros(3 * nt)),
                   **{f"{k}.npz": cs.npz_bytes(logits=np.random.default_rng(T).normal(size=(3 * nt, 2)).astype(np.float32))
                      for k in models}}
        cs.write_verified_files(d, payload, "DONE.json", {"digest": f"v1-{cid}", "shard_tiles": 2304,
                                                          "params": {"cid": cid, "m": 1200 / T, "T": T, "offsets": offsets,
                                                                     "alpha": 0.0 if a == 1 else 1.0,
                                                                     "a_alpha": a, "b_alpha": b}})
        if tamper:
            (d / "hybrid43.npz").write_bytes(b"tampered")

    legacy_cond("m1_a0_point", 1200, [0], 1.0, 0.0)
    legacy_cond("m1.25_a0_point", 960, [0, 240], 1.0, 0.0)
    legacy_cond("m2.165_a0_point", 554, [0, 323, 646], 1.0, 0.0)          # legacy level
    legacy_cond("m1_a1_point", 1200, [0], 1.31, 0.137)
    legacy_cond("m1.5_a0_point", 800, [0, 400], 1.0, 0.0, tamper=True)
    (root / "part_a" / "calibration").mkdir(parents=True)
    (root / "part_a" / "calibration" / "calibration.json").write_text(json.dumps({"point": {}}))
    for rel, obj in (("part_a/part_a_plan_frozen.json",
                      {"plan_digest": "legacyplan", "plan": {"models": models, "data": {"heldout_bases_digest": bsha},
                                                             "border_rule": "reflect101", "batch_size": 32}}),
                     ("part_a/part_a_summary.json", {"args": {"chunk_tiles": "288"}, "gate": {"cv2_version": "4.10.0"},
                                                     "environment": {"torch_config": torch_env}})):
        (root / rel).write_text(json.dumps(obj))
    legacy_spec = {"external_matched_T_used": 554, "external_matched": {"p_ext_um": 0.043, "p_ext_source": "config_rounded_3dp"}}
    (root / "geometry_spec.json").write_text(json.dumps(legacy_spec))

    snap = mig.snapshot(root, root / "geometry_spec.json")
    ensure(snap["legacy_geometry"]["outcome"] == "L", f"outcome {snap['legacy_geometry']}")
    # new spec and unit report (synthetic headers)
    ints = [tif(tmp / "i.tif", 1200, kind="imagej", um=float(P_INT))]
    exts = [tif(tmp / "e.tif", 2256, kind="ome", um=float(P_EXT))]
    spec = geo.build_geometry_spec(ints, exts, cfg=cfg(), expected_opencv=None, record_opencv=False)
    (root / "geometry_spec.json").write_text(json.dumps(spec))
    ud = root / "unit_checks"
    ud.mkdir()
    (ud / "unit_check_report.json").write_text(json.dumps({"checks": {
        "a2_C0_frozen_predictions": {"constants": models, "torch": torch_env},
        "d0_border_rule_and_cv2_parity": {"selected_border_rule": "reflect101"},
        "environment": {"opencv": {"cv2_version": "4.10.0"}}}}))
    plan = mig.make_plan(root, root / "geometry_spec.json", ud, split)
    cls = {e["legacy_cid"]: e["class"] for e in plan["conditions"]}
    ensure(cls == {"m1_a0_point": "INVARIANT", "m1.25_a0_point": "INVARIANT", "m2.165_a0_point": "RETIRED",
                   "m1_a1_point": "RECOMPUTE", "m1.5_a0_point": "REJECT"}, f"classes {cls}")
    conflict = root / "part_a" / "conditions_v2" / "T960_a0_point"
    cs.write_verified_files(conflict, {"preproc.npz": b"other"}, "DONE.json", {"digest": "x"})
    sha = cs.sha256_file(root / "migration" / "part_a_plan.json")
    raises(lambda: mig.apply(root, cfg(), sha), RuntimeError, "refusing to overwrite")
    for p in conflict.iterdir():
        p.unlink()
    conflict.rmdir()
    partial = root / "part_a" / "conditions_v2" / "T1200_a0_point"     # any partial result of the refused run
    if partial.exists():
        for p in partial.iterdir():
            p.unlink()
        partial.rmdir()
    raises(lambda: mig.apply(root, cfg(), "0" * 64), RuntimeError, "does not match")
    rec = mig.apply(root, cfg(), sha)
    ensure(sorted(rec["resigned"]) == ["T1200_a0_point", "T960_a0_point"], f"resigned {rec['resigned']}")
    lv = {x.T: x for x in geo.levels_from_spec(spec)}
    for e in plan["conditions"]:
        if e["class"] != "INVARIANT":
            continue
        d = root / "part_a" / "conditions_v2" / e["new_cid"]
        done = json.loads((d / "DONE.json").read_text())
        ensure(cs.verify_files(d, done), "re-signed files fail verification")
        ensure(all(done["files"][n]["sha256"] == e["files"][n]["sha256"] for n in e["files"]), "bytes changed")
        expect = pa.condition_digest_v2(bases_sha=bsha, T=e["T"], offsets=lv[e["T"]].offsets, margin=8,
                                        border_rule="reflect101", a_alpha=1.0, b_alpha=0.0, models=models, batch_size=32,
                                        chunk_tiles=288, shard_tiles=2304, cv2_version="4.10.0")
        ensure(done["digest"] == expect and done["migrated_from"]["legacy_cid"] == e["legacy_cid"], "digest/provenance")
    ensure((root / "legacy_v1" / "part_a" / "conditions" / "m2.165_a0_point" / "DONE.json").exists(), "legacy not archived")
    ensure(not (root / "part_a" / "conditions").exists(), "legacy directory left where the new run looks")
    raises(lambda: mig.apply(root, cfg(), sha), RuntimeError, "already applied")
    return {"classes": cls, "resigned": rec["resigned"], "archived": rec["moved_to_legacy_v1"]}


# =============================================================================
# Runner
# =============================================================================

def header_mode(files: List[Path]) -> None:
    ai, ae, pol = acqs()
    for f in files:
        rec = geo.read_spacing_candidates(f)
        acq = ae if rec["shape"][-1] == ae.n_px else ai
        cmp = [optics.compare_metadata(acq, v, k, optics.exact(pol["rel_tol"])) for k, v in rec["candidates"].items()]
        print(json.dumps({"file": str(f), "domain_by_shape": acq.domain, "shape": rec["shape"],
                          "candidates": rec["candidates"], "rejected": rec["rejected"],
                          "anisotropic": rec["anisotropic"], "comparison": cmp}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--header", nargs="+", type=Path, default=None, help="Print the spacing evidence of real TIFFs")
    a = ap.parse_args()
    if a.header:
        header_mode(a.header)
        return
    import inspect
    checks = [v for v in globals().values() if callable(v) and hasattr(v, "_cid")]
    for fn in checks:
        rec: Dict[str, Any] = {"id": fn._cid, "title": fn._title}
        try:
            if "tmp" in inspect.signature(fn).parameters:
                with tempfile.TemporaryDirectory() as d:
                    rec["detail"] = fn(Path(d))
            else:
                rec["detail"] = fn()
            rec["status"] = "PASS"
        except Exception as exc:  # recorded, reported, exit code 1
            rec["status"] = "FAIL"
            rec["error"] = f"{type(exc).__name__}: {exc}"
            rec["traceback"] = traceback.format_exc()
        RESULTS.append(rec)
        print(f"{rec['status']:<5} {rec['id']:<3} {rec['title']}" + (f"\n      -> {rec['error']}" if "error" in rec else ""))
    n_fail = sum(r["status"] != "PASS" for r in RESULTS)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    if a.json:
        a.json.write_text(json.dumps(RESULTS, indent=1, default=str))
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
