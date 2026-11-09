#!/usr/bin/env python3
"""
export_manuscript_artifacts.py

Reads only finished JSON/CSV outputs on Drive. Before writing anything it checks that
geometry_spec.json, calibration.json, the Part A plan and summary and the Part B plan
all carry the same geometry digest and calibration (fail loud otherwise).

  <drive_root>/manuscript_v2/tables/
      tab_partab_calibration.tex        optics (pitch, binning, magnification, spacing, FOV),
                                        external-matched T and m, and the label-free calibration
      tab_part_a_factorial.tex          C0, C-S, C-I, C-SI per model (slide BA, default-species slides, y, g, clip, R1-R3)
      tab_part_a_contrasts.tex          d_S, d_I, d_SxI: exact sign-flip p and Holm (primary y, secondary g)
      tab_part_a_uncertainty.tex        C-I and C-SI at the joint bootstrap extremes of (a, b)
      tab_part_b_metrics.tex            E-locked, E-cal, E-z per model: slide metrics with intervals
      tab_part_b_pipeline_contrasts.tex E-cal - E-locked, E-z - E-cal, E-z - E-locked (McNemar, sign-flip, Holm)
      tab_part_b_model_contrasts.tex    DenseNet-121 minus ViT+CCT+LSA_43 per pipeline (Holm m = 3)
      tab_part_b_recovery.tex           recovery ratio and the within-session slide E-alb_B
  <drive_root>/manuscript_v2/csv/          the same tables as CSV (full precision)
  <drive_root>/manuscript_v2/figs/
      fig_part_a_dose_response.{pdf,png}       slide BA versus alpha, one line per physical scale m = 1200/T
      fig_part_b_slide_probabilities.{pdf,png} 24 slide probabilities under L, C, Z per model
      part_a_sensitivity_surface.{pdf,png}     copied from part_b/figs
  <drive_root>/manuscript_v2/EXPORT_REPORT.md      key numbers, gate status, file list
  <drive_root>/manuscript_v2/ATTACH_MANIFEST.json  SHA-256 and size of every bundled file

Author: Rodrigo Sá
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

MODELS = ("hybrid43", "densenet121")
MODEL_TEX = {"hybrid43": r"ViT+CCT+LSA$_{43}$", "densenet121": "DenseNet-121"}
MODEL_TXT = {"hybrid43": "ViT+CCT+LSA_43", "densenet121": "DenseNet-121"}
PRIMARY = ("C0", "C-S", "C-I", "C-SI")
PIPES = ("E-locked", "E-cal", "E-z")
PIPE_CONTRASTS = ("E-cal-minus-E-locked", "E-z-minus-E-cal", "E-z-minus-E-locked")
CONTRAST_TEX = {"E-cal-minus-E-locked": r"E-cal $-$ E-locked", "E-z-minus-E-cal": r"E-z $-$ E-cal",
                "E-z-minus-E-locked": r"E-z $-$ E-locked"}
D_TEX = {"d_S": r"$d_S$", "d_I": r"$d_I$", "d_SxI": r"$d_{S\times I}$"}

# Reference palette
BLUE, ORANGE = "#2a78d6", "#eb6834"
BLUE_RAMP = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b")   # steps 250 to 700, assigned to
                                                                                    # levels in ascending physical m
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


# -----------------------------------------------------------------------------
# Formatting helpers
# -----------------------------------------------------------------------------

def num(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def f4(x: Any, d: int = 4) -> str:
    v = num(x)
    return "--" if v is None else f"{v:.{d}f}"


def fs(x: Any, d: int = 3) -> str:
    v = num(x)
    return "--" if v is None else f"{v:+.{d}f}"


def ci(lo: Any, hi: Any, d: int = 4) -> str:
    return "--" if num(lo) is None or num(hi) is None else f"[{f4(lo, d)}, {f4(hi, d)}]"


def pct(x: Any, d: int = 2) -> str:
    v = num(x)
    return "--" if v is None else f"{100 * v:.{d}f}"


def pv(x: Any) -> str:
    v = num(x)
    if v is None:
        return "--"
    return "$<$0.0001" if v < 1e-4 else f"{v:.4f}"


def yn(b: Any) -> str:
    return "--" if b is None else ("yes" if b else "no")


def g(d: Any, *keys: Any, default: Any = None) -> Any:
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default
    return d


def tex_table(label: str, caption: str, colspec: str, header: Sequence[str], rows: List[Sequence[str]],
              resize: bool = True, notes: Optional[str] = None) -> str:
    out = [r"\begin{table}[tbp]", r"\centering", r"\caption{" + caption + "}", r"\label{" + label + "}",
           r"\setlength{\tabcolsep}{4pt}"]
    if resize:
        out.append(r"\resizebox{\textwidth}{!}{%")
    out += [r"\begin{tabular}{" + colspec + "}", r"\hline", " & ".join(header) + r" \\", r"\hline"]
    for r in rows:
        out.append(r"\hline" if r == ["__RULE__"] else " & ".join(r) + r" \\")
    out += [r"\hline", r"\end{tabular}%" if resize else r"\end{tabular}"]
    if resize:
        out.append("}")
    if notes:
        out.append(r"\par\vspace{2pt}{\footnotesize " + notes + "}")
    out.append(r"\end{table}")
    return "\n".join(out) + "\n"


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols: List[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


# -----------------------------------------------------------------------------
# Part A tables
# -----------------------------------------------------------------------------

class ExportConsistencyError(RuntimeError):
    pass


def check_consistency(G: Dict[str, Any], A: Dict[str, Any], C: Dict[str, Any], PA: Dict[str, Any],
                      PB: Optional[Dict[str, Any]], B: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Every exported number must come from one geometry and one calibration (raises otherwise)."""
    gd = {"geometry_spec": G.get("geometry_digest"), "calibration": (C.get("geometry") or {}).get("geometry_digest"),
          "part_a_plan": (PA.get("plan") or {}).get("geometry_digest"),
          "part_a_summary": (A.get("geometry") or {}).get("geometry_digest")}
    if PB is not None:
        gd["part_b_plan"] = (PB.get("plan") or {}).get("geometry_digest")
    if None in gd.values() or len(set(gd.values())) != 1:
        raise ExportConsistencyError(f"geometry digests disagree or are missing (legacy outputs?): {gd}")
    cal = {"part_a_plan": (PA.get("plan") or {}).get("calibration_sha256"),
           "part_a_summary": (A.get("calibration") or {}).get("sha256")}
    if PB is not None:
        cal["part_b_plan"] = (PB.get("plan") or {}).get("calibration_sha256")
    if None in cal.values() or len(set(cal.values())) != 1:
        raise ExportConsistencyError(f"calibration SHA-256 disagrees: {cal}")
    if PB is not None and (PB.get("plan") or {}).get("part_a_plan_digest") != PA.get("plan_digest"):
        raise ExportConsistencyError("Part B plan was frozen against a different Part A plan")
    if B is not None and B.get("plan_digest") != (PB or {}).get("plan_digest"):
        raise ExportConsistencyError("Part B summary does not belong to the frozen Part B plan")
    return {"geometry_digest": gd["geometry_spec"], "calibration_sha256": cal["part_a_plan"]}


def cond_by_alias(A: Dict[str, Any], alias: str) -> Optional[Dict[str, Any]]:
    for rec in A.get("conditions", {}).values():
        if rec.get("alias") == alias:
            return rec
    return None


def table_calibration(A: Dict[str, Any], G: Dict[str, Any], out: Path) -> List[Dict[str, Any]]:
    cal, em, op = A.get("calibration", {}), G.get("external_matched", {}), G.get("optics", {})
    pt, sens = cal.get("point", {}), cal.get("sensitivity_contrast_definition") or {}
    lo, hi = cal.get("extreme_low") or {}, cal.get("extreme_high") or {}
    a95, b95 = cal.get("a_percentile_95") or [None, None], cal.get("b_percentile_95") or [None, None]
    noise = cal.get("factor_N_noise") or {}
    ms = em.get("metadata_status") or {}
    rows: List[Dict[str, Any]] = []
    for d in ("internal", "external"):
        o = op.get(d, {})
        rows += [
            {"quantity": f"{d} camera", "value": o.get("camera"), "detail": ""},
            {"quantity": f"{d} sensor pitch (um) x binning / total magnification",
             "value": f"{o.get('sensor_pitch_um')} x {o.get('binning')} / {o.get('total_magnification')}",
             "detail": f"native {o.get('native_n_px')} px" if o.get("native_n_px") else ""},
            {"quantity": f"{d} object-plane spacing (um/px)", "value": o.get("pixel_um"), "detail": o.get("pixel_um_exact")},
            {"quantity": f"{d} frame side (px) and FOV (um)", "value": o.get("n_px"), "detail": o.get("fov_um")},
            {"quantity": f"{d} pixel-size metadata", "value": ms.get(d), "detail": (em.get("metadata_value_um") or {}).get(d)},
        ]
    rows += [
        {"quantity": "T* = N_ext p_ext / p_int (px)", "value": em.get("T_star"), "detail": em.get("T_star_exact")},
        {"quantity": "External-matched tile T (px)", "value": G.get("external_matched_T_used"), "detail": ""},
        {"quantity": "External-matched physical scale m = N_int / T*", "value": em.get("m_external"), "detail": ""},
        {"quantity": "Spacing at 384 px: external / matched tile / internal full frame (um/px)",
         "value": em.get("spacing_external_at_384_um"),
         "detail": [em.get("spacing_tile_at_384_um"), em.get("spacing_internal_full_frame_at_384_um")]},
        {"quantity": "Relative spacing residual at 384 px (exact optics)", "value": em.get("relative_residual"), "detail": ""},
        {"quantity": "M_int (internal background)", "value": pt.get("M_int"), "detail": ""},
        {"quantity": "C_int (internal contrast)", "value": pt.get("C_int"), "detail": ""},
        {"quantity": "M_ext (external background)", "value": pt.get("M_ext"), "detail": ""},
        {"quantity": "C_ext (external contrast)", "value": pt.get("C_ext"), "detail": ""},
        {"quantity": "a (gain)", "value": pt.get("a"), "detail": a95},
        {"quantity": "b (offset)", "value": pt.get("b"), "detail": b95},
        {"quantity": "Joint extreme, 2.5th percentile of s* (a, b)", "value": lo.get("a"), "detail": lo.get("b")},
        {"quantity": "Joint extreme, 97.5th percentile of s* (a, b)", "value": hi.get("a"), "detail": hi.get("b")},
    ]
    for name in ("P1", "P10", "smoothed_P5"):
        rows.append({"quantity": f"a with contrast {name}", "value": g(sens, name, "a"), "detail": g(sens, name, "b")})
    for key in ("noise_var", "noise_mad2"):
        rows.append({"quantity": f"Noise ratio external/internal ({key}, internal units)",
                     "value": g(noise, key, "ratio_external_over_internal_internal_units"), "detail": ""})
    for k in MODELS:
        rows.append({"quantity": f"Training-mean shift under x' = a x + b, {MODEL_TXT[k]} (training SD)",
                     "value": g(cal, "training_mean_shift_under_forward_map", k, "delta_training_mean_in_training_sd"),
                     "detail": ""})
    write_csv(out / "csv" / "tab_partab_calibration.csv", rows)

    def val(x: Any) -> str:
        if isinstance(x, bool):
            return yn(x)
        v = num(x)
        return "--" if v is None else (f"{v:.0f}" if v == int(v) and abs(v) > 10 else f"{v:.5g}")

    oi, oe = op.get("internal", {}), op.get("external", {})

    def optic(o: Dict[str, Any]) -> str:
        return (f"{o.get('sensor_pitch_um')}$\\times${o.get('binning')}/{o.get('total_magnification')} = "
                f"{f4(o.get('pixel_um'), 6)}; FOV {f4(o.get('fov_um'), 2)}")

    tex_rows = [
        [r"Internal: pitch$\times$binning/magnification ($\mu$m) = spacing ($\mu$m/px); FOV ($\mu$m)", optic(oi),
         f"metadata: {ms.get('internal', '--')}"],
        [r"External: pitch$\times$binning/magnification ($\mu$m) = spacing ($\mu$m/px); FOV ($\mu$m)", optic(oe),
         f"metadata: {ms.get('external', '--')}"],
        [r"$T^{*}=N_{\mathrm{ext}}\,p_{\mathrm{ext}}/p_{\mathrm{int}}$ (px); $T$ used; $m = N_{\mathrm{int}}/T^{*}$",
         f"{f4(em.get('T_star'), 2)}; {val(G.get('external_matched_T_used'))}; {f4(em.get('m_external'), 4)}",
         "exact optics"],
        [r"Spacing at 384 px, external / matched tile ($\mu$m/px)",
         f"{f4(em.get('spacing_external_at_384_um'), 4)} / {f4(em.get('spacing_tile_at_384_um'), 4)}",
         f"residual {pct(em.get('relative_residual'), 3)}\\% (criterion $<0.1$\\%)"],
        [r"$M_{\mathrm{int}}$, $C_{\mathrm{int}}$", f"{f4(pt.get('M_int'), 5)}, {f4(pt.get('C_int'), 5)}", "48 development slides"],
        [r"$M_{\mathrm{ext}}$, $C_{\mathrm{ext}}$", f"{f4(pt.get('M_ext'), 5)}, {f4(pt.get('C_ext'), 5)}", "24 external slides, unlabeled"],
        ["$a$ [95\\% bootstrap]", f4(pt.get("a")), ci(*a95)],
        ["$b$ [95\\% bootstrap]", f4(pt.get("b")), ci(*b95)],
        ["Joint extremes $(a, b)$, low / high", f"({f4(lo.get('a'))}, {f4(lo.get('b'))})",
         f"({f4(hi.get('a'))}, {f4(hi.get('b'))})"],
        ["$a$ with P1 / P10 / smoothed P5 contrast", " / ".join(f4(g(sens, n, "a")) for n in ("P1", "P10", "smoothed_P5")), "sensitivity"],
        ["Noise power ratio, external/internal", f4(g(noise, "noise_var", "ratio_external_over_internal_internal_units"), 3),
         "high-pass variance, internal units"],
    ]
    for k in MODELS:
        tex_rows.append([f"Shift of the training mean, {MODEL_TEX[k]}",
                         fs(g(cal, "training_mean_shift_under_forward_map", k, "delta_training_mean_in_training_sd"), 2),
                         "units of the training SD"])
    cap = (r"Acquisition geometry, external-matched tiling and label-free affine calibration $x' = a x + b$ between "
           r"the internal and external detectors. Object-plane spacing is sensor pitch $\times$ binning / total "
           r"magnification (external: $2\times2$ binning, 4512 to 2256 px); pixel-size metadata is checked against it. "
           r"Bootstrap intervals resample slides within each domain ($B = 10{,}000$, seed 42); "
           r"the joint extremes are the replicates at the 2.5th and 97.5th percentiles of "
           r"$s^{*} = a^{*} M_{\mathrm{int}} + b^{*}$.")
    (out / "tables" / "tab_partab_calibration.tex").write_text(
        tex_table("tab:partab_calibration", cap, "lll", [r"\textbf{Quantity}", r"\textbf{Value}", r"\textbf{Interval or note}"],
                  tex_rows, resize=False), encoding="utf-8")
    return rows


def table_factorial(A: Dict[str, Any], out: Path) -> List[Dict[str, Any]]:
    geom = A.get("geometry", {})
    lv = {x["T"]: x for x in geom.get("levels", [])}.get(geom.get("T_external_matched"), {})
    EXT_NOTE = (f"$T = {geom.get('T_external_matched')}$, $m = {f4(geom.get('m_external'), 4)}$, "
                f"{f4(lv.get('output_spacing_um'), 3)}~$\\mu$m/px, FOV {f4(lv.get('fov_um'), 1)}~$\\mu$m")
    rows, tex = [], []
    for k in MODELS:
        for n in PRIMARY:
            c = cond_by_alias(A, n)
            if c is None:
                continue
            mk = c["models"][k]
            rep = mk.get("reproduction", {})
            r = {"model": MODEL_TXT[k], "condition": n, "T": c["params"]["T"], "m": c["params"]["m"],
                 "level_role": c["params"]["level_role"], "alpha": c["params"]["alpha"],
                 "a_alpha": c["params"]["a_alpha"], "b_alpha": c["params"]["b_alpha"],
                 "slide_balanced_accuracy": mk["classification"]["balanced_accuracy"],
                 "n_slides_pred_default": rep.get("n_slides_pred_default"), "default_species": rep.get("default_species"),
                 "mean_y": mk["mean_y"], "mean_g": mk["mean_g"],
                 "mean_frac_pixels_clipped": c["clip"]["mean_frac_pixels_clipped"],
                 "R1": rep.get("R1"), "R2": rep.get("R2"), "R3": rep.get("R3"), "reproduces": rep.get("reproduces")}
            rows.append(r)
            tex.append([MODEL_TEX[k] if n == "C0" else "", n, f"{c['params']['T']}", f"{c['params']['m']:.4f}",
                        f"{c['params']['alpha']:g}",
                        f4(r["slide_balanced_accuracy"]), f"{r['n_slides_pred_default']}/12",
                        fs(r["mean_y"], 2), fs(r["mean_g"], 2), pct(r["mean_frac_pixels_clipped"], 3),
                        "--" if n == "C0" else "".join("Y" if rep.get(x) else "N" for x in ("R1", "R2", "R3"))])
        tex.append(["__RULE__"])
    if tex and tex[-1] == ["__RULE__"]:
        tex.pop()
    write_csv(out / "csv" / "tab_part_a_factorial.csv", rows)
    cap = (r"Part A factorial conditions on the internal held-out cohort (12 slides, 821 fields). $T$ is the tile "
           r"side in internal pixels resampled to 384 px and $m = 1200/T$ the physical scale; C-S and C-SI use the "
           r"external-matched tile (" + EXT_NOTE + r"). $\alpha$ is the dose of the calibrated intensity map. $y$ is the slide mean of the "
           r"true-class log-odds and $g$ the slide mean of the \textit{C. glabrata} log-odds. Default-species slides "
           r"count slides predicted as the species each model assigns to the external cohort under E-locked "
           r"(\textit{C. albicans} for ViT+CCT+LSA$_{43}$, \textit{C. glabrata} for DenseNet-121). The last column "
           r"reports criteria R1, R2 and R3 (Y met, N not met).")
    (out / "tables" / "tab_part_a_factorial.tex").write_text(
        tex_table("tab:part_a_factorial", cap, "llccccccccc",
                  [r"\textbf{Model}", r"\textbf{Cond.}", "$T$", "$m$", r"$\alpha$", r"\textbf{Slide BA}", r"\textbf{Default}",
                   r"$\bar{y}$", r"$\bar{g}$", r"\textbf{Clipped (\%)}", r"\textbf{R1R2R3}"], tex), encoding="utf-8")
    return rows


def table_contrasts(A: Dict[str, Any], out: Path) -> List[Dict[str, Any]]:
    fy = g(A, "contrasts", "family_primary_holm_m6", default={}) or {}
    fg = g(A, "contrasts", "family_secondary_direction_holm_m6", default={}) or {}
    rows, tex = [], []
    for k in MODELS:
        for c in ("d_S", "d_I", "d_SxI"):
            t, u = fy.get(f"{k}:{c}"), fg.get(f"{k}:{c}")
            if t is None:
                continue
            r = {"model": MODEL_TXT[k], "contrast": c, "mean_y": t["mean"], "median_y": t["median"],
                 "n_positive": t["n_positive"], "n_negative": t["n_negative"], "p_exact_y": t["p_two_sided"],
                 "p_holm_y": t["p_holm"], "reject_y": t.get("reject_at_0.05"),
                 "mean_g": g(u, "mean"), "p_exact_g": g(u, "p_two_sided"), "p_holm_g": g(u, "p_holm")}
            rows.append(r)
            tex.append([MODEL_TEX[k] if c == "d_S" else "", D_TEX[c], fs(r["mean_y"], 3),
                        f"{r['n_positive']}/{r['n_negative']}", pv(r["p_exact_y"]), pv(r["p_holm_y"]),
                        fs(r["mean_g"], 3), pv(r["p_holm_g"])])
        tex.append(["__RULE__"])
    if tex and tex[-1] == ["__RULE__"]:
        tex.pop()
    write_csv(out / "csv" / "tab_part_a_contrasts.csv", rows)
    cap = (r"Part A slide-level contrasts: $d_S = y_{\mathrm{C\text{-}S}} - y_{\mathrm{C0}}$, "
           r"$d_I = y_{\mathrm{C\text{-}I}} - y_{\mathrm{C0}}$ and "
           r"$d_{S\times I} = (y_{\mathrm{C\text{-}SI}} - y_{\mathrm{C\text{-}S}}) - (y_{\mathrm{C\text{-}I}} - y_{\mathrm{C0}})$. "
           r"$p$ values come from the exact two-sided sign-flip test over all $2^{12} = 4{,}096$ sign patterns "
           r"(smallest attainable value 0.0005). Holm adjustment runs over the six primary tests on $y$; the six "
           r"tests on $g$ form a separate secondary family.")
    (out / "tables" / "tab_part_a_contrasts.tex").write_text(
        tex_table("tab:part_a_contrasts", cap, "llcccccc",
                  [r"\textbf{Model}", r"\textbf{Contrast}", r"Mean $\Delta y$", r"$+/-$ slides", r"$p$",
                   r"$p_{\mathrm{Holm}}$", r"Mean $\Delta g$", r"$p_{\mathrm{Holm}}$ ($g$)"], tex), encoding="utf-8")
    return rows


def table_uncertainty(A: Dict[str, Any], out: Path) -> List[Dict[str, Any]]:
    unc = A.get("uncertainty_propagation", {}) or {}
    rows, tex = [], []
    for k in MODELS:
        for src in ("extreme_low", "extreme_high"):
            u = unc.get(f"{k}:{src}")
            if u is None:
                continue
            r = {"model": MODEL_TXT[k], "extreme": src, "a": u["a"], "b": u["b"],
                 "mean_d_I": u["d_I"]["mean"], "p_d_I": u["d_I"]["p_two_sided"],
                 "mean_d_SxI": u["d_SxI"]["mean"], "p_d_SxI": u["d_SxI"]["p_two_sided"],
                 "C-I_reproduces": g(u, "C-I_reproduction", "reproduces"),
                 "C-SI_reproduces": g(u, "C-SI_reproduction", "reproduces")}
            rows.append(r)
            tex.append([MODEL_TEX[k] if src == "extreme_low" else "", src.replace("extreme_", ""),
                        f"({f4(r['a'])}, {f4(r['b'])})", fs(r["mean_d_I"], 3), pv(r["p_d_I"]),
                        fs(r["mean_d_SxI"], 3), pv(r["p_d_SxI"]), yn(r["C-I_reproduces"]), yn(r["C-SI_reproduces"])])
    write_csv(out / "csv" / "tab_part_a_uncertainty.csv", rows)
    cap = (r"Propagation of calibration uncertainty in Part A. C-I and C-SI are repeated with the joint bootstrap "
           r"replicates of $(a, b)$ at the 2.5th (low) and 97.5th (high) percentiles of $s^{*}$. $p$ values are "
           r"unadjusted and lie outside both Holm families.")
    (out / "tables" / "tab_part_a_uncertainty.tex").write_text(
        tex_table("tab:part_a_uncertainty", cap, "llccccccc",
                  [r"\textbf{Model}", r"\textbf{Extreme}", "$(a, b)$", r"Mean $d_I$", "$p$", r"Mean $d_{S\times I}$",
                   "$p$", r"C-I reproduces", r"C-SI reproduces"], tex), encoding="utf-8")
    return rows


# -----------------------------------------------------------------------------
# Part B tables
# -----------------------------------------------------------------------------

def table_b_metrics(B: Dict[str, Any], out: Path) -> List[Dict[str, Any]]:
    pb = B["part_b"]
    rows, tex = [], []
    for k in MODELS:
        for n in PIPES:
            m = pb["metrics"].get(f"{k}:{n}")
            if m is None:
                continue
            pt, bci, cp = m["point"], m.get("bootstrap_ci") or {}, m["clopper_pearson"]
            interp = g(m, "auroc_interpretation_rule", "interpretable")
            r = {"model": MODEL_TXT[k], "pipeline": n, "correct_slides": m["correct_slides"],
                 "balanced_accuracy": pt["balanced_accuracy"], "ba_ci_lower": g(bci, "balanced_accuracy", "ci_lower"),
                 "ba_ci_upper": g(bci, "balanced_accuracy", "ci_upper"), "mcc": pt["matthews_corrcoef"],
                 "mcc_ci_lower": g(bci, "matthews_corrcoef", "ci_lower"), "mcc_ci_upper": g(bci, "matthews_corrcoef", "ci_upper"),
                 "auroc": pt["auroc"], "auroc_ci_lower": g(bci, "auroc", "ci_lower"), "auroc_ci_upper": g(bci, "auroc", "ci_upper"),
                 "auroc_interpretable": interp, "recall_albicans": pt["recall_albicans"],
                 "recall_albicans_cp_lower": g(cp, "recall_albicans", "ci_lower"),
                 "recall_albicans_cp_upper": g(cp, "recall_albicans", "ci_upper"),
                 "recall_glabrata": pt["recall_glabrata"], "recall_glabrata_cp_lower": g(cp, "recall_glabrata", "ci_lower"),
                 "recall_glabrata_cp_upper": g(cp, "recall_glabrata", "ci_upper"), "brier": pt["brier"], "ece": pt["ece"],
                 "mean_frac_pixels_clipped": g(pb, "clip", n, "mean_frac_pixels_clipped")}
            rows.append(r)

            def cell(v: Any, lo: Any, hi: Any) -> str:
                return r"\begin{tabular}[c]{@{}c@{}}" + f4(v) + r"\\ {" + ci(lo, hi) + r"}\end{tabular}"

            tex.append([MODEL_TEX[k] if n == "E-locked" else "", n, m["correct_slides"],
                        cell(r["balanced_accuracy"], r["ba_ci_lower"], r["ba_ci_upper"]),
                        cell(r["mcc"], r["mcc_ci_lower"], r["mcc_ci_upper"]),
                        cell(r["auroc"], r["auroc_ci_lower"], r["auroc_ci_upper"]) + ("" if interp else r"$^{\dagger}$"),
                        cell(r["recall_albicans"], r["recall_albicans_cp_lower"], r["recall_albicans_cp_upper"]),
                        cell(r["recall_glabrata"], r["recall_glabrata_cp_lower"], r["recall_glabrata_cp_upper"]),
                        f4(r["brier"]), pct(r["mean_frac_pixels_clipped"], 3)])
        tex.append(["__RULE__"])
    if tex and tex[-1] == ["__RULE__"]:
        tex.pop()
    write_csv(out / "csv" / "tab_part_b_metrics.csv", rows)
    cap = (r"Part B slide-level external performance (24 slides, 1,253 images) under the locked-pixel (E-locked), "
           r"instrument-calibrated (E-cal) and per-image photometric (E-z) pipelines. Brackets give class-stratified "
           r"slide bootstrap intervals ($B = 10{,}000$, seed 42) for BA, MCC and AUROC and Clopper-Pearson intervals "
           r"for the recalls. $^{\dagger}$AUROC not interpretable: fewer than half of the 144 cross-class slide pairs "
           r"differ by at least $10^{-3}$ in probability. Clipped is the mean share of pixels clipped to $[0, 1]$.")
    (out / "tables" / "tab_part_b_metrics.tex").write_text(
        tex_table("tab:part_b_metrics", cap, "llcccccccc",
                  [r"\textbf{Model}", r"\textbf{Pipeline}", r"\textbf{Correct}", r"\textbf{BA}", r"\textbf{MCC}",
                   r"\textbf{AUROC}", r"\textbf{Recall} \textit{C. alb.}", r"\textbf{Recall} \textit{C. gla.}",
                   r"\textbf{Brier}", r"\textbf{Clipped (\%)}"], tex), encoding="utf-8")
    return rows


def holm_lookup(pb: Dict[str, Any], key: str) -> Optional[float]:
    for fam in ("pipeline_family_m12", "published_pipeline_family_m4", "new_pipeline_family_m8"):
        v = g(pb, "holm", fam, key, "p_holm")
        if v is not None:
            return v
    return None


def table_b_pipeline_contrasts(B: Dict[str, Any], out: Path) -> List[Dict[str, Any]]:
    pb = B["part_b"]
    scheme = B.get("holm_scheme", "protocol_m12")
    rows, tex = [], []
    for k in MODELS:
        for c in PIPE_CONTRASTS:
            v = pb["pipeline_contrasts"].get(f"{k}:{c}")
            if v is None:
                continue
            dba, mc, sf = v["delta_balanced_accuracy"], v["mcnemar"], v["signflip"]
            r = {"model": MODEL_TXT[k], "contrast": c, "delta_ba": dba["point_delta_b_minus_a"],
                 "delta_ba_ci_lower": dba["ci_lower"], "delta_ba_ci_upper": dba["ci_upper"],
                 "mcnemar_b_prev_correct_new_wrong": mc["table"]["a_correct_b_wrong"],
                 "mcnemar_c_prev_wrong_new_correct": mc["table"]["a_wrong_b_correct"],
                 "p_mcnemar": mc["p_value_two_sided_exact"], "p_holm_mcnemar": holm_lookup(pb, f"{k}:{c}:mcnemar"),
                 "mean_delta_p_true": sf.get("mean_all"), "n_nonzero": sf.get("n_nonzero"),
                 "p_signflip": sf.get("p_two_sided"), "p_holm_signflip": holm_lookup(pb, f"{k}:{c}:signflip"),
                 "holm_scheme": scheme}
            rows.append(r)
            tex.append([MODEL_TEX[k] if c == PIPE_CONTRASTS[0] else "", CONTRAST_TEX[c],
                        f"{fs(r['delta_ba'], 4)} {ci(r['delta_ba_ci_lower'], r['delta_ba_ci_upper'])}",
                        f"{r['mcnemar_b_prev_correct_new_wrong']}/{r['mcnemar_c_prev_wrong_new_correct']}",
                        pv(r["p_mcnemar"]), pv(r["p_holm_mcnemar"]), fs(r["mean_delta_p_true"], 4),
                        pv(r["p_signflip"]), pv(r["p_holm_signflip"])])
        tex.append(["__RULE__"])
    if tex and tex[-1] == ["__RULE__"]:
        tex.pop()
    write_csv(out / "csv" / "tab_part_b_pipeline_contrasts.csv", rows)
    fam = ("one family of 12 tests (3 contrasts, 2 tests, 2 models)" if scheme == "protocol_m12"
           else "the published family of 4 tests for E-z $-$ E-locked and a new family of 8 tests")
    cap = (r"Part B paired pipeline contrasts within each model. $\Delta$BA uses the paired class-stratified slide "
           r"bootstrap with identical indices for both pipelines. $b$ counts slides correct only under the "
           r"subtracted pipeline and $c$ slides correct only under the other pipeline of the pair. "
           r"$\Delta p_{\mathrm{true}}$ is the mean paired change in the slide true-class probability, tested with the "
           r"exact sign-flip test. Holm adjustment covers " + fam + ".")
    (out / "tables" / "tab_part_b_pipeline_contrasts.tex").write_text(
        tex_table("tab:part_b_pipeline_contrasts", cap, "llccccccc",
                  [r"\textbf{Model}", r"\textbf{Contrast}", r"$\Delta$BA [95\%]", "$b/c$", r"$p_{\mathrm{McN}}$",
                   r"$p_{\mathrm{Holm}}$", r"$\Delta p_{\mathrm{true}}$", r"$p_{\mathrm{SF}}$", r"$p_{\mathrm{Holm}}$"],
                  tex), encoding="utf-8")
    return rows


def table_b_model_contrasts(B: Dict[str, Any], out: Path) -> List[Dict[str, Any]]:
    mcs = B["part_b"]["model_contrasts"]
    rows, tex = [], []
    for n in PIPES:
        r0 = g(mcs, "by_pipeline", n)
        if r0 is None:
            continue
        d = r0["deltas"]["balanced_accuracy"]
        r = {"pipeline": n, "delta_ba_densenet_minus_hybrid": d["point_delta_b_minus_a"],
             "ci_lower": d["ci_lower"], "ci_upper": d["ci_upper"],
             "b_hybrid_only_correct": r0["mcnemar"]["table"]["a_correct_b_wrong"],
             "c_densenet_only_correct": r0["mcnemar"]["table"]["a_wrong_b_correct"],
             "p_mcnemar": r0["mcnemar"]["p_value_two_sided_exact"],
             "p_holm_m3": g(mcs, "holm_model_family_m3", n, "p_holm"),
             "p_holm_published_m2_recomputed": g(mcs, "holm_published_m2_recomputed", n),
             "p_holm_published_m2_printed": g(mcs, "holm_published_m2_printed", n),
             "prediction_agreement_rate": r0["prediction_agreement_rate"]}
        rows.append(r)
        tex.append([n, f"{fs(r['delta_ba_densenet_minus_hybrid'], 4)} {ci(r['ci_lower'], r['ci_upper'])}",
                    f"{r['b_hybrid_only_correct']}/{r['c_densenet_only_correct']}", pv(r["p_mcnemar"]),
                    pv(r["p_holm_m3"]), f4(r["prediction_agreement_rate"], 3)])
    write_csv(out / "csv" / "tab_part_b_model_contrasts.csv", rows)
    cap = (r"Paired model contrasts on the external cohort ($\Delta$ = DenseNet-121 minus ViT+CCT+LSA$_{43}$). "
           r"$b$ counts slides correct only for ViT+CCT+LSA$_{43}$ and $c$ slides correct only for DenseNet-121. "
           r"Holm adjustment runs over the three pipelines ($m = 3$).")
    (out / "tables" / "tab_part_b_model_contrasts.tex").write_text(
        tex_table("tab:part_b_model_contrasts", cap, "lccccc",
                  [r"\textbf{Pipeline}", r"$\Delta$BA [95\%]", "$b/c$", r"$p_{\mathrm{McN}}$",
                   r"$p_{\mathrm{Holm}}$", r"\textbf{Agreement}"], tex, resize=False), encoding="utf-8")
    return rows


def table_b_recovery(B: Dict[str, Any], out: Path) -> List[Dict[str, Any]]:
    pb = B["part_b"]
    alb = pb.get("E-alb_B", {})
    rows, tex = [], []
    for k in MODELS:
        rr = g(pb, "recovery_ratio", k, default={}) or {}
        r = {"model": MODEL_TXT[k], "rho": rr.get("point"), "rho_ci_lower": rr.get("ci_lower"),
             "rho_ci_upper": rr.get("ci_upper"), "n_undefined_replicates": rr.get("n_undefined_replicates")}
        for n in PIPES:
            r[f"E-alb_B_p_glabrata_{n}"] = g(alb, f"{k}:{n}", "p_glabrata")
            r[f"E-alb_B_correct_{n}"] = g(alb, f"{k}:{n}", "correct")
        rows.append(r)
        tex.append([MODEL_TEX[k], f4(r["rho"], 3), ci(r["rho_ci_lower"], r["rho_ci_upper"], 3),
                    str(r["n_undefined_replicates"]),
                    *[f4(r[f"E-alb_B_p_glabrata_{n}"]) + ("" if r[f"E-alb_B_correct_{n}"] else r"$^{*}$") for n in PIPES]])
    write_csv(out / "csv" / "tab_part_b_recovery.csv", rows)
    cap = (r"Recovery ratio $\rho = (\mathrm{BA}_{\mathrm{E\text{-}cal}} - \mathrm{BA}_{\mathrm{E\text{-}locked}}) / "
           r"(\mathrm{BA}_{\mathrm{E\text{-}z}} - \mathrm{BA}_{\mathrm{E\text{-}locked}})$ with a paired class-stratified "
           r"slide bootstrap interval (descriptive; replicates with a zero denominator are undefined), and the "
           r"\textit{C. glabrata} probability of \texttt{E-alb\_B}, the only \textit{C. albicans} slide acquired on "
           r"20260507. $^{*}$Misclassified.")
    (out / "tables" / "tab_part_b_recovery.tex").write_text(
        tex_table("tab:part_b_recovery", cap, "lcccccc",
                  [r"\textbf{Model}", r"$\rho$", r"95\% interval", "Undefined",
                   r"\texttt{E-alb\_B} L", "C", "Z"], tex, resize=False), encoding="utf-8")
    return rows


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------

def _style(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK2, labelsize=8, width=0.6)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def fig_dose_response(A: Dict[str, Any], figdir: Path) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surf = A.get("sensitivity_surface", {})
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharey=True, constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, k in zip(axes, MODELS):
        s = surf.get(k)
        _style(ax)
        ax.axhline(0.5, color=INK2, linewidth=0.8, linestyle=(0, (4, 3)))
        ax.text(1.0, 0.5, " chance", color=INK2, fontsize=7, va="bottom", ha="right", transform=ax.get_yaxis_transform())
        if s:
            alphas = s["cols_alpha"]
            ms = list(s["rows_m"])
            if any(a_ >= b_ for a_, b_ in zip(ms, ms[1:])) or len(ms) > len(BLUE_RAMP):
                raise ExportConsistencyError(f"surface rows must be strictly increasing in m: {ms}")
            for i, (m, role, row) in enumerate(zip(ms, s["rows_role"], s["grid"])):
                ys = [c["slide_balanced_accuracy"] if c else float("nan") for c in row]
                ext = role == "external_matched"
                ax.plot(alphas, ys, color=BLUE_RAMP[i], linewidth=2.6 if ext else 2, marker="o", markersize=4.5,
                        markeredgecolor=SURFACE, markeredgewidth=1.0, linestyle="-" if not ext else (0, (5, 2)),
                        label=f"m = {m:.3f}" + (" (ext.)" if ext else ""))
            ax.set_xticks(alphas)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel(r"intensity dose $\alpha$", color=INK, fontsize=9)
        ax.set_title(MODEL_TXT[k].replace("_43", r"$_{43}$"), color=INK, fontsize=10)
    axes[0].set_ylabel("slide balanced accuracy (12 slides)", color=INK, fontsize=9)
    h, l = axes[0].get_legend_handles_labels()
    if h:
        fig.legend(h, l, loc="outside right center", frameon=False, fontsize=8, title="physical scale m = 1200/T",
                   title_fontsize=8, labelcolor=INK)
    files = []
    for ext in ("pdf", "png"):
        p = figdir / f"fig_part_a_dose_response.{ext}"
        fig.savefig(p, dpi=300, facecolor=SURFACE)
        files.append(p.name)
    plt.close(fig)
    return files


def fig_slide_probabilities(B: Dict[str, Any], figdir: Path) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows = B["part_b"].get("slide_table", [])
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), sharey=True, constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    xs = [0, 1, 2]
    for ax, k in zip(axes, MODELS):
        _style(ax)
        ax.axhline(0.5, color=INK2, linewidth=0.8, linestyle=(0, (4, 3)))
        for r in rows:
            ys = [r[f"{k}:{n}:p_glabrata"] for n in PIPES]
            col = ORANGE if "glabrata" in str(r["species"]) else BLUE
            special = bool(r.get("within_session_contrast"))
            ax.plot(xs, ys, color=col, linewidth=2.6 if special else 1.0, alpha=1.0 if special else 0.55,
                    marker="o", markersize=6 if special else 4, markeredgecolor=SURFACE, markeredgewidth=1.0,
                    zorder=3 if special else 2)
            if special:
                ax.annotate(r["display"], (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                            fontsize=7.5, color=INK, va="center")
        ax.set_xticks(xs, ["E-locked", "E-cal", "E-z"])
        ax.set_xlim(-0.3, 2.6)
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(MODEL_TXT[k].replace("_43", r"$_{43}$"), color=INK, fontsize=10)
    axes[0].set_ylabel(r"slide mean $p$(C. glabrata)", color=INK, fontsize=9)
    handles = [Line2D([0], [0], color=BLUE, linewidth=2, marker="o", label="C. albicans slides"),
               Line2D([0], [0], color=ORANGE, linewidth=2, marker="o", label="C. glabrata slides")]
    fig.legend(handles=handles, loc="outside right center", frameon=False, fontsize=8, labelcolor=INK)
    files = []
    for ext in ("pdf", "png"):
        p = figdir / f"fig_part_b_slide_probabilities.{ext}"
        fig.savefig(p, dpi=300, facecolor=SURFACE)
        files.append(p.name)
    plt.close(fig)
    return files


# -----------------------------------------------------------------------------
# Report and bundle
# -----------------------------------------------------------------------------

BUNDLE = (
    "geometry_spec.json", "colab_config.json",
    "unit_checks/unit_check_report.json",
    "part_a/part_a_summary.json", "part_a/part_a_slide_level.csv", "part_a/part_a_plan_frozen.json",
    "part_a/c0_regression_gate.json", "part_a/calibration/calibration.json",
    "part_b/experimental_summary_for_manuscript.json", "part_b/part_b_statistics_full.json",
    "part_b/part_b_plan_frozen.json", "part_b/prep_equivalence_gate.json", "part_b/inference_record.json",
    "part_b/external_slides_extended.csv", "part_b/external_slides_extended.tex",
    "part_b/figs/part_a_sensitivity_surface.pdf", "part_b/figs/part_a_sensitivity_surface.png",
    "state/preflight_report.json", "state/partab_env.lock", "state/stage_timings.json", "state/geometry.done.json",
    "migration/legacy_manifest.json", "migration/part_a_plan.json", "migration/applied.json",
)
REQUIRED = ("geometry_spec.json", "unit_checks/unit_check_report.json", "part_a/part_a_summary.json",
            "part_a/part_a_plan_frozen.json", "part_a/calibration/calibration.json",
            "part_b/part_b_plan_frozen.json", "part_b/experimental_summary_for_manuscript.json")


def write_report(root: Path, out: Path, A: Dict[str, Any], B: Dict[str, Any], U: Dict[str, Any],
                 files: List[str], consistency: Dict[str, Any]) -> None:
    pb = B["part_b"]
    L = ["# Parts A and B: export report", "",
         f"Created {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} from {root}.", "",
         "## Gates", "",
         f"- Unit checks: {U.get('overall_status')}; conditions: {json.dumps(U.get('conditions'))}",
         f"- C0 regression gate (Part A): {json.dumps(A.get('c0_regression_gate'))}",
         f"- Part A complete: {A.get('complete')}; missing conditions: {A.get('missing_conditions')}",
         f"- Published E-locked/E-z metrics reproduced: "
         f"{ {k: v.get('reproduced') for k, v in (pb.get('published_metrics_check') or {}).items()} }",
         f"- Archive gate: { {k: v.get('status') for k, v in (pb.get('archive_gate') or {}).items()} }",
         f"- Holm scheme (Part B pipeline family): {B.get('holm_scheme')}", "",
         "## Geometry", "",
         f"- geometry digest {consistency.get('geometry_digest', '')[:16]}; external-matched T = "
         f"{g(A, 'geometry', 'T_external_matched')}, m = {f4(g(A, 'geometry', 'm_external'), 4)}; "
         f"FOV internal / external {f4(g(A, 'geometry', 'fov_um', 'internal'), 2)} / "
         f"{f4(g(A, 'geometry', 'fov_um', 'external'), 2)} um; metadata {g(A, 'geometry', 'metadata_status')}", "",
         "## Calibration", "",
         f"- a = {f4(g(A, 'calibration', 'point', 'a'))} {ci(*(g(A, 'calibration', 'a_percentile_95') or [None, None]))}, "
         f"b = {f4(g(A, 'calibration', 'point', 'b'))} {ci(*(g(A, 'calibration', 'b_percentile_95') or [None, None]))}", "",
         "## Part A reproduction verdicts", ""]
    for k, v in (A.get("reproduction_verdicts") or {}).items():
        L.append(f"- {v.get('display')}: reproduces {v.get('reproduces')}; calibration robustness "
                 f"{ {n: r.get('stable_across_calibration_interval') for n, r in (v.get('calibration_robustness') or {}).items()} }")
        for line in v.get("reading", []):
            L.append(f"  - {line}")
    L += ["", "## Part B balanced accuracy", ""]
    for k in MODELS:
        L.append(f"- {MODEL_TXT[k]}: " + ", ".join(
            f"{n} {f4(g(pb, 'metrics', f'{k}:{n}', 'point', 'balanced_accuracy'))}" for n in PIPES)
                 + f"; rho = {f4(g(pb, 'recovery_ratio', k, 'point'), 3)}")
    L += ["", "## Files in the bundle", ""] + [f"- {f}" for f in files]
    (out / "EXPORT_REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Manuscript tables, figures and bundle for Parts A and B.")
    p.add_argument("--drive-root", type=Path, required=True)
    p.add_argument("--config", type=Path, default=None, help="colab_config.json (copied into the bundle)")
    p.add_argument("--out", type=Path, default=None, help="Default: <drive-root>/manuscript_v2")
    a = p.parse_args()
    root = a.drive_root
    out = a.out or root / "manuscript_v2"
    missing = [f for f in REQUIRED if not (root / f).is_file()]
    if missing:
        print(f"ERROR: not ready for export; missing {missing}. Finish the earlier stages first.", file=sys.stderr)
        sys.exit(2)
    A = json.loads((root / "part_a/part_a_summary.json").read_text())
    B = json.loads((root / "part_b/experimental_summary_for_manuscript.json").read_text())
    U = json.loads((root / "unit_checks/unit_check_report.json").read_text())
    G = json.loads((root / "geometry_spec.json").read_text())
    C = json.loads((root / "part_a/calibration/calibration.json").read_text())
    PA = json.loads((root / "part_a/part_a_plan_frozen.json").read_text())
    PB = json.loads((root / "part_b/part_b_plan_frozen.json").read_text())
    if not A.get("complete"):
        print(f"ERROR: Part A incomplete: {A.get('missing_conditions')}", file=sys.stderr)
        sys.exit(2)
    try:
        consistency = check_consistency(G, A, C, PA, PB, B)
    except ExportConsistencyError as exc:
        print(f"ERROR: inconsistent inputs, nothing exported: {exc}", file=sys.stderr)
        sys.exit(2)
    for d in ("tables", "csv", "figs"):
        (out / d).mkdir(parents=True, exist_ok=True)
    if a.config and a.config.is_file():
        if a.config.resolve() != (root / "colab_config.json").resolve():
            shutil.copy2(a.config, root / "colab_config.json")

    table_calibration(A, G, out)
    table_factorial(A, out)
    table_contrasts(A, out)
    table_uncertainty(A, out)
    table_b_metrics(B, out)
    table_b_pipeline_contrasts(B, out)
    table_b_model_contrasts(B, out)
    table_b_recovery(B, out)
    fig_dose_response(A, out / "figs")
    fig_slide_probabilities(B, out / "figs")
    for ext in ("pdf", "png"):
        src = root / "part_b" / "figs" / f"part_a_sensitivity_surface.{ext}"
        if src.is_file():
            shutil.copy2(src, out / "figs" / src.name)

    produced = sorted(str(q.relative_to(root)) for d in ("tables", "csv", "figs") for q in (out / d).iterdir() if q.is_file())
    bundle = [f for f in BUNDLE if (root / f).is_file()] + produced
    write_report(root, out, A, B, U, bundle + [str((out / "EXPORT_REPORT.md").relative_to(root))], consistency)
    bundle.append(str((out / "EXPORT_REPORT.md").relative_to(root)))
    manifest = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "drive_root": str(root),
                "consistency": consistency,
                "absent_optional": [f for f in BUNDLE if not (root / f).is_file()],
                "files": [{"path": f, "bytes": (root / f).stat().st_size, "sha256": sha256(root / f)} for f in bundle]}
    (out / "ATTACH_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    zpath = out / "partAB_manuscript_bundle.zip"
    tmp = zpath.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in bundle:
            zf.write(root / f, arcname=f)
        zf.write(out / "ATTACH_MANIFEST.json", arcname=str((out / "ATTACH_MANIFEST.json").relative_to(root)))
    tmp.replace(zpath)
    print(f"tables: {len(list((out / 'tables').glob('*.tex')))}, figures: {len(list((out / 'figs').glob('*.pdf')))}, "
          f"bundle: {zpath} ({zpath.stat().st_size / 1e6:.1f} MB, {len(bundle) + 1} files)")
    if manifest["absent_optional"]:
        print(f"note: not bundled because absent: {manifest['absent_optional']}")


if __name__ == "__main__":
    main()
