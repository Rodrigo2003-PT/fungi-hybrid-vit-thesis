#!/usr/bin/env python3
"""
run_external_densenet.py

Colab (T4) orchestrator for the external validation of the DenseNet-121.

Required layout (all on Drive)
------------------------------
/content/drive/MyDrive/colab/denseNet121/validation/
  Run_DenseNet_External.ipynb
  README.md
  requirements_external.lock                     
  configs/external_validation_config.yaml        
  configs/external_validation_metadata.json      
  data/fungi.zip                                  (external raw TIFF archive)
  models/final_model_standard_cnn.pth                    
  outputs/                                        

Scope: locked_pixel_pipeline and photometric_mean_std_matched_pipeline

Optical geometry: optical_geometry.py is the single source of truth (external
0.08698 um/px, 2x2-binned Axiocam 820; development 0.1746 um/px). S1 verifies
every spacing and derived figure in the configs against it and writes
provenance/optical_geometry.json; the analysed pipelines do not consume spacing.

Author: Rodrigo Sá
Date: 2026
"""

from __future__ import annotations

import ext_common  # noqa: F401  (CUBLAS_WORKSPACE_CONFIG before torch)

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ext_common import (
    ANALYSED_PIPELINES,
    EXCLUDED_PIPELINES,
    LOCK_FILENAME,
    MODEL_DISPLAY,
    MODEL_FAMILY,
    PIPELINE_LOCKED,
    PIPELINE_ROLES,
    REFERENCE_DISPLAY,
    REFERENCE_HYBRID43_TABLE10,
    REFERENCE_LABEL,
    REFERENCE_PINS_FILENAME,
    archived_runs,
    assert_outputs_preflight,
    baseline_evaluation_dir,
    build_return_bundle,
    check_environment,
    environment_snapshot,
    expected_artifacts,
    read_json,
    reproduction_check,
    sha256_array_tree,
    sha256_file,
    verify_forbidden_absent,
    verify_pinned_files,
    verify_pre_run_records,
    write_json,
)
from ext_stats import fmt, fmt_ci_latex, holm_adjust, validate_reference_slide_csv
from optical_geometry import audit_config_geometry, plan_block as optical_geometry_plan_block

HERE = Path(__file__).resolve().parent                       # <validation>/scripts
DEFAULT_VALIDATION = Path("/content/drive/MyDrive/colab/denseNet121/validation")
DEFAULT_TRAINING_OUTPUT_CKPT = Path(
    "/content/drive/MyDrive/colab/denseNet121/output/standard_cnn/final_model_standard_cnn.pth")
FAMILY = MODEL_FAMILY
PIPELINES: Tuple[str, ...] = ANALYSED_PIPELINES

PRIMARY_ENDPOINT = ("slide-level balanced accuracy, locked_pixel_pipeline, class-stratified "
                    "slide bootstrap B=10000 seed=42")
DECISION_RULE = "mean slide probability, argmax (equivalent to 0.5), no tuning"
VOLATILE_PLAN_KEYS = ("created_utc", "plan_version", "supersedes", "scripts_sha256", "args")

# Pre-declared Holm family sizes (analysis plan "multiplicity_policy").
F1_TESTS_PER_PIPELINE = ("mcnemar_exact", "delong_auroc")
F2_LEGACY_PVALUE_KEYS = {
    "mcnemar_exact": "mcnemar_exact_pvalue_descriptive",
    "sign_flip_mean_delta_true_class_probability":
        "sign_flip_exact_pvalue_for_mean_delta_true_class_probability_descriptive",
}
_PLACEHOLDER_NOTE = re.compile(r"(?i)^\s*(<.*>|place\s*holder|todo|tbd|xxx+|n/?a|none|null|-+|\.+)\s*\.?\s*$")
MIN_PRIOR_NOTE_WORDS = 4


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# -----------------------------------------------------------------------------
# Plan resolution (pure; unit-tested)
# -----------------------------------------------------------------------------

def flatten_keys(d: Any, prefix: str = "") -> Dict[str, Any]:
    """Dotted paths of a nested dict; lists and scalars are leaves."""
    out: Dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict) and v:
                out.update(flatten_keys(v, key))
            else:
                out[key] = v
    return out


def plan_pipelines(plan: Dict[str, Any]) -> set:
    if isinstance(plan.get("analysed_pipelines"), dict):
        return set(plan["analysed_pipelines"])
    out = {PIPELINE_LOCKED} if PIPELINE_LOCKED in str(plan.get("primary_endpoint", "")) else set()
    out |= {k for k in (plan.get("post_hoc_probes") or {}) if k in ANALYSED_PIPELINES}
    return out


def find_archived_plans(archive_dir: Path) -> List[Tuple[Path, Dict[str, Any]]]:
    out = []
    for run in archived_runs(archive_dir):
        p = run / "analysis_plan_frozen.json"
        if p.exists():
            out.append((p, read_json(p)))
    return out


def validate_prior_execution_note(note: Optional[str]) -> str:
    """Empty means 'nothing to disclose'. A non-empty note must be a real sentence:
    placeholders ('Placeholder', 'TODO', '<...>') would be frozen into the plan as a disclosure."""
    n = (note or "").strip()
    if n and (_PLACEHOLDER_NOTE.match(n) or len(n.split()) < MIN_PRIOR_NOTE_WORDS):
        raise SystemExit(f"--prior-execution-note {n!r} looks like a placeholder. Pass an empty string if there "
                         f"was no earlier execution, or describe it in at least {MIN_PRIOR_NOTE_WORDS} words.")
    return n


def resolve_plan(out_dir: Path, archive_dir: Path, new_plan: Dict[str, Any],
                 amendment_reason: Optional[str],
                 prior_execution_note: Optional[str] = None) -> Dict[str, Any]:
    """
    ``prior_execution_note`` discloses an earlier execution whose outputs were not
    retained but whose outcomes were observed; it is stored in the plan
    (``prior_execution_disclosure``) so that v1 does not imply blinding.
    """
    out = Path(out_dir)
    plan = dict(new_plan)
    note = validate_prior_execution_note(prior_execution_note)
    plan["prior_execution_disclosure"] = None if not note else {
        "note": note,
        "outcomes_of_earlier_execution_observed": True,
        "earlier_outputs_retained": bool(find_archived_plans(archive_dir)),
    }
    prior = find_archived_plans(archive_dir)
    plan_path = out / "analysis_plan_frozen.json"
    if not prior:
        plan["plan_version"] = 1
        plan["supersedes"] = None
        write_json(plan, plan_path)
        return {"plan_path": str(plan_path), "amendment_path": None, "plan_version": 1,
                "disclosed_prior_execution": bool(note)}
    if not amendment_reason or not amendment_reason.strip():
        raise SystemExit("An archived analysis plan exists: this run is an amendment. "
                         "Pass --amendment-reason \"<why the analysis is re-executed>\".")
    prev_path, prev = prior[-1]
    prev_version = int(prev.get("plan_version") or 1)
    plan["plan_version"] = prev_version + 1
    plan["supersedes"] = {"archived_path": str(prev_path), "sha256": sha256_file(prev_path),
                          "created_utc": prev.get("created_utc"), "plan_version": prev_version}
    p_args, n_args = prev.get("args") or {}, plan.get("args") or {}
    invariants = {
        "primary_endpoint_unchanged": prev.get("primary_endpoint") == plan.get("primary_endpoint"),
        "decision_rule_unchanged": prev.get("decision_rule") == plan.get("decision_rule"),
        "analysed_pipelines_unchanged": plan_pipelines(prev) == plan_pipelines(plan),
        "checkpoint_unchanged": prev.get("checkpoint_sha256") == plan.get("checkpoint_sha256"),
        "bootstrap_B_unchanged": str(p_args.get("bootstrap")) == str(n_args.get("bootstrap")),
        "seed_unchanged": str(p_args.get("seed")) == str(n_args.get("seed")),
    }
    failed = [k for k, ok in invariants.items() if not ok]
    if failed:
        raise SystemExit(f"Amendment refused: pre-specified elements would change: {failed}.")

    def stable(k: str) -> bool:
        return k.split(".")[0] not in VOLATILE_PLAN_KEYS

    fp, fn = flatten_keys(prev), flatten_keys(plan)
    amendment = {
        "amendment_utc": utc_now(),
        "plan_version": plan["plan_version"],
        "reason": amendment_reason.strip(),
        "outcomes_observed_before_amendment": True,
        "statement": ("The frozen DenseNet-121 external analysis is re-executed under an amended plan "
                      "after its outcomes had been observed. Pre-specified elements listed in "
                      "'invariant_checks' are unchanged; removed and added keys are listed verbatim."),
        "prior_plan_path": str(prev_path),
        "prior_plan_sha256": sha256_file(prev_path),
        "prior_plan_created_utc": prev.get("created_utc"),
        "prior_plan_version": prev_version,
        "invariant_checks": invariants,
        "removed_keys": sorted(k for k in fp if k not in fn and stable(k)),
        "added_keys": sorted(k for k in fn if k not in fp and stable(k)),
        "changed_keys": sorted(k for k in fp if k in fn and stable(k) and fp[k] != fn[k]),
    }
    write_json(plan, plan_path)
    amendment_path = out / "analysis_plan_amendment.json"
    write_json(amendment, amendment_path)
    return {"plan_path": str(plan_path), "amendment_path": str(amendment_path),
            "plan_version": plan["plan_version"], "disclosed_prior_execution": bool(note)}


# -----------------------------------------------------------------------------
# Reference gate helpers
# -----------------------------------------------------------------------------

def cohort_slide_labels(metadata_json: Path) -> Dict[str, int]:
    md = read_json(metadata_json)
    lm = md["label_mapping"]
    out: Dict[str, int] = {}
    for species, slides in md["species_slide_sets"].items():
        for s in slides:
            out[s] = int(lm[species])
    return out


def reference_gate(ref_dir: Path, slide_labels: Dict[str, int], *, pin_if_missing: bool,
                   n_bootstrap: int, seed: int) -> Dict[str, Any]:
    """Validate both archived Hybrid_43 CSVs; create the SHA-256 pins once (trust on first use)."""
    pins_path = Path(ref_dir) / REFERENCE_PINS_FILENAME
    pins = read_json(pins_path) if pins_path.exists() else None
    if pins is None and not pin_if_missing:
        raise SystemExit(f"{pins_path} not found. After confirming R2, run once with --pin-reference; "
                         "the pins are written only if both CSVs reproduce Table 10.")
    status: Dict[str, Any] = {}
    for pl in PIPELINES:
        f = Path(ref_dir) / pl / "slide_predictions.csv"
        if not f.exists():
            raise SystemExit(f"Missing archived Hybrid_43 predictions: {f}")
        expected_sha = None if pins is None else pins["pipelines"][pl]["sha256"]
        status[pl] = validate_reference_slide_csv(
            f, pl, slide_labels, expected_sha, REFERENCE_HYBRID43_TABLE10[pl],
            n_bootstrap=n_bootstrap, seed=seed, require_sha256=pins is not None)
    pinned_now = False
    if pins is None:
        pins = {"created_utc": utc_now(),
                "method": ("trust on first use: SHA-256 recorded only after each CSV reproduced the "
                           "main.tex Table 10 point estimates, bootstrap intervals and Clopper-Pearson "
                           "intervals to 4 decimals"),
                "pipelines": {pl: {"path": f"{pl}/slide_predictions.csv", "sha256": status[pl]["sha256"]}
                              for pl in PIPELINES}}
        write_json(pins, pins_path)
        pinned_now = True
    return {"pins_path": str(pins_path), "pins_sha256": sha256_file(pins_path),
            "pinned_in_this_run": pinned_now, "pipelines": status}


# -----------------------------------------------------------------------------
# Results summary, inferential summary, LaTeX (pure on outputs/; unit-tested)
# -----------------------------------------------------------------------------

def _row_tex(label: str, ba: Sequence[Optional[float]], mcc: Sequence[Optional[float]],
             auroc: Sequence[Optional[float]], ra: Sequence[Optional[float]], rg: Sequence[Optional[float]]) -> str:
    cells = [fmt_ci_latex(*v) for v in (ba, mcc, auroc, ra, rg)]
    return f"{label} & " + " & ".join(cells) + r" \\"


def build_results_summary(out_dir: Path, pipelines: Sequence[str], *, comparisons_enabled: bool,
                          run_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build results_summary.json, inferential_summary.json and the LaTeX tables from scratch."""
    out = Path(out_dir)
    conditions: List[Dict[str, Any]] = []
    tex_rows: List[str] = []
    for pl in pipelines:
        s = read_json(out / FAMILY / pl / "supplement" / "supplement_summary.json")
        pm, bc, cp = s["slide_point_metrics"], s["slide_bootstrap_ci"], s["clopper_pearson_recall"]
        conditions.append({
            "model": FAMILY, "pipeline": pl, "role": PIPELINE_ROLES[pl],
            "slide_balanced_accuracy": bc["balanced_accuracy"],
            "slide_mcc": bc["matthews_corrcoef"], "slide_auroc": bc["auroc"],
            "recall_albicans_cp": cp["Candida albicans"], "recall_glabrata_cp": cp["Candida glabrata"],
            "brier": pm["brier"], "ece": pm["ece"],
            "mean_confidence_correct": pm["mean_confidence_correct"],
            "mean_confidence_incorrect": pm["mean_confidence_incorrect"],
            "mcc_denominator_zero": pm["mcc_denominator_zero"],
            "image_level": s["image_level_descriptive"],
            "misclassified_slides": [e["slide_id"] for e in s["misclassified_slides"]],
            "legacy_ci_crosscheck": s.get("legacy_crosscheck"),
        })
        g = lambda d: (d["point_estimate"], d["ci_lower"], d["ci_upper"])  # noqa: E731
        c = lambda d: (d["estimate"], d["ci_lower"], d["ci_upper"])        # noqa: E731
        tex_rows.append(f"% {MODEL_DISPLAY} / {pl}")
        tex_rows.append(_row_tex(f"{MODEL_DISPLAY} ({pl.split('_')[0]})", g(bc["balanced_accuracy"]),
                                 g(bc["matthews_corrcoef"]), g(bc["auroc"]),
                                 c(cp["Candida albicans"]), c(cp["Candida glabrata"])))

    ref_status_path = out / "provenance" / "reference_hybrid43_status.json"
    if comparisons_enabled and ref_status_path.exists():
        ref = read_json(ref_status_path)
        for pl in pipelines:
            o = ref["pipelines"][pl]["observed"]
            pt, bci, cpi = o["point"], o["bootstrap_ci"], o["clopper_pearson"]
            tex_rows.append(f"% {REFERENCE_LABEL} / {pl} (archived; equals main.tex Table 10)")
            tex_rows.append(_row_tex(f"{REFERENCE_DISPLAY} ({pl.split('_')[0]})",
                                     (pt["balanced_accuracy"], *bci["balanced_accuracy"]),
                                     (pt["matthews_corrcoef"], *bci["matthews_corrcoef"]),
                                     (pt["auroc"], *bci["auroc"]),
                                     (pt["recall_albicans"], *cpi["recall_albicans"]),
                                     (pt["recall_glabrata"], *cpi["recall_glabrata"])))

    comparisons: Dict[str, Any] = {}
    f1_raw: Dict[str, Optional[float]] = {}
    if comparisons_enabled:
        for pl in pipelines:
            cmp = read_json(out / "model_comparison" / pl / "model_comparison_summary.json")
            comparisons[pl] = {
                "paired_bootstrap_deltas": cmp["paired_bootstrap_deltas"],
                "mcnemar_exact": cmp["mcnemar_exact"],
                "delong_paired_auroc": {k: cmp["delong_paired_auroc"].get(k) for k in
                                        ("auc_a", "auc_b", "delta_auc_b_minus_a", "var_delta",
                                         "p_value_two_sided")},
                "prediction_agreement_rate": cmp["prediction_agreement_rate"],
            }
            f1_raw[f"mcnemar_exact|{pl}"] = cmp["mcnemar_exact"]["p_value_two_sided_exact"]
            f1_raw[f"delong_auroc|{pl}"] = cmp["delong_paired_auroc"].get("p_value_two_sided")
    f1_adj = holm_adjust(f1_raw)

    f2_raw: Dict[str, Optional[float]] = {}
    f2_missing: List[str] = []
    arms = [p for p in pipelines if p != PIPELINE_LOCKED]
    for arm in arms:
        pj = out / FAMILY / "pipeline_comparison" / f"pairwise_summary__{arm}.json"
        if not pj.exists():
            f2_missing.append(str(pj.relative_to(out)))
            continue
        s2 = read_json(pj)
        for test, legacy_key in F2_LEGACY_PVALUE_KEYS.items():
            if legacy_key not in s2:          # previously a silent None that shrank m
                f2_missing.append(f"{pj.name}:{legacy_key}")
            f2_raw[f"{test}|{arm}_vs_locked"] = s2.get(legacy_key)
    f2_adj = holm_adjust(f2_raw)
    f1_declared = len(F1_TESTS_PER_PIPELINE) * len(pipelines) if comparisons_enabled else 0
    f2_declared = len(F2_LEGACY_PVALUE_KEYS) * len(arms)

    primary = None
    if comparisons_enabled and PIPELINE_LOCKED in comparisons:
        d = comparisons[PIPELINE_LOCKED]["paired_bootstrap_deltas"]["balanced_accuracy"]
        primary = {"contrast": f"balanced accuracy, {MODEL_FAMILY} minus {REFERENCE_LABEL}",
                   "pipeline": PIPELINE_LOCKED, "estimate": d["point_delta_b_minus_a"],
                   "ci95": [d["ci_lower"], d["ci_upper"]], "n_valid_bootstrap": d["n_valid_bootstrap"],
                   "method": "paired class-stratified slide bootstrap, B=10000, seed=42, percentile CI",
                   "p_value": None}
    inferential = {
        "created_utc": utc_now(),
        "primary_contrast": primary,
        "family_F1_model_contrast": {"method": "Holm", "m": sum(v is not None for v in f1_raw.values()),
                                     "m_declared": f1_declared,
                                     "m_matches_declared": sum(v is not None for v in f1_raw.values()) == f1_declared,
                                     "raw": f1_raw, "adjusted": f1_adj},
        "family_F2_pipeline_contrast_within_densenet": {"method": "Holm",
                                                        "m": sum(v is not None for v in f2_raw.values()),
                                                        "m_declared": f2_declared,
                                                        "m_matches_declared": sum(
                                                            v is not None for v in f2_raw.values()) == f2_declared,
                                                        "missing_inputs": f2_missing,
                                                        "raw": f2_raw, "adjusted": f2_adj},
        "descriptive_only": ["all other paired-bootstrap deltas", "fraction_delta_* (not p-values)",
                             "image-level metrics", "acquisition-date breakdown", "calibration (Brier, ECE)"],
    }
    summary = {
        "created_utc": utc_now(),
        "model": FAMILY,
        "analysed_pipelines": {pl: PIPELINE_ROLES[pl] for pl in pipelines},
        "excluded_pipelines": EXCLUDED_PIPELINES,
        "optical_geometry": optical_geometry_plan_block(),
        "conditions": conditions,
        "model_comparisons_vs_archived_hybrid43": comparisons,
        "run": run_info or {},
    }
    write_json(summary, out / "results_summary.json")
    write_json(inferential, out / "inferential_summary.json")

    table = [
        r"% Auto-generated by run_external_densenet.py (slide level, n = 24)",
        r"% BA, MCC, AUROC: class-stratified slide bootstrap 95% CI (B=10000, seed 42); recalls: Clopper--Pearson",
        r"\begin{tabular}{lccccc}", r"\hline",
        r"\textbf{Condition} & \textbf{BA} & \textbf{MCC} & \textbf{AUROC} & \textbf{Rec. alb.} & \textbf{Rec. glab.} \\",
        r"\hline", *tex_rows, r"\hline", r"\end{tabular}",
    ]
    (out / "latex_external_table.tex").write_text("\n".join(table) + "\n", encoding="utf-8")

    if comparisons_enabled:
        lines = [r"% Paired contrasts DenseNet-121 minus archived Hybrid_43 (slide level, n = 24)",
                 r"% Delta: paired class-stratified bootstrap 95% CI (B=10000, seed 42); p: Holm-adjusted over family F1 (m = "
                 + str(inferential["family_F1_model_contrast"]["m"]) + ")",
                 r"\begin{tabular}{llcc}", r"\hline",
                 r"\textbf{Pipeline} & \textbf{Contrast} & \textbf{Estimate} & \textbf{95\% CI / Holm $p$} \\", r"\hline"]
        for pl in pipelines:
            c = comparisons[pl]
            dd = c["paired_bootstrap_deltas"]
            for k, name in (("balanced_accuracy", r"$\Delta$ BA"), ("matthews_corrcoef", r"$\Delta$ MCC"),
                            ("auroc", r"$\Delta$ AUROC"), ("brier", r"$\Delta$ Brier")):
                lines.append(f"{pl.split('_')[0]} & {name} & {fmt(dd[k]['point_delta_b_minus_a'])} & "
                             f"[{fmt(dd[k]['ci_lower'])}, {fmt(dd[k]['ci_upper'])}] \\\\")
            t = c["mcnemar_exact"]["table"]
            lines.append(f"{pl.split('_')[0]} & McNemar ($b$={t['a_correct_b_wrong']}, $c$={t['a_wrong_b_correct']}) "
                         f"& -- & {fmt(f1_adj.get(f'mcnemar_exact|{pl}'))} \\\\")
            lines.append(f"{pl.split('_')[0]} & DeLong $\\Delta$AUROC & "
                         f"{fmt(c['delong_paired_auroc']['delta_auc_b_minus_a'])} & "
                         f"{fmt(f1_adj.get(f'delong_auroc|{pl}'))} \\\\")
        lines += [r"\hline", r"\end{tabular}"]
        (out / "latex_model_comparison.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


# -----------------------------------------------------------------------------
# CLI and runner
# -----------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DenseNet-121 external validation (Colab T4).",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--validation-dir", type=Path, default=DEFAULT_VALIDATION)
    p.add_argument("--training-output-checkpoint", type=Path, default=DEFAULT_TRAINING_OUTPUT_CKPT,
                   help="Original file written by densenet_fungi.ipynb; the copy in models/ must be "
                        "byte-identical. Pass a non-existent path to skip this comparison.")
    p.add_argument("--local-work-dir", type=Path, default=Path("/content/densenet_external_work"))
    p.add_argument("--archive-dir", type=Path, default=None, help="Default: <validation>/archive")
    p.add_argument("--requirements-lock", type=Path, default=None, help=f"Default: <validation>/{LOCK_FILENAME}")
    p.add_argument("--amendment-reason", default=None,
                   help="Required when an archived analysis plan exists (re-execution after outcomes were seen).")
    p.add_argument("--prior-execution-note", default=None,
                   help="Disclose an earlier execution whose outputs were discarded but whose outcomes were "
                        "observed; recorded in the plan as prior_execution_disclosure.")
    p.add_argument("--reference-comparison", action=argparse.BooleanOptionalAction, default=True,
                   help="Paired comparison with archived Hybrid_43 (pre-declared primary contrast).")
    p.add_argument("--pin-reference", action=argparse.BooleanOptionalAction, default=False,
                   help="Create reference_hybrid43/reference_pins.json if absent (only after the Table 10 gate passes).")
    p.add_argument("--allow-unsafe-checkpoint-load", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--loader-workers", type=int, default=2)
    p.add_argument("--preprocess-workers", type=int, default=2)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


class Runner:
    def __init__(self, a: argparse.Namespace) -> None:
        self.a = a
        self.val = a.validation_dir.resolve()
        if HERE != (self.val / "scripts").resolve():
            raise SystemExit(f"This script must live in {self.val / 'scripts'} (found in {HERE}).")
        self.legacy = HERE / "legacy"
        self.config_yaml = self.val / "configs" / "external_validation_config.yaml"
        self.metadata_json = self.val / "configs" / "external_validation_metadata.json"
        self.raw_zip = self.val / "data" / "fungi.zip"
        self.ckpt = self.val / "models" / "final_model_standard_cnn.pth"
        self.ref = self.val / "reference_hybrid43"
        self.out = self.val / "outputs"
        self.prov = self.out / "provenance"
        self.archive = (a.archive_dir or (self.val / "archive")).resolve()
        self.lock = (a.requirements_lock or (self.val / LOCK_FILENAME)).resolve()
        self.work = a.local_work_dir
        self.manifest = self.out / "manifest" / "external_manifest.csv"
        self.log_path = self.out / "run_log.txt"
        self._log_open = False
        self.env = os.environ.copy()
        self.env["PYTHONPATH"] = ":".join([str(HERE), str(self.legacy)] +
                                          [x for x in self.env.get("PYTHONPATH", "").split(":") if x])
        self.env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        self.env["PYTHONDONTWRITEBYTECODE"] = "1"
        self.env.setdefault("PYTHONUNBUFFERED", "1")
        for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            self.env.setdefault(k, "1")   # identical to the published external run

    # ------------------------------------------------------------------ utils
    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        if self._log_open:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def run_cmd(self, cmd: List[Any]) -> None:
        cmd_s = [str(c) for c in cmd]
        self.log("RUN " + " ".join(cmd_s))
        with self.log_path.open("a", encoding="utf-8") as logf:
            proc = subprocess.Popen(cmd_s, cwd=str(HERE), env=self.env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1)
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                logf.write(line)
            rc = proc.wait()
        if rc != 0:
            self.log(f"FAILED rc={rc}")
            raise SystemExit(f"Command failed (rc={rc}): {' '.join(cmd_s)}")

    def require_inputs(self) -> None:
        needed = [self.config_yaml, self.metadata_json, self.raw_zip, self.ckpt, self.lock]
        if self.a.reference_comparison:
            needed += [self.ref / pl / "slide_predictions.csv" for pl in PIPELINES]
        missing = [str(p) for p in needed if not p.exists()]
        if missing:
            raise SystemExit("Missing required inputs:\n  " + "\n  ".join(missing))

    def copy_local(self, src: Path) -> Path:
        dst_dir = self.work / "inputs"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        src_sha = sha256_file(src)
        if not dst.exists() or sha256_file(dst) != src_sha:
            tmp = dst.with_suffix(dst.suffix + ".tmp")
            shutil.copy2(src, tmp)
            tmp.replace(dst)
        if sha256_file(dst) != src_sha:
            raise SystemExit(f"Local copy of {src} does not match the source.")
        return dst

    def pdir(self, pipeline: str) -> Path:
        return self.work / "preprocessed" / pipeline

    def scripts_fingerprint(self) -> Dict[str, str]:
        files = sorted(HERE.glob("*.py")) + sorted(self.legacy.glob("*.py")) + \
            sorted((HERE / "tests").glob("*.py")) + [self.config_yaml, self.metadata_json, self.lock]
        return {str(f.relative_to(self.val)): sha256_file(f) for f in files if f.exists()}

    def archived_env_snapshot(self) -> Optional[Dict[str, Any]]:
        for run in archived_runs(self.archive):
            for name in ("env_snapshot.json",):
                p = run / "provenance" / name
                if p.exists():
                    return {"path": str(p), **read_json(p)}
        return None

    # --------------------------------------------------------------- pipeline
    def build_manifest(self, raw_local: Path) -> None:
        self.run_cmd([sys.executable, self.legacy / "external_manifest.py",
                      "--input", raw_local, "--metadata", self.metadata_json,
                      "--config", self.config_yaml, "--output", self.manifest,
                      "--extract-dir", self.work / "raw_extracted", "--force-extract", "--strict"])

    def preprocess(self, pipeline: str, ckpt_local: Path) -> Path:
        d = self.pdir(pipeline)
        self.run_cmd([sys.executable, self.legacy / "preprocess_external.py",
                      "--manifest", self.manifest, "--checkpoint", ckpt_local,
                      "--config", self.config_yaml, "--output-dir", d, "--mode", pipeline,
                      "--image-size", "384", "--num-workers", str(self.a.preprocess_workers),
                      "--progress-every", "200", "--strict", "--force",
                      "--sidecar-output-dir", self.out / "preprocessing_sidecars" / pipeline])
        return d

    def evaluate(self, ckpt_local: Path, pipeline: str, pre_dir: Path, out_dir: Path) -> None:
        cmd: List[Any] = [sys.executable, HERE / "evaluate_external_model.py",
                          "--checkpoint", ckpt_local, "--manifest", self.manifest,
                          "--preprocessed-dir", pre_dir, "--preprocessing-mode", pipeline,
                          "--legacy-scripts-dir", self.legacy, "--output-dir", out_dir,
                          "--densenet-source", HERE / "densenet_model.py",
                          "--batch-size", str(self.a.batch_size),
                          "--num-workers", str(self.a.loader_workers), "--seed", str(self.a.seed),
                          "--strict"]
        if self.a.allow_unsafe_checkpoint_load:
            cmd.append("--allow-unsafe-checkpoint-load")
        self.run_cmd(cmd)

    def statistics(self, pipeline: str, eval_dir: Path, base: Path) -> None:
        self.run_cmd([sys.executable, self.legacy / "external_statistics.py",
                      "--image-predictions", eval_dir / "image_predictions.csv",
                      "--slide-predictions", eval_dir / "slide_predictions.csv",
                      "--manifest", self.manifest, "--config", self.config_yaml,
                      "--output-dir", base / "statistics", "--n-bootstrap", str(self.a.bootstrap),
                      "--seed", str(self.a.seed), "--strict"])
        self.run_cmd([sys.executable, HERE / "external_statistics_supplement.py",
                      "--image-predictions", eval_dir / "image_predictions.csv",
                      "--slide-predictions", eval_dir / "slide_predictions.csv",
                      "--output-dir", base / "supplement", "--model-label", FAMILY,
                      "--pipeline-label", pipeline, "--n-bootstrap", str(self.a.bootstrap),
                      "--seed", str(self.a.seed), "--require-legacy-crosscheck",
                      "--legacy-ci-json", base / "statistics" / "slide_metrics_with_ci.json"])

    def build_plan(self, ckpt_sha: str, ref: Optional[Dict[str, Any]], envc: Dict[str, Any]) -> Dict[str, Any]:
        a = self.a
        return {
            "created_utc": utc_now(),
            "stage": "frozen (single stage)",
            "model": "DenseNet-121 (frozen, eval-mode BatchNorm, FP32)",
            "analysed_pipelines": {pl: PIPELINE_ROLES[pl] for pl in PIPELINES},
            "excluded_pipelines": EXCLUDED_PIPELINES,
            "optical_geometry": optical_geometry_plan_block(),
            "primary_endpoint": PRIMARY_ENDPOINT,
            "primary_contrast": ("paired class-stratified slide bootstrap 95% CI of delta balanced accuracy "
                                 "(DenseNet-121 minus archived Hybrid_43), locked_pixel_pipeline, B=10000, seed=42"
                                 if a.reference_comparison else None),
            "secondary": ["MCC, weighted F1, AUROC (same bootstrap)", "per-class recall (Clopper-Pearson)",
                          "Brier, ECE (10 bins)", "image-level metrics (slide-cluster bootstrap, descriptive)"],
            "multiplicity_policy": {
                "F1_model_contrast": "exact McNemar and DeLong, both pipelines; Holm over m = 4",
                "F2_pipeline_contrast_within_densenet": "legacy McNemar and sign-flip, photometric vs locked; Holm over m = 2",
                "all_other": "descriptive; no p-values",
            },
            "decision_rule": DECISION_RULE,
            "precision": "FP32, TF32 disabled, cuDNN deterministic",
            "not_evaluated": ["internal held-out DenseNet results (not recomputed)",
                              "Hybrid_43 external results (archived predictions only)", "Hybrid_37"],
            "reference_comparison_enabled": bool(a.reference_comparison),
            "reference_pins_sha256": ref["pins_sha256"] if ref else None,
            "checkpoint_sha256": ckpt_sha,
            "unsafe_checkpoint_load_allowed": bool(a.allow_unsafe_checkpoint_load),
            "requirements_lock_sha256": envc["lock_sha256"],
            "scripts_sha256": self.scripts_fingerprint(),
            "args": {k: str(v) for k, v in vars(a).items()},
        }

    # ------------------------------------------------------------------- run
    def run(self) -> Dict[str, Any]:
        a = self.a
        t0 = time.time()
        validate_prior_execution_note(a.prior_execution_note)   # fail before anything is written

        # S0
        assert_outputs_preflight(self.out)
        forbidden = verify_forbidden_absent(self.val)
        self.out.mkdir(parents=True, exist_ok=True)
        self._log_open = True
        self.log("RUN start: DenseNet-121 frozen external validation")
        write_json(forbidden, self.prov / "forbidden_absent.json")
        write_json(verify_pre_run_records(self.out), self.prov / "pre_run_records.json")

        # S1
        self.require_inputs()
        write_json(verify_pinned_files(self.val), self.prov / "pinned_files.json")
        self.log("S1 pinned legacy scripts, configs and densenet_model.py are byte-identical.")
        geo = audit_config_geometry(self.config_yaml, self.metadata_json)
        write_json(geo, self.prov / "optical_geometry.json")
        self.log(f"S1 optical geometry: {geo['n_fields_checked']} config fields match optical_geometry.py "
                 f"(external {geo['external_spacing_um']} um/px, development {geo['training_spacing_um']} um/px).")

        # S2
        envc = check_environment(self.lock, strict=True)
        snap = environment_snapshot(include_pip_freeze=True)
        old = self.archived_env_snapshot()
        envc["torch_stack_vs_archived_run"] = None if old is None else {
            "archived_snapshot": old["path"],
            **{k: {"archived": old.get(k), "now": snap.get(k), "equal": old.get(k) == snap.get(k)}
               for k in ("torch", "torch_cuda_build", "cudnn", "torchvision", "cv2")},
        }
        write_json(envc, self.prov / "environment_check.json")
        write_json(snap, self.prov / "env_snapshot.json")
        self.log("S2 environment matches the lock file.")

        # S3
        ckpt_sha = sha256_file(self.ckpt)
        training_copy: Dict[str, Any] = {"path": str(a.training_output_checkpoint),
                                         "exists": a.training_output_checkpoint.exists()}
        if a.training_output_checkpoint.exists():
            training_copy["sha256"] = sha256_file(a.training_output_checkpoint)
            training_copy["identical_to_models_copy"] = training_copy["sha256"] == ckpt_sha
            if not training_copy["identical_to_models_copy"]:
                raise SystemExit("models/final_model_standard_cnn.pth differs from the training output "
                                 f"{a.training_output_checkpoint}. Re-copy it.")
        from load_frozen_model import load_frozen_model   # builds the network, no forward pass
        ident = load_frozen_model(self.ckpt, device="cpu",
                                  allow_unsafe_checkpoint_load=a.allow_unsafe_checkpoint_load)["metadata"]
        ident["training_output_copy_check"] = training_copy
        pre = read_json(self.prov / "checkpoint_identity_precheck.json")
        if pre.get("checkpoint_sha256") != ckpt_sha:
            raise SystemExit("Checkpoint changed since the precheck cell; rerun the precheck.")
        write_json(ident, self.prov / "checkpoint_identity.json")
        self.log(f"S3 checkpoint identity OK: sha256={ckpt_sha} load_mode={ident['checkpoint_load_mode']} "
                 f"mu/sigma={ident['mean']}/{ident['std']}")

        # S4
        ref: Optional[Dict[str, Any]] = None
        if a.reference_comparison:
            ref = reference_gate(self.ref, cohort_slide_labels(self.metadata_json),
                                 pin_if_missing=a.pin_reference, n_bootstrap=a.bootstrap, seed=a.seed)
            write_json(ref, self.prov / "reference_hybrid43_status.json")
            self.log(f"S4 archived Hybrid_43 CSVs reproduce Table 10 and match their pins "
                     f"(pinned in this run: {ref['pinned_in_this_run']}).")
        else:
            self.log("S4 reference comparison disabled by --no-reference-comparison (recorded in the plan).")

        # S5
        plan_info = resolve_plan(self.out, self.archive, self.build_plan(ckpt_sha, ref, envc),
                                 a.amendment_reason, a.prior_execution_note)
        amended = plan_info["amendment_path"] is not None
        if amended:
            when = " as an amendment after external outcomes were observed (see analysis_plan_amendment.json)."
        elif plan_info["disclosed_prior_execution"]:
            when = (" before this run's inference; an earlier, discarded execution whose outcomes were "
                    "observed is disclosed in prior_execution_disclosure.")
        else:
            when = " before any external inference."
        self.log(f"S5 analysis plan v{plan_info['plan_version']} written" + when)

        # S6
        ckpt_local = self.copy_local(self.ckpt)
        raw_local = self.copy_local(self.raw_zip)
        self.build_manifest(raw_local)

        # S7
        hashes: Dict[str, Any] = {}
        for pl in PIPELINES:
            hashes[pl] = sha256_array_tree(self.preprocess(pl, ckpt_local))
        write_json(hashes, self.prov / "preprocessing_array_hashes.json")

        # S8
        for pl in PIPELINES:
            base = self.out / FAMILY / pl
            self.evaluate(ckpt_local, pl, self.pdir(pl), base / "evaluation")
            self.statistics(pl, base / "evaluation", base)

        # S9
        comparison_arms = [pl for pl in PIPELINES if pl != PIPELINE_LOCKED]
        self.run_cmd([sys.executable, self.legacy / "compare_external_pipelines.py",
                      "--primary-slide-predictions",
                      self.out / FAMILY / PIPELINE_LOCKED / "evaluation" / "slide_predictions.csv",
                      "--primary-label", PIPELINE_LOCKED,
                      *[x for arm in comparison_arms for x in
                        ("--comparison", f"{arm}={self.out / FAMILY / arm / 'evaluation' / 'slide_predictions.csv'}")],
                      "--output-dir", self.out / FAMILY / "pipeline_comparison",
                      "--n-bootstrap", str(a.bootstrap), "--seed", str(a.seed)])

        # S10
        if ref is not None:
            for pl in PIPELINES:
                self.run_cmd([sys.executable, HERE / "compare_external_models.py",
                              "--model-a", f"{REFERENCE_LABEL}={self.ref / pl / 'slide_predictions.csv'}",
                              "--model-b", f"{FAMILY}={self.out / FAMILY / pl / 'evaluation' / 'slide_predictions.csv'}",
                              "--pipeline-label", pl, "--output-dir", self.out / "model_comparison" / pl,
                              "--n-bootstrap", str(a.bootstrap), "--seed", str(a.seed)])

        # S11
        rep = {pl: reproduction_check(self.out / FAMILY / pl / "evaluation",
                                      baseline_evaluation_dir(self.archive, pl)) for pl in PIPELINES}
        write_json(rep, self.prov / "reproduction_check.json")
        self.log("S11 reproduction check: " + ", ".join(f"{pl}={r['status']}" for pl, r in rep.items()))

        # S12
        build_results_summary(self.out, PIPELINES, comparisons_enabled=ref is not None,
                              run_info={"plan_version": plan_info["plan_version"], "amended": amended,
                                        "elapsed_seconds_before_bundle": time.time() - t0})
        self.log("S12 results_summary.json, inferential_summary.json and LaTeX tables written.")

        # S13 (the run log is closed first so that its bundled hash is final)
        self.log("S13 building the return bundle; the run log is closed now.")
        self._log_open = False
        manifest = build_return_bundle(self.out, expected_artifacts(PIPELINES, with_reference=ref is not None,
                                                                    amended=amended))
        print(f"[{time.strftime('%H:%M:%S')}] DONE: {manifest['n_files']} files, "
              f"{manifest['total_bytes'] / 1e6:.2f} MB, zip sha256={manifest['zip_sha256']}", flush=True)
        return manifest


def main() -> None:
    Runner(parse_args()).run()


if __name__ == "__main__":
    main()
