#!/usr/bin/env python3
"""
run_external_validation_colab_local.py

Drive-light Colab runner for the external new-microscope validation pipeline.

Scientific role
---------------
This runner orchestrates frozen external validation only. It does not train,
fine-tune, tune thresholds, select a model, or select preprocessing based on
external performance.

Pipeline (9 steps)
------------------
    1. external_manifest.py                      cohort truth table + optics validation
    2. audit_external_tiffs.py                   raw TIFF audit incl. calibration O-4/O-5
    3-5. locked_pixel_pipeline                   preprocess -> evaluate -> statistics (primary)
    6-8. photometric_mean_std_matched_pipeline   preprocess -> evaluate -> statistics (post hoc)
    9. compare_external_pipelines.py             paired slide-level comparison

Colab/Drive policy
------------------
Heavy transient data (extracted TIFFs, .npy arrays, checkpoint copies, caches)
stay on the Colab VM local disk. Only lightweight scientific outputs are
written to the Drive output directory (default ``outputs_external_v2``). Earlier
output roots are never modified or deleted by this runner.

Provenance
----------
``colab_local_run_metadata.json`` records the planned commands, resolved
settings, the environment (Python and package versions, git commit when
available) and SHA-256 hashes of every script, the config and the metadata.
``--dry-run`` prints the same plan as JSON without copying data or running any
step.

Expected project layout
-----------------------
/content/drive/MyDrive/colab/lsa_convtok_vit/validation/
  configs/   data/   models/   scripts/   source/ (training_logic.py and model code)

Author: Rodrigo Sá
Date: 2026
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pipeline_common import load_optional_yaml, resolve_setting, sha256_file

PRIMARY_LABEL = "locked_pixel_pipeline"
PHOTOMETRIC_LABEL = "photometric_mean_std_matched_pipeline"
RUN_METADATA_SCHEMA_VERSION = "2.0"
DEFAULT_OUTPUT_DIR_NAME = "outputs_external_v2"
PIPELINE_SCRIPTS = (
    "external_manifest.py",
    "audit_external_tiffs.py",
    "preprocess_external.py",
    "evaluate_external_generalization.py",
    "external_statistics.py",
    "compare_external_pipelines.py",
    "load_frozen_hybrid.py",
    "optics_provenance.py",
    "pipeline_common.py",
    "validation_stats.py",
    "run_external_validation_colab_local.py",
)
TRACKED_PACKAGES = ("numpy", "torch", "sklearn", "tifffile", "cv2", "yaml", "matplotlib")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run external validation in Colab using local scratch storage for heavy I/O.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--drive-project-dir", required=True, type=Path)
    parser.add_argument("--raw-zip", required=True, type=Path)
    parser.add_argument("--primary-checkpoint", required=True, type=Path)
    parser.add_argument("--provenance-checkpoint", type=Path, default=None)
    parser.add_argument("--metadata-json", required=True, type=Path)
    parser.add_argument("--config-yaml", required=True, type=Path)
    parser.add_argument("--drive-output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR_NAME),
                        help="Persistent output root; relative paths resolve under --drive-project-dir.")
    parser.add_argument("--local-work-dir", type=Path, default=Path("/content/external_validation_work"))
    parser.add_argument("--source-code-dir", type=Path, default=None,
                        help="Directory containing training_logic.py; defaults to <drive-project-dir>/source.")
    parser.add_argument("--comparison-script", type=Path, default=None,
                        help="Optional path to compare_external_pipelines.py (default: next to this runner).")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--preprocess-workers", type=int, default=2)
    parser.add_argument("--loader-workers", type=int, default=2)
    parser.add_argument("--bootstrap", type=int, default=None,
                        help="Bootstrap resamples. Precedence: flag > config evaluation.bootstrap_confidence_intervals.n_bootstrap > 10000.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed. Precedence: flag > config evaluation.bootstrap_confidence_intervals.seed > 42.")
    parser.add_argument("--input-distribution-max-images", type=int, default=256)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--run-photometric-sensitivity", action=argparse.BooleanOptionalAction, default=True,
                        help="Run the predefined post hoc photometric mean/std sensitivity analysis.")
    parser.add_argument("--run-multi-pipeline-comparison", action=argparse.BooleanOptionalAction, default=True,
                        help="Run the paired slide-level comparison after the sensitivity analysis.")
    parser.add_argument("--force-local-reset", action="store_true")
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--skip-audit", action="store_true", help="Diagnostic only. Do not use for final thesis runs.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the planned commands and provenance as JSON; copy nothing and run nothing.")
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def resolve_under(base: Path, p: Optional[Path]) -> Optional[Path]:
    if p is None:
        return None
    return p if p.is_absolute() else base / p


def existing_paths(paths: Sequence[Optional[Path]]) -> List[Path]:
    out: List[Path] = []
    for path in paths:
        if path is None:
            continue
        p = Path(path).resolve()
        if p.exists() and p not in out:
            out.append(p)
    return out


def run(cmd: List[object], *, cwd: Path, env: dict) -> None:
    cmd_str = [str(x) for x in cmd]
    print("\n" + "=" * 100, flush=True)
    print("RUNNING:\n" + " ".join(cmd_str), flush=True)
    print("=" * 100, flush=True)
    subprocess.run(cmd_str, cwd=str(cwd), env=env, check=True)


def copy_to_local(src: Path, dst_dir: Path) -> Path:
    if not src.exists():
        raise FileNotFoundError(f"Required input does not exist: {src}")
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if not dst.exists() or src.stat().st_size != dst.stat().st_size:
        print(f"Copying to local scratch: {src} -> {dst} ({src.stat().st_size / 2**20:.1f} MB)", flush=True)
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        if tmp.exists():
            tmp.unlink()
        shutil.copy2(src, tmp)
        tmp.replace(dst)
    else:
        print(f"Using existing local copy: {dst}", flush=True)
    return dst


def capture_environment(script_dir: Path, project_dir: Path, extra_files: Sequence[Path]) -> Dict[str, Any]:
    """Python/package versions, git commit (if available) and file hashes (F-14)."""
    versions: Dict[str, Optional[str]] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = str(getattr(importlib.import_module(name), "__version__", "unknown"))
        except Exception:
            versions[name] = None
    git_commit: Optional[str] = None
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(project_dir), capture_output=True, text=True, timeout=10, check=True
        ).stdout.strip() or None
    except Exception:
        git_commit = None
    hashes: Dict[str, Optional[str]] = {}
    for name in PIPELINE_SCRIPTS:
        p = script_dir / name
        hashes[name] = sha256_file(p) if p.exists() else None
    for p in extra_files:
        hashes[str(p)] = sha256_file(p) if p is not None and Path(p).exists() else None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
        "git_commit": git_commit,
        "file_sha256": hashes,
    }


def choose_comparison_script(script_dir: Path, user_value: Optional[Path]) -> Path:
    path = (user_value if user_value.is_absolute() else script_dir / user_value) if user_value else script_dir / "compare_external_pipelines.py"
    return path.resolve()


def plan_commands(
    *,
    args: argparse.Namespace,
    script_dir: Path,
    comparison_script: Path,
    inputs: Dict[str, Optional[Path]],
    drive_output: Path,
    local_work: Path,
    n_bootstrap: int,
    seed: int,
) -> Tuple[List[Tuple[str, List[object]]], Dict[str, Any]]:
    """Build the ordered (step_name, command) list for the 9-step pipeline."""
    manifest_csv = drive_output / "manifest" / "external_manifest.csv"
    audit_dir = drive_output / "audit"
    sidecars = drive_output / "preprocessing_sidecars"
    preprocessed_root = local_work / "preprocessed"
    py = sys.executable

    labels = [PRIMARY_LABEL] + ([PHOTOMETRIC_LABEL] if args.run_photometric_sensitivity else [])
    steps: List[Tuple[str, List[object]]] = [(
        "manifest",
        [py, script_dir / "external_manifest.py", "--input", inputs["raw_zip"], "--metadata", inputs["metadata"],
         "--config", inputs["config"], "--output", manifest_csv, "--extract-dir", local_work / "raw_extracted",
         "--force-extract", "--strict"],
    )]
    if not args.skip_audit:
        steps.append((
            "audit",
            [py, script_dir / "audit_external_tiffs.py", "--manifest", manifest_csv, "--metadata", inputs["metadata"],
             "--config", inputs["config"], "--output-dir", audit_dir, "--strict"],
        ))

    eval_dirs: Dict[str, Path] = {}
    for label in labels:
        pre_dir = preprocessed_root / label
        eval_dir = drive_output / "evaluation" / label
        stats_dir = drive_output / "statistics" / label
        eval_dirs[label] = eval_dir
        pre_cmd: List[object] = [
            py, script_dir / "preprocess_external.py", "--manifest", manifest_csv, "--checkpoint", inputs["primary_checkpoint"],
            "--metadata", inputs["metadata"], "--config", inputs["config"], "--image-size", str(args.image_size),
            "--num-workers", str(args.preprocess_workers), "--progress-every", "100", "--strict",
            "--output-dir", pre_dir, "--mode", label, "--sidecar-output-dir", sidecars / label,
        ]
        if args.force_preprocess:
            pre_cmd.append("--force")
        eval_cmd: List[object] = [
            py, script_dir / "evaluate_external_generalization.py", "--checkpoint", inputs["primary_checkpoint"],
            "--manifest", manifest_csv, "--config", inputs["config"], "--metadata", inputs["metadata"],
            "--batch-size", str(args.batch_size), "--num-workers", str(args.loader_workers),
            "--prefetch-factor", str(args.prefetch_factor),
            "--input-distribution-max-images", str(args.input_distribution_max_images), "--seed", str(seed),
            "--strict", "--preprocessed-dir", pre_dir, "--output-dir", eval_dir, "--preprocessing-mode", label,
        ]
        if inputs.get("provenance_checkpoint") is not None:
            eval_cmd += ["--provenance-checkpoint", inputs["provenance_checkpoint"]]
        stats_cmd: List[object] = [
            py, script_dir / "external_statistics.py", "--manifest", manifest_csv, "--config", inputs["config"],
            "--metadata", inputs["metadata"], "--n-bootstrap", str(n_bootstrap), "--seed", str(seed), "--strict",
            "--image-predictions", eval_dir / "image_predictions.csv",
            "--slide-predictions", eval_dir / "slide_predictions.csv", "--output-dir", stats_dir,
        ]
        if not args.skip_audit:
            stats_cmd += ["--tiff-audit-per-image", audit_dir / "tiff_audit_per_image.csv",
                          "--tiff-audit-per-slide", audit_dir / "tiff_audit_per_slide.csv"]
        steps += [(f"preprocess:{label}", pre_cmd), (f"evaluate:{label}", eval_cmd), (f"statistics:{label}", stats_cmd)]

    multi_dir = drive_output / "statistics" / "multi_pipeline_comparison"
    comparison_labels = [label for label in labels if label != PRIMARY_LABEL]
    if args.run_multi_pipeline_comparison and comparison_labels:
        cmp_cmd: List[object] = [
            py, comparison_script, "--primary-slide-predictions", eval_dirs[PRIMARY_LABEL] / "slide_predictions.csv",
            "--primary-label", PRIMARY_LABEL, "--output-dir", multi_dir, "--n-bootstrap", str(n_bootstrap),
            "--seed", str(seed),
        ]
        for label in comparison_labels:
            cmp_cmd += ["--comparison", f"{label}={eval_dirs[label] / 'slide_predictions.csv'}"]
        steps.append(("compare", cmp_cmd))

    layout = {
        "manifest_csv": str(manifest_csv),
        "audit_dir": None if args.skip_audit else str(audit_dir),
        "evaluation_dirs": {k: str(v) for k, v in eval_dirs.items()},
        "multi_pipeline_comparison_dir": str(multi_dir) if any(n == "compare" for n, _ in steps) else None,
        "local_preprocessed_dirs": [str(preprocessed_root / label) for label in labels],
    }
    return steps, layout


def main() -> None:
    args = parse_args()
    t0 = time.time()

    drive_project = args.drive_project_dir.resolve()
    script_dir = Path(__file__).resolve().parent
    source_code_dir = resolve_under(drive_project, args.source_code_dir) if args.source_code_dir else drive_project / "source"
    comparison_script = choose_comparison_script(script_dir, args.comparison_script)
    drive_output = resolve_under(drive_project, args.drive_output_dir)
    local_work = args.local_work_dir

    drive_inputs = {
        "raw_zip": resolve_under(drive_project, args.raw_zip),
        "primary_checkpoint": resolve_under(drive_project, args.primary_checkpoint),
        "provenance_checkpoint": resolve_under(drive_project, args.provenance_checkpoint),
        "metadata": resolve_under(drive_project, args.metadata_json),
        "config": resolve_under(drive_project, args.config_yaml),
    }

    config = load_optional_yaml(drive_inputs["config"]) if drive_inputs["config"] and drive_inputs["config"].exists() else {}
    settings: Dict[str, Any] = {}
    n_bootstrap = int(resolve_setting("n_bootstrap", args.bootstrap, config,
                                      ["evaluation", "bootstrap_confidence_intervals", "n_bootstrap"], 10000, settings))
    seed = int(resolve_setting("seed", args.seed, config,
                               ["evaluation", "bootstrap_confidence_intervals", "seed"], 42, settings))

    environment = capture_environment(
        script_dir, drive_project, [p for p in (drive_inputs["config"], drive_inputs["metadata"]) if p is not None]
    )

    if args.dry_run:
        steps, layout = plan_commands(
            args=args, script_dir=script_dir, comparison_script=comparison_script, inputs=drive_inputs,
            drive_output=drive_output, local_work=local_work, n_bootstrap=n_bootstrap, seed=seed,
        )
        print(json.dumps({
            "schema_version": RUN_METADATA_SCHEMA_VERSION,
            "dry_run": True,
            "steps": [{"name": n, "command": [str(x) for x in c]} for n, c in steps],
            "layout": layout,
            "settings_resolution": settings,
            "environment": environment,
        }, indent=2))
        return

    if args.force_local_reset and local_work.exists():
        shutil.rmtree(local_work)
    (local_work / "tmp").mkdir(parents=True, exist_ok=True)
    local_inputs: Dict[str, Optional[Path]] = {
        "raw_zip": copy_to_local(drive_inputs["raw_zip"], local_work / "inputs"),
        "primary_checkpoint": copy_to_local(drive_inputs["primary_checkpoint"], local_work / "checkpoints"),
        "provenance_checkpoint": (
            copy_to_local(drive_inputs["provenance_checkpoint"], local_work / "checkpoints")
            if drive_inputs["provenance_checkpoint"] else None
        ),
        "metadata": copy_to_local(drive_inputs["metadata"], local_work / "inputs"),
        "config": copy_to_local(drive_inputs["config"], local_work / "inputs"),
    }
    drive_output.mkdir(parents=True, exist_ok=True)

    pythonpath_roots = existing_paths([script_dir, source_code_dir, drive_project / "source", drive_project])
    env = os.environ.copy()
    old_pythonpath = [x for x in env.get("PYTHONPATH", "").split(":") if x]
    env["PYTHONPATH"] = ":".join([str(p) for p in pythonpath_roots] + old_pythonpath)
    env["EXTERNAL_VALIDATION_SOURCE_CODE_DIR"] = str(source_code_dir)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONPYCACHEPREFIX", str(local_work / "pycache"))
    env.setdefault("TMPDIR", str(local_work / "tmp"))
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env.setdefault(var, "1")

    if not (source_code_dir / "training_logic.py").exists() and not any((r / "training_logic.py").exists() for r in pythonpath_roots):
        raise FileNotFoundError(f"Could not find training_logic.py under {source_code_dir}; pass --source-code-dir.")

    steps, layout = plan_commands(
        args=args, script_dir=script_dir, comparison_script=comparison_script, inputs=local_inputs,
        drive_output=drive_output, local_work=local_work, n_bootstrap=n_bootstrap, seed=seed,
    )
    for _, cmd in steps:
        run(cmd, cwd=drive_project, env=env)

    manifest_summary = drive_output / "manifest" / "external_manifest_summary.json"
    optics_summary = None
    if manifest_summary.exists():
        with manifest_summary.open("r", encoding="utf-8") as f:
            optics_summary = json.load(f).get("optics")

    run_metadata: Dict[str, Any] = {
        "schema_version": RUN_METADATA_SCHEMA_VERSION,
        "elapsed_seconds": time.time() - t0,
        "drive_project_dir": str(drive_project),
        "drive_output_dir": str(drive_output),
        "local_work_dir": str(local_work),
        "source_code_dir": str(source_code_dir),
        "comparison_script": str(comparison_script),
        "pythonpath_roots": [str(p) for p in pythonpath_roots],
        "inputs_drive": {k: str(v) if v else None for k, v in drive_inputs.items()},
        "steps": [{"name": n, "command": [str(x) for x in c]} for n, c in steps],
        "layout": layout,
        "settings_resolution": settings,
        "environment": environment,
        "optics": optics_summary,
        "scientific_policy": {
            "external_set_used_for_training": False,
            "external_set_used_for_fine_tuning": False,
            "threshold_tuned_on_external_set": False,
            "primary_unit": "slide",
            "image_level_metrics": "secondary_descriptive_only",
            "primary_pipeline_role": "primary_external_validation",
            "photometric_pipeline_role": "post_hoc_photometric_mean_std_sensitivity_analysis",
            "comparison_role": "paired_slide_level_diagnostic_comparison_of_sensitivity_analyses_vs_primary",
        },
    }
    with (drive_output / "colab_local_run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(run_metadata, f, indent=2, ensure_ascii=False)

    print("\nCompleted external validation with Drive-light local scratch execution.", flush=True)
    print(f"Persistent outputs: {drive_output}", flush=True)
    print(f"Transient local work: {local_work}", flush=True)


if __name__ == "__main__":
    main()
