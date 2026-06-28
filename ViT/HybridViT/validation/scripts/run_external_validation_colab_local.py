#!/usr/bin/env python3
"""
run_external_validation_colab_local.py

Drive-light Colab runner for the external new-microscope validation pipeline.

Scientific role
---------------
This runner orchestrates frozen external validation only. It does not train,
fine-tune, tune thresholds, select a model, or select preprocessing based on
external performance.

Colab/Drive policy
------------------
Heavy transient inputs/arrays stay on the Colab VM local disk:
  - extracted TIFF files
  - preprocessed .npy arrays
  - local checkpoint copies
  - temporary files / Python cache

Only lightweight scientific outputs are written to Google Drive:
  - manifest/audit CSV/JSON/PNG
  - prediction CSVs and metric JSONs
  - bootstrap/statistics outputs
  - preprocessing sidecar metadata/failure CSVs, not .npy arrays
  - consolidated multi-pipeline comparison outputs

Expected project layout
-----------------------
/content/drive/MyDrive/colab/lsa_convtok_vit/validation/
  config/ or configs/
  data/
  models/
  scripts/       external validation scripts
  source/        original Hybrid training/model code, including training_logic.py

Author: Rodrigo Sá
Date: 2026
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


PRIMARY_LABEL = "locked_pixel_pipeline"
SCALE_LABEL = "scale_matched_physical_pipeline"
PHOTOMETRIC_LABEL = "photometric_mean_std_matched_pipeline"


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--drive-output-dir", required=True, type=Path)
    parser.add_argument("--local-work-dir", type=Path, default=Path("/content/external_validation_work"))
    parser.add_argument(
        "--source-code-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing original Hybrid model/training code, especially training_logic.py. "
            "Defaults to <drive-project-dir>/source. Relative paths are resolved under --drive-project-dir."
        ),
    )
    parser.add_argument(
        "--comparison-script",
        type=Path,
        default=None,
        help=(
            "Optional path to compare_external_pipelines.py. "
            "Defaults to the script directory containing this runner."
        ),
    )
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--preprocess-workers", type=int, default=2)
    parser.add_argument("--loader-workers", type=int, default=2)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-distribution-max-images", type=int, default=256)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--run-scale-sensitivity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--run-photometric-sensitivity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run one predefined post hoc photometric mean/std matching sensitivity analysis.",
    )
    parser.add_argument(
        "--run-multi-pipeline-comparison",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run the generalized multi-pipeline paired slide-level comparison after any requested sensitivity "
            "analyses complete."
        ),
    )
    parser.add_argument("--force-local-reset", action="store_true")
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--skip-audit", action="store_true", help="Diagnostic only. Do not use for final thesis runs.")
    return parser.parse_args()


def resolve_under(base: Path, p: Optional[Path]) -> Optional[Path]:
    if p is None:
        return None
    return p if p.is_absolute() else base / p


def existing_paths(paths: Sequence[Optional[Path]]) -> List[Path]:
    out: List[Path] = []
    seen = set()
    for path in paths:
        if path is None:
            continue
        p = Path(path).resolve()
        if p.exists() and str(p) not in seen:
            out.append(p)
            seen.add(str(p))
    return out


def run(cmd: List[object], *, cwd: Path, env: dict) -> None:
    cmd_str = [str(x) for x in cmd]
    print("\n" + "=" * 100, flush=True)
    print("RUNNING:", flush=True)
    print(" ".join(cmd_str), flush=True)
    print("=" * 100, flush=True)
    subprocess.run(cmd_str, cwd=str(cwd), env=env, check=True)


def copy_to_local(src: Path, dst_dir: Path) -> Path:
    if not src.exists():
        raise FileNotFoundError(f"Required input does not exist: {src}")
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if not dst.exists() or src.stat().st_size != dst.stat().st_size:
        size_mb = src.stat().st_size / (1024 * 1024)
        print(f"Copying to local scratch: {src} -> {dst} ({size_mb:.1f} MB)", flush=True)
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        if tmp.exists():
            tmp.unlink()
        shutil.copy2(src, tmp)
        tmp.replace(dst)
    else:
        print(f"Using existing local copy: {dst}", flush=True)
    return dst


def choose_comparison_script(script_dir: Path, user_value: Optional[Path]) -> Path:
    if user_value is not None:
        path = user_value if user_value.is_absolute() else (script_dir / user_value)
    else:
        preferred = script_dir / "compare_external_pipelines.py"
        fallback = script_dir / "compare_external_pipelines.py"
        path = preferred if preferred.exists() else fallback
    if not path.exists():
        raise FileNotFoundError(
            "Could not find a comparison script. Expected compare_external_pipelines.py "
            f"or compare_external_pipelines.py under {script_dir}, or provide --comparison-script explicitly."
        )
    return path.resolve()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    drive_project = args.drive_project_dir.resolve()
    script_dir = Path(__file__).resolve().parent
    source_code_dir = resolve_under(drive_project, args.source_code_dir) if args.source_code_dir else drive_project / "source"
    comparison_script = choose_comparison_script(script_dir, args.comparison_script)
    drive_output = args.drive_output_dir if args.drive_output_dir.is_absolute() else drive_project / args.drive_output_dir
    local_work = args.local_work_dir

    raw_zip_drive = resolve_under(drive_project, args.raw_zip)
    primary_ckpt_drive = resolve_under(drive_project, args.primary_checkpoint)
    provenance_ckpt_drive = resolve_under(drive_project, args.provenance_checkpoint)
    metadata_drive = resolve_under(drive_project, args.metadata_json)
    config_drive = resolve_under(drive_project, args.config_yaml)

    if args.force_local_reset and local_work.exists():
        shutil.rmtree(local_work)

    local_input_dir = local_work / "inputs"
    local_ckpt_dir = local_work / "checkpoints"
    local_extract_dir = local_work / "raw_extracted"
    local_preprocessed_root = local_work / "preprocessed"
    tmp_dir = local_work / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    raw_zip = copy_to_local(raw_zip_drive, local_input_dir)
    primary_ckpt = copy_to_local(primary_ckpt_drive, local_ckpt_dir)
    provenance_ckpt = copy_to_local(provenance_ckpt_drive, local_ckpt_dir) if provenance_ckpt_drive else None
    metadata_json = copy_to_local(metadata_drive, local_input_dir)
    config_yaml = copy_to_local(config_drive, local_input_dir)

    drive_output.mkdir(parents=True, exist_ok=True)
    manifest_csv = drive_output / "manifest" / "external_manifest.csv"
    audit_dir = drive_output / "audit"
    preprocessing_sidecars = drive_output / "preprocessing_sidecars"

    locked_preprocessed = local_preprocessed_root / PRIMARY_LABEL
    scale_preprocessed = local_preprocessed_root / SCALE_LABEL
    photometric_preprocessed = local_preprocessed_root / PHOTOMETRIC_LABEL

    locked_eval_dir = drive_output / "evaluation" / PRIMARY_LABEL
    locked_stats_dir = drive_output / "statistics" / PRIMARY_LABEL
    scale_eval_dir = drive_output / "evaluation" / SCALE_LABEL
    scale_stats_dir = drive_output / "statistics" / SCALE_LABEL
    photometric_eval_dir = drive_output / "evaluation" / PHOTOMETRIC_LABEL
    photometric_stats_dir = drive_output / "statistics" / PHOTOMETRIC_LABEL

    # New consolidated comparison output.
    multi_comparison_dir = drive_output / "statistics" / "multi_pipeline_comparison"

    pythonpath_roots = existing_paths([
        script_dir,
        source_code_dir,
        drive_project / "source",
        drive_project,
        drive_project.parent / "source",
        drive_project.parent,
    ])
    old_pythonpath = [Path(x) for x in os.environ.get("PYTHONPATH", "").split(":") if x]

    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join(str(p) for p in pythonpath_roots + old_pythonpath)
    env["EXTERNAL_VALIDATION_SOURCE_CODE_DIR"] = str(source_code_dir)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONPYCACHEPREFIX", str(local_work / "pycache"))
    env.setdefault("TMPDIR", str(tmp_dir))
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")

    print("External validation runner configuration:", flush=True)
    print(f"  drive_project_dir:  {drive_project}", flush=True)
    print(f"  scripts_dir:        {script_dir}", flush=True)
    print(f"  source_code_dir:    {source_code_dir} (exists={source_code_dir.exists()})", flush=True)
    print(f"  comparison_script:  {comparison_script}", flush=True)
    print(f"  local_work_dir:     {local_work}", flush=True)
    print("  PYTHONPATH roots:", flush=True)
    for root in pythonpath_roots:
        print(f"    - {root}", flush=True)

    candidates = [root / "training_logic.py" for root in pythonpath_roots]
    if not any(p.exists() for p in candidates):
        raise FileNotFoundError(
            "Could not find training_logic.py. Expected it under validation/source. "
            "Pass --source-code-dir if your source folder is elsewhere. Checked: "
            + ", ".join(str(p) for p in candidates)
        )

    run([
        sys.executable, script_dir / "external_manifest.py",
        "--input", raw_zip,
        "--metadata", metadata_json,
        "--config", config_yaml,
        "--output", manifest_csv,
        "--extract-dir", local_extract_dir,
        "--force-extract",
        "--strict",
    ], cwd=drive_project, env=env)

    if not args.skip_audit:
        run([
            sys.executable, script_dir / "audit_external_tiffs.py",
            "--manifest", manifest_csv,
            "--config", config_yaml,
            "--output-dir", audit_dir,
            "--strict",
        ], cwd=drive_project, env=env)

    preprocess_base = [
        sys.executable, script_dir / "preprocess_external.py",
        "--manifest", manifest_csv,
        "--checkpoint", primary_ckpt,
        "--config", config_yaml,
        "--image-size", str(args.image_size),
        "--num-workers", str(args.preprocess_workers),
        "--progress-every", "100",
        "--strict",
    ]
    if args.force_preprocess:
        preprocess_base.append("--force")

    eval_base = [
        sys.executable, script_dir / "evaluate_external_generalization.py",
        "--checkpoint", primary_ckpt,
        "--manifest", manifest_csv,
        "--config", config_yaml,
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.loader_workers),
        "--prefetch-factor", str(args.prefetch_factor),
        "--input-distribution-max-images", str(args.input_distribution_max_images),
        "--strict",
    ]
    if provenance_ckpt is not None:
        eval_base += ["--provenance-checkpoint", provenance_ckpt]

    stats_base = [
        sys.executable, script_dir / "external_statistics.py",
        "--manifest", manifest_csv,
        "--config", config_yaml,
        "--n-bootstrap", str(args.bootstrap),
        "--seed", str(args.seed),
        "--strict",
    ]
    if not args.skip_audit:
        stats_base += [
            "--tiff-audit-per-image", audit_dir / "tiff_audit_per_image.csv",
            "--tiff-audit-per-slide", audit_dir / "tiff_audit_per_slide.csv",
        ]

    # Track available comparison pipelines for unified comparison step.
    completed_comparisons: List[Tuple[str, Path]] = []

    # ------------------------------------------------------------------
    # Primary locked pipeline (always run)
    # ------------------------------------------------------------------
    run(preprocess_base + [
        "--output-dir", locked_preprocessed,
        "--mode", PRIMARY_LABEL,
        "--sidecar-output-dir", preprocessing_sidecars / PRIMARY_LABEL,
    ], cwd=drive_project, env=env)

    run(eval_base + [
        "--preprocessed-dir", locked_preprocessed,
        "--output-dir", locked_eval_dir,
        "--preprocessing-mode", PRIMARY_LABEL,
    ], cwd=drive_project, env=env)

    run(stats_base + [
        "--image-predictions", locked_eval_dir / "image_predictions.csv",
        "--slide-predictions", locked_eval_dir / "slide_predictions.csv",
        "--output-dir", locked_stats_dir,
    ], cwd=drive_project, env=env)

    # ------------------------------------------------------------------
    # Geometric scale sensitivity pipeline
    # ------------------------------------------------------------------
    if args.run_scale_sensitivity:
        run(preprocess_base + [
            "--output-dir", scale_preprocessed,
            "--mode", SCALE_LABEL,
            "--scale-match-padding", "median",
            "--training-pixel-spacing-um", "0.175",
            "--external-pixel-spacing-um", "0.043",
            "--scale-match-target-canvas-height-px", "1200",
            "--scale-match-target-canvas-width-px", "1200",
            "--sidecar-output-dir", preprocessing_sidecars / SCALE_LABEL,
        ], cwd=drive_project, env=env)

        run(eval_base + [
            "--preprocessed-dir", scale_preprocessed,
            "--output-dir", scale_eval_dir,
            "--preprocessing-mode", SCALE_LABEL,
        ], cwd=drive_project, env=env)

        run(stats_base + [
            "--image-predictions", scale_eval_dir / "image_predictions.csv",
            "--slide-predictions", scale_eval_dir / "slide_predictions.csv",
            "--output-dir", scale_stats_dir,
        ], cwd=drive_project, env=env)

        completed_comparisons.append((SCALE_LABEL, scale_eval_dir / "slide_predictions.csv"))

    # ------------------------------------------------------------------
    # Photometric sensitivity pipeline
    # ------------------------------------------------------------------
    if args.run_photometric_sensitivity:
        run(preprocess_base + [
            "--output-dir", photometric_preprocessed,
            "--mode", PHOTOMETRIC_LABEL,
            "--sidecar-output-dir", preprocessing_sidecars / PHOTOMETRIC_LABEL,
        ], cwd=drive_project, env=env)

        run(eval_base + [
            "--preprocessed-dir", photometric_preprocessed,
            "--output-dir", photometric_eval_dir,
            "--preprocessing-mode", PHOTOMETRIC_LABEL,
        ], cwd=drive_project, env=env)

        run(stats_base + [
            "--image-predictions", photometric_eval_dir / "image_predictions.csv",
            "--slide-predictions", photometric_eval_dir / "slide_predictions.csv",
            "--output-dir", photometric_stats_dir,
        ], cwd=drive_project, env=env)

        completed_comparisons.append((PHOTOMETRIC_LABEL, photometric_eval_dir / "slide_predictions.csv"))

    # ------------------------------------------------------------------
    # Unified multi-pipeline comparison
    # ------------------------------------------------------------------
    if args.run_multi_pipeline_comparison and completed_comparisons:
        comparison_cmd: List[object] = [
            sys.executable, comparison_script,
            "--primary-slide-predictions", locked_eval_dir / "slide_predictions.csv",
            "--primary-label", PRIMARY_LABEL,
            "--output-dir", multi_comparison_dir,
            "--n-bootstrap", str(args.bootstrap),
            "--seed", str(args.seed),
        ]
        for label, path in completed_comparisons:
            comparison_cmd += ["--comparison", f"{label}={path}"]

        run(comparison_cmd, cwd=drive_project, env=env)

    run_metadata: Dict[str, object] = {
        "elapsed_seconds": time.time() - t0,
        "drive_project_dir": str(drive_project),
        "drive_output_dir": str(drive_output),
        "local_work_dir": str(local_work),
        "source_code_dir": str(source_code_dir),
        "comparison_script": str(comparison_script),
        "pythonpath_roots": [str(p) for p in pythonpath_roots],
        "raw_zip_drive": str(raw_zip_drive),
        "raw_zip_local": str(raw_zip),
        "primary_checkpoint_local": str(primary_ckpt),
        "provenance_checkpoint_local": str(provenance_ckpt) if provenance_ckpt else None,
        "heavy_transient_outputs_kept_local": [
            str(local_extract_dir),
            str(locked_preprocessed),
            str(scale_preprocessed) if args.run_scale_sensitivity else None,
            str(photometric_preprocessed) if args.run_photometric_sensitivity else None,
        ],
        "persistent_outputs_on_drive": str(drive_output),
        "generated_comparison_pipelines": [
            {"label": label, "slide_predictions": str(path)} for label, path in completed_comparisons
        ],
        "comparison_outputs": {
            "multi_pipeline_comparison_dir": str(multi_comparison_dir) if (args.run_multi_pipeline_comparison and completed_comparisons) else None,
        },
        "scientific_policy": {
            "external_set_used_for_training": False,
            "external_set_used_for_fine_tuning": False,
            "threshold_tuned_on_external_set": False,
            "primary_unit": "slide",
            "image_level_metrics": "secondary_descriptive_only",
            "primary_pipeline_role": "primary_external_validation",
            "scale_matched_pipeline_role": "post_hoc_physical_scale_sensitivity_analysis",
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
