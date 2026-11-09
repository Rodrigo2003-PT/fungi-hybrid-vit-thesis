#!/usr/bin/env python3
"""
evaluate_external_model.py

Frozen FP32 external inference for DenseNet-121.

Parity with the published Hybrid_43 run is enforced by REUSING the legacy
implementation (byte-identical copy in scripts/legacy/, SHA-256 pinned in
ext_common.PINNED_SHA256, imported by path) for:
    - manifest reading and cohort validation (24 slides, 12 + 12),
    - ExternalNPYDataset: x_model = (x_[0,1] - mu_ckpt) / sigma_ckpt in float32,
    - slide aggregation: p_slide,c = mean_i p_i,c ; y_slide = argmax_c p_slide,c,
    - image/slide metric computation.
Only model loading (load_frozen_model.py), FP32/determinism enforcement,
preprocessing verification and numerical diagnostics are new.

Pipeline scope: locked_pixel_pipeline (primary) and
photometric_mean_std_matched_pipeline (post-hoc radiometric probe).

Usage
-----
    python evaluate_external_model.py \
        --checkpoint /content/densenet_external_work/inputs/final_model_standard_cnn.pth \
        --manifest <validation>/outputs/manifest/external_manifest.csv \
        --preprocessed-dir /content/densenet_external_work/preprocessed/locked_pixel_pipeline \
        --preprocessing-mode locked_pixel_pipeline \
        --output-dir <validation>/outputs/densenet121/locked_pixel_pipeline/evaluation

Author: Rodrigo Sá
Date: 2026
"""

from __future__ import annotations

import ext_common  # noqa: F401  (CUBLAS_WORKSPACE_CONFIG before torch)

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

from ext_common import (
    ANALYSED_PIPELINES,
    EXCLUDED_PIPELINES,
    PIPELINE_LOCKED,
    PIPELINE_PHOTOMETRIC,
    PIPELINE_ROLES,
    assert_fp32_inference_context,
    configure_deterministic_fp32,
    environment_snapshot,
    floats_equal_exact,
    import_module_from_path,
    read_json,
    write_csv,
    write_json,
)
from load_frozen_model import FAMILY, load_frozen_model

HERE = Path(__file__).resolve().parent

EXPECTED_SPATIAL_MODE = {
    PIPELINE_LOCKED: "pixel_locked_resize_to_model_input",
    PIPELINE_PHOTOMETRIC: "pixel_locked_resize_with_post_hoc_photometric_mean_std_matching",
}
assert set(EXPECTED_SPATIAL_MODE) == set(ANALYSED_PIPELINES), "spatial-mode table out of sync with scope"

EXTRA_IMAGE_COLUMNS = ["model_family", "logit_albicans", "logit_glabrata"]
EXTRA_SLIDE_COLUMNS = ["model_family"]


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Frozen external inference for DenseNet-121 (FP32, deterministic).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--preprocessed-dir", required=True, type=Path)
    p.add_argument("--preprocessing-mode", required=True, choices=list(ANALYSED_PIPELINES))
    p.add_argument("--legacy-scripts-dir", type=Path, default=HERE / "legacy",
                   help="Folder with the byte-identical legacy scripts")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--densenet-source", type=Path, default=HERE / "densenet_model.py")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allow-unsafe-checkpoint-load", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args(argv)


def verify_preprocessing(preprocessed_dir: Path, mode: str, n_manifest: int,
                         mean: List[float], std: List[float], strict: bool,
                         ckpt_sha256: str) -> Dict[str, Any]:
    meta_path = preprocessed_dir / "preprocessing_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}; run preprocess_external.py first.")
    meta = read_json(meta_path)
    counts = meta.get("counts", {})
    spatial_mode = (meta.get("spatial_policy") or {}).get("mode")
    ck = meta.get("checkpoint") or {}
    pre_mean = ck.get("checkpoint_mean")
    pre_std = ck.get("checkpoint_std")
    clip = (meta.get("spatial_policy") or {}).get("post_resize_clip_to_unit_interval")
    checks = {
        "n_failed_is_zero": counts.get("n_failed") == 0,
        "n_success_equals_manifest": counts.get("n_success") == n_manifest,
        "n_skipped_existing_is_zero": counts.get("n_skipped_existing") == 0,
        "spatial_mode_matches": spatial_mode == EXPECTED_SPATIAL_MODE[mode],
        "post_resize_clip_true": clip is True,
        "arrays_not_prestandardized": meta.get("save_standardized") is False,
        "preprocessing_checkpoint_sha_equals_model": ck.get("checkpoint_file_sha256") == ckpt_sha256,
        "preprocessing_mu_sigma_equal_model": (
            pre_mean is not None and pre_std is not None
            and floats_equal_exact(pre_mean, mean) and floats_equal_exact(pre_std, std)
        ),
    }
    # Photometric arrays embed the target mu/sigma; they MUST match the model's.
    required = [k for k in checks if k != "preprocessing_mu_sigma_equal_model"]
    if mode == PIPELINE_PHOTOMETRIC:
        required.append("preprocessing_mu_sigma_equal_model")
    failed = [k for k in required if not checks[k]]
    record = {
        "preprocessing_metadata": str(meta_path),
        "spatial_mode": spatial_mode,
        "preprocessing_checkpoint_mean": pre_mean,
        "preprocessing_checkpoint_std": pre_std,
        "preprocessing_checkpoint_sha256": ck.get("checkpoint_file_sha256"),
        "model_checkpoint_sha256": ckpt_sha256,
        "counts": counts,
        "checks": checks,
        "required_checks_failed": failed,
        "note": ("Locked-pixel arrays do not depend on mu/sigma; photometric arrays do "
                 "(target statistics are baked in)."),
    }
    if failed and strict:
        raise RuntimeError(f"Preprocessing verification failed: {failed}\n{record}")
    return record


def run_inference(model: torch.nn.Module, dataset, legacy, *, device: torch.device,
                  batch_size: int, num_workers: int, family: str) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "batch_size": int(batch_size), "shuffle": False, "num_workers": int(num_workers),
        "pin_memory": device.type == "cuda", "collate_fn": legacy.collate_batch,
    }
    loader = DataLoader(dataset, **kwargs)
    rows: List[Dict[str, Any]] = []
    max_abs_input = 0.0
    max_abs_logit = 0.0
    n_nonfinite = 0
    model.eval()
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader, start=1):
            assert_fp32_inference_context()
            x = batch["image"].to(device, non_blocking=True)
            if x.dtype != torch.float32:
                raise RuntimeError(f"Input dtype {x.dtype}; FP32 required.")
            max_abs_input = max(max_abs_input, float(x.abs().max().item()))
            logits = model(x)
            if logits.dtype != torch.float32:
                raise RuntimeError(f"Logit dtype {logits.dtype}; FP32 required.")
            if logits.ndim != 2 or logits.shape[1] != 2:
                raise RuntimeError(f"Expected logits [B,2], got {tuple(logits.shape)}")
            finite = torch.isfinite(logits).all(dim=1)
            n_nonfinite += int((~finite).sum().item())
            if finite.any():
                max_abs_logit = max(max_abs_logit, float(logits[finite].abs().max().item()))
            probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
            lg = logits.float().cpu().numpy()
            preds = probs.argmax(axis=1)
            labels = batch["label"].numpy()
            for i in range(len(labels)):
                pa, pg = float(probs[i, 0]), float(probs[i, 1])
                pred, true = int(preds[i]), int(labels[i])
                rows.append({
                    "relative_path": batch["relative_path"][i],
                    "filename": batch["filename"][i],
                    "slide_id": batch["slide_id"][i],
                    "species_name": batch["species_name"][i],
                    "true_label": true,
                    "pred_label": pred,
                    "pred_species": "Candida albicans" if pred == 0 else "Candida glabrata",
                    "prob_albicans": pa,
                    "prob_glabrata": pg,
                    "confidence": float(max(pa, pg)),
                    "correct": int(pred == true),
                    "hyphae_status": batch["hyphae_status"][i],
                    "model_family": family,
                    "logit_albicans": float(lg[i, 0]),
                    "logit_glabrata": float(lg[i, 1]),
                })
            if batch_idx % 5 == 0:
                print(f"  inference batches: {batch_idx}", flush=True)
    return {"rows": rows, "max_abs_standardized_input": max_abs_input,
            "max_abs_finite_logit": max_abs_logit, "n_nonfinite_logit_rows": n_nonfinite}


def main() -> None:
    args = parse_args()
    t0 = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    determinism = configure_deterministic_fp32(args.seed)

    legacy_dir = args.legacy_scripts_dir.resolve()
    if str(legacy_dir) not in sys.path:
        sys.path.insert(0, str(legacy_dir))
    legacy = import_module_from_path(
        "_legacy_evaluate_external_generalization",
        legacy_dir / "evaluate_external_generalization.py",
    )

    manifest_rows = legacy.read_manifest(args.manifest)
    manifest_warnings = legacy.validate_manifest_cohort(manifest_rows, strict=args.strict)

    bundle = load_frozen_model(args.checkpoint, device=args.device,
                               densenet_source=args.densenet_source, strict=args.strict,
                               allow_unsafe_checkpoint_load=args.allow_unsafe_checkpoint_load)
    model, device = bundle["model"], bundle["device"]
    mean, std = bundle["mean"], bundle["std"]

    prep = verify_preprocessing(args.preprocessed_dir, args.preprocessing_mode,
                                len(manifest_rows), mean, std, args.strict,
                                ckpt_sha256=bundle["metadata"]["checkpoint_sha256"])

    dataset = legacy.ExternalNPYDataset(
        manifest_rows=manifest_rows, preprocessed_dir=args.preprocessed_dir,
        mean=mean, std=std, arrays_already_standardized=False, strict=args.strict,
    )

    inf = run_inference(model, dataset, legacy, device=device, batch_size=args.batch_size,
                        num_workers=args.num_workers, family=FAMILY)
    image_rows = inf["rows"]
    diagnostics = {k: v for k, v in inf.items() if k != "rows"}
    if inf["n_nonfinite_logit_rows"] > 0:
        write_json({"diagnostics": diagnostics}, args.output_dir / "FAILED_nonfinite_logits.json")
        raise RuntimeError(f"{inf['n_nonfinite_logit_rows']} images produced non-finite logits in FP32.")

    legacy.add_preprocessing_mode_to_image_rows(image_rows, args.preprocessing_mode)
    slide_rows = legacy.aggregate_to_slides(image_rows, preprocessing_mode=args.preprocessing_mode)
    for r in slide_rows:
        r["model_family"] = FAMILY

    y_i = [int(r["true_label"]) for r in image_rows]
    yp_i = [int(r["pred_label"]) for r in image_rows]
    p_i = [float(r["prob_glabrata"]) for r in image_rows]
    y_s = [int(r["true_label"]) for r in slide_rows]
    yp_s = [int(r["pred_label"]) for r in slide_rows]
    p_s = [float(r["mean_prob_glabrata"]) for r in slide_rows]
    image_metrics = legacy.compute_binary_metrics(y_i, yp_i, p_i, unit_name="image")
    image_metrics["interpretation"] = "Secondary, descriptive: images within a slide are correlated."
    slide_metrics = legacy.compute_binary_metrics(y_s, yp_s, p_s, unit_name="slide")
    slide_metrics["interpretation"] = "Primary: the slide is the independent unit."
    slide_metrics["slide_aggregation_rule"] = ("mean_probability; argmax, i.e. C. glabrata iff "
                                               "mean p_glabrata > 0.5 (ties -> label 0)")
    n_ties = sum(1 for r in slide_rows if float(r["mean_prob_albicans"]) == float(r["mean_prob_glabrata"]))

    out = args.output_dir
    write_csv(image_rows, out / "image_predictions.csv",
              list(legacy.IMAGE_PREDICTION_COLUMNS) + EXTRA_IMAGE_COLUMNS)
    write_csv(slide_rows, out / "slide_predictions.csv",
              list(legacy.SLIDE_PREDICTION_COLUMNS) + EXTRA_SLIDE_COLUMNS)
    write_json(image_metrics, out / "metrics_image_level.json")
    write_json(slide_metrics, out / "metrics_slide_level.json")
    write_csv(legacy.confusion_matrix_long_rows(y_i, yp_i), out / "confusion_matrix_image.csv",
              legacy.CONFUSION_COLUMNS)
    write_csv(legacy.confusion_matrix_long_rows(y_s, yp_s), out / "confusion_matrix_slide.csv",
              legacy.CONFUSION_COLUMNS)

    metadata = {
        "schema_version": "3.0",
        "script": "evaluate_external_model.py",
        "elapsed_seconds": time.time() - t0,
        "model_family": FAMILY,
        "preprocessing_mode": args.preprocessing_mode,
        "scientific_role": PIPELINE_ROLES[args.preprocessing_mode],
        "excluded_pipelines": EXCLUDED_PIPELINES,
        "precision": "FP32 (no autocast; TF32 disabled)",
        "determinism": determinism,
        "environment": environment_snapshot(include_pip_freeze=False),
        "loader": bundle["metadata"],
        "preprocessing_verification": prep,
        "inference_diagnostics": {
            **diagnostics,
            "batch_size": int(args.batch_size),
            "n_images": len(image_rows),
            "n_slides": len(slide_rows),
            "n_slide_exact_ties": n_ties,
        },
        "manifest": str(args.manifest),
        "manifest_validation_warnings": manifest_warnings,
        "legacy_module": str(legacy_dir / "evaluate_external_generalization.py"),
        "threshold_tuning_performed": False,
        "external_labels_used_for_anything_but_scoring": False,
    }
    write_json(metadata, out / "external_validation_metadata.json")

    print("=" * 78)
    print(f"EVAL_DONE family={FAMILY} pipeline={args.preprocessing_mode}")
    print(f"  slide BA={slide_metrics['balanced_accuracy']:.4f} MCC={slide_metrics['matthews_corrcoef']:.4f} "
          f"AUROC={slide_metrics.get('auroc')}")
    print(f"  image BA={image_metrics['balanced_accuracy']:.4f} AUROC={image_metrics.get('auroc')}")
    print(f"  max|input|={diagnostics['max_abs_standardized_input']:.3f} "
          f"max|logit|={diagnostics['max_abs_finite_logit']:.3f} ties={n_ties}")
    print("=" * 78, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
