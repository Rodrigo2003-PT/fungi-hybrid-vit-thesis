#!/usr/bin/env python3
"""
profile_colab_t4.py

Author-run timing probe for the Colab T4 runtime (about 3 minutes). It times
each kind of work on a handful of real files and on random GPU inputs, then
projects the wall-clock time of every stage from the workload counts implied by
geometry_spec.json (tiles per field of every level, derived from the optics),
and compares the total with the 5-hour session limit.

The raw TIFFs are read from the extracted folders, so stage the inputs and build
the geometry specification first:
    !python /content/drive/MyDrive/partAB/code/run_all_colab.py \\
        --config /content/drive/MyDrive/partAB/colab_config.json --stop-after geometry

Author: Rodrigo Sá
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import geometry_harmonization as geo  # noqa: E402
import photometric_calibration as pc  # noqa: E402
import verify_unit_checks as vuc      # noqa: E402

# Fixed cohort sizes; tile counts per level come from geometry_spec.json
N_DEV_FIELDS, N_HELD, N_EXT = 2910, 821, 1253
ALPHAS = 5
B, N_METRIC_EVALS_CACHED, N_METRIC_EVALS_UNCACHED = 10000, 60_000, 240_000   # bootstrap replicate evaluations


def timeit(fn, n):
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def main() -> None:
    p = argparse.ArgumentParser()
    for n in ("split", "internal-raw-root", "external-manifest", "external-root", "pinned-scripts", "geometry-spec",
              "densenet-scripts", "densenet-ckpt", "hybrid-validation-scripts", "hybrid-source", "hybrid-ckpt",
              "hybrid-config", "out"):
        p.add_argument(f"--{n}", type=Path, required=True)
    p.add_argument("--allow-unsafe-densenet-load", action="store_true")
    p.add_argument("--gpu-batches", type=int, default=8)
    a = p.parse_args()
    R = {}
    split = json.loads(a.split.read_text())
    held = list(split["test_base_paths"])[:6]
    spec = json.loads(a.geometry_spec.read_text())
    level_list = geo.levels_from_spec(spec)
    levels = {lv.T: lv for lv in level_list}
    ext_T = geo.external_level(level_list).T
    TILES = {T: lv.n_tiles for T, lv in levels.items() if lv.role != geo.ROLE_SUPPLEMENTARY}
    EXTREME_LEVELS = (geo.N_INT, ext_T)                 # C-I and C-SI at the two bootstrap extremes
    pinned = pc.load_pinned_preprocess(a.pinned_scripts)
    rows = pc.read_external_manifest(a.external_manifest)[:4]

    # I/O and decode
    t0 = time.perf_counter()
    raws = [pc.read_internal_raw(pc.internal_raw_path(a.internal_raw_root, b)) for b in held]
    R["internal_read_s_per_file"] = (time.perf_counter() - t0) / len(held)
    t0 = time.perf_counter()
    for r in rows:
        pc.read_external_normalized(a.external_root / r["relative_path"], pinned)
    R["external_read_normalize_s_per_file"] = (time.perf_counter() - t0) / len(rows)

    # CPU preprocessing, Part A (alpha = 1 with representative a, b)
    rule = geo.identify_cv2_border_rule(pc.normalize_internal(raws[0])[0])["selected"]
    R["partA_prep_s_per_tile"] = {}
    for T, lv in levels.items():
        dt = timeit(lambda: pc.internal_condition(raws[1], lv, 1.3, 0.1, border_rule=rule), 3)
        R["partA_prep_s_per_tile"][str(T)] = dt / lv.n_tiles
    # Calibration statistics per tile (external-matched level)
    tiles, _ = geo.extract_level_tiles(pc.normalize_internal(raws[2])[0], levels[ext_T], border_rule=rule)
    R["calibration_stats_s_per_tile"] = timeit(lambda: [pc.image_statistics(t, pc.MaskParams()) for t in tiles], 2) / len(tiles)
    R["calibration_tile_extract_s_per_field"] = timeit(
        lambda: geo.extract_level_tiles(pc.normalize_internal(raws[2])[0], levels[ext_T], border_rule=rule), 3)
    # Part B preprocessing: per-pipeline (5 reads) versus one read
    path = a.external_root / rows[0]["relative_path"]
    R["partB_prep_per_pipeline_s_per_image"] = timeit(lambda: [pc.external_condition(path, md, pinned, a=1.3, b=0.1, mu=0.24, sigma=0.0074)
                                                               for md in ("E-locked", "E-z", "E-cal", "E-cal", "E-cal")], 2)
    R["partB_read_once_s_per_image"] = R["external_read_normalize_s_per_file"]

    # GPU throughput on random inputs (FP32, deterministic settings of the real runs)
    import torch
    vuc._configure_torch(42)
    models = vuc.load_models(a)
    R["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    R["img_per_s"] = {}
    for k, m in models.items():
        x = torch.randn(32, 1, 384, 384, device=next(m["model"].parameters()).device)
        with torch.inference_mode():
            for _ in range(2):
                m["model"](x)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(a.gpu_batches):
                m["model"](x)
            torch.cuda.synchronize()
        R["img_per_s"][k] = 32 * a.gpu_batches / (time.perf_counter() - t0)
    R["gpu_peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None

    # Statistics core
    sys.path.insert(0, str(a.densenet_scripts))
    import ext_stats
    rng = np.random.default_rng(0)
    y = np.array([0] * 12 + [1] * 12)
    R["binary_metrics_s"] = timeit(lambda: ext_stats.binary_metrics(y, rng.integers(0, 2, 24), rng.random(24)), 200)

    # Projection
    ips = sum(1.0 / v for v in R["img_per_s"].values())          # seconds per input, both models
    tiles_alpha = sum(N_HELD * t for t in TILES.values())
    tiles_a = ALPHAS * tiles_alpha + 2 * sum(N_HELD * TILES[m] for m in EXTREME_LEVELS)
    prep_a = sum(N_HELD * TILES[int(T)] * s for T, s in R["partA_prep_s_per_tile"].items() if int(T) in TILES) * ALPHAS \
        + 2 * sum(N_HELD * TILES[T] * R["partA_prep_s_per_tile"][str(T)] for T in EXTREME_LEVELS)
    gpu_a = tiles_a * ips
    proj = {
        "calibration_min": (N_DEV_FIELDS * (R["internal_read_s_per_file"] + R["calibration_tile_extract_s_per_field"]
                                            + levels[ext_T].n_tiles * R["calibration_stats_s_per_tile"])
                            + N_EXT * (R["external_read_normalize_s_per_file"] + R["calibration_stats_s_per_tile"])) / 60 / 1.6,
        "partA": {"tiles_per_model": tiles_a, "gpu_min": gpu_a / 60, "cpu_prep_min_single_core": prep_a / 60,
                  "wall_min_overlapped": max(gpu_a, prep_a / 1.6) / 60, "wall_min_serial_upper": (gpu_a + prep_a) / 60},
        "partB_infer": {"gpu_min": 5 * N_EXT * ips / 60,
                        "prep_min_per_pipeline_path": N_EXT * R["partB_prep_per_pipeline_s_per_image"] / 60,
                        "prep_min_single_read_path_estimate": N_EXT * (R["partB_read_once_s_per_image"]
                                                                       + 0.25 * R["partB_prep_per_pipeline_s_per_image"]) / 60},
        "partB_stats": {"min_with_metric_cache": N_METRIC_EVALS_CACHED * R["binary_metrics_s"] / 60,
                        "min_without_cache": N_METRIC_EVALS_UNCACHED * R["binary_metrics_s"] / 60},
    }
    proj["total_hours_optimized"] = (proj["calibration_min"] + 10 + proj["partA"]["wall_min_overlapped"]
                                     + max(proj["partB_infer"]["gpu_min"], proj["partB_infer"]["prep_min_single_read_path_estimate"])
                                     + proj["partB_stats"]["min_with_metric_cache"]) / 60
    R["projection"] = proj
    R["tiles_per_field_by_T"] = TILES
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(R, indent=2))
    print(json.dumps(proj, indent=2))
    print(f"Projected total {proj['total_hours_optimized']:.2f} h of compute against a 5 h session "
          "(plus input staging); every stage resumes across sessions.")


if __name__ == "__main__":
    main()
