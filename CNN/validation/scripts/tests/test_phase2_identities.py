"""
test_phase2_identities.py

Synthetic-only verification suite for the DenseNet-121 external validation (48 tests).

Section "O" covers the optical geometry (optical_geometry.py) and its
propagation into configs, plan and bundle. No test reads a real image,
checkpoint or notebook; the optical tests use arithmetic, the two configs
(text) and a 4 x 4 synthetic TIFF.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
VALIDATION_DIR = PKG.parent
sys.path.insert(0, str(PKG))

import ext_common  # noqa: E402,F401
import ext_stats as S  # noqa: E402
from ext_common import (  # noqa: E402
    ANALYSED_PIPELINES,
    PIPELINE_LOCKED,
    PIPELINE_PHOTOMETRIC,
    import_module_from_path,
)

try:  # torch is always present in Colab; it may be absent in a static audit sandbox.
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:  # pragma: no cover - environment dependent
    HAS_TORCH = False

LEGACY = Path(os.environ.get("LEGACY_SCRIPTS_DIR", PKG / "legacy"))
MU, SD = 0.23691008985042572, 0.007409718818962574          # Hybrid_43 / development-cohort constants
HAS_LEGACY = (LEGACY / "preprocess_external.py").exists()

needs_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
needs_legacy = pytest.mark.skipif(not HAS_LEGACY, reason="legacy scripts dir not found")


def _legacy(name: str):
    if str(LEGACY) not in sys.path:
        sys.path.insert(0, str(LEGACY))
    return import_module_from_path(f"_t_{name}", LEGACY / f"{name}.py")


def _write_csv(path: Path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _synthetic_slides(seed=0, pipeline=PIPELINE_LOCKED, family=None):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(24):
        y = 0 if i < 12 else 1
        pg = float(np.clip(rng.normal(0.35 + 0.3 * y, 0.25), 0.01, 0.99))
        pred = int(pg > 0.5)
        r = {"slide_id": f"2026050{7 + (i % 2)}_{'alb' if y == 0 else 'gla'}_{chr(65 + i)}",
             "species_name": "Candida albicans" if y == 0 else "Candida glabrata",
             "hyphae_status": "no_hyphae" if y == 0 else "not_applicable",
             "true_label": y, "pred_label": pred, "n_images": 3,
             "mean_prob_albicans": 1 - pg, "mean_prob_glabrata": pg,
             "confidence": max(pg, 1 - pg), "correct": int(pred == y),
             "preprocessing_mode": pipeline}
        if family is not None:
            r["model_family"] = family
        rows.append(r)
    return sorted(rows, key=lambda r: r["slide_id"])


def _synthetic_images(slides, k=3):
    imgs = []
    for r in slides:
        for j in range(k):
            pg = min(max(r["mean_prob_glabrata"] + 0.05 * (j - 1), 0.0), 1.0)
            imgs.append({"relative_path": f"{r['slide_id']}_{j}.tif", "slide_id": r["slide_id"],
                         "true_label": r["true_label"], "pred_label": int(pg > 0.5),
                         "prob_glabrata": pg, "prob_albicans": 1 - pg})
    return imgs


def _run(args, **kw):
    env = dict(os.environ, PYTHONPATH=str(PKG))
    return subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True, env=env, **kw)


# ============================================================================ K: kept (16)
# ----------------------------------------------------------------------------- preprocessing
@needs_legacy
@needs_torch
def test_14bit_locked_scaling_boundaries_and_dtype():
    pe = _legacy("preprocess_external")
    raw = np.array([[0, 4095], [16383, 65535]], dtype=np.uint16)
    x, den = pe.normalize_linear_raw(raw, 14)
    assert den == 16383 and x.dtype == np.float32
    assert x[0, 0] == 0.0 and x[1, 0] == 1.0 and x[1, 1] == 1.0          # 65535 clipped
    assert abs(float(x[0, 1]) - 4095 / 16383) < 6e-8                     # float32 rounding only
    assert pe.infer_effective_bit_depth(16383, 16, 14) == 14
    assert pe.infer_effective_bit_depth(16384, 16, 14) == 16
    assert pe.infer_effective_bit_depth(4095, 16, 14) == 14              # container rule


@needs_legacy
@needs_torch
def test_lanczos_resize_overshoot_is_clipped():
    pe = _legacy("preprocess_external")
    img = np.zeros((24, 24), np.float32)
    img[:, 12:] = 1.0                                                     # hard edge -> ringing
    raw = pe.cv2.resize(img, (37, 37), interpolation=pe.cv2.INTER_LANCZOS4)
    assert raw.max() > 1.0 or raw.min() < 0.0                             # overshoot exists
    out = pe.resize_to_square(img, 37, post_resize_clip=True)
    assert out.dtype == np.float32 and out.min() >= 0.0 and out.max() <= 1.0


@needs_legacy
@needs_torch
def test_photometric_pipeline_gives_per_image_zscore_and_clip_bounds():
    """Network input = (x - m_i)/s_i on the clip-inactive set, for ANY frozen (mu, sigma)."""
    pe = _legacy("preprocess_external")
    rng = np.random.default_rng(1)
    x = rng.uniform(0.40, 0.46, (24, 24)).astype(np.float32)
    for mu, sd in ((MU, SD), (0.25, 0.008)):
        y, _ = pe.photometric_mean_std_match_to_checkpoint(x, [mu], [sd], post_harmonization_clip=True)
        net = (y - np.float32(mu)) / np.float32(sd)                       # ExternalNPYDataset step
        z = (x - x.mean()) / x.std()
        assert y.dtype == np.float32 and np.abs(net - z).max() < 2e-5
    lo, hi = (0 - MU) / SD, (1 - MU) / SD
    assert abs(lo + 31.972885) < 1e-5 and abs(hi - 102.985002) < 1e-5
    spike = np.full((24, 24), 0.4, np.float32)
    spike[0, 0] = 1.0                                                     # z ~ 23.98 < 102.99: inactive
    y2, _ = pe.photometric_mean_std_match_to_checkpoint(spike, [MU], [SD])
    z2 = (spike - spike.mean()) / spike.std()
    assert abs((y2[0, 0] - MU) / SD - z2[0, 0]) < 1e-3


# ----------------------------------------------------------------------------- frozen BatchNorm
def _bn(c=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    bn = nn.BatchNorm2d(c)
    with torch.no_grad():
        bn.weight.copy_(torch.rand(c, generator=g) + 0.5)
        bn.bias.copy_(torch.randn(c, generator=g))
        bn.running_mean.copy_(torch.randn(c, generator=g))
        bn.running_var.copy_(torch.rand(c, generator=g) + 0.5)
    return bn.eval()


@needs_torch
def test_eval_bn_is_affine_and_passes_offset():
    bn = _bn()
    x = torch.randn(2, 3, 8, 8)
    manual = (bn.weight[None, :, None, None] * (x - bn.running_mean[None, :, None, None])
              / torch.sqrt(bn.running_var[None, :, None, None] + bn.eps) + bn.bias[None, :, None, None])
    assert torch.allclose(bn(x), manual, atol=1e-6)
    shift = bn(x + 26.2) - bn(x)                                          # = gamma * 26.2 / sqrt(var+eps)
    expect = bn.weight * 26.2 / torch.sqrt(bn.running_var + bn.eps)
    assert torch.allclose(shift.mean(dim=(0, 2, 3)), expect, atol=1e-4)


# ----------------------------------------------------------------------------- FP32 guard
@needs_torch
def test_fp32_guard_rejects_autocast():
    from ext_common import assert_fp32_inference_context
    assert_fp32_inference_context()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(RuntimeError):
            assert_fp32_inference_context()


# ----------------------------------------------------------------------------- aggregation & stats
@needs_legacy
@needs_torch
def test_slide_mean_pooling_and_tie_rule():
    ev = _legacy("evaluate_external_generalization")
    rows = [
        {"slide_id": "s1", "species_name": "Candida glabrata", "hyphae_status": "not_applicable",
         "true_label": 1, "prob_albicans": 0.7, "prob_glabrata": 0.3},
        {"slide_id": "s1", "species_name": "Candida glabrata", "hyphae_status": "not_applicable",
         "true_label": 1, "prob_albicans": 0.3, "prob_glabrata": 0.7},
        {"slide_id": "s2", "species_name": "Candida albicans", "hyphae_status": "no_hyphae",
         "true_label": 0, "prob_albicans": 0.2, "prob_glabrata": 0.8},
    ]
    out = {r["slide_id"]: r for r in ev.aggregate_to_slides(rows, preprocessing_mode="t")}
    assert out["s1"]["mean_prob_glabrata"] == 0.5 and out["s1"]["pred_label"] == 0   # tie -> albicans
    assert out["s2"]["pred_label"] == 1 and out["s2"]["correct"] == 0


def test_balanced_accuracy_brier_ece_formulas():
    y = np.array([0, 0, 1, 1])
    yp = np.array([0, 1, 1, 1])
    p = np.array([0.1, 0.6, 0.95, 0.85])
    m = S.binary_metrics(y, yp, p)
    assert m["balanced_accuracy"] == 0.5 * (1 / 2 + 2 / 2)
    assert abs(m["brier"] - np.mean((p - y) ** 2)) < 1e-15
    conf = np.maximum(p, 1 - p)
    correct = (y == yp).astype(float)
    # bins: 0.6 -> [0.6,0.7): acc 0; 0.85 -> [0.8,0.9); 0.9 and 0.95 -> [0.9,1.0]
    exp = (1 / 4) * abs(0 - 0.6) + (1 / 4) * abs(1 - 0.85) + (2 / 4) * abs(1 - (0.9 + 0.95) / 2)
    assert abs(S.expected_calibration_error(correct, conf)["ece"] - exp) < 1e-12


@needs_legacy
def test_bootstrap_ci_identical_to_legacy_external_statistics():
    es = _legacy("external_statistics")
    rows = [{k: str(v) for k, v in r.items()} for r in _synthetic_slides()]
    legacy_ci, _ = es.slide_level_bootstrap_ci(rows, n_bootstrap=500, ci=0.95, seed=42, stratified=True)
    y = np.array([int(r["true_label"]) for r in rows])
    yp = np.array([int(r["pred_label"]) for r in rows])
    p = np.array([float(r["mean_prob_glabrata"]) for r in rows])
    c = np.array([float(r["confidence"]) for r in rows])
    mine = S.marginal_bootstrap(y, yp, p, c, n_bootstrap=500, seed=42)
    for k in ("balanced_accuracy", "matthews_corrcoef", "auroc"):
        assert legacy_ci[k]["ci_lower"] == mine[k]["ci_lower"]
        assert legacy_ci[k]["ci_upper"] == mine[k]["ci_upper"]


def test_clopper_pearson_reproduces_manuscript_intervals():
    cases = {(12, 12): (0.7354, 1.0), (1, 12): (0.0021, 0.3848), (0, 12): (0.0, 0.2646),
             (9, 12): (0.4281, 0.9451), (11, 12): (0.6152, 0.9979)}
    for (k, n), (lo, hi) in cases.items():
        r = S.clopper_pearson(k, n)
        assert round(r["ci_lower"], 4) == lo and round(r["ci_upper"], 4) == hi


def test_mcnemar_exact():
    a = [1] * 19 + [0] * 5
    b = [1] * 24
    r = S.mcnemar_exact(a, b)
    assert r["table"]["a_wrong_b_correct"] == 5 and abs(r["p_value_two_sided_exact"] - 0.0625) < 1e-12
    assert S.mcnemar_exact(b, b)["p_value_two_sided_exact"] == 1.0


def test_delong_matches_sklearn_and_bruteforce_variance():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(3)
    y = np.array([0] * 12 + [1] * 12)
    s1 = rng.normal(y * 0.8, 1.0)
    s2 = np.round(rng.normal(y * 1.2, 1.0), 1)                           # includes ties
    r = S.delong_paired(y, s1, s2)
    assert abs(r["auc_a"] - roc_auc_score(y, s1)) < 1e-12
    assert abs(r["auc_b"] - roc_auc_score(y, s2)) < 1e-12
    pos, neg = y == 1, y == 0

    def psi(a, b):
        return 1.0 if a > b else (0.5 if a == b else 0.0)

    def comps(s):
        V10 = np.array([np.mean([psi(xp, xn) for xn in s[neg]]) for xp in s[pos]])
        V01 = np.array([np.mean([psi(xp, xn) for xp in s[pos]]) for xn in s[neg]])
        return V10, V01

    a10, a01 = comps(s1)
    b10, b01 = comps(s2)
    m, n = pos.sum(), neg.sum()
    var = (np.var(a10, ddof=1) + np.var(b10, ddof=1) - 2 * np.cov(a10, b10)[0, 1]) / m \
        + (np.var(a01, ddof=1) + np.var(b01, ddof=1) - 2 * np.cov(a01, b01)[0, 1]) / n
    assert abs(r["var_delta"] - var) < 1e-12


def test_paired_bootstrap_indices_are_shared():
    y = np.array([0] * 12 + [1] * 12)
    g1 = list(S.stratified_bootstrap_indices(y, 20, 42))
    g2 = list(S.stratified_bootstrap_indices(y, 20, 42))
    assert all(np.array_equal(a, b) for a, b in zip(g1, g2))
    assert all((y[i] == 0).sum() == 12 for i in g1)                       # class counts preserved


# ----------------------------------------------------------------------------- loader (no forward)
class _Opaque:                                                            # not allow-listed by weights_only
    pass


def _write_fake_densenet(tmp_path: Path, name="fake_densenet.pth", **overrides) -> Path:
    mod = import_module_from_path("_t_densenet_model", PKG / "densenet_model.py")
    torch.manual_seed(0)
    m = mod.DenseNet121Classifier(2, 1)
    ck = {"model_state_dict": m.state_dict(),
          "config": {"experiment_name": "standard_cnn", "image_size": 384, "channels": 1,
                     "model_config": {"num_classes": 2, "channels": 1}},
          "mean": [MU], "std": [SD],
          "class_to_idx": {"Candida albicans": 0, "Candida glabrata": 1},
          "optimal_epochs": 51, "training_history": {}, "metadata": {}}
    ck.update(overrides)
    p = tmp_path / name
    torch.save(ck, p)
    return p


@needs_torch
def test_loader_densenet_passes_and_fails_closed(tmp_path):
    from load_frozen_model import LoaderError, load_frozen_model
    good = _write_fake_densenet(tmp_path)
    b = load_frozen_model(good, device="cpu")
    meta = b["metadata"]
    assert meta["n_parameters"] == 6_949_634 and meta["n_batchnorm_layers"] == 121
    assert meta["forward_pass_executed"] is False and not b["model"].training
    assert meta["training_provenance"]["expected_steps_51_epochs_x_46_batches"] == 2346
    assert meta["training_provenance"]["consistent_with_final_training_run"] is False  # random init: 0 steps
    for bad in ({"class_to_idx": {"Candida glabrata": 0, "Candida albicans": 1}},
                {"optimal_epochs": 37},
                {"class_to_idx": None}):
        with pytest.raises(LoaderError):
            load_frozen_model(_write_fake_densenet(tmp_path, **bad), device="cpu")
    p = _write_fake_densenet(tmp_path)
    ck = torch.load(p, weights_only=True)
    del ck["mean"]
    torch.save(ck, p)
    with pytest.raises(LoaderError):
        load_frozen_model(p, device="cpu")


def test_pinned_files_are_byte_identical_to_published_versions():
    from ext_common import verify_pinned_files
    rep = verify_pinned_files(VALIDATION_DIR)
    assert all(v["status"] == "OK" for v in rep.values()) and len(rep) == 9


# ----------------------------------------------------------------------------- CLIs on synthetic CSVs
def test_supplement_and_model_comparison_cli_on_synthetic_tables(tmp_path):
    a = _synthetic_slides(0, family="densenet121")
    b = _synthetic_slides(1)
    for r_a, r_b in zip(a, b):
        r_b["slide_id"], r_b["true_label"] = r_a["slide_id"], r_a["true_label"]
        r_b["species_name"], r_b["hyphae_status"] = r_a["species_name"], r_a["hyphae_status"]
        r_b["correct"] = int(r_b["pred_label"] == r_b["true_label"])
    _write_csv(tmp_path / "a.csv", a)
    _write_csv(tmp_path / "b.csv", b)
    _write_csv(tmp_path / "img.csv", _synthetic_images(a))
    r1 = _run([PKG / "external_statistics_supplement.py", "--image-predictions", tmp_path / "img.csv",
               "--slide-predictions", tmp_path / "a.csv", "--output-dir", tmp_path / "supp",
               "--model-label", "densenet121", "--pipeline-label", PIPELINE_LOCKED,
               "--n-bootstrap", "200", "--no-require-legacy-crosscheck"])
    assert r1.returncode == 0, r1.stderr
    r2 = _run([PKG / "compare_external_models.py", "--model-a", f"A={tmp_path / 'b.csv'}",
               "--model-b", f"B={tmp_path / 'a.csv'}", "--pipeline-label", PIPELINE_LOCKED,
               "--output-dir", tmp_path / "cmp", "--n-bootstrap", "200"])
    assert r2.returncode == 0, r2.stderr
    summ = json.loads((tmp_path / "cmp" / "model_comparison_summary.json").read_text())
    assert summ["p_values_unadjusted"] is True and summ["inferential_role"]["delta_balanced_accuracy"].startswith("primary")
    assert "p=" not in (tmp_path / "cmp" / "latex_comparison_rows.tex").read_text()
    assert (tmp_path / "supp" / "latex_row.tex").exists()


@needs_legacy
@needs_torch
def test_preprocessing_metadata_contract_with_densenet_style_checkpoint(tmp_path):
    """Legacy preprocess_external.py exactly as the runner calls it (WITH the pinned --config),
    on two synthetic 24x24 uint16 TIFFs and a DenseNet-style checkpoint; verify_preprocessing()
    must find every key it needs, including the checkpoint SHA-256."""
    import tifffile
    from ext_common import sha256_file
    from evaluate_external_model import verify_preprocessing
    rng = np.random.default_rng(0)
    (tmp_path / "tif").mkdir()
    rows = []
    for sid, lab, sp, hy in (("20260507_gla_A", 1, "Candida glabrata", "not_applicable"),
                             ("20260508_alb_C", 0, "Candida albicans", "no_hyphae")):
        fn = f"{sid}_1.tif"
        tifffile.imwrite(tmp_path / "tif" / fn, rng.integers(5000, 9000, (24, 24), dtype=np.uint16))
        rows.append({"image_path": str(tmp_path / "tif" / fn), "relative_path": f"{sp}/{fn}",
                     "filename": fn, "slide_id": sid, "species_name": sp, "label": lab,
                     "hyphae_status": hy})
    _write_csv(tmp_path / "manifest.csv", rows)
    ckpt = _write_fake_densenet(tmp_path)
    for mode in ANALYSED_PIPELINES:
        out = tmp_path / "pre" / mode
        r = _run([LEGACY / "preprocess_external.py", "--manifest", tmp_path / "manifest.csv",
                  "--checkpoint", ckpt, "--config", VALIDATION_DIR / "configs" / "external_validation_config.yaml",
                  "--output-dir", out, "--mode", mode, "--image-size", "384",
                  "--num-workers", "1", "--force", "--strict"])
        assert r.returncode == 0, (r.stdout + r.stderr)[-3000:]
        rec = verify_preprocessing(out, mode, 2, [MU], [SD], strict=True, ckpt_sha256=sha256_file(ckpt))
        assert rec["required_checks_failed"] == []
        arr = np.load(next(out.rglob("*.npy")))
        assert arr.dtype == np.float32 and arr.shape == (384, 384) and arr.min() >= 0 and arr.max() <= 1


# ============================================================================ N: new (26)
def test_mcc_degenerate_single_predicted_class():
    """All-albicans predictions: MCC denominator 0 -> sklearn returns 0.0 (main.tex Table 10 note a)."""
    y = np.array([0] * 12 + [1] * 12)
    yp = np.zeros(24, int)
    p = np.linspace(0.01, 0.45, 24)
    m = S.binary_metrics(y, yp, p)
    assert m["matthews_corrcoef"] == 0.0 and m["mcc_denominator_zero"] is True
    assert m["balanced_accuracy"] == 0.5 and m["recall_glabrata"] == 0.0 and m["recall_albicans"] == 1.0
    if HAS_LEGACY:
        es = _legacy("external_statistics")
        assert es.compute_binary_metrics(y, yp, p, unit="slide")["matthews_corrcoef"] == 0.0
    boot = S.marginal_bootstrap(y, yp, p, np.maximum(p, 1 - p), n_bootstrap=300, seed=42)
    assert (boot["matthews_corrcoef"]["ci_lower"], boot["matthews_corrcoef"]["ci_upper"]) == (0.0, 0.0)
    assert (boot["balanced_accuracy"]["ci_lower"], boot["balanced_accuracy"]["ci_upper"]) == (0.5, 0.5)


def test_single_true_class_metrics():
    y = np.zeros(4, int)
    yp = np.array([0, 1, 0, 0])
    p = np.array([0.1, 0.7, 0.2, 0.3])
    m = S.binary_metrics(y, yp, p)
    assert m["auroc"] is None and m["mcc_denominator_zero"] is True
    assert abs(m["brier"] - np.mean((p - y) ** 2)) < 1e-15


def test_clopper_pearson_boundaries_closed_form():
    n, a = 12, 0.05
    r0, rn = S.clopper_pearson(0, n), S.clopper_pearson(n, n)
    assert r0["ci_lower"] == 0.0 and abs(r0["ci_upper"] - (1 - (a / 2) ** (1 / n))) < 1e-12
    assert rn["ci_upper"] == 1.0 and abs(rn["ci_lower"] - (a / 2) ** (1 / n)) < 1e-12
    assert round(r0["ci_upper"], 4) == 0.2646 and round(rn["ci_lower"], 4) == 0.7354
    los, his = [], []
    for k in range(n + 1):
        r = S.clopper_pearson(k, n)
        assert 0.0 <= r["ci_lower"] <= k / n <= r["ci_upper"] <= 1.0
        los.append(r["ci_lower"])
        his.append(r["ci_upper"])
    assert all(np.diff(los) > 0) and all(np.diff(his) > 0)
    assert S.clopper_pearson(0, 0)["estimate"] is None


def test_stratified_bootstrap_determinism_seed_and_order():
    import external_statistics_supplement as SUP
    rows = [{k: str(v) for k, v in r.items()} for r in _synthetic_slides(5)]
    _, _, y, yp, p, c = SUP.slide_arrays(rows)
    b1 = S.marginal_bootstrap(y, yp, p, c, n_bootstrap=300, seed=42)
    b2 = S.marginal_bootstrap(y, yp, p, c, n_bootstrap=300, seed=42)
    assert b1 == b2
    i42 = next(S.stratified_bootstrap_indices(y, 1, 42))
    i43 = next(S.stratified_bootstrap_indices(y, 1, 43))
    assert not np.array_equal(i42, i43)
    shuffled = list(reversed(rows))
    _, ids_s, ys, yps, ps, cs = SUP.slide_arrays(shuffled)
    assert ids_s == sorted(ids_s) and np.array_equal(ys, y) and np.array_equal(ps, p)
    assert S.marginal_bootstrap(ys, yps, ps, cs, n_bootstrap=300, seed=42) == b1


def test_replicate_omission_probability():
    """P(a stratified replicate omits one given slide of a 12-slide class) = (11/12)^12 = 0.3520."""
    y = np.array([0] * 12 + [1] * 12)
    target, B = 12, 20000                                                 # one designated C. glabrata slide
    omitted = sum(target not in idx for idx in S.stratified_bootstrap_indices(y, B, 42))
    p0 = (11 / 12) ** 12
    assert abs(omitted / B - p0) < 4 * np.sqrt(p0 * (1 - p0) / B)


def test_paired_contrast_identity_and_antisymmetry():
    import compare_external_models as CMP
    ra, rb = _synthetic_slides(2), _synthetic_slides(3)
    y = np.array([r["true_label"] for r in ra])

    def arr(rows):
        pred = np.array([int(r["mean_prob_glabrata"] > 0.5) for r in rows])
        p = np.array([r["mean_prob_glabrata"] for r in rows])
        return {"pred": pred, "p": p, "c": np.maximum(p, 1 - p), "correct": (pred == y).astype(int)}

    A, Bm = arr(ra), arr(rb)
    same = CMP.paired_contrast(y, A, A, n_bootstrap=200, seed=42)
    for k, d in same["deltas"].items():
        assert d["point_delta_b_minus_a"] == 0.0 and (d["ci_lower"], d["ci_upper"]) == (0.0, 0.0), k
    assert same["mcnemar"]["p_value_two_sided_exact"] == 1.0 and same["delong"]["p_value_two_sided"] == 1.0
    ab = CMP.paired_contrast(y, A, Bm, n_bootstrap=200, seed=42)["replicate_deltas"]
    ba = CMP.paired_contrast(y, Bm, A, n_bootstrap=200, seed=42)["replicate_deltas"]
    for k in ab:
        u, v = np.asarray(ab[k]), np.asarray(ba[k])
        fin = np.isfinite(u)
        assert np.array_equal(fin, np.isfinite(v)) and np.array_equal(u[fin], -v[fin]), k


def test_mcnemar_edge_cases():
    def p(b, c):
        a_ = [1] * b + [0] * c + [1] * 3
        b_ = [0] * b + [1] * c + [1] * 3
        return S.mcnemar_exact(a_, b_)["p_value_two_sided_exact"]
    assert S.mcnemar_exact([1, 0], [1, 0])["p_value_two_sided_exact"] == 1.0   # no discordant pairs
    assert p(1, 0) == 1.0
    assert abs(p(5, 0) - 0.0625) < 1e-15 and abs(p(6, 0) - 0.03125) < 1e-15
    for b, c in ((5, 2), (7, 1), (0, 4)):
        assert p(b, c) == p(c, b)


def test_delong_degenerate_and_perfect_separation():
    y = np.array([0] * 12 + [1] * 12)
    s = np.linspace(0, 1, 24)
    same = S.delong_paired(y, s, s)
    assert same["delta_auc_b_minus_a"] == 0.0 and same["var_delta"] == 0.0 and same["p_value_two_sided"] == 1.0
    perfect = S.delong_paired(y, y + 0.1 * s, 2.0 * y)
    assert perfect["auc_a"] == 1.0 and perfect["auc_b"] == 1.0 and perfect["p_value_two_sided"] == 1.0
    vs_const = S.delong_paired(y, y.astype(float), np.full(24, 0.3))
    assert vs_const["auc_a"] == 1.0 and abs(vs_const["auc_b"] - 0.5) < 1e-12   # midrank formula: 1-ulp rounding
    assert vs_const["var_delta"] == 0.0 and vs_const["p_value_two_sided"] is None   # undefined, not 0


def test_holm_adjust():
    adj = S.holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.5, "e": None})
    assert adj["e"] is None
    assert abs(adj["a"] - 0.04) < 1e-15 and abs(adj["c"] - 0.09) < 1e-12
    assert abs(adj["b"] - 0.09) < 1e-12 and adj["d"] == 0.5
    big = S.holm_adjust({"x": 0.4, "y": 0.6})
    assert big == {"x": 0.8, "y": 0.8}
    assert S.holm_adjust({}) == {}


def test_ece_bin_edges():
    one = S.expected_calibration_error([1], [1.0])
    assert one["ece"] == 0.0 and one["bins"][9]["n"] == 1
    assert S.expected_calibration_error([1], [0.9])["bins"][9]["n"] == 1
    assert S.expected_calibration_error([1], [0.8999])["bins"][8]["n"] == 1
    assert S.expected_calibration_error([0, 0], [1.0, 1.0])["ece"] == 1.0
    assert S.expected_calibration_error([], [])["ece"] is None


@needs_torch
def test_toy_eval_model_freeze_invariants():
    torch.manual_seed(0)
    conv, bn, head = nn.Conv2d(1, 2, 3, padding=1, bias=False), nn.BatchNorm2d(2), nn.Linear(2, 2)
    with torch.no_grad():
        bn.running_mean.copy_(torch.tensor([0.3, -0.2]))
        bn.running_var.copy_(torch.tensor([1.5, 0.7]))
        bn.num_batches_tracked.fill_(2346)

    class Toy(nn.Module):
        def forward(self, x):
            return head(torch.flatten(F.adaptive_avg_pool2d(F.relu(bn(conv(x))), 1), 1))

    model = Toy()
    model.add_module("conv", conv)
    model.add_module("bn", bn)
    model.add_module("head", head)
    model.eval()
    before = {k: v.clone() for k, v in bn.state_dict().items()}
    x = torch.randn(3, 1, 8, 8) + 26.2
    with torch.inference_mode():
        full = model(x)
        single = model(x[1:2])
    assert torch.allclose(full[1:2], single, atol=1e-6)                   # no cross-sample coupling
    for k, v in bn.state_dict().items():
        assert torch.equal(v, before[k]), k                               # running stats untouched
    assert not any(m.training for m in model.modules())


@needs_torch
def test_loader_freeze_flags_and_buffers(tmp_path):
    from load_frozen_model import load_frozen_model
    p = _write_fake_densenet(tmp_path)
    sd = torch.load(p, weights_only=True)["model_state_dict"]
    model = load_frozen_model(p, device="cpu")["model"]
    assert all(not q.requires_grad for q in model.parameters())
    assert not any(m.training for m in model.modules())
    for k, v in model.state_dict().items():
        assert torch.equal(v, sd[k]), k


@needs_torch
def test_loader_unsafe_pickle_refused_under_strict(tmp_path):
    from load_frozen_model import LoaderError, load_frozen_model
    p = _write_fake_densenet(tmp_path, name="opaque.pth", extra=_Opaque())
    with pytest.raises(LoaderError):
        load_frozen_model(p, device="cpu")
    meta = load_frozen_model(p, device="cpu", allow_unsafe_checkpoint_load=True)["metadata"]
    assert meta["unsafe_load_allowed"] is True and meta["checkpoint_load_mode"].startswith("weights_only=False")


def test_pipeline_scope_is_two_pipelines():
    import compare_external_models as CMP
    import external_statistics_supplement as SUP
    assert ANALYSED_PIPELINES == (PIPELINE_LOCKED, PIPELINE_PHOTOMETRIC)
    assert not hasattr(ext_common, "KNOWN_PIPELINES") and not hasattr(ext_common, "PIPELINE_SCALE")
    excluded = "scale_matched_physical_pipeline"
    assert excluded in ext_common.EXCLUDED_PIPELINES
    base_sup = ["--image-predictions", "i", "--slide-predictions", "s", "--output-dir", "o", "--model-label", "m"]
    base_cmp = ["--model-a", "a=x", "--model-b", "b=y", "--output-dir", "o"]
    for mod, base in ((SUP, base_sup), (CMP, base_cmp)):
        assert mod.parse_args(base + ["--pipeline-label", PIPELINE_LOCKED]).pipeline_label == PIPELINE_LOCKED
        with pytest.raises(SystemExit):
            mod.parse_args(base + ["--pipeline-label", excluded])
    if HAS_TORCH:
        import evaluate_external_model as EV
        assert set(EV.EXPECTED_SPATIAL_MODE) == set(ANALYSED_PIPELINES)
        base_ev = ["--checkpoint", "c", "--manifest", "m", "--preprocessed-dir", "d", "--output-dir", "o"]
        with pytest.raises(SystemExit):
            EV.parse_args(base_ev + ["--preprocessing-mode", excluded])


def test_no_bn_probe_artifacts_remain():
    """The removed probe leaves no file and no code-level use (imports, CLI flags, symbols)."""
    assert not (PKG / ("bn" + "_adaptation.py")).exists()
    tokens = ["import " + "bn_adaptation", "from " + "bn_adaptation", "--" + "bn-", "bn" + "_mode",
              "stage_" + "bn_probe", "PriorMix" + "BatchNorm2d", "apply_cohort" + "_reestimate",
              "apply_per_image" + "_prior", "--" + "stage"]
    targets = sorted(PKG.rglob("*.py"))
    targets += [p for p in (VALIDATION_DIR / "README.md", VALIDATION_DIR / "Run_DenseNet_External.ipynb") if p.exists()]
    hits = [(t.relative_to(VALIDATION_DIR).as_posix(), tok) for t in targets
            for tok in tokens if tok in t.read_text(encoding="utf-8", errors="replace")]
    assert hits == []
    assert not any("bn" + "_adaptation" in k for k in ext_common.PINNED_SHA256)


@needs_torch
def test_verify_preprocessing_rejects_checkpoint_sha_mismatch(tmp_path):
    from evaluate_external_model import EXPECTED_SPATIAL_MODE, verify_preprocessing
    for mode in ANALYSED_PIPELINES:
        d = tmp_path / mode
        d.mkdir()
        meta = {"counts": {"n_failed": 0, "n_success": 2, "n_skipped_existing": 0},
                "spatial_policy": {"mode": EXPECTED_SPATIAL_MODE[mode], "post_resize_clip_to_unit_interval": True},
                "save_standardized": False,
                "checkpoint": {"checkpoint_mean": [MU], "checkpoint_std": [SD], "checkpoint_file_sha256": "aa"}}
        (d / "preprocessing_metadata.json").write_text(json.dumps(meta))
        ok = verify_preprocessing(d, mode, 2, [MU], [SD], strict=False, ckpt_sha256="aa")
        assert ok["required_checks_failed"] == []
        bad = verify_preprocessing(d, mode, 2, [MU], [SD], strict=False, ckpt_sha256="bb")
        assert bad["required_checks_failed"] == ["preprocessing_checkpoint_sha_equals_model"]
        with pytest.raises(RuntimeError):
            verify_preprocessing(d, mode, 2, [MU], [SD], strict=True, ckpt_sha256="bb")


def test_reference_csv_gate(tmp_path):
    from ext_common import sha256_file
    rows = _synthetic_slides(7, pipeline=PIPELINE_LOCKED)
    labels = {r["slide_id"]: r["true_label"] for r in rows}
    path = tmp_path / "ref.csv"
    _write_csv(path, rows)
    y = np.array([r["true_label"] for r in rows])
    yp = np.array([r["pred_label"] for r in rows])
    p = np.array([r["mean_prob_glabrata"] for r in rows])
    c = np.array([r["confidence"] for r in rows])
    pt = S.binary_metrics(y, yp, p, c)
    bt = S.marginal_bootstrap(y, yp, p, c, n_bootstrap=200, seed=42, keys=("balanced_accuracy", "auroc"))
    t10 = {"point": {"balanced_accuracy": round(pt["balanced_accuracy"], 4), "auroc": round(pt["auroc"], 4)},
           "bootstrap_ci": {k: (round(bt[k]["ci_lower"], 4), round(bt[k]["ci_upper"], 4)) for k in bt},
           "clopper_pearson": {"recall_glabrata": tuple(round(S.clopper_pearson(
               pt["k_correct_glabrata"], 12)[x], 4) for x in ("ci_lower", "ci_upper"))}}
    sha = sha256_file(path)
    ok = S.validate_reference_slide_csv(path, PIPELINE_LOCKED, labels, sha, t10, n_bootstrap=200)
    assert ok["table10_reproduced"] is True

    def raises(tag, **kw):
        args = dict(path=path, pipeline=PIPELINE_LOCKED, expected_slide_labels=labels,
                    expected_sha256=sha, table10=t10, n_bootstrap=200)
        args.update(kw)
        with pytest.raises(ValueError, match=tag):
            S.validate_reference_slide_csv(**args)

    raises(r"\[pipeline\]", pipeline=PIPELINE_PHOTOMETRIC)
    raises(r"\[sha256\]", expected_sha256=None)
    raises(r"\[sha256\]", expected_sha256="0" * 64)
    raises(r"\[slide_set\]", expected_slide_labels={**labels, "20260509_alb_Z": 0})
    raises(r"\[labels\]", expected_slide_labels={k: 1 - v for k, v in labels.items()})
    raises(r"\[table10\]", table10={**t10, "point": {"balanced_accuracy": 0.1234}})
    bad = [dict(r) for r in rows]
    bad[0]["correct"] = 1 - bad[0]["correct"]
    _write_csv(tmp_path / "bad.csv", bad)
    raises(r"\[consistency\]", path=tmp_path / "bad.csv", expected_sha256=sha256_file(tmp_path / "bad.csv"))


def test_compare_cli_rejects_pipeline_mismatch(tmp_path):
    _write_csv(tmp_path / "a.csv", _synthetic_slides(0, pipeline=PIPELINE_LOCKED))
    _write_csv(tmp_path / "b.csv", _synthetic_slides(0, pipeline=PIPELINE_PHOTOMETRIC))
    r = _run([PKG / "compare_external_models.py", "--model-a", f"A={tmp_path / 'a.csv'}",
              "--model-b", f"B={tmp_path / 'b.csv'}", "--pipeline-label", PIPELINE_LOCKED,
              "--output-dir", tmp_path / "cmp", "--n-bootstrap", "50"])
    assert r.returncode != 0 and "preprocessing_mode" in r.stderr


def test_supplement_crosscheck_fails_closed(tmp_path):
    slides = _synthetic_slides(4, family="densenet121")
    _write_csv(tmp_path / "s.csv", slides)
    _write_csv(tmp_path / "i.csv", _synthetic_images(slides))
    base = [PKG / "external_statistics_supplement.py", "--image-predictions", tmp_path / "i.csv",
            "--slide-predictions", tmp_path / "s.csv", "--model-label", "densenet121",
            "--pipeline-label", PIPELINE_LOCKED, "--n-bootstrap", "200"]
    r0 = _run(base + ["--output-dir", tmp_path / "o0", "--no-require-legacy-crosscheck"])
    assert r0.returncode == 0, r0.stderr
    ci = json.loads((tmp_path / "o0" / "supplement_summary.json").read_text())["slide_bootstrap_ci"]
    same = {k: {"ci_lower": ci[k]["ci_lower"], "ci_upper": ci[k]["ci_upper"]}
            for k in ("balanced_accuracy", "matthews_corrcoef", "auroc")}
    (tmp_path / "same.json").write_text(json.dumps(same))
    differ = json.loads(json.dumps(same))
    differ["auroc"]["ci_upper"] = 0.123
    (tmp_path / "differ.json").write_text(json.dumps(differ))
    assert _run(base + ["--output-dir", tmp_path / "o1", "--legacy-ci-json", tmp_path / "same.json"]).returncode == 0
    assert _run(base + ["--output-dir", tmp_path / "o2", "--legacy-ci-json", tmp_path / "differ.json"]).returncode != 0
    assert _run(base + ["--output-dir", tmp_path / "o3"]).returncode != 0                    # required but missing
    _write_csv(tmp_path / "wrongfam.csv", _synthetic_slides(4, family="hybrid"))
    wrong = [str(x) if x != tmp_path / "s.csv" else str(tmp_path / "wrongfam.csv") for x in base]
    assert _run(wrong + ["--output-dir", tmp_path / "o4", "--no-require-legacy-crosscheck"]).returncode != 0


def test_plan_resolution_and_amendment(tmp_path):
    import run_external_densenet as R
    out, arch = tmp_path / "outputs", tmp_path / "archive"
    out.mkdir()
    new = {"created_utc": "x", "primary_endpoint": R.PRIMARY_ENDPOINT, "decision_rule": R.DECISION_RULE,
           "analysed_pipelines": {p: "" for p in ANALYSED_PIPELINES}, "checkpoint_sha256": "c" * 64,
           "multiplicity_policy": {"F1_model_contrast": "Holm"}, "args": {"bootstrap": "10000", "seed": "42"}}
    first = R.resolve_plan(out, arch, new, None)                          # no archive: v1, no amendment
    assert first["plan_version"] == 1 and first["amendment_path"] is None
    assert json.loads((out / "analysis_plan_frozen.json").read_text())["prior_execution_disclosure"] is None
    clean = R.resolve_plan(out, arch, new, None, "Earlier run discarded; outcomes were seen")
    disc = json.loads((out / "analysis_plan_frozen.json").read_text())["prior_execution_disclosure"]
    assert clean["plan_version"] == 1 and clean["disclosed_prior_execution"] is True
    assert disc["outcomes_of_earlier_execution_observed"] is True and disc["earlier_outputs_retained"] is False
    old_plan = {"created_utc": "2026-10-01T00:00:00Z", "stage": "frozen", "primary_endpoint": R.PRIMARY_ENDPOINT,
                "decision_rule": R.DECISION_RULE, "checkpoint_sha256": "c" * 64,
                "post_hoc_probes": {PIPELINE_PHOTOMETRIC: "probe", "bn" + "_probe": {"per_image_prior_N": 16.0}},
                "args": {"bootstrap": "10000", "seed": "42"}}
    (arch / "20261001T000000Z" / "outputs").mkdir(parents=True)
    (arch / "20261001T000000Z" / "outputs" / "analysis_plan_frozen.json").write_text(json.dumps(old_plan))
    with pytest.raises(SystemExit):
        R.resolve_plan(out, arch, new, None)                              # amendment without a reason
    with pytest.raises(SystemExit):
        R.resolve_plan(out, arch, {**new, "checkpoint_sha256": "d" * 64}, "reason")   # invariant broken
    info = R.resolve_plan(out, arch, new, "Supervisor decision: probe out of scope")
    amend = json.loads(Path(info["amendment_path"]).read_text())
    assert info["plan_version"] == 2 and amend["outcomes_observed_before_amendment"] is True
    assert ("post_hoc_probes." + "bn" + "_probe.per_image_prior_N") in amend["removed_keys"]
    assert "multiplicity_policy.F1_model_contrast" in amend["added_keys"] and all(amend["invariant_checks"].values())
    assert json.loads((out / "analysis_plan_frozen.json").read_text())["supersedes"]["sha256"] == amend["prior_plan_sha256"]


def test_results_summary_is_rebuilt_fresh(tmp_path):
    import run_external_densenet as R
    out = tmp_path / "outputs"
    (out).mkdir()
    (out / "results_summary.json").write_text(json.dumps({"elapsed_seconds_stale_stage": 1, "conditions": ["stale"]}))
    for pl in ANALYSED_PIPELINES:
        slides = _synthetic_slides(8, pipeline=pl, family="densenet121")
        _write_csv(tmp_path / f"{pl}.csv", slides)
        _write_csv(tmp_path / f"{pl}_img.csv", _synthetic_images(slides))
        r = _run([PKG / "external_statistics_supplement.py", "--image-predictions", tmp_path / f"{pl}_img.csv",
                  "--slide-predictions", tmp_path / f"{pl}.csv", "--output-dir", out / "densenet121" / pl / "supplement",
                  "--model-label", "densenet121", "--pipeline-label", pl, "--n-bootstrap", "100",
                  "--no-require-legacy-crosscheck"])
        assert r.returncode == 0, r.stderr
    summ = R.build_results_summary(out, ANALYSED_PIPELINES, comparisons_enabled=False)
    disk = json.loads((out / "results_summary.json").read_text())
    assert "elapsed_seconds_stale_stage" not in disk and set(disk) == set(summ)
    assert [c["pipeline"] for c in disk["conditions"]] == list(ANALYSED_PIPELINES)
    inf = json.loads((out / "inferential_summary.json").read_text())
    assert inf["primary_contrast"] is None and inf["family_F1_model_contrast"]["m"] == 0
    assert (out / "latex_external_table.tex").exists() and not (out / "latex_model_comparison.tex").exists()
    f2 = inf["family_F2_pipeline_contrast_within_densenet"]                 # no pairwise file: surfaced, not silent
    assert f2["m"] == 0 and f2["m_declared"] == 2 and f2["m_matches_declared"] is False and f2["missing_inputs"]
    pc = out / "densenet121" / "pipeline_comparison"
    pc.mkdir(parents=True)
    (pc / f"pairwise_summary__{PIPELINE_PHOTOMETRIC}.json").write_text(json.dumps(
        {"mcnemar_exact_pvalue_descriptive": 0.25}))                          # sign-flip key absent
    R.build_results_summary(out, ANALYSED_PIPELINES, comparisons_enabled=False)
    f2 = json.loads((out / "inferential_summary.json").read_text())["family_F2_pipeline_contrast_within_densenet"]
    assert f2["m"] == 1 and f2["m_matches_declared"] is False
    assert f2["missing_inputs"] == [f"pairwise_summary__{PIPELINE_PHOTOMETRIC}.json:"
                                    "sign_flip_exact_pvalue_for_mean_delta_true_class_probability_descriptive"]
    assert disk["optical_geometry"]["external_pixel_spacing_um"] == 0.08698


def test_return_bundle_exact_allowlist(tmp_path):
    from ext_common import BUNDLE_CONTENTS_NAME, build_return_bundle, expected_artifacts, sha256_file
    out = tmp_path / "outputs"
    exp = expected_artifacts(ANALYSED_PIPELINES, with_reference=True, amended=True)
    assert len(exp) == len(set(exp)) and "analysis_plan_amendment.json" in exp
    for rel in exp:
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).write_text(rel)
    man = build_return_bundle(out, exp)
    with zipfile.ZipFile(out / man["zip"]) as zf:
        assert sorted(zf.namelist()) == sorted(exp + [BUNDLE_CONTENTS_NAME])
    assert man["zip_sha256"] == sha256_file(out / man["zip"]) and man["n_files"] == len(exp)
    stray = out / "densenet121_bn_probe" / "x.json"                       # stale out-of-scope artifact
    stray.parent.mkdir()
    stray.write_text("{}")
    with pytest.raises(RuntimeError, match="extra"):
        build_return_bundle(out, exp)
    stray.unlink()
    (out / exp[0]).unlink()
    with pytest.raises(RuntimeError, match="missing"):
        build_return_bundle(out, exp)


def test_reproduction_check(tmp_path):
    from ext_common import reproduction_check

    def write(d, slide_p, img_p):
        d.mkdir(parents=True)
        _write_csv(d / "slide_predictions.csv",
                   [{"slide_id": f"s{i}", "pred_label": int(p > 0.5), "mean_prob_glabrata": p} for i, p in enumerate(slide_p)])
        _write_csv(d / "image_predictions.csv",
                   [{"relative_path": f"i{i}", "pred_label": int(p > 0.5), "prob_glabrata": p} for i, p in enumerate(img_p)])

    old, base = tmp_path / "old", [0.2, 0.7, 0.9]
    write(old, base, [0.1, 0.49999, 0.8])
    write(tmp_path / "same", base, [0.1, 0.49999, 0.8])
    r = reproduction_check(tmp_path / "same", old)
    assert r["status"] == "reproduced" and r["max_abs_delta_p_slide"] == 0.0 and r["within_p_tol"]
    write(tmp_path / "near", base, [0.1, 0.50001, 0.8])                 # flip within 1e-4 of 0.5: allowed
    assert reproduction_check(tmp_path / "near", old)["image_label_changes_near_boundary"] == 1
    write(tmp_path / "far", base, [0.1, 0.49999, 0.3])
    with pytest.raises(RuntimeError):
        reproduction_check(tmp_path / "far", old)
    write(tmp_path / "slideflip", [0.2, 0.4, 0.9], [0.1, 0.49999, 0.8])
    with pytest.raises(RuntimeError):
        reproduction_check(tmp_path / "slideflip", old)
    assert reproduction_check(tmp_path / "same", None)["status"] == "no_archived_baseline"


def test_image_cluster_bootstrap_shares_slide_draws():
    import external_statistics_supplement as SUP
    slides = _synthetic_slides(9)
    imgs = []
    for k, r in enumerate(slides):
        for j in range(1 + k % 4):                                        # 1..4 images per slide
            imgs.append({"slide_id": r["slide_id"], "true_label": r["true_label"]})
    ids = [r["slide_id"] for r in slides]
    y = np.array([r["true_label"] for r in slides])
    sizes = {r["slide_id"]: 1 + k % 4 for k, r in enumerate(slides)}
    ref = list(S.stratified_bootstrap_indices(y, 50, 42))
    for b, (picked, img_idx) in enumerate(SUP.cluster_bootstrap_draws(imgs, ids, y, n_bootstrap=50, seed=42)):
        assert np.array_equal(picked, ref[b])
        assert len(img_idx) == sum(sizes[ids[j]] for j in picked)
        assert all(imgs[i]["slide_id"] in {ids[j] for j in picked} for i in img_idx)


def test_environment_lock_check(tmp_path):
    from ext_common import check_environment, installed_version, parse_lock
    np_v, sp_v = installed_version("numpy"), installed_version("scipy")
    good = tmp_path / "good.lock"
    good.write_text(f"# header\nnumpy=={np_v}\nscipy=={sp_v}\n")
    assert parse_lock(good) == {"numpy": np_v, "scipy": sp_v}
    assert check_environment(good)["ok"] is True
    bad = tmp_path / "bad.lock"
    bad.write_text("numpy==0.0.0\n")
    with pytest.raises(RuntimeError, match="numpy"):
        check_environment(bad)
    assert check_environment(bad, strict=False)["mismatches"]["numpy"]["expected"] == "0.0.0"
    malformed = tmp_path / "m.lock"
    malformed.write_text("numpy>=1\n")
    with pytest.raises(ValueError):
        parse_lock(malformed)
    from ext_common import LOCK_PACKAGES, write_lock_from_environment
    if all(installed_version(p) for p in LOCK_PACKAGES):                  # always true in Colab after cell 3
        frozen = tmp_path / "frozen.lock"
        pins = write_lock_from_environment(frozen)
        assert parse_lock(frozen) == pins and check_environment(frozen)["ok"] is True
        assert {"numpy", "scipy", "scikit-learn", "tifffile", "pyyaml", "pytest"} <= set(pins)


# ============================================================================ O: optical geometry (7)
import optical_geometry as G  # noqa: E402


def test_optics_binning_derivation_matches_recorded_spacing():
    """0.08698 um/px = 2 x 2.74 um / 63 (2x2 binning of the 4512 px sensor); 0.1746 um/px = 11.0 um / 63."""
    E, I = G.EXTERNAL, G.INTERNAL
    assert E.native_px == E.binning * E.acquired_px == 4512 and E.binning == 2 and I.binning == 1
    assert abs(E.effective_pixel_pitch_um - 5.48) < 1e-12
    assert abs(E.derived_object_pitch_um - 5.48 / 63) < 1e-15
    assert abs(E.derived_object_pitch_um - 0.08698) / 0.08698 < 1e-4
    assert abs(I.derived_object_pitch_um - 0.1746) / 0.1746 < 1e-4
    assert abs(E.recorded_fov_um - 196.22688) < 1e-9 and abs(I.recorded_fov_um - 209.52) < 1e-9
    assert abs(E.native_px * E.sensor_pitch_um / 63 - E.derived_fov_um) < 1e-9           # full sensor imaged


def test_scale_geometry_values():
    g = G.GEOMETRY
    assert g == G.scale_geometry(0.08698, 0.1746)
    assert abs(g["fov_ratio_external_over_training_linear"] - 196.22688 / 209.52) < 1e-12
    assert abs(g["training_over_external_linear_sampling_density_ratio"] - 0.1746 / 0.08698) < 1e-12
    assert abs(g["locked_input_pitch_external_um"] - 196.22688 / 384) < 1e-12
    assert abs(g["locked_apparent_magnification_external_vs_training"] - 1.067744) < 1e-6
    assert g["locked_resize_factor_external"] == 2256 / 384 and g["locked_resize_factor_training"] == 1200 / 384
    assert g["scale_matched_tile_px"] == 1124 and g["scale_matched_tile_in_model_px"] == 360
    assert g["scale_matched_border_px_per_side"] == 38.0 and not g["scale_matched_tile_cropped"]
    assert abs(g["scale_matched_padding_fraction"] - (1 - (1124 / 1200) ** 2)) < 1e-15
    with pytest.raises(ValueError):
        G.scale_geometry(0.0, 0.1746)
    with pytest.raises(ValueError):
        G.scale_geometry(float("nan"), 0.1746)
    with pytest.raises(ValueError):
        G.scale_geometry(0.08698, 0.1746, canvas_px=0)


def test_scale_geometry_padding_and_fov_invariants_random():
    rng = np.random.default_rng(0)
    for _ in range(300):
        e, t = rng.uniform(0.02, 0.4), rng.uniform(0.05, 0.4)
        n, c, m = int(rng.integers(8, 64)), int(rng.integers(8, 64)), int(rng.integers(4, 32))
        g = G.scale_geometry(e, t, external_px=n, training_px=c, canvas_px=c, model_px=m)
        assert abs(g["scale_matched_content_fraction"] + g["scale_matched_padding_fraction"] - 1) < 1e-12
        assert 0.0 <= g["scale_matched_padding_fraction"] < 1.0
        assert abs(g["scale_matched_tile_px"] - n * e / t) <= 0.5                          # rounding only
        assert abs(g["scale_matched_tile_px"] * t - n * e) <= 0.5 * t + 1e-12              # physical FOV conserved
        assert g["scale_matched_tile_cropped"] == (g["scale_matched_tile_px"] > c)
        assert abs(g["scale_matched_input_pitch_um"] - g["locked_input_pitch_training_um"]) < 1e-12
        assert abs(g["locked_apparent_magnification_external_vs_training"]
                   * g["fov_ratio_external_over_training_linear"] - 1) < 1e-12
        assert abs(g["content_resampling_factor_external_to_training"]
                   * g["training_over_external_linear_sampling_density_ratio"] - 1) < 1e-12


def test_config_geometry_audit_is_fail_closed(tmp_path):
    import yaml
    cfg_p = VALIDATION_DIR / "configs" / "external_validation_config.yaml"
    md_p = VALIDATION_DIR / "configs" / "external_validation_metadata.json"
    rep = G.audit_config_geometry(cfg_p, md_p)                                       # the pinned configs
    assert rep["all_fields_match"] and rep["n_fields_checked"] == (
        len(G.CONFIG_SPACING_FIELDS) + len(G.METADATA_SPACING_FIELDS) + len(G.CONFIG_HARDWARE_FIELDS)
        + len(G.METADATA_HARDWARE_FIELDS) + len(G.CONFIG_DERIVED_FIELDS) + len(G.METADATA_DERIVED_FIELDS))
    assert rep["analysed_pipelines_consume_spacing"] is False and rep["external_spacing_um"] == 0.08698
    cfg, md = yaml.safe_load(cfg_p.read_text()), json.loads(md_p.read_text())

    def audit(c, m, tag):
        cp, mp = tmp_path / f"{tag}.yaml", tmp_path / f"{tag}.json"
        cp.write_text(yaml.safe_dump(c))
        mp.write_text(json.dumps(m))
        return G.audit_config_geometry(cp, mp)

    def mutate(doc, dotted, value):
        d = json.loads(json.dumps(doc))
        *head, last = dotted.split(".")
        cur = d
        for h in head:
            cur = cur[h]
        if value is None:
            del cur[last]
        else:
            cur[last] = value
        return d

    assert audit(cfg, md, "same")["all_fields_match"]                                # YAML round trip
    with pytest.raises(RuntimeError, match="does not match"):                       # one spacing off
        audit(cfg, mutate(md, "external_acquisition_metadata.pixel_spacing_um.y", 0.05), "spacing")
    with pytest.raises(RuntimeError, match="does not match"):                       # derived figure off
        audit(mutate(cfg, "scale_matching_sensitivity.expected_padding_fraction_in_training_canvas", 0.2), md, "pad")
    with pytest.raises(RuntimeError, match="does not match"):                       # binning off
        audit(mutate(cfg, "external_domain.binning", 1), md, "bin")
    with pytest.raises(RuntimeError, match="not found"):
        audit(mutate(cfg, "scale_matching_sensitivity.external_pixel_spacing_um", None), md, "gone")


def test_tiff_resolution_tags_give_spacing_without_reading_pixels(tmp_path):
    import tifffile
    assert abs(G.spacing_um_from_resolution((1_000_000_000, 8698), 3) - 0.08698) < 1e-12      # px/cm rational
    assert abs(G.spacing_um_from_resolution(25400 / 0.08698, 2) - 0.08698) < 1e-12           # px/inch float
    assert G.spacing_um_from_resolution((1, 1), 1) is None and G.spacing_um_from_resolution(None, 3) is None
    p = tmp_path / "ok.tif"
    tifffile.imwrite(p, np.zeros((4, 4), np.uint16), resolution=(1e4 / 0.08698, 1e4 / 0.08698), resolutionunit=3)
    r = G.read_tiff_pixel_spacing_um(p)
    assert abs(r["spacing_x_um"] - 0.08698) / 0.08698 < 1e-6 and r["matches_recorded_spacing"] and r["shape"] == (4, 4)
    bad = tmp_path / "bad.tif"
    tifffile.imwrite(bad, np.zeros((4, 4), np.uint16), resolution=(1e4 / 0.2, 1e4 / 0.2), resolutionunit=3)
    with pytest.raises(ValueError):
        G.read_tiff_pixel_spacing_um(bad)


def test_excluded_rationale_plan_and_bundle_carry_geometry():
    text = ext_common.EXCLUDED_PIPELINES["scale_matched_physical_pipeline"]
    for token in ("0.511", "0.546", "1.068x", "6.8%", "1124-px", "12.3%", "38-px"):
        assert token in text, token
    blk = G.plan_block()
    assert blk["external_pixel_spacing_um"] == 0.08698 and blk["external_binning"] == 2
    assert blk["locked_apparent_magnification_external_vs_training"] == 1.0677
    for amended in (False, True):
        assert ext_common.OPTICAL_GEOMETRY_PROVENANCE in ext_common.expected_artifacts(
            ANALYSED_PIPELINES, with_reference=amended, amended=amended)
    n_tests = sum(1 for k, v in globals().items() if k.startswith("test_") and callable(v))
    assert n_tests == ext_common.EXPECTED_TEST_COUNT == 48


def test_prior_execution_note_placeholder_is_refused(tmp_path):
    import run_external_densenet as R
    for bad in ("Placeholder", "placeholder.", "TODO", "<describe>", "n/a", "run discarded"):
        with pytest.raises(SystemExit):
            R.validate_prior_execution_note(bad)
    assert R.validate_prior_execution_note("") == "" and R.validate_prior_execution_note(None) == ""
    assert R.validate_prior_execution_note("  Earlier run discarded; outcomes were seen ") == \
        "Earlier run discarded; outcomes were seen"
    out = tmp_path / "outputs"
    out.mkdir()
    with pytest.raises(SystemExit):
        R.resolve_plan(out, tmp_path / "archive", {"args": {}}, None, "Placeholder")
    assert not (out / "analysis_plan_frozen.json").exists()
