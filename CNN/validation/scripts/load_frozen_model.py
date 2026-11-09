#!/usr/bin/env python3
"""
load_frozen_model.py

Fail-closed loader for the frozen DenseNet-121 used in external validation
(checkpoint ``final_model_standard_cnn.pth`` written by
``CNN/scripts/densenet_fungi.ipynb`` cell 7).

Guarantees (each failure raises unless --no-strict)
-------------------------------------------------
- Checkpoint read with ``weights_only=True``. A fallback to
  ``weights_only=False`` is refused unless ``allow_unsafe_checkpoint_load`` is
  set explicitly; the load mode is always recorded.
- ``mean``/``std`` taken from the checkpoint's top-level keys only; one
  channel, finite, std > 0. No default values exist in this module.
- ``class_to_idx`` == {'Candida albicans': 0, 'Candida glabrata': 1}.
- ``config.experiment_name`` == 'standard_cnn', image size 384, 1 channel.
- ``optimal_epochs`` == 51 (main.tex §3.4).
- Strict ``load_state_dict`` with an exact key-set match; 6,949,634
  parameters, 727 state entries, 121 BatchNorm layers (main.tex §3.3.2).
- Provenance fingerprint: every BatchNorm ``num_batches_tracked`` should equal
  51 epochs x ceil(2910 / 64) = 2,346 optimizer steps of the final training
  run. Reported (not enforced) because a resumed run can legitimately differ.
- All floating tensors FP32, model in eval(), gradients disabled, DenseNet
  drop_rate == 0 everywhere.

No forward pass is executed by this module.

Author: Rodrigo Sá
Date: 2026
"""

from __future__ import annotations

import ext_common  # noqa: F401  (sets CUBLAS_WORKSPACE_CONFIG before torch)

import argparse
import math
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from ext_common import (
    EXPECTED_CLASS_TO_IDX,
    import_module_from_path,
    resolve_device,
    sha256_file,
    write_json,
)

HERE = Path(__file__).resolve().parent
FAMILY = "densenet121"

SPEC: Dict[str, Any] = {
    "display_name": "DenseNet-121",
    "experiment_name": "standard_cnn",
    "expected_n_parameters": 6_949_634,       # main.tex §3.3.2
    "expected_n_state_entries": 727,
    "expected_n_batchnorm": 121,
    "expected_optimal_epochs": 51,            # main.tex §3.4
    "n_development_patches": 2910,            # main.tex §3.2
    "batch_size": 64,                         # main.tex Table A.1
}


class LoaderError(RuntimeError):
    pass


def _fail(msg: str, strict: bool, warnings_out: List[str]) -> None:
    if strict:
        raise LoaderError(msg)
    warnings_out.append(msg)


def read_checkpoint(path: Path, *, allow_unsafe: bool = False) -> Tuple[Dict[str, Any], str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
        mode = "weights_only=True"
    except Exception as exc:
        if not allow_unsafe:
            raise LoaderError(
                f"weights_only=True failed for {path.name} ({type(exc).__name__}: {exc}). "
                "Refusing to unpickle arbitrary objects. If this is your own artifact and the "
                "archived checkpoint_identity.json shows the same fallback, rerun with "
                "--allow-unsafe-checkpoint-load (recorded in the analysis plan)."
            ) from exc
        warnings.warn(
            f"weights_only=True failed for {path.name} ({type(exc).__name__}: {exc}); "
            "falling back to weights_only=False because --allow-unsafe-checkpoint-load was given."
        )
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        mode = f"weights_only=False (explicitly allowed; fallback after {type(exc).__name__})"
    if not isinstance(ckpt, dict):
        raise LoaderError(f"Checkpoint is not a dict: {type(ckpt)}")
    return ckpt, mode


def strict_mean_std(ckpt: Dict[str, Any]) -> Tuple[List[float], List[float]]:
    """One-channel mean/std from top-level keys only; never defaults."""
    if "mean" not in ckpt or "std" not in ckpt:
        raise LoaderError("Checkpoint lacks top-level 'mean'/'std'. Refusing to use defaults.")

    def as_list(v: Any) -> List[float]:
        if torch.is_tensor(v):
            v = v.detach().cpu().tolist()
        if isinstance(v, (int, float)):
            v = [v]
        if not isinstance(v, (list, tuple)):
            raise LoaderError(f"Unsupported mean/std type: {type(v)}")
        return [float(x) for x in v]

    mean = as_list(ckpt["mean"])
    std = as_list(ckpt["std"])
    if len(mean) != 1 or len(std) != 1:
        raise LoaderError(f"Expected one-channel mean/std, got mean={mean}, std={std}")
    if not all(math.isfinite(x) for x in mean + std) or std[0] <= 0:
        raise LoaderError(f"Invalid mean/std: mean={mean}, std={std}")
    return mean, std


def expected_bn_steps() -> int:
    return int(SPEC["expected_optimal_epochs"]) * math.ceil(
        SPEC["n_development_patches"] / SPEC["batch_size"])


def load_frozen_model(
    checkpoint_path: Path,
    *,
    device: str = "auto",
    densenet_source: Path = HERE / "densenet_model.py",
    strict: bool = True,
    allow_unsafe_checkpoint_load: bool = False,
) -> Dict[str, Any]:
    warnings_out: List[str] = []
    dev = resolve_device(device)
    checkpoint_path = Path(checkpoint_path)

    ckpt, load_mode = read_checkpoint(checkpoint_path, allow_unsafe=allow_unsafe_checkpoint_load)
    sd = ckpt.get("model_state_dict")
    if not isinstance(sd, dict):
        raise LoaderError("Checkpoint has no 'model_state_dict'.")
    mean, std = strict_mean_std(ckpt)
    config = ckpt.get("config")
    if not isinstance(config, dict):
        raise LoaderError("Checkpoint has no 'config' dict; refusing to reconstruct one.")

    if config.get("experiment_name") != SPEC["experiment_name"]:
        _fail(f"experiment_name={config.get('experiment_name')!r}, expected 'standard_cnn'",
              strict, warnings_out)
    if int(config.get("image_size", -1)) != 384 or int(config.get("channels", -1)) != 1:
        _fail(f"image_size/channels = {config.get('image_size')}/{config.get('channels')}, "
              "expected 384/1", strict, warnings_out)

    opt_ep = ckpt.get("optimal_epochs")
    if opt_ep is None or int(opt_ep) != int(SPEC["expected_optimal_epochs"]):
        _fail(f"optimal_epochs={opt_ep}, expected {SPEC['expected_optimal_epochs']}",
              strict, warnings_out)

    c2i = ckpt.get("class_to_idx")
    if c2i is None or dict(c2i) != EXPECTED_CLASS_TO_IDX:
        _fail(f"class_to_idx {c2i} != {EXPECTED_CLASS_TO_IDX}", strict, warnings_out)

    mod = import_module_from_path("_ext_densenet_model", densenet_source)
    model = mod.DenseNet121Classifier(num_classes=2, channels=1)

    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(sd.keys())
    if model_keys != ckpt_keys:
        raise LoaderError(
            "State-dict key mismatch: "
            f"missing_in_ckpt={sorted(model_keys - ckpt_keys)[:10]}, "
            f"unexpected_in_ckpt={sorted(ckpt_keys - model_keys)[:10]}"
        )
    result = model.load_state_dict(sd, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise LoaderError(f"load_state_dict incompatible keys: {result}")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    n_params = sum(p.numel() for p in model.parameters())
    n_state = len(model.state_dict())
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    dtypes = sorted({str(t.dtype) for t in model.state_dict().values() if t.is_floating_point()})
    drop_rates = sorted({float(m.drop_rate) for m in model.modules() if hasattr(m, "drop_rate")})

    if n_params != SPEC["expected_n_parameters"]:
        _fail(f"Parameter count {n_params} != {SPEC['expected_n_parameters']}", strict, warnings_out)
    if n_state != SPEC["expected_n_state_entries"]:
        _fail(f"State entries {n_state} != {SPEC['expected_n_state_entries']}", strict, warnings_out)
    if len(bns) != SPEC["expected_n_batchnorm"]:
        _fail(f"BatchNorm count {len(bns)} != {SPEC['expected_n_batchnorm']}", strict, warnings_out)
    if dtypes != ["torch.float32"]:
        _fail(f"Floating dtypes {dtypes}; FP32 required", strict, warnings_out)
    if drop_rates and drop_rates != [0.0]:
        _fail(f"Non-zero DenseNet drop_rate found: {drop_rates}", strict, warnings_out)
    if any(m.training for m in model.modules()):
        raise LoaderError("Some module is still in training mode after eval().")

    nbt = sorted({int(m.num_batches_tracked.item()) for m in bns})
    exp_steps = expected_bn_steps()
    provenance = {
        "bn_num_batches_tracked_unique": nbt,
        "expected_steps_51_epochs_x_46_batches": exp_steps,
        "consistent_with_final_training_run": nbt == [exp_steps],
        "note": ("All 121 BN layers counting exactly 2,346 training steps is consistent with "
                 "the 51-epoch final run on 2,910 patches at batch size 64 (2,295 if the "
                 "training loader dropped the last partial batch). It is a provenance "
                 "indicator, not a proof of identity with the held-out evaluation."),
    }
    if nbt != [exp_steps]:
        warnings_out.append(f"num_batches_tracked {nbt} != [{exp_steps}] (reported, not enforced)")

    meta = {
        "schema_version": "3.1",
        "script": "load_frozen_model.py",
        "created_unix_time": time.time(),
        "family": FAMILY,
        "display_name": SPEC["display_name"],
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "checkpoint_load_mode": load_mode,
        "unsafe_load_allowed": bool(allow_unsafe_checkpoint_load),
        "checkpoint_top_level_keys": sorted(str(k) for k in ckpt.keys()),
        "experiment_name": config.get("experiment_name"),
        "optimal_epochs": opt_ep,
        "class_to_idx": c2i,
        "mean": mean,
        "std": std,
        "n_parameters": int(n_params),
        "n_state_entries": int(n_state),
        "n_batchnorm_layers": len(bns),
        "floating_dtypes": dtypes,
        "drop_rates_unique": drop_rates,
        "training_provenance": provenance,
        "densenet_source": str(densenet_source),
        "densenet_source_sha256": sha256_file(densenet_source),
        "eval_mode": True,
        "gradients_disabled": True,
        "device": str(dev),
        "warnings": warnings_out,
        "forward_pass_executed": False,
    }
    model.to(dev)
    return {"model": model, "mean": mean, "std": std, "device": dev, "metadata": meta,
            "config": config}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Load and verify the frozen DenseNet-121 (no inference).")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--densenet-source", type=Path, default=HERE / "densenet_model.py")
    p.add_argument("--device", default="cpu")
    p.add_argument("--metadata-output", type=Path, default=None)
    p.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--allow-unsafe-checkpoint-load", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_frozen_model(args.checkpoint, device=args.device,
                               densenet_source=args.densenet_source, strict=args.strict,
                               allow_unsafe_checkpoint_load=args.allow_unsafe_checkpoint_load)
    meta = bundle["metadata"]
    if args.metadata_output:
        write_json(meta, args.metadata_output)
    print(f"LOADER_OK sha256={meta['checkpoint_sha256']} params={meta['n_parameters']} "
          f"epochs={meta['optimal_epochs']} mean={meta['mean']} std={meta['std']} "
          f"class_to_idx={meta['class_to_idx']} load_mode={meta['checkpoint_load_mode']} "
          f"bn_steps={meta['training_provenance']['bn_num_batches_tracked_unique']} "
          f"(expected {meta['training_provenance']['expected_steps_51_epochs_x_46_batches']})")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
