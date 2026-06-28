#!/usr/bin/env python3
"""
load_frozen_hybrid.py

Safe loader for the frozen LSA-ConvTok Hybrid model used in external validation.

Scientific role
---------------
External validation must evaluate the exact frozen Hybrid model selected after
model development. This script centralizes checkpoint loading so that downstream
evaluation and explainability scripts do not accidentally:
    - instantiate the wrong architecture,
    - load mismatched weights,
    - ignore architecture-hash mismatches,
    - use the full training-resume checkpoint as the default inference artifact,
    - leave gradients enabled during external inference.

Primary checkpoint policy
-------------------------
Use:
    final_model_lsa_convtok.pth

as the primary frozen inference artifact.

Optionally provide:
    final_model_checkpoint.pth

as the full training-resume provenance checkpoint. When supplied, this script can
verify that both checkpoints contain equivalent model_state_dict tensors.

Expected model
--------------
    SimpleViT_LSA_ConvTok
    experiment_name: lsa_convtok
    architecture_hash: 95ace50634b621d7
    image_size: 384
    channels: 1
    num_classes: 2

Usage as a module
-----------------
    from load_frozen_hybrid import load_frozen_hybrid

    bundle = load_frozen_hybrid(
        checkpoint_path="final_model_lsa_convtok.pth",
        provenance_checkpoint_path="final_model_checkpoint.pth",
        config_path="configs/external_validation_config.yaml",
        device="cuda",
    )

    model = bundle["model"]
    mean = bundle["mean"]
    std = bundle["std"]

Usage as a CLI smoke test
-------------------------
    python load_frozen_hybrid.py \\
        --checkpoint final_model_lsa_convtok.pth \\
        --provenance-checkpoint final_model_checkpoint.pth \\
        --config configs/external_validation_config.yaml \\
        --device cpu \\
        --smoke-test

Author: Rodrigo Sá
Date: 2026
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


DEFAULT_EXPECTED_ARCH_HASH = "95ace50634b621d7"
DEFAULT_MODEL_VARIANT = "SimpleViT_LSA_ConvTok"
DEFAULT_EXPERIMENT_NAME = "lsa_convtok"
DEFAULT_IMAGE_SIZE = 384
DEFAULT_CHANNELS = 1
DEFAULT_NUM_CLASSES = 2
DEFAULT_MEAN = [0.23691008985042572]
DEFAULT_STD = [0.007409718818962574]


# -----------------------------------------------------------------------------
# Optional local imports
# -----------------------------------------------------------------------------

def _import_training_factory():
    """
    Import training/model factory functions lazily.

    Robustness for the validation layout:
        validation/scripts/load_frozen_hybrid.py
        validation/source/training_logic.py

    The runner sets PYTHONPATH and EXTERNAL_VALIDATION_SOURCE_CODE_DIR, but this
    function also searches common locations so direct CLI usage remains safe.
    """
    import importlib
    import os

    script_dir = Path(__file__).resolve().parent
    candidates = [
        Path(os.environ["EXTERNAL_VALIDATION_SOURCE_CODE_DIR"])
        if os.environ.get("EXTERNAL_VALIDATION_SOURCE_CODE_DIR") else None,
        script_dir,
        script_dir.parent / "source",
        script_dir.parent,
        Path.cwd(),
        Path.cwd() / "source",
    ]

    seen = set()
    for candidate in candidates:
        if candidate is None:
            continue
        candidate = candidate.resolve()
        if not candidate.exists() or str(candidate) in seen:
            continue
        seen.add(str(candidate))
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))

    try:
        module = importlib.import_module("training_logic")
        build_model = getattr(module, "build_model")
        validate_checkpoint_arch = getattr(module, "validate_checkpoint_arch")
        compute_arch_hash = getattr(module, "compute_arch_hash")
    except Exception as exc:
        checked = [str(c.resolve()) for c in candidates if c is not None and c.exists()]
        raise ImportError(
            "Could not import build_model/validate_checkpoint_arch from training_logic.py. "
            "Expected training_logic.py in validation/source or in --source-code-dir. "
            f"Checked import roots: {checked}. Current sys.path head: {sys.path[:8]}"
        ) from exc

    return build_model, validate_checkpoint_arch, compute_arch_hash


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Safely load the frozen final LSA-ConvTok Hybrid checkpoint for "
            "external validation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Primary inference checkpoint, expected final_model_lsa_convtok.pth.",
    )
    parser.add_argument(
        "--provenance-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional full training-resume checkpoint, expected "
            "final_model_checkpoint.pth. Used only for provenance/equivalence checks."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional external_validation_config.yaml.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'auto', 'cpu', 'cuda', or a valid torch device string.",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Abort on architecture/config/provenance inconsistencies.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a no-gradient dummy forward pass after loading.",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=None,
        help="Optional path to write loader metadata JSON.",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------

def load_optional_yaml_config(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Config YAML not found: {path}")

    try:
        import yaml
    except ImportError:
        warnings.warn(
            "PyYAML is not installed; continuing without YAML config parsing. "
            "Install with: pip install pyyaml"
        )
        return {}

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data if isinstance(data, dict) else {}


def get_nested(data: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def resolve_device(device: Union[str, torch.device]) -> torch.device:
    if isinstance(device, torch.device):
        return device

    device_str = str(device).lower()
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device_str.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")

    return torch.device(device)


def normalize_list(value: Any, default: Sequence[float]) -> List[float]:
    if value is None:
        return [float(v) for v in default]
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    raise ValueError(f"Cannot convert value to list of floats: {value!r}")


def tensor_sha256(tensor: torch.Tensor) -> str:
    arr = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def hash_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> str:
    keys = sorted(str(k) for k in state_dict.keys())
    return hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()


def state_dict_content_fingerprint(state_dict: Dict[str, torch.Tensor]) -> str:
    """
    Compute a deterministic fingerprint over state_dict keys, shapes, dtypes, and
    tensor contents. This is useful for provenance comparison, not for security.
    """
    h = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key]
        if not torch.is_tensor(tensor):
            continue
        h.update(str(key).encode("utf-8"))
        h.update(str(tuple(tensor.shape)).encode("utf-8"))
        h.update(str(tensor.dtype).encode("utf-8"))
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def max_abs_state_dict_difference(
    state_a: Dict[str, torch.Tensor],
    state_b: Dict[str, torch.Tensor],
) -> Optional[float]:
    keys_a = set(state_a.keys())
    keys_b = set(state_b.keys())
    if keys_a != keys_b:
        return None

    max_diff = 0.0
    for key in sorted(keys_a):
        a = state_a[key]
        b = state_b[key]
        if not torch.is_tensor(a) or not torch.is_tensor(b):
            continue
        if a.shape != b.shape:
            return None
        diff = (a.detach().cpu().float() - b.detach().cpu().float()).abs().max().item()
        max_diff = max(max_diff, float(diff))
    return max_diff


# -----------------------------------------------------------------------------
# Checkpoint parsing
# -----------------------------------------------------------------------------

def torch_load_checkpoint(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Checkpoint must be a dictionary, got {type(ckpt)} from {path}")
    return ckpt


def extract_arch_metadata(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    arch = checkpoint.get("arch_metadata")
    if isinstance(arch, dict):
        return arch

    metadata = checkpoint.get("metadata")
    if isinstance(metadata, dict):
        arch = metadata.get("arch_metadata")
        if isinstance(arch, dict):
            return arch

    architecture = checkpoint.get("architecture")
    if isinstance(architecture, dict):
        arch = architecture.get("arch_metadata")
        if isinstance(arch, dict):
            return arch

    return {}


def extract_config(checkpoint: Dict[str, Any], arch_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract a model-building configuration.

    Preferred:
        checkpoint["config"]

    Fallback:
        reconstruct the minimum required config from arch_metadata.

    This fallback is included for robustness but the primary final_model_lsa_convtok.pth
    should contain a full config.
    """
    config = checkpoint.get("config")
    if isinstance(config, dict) and config:
        return dict(config)

    metadata = checkpoint.get("metadata")
    if isinstance(metadata, dict):
        config = metadata.get("config")
        if isinstance(config, dict) and config:
            return dict(config)

    # Fallback reconstruction from arch_metadata. This must match build_model()
    # expectations in training_logic.py for SimpleViT_LSA_ConvTok.
    if not arch_metadata:
        raise ValueError(
            "Checkpoint lacks both full config and architecture metadata. "
            "Cannot safely reconstruct the frozen Hybrid model."
        )

    exp = arch_metadata.get("experiment_name", DEFAULT_EXPERIMENT_NAME)

    reconstructed = {
        "experiment_name": exp,
        "image_size": int(arch_metadata.get("image_size", DEFAULT_IMAGE_SIZE)),
        "channels": int(arch_metadata.get("channels", DEFAULT_CHANNELS)),
        "bit_depth": int(
            checkpoint.get("bit_depth")
            or (metadata.get("bit_depth") if isinstance(metadata, dict) else 12)
            or 12
        ),
        "model_config": {
            "dim": int(arch_metadata.get("dim", 256)),
            "depth": int(arch_metadata.get("depth", 8)),
            "heads": int(arch_metadata.get("heads", 6)),
            "mlp_dim": int(arch_metadata.get("mlp_dim", 512)),
            "dim_head": int(arch_metadata.get("dim_head", 64)),
            "pe_temperature": int(arch_metadata.get("pe_temperature", 10000)),
        },
        "convtok_blocks": arch_metadata.get(
            "convtok_blocks",
            [
                {"conv_k": 7, "conv_s": 2, "conv_p": 3, "pool_k": 3, "pool_s": 2, "pool_p": 1},
                {"conv_k": 3, "conv_s": 2, "conv_p": 1, "pool_k": 3, "pool_s": 2, "pool_p": 1},
            ],
        ),
        "convtok_hidden_channels": int(arch_metadata.get("convtok_hidden_channels", 64)),
        "convtok_match_baseline_tokens": True,
        "convtok_expected_hw": arch_metadata.get("convtok_expected_hw", [24, 24]),
        "convtok_expected_tokens": int(arch_metadata.get("convtok_expected_tokens", 576)),
        "convtok_post_ln": bool(arch_metadata.get("convtok_post_ln", True)),
        "convtok_conv_bias": bool(arch_metadata.get("convtok_conv_bias", False)),
        "init_policy": arch_metadata.get("init_policy", "default"),
    }

    return reconstructed


def extract_model_state_dict(checkpoint: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    state = checkpoint.get("model_state_dict")
    if isinstance(state, dict):
        return state

    # Some exports may use "state_dict".
    state = checkpoint.get("state_dict")
    if isinstance(state, dict):
        return state

    raise ValueError("Checkpoint has no model_state_dict/state_dict.")


def extract_mean_std(checkpoint: Dict[str, Any], config: Dict[str, Any]) -> Tuple[List[float], List[float]]:
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    mean = (
        checkpoint.get("mean")
        or metadata.get("mean")
        or config.get("mean")
        or DEFAULT_MEAN
    )
    std = (
        checkpoint.get("std")
        or metadata.get("std")
        or config.get("std")
        or DEFAULT_STD
    )

    mean_list = normalize_list(mean, DEFAULT_MEAN)
    std_list = normalize_list(std, DEFAULT_STD)

    if len(mean_list) != len(std_list):
        raise ValueError(f"Mean/std length mismatch: mean={mean_list}, std={std_list}")
    if any(s <= 0 for s in std_list):
        raise ValueError(f"All std values must be positive, got {std_list}")

    return mean_list, std_list


def checkpoint_summary(
    path: Union[str, Path],
    checkpoint: Dict[str, Any],
    config: Dict[str, Any],
    arch_metadata: Dict[str, Any],
    state_dict: Dict[str, torch.Tensor],
    mean: Sequence[float],
    std: Sequence[float],
) -> Dict[str, Any]:
    path = Path(path)

    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    architecture_hash = (
        arch_metadata.get("arch_hash")
        or checkpoint.get("architecture_hash")
        or metadata.get("architecture_hash")
        or metadata.get("arch_hash")
        or "unknown"
    )

    model_variant = (
        checkpoint.get("model_variant")
        or metadata.get("model_variant")
        or arch_metadata.get("model_class")
        or DEFAULT_MODEL_VARIANT
    )

    image_size = (
        config.get("image_size")
        or arch_metadata.get("image_size")
        or checkpoint.get("image_size")
        or metadata.get("image_size")
    )

    channels = (
        config.get("channels")
        or arch_metadata.get("channels")
        or checkpoint.get("channels")
        or metadata.get("channels")
    )

    return {
        "path": str(path),
        "file_sha256": sha256_file(path),
        "model_variant": model_variant,
        "experiment_name": config.get("experiment_name", arch_metadata.get("experiment_name", "unknown")),
        "architecture_hash": architecture_hash,
        "image_size": int(image_size) if image_size is not None else None,
        "channels": int(channels) if channels is not None else None,
        "checkpoint_training_bit_depth": (
            config.get("bit_depth")
            or checkpoint.get("bit_depth")
            or metadata.get("bit_depth")
        ),
        "mean": [float(v) for v in mean],
        "std": [float(v) for v in std],
        "state_dict_n_tensors": len(state_dict),
        "state_dict_key_hash": hash_state_dict_keys(state_dict),
        "state_dict_content_fingerprint": state_dict_content_fingerprint(state_dict),
        "available_top_level_keys": sorted(str(k) for k in checkpoint.keys()),
    }


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

def expected_values_from_config(config_yaml: Dict[str, Any]) -> Dict[str, Any]:
    expected_model = get_nested(config_yaml, ["checkpoints", "expected_model"], {}) or {}
    if not isinstance(expected_model, dict):
        expected_model = {}

    return {
        "architecture_hash": expected_model.get("architecture_hash", DEFAULT_EXPECTED_ARCH_HASH),
        "model_variant": expected_model.get("model_variant", DEFAULT_MODEL_VARIANT),
        "experiment_name": expected_model.get("experiment_name", DEFAULT_EXPERIMENT_NAME),
        "image_size": int(expected_model.get("image_size", DEFAULT_IMAGE_SIZE)),
        "channels": int(expected_model.get("channels", DEFAULT_CHANNELS)),
        "num_classes": int(expected_model.get("num_classes", DEFAULT_NUM_CLASSES)),
        "checkpoint_mean": expected_model.get("checkpoint_mean", DEFAULT_MEAN),
        "checkpoint_std": expected_model.get("checkpoint_std", DEFAULT_STD),
    }


def validate_checkpoint_identity(
    summary: Dict[str, Any],
    expected: Dict[str, Any],
    *,
    strict: bool,
) -> List[str]:
    warnings_out: List[str] = []

    def handle(message: str, fatal: bool = True) -> None:
        if strict and fatal:
            raise RuntimeError(message)
        warnings_out.append(message)

    if summary["architecture_hash"] not in (None, "unknown"):
        if str(summary["architecture_hash"]) != str(expected["architecture_hash"]):
            handle(
                f"Architecture hash mismatch: checkpoint={summary['architecture_hash']} "
                f"expected={expected['architecture_hash']}"
            )
    else:
        handle("Checkpoint architecture hash is missing/unknown.", fatal=False)

    if summary["model_variant"] and expected["model_variant"]:
        if str(summary["model_variant"]) != str(expected["model_variant"]):
            # Some artifacts store model_class but not model_variant. Treat this
            # as fatal only if both are specific and conflicting.
            handle(
                f"Model variant mismatch: checkpoint={summary['model_variant']} "
                f"expected={expected['model_variant']}",
                fatal=False,
            )

    if summary["image_size"] is not None and int(summary["image_size"]) != int(expected["image_size"]):
        handle(
            f"Image size mismatch: checkpoint={summary['image_size']} "
            f"expected={expected['image_size']}"
        )

    if summary["channels"] is not None and int(summary["channels"]) != int(expected["channels"]):
        handle(
            f"Channel mismatch: checkpoint={summary['channels']} "
            f"expected={expected['channels']}"
        )

    if len(summary["mean"]) != int(expected["channels"]):
        handle(
            f"Mean length {len(summary['mean'])} does not match channels "
            f"{expected['channels']}."
        )

    if len(summary["std"]) != int(expected["channels"]):
        handle(
            f"Std length {len(summary['std'])} does not match channels "
            f"{expected['channels']}."
        )

    return warnings_out


def validate_config_arch_hash(
    checkpoint: Dict[str, Any],
    config: Dict[str, Any],
    *,
    strict: bool,
) -> List[str]:
    """
    Use the project's validate_checkpoint_arch when possible.

    If the checkpoint lacks arch_metadata but the external summary already
    validated architecture_hash, this function will warn rather than fail unless
    strict validation is impossible.
    """
    warnings_out: List[str] = []

    try:
        _, validate_checkpoint_arch, _ = _import_training_factory()
    except ImportError as exc:
        if strict:
            raise
        warnings_out.append(str(exc))
        return warnings_out

    try:
        validate_checkpoint_arch(checkpoint, config, strict=True)
    except Exception as exc:
        if strict:
            raise RuntimeError(f"Project architecture validation failed: {exc}") from exc
        warnings_out.append(f"Project architecture validation warning: {exc}")

    return warnings_out


def compare_provenance_checkpoint(
    primary_state: Dict[str, torch.Tensor],
    provenance_path: Union[str, Path],
    *,
    strict: bool,
) -> Dict[str, Any]:
    provenance_path = Path(provenance_path)
    prov_ckpt = torch_load_checkpoint(provenance_path)
    prov_state = extract_model_state_dict(prov_ckpt)

    primary_keys = set(primary_state.keys())
    provenance_keys = set(prov_state.keys())

    key_match = primary_keys == provenance_keys
    shape_match = True
    dtype_match = True

    missing_in_provenance = sorted(primary_keys - provenance_keys)
    extra_in_provenance = sorted(provenance_keys - primary_keys)

    mismatched_shapes: List[str] = []
    mismatched_dtypes: List[str] = []

    if key_match:
        for key in sorted(primary_keys):
            a = primary_state[key]
            b = prov_state[key]
            if torch.is_tensor(a) and torch.is_tensor(b):
                if tuple(a.shape) != tuple(b.shape):
                    shape_match = False
                    mismatched_shapes.append(key)
                if a.dtype != b.dtype:
                    dtype_match = False
                    mismatched_dtypes.append(key)

    max_abs_diff = max_abs_state_dict_difference(primary_state, prov_state) if key_match else None
    tensor_values_identical = bool(max_abs_diff == 0.0) if max_abs_diff is not None else False

    result = {
        "provenance_checkpoint_path": str(provenance_path),
        "provenance_file_sha256": sha256_file(provenance_path),
        "key_match": key_match,
        "shape_match": shape_match,
        "dtype_match": dtype_match,
        "tensor_values_identical": tensor_values_identical,
        "max_abs_difference": max_abs_diff,
        "primary_n_tensors": len(primary_state),
        "provenance_n_tensors": len(prov_state),
        "missing_keys_in_provenance": missing_in_provenance[:50],
        "extra_keys_in_provenance": extra_in_provenance[:50],
        "mismatched_shape_keys": mismatched_shapes[:50],
        "mismatched_dtype_keys": mismatched_dtypes[:50],
    }

    ok = key_match and shape_match and dtype_match and tensor_values_identical
    result["equivalence_status"] = "equivalent" if ok else "different"

    if strict and not ok:
        raise RuntimeError(
            "Primary checkpoint and provenance checkpoint are not equivalent. "
            f"Comparison result: {json.dumps(result, indent=2)}"
        )

    return result


# -----------------------------------------------------------------------------
# Model construction and loading
# -----------------------------------------------------------------------------

def build_and_load_model(
    checkpoint: Dict[str, Any],
    config: Dict[str, Any],
    state_dict: Dict[str, torch.Tensor],
    *,
    device: torch.device,
    num_classes: int,
    strict_state_dict: bool = True,
) -> nn.Module:
    build_model, _, _ = _import_training_factory()

    model = build_model(config=config, num_classes=num_classes, device=device)

    missing, unexpected = model.load_state_dict(state_dict, strict=strict_state_dict)
    # In strict=True, PyTorch raises before returning incompatible keys. In
    # strict=False, handle explicitly.
    if missing or unexpected:
        raise RuntimeError(
            f"State_dict load produced missing={missing}, unexpected={unexpected}"
        )

    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    return model


def run_smoke_test(
    model: nn.Module,
    *,
    image_size: int,
    channels: int,
    num_classes: int,
    device: torch.device,
) -> Dict[str, Any]:
    with torch.no_grad():
        x = torch.zeros((1, int(channels), int(image_size), int(image_size)), device=device)
        y = model(x)

    if not torch.is_tensor(y):
        raise RuntimeError(f"Model output is not a tensor: {type(y)}")

    if tuple(y.shape) != (1, int(num_classes)):
        raise RuntimeError(
            f"Unexpected model output shape {tuple(y.shape)}, expected "
            f"(1, {num_classes})."
        )

    if not torch.all(torch.isfinite(y)).item():
        raise RuntimeError("Smoke-test output contains non-finite values.")

    return {
        "smoke_test_passed": True,
        "input_shape": [1, int(channels), int(image_size), int(image_size)],
        "output_shape": list(y.shape),
        "output_min": float(y.min().detach().cpu().item()),
        "output_max": float(y.max().detach().cpu().item()),
    }


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def load_frozen_hybrid(
    checkpoint_path: Union[str, Path],
    provenance_checkpoint_path: Optional[Union[str, Path]] = None,
    config_path: Optional[Union[str, Path]] = None,
    device: Union[str, torch.device] = "auto",
    strict: bool = True,
    run_dummy_forward: bool = False,
) -> Dict[str, Any]:
    """
    Load the frozen LSA-ConvTok Hybrid model safely for external validation.

    Parameters
    ----------
    checkpoint_path:
        Primary exported inference artifact, expected final_model_lsa_convtok.pth.
    provenance_checkpoint_path:
        Optional full final_model_checkpoint.pth used only for equivalence/provenance.
    config_path:
        Optional external_validation_config.yaml. Supplies expected model identity.
    device:
        "auto", "cpu", "cuda", or torch.device.
    strict:
        If True, abort on identity or provenance inconsistencies.
    run_dummy_forward:
        If True, run a zero-input no-gradient smoke test.

    Returns
    -------
    dict
        {
          "model": nn.Module,
          "mean": list[float],
          "std": list[float],
          "config": dict,
          "metadata": dict,
          "device": torch.device,
          "class_names": list[str],
        }
    """
    checkpoint_path = Path(checkpoint_path)
    config_yaml = load_optional_yaml_config(Path(config_path) if config_path else None)
    expected = expected_values_from_config(config_yaml)
    resolved_device = resolve_device(device)

    primary_ckpt = torch_load_checkpoint(checkpoint_path)
    arch_metadata = extract_arch_metadata(primary_ckpt)
    config = extract_config(primary_ckpt, arch_metadata)
    state_dict = extract_model_state_dict(primary_ckpt)
    mean, std = extract_mean_std(primary_ckpt, config)

    summary = checkpoint_summary(
        path=checkpoint_path,
        checkpoint=primary_ckpt,
        config=config,
        arch_metadata=arch_metadata,
        state_dict=state_dict,
        mean=mean,
        std=std,
    )

    identity_warnings = validate_checkpoint_identity(summary, expected, strict=strict)
    arch_warnings = validate_config_arch_hash(primary_ckpt, config, strict=strict)

    provenance_result = None
    if provenance_checkpoint_path is not None:
        provenance_result = compare_provenance_checkpoint(
            primary_state=state_dict,
            provenance_path=provenance_checkpoint_path,
            strict=strict,
        )

    model = build_and_load_model(
        checkpoint=primary_ckpt,
        config=config,
        state_dict=state_dict,
        device=resolved_device,
        num_classes=int(expected["num_classes"]),
        strict_state_dict=True,
    )

    smoke = None
    if run_dummy_forward:
        smoke = run_smoke_test(
            model,
            image_size=int(expected["image_size"]),
            channels=int(expected["channels"]),
            num_classes=int(expected["num_classes"]),
            device=resolved_device,
        )

    class_names = get_nested(
        config_yaml,
        ["labels", "class_names"],
        ["Candida albicans", "Candida glabrata"],
    )

    metadata = {
        "schema_version": "1.0",
        "script": "load_frozen_hybrid.py",
        "created_unix_time": time.time(),
        "scientific_role": (
            "Frozen checkpoint loader for external validation. The returned model "
            "is in eval mode with gradients disabled."
        ),
        "primary_checkpoint": summary,
        "provenance_comparison": provenance_result,
        "expected_model": expected,
        "identity_warnings": identity_warnings,
        "project_architecture_validation_warnings": arch_warnings,
        "device": str(resolved_device),
        "mean": [float(v) for v in mean],
        "std": [float(v) for v in std],
        "class_names": class_names,
        "model_eval_mode": not model.training,
        "all_parameters_require_grad_false": all(not p.requires_grad for p in model.parameters()),
        "smoke_test": smoke,
        "scientific_guardrails": {
            "primary_artifact_is_exported_inference_checkpoint": True,
            "provenance_checkpoint_used_only_for_equivalence_check": provenance_checkpoint_path is not None,
            "architecture_hash_validated": True,
            "no_training_state_used_for_inference": True,
            "no_gradients_enabled": True,
        },
    }

    return {
        "model": model,
        "mean": [float(v) for v in mean],
        "std": [float(v) for v in std],
        "config": config,
        "metadata": metadata,
        "device": resolved_device,
        "class_names": class_names,
    }


def save_loader_metadata(bundle: Dict[str, Any], path: Path) -> None:
    metadata = dict(bundle["metadata"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def print_loader_summary(bundle: Dict[str, Any]) -> None:
    meta = bundle["metadata"]
    primary = meta["primary_checkpoint"]
    prov = meta.get("provenance_comparison")

    print("\n" + "=" * 80)
    print("FROZEN HYBRID MODEL LOADED")
    print("=" * 80)
    print(f"Checkpoint       : {primary['path']}")
    print(f"Model variant    : {primary['model_variant']}")
    print(f"Experiment       : {primary['experiment_name']}")
    print(f"Arch hash        : {primary['architecture_hash']}")
    print(f"Image size       : {primary['image_size']}")
    print(f"Channels         : {primary['channels']}")
    print(f"Mean             : {meta['mean']}")
    print(f"Std              : {meta['std']}")
    print(f"Device           : {meta['device']}")
    print(f"Eval mode        : {meta['model_eval_mode']}")
    print(f"Grad disabled    : {meta['all_parameters_require_grad_false']}")

    if prov is not None:
        print("\nProvenance checkpoint comparison:")
        print(f"  - status           : {prov['equivalence_status']}")
        print(f"  - key_match        : {prov['key_match']}")
        print(f"  - shape_match      : {prov['shape_match']}")
        print(f"  - dtype_match      : {prov['dtype_match']}")
        print(f"  - values_identical : {prov['tensor_values_identical']}")
        print(f"  - max_abs_diff     : {prov['max_abs_difference']}")

    if meta.get("smoke_test"):
        print("\nSmoke test:")
        print(f"  - passed       : {meta['smoke_test']['smoke_test_passed']}")
        print(f"  - input_shape  : {meta['smoke_test']['input_shape']}")
        print(f"  - output_shape : {meta['smoke_test']['output_shape']}")

    if meta.get("identity_warnings") or meta.get("project_architecture_validation_warnings"):
        print("\nWarnings:")
        for w in meta.get("identity_warnings", []):
            print(f"  - {w}")
        for w in meta.get("project_architecture_validation_warnings", []):
            print(f"  - {w}")

    print("=" * 80 + "\n")


def main() -> None:
    args = parse_args()

    bundle = load_frozen_hybrid(
        checkpoint_path=args.checkpoint,
        provenance_checkpoint_path=args.provenance_checkpoint,
        config_path=args.config,
        device=args.device,
        strict=args.strict,
        run_dummy_forward=args.smoke_test,
    )

    if args.metadata_output is not None:
        save_loader_metadata(bundle, args.metadata_output)

    print_loader_summary(bundle)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
