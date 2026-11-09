#!/usr/bin/env python3
"""
load_frozen_hybrid.py

Safe loader for the frozen LSA-ConvTok Hybrid model used in external validation.

Scientific role
---------------
External validation must evaluate the exact frozen Hybrid model selected after
model development. This module centralises checkpoint loading so that
downstream evaluation scripts cannot:
    - instantiate the wrong architecture,
    - load mismatched weights,
    - report an architecture check that did not actually happen,
    - fall back silently to hard-coded normalisation statistics,
    - leave gradients enabled during external inference.

Architecture identity (F-01)
----------------------------
``training_logic.validate_checkpoint_arch`` returns True with only a warning
when a checkpoint has no ``arch_metadata``. This loader therefore establishes
identity itself, in priority order:

    1. stored_arch_metadata             checkpoint arch_metadata.arch_hash
    2. recomputed_from_embedded_config  training_logic.compute_arch_hash(config)
    3. provenance_arch_metadata         provenance checkpoint arch_metadata.arch_hash

Every available hash must equal the expected hash; any mismatch is fatal in
strict mode. ``architecture_hash_validated`` is the computed outcome, never a
constant. Strict ``load_state_dict`` additionally verifies keys and shapes.

Normalisation statistics (F-03)
-------------------------------
Mean/std are located by explicit key search and the source key is recorded.
A fallback to the built-in defaults is an error in strict mode. Values are
compared with the config's expected values and, when supplied, with the
provenance checkpoint.

Primary checkpoint policy
-------------------------
    final_model_lsa_convtok.pth  primary frozen inference artifact
    final_model_checkpoint.pth   optional training-resume provenance artifact

Author: Rodrigo Sá
Date: 2026
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

LOADER_SCHEMA_VERSION = "2.0"
DEFAULT_EXPECTED_ARCH_HASH = "95ace50634b621d7"
DEFAULT_MODEL_VARIANT = "SimpleViT_LSA_ConvTok"
DEFAULT_EXPERIMENT_NAME = "lsa_convtok"
DEFAULT_IMAGE_SIZE = 384
DEFAULT_CHANNELS = 1
DEFAULT_NUM_CLASSES = 2
DEFAULT_MEAN = [0.23691008985042572]
DEFAULT_STD = [0.007409718818962574]
MEAN_STD_REL_TOLERANCE = 1e-6

MEAN_KEY_PATHS: Tuple[Tuple[str, ...], ...] = (("mean",), ("metadata", "mean"), ("config", "mean"))
STD_KEY_PATHS: Tuple[Tuple[str, ...], ...] = (("std",), ("metadata", "std"), ("config", "std"))


# -----------------------------------------------------------------------------
# training_logic import (F-23)
# -----------------------------------------------------------------------------

def _training_logic_candidates() -> List[Path]:
    """Import roots in priority order: explicit env dir first, cwd/source last."""
    script_dir = Path(__file__).resolve().parent
    env_dir = os.environ.get("EXTERNAL_VALIDATION_SOURCE_CODE_DIR")
    raw = [
        Path(env_dir) if env_dir else None,
        script_dir.parent / "source",
        script_dir,
        script_dir.parent,
        Path.cwd() / "source",
        Path.cwd(),
    ]
    out: List[Path] = []
    for c in raw:
        if c is None:
            continue
        c = c.resolve()
        if c.exists() and c not in out:
            out.append(c)
    return out


def _import_training_factory():
    """
    Import ``build_model``, ``validate_checkpoint_arch`` and ``compute_arch_hash``
    from training_logic.py.

    The candidate roots are placed at the front of ``sys.path`` in priority
    order (moving them if already present), so EXTERNAL_VALIDATION_SOURCE_CODE_DIR
    always wins over a stray training_logic.py in the working directory.
    """
    import importlib

    module = sys.modules.get("training_logic")
    candidates = _training_logic_candidates()
    if module is None:
        ordered = [str(c) for c in candidates]
        sys.path[:] = ordered + [p for p in sys.path if p not in ordered]
        try:
            module = importlib.import_module("training_logic")
        except Exception as exc:
            raise ImportError(
                "Could not import training_logic.py. Expected it in validation/source or in "
                f"EXTERNAL_VALIDATION_SOURCE_CODE_DIR. Checked import roots: {ordered}."
            ) from exc
    try:
        build_model = getattr(module, "build_model")
        validate_checkpoint_arch = getattr(module, "validate_checkpoint_arch")
        compute_arch_hash = getattr(module, "compute_arch_hash")
    except AttributeError as exc:
        raise ImportError(f"training_logic at {getattr(module, '__file__', '?')} lacks a required function.") from exc
    return build_model, validate_checkpoint_arch, compute_arch_hash


def training_logic_file() -> Optional[str]:
    module = sys.modules.get("training_logic")
    return str(Path(module.__file__).resolve()) if module is not None and getattr(module, "__file__", None) else None


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Safely load the frozen final LSA-ConvTok Hybrid checkpoint for external validation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="Primary inference checkpoint, expected final_model_lsa_convtok.pth.")
    parser.add_argument("--provenance-checkpoint", type=Path, default=None,
                        help="Optional full training-resume checkpoint used only for provenance checks.")
    parser.add_argument("--config", type=Path, default=None, help="Optional external_validation_config.yaml.")
    parser.add_argument("--device", type=str, default="auto", help="'auto', 'cpu', 'cuda', or a torch device.")
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True,
                        help="Abort on architecture/config/provenance inconsistencies.")
    parser.add_argument("--smoke-test", action="store_true", help="Run a no-gradient dummy forward pass.")
    parser.add_argument("--metadata-output", type=Path, default=None, help="Optional loader metadata JSON path.")
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------

def load_optional_yaml_config(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config YAML not found: {path}")
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to read the config. Install with: pip install pyyaml") from exc
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
    with Path(path).open("rb") as f:
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


def normalize_list(value: Any) -> List[float]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, np.ndarray):
        value = value.reshape(-1).tolist()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [float(value)]
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    raise ValueError(f"Cannot convert value to list of floats: {value!r}")


def hash_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> str:
    keys = sorted(str(k) for k in state_dict.keys())
    return hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()


def state_dict_content_fingerprint(state_dict: Dict[str, torch.Tensor]) -> str:
    """Deterministic fingerprint over keys, shapes, dtypes and tensor bytes (provenance only)."""
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


def max_abs_state_dict_difference(state_a: Dict[str, torch.Tensor], state_b: Dict[str, torch.Tensor]) -> Optional[float]:
    if set(state_a.keys()) != set(state_b.keys()):
        return None
    max_diff = 0.0
    for key in sorted(state_a.keys()):
        a, b = state_a[key], state_b[key]
        if not torch.is_tensor(a) or not torch.is_tensor(b):
            continue
        if a.shape != b.shape:
            return None
        max_diff = max(max_diff, float((a.detach().cpu().float() - b.detach().cpu().float()).abs().max().item()))
    return max_diff


def _lookup(data: Dict[str, Any], path: Sequence[str]) -> Any:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


# -----------------------------------------------------------------------------
# Checkpoint parsing
# -----------------------------------------------------------------------------

def torch_load_checkpoint(path: Union[str, Path]) -> Tuple[Dict[str, Any], str]:
    """
    Load a checkpoint dict. ``weights_only=True`` is attempted first; only when
    the file requires arbitrary unpickling is ``weights_only=False`` used, and
    that is recorded as ``unsafe_pickle_trusted_artifact`` (F-22).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
        mode = "weights_only"
    except Exception:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        mode = "unsafe_pickle_trusted_artifact"
    if not isinstance(ckpt, dict):
        raise ValueError(f"Checkpoint must be a dictionary, got {type(ckpt)} from {path}")
    return ckpt, mode


def extract_arch_metadata(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    for path in (("arch_metadata",), ("metadata", "arch_metadata"), ("architecture", "arch_metadata")):
        value = _lookup(checkpoint, path)
        if isinstance(value, dict):
            return value
    return {}


def extract_config(checkpoint: Dict[str, Any], arch_metadata: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """
    Return (model-building config, config_source).

    config_source is ``embedded_config``, ``embedded_metadata_config`` or
    ``reconstructed_from_arch_metadata``. Reconstruction is a robustness
    fallback; it is recorded so reports can state it.
    """
    config = checkpoint.get("config")
    if isinstance(config, dict) and config:
        return dict(config), "embedded_config"
    metadata = checkpoint.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("config"), dict) and metadata["config"]:
        return dict(metadata["config"]), "embedded_metadata_config"
    if not arch_metadata:
        raise ValueError(
            "Checkpoint lacks both full config and architecture metadata. "
            "Cannot safely reconstruct the frozen Hybrid model."
        )
    bit_depth = checkpoint.get("bit_depth")
    if bit_depth is None and isinstance(metadata, dict):
        bit_depth = metadata.get("bit_depth")
    reconstructed = {
        "experiment_name": arch_metadata.get("experiment_name", DEFAULT_EXPERIMENT_NAME),
        "image_size": int(arch_metadata.get("image_size", DEFAULT_IMAGE_SIZE)),
        "channels": int(arch_metadata.get("channels", DEFAULT_CHANNELS)),
        "bit_depth": int(bit_depth) if bit_depth is not None else 12,
        "model_config": {
            "dim": int(arch_metadata.get("dim", 256)),
            "depth": int(arch_metadata.get("depth", 8)),
            "heads": int(arch_metadata.get("heads", 6)),
            "mlp_dim": int(arch_metadata.get("mlp_dim", 512)),
            "dim_head": int(arch_metadata.get("dim_head", 64)),
            "pe_temperature": int(arch_metadata.get("pe_temperature", 10000)),
        },
        "convtok_blocks": arch_metadata.get("convtok_blocks"),
        "convtok_hidden_channels": int(arch_metadata.get("convtok_hidden_channels", 64)),
        "convtok_match_baseline_tokens": True,
        "convtok_expected_hw": arch_metadata.get("convtok_expected_hw", [24, 24]),
        "convtok_expected_tokens": int(arch_metadata.get("convtok_expected_tokens", 576)),
        "convtok_post_ln": bool(arch_metadata.get("convtok_post_ln", True)),
        "convtok_conv_bias": bool(arch_metadata.get("convtok_conv_bias", False)),
        "init_policy": arch_metadata.get("init_policy", "default"),
    }
    return reconstructed, "reconstructed_from_arch_metadata"


def extract_model_state_dict(checkpoint: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    for key in ("model_state_dict", "state_dict"):
        state = checkpoint.get(key)
        if isinstance(state, dict):
            return state
    raise ValueError("Checkpoint has no model_state_dict/state_dict.")


def resolve_normalization_stats(checkpoint: Dict[str, Any]) -> Tuple[Optional[List[float]], Optional[List[float]], str]:
    """
    Locate checkpoint mean/std by explicit key search (``is None`` tests, no
    truthiness). Returns (mean, std, source); source is ``"<path>"`` of the
    first key holding both values, or ``"absent"``.
    """
    for m_path, s_path in zip(MEAN_KEY_PATHS, STD_KEY_PATHS):
        m, s = _lookup(checkpoint, m_path), _lookup(checkpoint, s_path)
        if m is not None and s is not None:
            return normalize_list(m), normalize_list(s), ".".join(m_path) + "/" + ".".join(s_path)
    return None, None, "absent"


def resolve_mean_std(
    checkpoint: Dict[str, Any],
    *,
    expected_mean: Optional[Sequence[float]],
    expected_std: Optional[Sequence[float]],
    strict: bool,
) -> Tuple[List[float], List[float], Dict[str, Any]]:
    """Resolve mean/std with source tracking and consistency checks (F-03)."""
    mean, std, source = resolve_normalization_stats(checkpoint)
    info: Dict[str, Any] = {"mean_std_source": source}
    if mean is None or std is None:
        if strict:
            raise RuntimeError(
                "Checkpoint mean/std not found under keys mean/std, metadata.mean/std or config.mean/std. "
                "Refusing mean/std fallback to built-in defaults in strict mode."
            )
        warnings.warn("Checkpoint mean/std absent; using built-in defaults (non-strict run).")
        mean, std = list(DEFAULT_MEAN), list(DEFAULT_STD)
        info["mean_std_source"] = "built_in_default"
    if len(mean) != len(std):
        raise ValueError(f"Mean/std length mismatch: mean={mean}, std={std}")
    if any((not math.isfinite(s)) or s <= 0 for s in std):
        raise ValueError(f"All std values must be finite and positive, got {std}")

    matches = None
    if expected_mean is not None and expected_std is not None:
        exp_m, exp_s = normalize_list(expected_mean), normalize_list(expected_std)
        matches = len(exp_m) == len(mean) and len(exp_s) == len(std) and all(
            abs(a - b) <= MEAN_STD_REL_TOLERANCE * max(abs(b), 1e-12) for a, b in zip(mean + std, exp_m + exp_s)
        )
        if not matches:
            msg = f"Checkpoint mean/std {mean}/{std} differ from config expected {exp_m}/{exp_s}."
            if strict:
                raise RuntimeError(msg)
            warnings.warn(msg)
    info["mean_std_matches_config"] = matches
    return [float(v) for v in mean], [float(v) for v in std], info


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
    metadata = checkpoint.get("metadata") if isinstance(checkpoint.get("metadata"), dict) else {}

    def first(*values: Any) -> Any:
        for v in values:
            if v is not None:
                return v
        return None

    stored_hash = first(arch_metadata.get("arch_hash"), checkpoint.get("architecture_hash"),
                        metadata.get("architecture_hash"), metadata.get("arch_hash"))
    image_size = first(config.get("image_size"), arch_metadata.get("image_size"),
                       checkpoint.get("image_size"), metadata.get("image_size"))
    channels = first(config.get("channels"), arch_metadata.get("channels"),
                     checkpoint.get("channels"), metadata.get("channels"))
    return {
        "path": str(path),
        "file_sha256": sha256_file(path),
        "model_variant": first(checkpoint.get("model_variant"), metadata.get("model_variant"),
                               arch_metadata.get("model_class")),
        "experiment_name": first(config.get("experiment_name"), arch_metadata.get("experiment_name")),
        "stored_architecture_hash": stored_hash,
        "image_size": int(image_size) if image_size is not None else None,
        "channels": int(channels) if channels is not None else None,
        "checkpoint_training_bit_depth": first(config.get("bit_depth"), checkpoint.get("bit_depth"), metadata.get("bit_depth")),
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
        "checkpoint_mean": expected_model.get("checkpoint_mean"),
        "checkpoint_std": expected_model.get("checkpoint_std"),
    }


def validate_checkpoint_identity(summary: Dict[str, Any], expected: Dict[str, Any], *, strict: bool) -> List[str]:
    """Non-hash identity checks: variant, image size, channels, mean/std length."""
    warnings_out: List[str] = []

    def handle(message: str, fatal: bool = True) -> None:
        if strict and fatal:
            raise RuntimeError(message)
        warnings_out.append(message)

    if summary["model_variant"] and expected["model_variant"] and str(summary["model_variant"]) != str(expected["model_variant"]):
        handle(f"Model variant mismatch: checkpoint={summary['model_variant']} expected={expected['model_variant']}",
               fatal=False)
    if summary["image_size"] is not None and int(summary["image_size"]) != int(expected["image_size"]):
        handle(f"Image size mismatch: checkpoint={summary['image_size']} expected={expected['image_size']}")
    if summary["channels"] is not None and int(summary["channels"]) != int(expected["channels"]):
        handle(f"Channel mismatch: checkpoint={summary['channels']} expected={expected['channels']}")
    if len(summary["mean"]) != int(expected["channels"]) or len(summary["std"]) != int(expected["channels"]):
        handle(f"Mean/std length does not match channels {expected['channels']}.")
    return warnings_out


def verify_architecture_identity(
    *,
    checkpoint: Dict[str, Any],
    config: Dict[str, Any],
    config_source: str,
    arch_metadata: Dict[str, Any],
    provenance_checkpoint: Optional[Dict[str, Any]],
    expected_hash: str,
    strict: bool,
) -> Dict[str, Any]:
    """
    Establish architecture identity independently of training_logic's lenient
    validator (F-01). Returns a record; raises in strict mode on any mismatch
    or when no hash could be obtained.
    """
    _, validate_checkpoint_arch, compute_arch_hash = _import_training_factory()

    candidates: List[Tuple[str, Optional[str]]] = []
    stored = arch_metadata.get("arch_hash") if arch_metadata else None
    candidates.append(("stored_arch_metadata", str(stored) if stored else None))

    recomputed: Optional[str] = None
    recompute_error: Optional[str] = None
    try:
        recomputed = str(compute_arch_hash(config))
    except Exception as exc:  # recorded; becomes fatal only if nothing else validates
        recompute_error = f"{type(exc).__name__}: {exc}"
    candidates.append(("recomputed_from_embedded_config", recomputed))

    prov_hash: Optional[str] = None
    if provenance_checkpoint is not None:
        prov_meta = extract_arch_metadata(provenance_checkpoint)
        prov_hash = str(prov_meta["arch_hash"]) if prov_meta.get("arch_hash") else None
    candidates.append(("provenance_arch_metadata", prov_hash))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            validate_checkpoint_arch(checkpoint, config, strict=True)
            training_logic_validator = "passed_or_skipped"
        except Exception as exc:
            training_logic_validator = f"failed: {exc}"
    validator_warnings = [str(w.message) for w in caught]

    mismatches = [(m, h) for m, h in candidates if h is not None and h != str(expected_hash)]
    matches = [m for m, h in candidates if h is not None and h == str(expected_hash)]
    method = matches[0] if matches else "not_validated"
    validated = bool(matches) and not mismatches and not training_logic_validator.startswith("failed")

    record = {
        "architecture_hash_expected": str(expected_hash),
        "architecture_hash_observed": recomputed,
        "architecture_hash_stored": stored,
        "architecture_hash_provenance": prov_hash,
        "architecture_validation_method": method,
        "architecture_hash_validated": validated,
        "config_source": config_source,
        "recompute_error": recompute_error,
        "training_logic_validator": training_logic_validator,
        "training_logic_validator_warnings": validator_warnings,
        "candidate_hashes": {m: h for m, h in candidates},
    }
    if mismatches or not matches or training_logic_validator.startswith("failed"):
        msg = (
            "Architecture hash mismatch or architecture not validated: "
            f"expected={expected_hash}, candidates={record['candidate_hashes']}, "
            f"recompute_error={recompute_error}, training_logic_validator={training_logic_validator}"
        )
        if strict:
            raise RuntimeError(msg)
        warnings.warn(msg)
    return record


def compare_provenance_checkpoint(
    primary_state: Dict[str, torch.Tensor],
    primary_mean: Sequence[float],
    primary_std: Sequence[float],
    provenance_ckpt: Dict[str, Any],
    provenance_path: Union[str, Path],
    *,
    strict: bool,
) -> Dict[str, Any]:
    """Weight and normalisation equivalence between primary and provenance checkpoints."""
    provenance_path = Path(provenance_path)
    prov_state = extract_model_state_dict(provenance_ckpt)
    primary_keys, provenance_keys = set(primary_state.keys()), set(prov_state.keys())
    key_match = primary_keys == provenance_keys
    mismatched_shapes: List[str] = []
    mismatched_dtypes: List[str] = []
    if key_match:
        for key in sorted(primary_keys):
            a, b = primary_state[key], prov_state[key]
            if torch.is_tensor(a) and torch.is_tensor(b):
                if tuple(a.shape) != tuple(b.shape):
                    mismatched_shapes.append(key)
                if a.dtype != b.dtype:
                    mismatched_dtypes.append(key)
    max_abs_diff = max_abs_state_dict_difference(primary_state, prov_state) if key_match else None
    values_identical = bool(max_abs_diff == 0.0) if max_abs_diff is not None else False

    prov_mean, prov_std, prov_source = resolve_normalization_stats(provenance_ckpt)
    mean_std_matches = None
    if prov_mean is not None and prov_std is not None:
        mean_std_matches = all(
            abs(a - b) <= MEAN_STD_REL_TOLERANCE * max(abs(b), 1e-12)
            for a, b in zip(list(primary_mean) + list(primary_std), prov_mean + prov_std)
        ) and len(prov_mean) == len(primary_mean)

    result = {
        "provenance_checkpoint_path": str(provenance_path),
        "provenance_file_sha256": sha256_file(provenance_path),
        "key_match": key_match,
        "shape_match": not mismatched_shapes,
        "dtype_match": not mismatched_dtypes,
        "tensor_values_identical": values_identical,
        "max_abs_difference": max_abs_diff,
        "primary_n_tensors": len(primary_state),
        "provenance_n_tensors": len(prov_state),
        "missing_keys_in_provenance": sorted(primary_keys - provenance_keys)[:50],
        "extra_keys_in_provenance": sorted(provenance_keys - primary_keys)[:50],
        "mismatched_shape_keys": mismatched_shapes[:50],
        "mismatched_dtype_keys": mismatched_dtypes[:50],
        "provenance_mean_std_source": prov_source,
        "mean_std_matches_provenance": mean_std_matches,
    }
    ok = key_match and not mismatched_shapes and not mismatched_dtypes and values_identical and mean_std_matches is not False
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
    config: Dict[str, Any],
    state_dict: Dict[str, torch.Tensor],
    *,
    device: torch.device,
    num_classes: int,
) -> nn.Module:
    build_model, _, _ = _import_training_factory()
    model = build_model(config=config, num_classes=num_classes, device=device)
    model.load_state_dict(state_dict, strict=True)  # raises on missing/unexpected keys or shapes
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def run_smoke_test(model: nn.Module, *, image_size: int, channels: int, num_classes: int,
                   device: torch.device) -> Dict[str, Any]:
    with torch.no_grad():
        x = torch.zeros((1, int(channels), int(image_size), int(image_size)), device=device)
        y = model(x)
    if not torch.is_tensor(y) or tuple(y.shape) != (1, int(num_classes)):
        raise RuntimeError(f"Unexpected smoke-test output {getattr(y, 'shape', type(y))}.")
    if not torch.all(torch.isfinite(y)).item():
        raise RuntimeError("Smoke-test output contains non-finite values.")
    return {
        "smoke_test_passed": True,
        "input_shape": [1, int(channels), int(image_size), int(image_size)],
        "output_shape": list(y.shape),
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
    Load the frozen model for external validation.

    Returns {"model", "mean", "std", "config", "metadata", "device", "class_names"}.
    The model is in eval mode with gradients disabled.
    """
    checkpoint_path = Path(checkpoint_path)
    config_yaml = load_optional_yaml_config(Path(config_path) if config_path else None)
    expected = expected_values_from_config(config_yaml)
    resolved_device = resolve_device(device)

    primary_ckpt, primary_load_mode = torch_load_checkpoint(checkpoint_path)
    arch_metadata = extract_arch_metadata(primary_ckpt)
    config, config_source = extract_config(primary_ckpt, arch_metadata)
    state_dict = extract_model_state_dict(primary_ckpt)
    mean, std, mean_std_info = resolve_mean_std(
        primary_ckpt,
        expected_mean=expected["checkpoint_mean"],
        expected_std=expected["checkpoint_std"],
        strict=strict,
    )

    summary = checkpoint_summary(checkpoint_path, primary_ckpt, config, arch_metadata, state_dict, mean, std)
    identity_warnings = validate_checkpoint_identity(summary, expected, strict=strict)

    provenance_ckpt: Optional[Dict[str, Any]] = None
    provenance_load_mode: Optional[str] = None
    if provenance_checkpoint_path is not None:
        provenance_ckpt, provenance_load_mode = torch_load_checkpoint(provenance_checkpoint_path)

    architecture = verify_architecture_identity(
        checkpoint=primary_ckpt,
        config=config,
        config_source=config_source,
        arch_metadata=arch_metadata,
        provenance_checkpoint=provenance_ckpt,
        expected_hash=str(expected["architecture_hash"]),
        strict=strict,
    )

    provenance_result = None
    if provenance_ckpt is not None:
        provenance_result = compare_provenance_checkpoint(
            state_dict, mean, std, provenance_ckpt, provenance_checkpoint_path, strict=strict
        )

    model = build_and_load_model(config, state_dict, device=resolved_device, num_classes=int(expected["num_classes"]))

    smoke = None
    if run_dummy_forward:
        smoke = run_smoke_test(model, image_size=int(expected["image_size"]), channels=int(expected["channels"]),
                               num_classes=int(expected["num_classes"]), device=resolved_device)

    class_names = get_nested(config_yaml, ["labels", "class_names"], ["Candida albicans", "Candida glabrata"])

    metadata = {
        "schema_version": LOADER_SCHEMA_VERSION,
        "script": "load_frozen_hybrid.py",
        "created_unix_time": time.time(),
        "scientific_role": "Frozen checkpoint loader for external validation (eval mode, gradients disabled).",
        "primary_checkpoint": summary,
        "checkpoint_load_mode": primary_load_mode,
        "provenance_checkpoint_load_mode": provenance_load_mode,
        "provenance_comparison": provenance_result,
        "expected_model": expected,
        "architecture_identity": architecture,
        "identity_warnings": identity_warnings,
        **mean_std_info,
        "mean_std_matches_provenance": (provenance_result or {}).get("mean_std_matches_provenance"),
        "training_logic_file": training_logic_file(),
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
            "architecture_hash_validated": bool(architecture["architecture_hash_validated"]),
            "architecture_validation_method": architecture["architecture_validation_method"],
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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(dict(bundle["metadata"]), f, indent=2, ensure_ascii=False)


def print_loader_summary(bundle: Dict[str, Any]) -> None:
    meta = bundle["metadata"]
    arch = meta["architecture_identity"]
    print("\n" + "=" * 80)
    print("FROZEN HYBRID MODEL LOADED")
    print("=" * 80)
    print(f"Checkpoint        : {meta['primary_checkpoint']['path']}")
    print(f"Load mode         : {meta['checkpoint_load_mode']}")
    print(f"Arch hash expected: {arch['architecture_hash_expected']}")
    print(f"Arch validated    : {arch['architecture_hash_validated']} ({arch['architecture_validation_method']})")
    print(f"Mean/std          : {meta['mean']} / {meta['std']} (source {meta['mean_std_source']})")
    print(f"training_logic    : {meta['training_logic_file']}")
    print(f"Device            : {meta['device']}")
    if meta.get("provenance_comparison"):
        print(f"Provenance        : {meta['provenance_comparison']['equivalence_status']}")
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
