#!/usr/bin/env python3
"""
pipeline_common.py

Small shared helpers for the external-validation scripts: YAML/JSON loading,
nested lookups, explicit CLI > config > default precedence, hashing, and
cohort-count resolution from the metadata JSON.

Precedence rule (F-13)
----------------------
Every tunable setting is resolved as:

    explicit CLI value (argparse default is None)  >  config value  >  built-in default

and the source is recorded so the run metadata states where each value came
from.

Author: Rodrigo Sá
Date: 2026
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


def load_optional_yaml(path: Optional[Path]) -> Dict[str, Any]:
    """Load a YAML mapping. Missing PyYAML is fatal: silent config loss is not allowed."""
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config YAML not found: {path}")
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to read the external-validation config. Install with: pip install pyyaml"
        ) from exc
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def read_json(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return data


def write_json(data: Mapping[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=True)


def get_nested(data: Mapping[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def resolve_setting(
    name: str,
    cli_value: Any,
    config: Mapping[str, Any],
    config_keys: Sequence[str],
    default: Any,
    resolution_log: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Any:
    """Resolve CLI > config > default and optionally log the source."""
    if cli_value is not None:
        value, source = cli_value, "cli"
    else:
        cfg_value = get_nested(config, config_keys, None)
        if cfg_value is not None:
            value, source = cfg_value, "config:" + ".".join(config_keys)
        else:
            value, source = default, "built_in_default"
    if resolution_log is not None:
        resolution_log[name] = {"value": value, "resolved_from": source}
    return value


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def expected_cohort_counts(
    metadata: Optional[Mapping[str, Any]],
    config: Optional[Mapping[str, Any]],
) -> Tuple[Optional[int], Dict[str, int], str]:
    """
    Expected slide counts: metadata ``expected_slide_counts`` first, then config
    ``expected_external_cohort``. Returns (total, per_species, source).
    """
    if metadata:
        counts = metadata.get("expected_slide_counts")
        labels = metadata.get("label_mapping", {}) or {}
        if isinstance(counts, Mapping) and counts.get("total_slides") is not None:
            per_species = {k: int(v) for k, v in counts.items() if k in labels}
            return int(counts["total_slides"]), per_species, "metadata.expected_slide_counts"
    if config:
        cohort = config.get("expected_external_cohort", {})
        if isinstance(cohort, Mapping) and cohort.get("total_slides") is not None:
            per_species = {k: int(v) for k, v in (cohort.get("slides_per_species", {}) or {}).items()}
            return int(cohort["total_slides"]), per_species, "config.expected_external_cohort"
    return None, {}, "not_specified"


def warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)
