#!/usr/bin/env python3
"""
external_manifest.py

Create the authoritative image-level manifest for external validation of the
frozen LSA-ConvTok Hybrid model under microscope-domain shift.

Scientific role
---------------
The external new-microscope cohort is an independent acquisition-domain
validation set. It must not be split into train/test subsets, used for
fine-tuning, used for threshold selection, or used for preprocessing/model
selection. This script fixes the biological/acquisition truth table before any
TIFF audit, preprocessing, inference, or statistical analysis is performed.

The script:
    1. Reads a raw external dataset supplied as either a ZIP archive or an
       extracted directory.
    2. If a ZIP is supplied, extracts it deterministically to an extraction
       directory so downstream scripts can access real filesystem paths.
    3. Parses image filenames of the form:
           20250813_alb_A_1.tif
       into acquisition_date, species_code, slide_letter, image_index, and
       slide_id = date_speciesCode_slideLetter.
    4. Assigns labels and C. albicans hyphae/no-hyphae status from an external
       metadata JSON file, never from model predictions.
    5. Writes an image-level CSV manifest and a JSON validation summary.
    6. Fails loudly in strict mode if the observed slides do not match the
       pre-specified metadata and expected cohort structure.

Inputs
------
    - fungi.zip or extracted dataset directory
    - external_validation_metadata.json
    - optional external_validation_config.yaml

Outputs
-------
    - external_manifest.csv
    - external_manifest_summary.json
    - optional extracted raw dataset directory if input is a ZIP

Example
-------
    python external_manifest.py \
        --input fungi.zip \
        --metadata configs/external_validation_metadata.json \
        --output outputs_external/manifest/external_manifest.csv

Author: Rodrigo Sá / generated implementation
Date: 2026
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


SUPPORTED_TIFF_EXTENSIONS = {".tif", ".tiff", ".TIF", ".TIFF"}
DEFAULT_SPECIES_CODE_MAP = {
    "alb": "Candida albicans",
    "gla": "Candida glabrata",
}
DEFAULT_HYPHAE_STATUS_GLabrata = "not_applicable"
VALID_HYPHAE_STATUS = {"hyphae", "no_hyphae", "not_applicable"}

MANIFEST_COLUMNS = [
    "image_path",
    "relative_path",
    "filename",
    "species_name",
    "species_code",
    "label",
    "acquisition_date",
    "slide_letter",
    "slide_id",
    "image_index",
    "domain",
    "microscope",
    "camera",
    "objective",
    "software",
    "training_domain_bit_depth",
    "external_domain_bit_depth",
    "training_pixel_spacing_um",
    "external_pixel_spacing_um",
    "scale_factor_external_to_training",
    "training_illumination",
    "external_illumination",
    "training_exposure_ms",
    "external_exposure_ms",
    "sample_preparation",
    "sampling_strategy",
    "calibration",
    "hyphae_status",
    "dataset_role",
    "source_input",
    "source_input_sha256",
]

SUMMARY_FILENAME = "external_manifest_summary.json"
CONFIG_SNAPSHOT_FILENAME = "external_validation_config_snapshot.yaml"
METADATA_SNAPSHOT_FILENAME = "external_validation_metadata_snapshot.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a strict image-level external-validation manifest from a "
            "fungal microscopy ZIP/directory and slide-level metadata JSON."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Raw external dataset as fungi.zip or extracted dataset directory.",
    )
    parser.add_argument(
        "--metadata",
        required=True,
        type=Path,
        help="external_validation_metadata.json with slide labels and hyphae status.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output external_manifest.csv path.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional external_validation_config.yaml; used for additional expected values and snapshotting.",
    )
    parser.add_argument(
        "--extract-dir",
        type=Path,
        default=None,
        help=(
            "Directory where ZIP input is extracted. Defaults to "
            "<output_dir>/extracted_raw. Ignored for directory input."
        ),
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Delete and re-extract --extract-dir when input is a ZIP.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Optional path for external_manifest_summary.json.",
    )
    parser.add_argument(
        "--expected-images-min",
        type=int,
        default=None,
        help="Optional warning threshold for minimum images per slide.",
    )
    parser.add_argument(
        "--expected-images-max",
        type=int,
        default=None,
        help="Optional warning threshold for maximum images per slide.",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Abort on metadata, filename, slide-count, or label inconsistencies.",
    )
    parser.add_argument(
        "--allow-unlisted-slides",
        action="store_true",
        help=(
            "Diagnostic mode only: allow observed slide IDs not listed in metadata. "
            "Not recommended for final thesis runs."
        ),
    )
    parser.add_argument(
        "--allow-missing-listed-slides",
        action="store_true",
        help=(
            "Diagnostic mode only: allow metadata-listed slides absent from the raw dataset. "
            "Not recommended for final thesis runs."
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Metadata JSON not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Metadata JSON must contain an object: {path}")
    return data


def load_optional_yaml(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Config YAML not found: {path}")
    try:
        import yaml  # type: ignore
    except ImportError:
        print(
            "WARNING: PyYAML not installed; continuing without parsing YAML config.",
            file=sys.stderr,
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


def write_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=True)


def write_csv(rows: List[Dict[str, Any]], path: Path, columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def snapshot_file(source: Optional[Path], destination: Path) -> None:
    if source is None or not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def safe_zip_member(member: str) -> bool:
    """Reject absolute paths and path traversal inside ZIP archives."""
    p = PurePosixPath(member)
    if p.is_absolute():
        return False
    if any(part == ".." for part in p.parts):
        return False
    if member.endswith("/"):
        return True
    # Reject Windows-style absolute paths such as C:/...
    if re.match(r"^[A-Za-z]:", member):
        return False
    return True


def extract_zip_safely(zip_path: Path, extract_dir: Path, force: bool = False) -> Path:
    if not zip_path.exists():
        raise FileNotFoundError(f"Input ZIP not found: {zip_path}")
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"Input has .zip suffix but is not a valid ZIP archive: {zip_path}")

    if extract_dir.exists() and force:
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    sentinel = extract_dir / ".external_manifest_extracted"
    if sentinel.exists() and not force:
        return extract_dir

    with zipfile.ZipFile(zip_path, "r") as zf:
        bad = [name for name in zf.namelist() if not safe_zip_member(name)]
        if bad:
            preview = bad[:10]
            raise RuntimeError(f"Unsafe ZIP member paths detected: {preview}")
        zf.extractall(extract_dir)

    sentinel.write_text(
        json.dumps(
            {
                "source_zip": str(zip_path),
                "source_sha256": sha256_file(zip_path),
                "extracted_at_unix_time": time.time(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return extract_dir


def find_dataset_root(input_root: Path, class_names: Sequence[str]) -> Path:
    """
    Find the directory containing all expected class subdirectories.

    Handles both:
        fungi/Candida albicans/*.tif
        extracted_zip/fungi/Candida albicans/*.tif
    and nested one-directory wrappers.
    """
    candidates = [input_root]
    for p in input_root.rglob("*"):
        if p.is_dir():
            candidates.append(p)

    class_set = set(class_names)
    matches: List[Tuple[int, Path]] = []
    for cand in candidates:
        child_names = {d.name for d in cand.iterdir() if d.is_dir()} if cand.exists() else set()
        if class_set.issubset(child_names):
            # Prefer the shallowest matching root to avoid matching class subtrees.
            depth = len(cand.relative_to(input_root).parts) if cand != input_root else 0
            matches.append((depth, cand))

    if not matches:
        raise RuntimeError(
            "Could not find dataset root containing class subdirectories: "
            f"{sorted(class_set)} under {input_root}"
        )

    matches.sort(key=lambda x: (x[0], str(x[1])))
    return matches[0][1]


def iter_tiff_paths(dataset_root: Path, class_names: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for cls in sorted(class_names):
        class_dir = dataset_root / cls
        if not class_dir.exists() or not class_dir.is_dir():
            raise FileNotFoundError(f"Expected class directory not found: {class_dir}")
        for p in class_dir.rglob("*"):
            if p.is_file() and p.suffix in SUPPORTED_TIFF_EXTENSIONS:
                paths.append(p)
    return sorted(paths, key=lambda p: str(p.relative_to(dataset_root)).lower())


def parse_filename(filename: str) -> Dict[str, Any]:
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) < 4:
        raise ValueError(
            f"Filename must contain at least date_species_slide_index fields: {filename}"
        )

    acquisition_date = parts[0]
    species_code = parts[1].lower()
    slide_letter = parts[2]
    image_index_token = parts[3]

    if not re.fullmatch(r"\d{8}", acquisition_date):
        raise ValueError(f"Invalid acquisition date in filename {filename!r}: {acquisition_date!r}")
    if not re.fullmatch(r"[A-Za-z]+", species_code):
        raise ValueError(f"Invalid species code in filename {filename!r}: {species_code!r}")
    if not re.fullmatch(r"[A-Za-z]+", slide_letter):
        raise ValueError(f"Invalid slide letter in filename {filename!r}: {slide_letter!r}")
    if not re.fullmatch(r"\d+", image_index_token):
        raise ValueError(
            f"Invalid image index in filename {filename!r}: {image_index_token!r}. "
            "Expected a numeric fourth underscore-separated field."
        )

    slide_id = f"{acquisition_date}_{species_code}_{slide_letter}"
    return {
        "acquisition_date": acquisition_date,
        "species_code": species_code,
        "slide_letter": slide_letter,
        "slide_id": slide_id,
        "image_index": int(image_index_token),
    }


def build_expected_slide_maps(metadata: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return slide_id -> species_name and slide_id -> hyphae_status."""
    species_slide_sets = metadata.get("species_slide_sets")
    if not isinstance(species_slide_sets, dict) or not species_slide_sets:
        raise ValueError("metadata must contain non-empty 'species_slide_sets'.")

    slide_to_species: Dict[str, str] = {}
    duplicate_slides: List[str] = []
    for species_name, slides in species_slide_sets.items():
        if not isinstance(slides, list):
            raise ValueError(f"species_slide_sets[{species_name!r}] must be a list.")
        for sid in slides:
            if not isinstance(sid, str) or not sid:
                raise ValueError(f"Invalid slide ID in species_slide_sets[{species_name!r}]: {sid!r}")
            if sid in slide_to_species and slide_to_species[sid] != species_name:
                duplicate_slides.append(sid)
            slide_to_species[sid] = species_name
    if duplicate_slides:
        raise ValueError(f"Slide IDs assigned to multiple species: {sorted(set(duplicate_slides))}")

    morphology = metadata.get("albicans_morphology_subgroups", {})
    if not isinstance(morphology, dict):
        raise ValueError("metadata['albicans_morphology_subgroups'] must be an object.")

    slide_to_hyphae: Dict[str, str] = {}
    for status in ("hyphae", "no_hyphae", "not_applicable"):
        slides = morphology.get(status, [])
        if not isinstance(slides, list):
            raise ValueError(f"albicans_morphology_subgroups[{status!r}] must be a list.")
        for sid in slides:
            if sid in slide_to_hyphae and slide_to_hyphae[sid] != status:
                raise ValueError(
                    f"Slide {sid!r} appears in multiple morphology subgroups: "
                    f"{slide_to_hyphae[sid]!r} and {status!r}"
                )
            slide_to_hyphae[sid] = status

    return slide_to_species, slide_to_hyphae


def get_float_spacing(spacing_obj: Any, axis: str = "x") -> Optional[float]:
    if isinstance(spacing_obj, dict):
        v = spacing_obj.get(axis)
        return float(v) if v is not None else None
    if isinstance(spacing_obj, (int, float)):
        return float(spacing_obj)
    return None


def compute_scale_factor(metadata: Dict[str, Any]) -> Optional[float]:
    domain_shift = metadata.get("domain_shift_metadata", {})
    if isinstance(domain_shift, dict):
        val = domain_shift.get("external_to_training_pixel_spacing_ratio")
        if val is not None:
            return float(val)

    train = metadata.get("training_domain_reference", {})
    ext = metadata.get("external_acquisition_metadata", {})
    train_x = get_float_spacing(train.get("pixel_spacing_um"), "x") if isinstance(train, dict) else None
    ext_x = get_float_spacing(ext.get("pixel_spacing_um"), "x") if isinstance(ext, dict) else None
    if train_x is not None and ext_x is not None and ext_x > 0:
        return float(train_x / ext_x)
    return None


def as_scalar_spacing(spacing_obj: Any) -> str:
    """Return 'x' spacing as a scalar string for downstream CSV compatibility."""
    val = get_float_spacing(spacing_obj, "x")
    return "" if val is None else f"{val:.12g}"


def create_manifest_rows(
    image_paths: List[Path],
    dataset_root: Path,
    metadata: Dict[str, Any],
    source_input: Path,
    source_sha256: str,
    *,
    allow_unlisted_slides: bool,
) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    warnings_out: List[str] = []
    errors: List[str] = []

    label_mapping = metadata.get("label_mapping")
    if not isinstance(label_mapping, dict) or not label_mapping:
        raise ValueError("metadata must contain non-empty 'label_mapping'.")

    slide_to_species, slide_to_hyphae = build_expected_slide_maps(metadata)

    external = metadata.get("external_acquisition_metadata", {})
    training = metadata.get("training_domain_reference", {})
    if not isinstance(external, dict):
        external = {}
    if not isinstance(training, dict):
        training = {}

    domain = str(metadata.get("domain", "external_new_microscope"))
    dataset_role = str(metadata.get("dataset_role", "external_validation"))
    scale_factor = compute_scale_factor(metadata)

    rows: List[Dict[str, Any]] = []
    seen_rel_paths: Set[str] = set()

    for image_path in image_paths:
        rel_path = image_path.relative_to(dataset_root).as_posix()
        if rel_path in seen_rel_paths:
            errors.append(f"Duplicate relative path detected: {rel_path}")
            continue
        seen_rel_paths.add(rel_path)

        class_folder = image_path.relative_to(dataset_root).parts[0]
        try:
            parsed = parse_filename(image_path.name)
        except Exception as exc:
            errors.append(f"Filename parse failed for {rel_path}: {exc}")
            continue

        species_from_code = DEFAULT_SPECIES_CODE_MAP.get(parsed["species_code"])
        if species_from_code is None:
            errors.append(
                f"Unknown species code {parsed['species_code']!r} in {rel_path}. "
                f"Known codes: {sorted(DEFAULT_SPECIES_CODE_MAP)}"
            )
            continue

        if class_folder != species_from_code:
            errors.append(
                f"Folder/species-code mismatch for {rel_path}: folder={class_folder!r}, "
                f"filename code maps to {species_from_code!r}."
            )
            continue

        slide_id = parsed["slide_id"]
        expected_species = slide_to_species.get(slide_id)
        if expected_species is None:
            msg = f"Observed slide {slide_id!r} is not listed in metadata species_slide_sets."
            if allow_unlisted_slides:
                warnings_out.append(msg)
                expected_species = species_from_code
            else:
                errors.append(msg)
                continue

        if expected_species != species_from_code:
            errors.append(
                f"Metadata/species mismatch for slide {slide_id}: metadata={expected_species!r}, "
                f"filename/folder={species_from_code!r}."
            )
            continue

        if expected_species not in label_mapping:
            errors.append(f"Species {expected_species!r} missing from label_mapping.")
            continue

        hyphae_status = slide_to_hyphae.get(slide_id)
        if hyphae_status is None:
            if expected_species == "Candida glabrata":
                hyphae_status = DEFAULT_HYPHAE_STATUS_GLabrata
                warnings_out.append(
                    f"Slide {slide_id} missing from morphology metadata; assigned not_applicable because it is C. glabrata."
                )
            else:
                errors.append(f"C. albicans slide {slide_id!r} has no hyphae_status in metadata.")
                continue

        if hyphae_status not in VALID_HYPHAE_STATUS:
            errors.append(f"Invalid hyphae_status for {slide_id}: {hyphae_status!r}")
            continue
        if expected_species == "Candida glabrata" and hyphae_status != "not_applicable":
            errors.append(
                f"C. glabrata slide {slide_id!r} must have hyphae_status='not_applicable', got {hyphae_status!r}."
            )
            continue
        if expected_species == "Candida albicans" and hyphae_status not in {"hyphae", "no_hyphae"}:
            errors.append(
                f"C. albicans slide {slide_id!r} must have hyphae/no_hyphae status, got {hyphae_status!r}."
            )
            continue

        row = {
            "image_path": str(image_path.resolve()),
            "relative_path": rel_path,
            "filename": image_path.name,
            "species_name": expected_species,
            "species_code": parsed["species_code"],
            "label": int(label_mapping[expected_species]),
            "acquisition_date": parsed["acquisition_date"],
            "slide_letter": parsed["slide_letter"],
            "slide_id": slide_id,
            "image_index": int(parsed["image_index"]),
            "domain": domain,
            "microscope": external.get("microscope", ""),
            "camera": external.get("camera", ""),
            "objective": external.get("objective", ""),
            "software": external.get("software", ""),
            "training_domain_bit_depth": training.get("bit_depth", ""),
            "external_domain_bit_depth": external.get("bit_depth", external.get("bit_depth_reported_by_lab", "")),
            "training_pixel_spacing_um": as_scalar_spacing(training.get("pixel_spacing_um")),
            "external_pixel_spacing_um": as_scalar_spacing(external.get("pixel_spacing_um")),
            "scale_factor_external_to_training": "" if scale_factor is None else f"{scale_factor:.12g}",
            "training_illumination": training.get("illumination", ""),
            "external_illumination": external.get("illumination", ""),
            "training_exposure_ms": training.get("exposure_time_ms", ""),
            "external_exposure_ms": external.get("exposure_time_ms", ""),
            "sample_preparation": external.get("sample_preparation", ""),
            "sampling_strategy": external.get("sampling_strategy", ""),
            "calibration": external.get("calibration", ""),
            "hyphae_status": hyphae_status,
            "dataset_role": dataset_role,
            "source_input": str(source_input),
            "source_input_sha256": source_sha256,
        }
        rows.append(row)

    rows.sort(
        key=lambda r: (
            str(r["species_name"]),
            str(r["slide_id"]),
            int(r["image_index"]),
            str(r["relative_path"]),
        )
    )
    return rows, warnings_out, errors


def validate_manifest_rows(
    rows: List[Dict[str, Any]],
    metadata: Dict[str, Any],
    *,
    expected_images_min: Optional[int],
    expected_images_max: Optional[int],
    allow_missing_listed_slides: bool,
    allow_unlisted_slides: bool,
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    warnings_out: List[str] = []
    errors: List[str] = []

    if not rows:
        errors.append("No valid TIFF images were included in the manifest.")
        return warnings_out, errors, {}

    slide_to_species_expected, slide_to_hyphae_expected = build_expected_slide_maps(metadata)
    expected_slides = set(slide_to_species_expected.keys())
    observed_slides = {str(r["slide_id"]) for r in rows}

    missing_listed = sorted(expected_slides - observed_slides)
    unlisted_observed = sorted(observed_slides - expected_slides)
    if missing_listed:
        msg = f"Metadata-listed slides absent from raw dataset: {missing_listed}"
        if allow_missing_listed_slides:
            warnings_out.append(msg)
        else:
            errors.append(msg)
    if unlisted_observed:
        msg = f"Observed slides not listed in metadata: {unlisted_observed}"
        if allow_unlisted_slides:
            warnings_out.append(msg)
        else:
            errors.append(msg)

    # One species/label/hyphae status per slide.
    by_slide: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_slide[str(row["slide_id"])].append(row)

    for sid, srows in sorted(by_slide.items()):
        species_values = {str(r["species_name"]) for r in srows}
        label_values = {str(r["label"]) for r in srows}
        hyphae_values = {str(r["hyphae_status"]) for r in srows}
        if len(species_values) != 1:
            errors.append(f"Slide {sid} has mixed species labels: {sorted(species_values)}")
        if len(label_values) != 1:
            errors.append(f"Slide {sid} has mixed numeric labels: {sorted(label_values)}")
        if len(hyphae_values) != 1:
            errors.append(f"Slide {sid} has mixed hyphae_status values: {sorted(hyphae_values)}")

        n_images = len(srows)
        if expected_images_min is not None and n_images < expected_images_min:
            warnings_out.append(
                f"Slide {sid} has {n_images} images, below warning threshold {expected_images_min}."
            )
        if expected_images_max is not None and n_images > expected_images_max:
            warnings_out.append(
                f"Slide {sid} has {n_images} images, above warning threshold {expected_images_max}."
            )

        image_indices = sorted(int(r["image_index"]) for r in srows)
        duplicate_indices = sorted(k for k, v in Counter(image_indices).items() if v > 1)
        if duplicate_indices:
            errors.append(f"Slide {sid} has duplicate image_index values: {duplicate_indices}")

    expected_counts = metadata.get("expected_slide_counts", {})
    if not isinstance(expected_counts, dict):
        expected_counts = {}

    total_slides_observed = len(observed_slides)
    expected_total = expected_counts.get("total_slides")
    if expected_total is not None and total_slides_observed != int(expected_total):
        errors.append(
            f"Expected {int(expected_total)} total slides, observed {total_slides_observed}."
        )

    species_by_slide = {
        sid: next(iter({str(r["species_name"]) for r in srows}))
        for sid, srows in by_slide.items()
    }
    species_counts = Counter(species_by_slide.values())
    # Only validate true species keys here. Morphology count keys such as
    # "Candida albicans with hyphae" are validated separately below.
    label_mapping = metadata.get("label_mapping", {})
    species_keys = [k for k in label_mapping.keys() if k in expected_counts]
    for species_name in sorted(species_keys):
        observed = species_counts.get(species_name, 0)
        expected = int(expected_counts[species_name])
        if observed != expected:
            errors.append(
                f"Expected {expected} slides for {species_name}, observed {observed}."
            )

    # Morphology counts at slide level.
    hyphae_by_slide = {
        sid: next(iter({str(r["hyphae_status"]) for r in srows}))
        for sid, srows in by_slide.items()
    }
    observed_alb_hyphae = sum(
        1
        for sid, status in hyphae_by_slide.items()
        if species_by_slide.get(sid) == "Candida albicans" and status == "hyphae"
    )
    observed_alb_no_hyphae = sum(
        1
        for sid, status in hyphae_by_slide.items()
        if species_by_slide.get(sid) == "Candida albicans" and status == "no_hyphae"
    )
    if "Candida albicans with hyphae" in expected_counts:
        exp = int(expected_counts["Candida albicans with hyphae"])
        if observed_alb_hyphae != exp:
            errors.append(
                f"Expected {exp} C. albicans hyphae slides, observed {observed_alb_hyphae}."
            )
    if "Candida albicans without hyphae" in expected_counts:
        exp = int(expected_counts["Candida albicans without hyphae"])
        if observed_alb_no_hyphae != exp:
            errors.append(
                f"Expected {exp} C. albicans no_hyphae slides, observed {observed_alb_no_hyphae}."
            )

    # Validate morphology metadata disjointness and coverage.
    morph = metadata.get("albicans_morphology_subgroups", {})
    if isinstance(morph, dict):
        h = set(morph.get("hyphae", []) or [])
        nh = set(morph.get("no_hyphae", []) or [])
        na = set(morph.get("not_applicable", []) or [])
        if h & nh:
            errors.append(f"Slides present in both hyphae and no_hyphae metadata: {sorted(h & nh)}")
        if (h | nh) & na:
            errors.append(f"Albicans morphology slides overlap not_applicable: {sorted((h | nh) & na)}")

    image_count_by_slide = {sid: len(srows) for sid, srows in sorted(by_slide.items())}
    summary_counts = {
        "n_images": len(rows),
        "n_slides": total_slides_observed,
        "species_slide_counts": dict(sorted(species_counts.items())),
        "species_image_counts": dict(sorted(Counter(str(r["species_name"]) for r in rows).items())),
        "hyphae_slide_counts": dict(sorted(Counter(hyphae_by_slide.values()).items())),
        "hyphae_image_counts": dict(sorted(Counter(str(r["hyphae_status"]) for r in rows).items())),
        "image_count_by_slide": image_count_by_slide,
        "images_per_slide_min": min(image_count_by_slide.values()) if image_count_by_slide else None,
        "images_per_slide_max": max(image_count_by_slide.values()) if image_count_by_slide else None,
        "images_per_slide_mean": (
            sum(image_count_by_slide.values()) / len(image_count_by_slide)
            if image_count_by_slide
            else None
        ),
        "observed_slide_ids": sorted(observed_slides),
        "missing_metadata_slides": missing_listed,
        "unlisted_observed_slides": unlisted_observed,
    }
    return warnings_out, errors, summary_counts



def enrich_metadata_from_config(metadata: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fill missing acquisition/reference fields in metadata from the YAML config.

    The JSON metadata remains the authority for slide IDs, labels, and hyphae
    status. The YAML config is only used as a non-destructive fallback for
    acquisition-domain descriptors such as microscope, camera, illumination,
    exposure, and pixel spacing.
    """
    if not config:
        return metadata

    enriched = json.loads(json.dumps(metadata))

    cfg_train = config.get("training_domain_reference", {})
    cfg_ext = config.get("external_domain", {})
    cfg_domain_shift = config.get("domain_shift", {})

    if isinstance(cfg_train, dict):
        train = enriched.setdefault("training_domain_reference", {})
        if isinstance(train, dict):
            for k, v in cfg_train.items():
                train.setdefault(k, v)

    if isinstance(cfg_ext, dict):
        ext = enriched.setdefault("external_acquisition_metadata", {})
        if isinstance(ext, dict):
            for k, v in cfg_ext.items():
                if k == "bit_depth_reported_by_lab" and "bit_depth" not in ext:
                    ext["bit_depth"] = v
                ext.setdefault(k, v)

    if isinstance(cfg_domain_shift, dict):
        ds = enriched.setdefault("domain_shift_metadata", {})
        if isinstance(ds, dict):
            px = cfg_domain_shift.get("pixel_spacing")
            if isinstance(px, dict):
                ratio = px.get("external_to_training_linear_scale_factor")
                if ratio is not None:
                    ds.setdefault("external_to_training_pixel_spacing_ratio", ratio)

    return enriched

def make_summary(
    *,
    args: argparse.Namespace,
    metadata: Dict[str, Any],
    config: Dict[str, Any],
    dataset_root: Path,
    rows: List[Dict[str, Any]],
    counts: Dict[str, Any],
    warnings_out: List[str],
    errors: List[str],
    elapsed_seconds: float,
    source_sha256: str,
) -> Dict[str, Any]:
    return {
        "schema_version": "1.0",
        "script": "external_manifest.py",
        "scientific_role": (
            "Authoritative image-level manifest for an independent external "
            "acquisition-domain validation cohort. No train/test split is created."
        ),
        "inputs": {
            "raw_input": str(args.input),
            "raw_input_sha256": source_sha256,
            "metadata_json": str(args.metadata),
            "config_yaml": str(args.config) if args.config else None,
        },
        "outputs": {
            "manifest_csv": str(args.output),
            "summary_json": str(args.summary_output or args.output.with_name(SUMMARY_FILENAME)),
        },
        "dataset_root_used_for_relative_paths": str(dataset_root),
        "strict_mode": bool(args.strict),
        "diagnostic_relaxations": {
            "allow_unlisted_slides": bool(args.allow_unlisted_slides),
            "allow_missing_listed_slides": bool(args.allow_missing_listed_slides),
        },
        "metadata_role": metadata.get("metadata_role", "external_validation_slide_and_acquisition_metadata"),
        "dataset_role": metadata.get("dataset_role", "external_validation"),
        "domain": metadata.get("domain", "external_new_microscope"),
        "label_mapping": metadata.get("label_mapping", {}),
        "expected_slide_counts": metadata.get("expected_slide_counts", {}),
        "counts": counts,
        "guardrails": {
            "no_train_test_split_created": True,
            "labels_from_metadata_and_folder_names_only": True,
            "hyphae_status_used_for_subgroup_analysis_only": True,
            "model_predictions_used": False,
        },
        "validation_status": "passed" if not errors else "failed",
        "warnings": warnings_out,
        "errors": errors,
        "elapsed_seconds": elapsed_seconds,
        "config_present": bool(config),
        "manifest_columns": MANIFEST_COLUMNS,
    }


def infer_expected_image_thresholds(args: argparse.Namespace, config: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    if args.expected_images_min is not None or args.expected_images_max is not None:
        return args.expected_images_min, args.expected_images_max
    min_v = get_nested(
        config,
        ["expected_external_cohort", "expected_images_per_slide", "acceptable_minimum_for_audit_warning"],
        None,
    )
    max_v = get_nested(
        config,
        ["expected_external_cohort", "expected_images_per_slide", "acceptable_maximum_for_audit_warning"],
        None,
    )
    return (int(min_v) if min_v is not None else None, int(max_v) if max_v is not None else None)


def main() -> None:
    start = time.time()
    args = parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.summary_output or args.output.with_name(SUMMARY_FILENAME)

    metadata_raw = read_json(args.metadata)
    config = load_optional_yaml(args.config)
    metadata = enrich_metadata_from_config(metadata_raw, config)
    class_names = list(metadata.get("label_mapping", {}).keys())
    if not class_names:
        raise ValueError("metadata['label_mapping'] must define class names.")

    source_input = args.input
    if source_input.is_file() and source_input.suffix.lower() == ".zip":
        source_sha256 = sha256_file(source_input)
        extract_dir = args.extract_dir or (args.output.parent / "extracted_raw")
        scan_root = extract_zip_safely(source_input, extract_dir, force=args.force_extract)
    elif source_input.is_dir():
        # Directory hash is intentionally not computed to avoid expensive reads here.
        source_sha256 = "directory_input_not_hashed"
        scan_root = source_input
    else:
        raise FileNotFoundError(f"Input must be a ZIP file or directory: {source_input}")

    dataset_root = find_dataset_root(scan_root, class_names)
    image_paths = iter_tiff_paths(dataset_root, class_names)
    if not image_paths:
        raise RuntimeError(f"No TIFF files found under dataset root: {dataset_root}")

    rows, row_warnings, row_errors = create_manifest_rows(
        image_paths=image_paths,
        dataset_root=dataset_root,
        metadata=metadata,
        source_input=source_input,
        source_sha256=source_sha256,
        allow_unlisted_slides=args.allow_unlisted_slides,
    )

    expected_min, expected_max = infer_expected_image_thresholds(args, config)
    val_warnings, val_errors, counts = validate_manifest_rows(
        rows,
        metadata,
        expected_images_min=expected_min,
        expected_images_max=expected_max,
        allow_missing_listed_slides=args.allow_missing_listed_slides,
        allow_unlisted_slides=args.allow_unlisted_slides,
    )

    warnings_out = row_warnings + val_warnings
    errors = row_errors + val_errors
    elapsed = time.time() - start

    summary = make_summary(
        args=args,
        metadata=metadata,
        config=config,
        dataset_root=dataset_root,
        rows=rows,
        counts=counts,
        warnings_out=warnings_out,
        errors=errors,
        elapsed_seconds=elapsed,
        source_sha256=source_sha256,
    )

    # Always write summary for debugging, even on strict failure.
    write_json(summary, summary_path)

    if errors and args.strict:
        preview = "\n  - ".join(errors[:25])
        raise RuntimeError(
            "External manifest validation failed in strict mode. "
            f"Summary written to {summary_path}.\n  - {preview}"
        )

    write_csv(rows, args.output, MANIFEST_COLUMNS)

    # Snapshots support reproducibility and auditability.
    snapshot_file(args.metadata, args.output.parent / METADATA_SNAPSHOT_FILENAME)
    snapshot_file(args.config, args.output.parent / CONFIG_SNAPSHOT_FILENAME)

    print("\nExternal manifest created successfully")
    print("=" * 72)
    print(f"Manifest CSV : {args.output}")
    print(f"Summary JSON : {summary_path}")
    print(f"Dataset root : {dataset_root}")
    print(f"Images       : {counts.get('n_images', len(rows))}")
    print(f"Slides       : {counts.get('n_slides')}")
    print(f"Species slides: {counts.get('species_slide_counts')}")
    print(f"Hyphae slides : {counts.get('hyphae_slide_counts')}")
    if warnings_out:
        print(f"Warnings     : {len(warnings_out)}; see summary JSON")
    print("No train/test split was created; all slides remain external validation data.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
