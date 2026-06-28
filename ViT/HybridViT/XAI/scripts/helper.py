"""
Helper Utilities

Author: Rodrigo Sá
Date: 2025
"""

import re
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from collections import Counter
import warnings


def extract_slide_id_from_filename(filename: str, validate: bool = True) -> str:
    """
    Extract slide ID from preprocessed filename with validation.
    
    Args:
        filename: Filename (with or without extension)
        validate: Perform format validation
        
    Returns:
        slide_id: Slide identifier string (e.g., "20230115_01_A")
        
    Raises:
        ValueError: If filename doesn't match expected pattern
    """
    # Remove extension if present
    stem = Path(filename).stem
    
    # Split by underscore
    parts = stem.split('_')
    
    # Validation (if requested)
    if validate:
        if len(parts) < 3:
            raise ValueError(
                f"Filename '{filename}' doesn't match expected pattern.\n"
                f"Expected: YYYYMMDD_XX_Y_patch_NNNN.npy\n"
                f"Got only {len(parts)} components after splitting by '_'"
            )
        
        # Validate date component (8 digits)
        if not re.match(r'^\d{8}$', parts[0]):
            warnings.warn(
                f"Date component '{parts[0]}' in '{filename}' doesn't match "
                f"expected format YYYYMMDD (8 digits). Proceeding anyway."
            )
        
        # Validate species code (3 digits)
        if not re.match(r'^[a-zA-Z]{3}$', parts[1]):
            warnings.warn(
                f"Species code '{parts[1]}' in '{filename}' doesn't match "
                f"expected format XXX (3 letters). Proceeding anyway."
            )
        
        # Validate letter identifier (single letter)
        if not re.match(r'^[A-Za-z]$', parts[2]):
            warnings.warn(
                f"Letter identifier '{parts[2]}' in '{filename}' is not a "
                f"single letter. Proceeding anyway."
            )
    
    # Take first 3 components: date, species code, letter
    if len(parts) >= 3:
        slide_id = f"{parts[0]}_{parts[1]}_{parts[2]}"
    else:
        # Fallback: return full stem (should not reach here if validate=True)
        warnings.warn(
            f"Using full filename stem '{stem}' as slide ID due to unexpected format"
        )
        slide_id = stem
    
    return slide_id


def create_slide_id_provenance_map(
    file_paths: List[str],
    slide_ids: np.ndarray,
    labels: np.ndarray
) -> Dict[str, Dict]:
    """
    Create comprehensive provenance mapping: Slide ID → metadata.
    
    Args:
        file_paths: List of file paths [N]
        slide_ids: Array of slide IDs [N]
        labels: Array of labels [N]
        
    Returns:
        provenance_map: Dictionary mapping slide_id to metadata:
            {
                "slide_id": {
                    "label": int,  # Ground truth class (validated unanimous)
                    "n_patches": int,  # Number of patches from this slide
                    "file_paths": List[str],  # All patch file paths
                    "patch_indices": List[int]  # Dataset indices
                }
            }
    """
    provenance_map = {}
    unique_slides = np.unique(slide_ids)
    
    for slide_id in unique_slides:
        slide_mask = slide_ids == slide_id
        slide_indices = np.where(slide_mask)[0]
        slide_labels = labels[slide_mask]
        
        # Validate unanimous label
        unique_labels = np.unique(slide_labels)
        if len(unique_labels) > 1:
            raise ValueError(
                f"Data integrity error: Slide '{slide_id}' has conflicting labels: "
                f"{unique_labels}. All patches from same slide must share label."
            )
        
        slide_label = int(slide_labels[0])
        slide_file_paths = [file_paths[i] for i in slide_indices]
        
        provenance_map[slide_id] = {
            "label": slide_label,
            "n_patches": int(len(slide_indices)),
            "file_paths": slide_file_paths,
            "patch_indices": slide_indices.tolist()
        }
    
    return provenance_map


def validate_slide_grouping(
    slide_ids: np.ndarray,
    labels: np.ndarray
) -> Tuple[bool, Optional[str]]:
    """
    Validate that all patches from same slide have same label.

    Args:
        slide_ids: Array of slide identifiers [N]
        labels: Array of ground truth labels [N]
        
    Returns:
        is_valid: True if validation passes
        error_message: Description of error if validation fails
    """
    unique_slides = np.unique(slide_ids)
    
    for slide_id in unique_slides:
        slide_mask = slide_ids == slide_id
        slide_labels = labels[slide_mask]
        
        unique_labels = np.unique(slide_labels)
        if len(unique_labels) > 1:
            label_counts = Counter(slide_labels)
            return False, (
                f"Slide '{slide_id}' has conflicting labels: {dict(label_counts)}. "
                f"All patches from same slide must have same ground truth label. "
                f"This indicates a data integrity issue that must be resolved."
            )
    
    return True, None