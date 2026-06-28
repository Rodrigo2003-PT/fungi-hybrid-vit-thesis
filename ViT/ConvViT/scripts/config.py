"""
Configuration Module

Config fields
-------------
convtok_blocks : list[dict]
    Ordered block configurations for ConvTokenizer. Each dict:
      conv_k, conv_s, conv_p  – Conv2d kernel/stride/padding
      pool_k, pool_s, pool_p  – MaxPool2d kernel/stride/padding
    Default targets ×16 overall downsampling → 24×24 tokens for 384 input.

convtok_match_baseline_tokens : bool
    If True, a runtime assertion enforces (H', W') == convtok_expected_hw.

convtok_expected_hw : list[int]
    Expected [H', W'] grid. Stored as a list for JSON round-trip safety.
    Equivalent constraint: N = H' * W' == convtok_expected_tokens.

convtok_expected_tokens : int
    Derived from convtok_expected_hw for convenience checks.

convtok_post_ln : bool
    Always True (ConvTok post-reshape LayerNorm).

convtok_conv_bias : bool
    Conv2d bias flag. Hassani et al. CCT reference uses False.

init_policy : str
    "default"           – PyTorch default init (primary ablation).
    "explicit_kaiming"  – Kaiming Normal for Conv2d / trunc-normal for Linear
                          (separate ablation factor;).

Author: Rodrigo Sá
Date: 2025
"""

import json
import os
from typing import Dict, Any

# ---------------------------------------------------------------------------
# Base configuration
# ---------------------------------------------------------------------------

BASE_CONFIG: Dict[str, Any] = {
    'random_seed': 42,
    'image_size': 384,
    'bit_depth': 12,
    'n_folds': 5,
    'n_iterations': 30,
    'num_epochs': 100,
    'patience': 20,
    'batch_size': 64,
    'learning_rate': 3e-4,
    'weight_decay': 0.05,
    'warmup_epochs': 5,
    # Early stopping
    'ema_alpha': 0.3,
    'min_delta': 0.01,
    # Lexicographic selection
    'epsilon_slide_ba': 1e-6,
    'epsilon_val_loss': 1e-6,
    'checkpoint_every_n_epochs': 25,
    'num_workers': 2,
    'use_amp': True,
    'gradient_clip_norm': 1.0,
    'class_weight_gamma': 0.0,
    'channels': 1,
    'model_config': {
        'dim': 256,
        'depth': 8,
        'heads': 6,
        'mlp_dim': 512,
        'dim_head': 64,          # per-head attention dimension
        'pe_temperature': 10000, # sin-cos PE temperature;
    },
    # -----------------------------------------------------------------------
    # ConvTok tokenizer configuration
    # -----------------------------------------------------------------------
    'convtok_blocks': [
        {"conv_k": 7, "conv_s": 2, "conv_p": 3, "pool_k": 3, "pool_s": 2, "pool_p": 1},
        {"conv_k": 3, "conv_s": 2, "conv_p": 1, "pool_k": 3, "pool_s": 2, "pool_p": 1},
    ],
    'convtok_hidden_channels': 64,
    'convtok_match_baseline_tokens': True,
    'convtok_expected_hw': [24, 24],
    'convtok_expected_tokens': 576,
    'convtok_post_ln': True, 
    'convtok_conv_bias': False,      # matches Hassani et al. CCT reference
    # Initialisation policy
    'init_policy': 'default',        # "default" | "explicit_kaiming"
}

# ---------------------------------------------------------------------------
# Experiment presets
# ---------------------------------------------------------------------------

# Primary ConvTok experiment — two-block cascade, default init, N=576 matched.
CONVTOK_CONFIG: Dict[str, Any] = {
    'augmentation': {
        'random_resized_crop_scale': (0.9, 1.0),
        'random_resized_crop_ratio': (0.95, 1.05),
        'horizontal_flip_p': 0.5,
        'vertical_flip_p': 0.5,
        'rotation_degrees': 30,
        'use_mixup': True,
        'mixup_alpha': 0.2,
        'mixup_prob': 0.5,
    },
    'class_weight_gamma': 0.0,
}
CONVTOK_EXPLICIT_INIT_CONFIG: Dict[str, Any] = {
    **CONVTOK_CONFIG,
    'init_policy': 'explicit_kaiming',
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_config(experiment_name: str = 'convtok_matched_n', **kwargs) -> Dict[str, Any]:
    """
    Build configuration for a named ConvTok experiment.

    Parameters
    ----------
    experiment_name : str
        - 'convtok'     :           — default PyTorch init
        - 'convtok_explicit_init' : Kaiming Normal for Conv2d, 
                                    trunc-normal for Linear.

    **kwargs
        Additional overrides applied after the preset, enabling ad-hoc ablations

    Returns
    -------
    dict
        Complete, JSON-serialisable configuration dictionary.

    Raises
    ------
    ValueError
        If experiment_name is not recognised.
    """
    config = json.loads(json.dumps(BASE_CONFIG))  # deep copy, JSON-safe

    if experiment_name == 'convtok':
        config.update(CONVTOK_CONFIG)
    elif experiment_name == 'convtok_explicit_init':
        config.update(CONVTOK_EXPLICIT_INIT_CONFIG)
    else:
        raise ValueError(
            f"Unknown experiment_name: '{experiment_name}'. "
            f"Valid options: 'convtok', 'convtok_explicit_init'."
        )

    # Apply caller overrides
    config.update(kwargs)

    config['experiment_name'] = experiment_name

    # Slide-level split quality constraints
    config['min_slides_per_class_validation'] = 4
    config['min_slide_balance_ratio'] = 0.5
    config['max_image_imbalance_ratio'] = 0.2

    # Recompute derived field after any override
    hw = config.get('convtok_expected_hw')
    if hw is not None:
        config['convtok_expected_tokens'] = int(hw[0]) * int(hw[1])

    _validate_config(config)

    return config
    
def _validate_config(config: Dict[str, Any]) -> None:
    """Run sanity checks on the assembled config."""
    blocks = config.get('convtok_blocks')
    if not blocks or len(blocks) == 0:
        raise ValueError("convtok_blocks must be a non-empty list.")

    for i, blk in enumerate(blocks):
        required_keys = {'conv_k', 'conv_s', 'conv_p', 'pool_k', 'pool_s', 'pool_p'}
        missing = required_keys - set(blk.keys())
        if missing:
            raise ValueError(f"convtok_blocks[{i}] is missing keys: {missing}")

    if config.get('convtok_post_ln', True) is not True:
        raise ValueError(
            "convtok_post_ln must be True to match the normalised-token interface "
        )

    ip = config.get('init_policy', 'default')
    if ip not in ('default', 'explicit_kaiming'):
        raise ValueError(
            f"init_policy must be 'default' or 'explicit_kaiming', got '{ip}'."
        )

    hc = config.get('convtok_hidden_channels', 64)
    if not isinstance(hc, int) or hc < 1:
        raise ValueError(
            f"convtok_hidden_channels must be a positive integer, got {hc!r}."
        )

    mc = config.get('model_config', {})
    dh = mc.get('dim_head', 64)
    if not isinstance(dh, int) or dh < 1:
        raise ValueError(
            f"model_config['dim_head'] must be a positive integer, got {dh!r}."
        )

    pt = mc.get('pe_temperature', 10000)
    if not isinstance(pt, int) or pt < 1:
        raise ValueError(
            f"model_config['pe_temperature'] must be a positive integer, got {pt!r}."
        )


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Persist configuration as a JSON file named after the experiment.

    Parameters
    ----------
    config : dict
        Configuration dictionary (must contain 'experiment_name').
    output_path : str
        Directory where the JSON file is written.
    """
    os.makedirs(output_path, exist_ok=True)
    file_path = os.path.join(
        output_path, f'config_{config["experiment_name"]}.json'
    )

    with open(file_path, 'w') as f:
        json.dump(config, f, indent=4)

    print(f"  Config saved to: {file_path}")
