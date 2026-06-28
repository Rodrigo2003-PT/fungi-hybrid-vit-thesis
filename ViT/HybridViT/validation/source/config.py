"""
Configuration Module

Author: Rodrigo Sá
Date: 2026
"""

import json
import os
from typing import Any, Dict

BASE_CONFIG: Dict[str, Any] = {
    'random_seed': 42,
    'image_size': 384,
    'bit_depth': 12,
    'channels': 1,
    'n_folds': 5,
    'n_iterations': 30,
    'num_epochs': 100,
    'patience': 20,
    'batch_size': 64,
    'learning_rate': 3e-4,
    'weight_decay': 0.05,
    'warmup_epochs': 5,
    'ema_alpha': 0.3,
    'min_delta': 0.01,
    'epsilon_slide_ba': 1e-6,
    'epsilon_val_loss': 1e-6,
    'checkpoint_every_n_epochs': 25,
    'num_workers': 2,
    'use_amp': True,
    'gradient_clip_norm': 1.0,
    'class_weight_gamma': 0.0,
    'model_config': {
        'dim': 256,
        'depth': 8,
        'heads': 6,
        'mlp_dim': 512,
        'dim_head': 64,      
        'pe_temperature': 10000,
    },
    'convtok_blocks': [
        {"conv_k": 7, "conv_s": 2, "conv_p": 3, "pool_k": 3, "pool_s": 2, "pool_p": 1},
        {"conv_k": 3, "conv_s": 2, "conv_p": 1, "pool_k": 3, "pool_s": 2, "pool_p": 1},
    ],
    'convtok_hidden_channels': 64,
    'convtok_match_baseline_tokens': True,
    'convtok_expected_hw': [24, 24],
    'convtok_expected_tokens': 576,
    'convtok_post_ln': True,
    'convtok_conv_bias': False,
    'init_policy': 'default',
}

_SHARED_AUGMENTATION: Dict[str, Any] = {
    'random_resized_crop_scale': (0.9, 1.0),
    'random_resized_crop_ratio': (0.95, 1.05),
    'horizontal_flip_p': 0.5,
    'vertical_flip_p': 0.5,
    'rotation_degrees': 30,
    'use_mixup': True,
    'mixup_alpha': 0.2,
    'mixup_prob': 0.5,
}

LSA_CONVTOK_CONFIG: Dict[str, Any] = {
    'augmentation': _SHARED_AUGMENTATION,
    'class_weight_gamma': 0.0,
}

LSA_CONVTOK_EXPLICIT_INIT_CONFIG: Dict[str, Any] = {
    **LSA_CONVTOK_CONFIG,
    'init_policy': 'explicit_kaiming',
}

_VALID_EXPERIMENTS = (
    'lsa_convtok',
    'lsa_convtok_explicit_init',
)


def get_config(experiment_name: str = 'convtok', **kwargs) -> Dict[str, Any]:
    """
    Build a complete, validated configuration for a named experiment.

    Parameters
    ----------
    experiment_name : str
        One of:
        - ``'lsa_convtok'``            — LSA-ConvTok (aggregate), Policy A
        - ``'lsa_convtok_explicit_init'`` — LSA-ConvTok, Policy B
    **kwargs
        Optional overrides applied after the preset (ad-hoc ablations).

    Returns
    -------
    dict
        Complete, JSON-serialisable configuration dictionary.

    Raises
    ------
    ValueError
        If ``experiment_name`` is not recognised or config fails validation.
    """
    # Deep copy via JSON round-trip (safe for all JSON-serialisable types)
    config = json.loads(json.dumps(BASE_CONFIG))

    if experiment_name == 'lsa_convtok':
        config.update(LSA_CONVTOK_CONFIG)
    elif experiment_name == 'lsa_convtok_explicit_init':
        config.update(LSA_CONVTOK_EXPLICIT_INIT_CONFIG)
    else:
        raise ValueError(
            f"Unknown experiment_name: '{experiment_name}'. "
            f"Valid options: {_VALID_EXPERIMENTS}."
        )

    # Apply caller overrides
    config.update(kwargs)
    config['experiment_name'] = experiment_name

    # Slide-level split quality constraints
    config['min_slides_per_class_validation'] = 4
    config['min_slide_balance_ratio']         = 0.5
    config['max_image_imbalance_ratio']        = 0.2

    # Recompute derived field after any hw override
    hw = config.get('convtok_expected_hw')
    if hw is not None:
        config['convtok_expected_tokens'] = int(hw[0]) * int(hw[1])

    _validate_config(config)
    return config


def _validate_config(config: Dict[str, Any]) -> None:
    """Run sanity checks on the assembled config."""
    exp = config.get('experiment_name', '')

    if 'patch_size' in config and ('convtok' in exp):
        raise ValueError(
            f"'patch_size' found in config for experiment '{exp}'. "
            "ConvTokenizer experiments derive tokenization geometry from "
            "'convtok_blocks', not patch_size.  Remove patch_size from "
            "your config or overrides to prevent silent cross-pipeline "
            "confusion."
        )

    # convtok_blocks
    blocks = config.get('convtok_blocks')
    if not blocks or len(blocks) == 0:
        raise ValueError("convtok_blocks must be a non-empty list.")
    required_keys = {'conv_k', 'conv_s', 'conv_p', 'pool_k', 'pool_s', 'pool_p'}
    for i, blk in enumerate(blocks):
        missing = required_keys - set(blk.keys())
        if missing:
            raise ValueError(f"convtok_blocks[{i}] is missing keys: {missing}")

    # post_ln must always be True
    if config.get('convtok_post_ln', True) is not True:
        raise ValueError(
            "convtok_post_ln must be True to maintain the normalised-token "
            "interface required for a fair ablation."
        )

    # init_policy
    ip = config.get('init_policy', 'default')
    if ip not in ('default', 'explicit_kaiming'):
        raise ValueError(
            f"init_policy must be 'default' or 'explicit_kaiming', got '{ip}'."
        )

    # hidden_channels
    hc = config.get('convtok_hidden_channels', 64)
    if not isinstance(hc, int) or hc < 1:
        raise ValueError(
            f"convtok_hidden_channels must be a positive integer, got {hc!r}."
        )

    # model_config fields
    mc   = config.get('model_config', {})
    for field, default in [('dim_head', 64), ('pe_temperature', 10000)]:
        val = mc.get(field, default)
        if not isinstance(val, int) or val < 1:
            raise ValueError(
                f"model_config['{field}'] must be a positive integer, got {val!r}."
            )

    # Training hyperparameter sanity
    if config.get('learning_rate', 1e-4) <= 0:
        raise ValueError("learning_rate must be positive.")
    if config.get('weight_decay', 0) < 0:
        raise ValueError("weight_decay must be non-negative.")
    if config.get('batch_size', 1) < 1:
        raise ValueError("batch_size must be >= 1.")


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Persist configuration as a JSON file named after the experiment.

    Parameters
    ----------
    config      : dict — must contain 'experiment_name'.
    output_path : str  — directory where the JSON file is written.
    """
    os.makedirs(output_path, exist_ok=True)
    file_path = os.path.join(
        output_path, f'config_{config["experiment_name"]}.json'
    )
    with open(file_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"  Config saved to: {file_path}")
