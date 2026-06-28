"""
Configuration Module

Author: Rodrigo Sá
Date: 2025
"""

import json
import os
from typing import Dict, Any


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
    'ema_alpha': 0.3,
    'min_delta': 0.01,
    'epsilon_slide_ba': 1e-6,
    'epsilon_val_loss': 1e-6,
    'checkpoint_every_n_epochs': 25,
    'num_workers': 2,
    'use_amp': True,
    'gradient_clip_norm': 1.0,
    'class_weight_gamma': 0.0,
    'channels': 1,
    'model_config': {
        'num_classes': 2,
        'channels': 1,
    },
}

STANDARD_CONFIG: Dict[str, Any] = {
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


def get_config(experiment_name: str = 'standard_cnn', **kwargs) -> Dict[str, Any]:
    """
    Get configuration for the DenseNet-121 CNN baseline experiment.

    Args:
        experiment_name: Must be 'standard_cnn'.
        **kwargs: Additional overrides.

    Returns:
        Configuration dictionary.

    Raises:
        ValueError: If an unknown experiment_name is provided.
    """
    # Deep copy to prevent mutation of module-level constants
    config = json.loads(json.dumps(BASE_CONFIG))

    if experiment_name == 'standard_cnn':
        config.update(STANDARD_CONFIG)
    else:
        raise ValueError(
            f"Unknown experiment_name: '{experiment_name}'. "
            f"Valid options: 'standard_cnn'"
        )

    # Apply any explicit overrides
    config.update(kwargs)

    config['experiment_name'] = experiment_name

    config['min_slides_per_class_validation'] = 4
    config['min_slide_balance_ratio'] = 0.5
    config['max_image_imbalance_ratio'] = 0.2

    return config


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save configuration to JSON file.

    Args:
        config: Configuration dictionary.
        output_path: Directory to save config.
    """
    os.makedirs(output_path, exist_ok=True)
    file_path = os.path.join(output_path, f'config_{config["experiment_name"]}.json')

    with open(file_path, 'w') as f:
        json.dump(config, f, indent=4)

    print(f"  Config saved to: {file_path}")
