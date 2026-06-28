"""
Training Utilities
Author: Rodrigo Sá
Date: 2025
"""

import torch
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from collections import Counter
import numpy as np
from numpy.random import SeedSequence
import warnings
from typing import Dict, List, Tuple, Callable, Optional
from sklearn.metrics import (
    balanced_accuracy_score,
    matthews_corrcoef,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold


class LexBestModelTracker:
    """
    Tracks best model: (maximize A, minimize L, minimize t).
    
    Selection Rule (Formal Specification):
    ----------------------------------------
    For epochs t ∈ {0, 1, ..., T-1}, define:
        A_t = slide-level balanced accuracy at epoch t
        L_t = validation loss at epoch t
        
    The best epoch t* is selected via lexicographic ordering:
        1. t* ∈ argmax_t A_t
        2. If tie: t* ∈ argmin_{t: A_t = max_k A_k} L_t
        3. If still tied: t* = min{t : (A_t, L_t) satisfy (1) and (2)}
    """
    
    def __init__(
        self, 
        epsilon_A: float = 1e-6,
        epsilon_L: float = 1e-6
    ):
        """
        Args:
            epsilon_A: Tolerance for comparing slide balanced accuracy
            epsilon_L: Tolerance for comparing validation loss
        """
        if epsilon_A < 0:
            raise ValueError(f"epsilon_A must be >= 0, got {epsilon_A}")
        if epsilon_L < 0:
            raise ValueError(f"epsilon_L must be >= 0, got {epsilon_L}")
            
        self.epsilon_A = epsilon_A
        self.epsilon_L = epsilon_L
        
        # Best model state
        self.best_epoch_idx: int = -1
        self.best_slide_ba: Optional[float] = None
        self.best_val_loss: Optional[float] = None
        
        # Full history for analysis
        self.slide_ba_history: List[float] = []
        self.val_loss_history: List[float] = []
    
    def update(
        self, 
        current_epoch: int,
        slide_ba: float,
        val_loss: float
    ) -> Dict[str, any]:
        """
        Update tracker with metrics from current epoch.
        
        Args:
            current_epoch: Current epoch index (0-based)
            slide_ba: Slide-level balanced accuracy at current epoch
            val_loss: Validation loss at current epoch
            
        Returns:
            Dictionary containing:
                - 'improved': Whether this epoch became new best
                - 'best_epoch': Current best epoch index
                - 'best_slide_ba': Slide BA at best epoch
                - 'best_val_loss': Val loss at best epoch
                - 'improvement_reason': String explaining why (if improved)
        """
        # Record history
        self.slide_ba_history.append(slide_ba)
        self.val_loss_history.append(val_loss)
        
        improved = False
        improvement_reason = None
        
        # Initialize on first epoch
        if self.best_epoch_idx < 0:
            self.best_epoch_idx = current_epoch
            self.best_slide_ba = slide_ba
            self.best_val_loss = val_loss
            improved = True
            improvement_reason = "first_epoch"
        else:
            
            # 1.maximize slide_ba
            if slide_ba > self.best_slide_ba + self.epsilon_A:
                self.best_epoch_idx = current_epoch
                self.best_slide_ba = slide_ba
                self.best_val_loss = val_loss
                improved = True
                improvement_reason = "better_slide_ba"
                
            elif abs(slide_ba - self.best_slide_ba) <= self.epsilon_A:
                
                # 2.minimize val_loss
                if val_loss < self.best_val_loss - self.epsilon_L:
                    self.best_epoch_idx = current_epoch
                    self.best_slide_ba = slide_ba
                    self.best_val_loss = val_loss
                    improved = True
                    improvement_reason = "tied_slide_ba_better_loss"
                    
                elif abs(val_loss - self.best_val_loss) <= self.epsilon_L:
                    improvement_reason = "tied_both_metrics_keep_earlier"
        
        return {
            'improved': improved,
            'best_epoch': self.best_epoch_idx,
            'best_slide_ba': self.best_slide_ba,
            'best_val_loss': self.best_val_loss,
            'improvement_reason': improvement_reason,
            'current_slide_ba': slide_ba,
            'current_val_loss': val_loss,
        }
    
    def get_state(self) -> Dict:
        """Get complete state for checkpointing."""
        return {
            'epsilon_A': self.epsilon_A,
            'epsilon_L': self.epsilon_L,
            'best_epoch_idx': self.best_epoch_idx,
            'best_slide_ba': self.best_slide_ba,
            'best_val_loss': self.best_val_loss,
            'slide_ba_history': self.slide_ba_history.copy(),
            'val_loss_history': self.val_loss_history.copy(),
        }
    
    def load_state(self, state: Dict):
        """Restore state from checkpoint."""
        self.epsilon_A = state['epsilon_A']
        self.epsilon_L = state['epsilon_L']
        self.best_epoch_idx = state['best_epoch_idx']
        self.best_slide_ba = state['best_slide_ba']
        self.best_val_loss = state['best_val_loss']
        self.slide_ba_history = state['slide_ba_history'].copy()
        self.val_loss_history = state['val_loss_history'].copy()
    
    def reset(self):
        """Reset tracker to initial state."""
        self.best_epoch_idx = -1
        self.best_slide_ba = None
        self.best_val_loss = None
        self.slide_ba_history = []
        self.val_loss_history = []


class ExponentialMovingAverage:
    """
    Exponential Moving Average tracker for smoothing noisy validation metrics.
    
    Mathematical formulation:
        S_t = α * M_t + (1 - α) * S_{t-1}
    
    Where:
        S_t = Smoothed metric at epoch t
        M_t = Raw metric at epoch t
        α   = Smoothing factor (0 < α ≤ 1)
    """
    
    def __init__(self, alpha: float = 0.3, maximize: bool = True, min_delta: float = 0.0):
        """
        
        Args:
            alpha: Smoothing factor. Lower α → more smoothing
            maximize: If True, track maximum (e.g., accuracy). If False, track minimum 
            min_delta: Minimum change to qualify as improvement 
        """
        if not 0 < alpha <= 1:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        
        if min_delta < 0:
            raise ValueError(f"min_delta must be >= 0, got {min_delta}")
        
        self.alpha = alpha
        self.maximize = maximize
        self.min_delta = min_delta
        
        # Tracking variables
        self.smoothed_value: Optional[float] = None
        self.raw_history: list = []
        self.smoothed_history: list = []
        
        # For early stopping (smoothed metric only)
        self.best_smoothed: Optional[float] = None
        self.best_smoothed_epoch: int = -1
    
    def update(self, raw_value: float, epoch: int) -> dict:
        """
        Update EMA with new raw metric value.
        
        Args:
            raw_value: Raw metric value at current epoch
            epoch: Current epoch number
            
        Returns:
            Dictionary containing:
                - 'smoothed': Current smoothed value
                - 'raw': Current raw value
                - 'improved_smoothed': Whether smoothed value improved beyond best + min_delta
                - 'best_smoothed': Best smoothed value seen
                - 'best_smoothed_epoch': Epoch of best smoothed value
        """
        if self.smoothed_value is None:
            self.smoothed_value = raw_value
            self.best_smoothed = raw_value
            self.best_smoothed_epoch = epoch
        else:
            # EMA update: S_t = α * M_t + (1 - α) * S_{t-1}
            self.smoothed_value = (
                self.alpha * raw_value + 
                (1 - self.alpha) * self.smoothed_value
            )
        
        # Track best SMOOTHED value
        improved_smoothed = False
        if self.maximize:
            if self.smoothed_value > self.best_smoothed + self.min_delta:
                self.best_smoothed = self.smoothed_value
                self.best_smoothed_epoch = epoch
                improved_smoothed = True
        else:
            if self.smoothed_value < self.best_smoothed - self.min_delta:
                self.best_smoothed = self.smoothed_value
                self.best_smoothed_epoch = epoch
                improved_smoothed = True
        
        # Record history
        self.raw_history.append(raw_value)
        self.smoothed_history.append(self.smoothed_value)
        
        return {
            'smoothed': self.smoothed_value,
            'raw': raw_value,
            'improved_smoothed': improved_smoothed,
            'best_smoothed': self.best_smoothed,
            'best_smoothed_epoch': self.best_smoothed_epoch,
        }
    
    def get_state(self) -> dict:
        """Get complete state for checkpointing."""
        return {
            'alpha': self.alpha,
            'maximize': self.maximize,
            'min_delta': self.min_delta,
            'smoothed_value': self.smoothed_value,
            'raw_history': self.raw_history.copy(),
            'smoothed_history': self.smoothed_history.copy(),
            'best_smoothed': self.best_smoothed,
            'best_smoothed_epoch': self.best_smoothed_epoch,
        }
    
    def load_state(self, state: dict):
        """Restore state from checkpoint."""
        self.alpha = state['alpha']
        self.maximize = state['maximize']
        self.min_delta = state['min_delta']
        self.smoothed_value = state['smoothed_value']
        self.raw_history = state['raw_history'].copy()
        self.smoothed_history = state['smoothed_history'].copy()
        self.best_smoothed = state['best_smoothed']
        self.best_smoothed_epoch = state['best_smoothed_epoch']
    
    def reset(self):
        """Reset tracker to initial state."""
        self.smoothed_value = None
        self.raw_history = []
        self.smoothed_history = []
        self.best_smoothed = None
        self.best_smoothed_epoch = -1


class PatienceTracker:
    """
    Patience-based early stopping tracker using smoothed metrics.
    """
    
    def __init__(self, patience: int, min_delta: float = 0.0):
        """
        Initialize patience tracker.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
        """
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        
        if min_delta < 0:
            raise ValueError(f"min_delta must be >= 0, got {min_delta}")
        
        self.patience = patience
        self.min_delta = min_delta
        self.epochs_no_improve = 0
        self.should_stop = False
    
    def update(self, improved_smoothed: bool) -> dict:
        """
        Update patience counter based on smoothed metric improvement.
        
        Args:
            improved_smoothed: Whether smoothed metric improved beyond best + min_delta
            
        Returns:
            Dictionary with patience status
        """
        if improved_smoothed:
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1
        
        self.should_stop = self.epochs_no_improve >= self.patience
        
        return {
            'epochs_no_improve': self.epochs_no_improve,
            'should_stop': self.should_stop,
            'patience_remaining': max(0, self.patience - self.epochs_no_improve),
        }
    
    def get_state(self) -> dict:
        """Get state for checkpointing."""
        return {
            'patience': self.patience,
            'min_delta': self.min_delta,
            'epochs_no_improve': self.epochs_no_improve,
            'should_stop': self.should_stop,
        }
    
    def load_state(self, state: dict):
        """Restore state from checkpoint."""
        self.patience = state['patience']
        self.min_delta = state['min_delta']
        self.epochs_no_improve = state['epochs_no_improve']
        self.should_stop = state['should_stop']
    
    def reset(self):
        """Reset tracker."""
        self.epochs_no_improve = 0
        self.should_stop = False


class WarmupCosineScheduler(optim.lr_scheduler._LRScheduler):
    """
    Learning rate scheduler with linear warmup followed by cosine decay.
    """
    
    def __init__(self, optimizer, warmup_epochs: int, max_epochs: int, 
                 min_lr: float = 1e-6, last_epoch: int = -1):
        if warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be >= 0, got {warmup_epochs}")
        if max_epochs <= 0:
            raise ValueError(f"max_epochs must be > 0, got {max_epochs}")
        if warmup_epochs >= max_epochs:
            warnings.warn(f"warmup_epochs ({warmup_epochs}) >= max_epochs ({max_epochs}). "
                         "Adjusting to have at least 1 cosine decay epoch.")
            warmup_epochs = max(0, max_epochs - 1)
        
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            if self.warmup_epochs == 0:
                return list(self.base_lrs)
            else:
                return [base_lr / self.warmup_epochs for base_lr in self.base_lrs]
        
        current_epoch = self.last_epoch
        
        if current_epoch < self.warmup_epochs:
            alpha = (current_epoch + 1) / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        
        cosine_epochs = self.max_epochs - self.warmup_epochs
        if cosine_epochs > 0:
            progress = (current_epoch - self.warmup_epochs) / cosine_epochs
            progress = min(1.0, max(0.0, progress))
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
        else:
            cosine_decay = 0.0
        
        return [self.min_lr + (base_lr - self.min_lr) * cosine_decay 
               for base_lr in self.base_lrs]


class MixUpTransform:
    """
    MixUp augmentation for batch-level mixing.
    """
    
    def __init__(self, alpha: float = 0.2, prob: float = 0.5):
        self.alpha = alpha
        self.prob = prob
        if self.alpha > 0:
            self.beta_dist = torch.distributions.beta.Beta(
                torch.tensor(self.alpha), torch.tensor(self.alpha)
            )
    
    def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        if torch.rand(1).item() > self.prob:
            return images, labels, labels, 1.0
        
        batch_size = images.size(0)
        
        if self.alpha > 0:
            lam = self.beta_dist.sample().item()
        else:
            lam = 1.0
        
        index = torch.randperm(batch_size, device=images.device)
        mixed_images = lam * images + (1 - lam) * images[index]
        labels_a = labels
        labels_b = labels[index]
        
        return mixed_images, labels_a, labels_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute MixUp loss as weighted combination of two losses."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class NPYImageFolder(datasets.DatasetFolder):
    """Dataset loader for .npy float32 images"""

    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(
            root,
            loader=self._npy_loader,
            extensions=('.npy',),
            transform=transform,
            target_transform=target_transform,
        )
        print(f"NPYImageFolder initialized: {len(self.samples)} samples (grayscale)")

    @staticmethod
    def _npy_loader(path: str) -> torch.Tensor:
        arr = np.load(path)
        if arr.dtype != np.float32:
            warnings.warn(f"Expected float32, got {arr.dtype}. Converting.")
            arr = arr.astype(np.float32)
        if arr.min() < -0.01 or arr.max() > 1.01:
            warnings.warn(
                f"Values outside [0,1]: [{arr.min():.4f}, {arr.max():.4f}], clipping."
            )
            arr = np.clip(arr, 0.0, 1.0)
        return torch.from_numpy(arr).unsqueeze(0)


class TransformSubset(torch.utils.data.Dataset):
    """Wrapper to apply specific transforms to a subset of data."""
    
    def __init__(self, dataset, indices: List[int], transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        img_path, label = self.dataset.samples[actual_idx]
        img_tensor = self.dataset.loader(img_path)
        if self.transform is not None:
            img_tensor = self.transform(img_tensor)
        return img_tensor, label
    
    def __len__(self):
        return len(self.indices)


def create_transforms(mean: List[float], std: List[float],
                     config: Dict, train: bool = True):
    """Create transforms for microscopy images"""
    if train:
        aug = config['augmentation']
        transform_list = []
        
        transform_list.append(
            transforms.RandomResizedCrop(
                config['image_size'],
                scale=aug.get('random_resized_crop_scale', (0.9, 1.0)),
                ratio=aug.get('random_resized_crop_ratio', (0.95, 1.05))
            )
        )
        if aug.get('horizontal_flip_p', 0) > 0:
            transform_list.append(
                transforms.RandomHorizontalFlip(p=aug['horizontal_flip_p'])
            )
        if aug.get('vertical_flip_p', 0) > 0:
            transform_list.append(
                transforms.RandomVerticalFlip(p=aug['vertical_flip_p'])
            )
        if aug.get('rotation_degrees', 0) > 0:
            transform_list.append(
                transforms.RandomRotation(degrees=aug['rotation_degrees'])
            )
        transform_list.append(transforms.Normalize(mean=mean, std=std))
        
        return transforms.Compose(transform_list)
    
    else:
        return transforms.Compose([
            transforms.Normalize(mean=mean, std=std)
        ])


def create_data_loaders(dataset, train_indices: List[int], val_indices: List[int],
                        train_transform, val_transform, batch_size: int,
                        num_workers: int = 2) -> Tuple[DataLoader, DataLoader]:

    persist_flag = num_workers > 0
    
    if train_indices:
        train_dataset = TransformSubset(dataset, train_indices, train_transform)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=persist_flag,
            drop_last=False
        )
    else:
        train_loader = None
    
    if val_indices:
        val_dataset = TransformSubset(dataset, val_indices, val_transform)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=persist_flag,
            drop_last=False
        )
    else:
        val_loader = None
    
    return train_loader, val_loader


def calculate_dataset_stats(dataset, indices: List[int],
                            batch_size: int = 32,
                            num_workers: int = 2) -> Tuple[List[float], List[float]]:
    """Calculate per-channel mean and std for normalization."""
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    channel_sum = None
    channel_sq_sum = None
    num_pixels = 0

    for imgs, _ in loader:
        imgs = imgs.double()
        b, c, h, w = imgs.shape
        if channel_sum is None:
            channel_sum = torch.zeros(c, dtype=torch.float64)
            channel_sq_sum = torch.zeros(c, dtype=torch.float64)
        num_pixels += b * h * w
        channel_sum += imgs.sum(dim=[0, 2, 3])
        channel_sq_sum += (imgs ** 2).sum(dim=[0, 2, 3])

    mean = channel_sum / num_pixels
    var = (channel_sq_sum / num_pixels) - (mean ** 2)
    var = torch.clamp(var, min=1e-8)
    std = torch.sqrt(var)

    return mean.float().tolist(), std.float().tolist()


def calculate_class_weights(
    targets: List[int],
    num_classes: int,
    device: torch.device,
    gamma: float = 0.0
) -> torch.Tensor:
    """Compute adaptive class weights for imbalanced datasets."""
    if gamma == 0.0:
        weights = torch.ones(num_classes, dtype=torch.float32, device=device)
        print(f"Using uniform class weights (gamma=0, recommended with MixUp)")
        return weights

    counts = Counter(targets)
    total = sum(counts.values())

    weights = []
    for c in range(num_classes):
        count = counts.get(c, 1)
        w = (total / (num_classes * count)) ** gamma
        weights.append(w)

    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    if weights.max() > 0:
        weights /= weights.max()

    print(f"Calculated class weights (gamma={gamma}): {[round(w.item(), 3) for w in weights]}")
    return weights


def compute_slide_statistics(
    indices: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    class_names: List[str]
) -> Dict:
    """Compute slide-level statistics."""
    subset_labels = labels[indices]
    subset_groups = groups[indices]
    
    image_counts = Counter(subset_labels)
    n_images = len(indices)
    
    unique_slides = np.unique(subset_groups)
    n_slides = len(unique_slides)
    
    slide_to_class = {}
    for slide_id in unique_slides:
        slide_mask = subset_groups == slide_id
        slide_labels = subset_labels[slide_mask]
        slide_class = Counter(slide_labels).most_common(1)[0][0]
        slide_to_class[slide_id] = slide_class
    
    slide_counts = Counter(slide_to_class.values())
    
    image_class_sizes = [image_counts.get(i, 0) for i in range(len(class_names))]
    slide_class_sizes = [slide_counts.get(i, 0) for i in range(len(class_names))]
    
    nonzero_image_sizes = [s for s in image_class_sizes if s > 0]
    nonzero_slide_sizes = [s for s in slide_class_sizes if s > 0]
    
    image_balance_ratio = (
        min(nonzero_image_sizes) / max(nonzero_image_sizes)
        if nonzero_image_sizes else 0.0
    )
    slide_balance_ratio = (
        min(nonzero_slide_sizes) / max(nonzero_slide_sizes)
        if nonzero_slide_sizes else 0.0
    )
    
    images_per_slide = []
    for slide_id in unique_slides:
        n_imgs = np.sum(subset_groups == slide_id)
        images_per_slide.append(n_imgs)
    
    images_per_class_py = {int(k): int(v) for k, v in image_counts.items()}
    slides_per_class_py = {int(k): int(v) for k, v in slide_counts.items()}

    return {
        "n_images": int(n_images),
        "n_slides": int(n_slides),
        "images_per_class": images_per_class_py,
        "slides_per_class": slides_per_class_py,
        "image_balance_ratio": float(image_balance_ratio),
        "slide_balance_ratio": float(slide_balance_ratio),
        "images_per_slide_stats": {
            "min": int(np.min(images_per_slide)) if images_per_slide else 0,
            "max": int(np.max(images_per_slide)) if images_per_slide else 0,
            "mean": float(np.mean(images_per_slide)) if images_per_slide else 0.0,
            "std": float(np.std(images_per_slide)) if images_per_slide else 0.0,
        }
    }


def validate_split_quality(
    train_stats: Dict,
    val_stats: Dict,
    class_names: List[str],
    min_slides_per_class: int = 2,
    min_slide_balance_ratio: float = 0.4,
    max_image_imbalance_ratio: float = 0.2,
) -> Tuple[bool, Optional[str]]:
    """Validate split quality."""
    val_slide_counts = val_stats['slides_per_class']
    
    for class_idx, class_name in enumerate(class_names):
        n_val_slides = val_slide_counts.get(class_idx, 0)
        if n_val_slides < min_slides_per_class:
            return False, (
                f"INVALID: Validation set has only {n_val_slides} slide(s) of "
                f"'{class_name}' (minimum: {min_slides_per_class}). "
                f"Slide-level metrics will have excessive variance."
            )
    
    val_slide_balance = val_stats['slide_balance_ratio']
    if val_slide_balance < min_slide_balance_ratio:
        return False, (
            f"INVALID: Validation slide balance ratio {val_slide_balance:.3f} "
            f"below threshold {min_slide_balance_ratio:.3f}. "
            f"Distribution: {val_stats['slides_per_class']}"
        )
    
    train_image_balance = train_stats['image_balance_ratio']
    if train_image_balance < max_image_imbalance_ratio:
        warnings.warn(
            f"Training image balance {train_image_balance:.3f} is poor "
            f"(threshold: {max_image_imbalance_ratio:.3f}). "
            f"Consider adjusting class weights. Distribution: {train_stats['images_per_class']}"
        )
    
    return True, None


def create_slide_level_folds(
    image_indices: np.ndarray,
    image_labels: np.ndarray,
    slide_groups: np.ndarray,
    n_splits: int,
    random_state: int,
    class_names: List[str],
    min_slides_per_class: int = 2,
    min_slide_balance_ratio: float = 0.4,
    max_image_imbalance_ratio: float = 0.2,
    max_retries: int = 50,
    verbose: bool = True
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Dict]:
    """Create slide-level stratified folds with quality validation."""
    
    subset_labels = image_labels[image_indices]
    subset_groups = slide_groups[image_indices]
    
    unique_slides = np.unique(subset_groups)
    slide_to_label = {}
    slide_to_indices = {}
    
    for slide_id in unique_slides:
        slide_mask = subset_groups == slide_id
        slide_labels = subset_labels[slide_mask]
        
        unique_labels = np.unique(slide_labels)
        if len(unique_labels) > 1:
            raise ValueError(
                f"Data integrity error: Slide '{slide_id}' has conflicting labels: "
                f"{unique_labels}. All patches from a slide must have the same label."
            )
        
        slide_to_label[slide_id] = int(slide_labels[0])
        slide_to_indices[slide_id] = image_indices[slide_mask]
    
    slides_array = np.array(list(slide_to_label.keys()))
    slide_labels_array = np.array([slide_to_label[s] for s in slides_array])
    
    for retry_attempt in range(max_retries):
        if retry_attempt == 0:
            current_seed = random_state
        else:
            ss = SeedSequence(random_state, spawn_key=(retry_attempt,))
            current_seed = int(ss.generate_state(1)[0])
        
        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=current_seed
        )
        
        slide_fold_splits = list(skf.split(slides_array, slide_labels_array))
        
        all_folds_valid = True
        image_fold_splits = []
        fold_metadata = []
        
        for fold_idx, (train_slide_idx, val_slide_idx) in enumerate(slide_fold_splits):

            train_slides = slides_array[train_slide_idx]
            val_slides = slides_array[val_slide_idx]
            
            train_image_idx = np.concatenate([
                slide_to_indices[slide_id] for slide_id in train_slides
            ])
            val_image_idx = np.concatenate([
                slide_to_indices[slide_id] for slide_id in val_slides
            ])
            
            train_stats = compute_slide_statistics(
                train_image_idx, image_labels, slide_groups, class_names
            )
            val_stats = compute_slide_statistics(
                val_image_idx, image_labels, slide_groups, class_names
            )
            
            is_valid, error_msg = validate_split_quality(
                train_stats, val_stats, class_names,
                min_slides_per_class, min_slide_balance_ratio, max_image_imbalance_ratio
            )
            
            if not is_valid:
                if verbose and retry_attempt == 0:
                    print(f"\n  Fold {fold_idx + 1}: {error_msg}")
                    print(f"  Attempting retry with perturbed seed...")
                elif verbose and retry_attempt < 5:
                    print(f"  Retry {retry_attempt}, Fold {fold_idx + 1}: Still invalid")
                
                all_folds_valid = False
                break
            
            image_fold_splits.append((train_image_idx, val_image_idx))
            fold_metadata.append({
                'fold_idx': fold_idx,
                'train_stats': train_stats,
                'val_stats': val_stats
            })
        
        if all_folds_valid:
            if verbose:
                if retry_attempt > 0:
                    print(f"\n✓ Valid splits found after {retry_attempt} "
                          f"{'retry' if retry_attempt == 1 else 'retries'} "
                          f"(effective_seed={current_seed})")
                else:
                    print(f"\n✓ Valid splits found on first attempt")
            
            metadata = {
                'random_state_effective': current_seed,
                'retry_count': retry_attempt,
                'n_folds': n_splits,
                'fold_metadata': fold_metadata
            }
            
            return image_fold_splits, metadata
    
    raise RuntimeError(
        f"Failed to generate valid slide-level stratified splits after {max_retries} attempts"
    )


def compute_slide_level_metrics(
    y_true_patches: np.ndarray,
    y_pred_patches: np.ndarray,
    y_proba_patches: np.ndarray,
    groups: np.ndarray,
    num_classes: int,
    verbose: bool = False
) -> Dict[str, float]:
    """Compute slide-level performance metrics via probability aggregation."""
    y_true_patches = np.asarray(y_true_patches)
    y_pred_patches = np.asarray(y_pred_patches)
    y_proba_patches = np.asarray(y_proba_patches)
    groups = np.asarray(groups)
    
    if not (len(y_true_patches) == len(y_pred_patches) == 
            len(y_proba_patches) == len(groups)):
        raise ValueError(
            f"Shape mismatch: y_true={len(y_true_patches)}, "
            f"y_pred={len(y_pred_patches)}, y_proba={len(y_proba_patches)}, "
            f"groups={len(groups)}"
        )
    
    if y_proba_patches.ndim != 2 or y_proba_patches.shape[1] != num_classes:
        raise ValueError(
            f"y_proba must be shape (N, {num_classes}), "
            f"got {y_proba_patches.shape}"
        )
    
    try:
        slide_true, slide_pred, slide_proba, unique_slides = aggregate_predictions_by_group(
            y_true=y_true_patches,
            y_pred=y_pred_patches,
            y_proba=y_proba_patches,
            groups=groups,
            num_classes=num_classes
        )
    except Exception as e:
        warnings.warn(f"Failed to aggregate to slide level: {e}")
        return {
            'slide_bal_acc': 0.0,
            'slide_mcc': 0.0,
            'slide_f1': 0.0,
            'slide_auc': 0.0,
            'n_slides': 0
        }
    
    n_slides = len(slide_true)
    
    if n_slides < 2:
        warnings.warn(f"Only {n_slides} slide(s) in validation set - metrics unreliable")
        return {
            'slide_bal_acc': 0.0,
            'slide_mcc': 0.0,
            'slide_f1': 0.0,
            'slide_auc': 0.0,
            'n_slides': n_slides
        }
    
    metrics = {}
    
    try:
        metrics['slide_bal_acc'] = balanced_accuracy_score(slide_true, slide_pred)
    except Exception as e:
        warnings.warn(f"Could not compute slide balanced accuracy: {e}")
        metrics['slide_bal_acc'] = 0.0
    
    try:
        metrics['slide_mcc'] = matthews_corrcoef(slide_true, slide_pred)
    except Exception as e:
        warnings.warn(f"Could not compute slide MCC: {e}")
        metrics['slide_mcc'] = 0.0
    
    try:
        metrics['slide_f1'] = f1_score(
            slide_true, slide_pred, average='weighted', zero_division=0
        )
    except Exception as e:
        warnings.warn(f"Could not compute slide F1: {e}")
        metrics['slide_f1'] = 0.0
    
    try:
        metrics['slide_auc'] = roc_auc_score(slide_true, slide_proba[:, 1])
    except Exception as e:
        warnings.warn(f"Could not compute slide AUC: {e}")
        metrics['slide_auc'] = 0.0
    
    metrics['n_slides'] = n_slides
    
    if verbose:
        print(f"\n  Slide-Level Metrics ({n_slides} slides):")
        print(f"    Balanced Acc: {metrics['slide_bal_acc']:.4f}")
        print(f"    MCC:          {metrics['slide_mcc']:.4f}")
        print(f"    F1 (weighted): {metrics['slide_f1']:.4f}")
        print(f"    AUC:          {metrics['slide_auc']:.4f}")
    
    return metrics


def aggregate_predictions_by_group(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    groups: np.ndarray,
    num_classes: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate patch-level predictions to slide-level using probability averaging."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba)
    groups = np.asarray(groups)

    if y_proba.ndim != 2:
        raise ValueError(f"y_proba must be 2D (N, C), got shape {y_proba.shape}")

    if not (len(y_true) == len(y_pred) == len(y_proba) == len(groups)):
        raise ValueError("All inputs must have the same length")

    unique_groups = np.unique(groups)
    slide_true = []
    slide_pred = []
    slide_proba = []

    for g in unique_groups:
        mask = groups == g
        g_true = y_true[mask]
        g_proba = y_proba[mask]

        if g_true.size == 0:
            continue

        unique_labels = np.unique(g_true)
        if len(unique_labels) > 1:
            raise ValueError(
                f"Data quality error: Slide '{g}' has conflicting patch labels: {unique_labels}."
            )
        
        true_label = int(g_true[0])
        slide_true.append(true_label)

        mean_proba = g_proba.mean(axis=0)
        slide_proba.append(mean_proba)
        slide_pred.append(int(mean_proba.argmax()))

    if not slide_true:
        raise RuntimeError("No groups found when aggregating predictions by group.")

    slide_true = np.asarray(slide_true, dtype=int)
    slide_pred = np.asarray(slide_pred, dtype=int)
    slide_proba = np.vstack(slide_proba)

    return slide_true, slide_pred, slide_proba, unique_groups


def grouped_bootstrap_ci(
    metric_fn: Callable,
    metric_args: List[np.ndarray],
    groups: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
    verbose: bool = False,
) -> Tuple[float, float]:
    """Cluster-robust bootstrap confidence interval using pairs cluster bootstrap."""
    if not metric_args:
        raise ValueError("metric_args list cannot be empty.")

    try:
        metric_args = [np.asarray(arg) for arg in metric_args]
    except Exception as e:
        raise ValueError(f"Could not convert metric_args to numpy arrays: {e}")

    n_samples = len(metric_args[0])
    if n_samples == 0:
        warnings.warn("Empty arrays for grouped bootstrap CI. Returning (0.0, 0.0).")
        return (0.0, 0.0)

    if not all(len(arg) == n_samples for arg in metric_args):
        arg_lengths = [len(arg) for arg in metric_args]
        raise ValueError(
            f"All arrays in metric_args must have same length. Got: {arg_lengths}"
        )

    groups = np.asarray(groups)
    if len(groups) != n_samples:
        raise ValueError(
            f"'groups' length ({len(groups)}) must match n_samples ({n_samples})."
        )

    y_true_orig = metric_args[0]
    if len(np.unique(y_true_orig)) < 2:
        warnings.warn(
            "Only one class in original data. CI calculation unreliable. "
            "Returning (0.0, 0.0)."
        )
        return (0.0, 0.0)

    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    
    if n_groups < 2:
        warnings.warn(
            "Fewer than 2 unique groups. Grouped bootstrap not meaningful. "
            "Returning (0.0, 0.0)."
        )
        return (0.0, 0.0)

    group_to_indices = {g: np.where(groups == g)[0] for g in unique_groups}

    rng = np.random.RandomState(seed)
    scores = []
    failed_count = 0
    warn_once = True

    for b in range(n_bootstrap):
        try:
            sampled_groups = rng.choice(unique_groups, size=n_groups, replace=True)
            boot_indices = np.concatenate([group_to_indices[g] for g in sampled_groups])
            boot_args = [arg[boot_indices] for arg in metric_args]

            if len(np.unique(boot_args[0])) < 2:
                if verbose and warn_once:
                    warnings.warn(
                        "Some bootstrap samples have only one class. Skipping."
                    )
                    warn_once = False
                failed_count += 1
                continue

            score = metric_fn(*boot_args)
            
            if np.isnan(score) or np.isinf(score):
                if verbose and warn_once:
                    warnings.warn("Bootstrap sample produced NaN/Inf. Skipping.")
                    warn_once = False
                failed_count += 1
                continue

            scores.append(score)

        except Exception as e:
            if verbose and warn_once:
                warnings.warn(f"Bootstrap sample failed: {e}")
                warn_once = False
            failed_count += 1
            continue

    failure_rate = failed_count / n_bootstrap
    if failure_rate > 0.05 and failure_rate < 1.0:
        warnings.warn(
            f"HIGH BOOTSTRAP FAILURE RATE: {failure_rate*100:.1f}% "
            f"({failed_count}/{n_bootstrap} samples failed)."
        )

    if len(scores) == 0:
        warnings.warn(
            f"All {n_bootstrap} bootstrap samples failed! "
            "Returning baseline metric as CI."
        )
        try:
            baseline_score = metric_fn(*metric_args)
            return (baseline_score, baseline_score)
        except Exception:
            return (0.0, 0.0)

    scores = np.array(scores)
    alpha = (1 - ci) / 2
    lower = np.percentile(scores, alpha * 100)
    upper = np.percentile(scores, (1 - alpha) * 100)

    lower_clamped = max(-1.0, min(1.0, lower)) if not np.isnan(lower) else 0.0
    upper_clamped = max(-1.0, min(1.0, upper)) if not np.isnan(upper) else 1.0

    if (lower_clamped != lower or upper_clamped != upper) and verbose:
        warnings.warn(
            f"CI bounds [{lower:.4f}, {upper:.4f}] clamped to "
            f"[{lower_clamped:.4f}, {upper_clamped:.4f}]"
        )

    return float(lower_clamped), float(upper_clamped)


def print_data_distribution(targets: List[int], class_names: List[str], 
                           split_name: str = "Dataset"):
    """Print class distribution statistics for quality control."""
    if not targets:
        print(f"{split_name}: No samples")
        return
    
    class_counts = Counter(targets)
    total = len(targets)
    
    print(f"\n{split_name} Distribution:")
    print(f"  Total samples: {total}")
    for i, class_name in enumerate(class_names):
        count = class_counts.get(i, 0)
        percentage = (count / total) * 100 if total > 0 else 0
        print(f"  {class_name}: {count} ({percentage:.1f}%)")


def verify_data_splits(train_idx: List[int], val_idx: List[int], 
                      test_idx: List[int]) -> bool:
    """Verify that data splits have no overlap."""
    train_set = set(train_idx) if train_idx else set()
    val_set = set(val_idx) if val_idx else set()
    test_set = set(test_idx) if test_idx else set()
    
    assert len(train_set & val_set) == 0, "Train/Val overlap detected!"
    assert len(train_set & test_set) == 0, "Train/Test overlap detected!"
    assert len(val_set & test_set) == 0, "Val/Test overlap detected!"
    
    print("✓ Data splits verified: No overlap detected")
    return True


def get_device_info() -> torch.device:
    """Get device and print information."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice Information:")
    print(f"  Device: {device}")
    
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"  CUDA Version: {torch.version.cuda}")
    else:
        print("  Running on CPU")
    
    return device