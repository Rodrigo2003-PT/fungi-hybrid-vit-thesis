"""
Training Logic

Author: Rodrigo Sá
Date: 2026
"""

import copy
import gc
import math
import random
import warnings
from timeit import default_timer as timer
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.random import SeedSequence
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from torch.utils.data import DataLoader

# ------------------------------------------------------------------
# Model imports
# ------------------------------------------------------------------
from lsa_convtok_vit import (
    SimpleViT_LSA_ConvTok,
    compute_arch_hash as _lsa_convtok_arch_hash,
)

# LSALogger operates on any model containing LSAAttention layers.
from lsa_logger import LSALogger

from training_utils import (
    ExponentialMovingAverage,
    LexBestModelTracker,
    MixUpTransform,
    PatienceTracker,
    WarmupCosineScheduler,
    compute_slide_level_metrics,
)


# ---------------------------------------------------------------------------
# Arch hash dispatcher
# ---------------------------------------------------------------------------

def compute_arch_hash(config: Dict[str, Any]) -> str:
    """
    Dispatch to the correct ``compute_arch_hash`` for the current
    experiment family so that checkpoint identity validation is
    always family-correct.

    Parameters
    ----------
    config : dict — must contain ``'experiment_name'``.

    Returns
    -------
    str — 16-char SHA-256 prefix.
    """
    exp = config.get('experiment_name', '')
    if exp.startswith('lsa_convtok'):
        return _lsa_convtok_arch_hash(config)
    else:
        raise ValueError(
            f"compute_arch_hash: unrecognised experiment_name '{exp}'. "
            "Add a dispatch branch for new experiment families."
        )


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(
    config: Dict[str, Any],
    num_classes: int,
    device: torch.device,
) -> nn.Module:
    """
    Instantiate the correct model for the given experiment and move it
    to ``device``.

    Parameters
    ----------
    config      : complete config dict (from ``get_config``).
    num_classes : int
    device      : torch.device

    Returns
    -------
    nn.Module   — model on ``device``, ready for training.

    Raises
    ------
    ValueError  — if ``experiment_name`` is not recognised.
    """
    exp      = config.get('experiment_name', '')
    mc       = config['model_config']
    dim_head = mc.get('dim_head', 64)
    pe_temp  = mc.get('pe_temperature', 10000)

    if exp.startswith('lsa_convtok'):
        model = SimpleViT_LSA_ConvTok(
            image_size=config['image_size'],
            num_classes=num_classes,
            dim=mc['dim'],
            depth=mc['depth'],
            heads=mc['heads'],
            mlp_dim=mc['mlp_dim'],
            channels=config['channels'],
            dim_head=dim_head,
            convtok_hidden_channels=config.get('convtok_hidden_channels', 64),
            convtok_blocks=config.get('convtok_blocks'),
            convtok_match_baseline_tokens=config.get(
                'convtok_match_baseline_tokens', True
            ),
            convtok_expected_hw=config.get('convtok_expected_hw', [24, 24]),
            convtok_conv_bias=config.get('convtok_conv_bias', False),
            init_policy=config.get('init_policy', 'default'),
            pe_temperature=pe_temp,
        ).to(device)

    else:
        raise ValueError(
            f"build_model: unrecognised experiment_name '{exp}'. "
            f"Valid prefixes: 'convtok', 'lsa_convtok'."
        )

    return model


# ---------------------------------------------------------------------------
# Arch metadata
# ---------------------------------------------------------------------------

def create_model_arch_metadata(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a self-describing architecture metadata dict embedded in
    checkpoints and fold histories.  Fields are experiment-family-aware.

    Parameters
    ----------
    config : complete config dict.

    Returns
    -------
    dict — JSON-serialisable; stored under ``'arch_metadata'`` key.
    """
    exp = config.get('experiment_name', '')
    mc  = config.get('model_config', {})
    dh  = mc.get('dim_head', 64)

    meta: Dict[str, Any] = {
        # Identity
        'experiment_name':         exp,
        'model_class':             (
            'SimpleViT_LSA_ConvTok' if exp.startswith('lsa_convtok')
            else 'SimpleViT_ConvTok'
        ),
        'attention_type':          (
            'lsa_diagonal' if exp.startswith('lsa_convtok')
            else 'global'
        ),
        # Tokenizer
        'convtok_blocks':          config.get('convtok_blocks'),
        'convtok_hidden_channels': config.get('convtok_hidden_channels', 64),
        'convtok_post_ln':         config.get('convtok_post_ln', True),
        'convtok_expected_hw':     config.get('convtok_expected_hw'),
        'convtok_expected_tokens': config.get('convtok_expected_tokens'),
        'convtok_conv_bias':       config.get('convtok_conv_bias', False),
        # Transformer
        'dim':                     mc.get('dim'),
        'depth':                   mc.get('depth'),
        'heads':                   mc.get('heads'),
        'mlp_dim':                 mc.get('mlp_dim'),
        'dim_head':                dh,
        'pe_temperature':          mc.get('pe_temperature', 10000),
        # Input
        'image_size':              config.get('image_size'),
        'channels':                config.get('channels'),
        # Initialisation
        'init_policy':             config.get('init_policy', 'default'),
        # Arch hash
        'arch_hash':               compute_arch_hash(config),
    }

    # LSA-specific fields
    if exp.startswith('lsa_convtok'):
        depth = mc.get('depth', 8)
        meta['lsa_temperature_init']    = float(math.log(dh ** -0.5))
        meta['lsa_n_temperature_params'] = depth
        meta['lsa_temperature_policy']   = (
            "excluded_from_policy_b"
            if config.get('init_policy') == 'explicit_kaiming'
            else "default_log_scale"
        )

    return meta


def validate_checkpoint_arch(
    checkpoint: Dict[str, Any],
    current_config: Dict[str, Any],
    strict: bool = True,
) -> bool:
    """
    Verify that a loaded checkpoint was produced by the same architecture
    as the current config.

    Parameters
    ----------
    checkpoint     : loaded checkpoint dict.
    current_config : config dict for the current run.
    strict         : if True, raise RuntimeError on mismatch; else warn.

    Returns
    -------
    bool — True if hashes match.
    """
    stored_meta = checkpoint.get('arch_metadata')
    if stored_meta is None:
        warnings.warn(
            "Checkpoint has no 'arch_metadata'. Skipping architecture "
            "validation.  This checkpoint predates the arch-hash system."
        )
        return True

    current_hash = compute_arch_hash(current_config)
    stored_hash  = stored_meta.get('arch_hash')

    if current_hash != stored_hash:
        msg = (
            f"Architecture mismatch between checkpoint and current config!\n"
            f"  Checkpoint model_class : {stored_meta.get('model_class', '?')}\n"
            f"  Checkpoint arch_hash   : {stored_hash}\n"
            f"  Current    arch_hash   : {current_hash}\n"
            f"  Current experiment     : {current_config.get('experiment_name')}\n"
        )
        if strict:
            raise RuntimeError(msg)
        else:
            warnings.warn(msg)
            return False

    return True


# ---------------------------------------------------------------------------
# ModelTrainer
# ---------------------------------------------------------------------------

class ModelTrainer:
    """
    Unified trainer for all CTT and LSA-ConvTok ablation experiments.

    Dispatches model instantiation via ``build_model`` and activates
    LSA-specific diagnostics automatically when the constructed model
    contains LSAAttention layers.

    Parameters
    ----------
    config             : complete config dict from ``get_config``.
    device             : torch.device
    checkpoint_manager : CheckpointManager instance
    num_classes        : int
    """

    def __init__(
        self,
        config: Dict[str, Any],
        device: torch.device,
        checkpoint_manager,
        num_classes: int,
    ):
        self.config             = config
        self.device             = device
        self.checkpoint_manager = checkpoint_manager
        self.num_classes        = num_classes

        # AMP setup
        self.use_amp = config.get('use_amp', False) and torch.cuda.is_available()
        if config.get('use_amp', False) and not torch.cuda.is_available():
            warnings.warn("AMP requested but CUDA not available. Disabling AMP.")

        if self.use_amp:
            from torch.amp import autocast, GradScaler
            self.autocast  = lambda: autocast(device_type='cuda')
            self.GradScaler = lambda: GradScaler('cuda')
        else:
            self.autocast   = None
            self.GradScaler = None

        # MixUp setup
        aug = config.get('augmentation', {})
        self.use_mixup = aug.get('use_mixup', True)
        if self.use_mixup:
            self.mixup_transform = MixUpTransform(
                alpha=aug.get('mixup_alpha', 0.2),
                prob=aug.get('mixup_prob', 0.5),
            )
            print(f"MixUp enabled: alpha={aug.get('mixup_alpha')}, "
                  f"prob={aug.get('mixup_prob')}")
        else:
            self.mixup_transform = None
            print("MixUp disabled")

    # ------------------------------------------------------------------
    # Model creation
    # ------------------------------------------------------------------

    def create_model(self, model_seed: int) -> nn.Module:
        """
        Set seed, call ``build_model``, print parameter counts and arch hash.
        """
        self._set_seeds(model_seed)

        mc        = self.config['model_config']
        exp       = self.config.get('experiment_name', '')
        init_pol  = self.config.get('init_policy', 'default')

        print(f"  Model config:")
        print(f"    experiment       : {exp}")
        print(f"    init_policy      : {init_pol}")
        print(f"    dim_head         : {mc.get('dim_head', 64)}")
        print(f"    pe_temperature   : {mc.get('pe_temperature', 10000)}")
        print(f"    convtok_hidden_ch: {self.config.get('convtok_hidden_channels', 64)}")
        print(f"    arch_hash        : {compute_arch_hash(self.config)}")

        model = build_model(self.config, self.num_classes, self.device)

        if exp.startswith('lsa_convtok'):
            temp_params = [
                n for n, _ in model.named_parameters() if 'temperature' in n
            ]
            depth = mc.get('depth', 8)
            if len(temp_params) != depth:
                warnings.warn(
                    f"Expected {depth} temperature parameters, found "
                    f"{len(temp_params)}.  Check LSAAttention layer count."
                )
            else:
                print(f"  LSA temperatures: {len(temp_params)} params "
                      f"(τ₀ = {math.log(mc.get('dim_head', 64) ** -0.5):.4f})")

        return model

    def create_optimizer_and_scheduler(
        self,
        model: nn.Module,
        max_epochs: int,
    ) -> Tuple[optim.Optimizer, Any, Optional[Any]]:
        """AdamW + WarmupCosineScheduler + optional AMP scaler."""
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
        )
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=self.config['warmup_epochs'],
            max_epochs=max_epochs,
        )
        scaler = self.GradScaler() if self.use_amp else None
        return optimizer, scheduler, scaler

    # ------------------------------------------------------------------
    # CV fold training
    # ------------------------------------------------------------------

    def train_cv_fold(
        self,
        iter_idx: int,
        fold_idx: int,
        iteration_seed: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        fold_mean: List[float],
        fold_std: List[float],
        class_weights: torch.Tensor,
        val_groups_fold: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Train a single cross-validation fold.

        Parameters
        ----------
        val_groups_fold : np.ndarray, required
            Slide IDs for each validation patch, used for slide-level
            metric computation and early stopping.
        """
        if val_groups_fold is None:
            raise ValueError(
                "val_groups_fold is REQUIRED for slide-level early stopping."
            )

        exp = self.config.get('experiment_name', '')
        print(f"\n{'='*70}")
        print(
            f"[{exp.upper()}] "
            f"ITERATION {iter_idx+1}/{self.config['n_iterations']} | "
            f"FOLD {fold_idx+1}/{self.config['n_folds']}"
        )
        print(f"Iteration seed: {iteration_seed}")
        print(
            f"Validation: ~{len(np.unique(val_groups_fold))} slides, "
            f"{len(val_groups_fold)} patches"
        )

        # Hyperparameters
        ema_alpha = self.config.get('ema_alpha', 0.3)
        min_delta = self.config.get('min_delta', 0.0)
        epsilon_A = self.config.get('epsilon_slide_ba', 1e-6)
        epsilon_L = self.config.get('epsilon_val_loss',  1e-6)

        # ------------------------------------------------------------------
        # Checkpoint / history resume
        # ------------------------------------------------------------------
        history    = self.checkpoint_manager.load_fold_history(
            iter_idx, fold_idx, verbose=False
        )
        checkpoint = self._load_fold_checkpoint(iter_idx, fold_idx)

        if history is not None and checkpoint is None:
            print(
                f"\n  WARNING: Fold {fold_idx+1} already completed "
                "(history exists, checkpoint cleaned). Returning cached summary."
            )
            return self._create_fold_summary_from_history(
                history, iter_idx, fold_idx, iteration_seed
            )

        # ------------------------------------------------------------------
        # Initialise or restore training state
        # ------------------------------------------------------------------
        if checkpoint is None:
            fold_results     = self._initialize_fold_results()
            start_epoch      = 0
            best_model_state = None

            lexicographic_tracker = LexBestModelTracker(
                epsilon_A=epsilon_A, epsilon_L=epsilon_L
            )
            ema_tracker = ExponentialMovingAverage(
                alpha=ema_alpha, maximize=True, min_delta=min_delta
            )
            patience_tracker = PatienceTracker(
                patience=self.config['patience'], min_delta=min_delta
            )
            print("Starting fold from scratch")

        else:
            validate_checkpoint_arch(checkpoint, self.config, strict=True)
            fold_results     = checkpoint['fold_results']
            start_epoch      = checkpoint['epoch'] + 1
            best_model_state = checkpoint.get('best_model_state_dict', None)

            lexicographic_tracker = LexBestModelTracker(
                epsilon_A=epsilon_A, epsilon_L=epsilon_L
            )
            ema_tracker = ExponentialMovingAverage(
                alpha=ema_alpha, maximize=True, min_delta=min_delta
            )
            patience_tracker = PatienceTracker(
                patience=self.config['patience'], min_delta=min_delta
            )
            if checkpoint.get('lexicographic_state'):
                lexicographic_tracker.load_state(checkpoint['lexicographic_state'])
            if checkpoint.get('ema_state'):
                ema_tracker.load_state(checkpoint['ema_state'])
            if checkpoint.get('patience_state'):
                patience_tracker.load_state(checkpoint['patience_state'])

            print(
                f"Resuming from epoch {checkpoint['epoch']} | "
                f"best epoch: {lexicographic_tracker.best_epoch_idx+1}"
            )

        # ------------------------------------------------------------------
        # Model, optimiser, loss
        # ------------------------------------------------------------------
        fold_seed_seq = SeedSequence(iteration_seed).spawn(self.config['n_folds'])
        model_seed    = int(fold_seed_seq[fold_idx].generate_state(1)[0])
        print(f"Model init seed: {model_seed}")

        model                        = self.create_model(model_seed)
        optimizer, scheduler, scaler = self.create_optimizer_and_scheduler(
            model, self.config['num_epochs']
        )
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        if checkpoint:
            self.checkpoint_manager.restore_training_state(
                checkpoint, model, optimizer, scheduler, scaler
            )

        arch_meta = create_model_arch_metadata(self.config)

        # ------------------------------------------------------------------
        # LSA diagnostic logger
        # ------------------------------------------------------------------
        lsa_logger: Optional[LSALogger] = None
        if LSALogger.is_lsa_model(model):
            lsa_logger = LSALogger(
                model=model,
                fold_results=fold_results,
                iter_idx=iter_idx,
                fold_idx=fold_idx,
                tau_warn_threshold=3.0,
            )
            print(
                f"  LSALogger active: {lsa_logger._n_layers} attention layers tracked"
            )

        # ------------------------------------------------------------------
        # Training loop
        # ------------------------------------------------------------------
        fold_start = timer()
        epoch      = start_epoch - 1

        for epoch in range(start_epoch, self.config['num_epochs']):

            train_loss, train_hard_acc, train_soft_acc = self._train_epoch(
                model, train_loader, loss_fn, optimizer, scaler
            )

            # Register entropy hooks before validation
            if lsa_logger is not None:
                lsa_logger.start_validation_epoch()

            val_loss, val_acc, val_patch_metrics, val_slide_metrics = (
                self._validate_epoch(model, val_loader, loss_fn, val_groups_fold)
            )

            # Collect temperatures / entropy, remove hooks
            if lsa_logger is not None:
                lsa_logger.end_validation_epoch(
                    epoch=epoch,
                    verbose=(epoch % 10 == 0 or epoch == 0),
                )

            scheduler.step()
            current_lr        = scheduler.get_last_lr()[0]
            current_slide_ba  = val_slide_metrics['slide_bal_acc']

            lexico_update   = lexicographic_tracker.update(
                current_epoch=epoch,
                slide_ba=current_slide_ba,
                val_loss=val_loss,
            )
            ema_update      = ema_tracker.update(current_slide_ba, epoch)
            patience_update = patience_tracker.update(ema_update['improved_smoothed'])

            if lexico_update['improved']:
                best_model_state = copy.deepcopy(model.state_dict())
                print(
                    f"\n  BEST MODEL epoch {epoch+1} "
                    f"({lexico_update['improvement_reason']}): "
                    f"slide_BA={current_slide_ba:.4f} | val_loss={val_loss:.4f}"
                )

            print(
                f"Iter {iter_idx+1}, Fold {fold_idx+1}, "
                f"Epoch {epoch+1}/{self.config['num_epochs']} | "
                f"train_loss={train_loss:.4f} | train_acc={train_hard_acc:.4f}"
            )
            print(
                f"  PATCH : val_loss={val_loss:.4f}, "
                f"bal_acc={val_patch_metrics['bal_acc']:.4f}"
            )
            print(
                f"  SLIDE : BA={current_slide_ba:.4f}, "
                f"EMA={ema_update['smoothed']:.4f} | "
                f"best BA={lexico_update['best_slide_ba']:.4f} "
                f"(epoch {lexico_update['best_epoch']+1})"
            )
            print(
                f"  Patience: {patience_update['epochs_no_improve']}/"
                f"{self.config['patience']} | LR={current_lr:.6f}"
            )

            # Record metrics
            fold_results['train_loss'].append(train_loss)
            fold_results['train_acc'].append(train_hard_acc)
            fold_results['train_acc_soft'].append(train_soft_acc)
            fold_results['val_loss'].append(val_loss)
            fold_results['val_acc'].append(val_acc)
            fold_results['val_bal_acc'].append(val_patch_metrics['bal_acc'])
            fold_results['val_mcc'].append(val_patch_metrics['mcc'])
            fold_results['val_f1'].append(val_patch_metrics['f1'])
            fold_results['val_auc'].append(val_patch_metrics['val_auc'])
            fold_results['val_slide_bal_acc'].append(current_slide_ba)
            fold_results['val_slide_bal_acc_smoothed'].append(
                ema_update['smoothed']
            )
            fold_results['val_slide_mcc'].append(val_slide_metrics['slide_mcc'])
            fold_results['val_slide_f1'].append(val_slide_metrics['slide_f1'])
            fold_results['val_slide_auc'].append(val_slide_metrics['slide_auc'])

            # Periodic checkpoint
            should_save = (
                (epoch + 1) % self.config.get('checkpoint_every_n_epochs', 25) == 0
                or patience_tracker.should_stop
                or (epoch + 1) >= self.config['num_epochs']
            )
            if should_save:
                ckpt_ok, hist_ok = self.checkpoint_manager.create_cv_checkpoint(
                    iter_idx=iter_idx,
                    fold_idx=fold_idx,
                    current_epoch=epoch,
                    best_epoch=lexicographic_tracker.best_epoch_idx,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    fold_results=fold_results,
                    best_slide_metric_raw=lexicographic_tracker.best_slide_ba,
                    epochs_no_improve=patience_tracker.epochs_no_improve,
                    fold_mean=fold_mean,
                    fold_std=fold_std,
                    scaler=scaler,
                    best_model_state=best_model_state,
                    lexicographic_state=lexicographic_tracker.get_state(),
                    ema_state=ema_tracker.get_state(),
                    patience_state=patience_tracker.get_state(),
                    best_val_loss=lexicographic_tracker.best_val_loss,
                    arch_metadata=arch_meta,
                )
                if not ckpt_ok:
                    raise RuntimeError(
                        f"CRITICAL: checkpoint save failed at epoch {epoch+1}. "
                        "Training cannot safely continue without backup."
                    )
                if not hist_ok:
                    warnings.warn(
                        f"Fold history save failed at epoch {epoch+1}."
                    )

            if patience_tracker.should_stop:
                print(
                    f"\n  EARLY STOPPING at epoch {epoch+1} | "
                    f"best epoch={lexicographic_tracker.best_epoch_idx+1}, "
                    f"slide_BA={lexicographic_tracker.best_slide_ba:.4f}"
                )
                break

        fold_end      = timer()
        training_time = fold_end - fold_start

        # ------------------------------------------------------------------
        # Restore best weights
        # ------------------------------------------------------------------
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            best_epoch_idx = lexicographic_tracker.best_epoch_idx
            print(
                f"\n  RESTORED best weights from epoch {best_epoch_idx+1} | "
                f"slide_BA={lexicographic_tracker.best_slide_ba:.4f} | "
                f"val_loss={lexicographic_tracker.best_val_loss:.4f}"
            )
        else:
            warnings.warn("No best model state found — using final epoch weights.")
            best_epoch_idx = epoch

        # LSA fold summary
        if lsa_logger is not None:
            lsa_logger.log_fold_summary(best_epoch_idx)

        # Final checkpoint (persists best weights and complete history)
        ckpt_ok, hist_ok = self.checkpoint_manager.create_cv_checkpoint(
            iter_idx=iter_idx,
            fold_idx=fold_idx,
            current_epoch=epoch,
            best_epoch=lexicographic_tracker.best_epoch_idx,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            fold_results=fold_results,
            best_slide_metric_raw=lexicographic_tracker.best_slide_ba,
            epochs_no_improve=patience_tracker.epochs_no_improve,
            fold_mean=fold_mean,
            fold_std=fold_std,
            scaler=scaler,
            best_model_state=best_model_state,
            lexicographic_state=lexicographic_tracker.get_state(),
            ema_state=ema_tracker.get_state(),
            patience_state=patience_tracker.get_state(),
            best_val_loss=lexicographic_tracker.best_val_loss,
            arch_metadata=arch_meta,
        )
        if not ckpt_ok or not hist_ok:
            warnings.warn("Failed to save final fold checkpoint or history.")

        fold_summary = self._create_fold_summary(
            iter_idx=iter_idx,
            fold_idx=fold_idx,
            iteration_seed=iteration_seed,
            model_seed=model_seed,
            best_epoch_idx=best_epoch_idx,
            fold_results=fold_results,
            lexicographic_tracker=lexicographic_tracker,
            ema_tracker=ema_tracker,
            min_delta=min_delta,
            training_time=training_time,
            fold_mean=fold_mean,
            fold_std=fold_std,
        )

        # Augment with LSA fields (no-op for CTT experiments)
        fold_summary['lsa_best_epoch_temperatures'] = fold_results.get(
            'lsa_best_epoch_temperatures', []
        )
        fold_summary['lsa_best_epoch_entropy'] = fold_results.get(
            'lsa_best_epoch_entropy', []
        )

        self.checkpoint_manager.cleanup_previous_fold_checkpoint(iter_idx, fold_idx)
        del model, optimizer, scheduler, loss_fn
        if scaler:
            del scaler
        self._cleanup_memory()

        return fold_summary

    # ------------------------------------------------------------------
    # Final model training
    # ------------------------------------------------------------------

    def train_final_model(
        self,
        train_loader: DataLoader,
        optimal_epochs: int,
        final_mean: List[float],
        final_std: List[float],
        class_weights: torch.Tensor,
    ) -> Tuple[nn.Module, Dict]:
        """Train the final model on all training data for ``optimal_epochs``."""
        exp = self.config.get('experiment_name', '')
        print(f"\n{'='*70}")
        print(f"[{exp.upper()}] TRAINING FINAL MODEL FOR {optimal_epochs} EPOCHS")
        print(f"{'='*70}")

        checkpoint = self._load_final_checkpoint()
        arch_meta  = create_model_arch_metadata(self.config)

        if checkpoint is None:
            final_history = {
                'train_loss':     [],
                'train_acc':      [],
                'train_acc_soft': [],
                # LSA fields pre-declared
                'lsa_temperatures':  [],
                'lsa_tau_raw':       [],
                'lsa_attn_entropy':  [],
            }
            start_epoch = 0
            print("Starting final training from scratch")
        else:
            validate_checkpoint_arch(checkpoint, self.config, strict=True)
            final_history = checkpoint['training_history']
            start_epoch   = checkpoint['epoch'] + 1
            for key in ('train_acc_soft', 'lsa_temperatures',
                        'lsa_tau_raw', 'lsa_attn_entropy'):
                final_history.setdefault(key, [])
            print(f"Resuming final training from epoch {start_epoch}")

        model                        = self.create_model(
            model_seed=self.config['random_seed']
        )
        optimizer, scheduler, scaler = self.create_optimizer_and_scheduler(
            model, optimal_epochs
        )
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        if checkpoint:
            self.checkpoint_manager.restore_training_state(
                checkpoint, model, optimizer, scheduler, scaler
            )

        # LSA temperature logger
        lsa_logger: Optional[LSALogger] = None
        if LSALogger.is_lsa_model(model):
            lsa_logger = LSALogger(
                model=model,
                fold_results=final_history,
                iter_idx=-1,
                fold_idx=-1,
                tau_warn_threshold=3.0,
            )

        final_start = timer()

        for epoch in range(start_epoch, optimal_epochs):
            train_loss, train_hard_acc, train_soft_acc = self._train_epoch(
                model, train_loader, loss_fn, optimizer, scaler
            )
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            print(
                f"Epoch {epoch+1}/{optimal_epochs} | "
                f"loss={train_loss:.4f} | acc={train_hard_acc:.4f} | "
                f"LR={current_lr:.6f}"
            )

            final_history['train_loss'].append(train_loss)
            final_history['train_acc'].append(train_hard_acc)
            final_history['train_acc_soft'].append(train_soft_acc)

            # Temperature logging
            if lsa_logger is not None:
                lsa_logger.start_validation_epoch()
                lsa_logger._remove_entropy_hooks()
                temps = lsa_logger._read_temperatures()
                final_history['lsa_temperatures'].append(temps['tau_exp'])
                final_history['lsa_tau_raw'].append(temps['tau_raw'])
                final_history['lsa_attn_entropy'].append(
                    [float('nan')] * lsa_logger._n_layers
                )

            should_save = (
                (epoch + 1) % self.config.get('checkpoint_every_n_epochs', 25) == 0
                or epoch == optimal_epochs - 1
            )
            if should_save:
                self.checkpoint_manager.create_final_checkpoint(
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    training_history=final_history,
                    mean=final_mean,
                    std=final_std,
                    scaler=scaler,
                    arch_metadata=arch_meta,
                )

        final_end     = timer()
        training_time = final_end - final_start
        print(f"\nFinal model training time: {training_time:.2f}s")

        self.checkpoint_manager.backup_to_drive(
            self.checkpoint_manager.get_final_checkpoint_path(use_local=True),
            self.checkpoint_manager.get_final_checkpoint_path(use_local=False),
            verbose=True,
        )

        return model, final_history

    # ------------------------------------------------------------------
    # Epoch-level methods
    # ------------------------------------------------------------------

    def _train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        scaler: Optional[object],
    ) -> Tuple[float, float, float]:
        model.train()
        running_loss  = 0.0
        hard_correct  = 0
        soft_correct  = 0.0
        total         = 0

        for X, y in train_loader:
            X, y = X.to(self.device), y.to(self.device)

            if self.use_mixup and self.mixup_transform is not None:
                X, y_a, y_b, lam, use_mixup_loss = self.mixup_transform(X, y)
            else:
                X, y_a, y_b, lam, use_mixup_loss = X, y, y, 1.0, False

            optimizer.zero_grad(set_to_none=True)

            if self.use_amp and scaler is not None:
                with self.autocast():
                    y_pred = model(X)
                    loss   = (
                        _mixup_criterion(loss_fn, y_pred, y_a, y_b, lam)
                        if use_mixup_loss else loss_fn(y_pred, y_a)
                    )
                scaler.scale(loss).backward()
                if self.config.get('gradient_clip_norm'):
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(
                        model.parameters(), self.config['gradient_clip_norm']
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                y_pred = model(X)
                loss   = (
                    _mixup_criterion(loss_fn, y_pred, y_a, y_b, lam)
                    if use_mixup_loss else loss_fn(y_pred, y_a)
                )
                loss.backward()
                if self.config.get('gradient_clip_norm'):
                    nn.utils.clip_grad_norm_(
                        model.parameters(), self.config['gradient_clip_norm']
                    )
                optimizer.step()

            running_loss  += loss.detach().item() * y_a.size(0)
            with torch.no_grad():
                preds         = y_pred.argmax(dim=1)
                hard_correct  += (preds == y_a).sum().item()
                soft_correct  += (
                    lam * (preds == y_a).sum().item()
                    + (1 - lam) * (preds == y_b).sum().item()
                    if use_mixup_loss
                    else (preds == y_a).sum().item()
                )
                total         += y_a.size(0)

        epoch_loss     = running_loss / total if total > 0 else 0.0
        epoch_hard_acc = hard_correct / total if total > 0 else 0.0
        epoch_soft_acc = soft_correct / total if total > 0 else 0.0
        return epoch_loss, epoch_hard_acc, epoch_soft_acc

    def _validate_epoch(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        val_groups: Optional[np.ndarray] = None,
    ) -> Tuple[float, float, Dict, Dict]:
        """Validate for one epoch; returns patch- and slide-level metrics."""
        model.eval()
        running_loss = 0.0
        correct      = 0
        total        = 0
        y_preds_list = []
        y_true_list  = []
        y_probs_list = []

        with torch.inference_mode():
            for X, y in val_loader:
                X, y = X.to(self.device), y.to(self.device)

                if self.use_amp and self.autocast is not None:
                    with self.autocast():
                        logits = model(X)
                        loss   = loss_fn(logits, y)
                else:
                    logits = model(X)
                    loss   = loss_fn(logits, y)

                running_loss += loss.item() * y.size(0)
                preds   = logits.argmax(dim=1)
                probs   = logits.softmax(dim=1)
                correct += (preds == y).sum().item()
                total   += y.size(0)

                y_preds_list.extend(preds.cpu().numpy())
                y_true_list.extend(y.cpu().numpy())
                y_probs_list.extend(probs.cpu().numpy())

        y_true  = np.array(y_true_list)
        y_pred  = np.array(y_preds_list)
        y_probs = np.array(y_probs_list)

        val_loss = running_loss / total if total > 0 else 0.0
        val_acc  = correct     / total if total > 0 else 0.0

        val_patch_metrics = {
            'bal_acc': balanced_accuracy_score(y_true, y_pred),
            'mcc':     matthews_corrcoef(y_true, y_pred),
            'f1':      f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'val_auc': (
                roc_auc_score(y_true, y_probs[:, 1])
                if y_probs.shape[1] == 2 else float('nan')
            ),
        }

        val_slide_metrics = compute_slide_level_metrics(
            y_true, y_pred, y_probs, val_groups, self.num_classes
        ) if val_groups is not None else {
            'slide_bal_acc': val_patch_metrics['bal_acc'],
            'slide_mcc':     val_patch_metrics['mcc'],
            'slide_f1':      val_patch_metrics['f1'],
            'slide_auc':     val_patch_metrics['val_auc'],
        }

        return val_loss, val_acc, val_patch_metrics, val_slide_metrics

    # ------------------------------------------------------------------
    # Fold summary builders
    # ------------------------------------------------------------------

    def _initialize_fold_results(self) -> Dict[str, List]:

        return {
            'train_loss':                  [],
            'train_acc':                   [],
            'train_acc_soft':              [],
            'val_loss':                    [],
            'val_acc':                     [],
            'val_bal_acc':                 [],
            'val_mcc':                     [],
            'val_f1':                      [],
            'val_auc':                     [],
            'val_slide_bal_acc':           [],
            'val_slide_bal_acc_smoothed':  [],
            'val_slide_mcc':               [],
            'val_slide_f1':                [],
            'val_slide_auc':               [],
            'lsa_temperatures':            [],
            'lsa_tau_raw':                 [],
            'lsa_attn_entropy':            [],
        }

    def _create_fold_summary(
        self,
        iter_idx: int,
        fold_idx: int,
        iteration_seed: int,
        model_seed: int,
        best_epoch_idx: int,
        fold_results: Dict[str, List],
        lexicographic_tracker: LexBestModelTracker,
        ema_tracker: ExponentialMovingAverage,
        min_delta: float,
        training_time: float,
        fold_mean: List[float],
        fold_std: List[float],
    ) -> Dict[str, Any]:
        if not fold_results['val_loss'] or best_epoch_idx < 0:
            warnings.warn(
                f"Iter {iter_idx+1} Fold {fold_idx+1} has no validation results!"
            )
            return self._create_empty_fold_summary(
                iter_idx, fold_idx, iteration_seed, model_seed,
                training_time, fold_mean, fold_std,
            )

        exp = self.config.get('experiment_name', '')
        return {
            'iteration':      iter_idx,
            'fold':           fold_idx + 1,
            'iteration_seed': iteration_seed,
            'model_init_seed': model_seed,
            'experiment_name': exp,
            'arch_hash':       compute_arch_hash(self.config),
            'init_policy':     self.config.get('init_policy', 'default'),
            'best_epoch':      best_epoch_idx + 1,
            'epochs_trained':  len(fold_results['train_loss']),
            # Training metrics at best epoch
            'train_loss':      fold_results['train_loss'][best_epoch_idx],
            'train_acc':       fold_results['train_acc'][best_epoch_idx],
            # Patch-level validation metrics
            'val_loss':            fold_results['val_loss'][best_epoch_idx],
            'val_loss_at_best':    fold_results['val_loss'][best_epoch_idx],
            'val_acc':             fold_results['val_acc'][best_epoch_idx],
            'val_bal_acc':         fold_results['val_bal_acc'][best_epoch_idx],
            'val_mcc':             fold_results['val_mcc'][best_epoch_idx],
            'val_f1':              fold_results['val_f1'][best_epoch_idx],
            'val_auc':             fold_results['val_auc'][best_epoch_idx],
            # Slide-level validation metrics
            'val_slide_bal_acc_raw':      fold_results['val_slide_bal_acc'][best_epoch_idx],
            'val_slide_bal_acc_smoothed': fold_results['val_slide_bal_acc_smoothed'][best_epoch_idx],
            'val_slide_mcc_best':         fold_results['val_slide_mcc'][best_epoch_idx],
            'val_slide_f1_best':          fold_results['val_slide_f1'][best_epoch_idx],
            'val_slide_auc_best':         fold_results['val_slide_auc'][best_epoch_idx],
            # Model selection metadata
            'selection_criterion': 'lexicographic_slide_ba_val_loss',
            'stopping_criterion':  'smoothed_slide_bal_acc',
            'epsilon_slide_ba':    lexicographic_tracker.epsilon_A,
            'epsilon_val_loss':    lexicographic_tracker.epsilon_L,
            'ema_alpha':           ema_tracker.alpha,
            'min_delta':           min_delta,
            # Timing and normalisation
            'training_time_s':     training_time,
            'fold_mean':           fold_mean,
            'fold_std':            fold_std,
        }

    def _create_fold_summary_from_history(
        self,
        history: Dict[str, Any],
        iter_idx: int,
        fold_idx: int,
        iteration_seed: int,
    ) -> Dict[str, Any]:
        """Reconstruct fold summary from saved history (checkpoint deleted)."""
        fold_results   = history.get('history', {})
        best_epoch_idx = history.get('best_epoch', 0)

        if not fold_results or best_epoch_idx < 0:
            return self._create_empty_fold_summary(
                iter_idx, fold_idx, iteration_seed, -1,
                0.0, history.get('fold_mean', []), history.get('fold_std', []),
            )

        summary = {
            'iteration':       iter_idx,
            'fold':            fold_idx + 1,
            'iteration_seed':  iteration_seed,
            'model_init_seed': -1,
            'experiment_name': self.config.get('experiment_name'),
            'arch_hash':       compute_arch_hash(self.config),
            'init_policy':     self.config.get('init_policy', 'default'),
            'best_epoch':      best_epoch_idx + 1,
            'epochs_trained':  history.get('n_epochs', 0),
            'train_loss':      fold_results.get('train_loss', [float('nan')])[best_epoch_idx],
            'train_acc':       fold_results.get('train_acc',  [float('nan')])[best_epoch_idx],
            'val_loss':        fold_results.get('val_loss',   [float('nan')])[best_epoch_idx],
            'val_loss_at_best': fold_results.get('val_loss',  [float('nan')])[best_epoch_idx],
            'val_acc':         fold_results.get('val_acc',    [float('nan')])[best_epoch_idx],
            'val_bal_acc':     fold_results.get('val_bal_acc',[float('nan')])[best_epoch_idx],
            'val_mcc':         fold_results.get('val_mcc',    [float('nan')])[best_epoch_idx],
            'val_f1':          fold_results.get('val_f1',     [float('nan')])[best_epoch_idx],
            'val_auc':         fold_results.get('val_auc',    [float('nan')])[best_epoch_idx],
            'val_slide_bal_acc_raw':      fold_results.get('val_slide_bal_acc', [float('nan')])[best_epoch_idx],
            'val_slide_bal_acc_smoothed': fold_results.get('val_slide_bal_acc_smoothed', [float('nan')])[best_epoch_idx],
            'val_slide_mcc_best':         fold_results.get('val_slide_mcc',     [float('nan')])[best_epoch_idx],
            'val_slide_f1_best':          fold_results.get('val_slide_f1',      [float('nan')])[best_epoch_idx],
            'val_slide_auc_best':         fold_results.get('val_slide_auc',     [float('nan')])[best_epoch_idx],
            'selection_criterion': 'lexicographic_slide_ba_val_loss',
            'stopping_criterion':  'smoothed_slide_bal_acc',
            'epsilon_slide_ba':    1e-6,
            'epsilon_val_loss':    1e-6,
            'ema_alpha':           0.0,
            'min_delta':           0.0,
            'training_time_s':     0.0,
            'fold_mean':           history.get('fold_mean', []),
            'fold_std':            history.get('fold_std',  []),
            # LSA fields
            'lsa_best_epoch_temperatures': fold_results.get('lsa_best_epoch_temperatures', []),
            'lsa_best_epoch_entropy':       fold_results.get('lsa_best_epoch_entropy', []),
        }
        return summary

    def _create_empty_fold_summary(
        self,
        iter_idx: int,
        fold_idx: int,
        iteration_seed: int,
        model_seed: int,
        training_time: float,
        fold_mean: List[float],
        fold_std: List[float],
    ) -> Dict[str, Any]:
        return {
            'iteration':       iter_idx,
            'fold':            fold_idx + 1,
            'iteration_seed':  iteration_seed,
            'model_init_seed': model_seed,
            'experiment_name': self.config.get('experiment_name'),
            'arch_hash':       compute_arch_hash(self.config),
            'init_policy':     self.config.get('init_policy', 'default'),
            'best_epoch':      0,
            'epochs_trained':  0,
            'train_loss':      float('inf'),
            'train_acc':       0.0,
            'val_loss':        float('inf'),
            'val_loss_at_best': float('inf'),
            'val_acc':         0.0,
            'val_bal_acc':     0.0,
            'val_mcc':         0.0,
            'val_f1':          0.0,
            'val_auc':         0.0,
            'val_slide_bal_acc_raw':      0.0,
            'val_slide_bal_acc_smoothed': 0.0,
            'val_slide_mcc_best':         0.0,
            'val_slide_f1_best':          0.0,
            'val_slide_auc_best':         0.0,
            'selection_criterion': 'lexicographic_slide_ba_val_loss',
            'stopping_criterion':  'smoothed_slide_bal_acc',
            'epsilon_slide_ba':    1e-6,
            'epsilon_val_loss':    1e-6,
            'ema_alpha':           0.0,
            'min_delta':           0.0,
            'training_time_s':     training_time,
            'fold_mean':           fold_mean,
            'fold_std':            fold_std,
            'lsa_best_epoch_temperatures': [],
            'lsa_best_epoch_entropy':      [],
        }

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _load_fold_checkpoint(
        self, iter_idx: int, fold_idx: int
    ) -> Optional[Dict[str, Any]]:
        drive_path = self.checkpoint_manager.get_cv_checkpoint_path(
            iter_idx, fold_idx, use_local=False
        )
        ckpt = self.checkpoint_manager.load_checkpoint(drive_path, verbose=False)
        if ckpt is not None:
            return ckpt
        local_path = self.checkpoint_manager.get_cv_checkpoint_path(
            iter_idx, fold_idx, use_local=True
        )
        return self.checkpoint_manager.load_checkpoint(local_path, verbose=False)

    def _load_final_checkpoint(self) -> Optional[Dict[str, Any]]:
        drive_path = self.checkpoint_manager.get_final_checkpoint_path(use_local=False)
        ckpt = self.checkpoint_manager.load_checkpoint(drive_path, verbose=False)
        if ckpt is not None:
            return ckpt
        local_path = self.checkpoint_manager.get_final_checkpoint_path(use_local=True)
        return self.checkpoint_manager.load_checkpoint(local_path, verbose=False)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _set_seeds(self, seed: int) -> None:
        seed = int(seed) & (2 ** 64 - 1)
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
        torch.use_deterministic_algorithms(True, warn_only=True)

    def _cleanup_memory(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _mixup_criterion(
    loss_fn: nn.Module,
    y_pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    return lam * loss_fn(y_pred, y_a) + (1 - lam) * loss_fn(y_pred, y_b)


def determine_optimal_epochs(
    cv_fold_details: List[Dict],
    method: str = "percentile_75",
    config: Dict = None,
    verbose: bool = True,
) -> int:
    """
    Determine optimal training duration from CV results.

    Parameters
    ----------
    cv_fold_details : list[dict] — fold summaries from cross-validation.
    method          : str — 'percentile_75' | 'median' | 'mean' | 'mean_plus_std'.
    config          : dict — used for num_epochs upper bound.
    verbose         : bool

    Returns
    -------
    int — epoch count for final model training.
    """
    if not cv_fold_details:
        warnings.warn("No CV fold details provided. Using default from config.")
        return config.get('num_epochs', 100) if config else 100

    try:
        df = pd.DataFrame(cv_fold_details)
    except Exception as e:
        raise ValueError(f"Could not convert cv_fold_details to DataFrame: {e}")

    required_cols = ['iteration', 'best_epoch']
    missing       = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"cv_fold_details missing columns: {missing}")

    valid_mask = (df['best_epoch'] > 0) & df['best_epoch'].notna()
    if not valid_mask.any():
        raise RuntimeError("No valid best_epoch values in cv_fold_details.")

    df_valid      = df[valid_mask].copy()
    n_valid       = len(df_valid)
    n_total       = len(df)
    if n_valid < n_total:
        warnings.warn(f"Only {n_valid}/{n_total} folds have valid best_epoch.")

    if verbose:
        print(f"\n{'='*70}")
        print("OPTIMAL EPOCHS CALCULATION")
        print(f"{'='*70}")
        print(
            f"Input: {n_valid} valid folds from "
            f"{df_valid['iteration'].nunique()} iterations | method: {method}"
        )

    iter_means = (
        df_valid.groupby('iteration')['best_epoch'].mean().values
    )

    if method == "percentile_75":
        raw = np.percentile(iter_means, 75)
    elif method == "median":
        raw = np.median(iter_means)
    elif method == "mean":
        raw = np.mean(iter_means)
    elif method == "mean_plus_std":
        raw = np.mean(iter_means) + np.std(iter_means)
    else:
        warnings.warn(f"Unknown method '{method}', defaulting to 'percentile_75'.")
        raw = np.percentile(iter_means, 75)

    optimal   = int(np.ceil(raw))
    min_ep    = 5
    max_ep    = config.get('num_epochs', 100) * 2 if config else 200
    optimal   = max(min_ep, min(optimal, max_ep))

    if verbose:
        print(f"  Raw value : {float(np.ceil(raw)):.1f} → bounded: {optimal} epochs")
        print(f"  Range     : [{min_ep}, {max_ep}]")
        print(f"\n  Train final model for {optimal} epochs")
        print(f"{'='*70}\n")

    return optimal
