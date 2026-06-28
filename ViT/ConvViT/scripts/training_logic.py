"""
Training Logic Module

Author: Rodrigo Sá
Date: 2025
"""

import copy
import gc
import warnings
from timeit import default_timer as timer
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.random import SeedSequence
import random

from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from convViT import SimpleViT_ConvTok, compute_arch_hash, count_parameters

from training_utils import (
    ExponentialMovingAverage,
    LexBestModelTracker,
    MixUpTransform,
    PatienceTracker,
    WarmupCosineScheduler,
    compute_slide_level_metrics,
)

# ---------------------------------------------------------------------------
# Arch metadata helper
# ---------------------------------------------------------------------------

def create_model_arch_metadata(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a self-describing architecture metadata dict to embed in checkpoints.
    """
    model_cfg = config.get("model_config", {})
    return {
        # Experiment identity
        "experiment_name":          config.get("experiment_name"),
        # Tokenizer architecture
        "convtok_blocks":           config.get("convtok_blocks"),
        "convtok_hidden_channels":  config.get("convtok_hidden_channels", 64),
        "convtok_post_ln":          config.get("convtok_post_ln", True),
        "convtok_expected_hw":      config.get("convtok_expected_hw"),
        "convtok_expected_tokens":  config.get("convtok_expected_tokens"),
        "convtok_conv_bias":        config.get("convtok_conv_bias", False),
        # Transformer architecture
        "dim":                      model_cfg.get("dim"),
        "depth":                    model_cfg.get("depth"),
        "heads":                    model_cfg.get("heads"),
        "mlp_dim":                  model_cfg.get("mlp_dim"),
        "dim_head":                 model_cfg.get("dim_head", 64),
        "pe_temperature":           model_cfg.get("pe_temperature", 10000),
        # Input geometry
        "image_size":               config.get("image_size"),
        "channels":                 config.get("channels"),
        # Initialisation policy
        "init_policy":              config.get("init_policy", "default"),
        # Hash
        "arch_hash":                compute_arch_hash(config),
    }


def validate_checkpoint_arch(
    checkpoint: Dict[str, Any],
    current_config: Dict[str, Any],
    strict: bool = True,
) -> bool:
    """
    Parameters
    ----------
    checkpoint   : loaded checkpoint dict
    current_config : the config being used for the current run
    strict       : if True, raise RuntimeError on mismatch; else warn

    Returns
    -------
    bool  True if match, False if mismatch (only when strict=False)
    """
    stored_meta = checkpoint.get("arch_metadata")
    if stored_meta is None:
        warnings.warn(
            "Checkpoint has no 'arch_metadata'. Skipping architecture validation. "
        )
        return True

    current_hash = compute_arch_hash(current_config)
    stored_hash = stored_meta.get("arch_hash")

    if current_hash != stored_hash:
        msg = (
            f"Architecture mismatch between checkpoint and current config!\n"
            f"  Checkpoint arch_hash : {stored_hash}\n"
            f"  Current    arch_hash : {current_hash}\n"
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
    """Handles training for the SimpleViT_ConvTok pipeline."""

    def __init__(
        self,
        config: Dict[str, Any],
        device: torch.device,
        checkpoint_manager,
        num_classes: int,
    ):
        self.config = config
        self.device = device
        self.checkpoint_manager = checkpoint_manager
        self.num_classes = num_classes

        # AMP setup
        self.use_amp = config.get("use_amp", False) and torch.cuda.is_available()
        if config.get("use_amp", False) and not torch.cuda.is_available():
            warnings.warn("AMP requested but CUDA not available. Disabling AMP.")
        
        if self.use_amp:
            from torch.amp import autocast, GradScaler
            self.autocast = lambda: autocast(device_type='cuda')
            self.GradScaler = lambda: GradScaler('cuda')
        else:
            self.autocast = None
            self.GradScaler = None

        # MixUp setup
        self.use_mixup = config["augmentation"].get("use_mixup", True)
        if self.use_mixup:
            mixup_alpha = config["augmentation"].get("mixup_alpha", 0.2)
            mixup_prob = config["augmentation"].get("mixup_prob", 0.5)
            self.mixup_transform = MixUpTransform(alpha=mixup_alpha, prob=mixup_prob)
            print(f"MixUp enabled: alpha={mixup_alpha}, prob={mixup_prob}")
        else:
            self.mixup_transform = None
            print("MixUp disabled")

    # ------------------------------------------------------------------
    # Model creation
    # ------------------------------------------------------------------
    
    def create_model(self, model_seed: int) -> nn.Module:
        """
        Instantiate and return a SimpleViT_ConvTok model.
        """
        self._set_seeds(model_seed)

        init_policy = self.config.get("init_policy", "default")
        model_cfg = self.config["model_config"]

        dim_head         = model_cfg.get("dim_head", 64)
        pe_temperature   = model_cfg.get("pe_temperature", 10000)
        hidden_channels  = self.config.get("convtok_hidden_channels", 64)

        print(f"  Model config:")
        print(f"    init_policy          : {init_policy}")
        print(f"    dim_head             : {dim_head}")
        print(f"    pe_temperature       : {pe_temperature}")
        print(f"    convtok_hidden_ch    : {hidden_channels}")
        print(f"    arch_hash            : {compute_arch_hash(self.config)}")

        model = SimpleViT_ConvTok(
            image_size=self.config["image_size"],
            num_classes=self.num_classes,
            dim=model_cfg["dim"],
            depth=model_cfg["depth"],
            heads=model_cfg["heads"],
            mlp_dim=model_cfg["mlp_dim"],
            channels=self.config["channels"],
            dim_head=dim_head,
            convtok_hidden_channels=hidden_channels,
            convtok_blocks=self.config.get("convtok_blocks"),
            convtok_match_baseline_tokens=self.config.get(
                "convtok_match_baseline_tokens", True
            ),
            convtok_expected_hw=self.config.get("convtok_expected_hw", [24, 24]),
            convtok_conv_bias=self.config.get("convtok_conv_bias", False),
            init_policy=init_policy,
            pe_temperature=pe_temperature,
        ).to(self.device)

        total_params, trainable_params = count_parameters(model)
        print(
            f"  Model created (seed={model_seed}): "
            f"{total_params:,} params ({trainable_params:,} trainable)"
        )

        self._log_token_stats(model)

        return model

    def _log_token_stats(self, model: nn.Module):
        """
        Log mean/std of token vectors (post-ConvTok+LN) on a fixed dummy batch.
        Confirms that LayerNorm is functioning (mean ≈ 0) before any training.
        """
        model.eval()
        dummy = torch.randn(
            2,
            self.config["channels"],
            self.config["image_size"],
            self.config["image_size"],
            device=self.device,
        )
        with torch.no_grad():
            toks, h_p, w_p = model.tokenizer(dummy)
        print(
            f"  [ConvTok diag] Token stats (post-ConvTok+LN, dummy batch): "
            f"mean={toks.mean().item():.4f}, std={toks.std().item():.4f}, "
            f"grid=({h_p},{w_p}), N={h_p*w_p}"
        )
        model.train()

    # ------------------------------------------------------------------
    # Optimiser / scheduler
    # ------------------------------------------------------------------

    def create_optimizer_and_scheduler(
        self,
        model: nn.Module,
        max_epochs: int,
    ) -> Tuple[optim.Optimizer, Any, Optional[Any]]:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=self.config["warmup_epochs"],
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
        """Train a single CV fold."""
        if val_groups_fold is None:
            raise ValueError(
                "val_groups_fold is REQUIRED for slide-level early stopping."
            )

        print(f"\n{'='*70}")
        print(
            f"ITERATION {iter_idx + 1}/{self.config['n_iterations']} | "
            f"FOLD {fold_idx + 1}/{self.config['n_folds']}"
        )
        print(f"Iteration Seed: {iteration_seed}")
        print(
            f"Validation: ~{len(np.unique(val_groups_fold))} slides, "
            f"{len(val_groups_fold)} patches"
        )

        ema_alpha = self.config.get('ema_alpha', 0.3)
        min_delta = self.config.get('min_delta', 0.0)
        epsilon_A = self.config.get('epsilon_slide_ba', 1e-6)
        epsilon_L = self.config.get('epsilon_val_loss', 1e-6)

        history = self.checkpoint_manager.load_fold_history(iter_idx, fold_idx, verbose=False)
        checkpoint = self._load_fold_checkpoint(iter_idx, fold_idx)

        if history is not None and checkpoint is None:
            print(f"\n WARNING: Fold {fold_idx+1} already completed")
            print(f"   History exists but checkpoint was cleaned")
            print(f"   Returning cached fold summary\n")
            return self._create_fold_summary_from_history(
                history, iter_idx, fold_idx, iteration_seed
            )

        if checkpoint is None:
            fold_results = self._initialize_fold_results()
            start_epoch = 0
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

            fold_results = checkpoint["fold_results"]
            start_epoch = checkpoint["epoch"] + 1
            best_model_state = checkpoint.get("best_model_state_dict", None)

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

            best_epoch = lexicographic_tracker.best_epoch_idx
            print(f"Resuming from checkpoint:")
            print(f"  Last completed epoch: {checkpoint['epoch']}")
            print(f"  Next epoch to train: {start_epoch}")
            print(f"  Best model: epoch {best_epoch + 1}")

        # Model creation
        fold_seed_sequence = SeedSequence(iteration_seed).spawn(self.config["n_folds"])
        model_seed = int(fold_seed_sequence[fold_idx].generate_state(1)[0])
        print(f"Model initialization seed: {model_seed}")

        model = self.create_model(model_seed=model_seed)
        optimizer, scheduler, scaler = self.create_optimizer_and_scheduler(
            model, self.config["num_epochs"]
        )
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        if checkpoint:
            self.checkpoint_manager.restore_training_state(
                checkpoint, model, optimizer, scheduler, scaler
            )
            print(f"  Restored optimizer, scheduler, and RNG states")

        fold_start = timer()
        arch_metadata = create_model_arch_metadata(self.config)

        # ========== TRAINING LOOP ==========
        for epoch in range(start_epoch, self.config["num_epochs"]):
            train_loss, train_hard_acc, train_soft_acc = self._train_epoch(
                model, train_loader, loss_fn, optimizer, scaler
            )
            val_loss, val_acc, val_patch_metrics, val_slide_metrics = self._validate_epoch(
                model, val_loader, loss_fn, val_groups_fold
            )
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            current_slide_ba = val_slide_metrics['slide_bal_acc']
            lexico_update = lexicographic_tracker.update(
                current_epoch=epoch,
                slide_ba=current_slide_ba,
                val_loss=val_loss,
            )
            ema_update = ema_tracker.update(current_slide_ba, epoch)
            patience_update = patience_tracker.update(ema_update['improved_smoothed'])

            if lexico_update['improved']:
                best_model_state = copy.deepcopy(model.state_dict())
                reason = lexico_update['improvement_reason']
                print(f"\n BEST MODEL at epoch {epoch+1} (reason: {reason})")
                print(f"   slide_BA: {current_slide_ba:.4f} | val_loss: {val_loss:.4f}")

            print(
                f"Iter {iter_idx+1}, Fold {fold_idx+1}, Epoch {epoch+1}/{self.config['num_epochs']} | "
                f"train_loss: {train_loss:.4f} | train_acc: {train_hard_acc:.4f}"
            )
            print(
                f"  PATCH-level: val_loss={val_loss:.4f}, bal_acc={val_patch_metrics['bal_acc']:.4f}"
            )
            print(
                f"  SLIDE-level: BA={current_slide_ba:.4f} | EMA smoothed={ema_update['smoothed']:.4f}"
            )
            print(
                f"  Best model: epoch {lexico_update['best_epoch']+1}, "
                f"BA={lexico_update['best_slide_ba']:.4f}, loss={lexico_update['best_val_loss']:.4f}"
            )
            print(
                f"  Best EMA smoothed: {ema_update['best_smoothed']:.4f} "
                f"(epoch {ema_update['best_smoothed_epoch']+1})"
            )
            print(
                f"  Patience: {patience_update['epochs_no_improve']}/{self.config['patience']} "
                f"(remaining: {patience_update['patience_remaining']})"
            )
            print(f"  LR: {current_lr:.6f}")

            # Record metrics
            fold_results["train_loss"].append(train_loss)
            fold_results["train_acc"].append(train_hard_acc)
            fold_results["train_acc_soft"].append(train_soft_acc)
            fold_results["val_loss"].append(val_loss)
            fold_results["val_acc"].append(val_acc)
            fold_results["val_bal_acc"].append(val_patch_metrics["bal_acc"])
            fold_results["val_mcc"].append(val_patch_metrics["mcc"])
            fold_results["val_f1"].append(val_patch_metrics["f1"])
            fold_results["val_auc"].append(val_patch_metrics["val_auc"])
            fold_results["val_slide_bal_acc"].append(current_slide_ba)
            fold_results["val_slide_bal_acc_smoothed"].append(ema_update['smoothed'])
            fold_results["val_slide_mcc"].append(val_slide_metrics["slide_mcc"])
            fold_results["val_slide_f1"].append(val_slide_metrics["slide_f1"])
            fold_results["val_slide_auc"].append(val_slide_metrics["slide_auc"])

            # Checkpoint
            should_save = (
                (epoch + 1) % self.config.get("checkpoint_every_n_epochs", 25) == 0
                or patience_tracker.should_stop
                or (epoch + 1) >= self.config["num_epochs"]
            )

            if should_save:
                checkpoint_success, history_success = (
                    self.checkpoint_manager.create_cv_checkpoint(
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
                        arch_metadata=arch_metadata,
                    )
                )

                if not checkpoint_success:
                    raise RuntimeError(
                        f"CRITICAL: Failed to save checkpoint at epoch {epoch+1}. "
                        "Training cannot safely continue without checkpoint backup."
                    )
                if not history_success:
                    warnings.warn(
                        f"Failed to save fold history at epoch {epoch+1}. "
                        "Analysis may be incomplete."
                    )

            if patience_tracker.should_stop:
                print(f"\n EARLY STOPPING triggered at epoch {epoch+1}")
                print(f"   Best model: epoch {lexicographic_tracker.best_epoch_idx+1}")
                print(
                    f"   slide_BA={lexicographic_tracker.best_slide_ba:.4f}, "
                    f"   val_loss={lexicographic_tracker.best_val_loss:.4f}"
                )
                print(
                    f"   Best EMA smoothed: {ema_tracker.best_smoothed:.4f} "
                    f"(epoch {ema_tracker.best_smoothed_epoch+1})"
                )
                break

        fold_end = timer()
        training_time = fold_end - fold_start

        # Restore best model weights
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            best_epoch_idx = lexicographic_tracker.best_epoch_idx
            print(f"\nRESTORED best model weights from epoch {best_epoch_idx+1}")
            print(f"      1. slide_BA (maximize): {lexicographic_tracker.best_slide_ba:.4f}")
            print(f"      2. val_loss (minimize): {lexicographic_tracker.best_val_loss:.4f}")
            print(
                f"    EMA smoothed at that epoch: "
                f"{fold_results['val_slide_bal_acc_smoothed'][best_epoch_idx]:.4f}"
            )
        else:
            warnings.warn("No best model state found – using final epoch weights.")
            best_epoch_idx = epoch

        # Final checkpoint
        print(
            f"\nSaving FINAL checkpoint and history for "
            f"Iter {iter_idx + 1} Fold {fold_idx + 1}..."
        )
        checkpoint_success, history_success = (
            self.checkpoint_manager.create_cv_checkpoint(
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
                arch_metadata=arch_metadata,
            )
        )

        if not checkpoint_success or not history_success:
            warnings.warn("Failed to save final checkpoint or history.")

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

        print(
            f"\nIter {iter_idx + 1} Fold {fold_idx + 1} Summary "
            f"(at best epoch {fold_summary['best_epoch']}):"
        )
        print(f"  SLIDE-LEVEL:")
        print(f"    Balanced Acc: {fold_summary['val_slide_bal_acc_raw']:.4f}")
        print(f"    Val Loss:     {fold_summary['val_loss_at_best']:.4f}")
        print(f"    EMA smoothed: {fold_summary['val_slide_bal_acc_smoothed']:.4f}")
        print(f"    MCC: {fold_summary['val_slide_mcc_best']:.4f}")
        print(f"    AUC: {fold_summary['val_slide_auc_best']:.4f}")
        print(f"  PATCH-LEVEL:")
        print(f"    Balanced Acc: {fold_summary['val_bal_acc']:.4f}")
        print(f"    MCC: {fold_summary['val_mcc']:.4f}")
        print(f"  Training time: {training_time:.2f}s")

        del model, optimizer, scheduler, loss_fn
        if scaler:
            del scaler
        self._cleanup_memory()

        return fold_summary

    # ------------------------------------------------------------------
    # _train_epoch / _validate_epoch / helpers
    # ------------------------------------------------------------------

    def _initialize_fold_results(self) -> Dict[str, List]:
        return {
            "train_loss": [], "train_acc": [], "train_acc_soft": [],
            "val_loss": [], "val_acc": [],
            "val_bal_acc": [], "val_mcc": [], "val_f1": [], "val_auc": [],
            "val_slide_bal_acc": [], "val_slide_bal_acc_smoothed": [],
            "val_slide_mcc": [], "val_slide_f1": [], "val_slide_auc": [],
        }

    def _train_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        scaler,
    ) -> Tuple[float, float, float]:
        model.train()
        total_loss = 0.0
        n_correct_hard = 0
        n_correct_soft = 0
        n_total = 0

        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if self.use_mixup and self.mixup_transform is not None:
                images, labels_a, labels_b, lam, mixed = (
                    self.mixup_transform(images, labels)
                )
            else:
                mixed = False

            optimizer.zero_grad()

            if self.use_amp and scaler is not None:
                with self.autocast():
                    logits = model(images)
                    if mixed:
                        loss = lam * loss_fn(logits, labels_a) + (1 - lam) * loss_fn(logits, labels_b)
                    else:
                        loss = loss_fn(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    model.parameters(), self.config.get("gradient_clip_norm", 1.0)
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images)
                if mixed:
                    loss = lam * loss_fn(logits, labels_a) + (1 - lam) * loss_fn(logits, labels_b)
                else:
                    loss = loss_fn(logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(), self.config.get("gradient_clip_norm", 1.0)
                )
                optimizer.step()

            preds = logits.argmax(dim=1)
            n_correct_hard += (preds == labels).sum().item()
            
            if mixed:
                # Soft accuracy: expectation over mixup targets
                soft_correct = (
                    lam * (preds == labels_a).float() +
                    (1 - lam) * (preds == labels_b).float()
                ).sum().item()
                n_correct_soft += soft_correct
            else:
                n_correct_soft += (preds == labels).sum().item()

            total_loss += loss.item() * images.size(0)
            n_total += images.size(0)

        avg_loss = total_loss / max(n_total, 1)
        hard_acc = n_correct_hard / max(n_total, 1)
        soft_acc = n_correct_soft / max(n_total, 1)
        return avg_loss, hard_acc, soft_acc

    def _validate_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        loss_fn: nn.Module,
        val_groups: Optional[np.ndarray] = None,
    ) -> Tuple[float, float, Dict, Dict]:
        model.eval()
        total_loss = 0.0
        n_correct = 0
        n_total = 0
        y_true_list: List[int] = []
        y_preds_list: List[int] = []
        y_probs_list: List[np.ndarray] = []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                if self.use_amp:
                    with self.autocast():
                        logits = model(images)
                        loss = loss_fn(logits, labels)
                else:
                    logits = model(images)
                    loss = loss_fn(logits, labels)

                probs = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)

                n_correct += (preds == labels).sum().item()
                total_loss += loss.item() * images.size(0)
                n_total += images.size(0)

                y_true_list.extend(labels.cpu().numpy().tolist())
                y_preds_list.extend(preds.cpu().numpy().tolist())
                y_probs_list.append(probs.cpu().numpy())

        val_loss = total_loss / max(n_total, 1)
        val_acc = n_correct / max(n_total, 1)

        all_true = np.array(y_true_list)
        all_preds = np.array(y_preds_list)
        all_probs = np.concatenate(y_probs_list)

        # Patch-level metrics
        patch_metrics = {}
        if y_true_list:
            try:
                patch_metrics["bal_acc"] = balanced_accuracy_score(all_true, all_preds)
            except Exception as e:
                warnings.warn(f"Could not calculate patch balanced accuracy: {e}")
                patch_metrics["bal_acc"] = 0.0
            try:
                patch_metrics["mcc"] = matthews_corrcoef(all_true, all_preds)
            except Exception as e:
                warnings.warn(f"Could not calculate patch MCC: {e}")
                patch_metrics["mcc"] = 0.0
            try:
                patch_metrics["f1"] = f1_score(
                    all_true, all_preds, average="weighted", zero_division=0
                )
            except Exception as e:
                warnings.warn(f"Could not calculate patch F1: {e}")
                patch_metrics["f1"] = 0.0
            try:
                patch_metrics["val_auc"] = roc_auc_score(all_true, all_probs[:, 1])
            except Exception as e:
                warnings.warn(f"Could not calculate patch AUC: {e}")
                patch_metrics["val_auc"] = 0.0
        else:
            patch_metrics = {"bal_acc": 0.0, "mcc": 0.0, "f1": 0.0, "val_auc": 0.0}

        # Slide-level metrics
        if val_groups is not None:
            slide_metrics = compute_slide_level_metrics(
                y_true_patches=all_true,
                y_pred_patches=all_preds,
                y_proba_patches=all_probs,
                groups=val_groups,
                num_classes=self.num_classes,
                verbose=False,
            )
        else:
            warnings.warn("val_groups not provided – cannot compute slide metrics.")
            slide_metrics = {
                'slide_bal_acc': 0.0, 'slide_mcc': 0.0,
                'slide_f1': 0.0, 'slide_auc': 0.0, 'n_slides': 0,
            }

        return val_loss, val_acc, patch_metrics, slide_metrics

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
    ) -> Tuple[nn.Module, Dict[str, List[float]]]:
        """Train the final model on the full training set."""
        print(f"\n{'='*70}")
        print("FINAL MODEL TRAINING")
        print(f"  Epochs      : {optimal_epochs}")
        model_seed = self.config["random_seed"]
        print(f"  Model seed  : {model_seed}")
        print(f"  Experiment  : {self.config.get('experiment_name')}")
        print(f"{'='*70}")

        model = self.create_model(model_seed=model_seed)
        optimizer, scheduler, scaler = self.create_optimizer_and_scheduler(
            model, optimal_epochs
        )
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        checkpoint = self._load_final_checkpoint()
        start_epoch = 0
        history: Dict[str, List[float]] = {
            "train_loss": [], "train_acc": [], "train_acc_soft": []
        }

        if checkpoint is not None:
            validate_checkpoint_arch(checkpoint, self.config, strict=True)
            start_epoch = checkpoint["epoch"] + 1
            self.checkpoint_manager.restore_training_state(
                checkpoint, model, optimizer, scheduler, scaler
            )
            history = checkpoint.get("training_history", history)
            print(f"  Resumed final model from epoch {start_epoch}")

        arch_metadata = create_model_arch_metadata(self.config)

        for epoch in range(start_epoch, optimal_epochs):
            train_loss, hard_acc, soft_acc = self._train_epoch(
                model, train_loader, loss_fn, optimizer, scaler
            )
            scheduler.step()
            lr = scheduler.get_last_lr()[0]

            history["train_loss"].append(train_loss)
            history["train_acc"].append(hard_acc)
            history["train_acc_soft"].append(soft_acc)

            print(
                f"Final Epoch {epoch+1}/{optimal_epochs} | "
                f"loss={train_loss:.4f} | acc={hard_acc:.4f} | lr={lr:.6f}"
            )

            # Periodic checkpoint
            if (epoch + 1) % self.config.get("checkpoint_every_n_epochs", 25) == 0 or \
               (epoch + 1) >= optimal_epochs:
                self.checkpoint_manager.create_final_checkpoint(
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    training_history=history,
                    mean=final_mean,
                    std=final_std,
                    scaler=scaler,
                    arch_metadata=arch_metadata,
                )

        return model, history

    # ------------------------------------------------------------------
    # Fold summary helpers
    # ------------------------------------------------------------------

    def _create_fold_summary(
        self,
        iter_idx: int,
        fold_idx: int,
        iteration_seed: int,
        model_seed: int,
        best_epoch_idx: int,
        fold_results: Dict[str, List[float]],
        lexicographic_tracker,
        ema_tracker,
        min_delta: float,
        training_time: float,
        fold_mean: List[float],
        fold_std: List[float],
    ) -> Dict[str, Any]:
        if not fold_results["val_loss"] or best_epoch_idx < 0:
            warnings.warn(f"Iter {iter_idx+1} Fold {fold_idx + 1} has no validation results!")
            return self._create_empty_fold_summary(
                iter_idx, fold_idx, iteration_seed, model_seed,
                training_time, fold_mean, fold_std,
            )

        return {
            "iteration": iter_idx,
            "fold": fold_idx + 1,
            "iteration_seed": iteration_seed,
            "model_init_seed": model_seed,
            "experiment_name": self.config.get("experiment_name"),
            "init_policy": self.config.get("init_policy", "default"),
            "arch_hash": compute_arch_hash(self.config),
            "best_epoch": best_epoch_idx + 1,
            "epochs_trained": len(fold_results["train_loss"]),
            "train_loss": fold_results["train_loss"][best_epoch_idx],
            "train_acc": fold_results["train_acc"][best_epoch_idx],
            "val_loss": fold_results["val_loss"][best_epoch_idx],
            "val_loss_at_best": fold_results["val_loss"][best_epoch_idx],
            "val_acc": fold_results["val_acc"][best_epoch_idx],
            "val_bal_acc": fold_results["val_bal_acc"][best_epoch_idx],
            "val_mcc": fold_results["val_mcc"][best_epoch_idx],
            "val_f1": fold_results["val_f1"][best_epoch_idx],
            "val_auc": fold_results["val_auc"][best_epoch_idx],
            "val_slide_bal_acc_raw": fold_results["val_slide_bal_acc"][best_epoch_idx],
            "val_slide_bal_acc_smoothed": fold_results["val_slide_bal_acc_smoothed"][best_epoch_idx],
            "val_slide_mcc_best": fold_results["val_slide_mcc"][best_epoch_idx],
            "val_slide_f1_best": fold_results["val_slide_f1"][best_epoch_idx],
            "val_slide_auc_best": fold_results["val_slide_auc"][best_epoch_idx],
            "lexicographic_best_slide_ba": lexicographic_tracker.best_slide_ba,
            "lexicographic_best_val_loss": lexicographic_tracker.best_val_loss,
            "ema_best_smoothed": ema_tracker.best_smoothed,
            "ema_best_smoothed_epoch": ema_tracker.best_smoothed_epoch + 1,
            "training_time_s": training_time,
            "fold_mean": fold_mean,
            "fold_std": fold_std,
        }

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
            "iteration": iter_idx,
            "fold": fold_idx + 1,
            "iteration_seed": iteration_seed,
            "model_init_seed": model_seed,
            "experiment_name": self.config.get("experiment_name"),
            "init_policy": self.config.get("init_policy", "default"),
            "arch_hash": compute_arch_hash(self.config),
            "best_epoch": -1,
            "epochs_trained": 0,
            "train_loss": float('nan'),
            "train_acc": float('nan'),
            "val_loss": float('nan'),
            "val_loss_at_best": float('nan'),
            "val_acc": float('nan'),
            "val_bal_acc": float('nan'),
            "val_mcc": float('nan'),
            "val_f1": float('nan'),
            "val_auc": float('nan'),
            "val_slide_bal_acc_raw": float('nan'),
            "val_slide_bal_acc_smoothed": float('nan'),
            "val_slide_mcc_best": float('nan'),
            "val_slide_f1_best": float('nan'),
            "val_slide_auc_best": float('nan'),
            "training_time_s": training_time,
            "fold_mean": fold_mean,
            "fold_std": fold_std,
        }

    def _create_fold_summary_from_history(
        self,
        history: Dict,
        iter_idx: int,
        fold_idx: int,
        iteration_seed: int,
    ) -> Dict[str, Any]:
        fold_seed_sequence = SeedSequence(iteration_seed).spawn(self.config["n_folds"])
        model_seed = int(fold_seed_sequence[fold_idx].generate_state(1)[0])

        best_epoch_idx = history.get('best_epoch', 0)
        fold_results = history.get('history', {})

        if not fold_results or best_epoch_idx < 0:
            return self._create_empty_fold_summary(
                iter_idx, fold_idx, iteration_seed, model_seed,
                0.0, history.get('fold_mean', []), history.get('fold_std', []),
            )

        return {
            "iteration": iter_idx,
            "fold": fold_idx + 1,
            "iteration_seed": iteration_seed,
            "model_init_seed": model_seed,
            "experiment_name": self.config.get("experiment_name"),
            "init_policy": self.config.get("init_policy", "default"),
            "arch_hash": compute_arch_hash(self.config),
            "best_epoch": best_epoch_idx + 1,
            "epochs_trained": len(fold_results.get("train_loss", [])),
            "train_loss": fold_results.get("train_loss", [float('nan')])[best_epoch_idx],
            "train_acc": fold_results.get("train_acc", [float('nan')])[best_epoch_idx],
            "val_loss": fold_results.get("val_loss", [float('nan')])[best_epoch_idx],
            "val_loss_at_best": fold_results.get("val_loss", [float('nan')])[best_epoch_idx],
            "val_acc": fold_results.get("val_acc", [float('nan')])[best_epoch_idx],
            "val_bal_acc": fold_results.get("val_bal_acc", [float('nan')])[best_epoch_idx],
            "val_mcc": fold_results.get("val_mcc", [float('nan')])[best_epoch_idx],
            "val_f1": fold_results.get("val_f1", [float('nan')])[best_epoch_idx],
            "val_auc": fold_results.get("val_auc", [float('nan')])[best_epoch_idx],
            "val_slide_bal_acc_raw": fold_results.get("val_slide_bal_acc", [float('nan')])[best_epoch_idx],
            "val_slide_bal_acc_smoothed": fold_results.get("val_slide_bal_acc_smoothed", [float('nan')])[best_epoch_idx],
            "val_slide_mcc_best": fold_results.get("val_slide_mcc", [float('nan')])[best_epoch_idx],
            "val_slide_f1_best": fold_results.get("val_slide_f1", [float('nan')])[best_epoch_idx],
            "val_slide_auc_best": fold_results.get("val_slide_auc", [float('nan')])[best_epoch_idx],
            "training_time_s": 0.0,
            "fold_mean": history.get('fold_mean', []),
            "fold_std": history.get('fold_std', []),
        }

    # ------------------------------------------------------------------
    # Checkpoint loading helpers
    # ------------------------------------------------------------------

    def _load_fold_checkpoint(
        self, iter_idx: int, fold_idx: int
    ) -> Optional[Dict[str, Any]]:
        drive_path = self.checkpoint_manager.get_cv_checkpoint_path(
            iter_idx, fold_idx, use_local=False
        )
        checkpoint = self.checkpoint_manager.load_checkpoint(drive_path, verbose=False)
        if checkpoint is not None:
            return checkpoint
        local_path = self.checkpoint_manager.get_cv_checkpoint_path(
            iter_idx, fold_idx, use_local=True
        )
        return self.checkpoint_manager.load_checkpoint(local_path, verbose=False)

    def _load_final_checkpoint(self) -> Optional[Dict[str, Any]]:
        drive_path = self.checkpoint_manager.get_final_checkpoint_path(use_local=False)
        checkpoint = self.checkpoint_manager.load_checkpoint(drive_path, verbose=False)
        if checkpoint is not None:
            return checkpoint
        local_path = self.checkpoint_manager.get_final_checkpoint_path(use_local=True)
        return self.checkpoint_manager.load_checkpoint(local_path, verbose=False)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _set_seeds(self, seed: int):
        seed = int(seed) & (2 ** 64 - 1)
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True,warn_only=True)

    def _cleanup_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# determine_optimal_epochs
# ---------------------------------------------------------------------------

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
    cv_fold_details : list[dict]
        List of fold summary dictionaries.
    method : str
        Aggregation method: 'percentile_75', 'median', 'mean', 'mean_plus_std'.
    config : dict
        Config dict (used for num_epochs bound).
    verbose : bool

    Returns
    -------
    int  Optimal epoch count for final model training.
    """
    if not cv_fold_details:
        warnings.warn("No CV fold details provided. Using default from config.")
        return config.get('num_epochs', 100)

    try:
        df = pd.DataFrame(cv_fold_details)
    except Exception as e:
        raise ValueError(f"Could not convert cv_fold_details to DataFrame: {e}")

    required_cols = ['iteration', 'best_epoch']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"cv_fold_details missing required columns: {missing_cols}."
        )

    valid_mask = (df['best_epoch'] > 0) & (df['best_epoch'].notna())
    if not valid_mask.any():
        raise RuntimeError("No valid best_epoch values found in cv_fold_details.")

    df_valid = df[valid_mask].copy()
    n_valid_folds = len(df_valid)
    n_total_folds = len(df)
    if n_valid_folds < n_total_folds:
        warnings.warn(
            f"Only {n_valid_folds}/{n_total_folds} folds have valid best_epoch."
        )

    if verbose:
        print(f"\n{'='*70}")
        print("OPTIMAL EPOCHS CALCULATION (Hierarchically Correct)")
        print(f"{'='*70}")
        print(
            f"Input: {n_valid_folds} valid folds from "
            f"{df_valid['iteration'].nunique()} iterations"
        )
        print(f"Method: {method}")

    iteration_stats = df_valid.groupby('iteration')['best_epoch'].agg(
        mean='mean', std='std', min='min', max='max', count='count'
    ).reset_index()
    iteration_means = iteration_stats['mean'].values

    if method == "percentile_75":
        optimal_raw = np.percentile(iteration_means, 75)
    elif method == "median":
        optimal_raw = np.median(iteration_means)
    elif method == "mean":
        optimal_raw = np.mean(iteration_means)
    elif method == "mean_plus_std":
        optimal_raw = np.mean(iteration_means) + np.std(iteration_means)
    else:
        warnings.warn(f"Unknown method '{method}', defaulting to 'percentile_75'.")
        optimal_raw = np.percentile(iteration_means, 75)

    optimal = int(np.ceil(optimal_raw))
    min_epochs = 5
    max_epochs = config.get("num_epochs", 100) * 2 if config else 200
    optimal_bounded = max(min_epochs, min(optimal, max_epochs))

    if verbose:
        print(f"\n  Raw value : {optimal_raw:.2f} epochs")
        if optimal != optimal_bounded:
            print(f"  Bounded   : {optimal} → {optimal_bounded} (limits [{min_epochs}, {max_epochs}])")
        print(f"\n Train final model for {optimal_bounded} epochs")
        print(f"{'='*70}\n")

    return optimal_bounded
