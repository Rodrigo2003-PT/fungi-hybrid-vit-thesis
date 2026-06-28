"""
Training Logic Module

Author: Rodrigo Sá
Date: 2025
"""

import copy
import gc
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

from densenet_model import DenseNet121Classifier
from training_utils import (
    ExponentialMovingAverage,
    LexBestModelTracker,
    PatienceTracker,
    WarmupCosineScheduler,
)


class ModelTrainer:
    """
    Handles all training phases for the DenseNet-121 CNN.
    """

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
            from torch.amp import GradScaler, autocast
            self.autocast = lambda: autocast(device_type='cuda')
            self.GradScaler = lambda: GradScaler('cuda')
        else:
            self.autocast = None
            self.GradScaler = None

        self.use_mixup = config["augmentation"].get("use_mixup", True)
        if self.use_mixup:
            from training_utils import MixUpTransform
            mixup_alpha = config["augmentation"].get("mixup_alpha", 0.2)
            mixup_prob = config["augmentation"].get("mixup_prob", 0.5)
            self.mixup_transform = MixUpTransform(alpha=mixup_alpha, prob=mixup_prob)
            print(f"MixUp enabled: alpha={mixup_alpha}, prob={mixup_prob}")
        else:
            self.mixup_transform = None
            print("MixUp disabled")

    def create_model(self, model_seed: int) -> nn.Module:
        """
        Args:
            model_seed: Integer seed for this fold's model initialization.

        Returns:
            Initialized DenseNet121Classifier on self.device.
        """
        self._set_seeds(model_seed)

        model = DenseNet121Classifier(
            num_classes=self.num_classes,
            **{k: v for k, v in self.config["model_config"].items()
               if k != 'num_classes'},
        ).to(self.device)

        counts = model.count_parameters()
        print(
            f"  DenseNet121Classifier created with seed {model_seed}: "
            f"{counts['total']:,} params ({counts['trainable']:,} trainable)"
        )

        return model

    def create_optimizer_and_scheduler(
        self,
        model: nn.Module,
        max_epochs: int,
    ) -> Tuple[optim.Optimizer, Any, Optional[Any]]:
        """
        Create AdamW optimizer and WarmupCosineScheduler.
        """
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

        # Configuration
        ema_alpha = self.config.get('ema_alpha', 0.3)
        min_delta = self.config.get('min_delta', 0.0)
        epsilon_A = self.config.get('epsilon_slide_ba', 1e-6)
        epsilon_L = self.config.get('epsilon_val_loss', 1e-6)

        history = self.checkpoint_manager.load_fold_history(
            iter_idx, fold_idx, verbose=False
        )
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
                epsilon_A=epsilon_A,
                epsilon_L=epsilon_L,
            )
            ema_tracker = ExponentialMovingAverage(
                alpha=ema_alpha,
                maximize=True,
                min_delta=min_delta,
            )
            patience_tracker = PatienceTracker(
                patience=self.config['patience'],
                min_delta=min_delta,
            )
            print("Starting fold from scratch")

        else:
            fold_results = checkpoint["fold_results"]
            start_epoch = checkpoint["epoch"] + 1
            best_model_state = checkpoint.get("best_model_state_dict", None)

            lexicographic_tracker = LexBestModelTracker(
                epsilon_A=epsilon_A,
                epsilon_L=epsilon_L,
            )
            ema_tracker = ExponentialMovingAverage(
                alpha=ema_alpha,
                maximize=True,
                min_delta=min_delta,
            )
            patience_tracker = PatienceTracker(
                patience=self.config['patience'],
                min_delta=min_delta,
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

        # Derive fold model seed from iteration seed
        fold_seed_sequence = SeedSequence(iteration_seed).spawn(
            self.config["n_folds"]
        )
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

        # ========== TRAINING LOOP ==========
        for epoch in range(start_epoch, self.config["num_epochs"]):

            train_loss, train_hard_acc, train_soft_acc = self._train_epoch(
                model, train_loader, loss_fn, optimizer, scaler
            )

            val_loss, val_acc, val_patch_metrics, val_slide_metrics = \
                self._validate_epoch(model, val_loader, loss_fn, val_groups_fold)

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
                print(
                    f"\n BEST MODEL at epoch {epoch+1} (reason: {reason})"
                )
                print(
                    f"   slide_BA: {current_slide_ba:.4f} | "
                    f"val_loss: {val_loss:.4f}"
                )

            print(
                f"Iter {iter_idx+1}, Fold {fold_idx+1}, "
                f"Epoch {epoch+1}/{self.config['num_epochs']} | "
                f"train_loss: {train_loss:.4f} | "
                f"train_acc: {train_hard_acc:.4f}"
            )
            print(
                f"  PATCH-level: val_loss={val_loss:.4f}, "
                f"bal_acc={val_patch_metrics['bal_acc']:.4f}"
            )
            print(
                f"  SLIDE-level: BA={current_slide_ba:.4f} | "
                f"EMA smoothed={ema_update['smoothed']:.4f}"
            )
            print(
                f"  Best model: epoch {lexico_update['best_epoch']+1}, "
                f"BA={lexico_update['best_slide_ba']:.4f}, "
                f"loss={lexico_update['best_val_loss']:.4f}"
            )
            print(
                f"  Best EMA smoothed: {ema_update['best_smoothed']:.4f} "
                f"(epoch {ema_update['best_smoothed_epoch']+1})"
            )
            print(
                f"  Patience: {patience_update['epochs_no_improve']}/"
                f"{self.config['patience']} "
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
            fold_results["val_slide_bal_acc_smoothed"].append(
                ema_update['smoothed']
            )
            fold_results["val_slide_mcc"].append(val_slide_metrics["slide_mcc"])
            fold_results["val_slide_f1"].append(val_slide_metrics["slide_f1"])
            fold_results["val_slide_auc"].append(val_slide_metrics["slide_auc"])

            # Periodic checkpointing
            should_save_checkpoint = (
                (epoch + 1) % self.config.get("checkpoint_every_n_epochs", 2) == 0
                or patience_tracker.should_stop
                or (epoch + 1) >= self.config["num_epochs"]
            )

            if should_save_checkpoint:
                checkpoint_success, history_success = \
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
                    )

                if not checkpoint_success:
                    raise RuntimeError(
                        f"CRITICAL: Failed to save checkpoint at epoch {epoch+1}. "
                        f"Training cannot safely continue without checkpoint backup."
                    )

                if not history_success:
                    warnings.warn(
                        f"Failed to save fold history at epoch {epoch+1}. "
                        f"Analysis may be incomplete."
                    )

            if patience_tracker.should_stop:
                print(f"\n EARLY STOPPING triggered at epoch {epoch+1}")
                print(
                    f"   Best model: epoch "
                    f"{lexicographic_tracker.best_epoch_idx+1}"
                )
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
            print(
                f"\nRESTORED best model weights from epoch {best_epoch_idx+1}"
            )
            print(
                f"      1. slide_BA (maximize): "
                f"{lexicographic_tracker.best_slide_ba:.4f}"
            )
            print(
                f"      2. val_loss (minimize): "
                f"{lexicographic_tracker.best_val_loss:.4f}"
            )
            print(
                f"    EMA smoothed at that epoch: "
                f"{fold_results['val_slide_bal_acc_smoothed'][best_epoch_idx]:.4f}"
            )
        else:
            warnings.warn("No best model state found - using final epoch weights.")
            best_epoch_idx = epoch

        # Save final checkpoint and history
        print(
            f"\nSaving FINAL checkpoint and history for "
            f"Iter {iter_idx + 1} Fold {fold_idx + 1}..."
        )
        checkpoint_success, history_success = \
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
            )

        if not checkpoint_success or not history_success:
            warnings.warn("Failed to save final checkpoint or history")

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
        print(
            f"    Balanced Acc: {fold_summary['val_slide_bal_acc_raw']:.4f}"
        )
        print(f"    Val Loss:     {fold_summary['val_loss_at_best']:.4f}")
        print(
            f"    EMA smoothed: "
            f"{fold_summary['val_slide_bal_acc_smoothed']:.4f}"
        )
        print(f"    MCC: {fold_summary['val_slide_mcc_best']:.4f}")
        print(f"    AUC: {fold_summary['val_slide_auc_best']:.4f}")
        print(f"  PATCH-LEVEL:")
        print(f"    Balanced Acc: {fold_summary['val_bal_acc']:.4f}")
        print(f"    MCC: {fold_summary['val_mcc']:.4f}")
        print(f"  Training time: {training_time:.2f}s")

        self.checkpoint_manager.cleanup_previous_fold_checkpoint(
            iter_idx, fold_idx
        )
        del model, optimizer, scheduler, loss_fn
        if scaler:
            del scaler
        self._cleanup_memory()

        return fold_summary

    def train_final_model(
        self,
        train_loader: DataLoader,
        optimal_epochs: int,
        final_mean: List[float],
        final_std: List[float],
        class_weights: torch.Tensor,
    ) -> Tuple:

        print(f"\n{'='*70}")
        print(f"TRAINING FINAL MODEL FOR {optimal_epochs} EPOCHS")
        print(f"Using Base Seed: {self.config['random_seed']}")
        print(f"{'='*70}")

        checkpoint = self._load_final_checkpoint()

        if checkpoint is None:
            final_training_history = {
                "train_loss": [],
                "train_acc": [],
                "train_acc_soft": [],
            }
            start_epoch = 0
            print("Starting final training from scratch")
        else:
            final_training_history = checkpoint["training_history"]
            start_epoch = checkpoint["epoch"] + 1

            if "train_acc_soft" not in final_training_history:
                final_training_history["train_acc_soft"] = [0.0] * len(
                    final_training_history["train_loss"]
                )
            print(f"Resuming from epoch {start_epoch}")

        model = self.create_model(model_seed=self.config["random_seed"])
        optimizer, scheduler, scaler = self.create_optimizer_and_scheduler(
            model, optimal_epochs
        )
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        if checkpoint:
            self.checkpoint_manager.restore_training_state(
                checkpoint, model, optimizer, scheduler, scaler
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
                f"Loss: {train_loss:.4f} | "
                f"Acc: {train_hard_acc:.4f} | "
                f"LR: {current_lr:.6f}"
            )

            final_training_history["train_loss"].append(train_loss)
            final_training_history["train_acc"].append(train_hard_acc)
            final_training_history["train_acc_soft"].append(train_soft_acc)

            should_save = (
                (epoch + 1) % self.config.get("checkpoint_every_n_epochs", 5) == 0
                or epoch == optimal_epochs - 1
            )

            if should_save:
                self.checkpoint_manager.create_final_checkpoint(
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    final_training_history,
                    final_mean,
                    final_std,
                    scaler,
                )

        final_end = timer()
        training_time = final_end - final_start
        print(f"\nFinal model training time: {training_time:.2f}s")

        self.checkpoint_manager.backup_to_drive(
            self.checkpoint_manager.get_final_checkpoint_path(use_local=True),
            self.checkpoint_manager.get_final_checkpoint_path(use_local=False),
            verbose=True,
        )

        return model, final_training_history

    # ----------------------------------------------------------------------
    # Internal training primitives
    # ----------------------------------------------------------------------

    def _train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        scaler: Optional[object],
    ) -> Tuple[float, float, float]:
        """Train for one epoch. Verbatim from ViT training_logic.py."""
        model.train()
        running_loss, hard_correct, soft_correct, total = 0.0, 0, 0, 0

        for X, y in train_loader:
            X, y = X.to(self.device), y.to(self.device)

            if self.use_mixup and self.mixup_transform is not None:
                X, y_a, y_b, lam = self.mixup_transform(X, y)
                use_mixup_loss = True
            else:
                X, y_a, y_b, lam = X, y, y, 1.0
                use_mixup_loss = False

            optimizer.zero_grad(set_to_none=True)

            if self.use_amp and scaler is not None:
                with self.autocast():
                    y_pred = model(X)
                    if use_mixup_loss:
                        from training_utils import mixup_criterion
                        loss = mixup_criterion(loss_fn, y_pred, y_a, y_b, lam)
                    else:
                        loss = loss_fn(y_pred, y_a)

                scaler.scale(loss).backward()

                if self.config.get("gradient_clip_norm"):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        self.config["gradient_clip_norm"],
                    )

                scaler.step(optimizer)
                scaler.update()
            else:
                y_pred = model(X)
                if use_mixup_loss:
                    from training_utils import mixup_criterion
                    loss = mixup_criterion(loss_fn, y_pred, y_a, y_b, lam)
                else:
                    loss = loss_fn(y_pred, y_a)
                loss.backward()

                if self.config.get("gradient_clip_norm"):
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        self.config["gradient_clip_norm"],
                    )

                optimizer.step()

            batch_loss = loss.detach().item()
            running_loss += batch_loss * y_a.size(0)

            with torch.no_grad():
                preds = y_pred.argmax(dim=1)
                hard_correct += (preds == y_a).sum().item()

                if use_mixup_loss:
                    soft_correct += (
                        lam * (preds == y_a).sum().item()
                        + (1 - lam) * (preds == y_b).sum().item()
                    )
                else:
                    soft_correct += (preds == y_a).sum().item()

                total += y_a.size(0)

        epoch_loss = running_loss / total if total > 0 else 0.0
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
        """
        Validate for one epoch with both patch and slide metrics.
        Verbatim from ViT training_logic.py.
        """
        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        y_preds_list, y_true_list, y_probs_list = [], [], []

        with torch.inference_mode():
            for X, y in val_loader:
                X, y = X.to(self.device), y.to(self.device)

                if self.use_amp and self.autocast is not None:
                    with self.autocast():
                        logits = model(X)
                        loss = loss_fn(logits, y)
                else:
                    logits = model(X)
                    loss = loss_fn(logits, y)

                running_loss += loss.item() * y.size(0)
                preds = logits.argmax(dim=1)
                probs = logits.softmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

                y_preds_list.extend(preds.cpu().numpy())
                y_true_list.extend(y.cpu().numpy())
                y_probs_list.append(probs.cpu().numpy())

        val_loss = running_loss / total if total > 0 else float("inf")
        val_acc = correct / total if total > 0 else 0.0

        all_true = np.array(y_true_list)
        all_preds = np.array(y_preds_list)
        all_probs = np.concatenate(y_probs_list)

        # 1. PATCH-LEVEL METRICS
        patch_metrics = {}
        if y_true_list and y_preds_list:
            try:
                patch_metrics["bal_acc"] = balanced_accuracy_score(
                    all_true, all_preds
                )
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
                patch_metrics["val_auc"] = roc_auc_score(
                    all_true, all_probs[:, 1]
                )
            except Exception as e:
                warnings.warn(f"Could not calculate patch AUC: {e}")
                patch_metrics["val_auc"] = 0.0
        else:
            patch_metrics = {
                "bal_acc": 0.0, "mcc": 0.0, "f1": 0.0, "val_auc": 0.0
            }

        # 2. SLIDE-LEVEL METRICS
        if val_groups is not None:
            from training_utils import compute_slide_level_metrics

            slide_metrics = compute_slide_level_metrics(
                y_true_patches=all_true,
                y_pred_patches=all_preds,
                y_proba_patches=all_probs,
                groups=val_groups,
                num_classes=self.num_classes,
                verbose=False,
            )
        else:
            warnings.warn("val_groups not provided - cannot compute slide metrics")
            slide_metrics = {
                'slide_bal_acc': 0.0,
                'slide_mcc': 0.0,
                'slide_f1': 0.0,
                'slide_auc': 0.0,
                'n_slides': 0,
            }

        return val_loss, val_acc, patch_metrics, slide_metrics

    # ---------------------------------------------------------------------- #
    # Checkpoint helpers — verbatim from ViT training_logic.py               #
    # ---------------------------------------------------------------------- #

    def _load_fold_checkpoint(
        self, iter_idx: int, fold_idx: int
    ) -> Optional[Dict[str, Any]]:
        """Load checkpoint for CV fold. Try Drive first, fall back to local."""
        drive_path = self.checkpoint_manager.get_cv_checkpoint_path(
            iter_idx, fold_idx, use_local=False
        )
        checkpoint = self.checkpoint_manager.load_checkpoint(
            drive_path, verbose=False
        )
        if checkpoint is not None:
            return checkpoint

        local_path = self.checkpoint_manager.get_cv_checkpoint_path(
            iter_idx, fold_idx, use_local=True
        )
        return self.checkpoint_manager.load_checkpoint(local_path, verbose=False)

    def _load_final_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load checkpoint for final model training. Try Drive first."""
        drive_path = self.checkpoint_manager.get_final_checkpoint_path(
            use_local=False
        )
        checkpoint = self.checkpoint_manager.load_checkpoint(
            drive_path, verbose=False
        )
        if checkpoint is not None:
            return checkpoint

        local_path = self.checkpoint_manager.get_final_checkpoint_path(
            use_local=True
        )
        return self.checkpoint_manager.load_checkpoint(local_path, verbose=False)

    # ----------------------------------------------------------------------
    # Fold summary helpers
    # ----------------------------------------------------------------------

    def _create_fold_summary(
        self,
        iter_idx: int,
        fold_idx: int,
        iteration_seed: int,
        model_seed: int,
        best_epoch_idx: int,
        fold_results: Dict[str, List[float]],
        lexicographic_tracker: LexBestModelTracker,
        ema_tracker: ExponentialMovingAverage,
        min_delta: float,
        training_time: float,
        fold_mean: List[float],
        fold_std: List[float],
    ) -> Dict[str, Any]:
        """Create fold summary. Verbatim from ViT training_logic.py."""
        if not fold_results["val_loss"] or best_epoch_idx < 0:
            warnings.warn(
                f"Iter {iter_idx+1} Fold {fold_idx + 1} has no validation results!"
            )
            return self._create_empty_fold_summary(
                iter_idx, fold_idx, iteration_seed, model_seed,
                training_time, fold_mean, fold_std
            )

        return {
            "iteration": iter_idx,
            "fold": fold_idx + 1,
            "iteration_seed": iteration_seed,
            "model_init_seed": model_seed,
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

            "selection_criterion": "lexicographic_slide_ba_val_loss",
            "stopping_criterion": "smoothed_slide_bal_acc",
            "epsilon_slide_ba": lexicographic_tracker.epsilon_A,
            "epsilon_val_loss": lexicographic_tracker.epsilon_L,
            "ema_alpha": ema_tracker.alpha,
            "min_delta": min_delta,

            "training_time": training_time,
            "pixel_channel_mean": fold_mean,
            "pixel_channel_std": fold_std,
        }

    def _create_fold_summary_from_history(
        self,
        history: Dict[str, Any],
        iter_idx: int,
        fold_idx: int,
        iteration_seed: int,
    ) -> Dict[str, Any]:
        """Create fold summary from saved history (when checkpoint deleted)."""
        fold_results = history['history']
        best_epoch_idx = history['best_epoch']

        return {
            "iteration": iter_idx,
            "fold": fold_idx + 1,
            "iteration_seed": iteration_seed,
            "model_init_seed": -1,
            "best_epoch": best_epoch_idx + 1,
            "epochs_trained": history['n_epochs'],

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
            "val_slide_bal_acc_smoothed": fold_results.get(
                "val_slide_bal_acc_smoothed", [0]
            )[best_epoch_idx],
            "val_slide_mcc_best": fold_results["val_slide_mcc"][best_epoch_idx],
            "val_slide_f1_best": fold_results["val_slide_f1"][best_epoch_idx],
            "val_slide_auc_best": fold_results["val_slide_auc"][best_epoch_idx],

            "selection_criterion": "lexicographic_slide_ba_val_loss",
            "stopping_criterion": "smoothed_slide_bal_acc",
            "epsilon_slide_ba": 1e-6,
            "epsilon_val_loss": 1e-6,
            "ema_alpha": 0.0,
            "min_delta": 0.0,

            "training_time": 0.0,
            "pixel_channel_mean": history.get('fold_mean', []),
            "pixel_channel_std": history.get('fold_std', []),
        }

    def _initialize_fold_results(self) -> Dict[str, List[float]]:
        """Initialize empty fold results dictionary."""
        return {
            "train_loss": [],
            "train_acc": [],
            "train_acc_soft": [],
            "val_loss": [],
            "val_acc": [],
            "val_bal_acc": [],
            "val_mcc": [],
            "val_f1": [],
            "val_auc": [],
            "val_slide_bal_acc": [],
            "val_slide_bal_acc_smoothed": [],
            "val_slide_mcc": [],
            "val_slide_f1": [],
            "val_slide_auc": [],
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
        """Create empty fold summary for failed folds."""
        return {
            "iteration": iter_idx,
            "fold": fold_idx + 1,
            "iteration_seed": iteration_seed,
            "model_init_seed": model_seed,
            "best_epoch": 0,
            "epochs_trained": 0,
            "train_loss": float("inf"),
            "train_acc": 0.0,
            "val_loss": float("inf"),
            "val_loss_at_best": float("inf"),
            "val_acc": 0.0,
            "val_bal_acc": 0.0,
            "val_mcc": 0.0,
            "val_f1": 0.0,
            "val_auc": 0.0,
            "val_slide_bal_acc_raw": 0.0,
            "val_slide_bal_acc_smoothed": 0.0,
            "val_slide_mcc_best": 0.0,
            "val_slide_f1_best": 0.0,
            "val_slide_auc_best": 0.0,
            "selection_criterion": "lexicographic_slide_ba_val_loss",
            "stopping_criterion": "smoothed_slide_bal_acc",
            "epsilon_slide_ba": 1e-6,
            "epsilon_val_loss": 1e-6,
            "ema_alpha": 0.0,
            "min_delta": 0.0,
            "training_time": training_time,
            "pixel_channel_mean": fold_mean,
            "pixel_channel_std": fold_std,
        }

    # ---------------------------------------------------------------------- 
    # Utility methods
    # ----------------------------------------------------------------------

    def _set_seeds(self, seed: int):
        """Set all random seeds for full reproducibility."""
        seed = int(seed) & (2**64 - 1)
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

    def _cleanup_memory(self):
        """Clean up GPU memory between folds."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Epoch count selection
# ---------------------------------------------------------------------------

def determine_optimal_epochs(
    cv_fold_details: List[Dict],
    method: str = "median",
    config: Dict = None,
    verbose: bool = True,
) -> int:
    """
    Determine optimal training duration from CV results.

    Args:
        cv_fold_details: List of fold summary dictionaries.
        method: Statistical aggregation method. Default 'median' as per
            the workplan's "Other Aspects" clause.
        config: Configuration dictionary.
        verbose: Print diagnostic information.

    Returns:
        Optimal number of epochs for final training (integer).
    """
    if not cv_fold_details:
        warnings.warn("No CV fold details provided. Using default from config.")
        return config.get('num_epochs', 100)

    try:
        df = pd.DataFrame(cv_fold_details)
    except Exception as e:
        raise ValueError(
            f"Could not convert cv_fold_details to DataFrame: {e}"
        )

    required_cols = ['iteration', 'best_epoch']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"cv_fold_details missing required columns: {missing_cols}. "
            f"Available columns: {df.columns.tolist()}"
        )

    valid_mask = (df['best_epoch'] > 0) & (df['best_epoch'].notna())
    if not valid_mask.any():
        raise RuntimeError(
            "No valid best_epoch values found in cv_fold_details."
        )

    df_valid = df[valid_mask].copy()
    n_valid_folds = len(df_valid)
    n_total_folds = len(df)

    if n_valid_folds < n_total_folds:
        warnings.warn(
            f"Only {n_valid_folds}/{n_total_folds} folds have valid "
            f"best_epoch. Dropped {n_total_folds - n_valid_folds} invalid."
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

    iteration_stats = df_valid.groupby('iteration')['best_epoch'].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max'),
        ('count', 'count'),
    ]).reset_index()

    iteration_means = iteration_stats['mean'].values

    if method == "median":
        optimal_raw = np.median(iteration_means)
        if verbose:
            print(f"\n  Selected method: Median of iteration means")
            print(f"  Raw value: {optimal_raw:.2f} epochs")

    elif method == "percentile_75":
        optimal_raw = np.percentile(iteration_means, 75)
        if verbose:
            print(f"\n  Selected method: 75th percentile of iteration means")
            print(f"  Raw value: {optimal_raw:.2f} epochs")

    elif method == "mean":
        optimal_raw = np.mean(iteration_means)
        if verbose:
            print(f"\n  Selected method: Mean of iteration means")
            print(f"  Raw value: {optimal_raw:.2f} epochs")

    elif method == "mean_plus_std":
        optimal_raw = np.mean(iteration_means) + np.std(iteration_means)
        if verbose:
            print(f"\n  Selected method: Mean + 1 std of iteration means")
            print(f"  Raw value: {optimal_raw:.2f} epochs")

    else:
        warnings.warn(
            f"Unknown method '{method}', defaulting to 'median'."
        )
        optimal_raw = np.median(iteration_means)

    optimal = int(np.ceil(optimal_raw))

    min_epochs = 5
    max_epochs = config.get("num_epochs", 100) * 2 if config else 200

    optimal_bounded = max(min_epochs, min(optimal, max_epochs))

    if optimal != optimal_bounded:
        if verbose:
            print(f"\n  BOUNDS APPLIED:")
            print(f"    Calculated: {optimal} epochs")
            print(f"    Bounded to: [{min_epochs}, {max_epochs}]")
            print(f"    Final:      {optimal_bounded} epochs")
        optimal = optimal_bounded

    if verbose:
        print(f"\n Train final model for {optimal} epochs")
        print(f"{'='*70}\n")

    return optimal
