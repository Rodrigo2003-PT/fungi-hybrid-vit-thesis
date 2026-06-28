"""
Checkpoint Module

Author: Rodrigo Sá
Date: 2025
"""

import glob
import json
import os
import random
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


class CheckpointManager:
    """Manages checkpoints for CV folds and final model training."""

    def __init__(
        self,
        local_checkpoint_dir: str,
        drive_checkpoint_dir: str,
        n_folds: int = 5,
        use_atomic_save: bool = True,
        auto_cleanup: bool = True,
    ):
        self.local_checkpoint_dir = local_checkpoint_dir
        self.drive_checkpoint_dir = drive_checkpoint_dir
        self.n_folds = n_folds
        self.use_atomic_save = use_atomic_save
        self.auto_cleanup = auto_cleanup

        os.makedirs(local_checkpoint_dir, exist_ok=True)
        os.makedirs(drive_checkpoint_dir, exist_ok=True)

        self.histories_dir = os.path.join(drive_checkpoint_dir, 'fold_histories')
        os.makedirs(self.histories_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def get_cv_checkpoint_path(
        self, iter_idx: int, fold_idx: int, use_local: bool = True
    ) -> str:
        d = self.local_checkpoint_dir if use_local else self.drive_checkpoint_dir
        return os.path.join(d, f'cv_iter_{iter_idx}_fold_{fold_idx}_checkpoint.pth')

    def get_fold_history_path(self, iter_idx: int, fold_idx: int) -> str:
        return os.path.join(
            self.histories_dir, f'history_iter_{iter_idx}_fold_{fold_idx}.json'
        )

    def get_final_checkpoint_path(self, use_local: bool = True) -> str:
        d = self.local_checkpoint_dir if use_local else self.drive_checkpoint_dir
        return os.path.join(d, 'final_model_checkpoint.pth')

    def get_master_state_path(self, use_local: bool = True) -> str:
        d = self.local_checkpoint_dir if use_local else self.drive_checkpoint_dir
        return os.path.join(d, 'master_training_state.json')

    def get_results_csv_path(
        self, experiment_name: str, use_local: bool = True
    ) -> str:
        d = self.local_checkpoint_dir if use_local else self.drive_checkpoint_dir
        return os.path.join(d, f'cv_results_{experiment_name}.csv')

    # ------------------------------------------------------------------
    # Core save / load
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        checkpoint_path: str,
        checkpoint_data: Dict[str, Any],
        verbose: bool = True,
    ) -> bool:
        """Save checkpoint with atomic write."""
        self._validate_checkpoint_data(checkpoint_data, checkpoint_path)

        try:
            checkpoint_dir = os.path.dirname(checkpoint_path)

            if self.use_atomic_save:
                can_atomic_rename = self._is_same_filesystem(
                    checkpoint_dir, checkpoint_path
                )
                if can_atomic_rename:
                    temp_path = checkpoint_path + '.tmp'
                    torch.save(checkpoint_data, temp_path)
                    os.replace(temp_path, checkpoint_path)
                else:
                    with tempfile.NamedTemporaryFile(
                        mode='wb',
                        dir=checkpoint_dir,
                        prefix='.tmp_checkpoint_',
                        suffix='.pth',
                        delete=False,
                    ) as tmp_file:
                        temp_path = tmp_file.name
                    try:
                        torch.save(checkpoint_data, temp_path)
                        shutil.move(temp_path, checkpoint_path)
                    except Exception as e:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        raise e
            else:
                torch.save(checkpoint_data, checkpoint_path)

            if verbose:
                msg = f"Checkpoint saved: {os.path.basename(checkpoint_path)}"
                if 'best_epoch' in checkpoint_data:
                    msg += f" (best_epoch={checkpoint_data['best_epoch']})"
                # Log arch identity if present
                arch_meta = checkpoint_data.get('arch_metadata')
                if arch_meta:
                    msg += (
                        f"[hash={arch_meta.get('arch_hash','?')}]"
                    )
                print(msg)
            return True

        except Exception as e:
            print(f"Failed to save checkpoint {os.path.basename(checkpoint_path)}: {e}")
            return False

    def load_checkpoint(
        self,
        checkpoint_path: str,
        verbose: bool = True,
        map_location: str = 'cpu',
    ) -> Optional[Dict[str, Any]]:
        """Load checkpoint from disk."""
        if not os.path.exists(checkpoint_path):
            return None

        try:
            checkpoint = torch.load(
                checkpoint_path, map_location=map_location, weights_only=False
            )
            if verbose:
                msg = f"Checkpoint loaded: {os.path.basename(checkpoint_path)}"
                if 'epoch' in checkpoint:
                    msg += f" (epoch={checkpoint['epoch']})"
                arch_meta = checkpoint.get('arch_metadata')
                if arch_meta:
                    msg += (
                        f"[hash={arch_meta.get('arch_hash','?')}]"
                    )
                print(msg)
            return checkpoint

        except Exception as e:
            print(f"Failed to load checkpoint {checkpoint_path}: {e}")
            return None

    # ------------------------------------------------------------------
    # Fold history
    # ------------------------------------------------------------------

    def save_fold_history(
        self,
        iter_idx: int,
        fold_idx: int,
        history_data: Dict[str, Any],
        verbose: bool = True,
    ) -> bool:
        history_path = self.get_fold_history_path(iter_idx, fold_idx)
        try:
            serializable_data = self._make_json_serializable(history_data)
            with tempfile.NamedTemporaryFile(
                mode='w',
                dir=self.histories_dir,
                prefix='.tmp_history_',
                suffix='.json',
                delete=False,
            ) as tmp_file:
                temp_path = tmp_file.name
                json.dump(serializable_data, tmp_file, indent=2)
            shutil.move(temp_path, history_path)
            if verbose:
                print(f"  History saved: {os.path.basename(history_path)}")
            return True
        except Exception as e:
            print(f"Failed to save fold history: {e}")
            return False

    def load_fold_history(
        self, iter_idx: int, fold_idx: int, verbose: bool = False
    ) -> Optional[Dict[str, Any]]:
        history_path = self.get_fold_history_path(iter_idx, fold_idx)
        if not os.path.exists(history_path):
            return None
        try:
            with open(history_path, 'r') as f:
                data = json.load(f)
            if verbose:
                print(f"  History loaded: {os.path.basename(history_path)}")
            return data
        except Exception as e:
            if verbose:
                print(f"Failed to load fold history: {e}")
            return None

    def load_all_training_histories(
        self, n_iterations: int, n_folds: int, verbose: bool = False
    ) -> List[Dict[str, Any]]:
        histories = []
        for iter_idx in range(n_iterations):
            for fold_idx in range(n_folds):
                history = self.load_fold_history(iter_idx, fold_idx, verbose=False)
                if history is None:
                    if verbose:
                        print(
                            f"  Warning: Could not load history "
                            f"for iter {iter_idx+1}, fold {fold_idx+1}"
                        )
                    continue
                if 'history' not in history or 'n_epochs' not in history:
                    if verbose:
                        print(
                            f"  Warning: Incomplete history "
                            f"for iter {iter_idx+1}, fold {fold_idx+1}"
                        )
                    continue
                histories.append(history)
                if verbose:
                    print(
                        f"  Loaded: Iter {iter_idx+1} Fold {fold_idx+1} "
                        f"({history['n_epochs']} epochs)"
                    )
        if verbose:
            print(f"\nSuccessfully loaded {len(histories)} training histories")
        return histories

    # ------------------------------------------------------------------
    # Cleanup helpers
    # ------------------------------------------------------------------

    def cleanup_previous_fold_checkpoint(
        self, current_iter: int, current_fold: int, verbose: bool = True
    ) -> int:
        if not self.auto_cleanup or (current_iter == 0 and current_fold == 0):
            return 0

        prev_fold = current_fold - 1
        prev_iter = current_iter

        if prev_fold < 0:
            prev_fold = self.n_folds - 1
            prev_iter = current_iter - 1

        if prev_iter < 0:
            return 0

        cleaned = 0
        for use_local in [True, False]:
            path = self.get_cv_checkpoint_path(prev_iter, prev_fold, use_local=use_local)
            if os.path.exists(path):
                try:
                    os.remove(path)
                    cleaned += 1
                    if verbose:
                        loc = "local" if use_local else "Drive"
                        print(
                            f"  Cleaned {loc} checkpoint: "
                            f"iter_{prev_iter}_fold_{prev_fold}"
                        )
                except Exception as e:
                    if verbose:
                        print(f"  Failed to remove {path}: {e}")
        return cleaned

    def cleanup_all_fold_checkpoints(self, verbose: bool = True) -> int:
        if not self.auto_cleanup:
            return 0
        cleaned = 0
        for d in [self.local_checkpoint_dir, self.drive_checkpoint_dir]:
            for p in glob.glob(os.path.join(d, 'cv_iter_*_fold_*_checkpoint.pth')):
                try:
                    os.remove(p)
                    cleaned += 1
                except Exception as e:
                    if verbose:
                        print(f"  Failed to remove {p}: {e}")
        if verbose and cleaned > 0:
            print(f"\n Cleaned up {cleaned} CV checkpoint(s) after analysis completion")
        return cleaned

    def cleanup_fold_histories(self, verbose: bool = True) -> int:
        if not self.auto_cleanup:
            return 0
        cleaned = 0
        for p in glob.glob(
            os.path.join(self.histories_dir, 'history_iter_*_fold_*.json')
        ):
            try:
                os.remove(p)
                cleaned += 1
            except Exception as e:
                if verbose:
                    print(f"  Failed to remove {p}: {e}")
        if verbose and cleaned > 0:
            print(f"\n Cleaned up {cleaned} fold history file(s)")
        return cleaned

    def cleanup_temp_files(self, verbose: bool = True) -> int:
        cleaned = 0
        for loc in [
            self.local_checkpoint_dir,
            self.drive_checkpoint_dir,
            self.histories_dir,
        ]:
            tmps = glob.glob(os.path.join(loc, '*.tmp'))
            tmps += glob.glob(os.path.join(loc, '.tmp_*'))
            for t in tmps:
                try:
                    os.remove(t)
                    cleaned += 1
                    if verbose:
                        print(f"  Removed temp file: {os.path.basename(t)}")
                except Exception as e:
                    if verbose:
                        print(f"  Failed to remove {t}: {e}")
        if verbose and cleaned > 0:
            print(f"Cleaned up {cleaned} temporary file(s)")
        return cleaned

    # ------------------------------------------------------------------
    # Drive backup / sync
    # ------------------------------------------------------------------

    def backup_to_drive(
        self,
        local_path: str,
        drive_path: Optional[str] = None,
        verbose: bool = True,
        verify: bool = True,
    ) -> bool:
        if drive_path is None:
            drive_path = local_path.replace(
                self.local_checkpoint_dir, self.drive_checkpoint_dir
            )
        try:
            os.makedirs(os.path.dirname(drive_path), exist_ok=True)
            shutil.copy2(local_path, drive_path)
            if verify:
                if os.path.getsize(local_path) != os.path.getsize(drive_path):
                    warnings.warn("Backup size mismatch.")
                    return False
            if verbose:
                print(f"  Backed up to Drive: {os.path.basename(drive_path)}")
            return True
        except Exception as e:
            print(f"Drive backup failed: {e}")
            return False

    def sync_from_drive(self, verbose: bool = True) -> int:
        patterns = [
            (self.drive_checkpoint_dir, self.local_checkpoint_dir, '*.pth'),
            (self.drive_checkpoint_dir, self.local_checkpoint_dir, '*.json'),
            (self.drive_checkpoint_dir, self.local_checkpoint_dir, '*.csv'),
        ]
        synced = 0
        for src_dir, dst_dir, pattern in patterns:
            for src_file in glob.glob(os.path.join(src_dir, pattern)):
                if src_file.endswith('.tmp') or os.path.basename(src_file).startswith('.tmp'):
                    continue
                dst_file = os.path.join(dst_dir, os.path.basename(src_file))
                should_copy = not os.path.exists(dst_file)
                if not should_copy:
                    try:
                        should_copy = (
                            os.path.getmtime(src_file) > os.path.getmtime(dst_file)
                        )
                    except OSError:
                        should_copy = True
                if should_copy:
                    try:
                        os.makedirs(dst_dir, exist_ok=True)
                        shutil.copy2(src_file, dst_file)
                        synced += 1
                        if verbose:
                            print(f"  Synced: {os.path.basename(src_file)}")
                    except Exception as e:
                        print(f"  Failed to sync {os.path.basename(src_file)}: {e}")
        if verbose and synced > 0:
            print(f"Sync complete: {synced} file(s) updated")
        return synced

    # ------------------------------------------------------------------
    # Master state
    # ------------------------------------------------------------------

    def save_master_state(
        self, state: Dict[str, Any], backup_to_drive: bool = True
    ) -> bool:
        local_path = self.get_master_state_path(use_local=True)
        try:
            serializable = self._make_json_serializable(state)
            with tempfile.NamedTemporaryFile(
                'w', dir=self.local_checkpoint_dir, delete=False
            ) as f:
                json.dump(serializable, f, indent=2)
                temp_path = f.name
            os.replace(temp_path, local_path)
            if backup_to_drive:
                drive_path = self.get_master_state_path(use_local=False)
                shutil.copy2(local_path, drive_path)
            return True
        except Exception as e:
            print(f"Failed to save master state: {e}")
            return False

    def load_master_state(
        self, prefer_drive: bool = True
    ) -> Optional[Dict[str, Any]]:
        local_path = self.get_master_state_path(use_local=True)
        drive_path = self.get_master_state_path(use_local=False)
        path_to_load = None
        if prefer_drive and os.path.exists(drive_path):
            path_to_load = drive_path
        elif os.path.exists(local_path):
            path_to_load = local_path
        elif os.path.exists(drive_path):
            path_to_load = drive_path
        if path_to_load is None:
            return None
        try:
            with open(path_to_load, 'r') as f:
                state = json.load(f)
            print(f"Loaded master state from: {os.path.basename(path_to_load)}")
            return state
        except Exception as e:
            print(f"Failed to load master state: {e}")
            return None

    # ------------------------------------------------------------------
    # CV checkpoint creation
    # ------------------------------------------------------------------

    def create_cv_checkpoint(
        self,
        iter_idx: int,
        fold_idx: int,
        current_epoch: int,
        best_epoch: int,
        model,
        optimizer,
        scheduler,
        fold_results: Dict,
        epochs_no_improve: int,
        fold_mean: list,
        fold_std: list,
        scaler=None,
        best_model_state=None,
        best_slide_metric_raw: float = -float('inf'),
        lexicographic_state: Dict = None,
        ema_state: Dict = None,
        patience_state: Dict = None,
        best_val_loss: float = float('inf'),
        arch_metadata: Optional[Dict[str, Any]] = None,  # NEW
    ) -> Tuple[bool, bool]:
        """
        Create CV fold checkpoint.
        """
        checkpoint_data: Dict[str, Any] = {
            'iter_idx': iter_idx,
            'fold_idx': fold_idx,
            'epoch': current_epoch,
            'best_epoch': best_epoch,
            'epochs_no_improve': epochs_no_improve,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'best_model_state_dict': best_model_state,
            'fold_results': fold_results,
            'fold_mean': fold_mean,
            'fold_std': fold_std,
            'best_slide_metric_raw': best_slide_metric_raw,
            'best_val_loss': best_val_loss,
            'lexicographic_state': lexicographic_state,
            'ema_state': ema_state,
            'patience_state': patience_state,
            'rng_state': {
                'torch': torch.get_rng_state(),
                'torch_cuda': (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available() else None
                ),
                'numpy': np.random.get_state(),
                'random': random.getstate(),
            },
        }
        if arch_metadata is not None:
            checkpoint_data['arch_metadata'] = arch_metadata

        local_path = self.get_cv_checkpoint_path(iter_idx, fold_idx, use_local=True)
        checkpoint_success = self.save_checkpoint(local_path, checkpoint_data, verbose=True)

        if checkpoint_success:
            drive_path = self.get_cv_checkpoint_path(iter_idx, fold_idx, use_local=False)
            self.backup_to_drive(local_path, drive_path, verbose=False)

        # Lightweight history
        history_data = {
            'iteration': iter_idx,
            'fold': fold_idx + 1,
            'history': fold_results,
            'n_epochs': len(fold_results.get('train_loss', [])),
            'best_epoch': best_epoch,
            'fold_mean': fold_mean,
            'fold_std': fold_std,
            'best_slide_metric_raw': best_slide_metric_raw,
            'best_val_loss': best_val_loss,
        }
        if arch_metadata is not None:
            history_data['arch_metadata'] = arch_metadata

        history_success = self.save_fold_history(
            iter_idx, fold_idx, history_data, verbose=False
        )

        return checkpoint_success, history_success

    # ------------------------------------------------------------------
    # Final model checkpoint
    # ------------------------------------------------------------------

    def create_final_checkpoint(
        self,
        epoch: int,
        model,
        optimizer,
        scheduler,
        training_history: Dict,
        mean: list,
        std: list,
        scaler=None,
        arch_metadata: Optional[Dict[str, Any]] = None,  # NEW
    ) -> bool:
        """
        Create final model checkpoint.
        """
        checkpoint_data: Dict[str, Any] = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'training_history': training_history,
            'mean': mean,
            'std': std,
            'rng_state': {
                'torch': torch.get_rng_state(),
                'torch_cuda': (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available() else None
                ),
                'numpy': np.random.get_state(),
                'random': random.getstate(),
            },
        }

        if arch_metadata is not None:
            checkpoint_data['arch_metadata'] = arch_metadata

        local_path = self.get_final_checkpoint_path(use_local=True)
        success = self.save_checkpoint(local_path, checkpoint_data)

        if success:
            drive_path = self.get_final_checkpoint_path(use_local=False)
            self.backup_to_drive(local_path, drive_path, verbose=True)

        return success

    # ------------------------------------------------------------------
    # State restoration
    # ------------------------------------------------------------------

    def restore_training_state(
        self,
        checkpoint: Dict,
        model,
        optimizer,
        scheduler=None,
        scaler=None,
    ) -> None:
        """Restore training state from checkpoint."""
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if scaler and checkpoint.get('scaler_state_dict'):
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

        rng_state = checkpoint.get('rng_state', {})
        if 'torch' in rng_state:
            torch.set_rng_state(rng_state['torch'])
        if (
            'torch_cuda' in rng_state
            and rng_state['torch_cuda'] is not None
            and torch.cuda.is_available()
        ):
            torch.cuda.set_rng_state_all(rng_state['torch_cuda'])
        if 'numpy' in rng_state:
            np.random.set_state(rng_state['numpy'])
        if 'random' in rng_state:
            random.setstate(rng_state['random'])

        print("Model, optimizer, scheduler, scaler, and RNG states restored")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_checkpoint_data(
        self, checkpoint_data: Dict[str, Any], checkpoint_path: str
    ) -> None:
        if 'fold_idx' in checkpoint_data:
            required = ['best_epoch', 'epochs_no_improve']
            missing = [f for f in required if f not in checkpoint_data]
            if missing:
                raise ValueError(f"Checkpoint missing required fields: {missing}")
            if checkpoint_data['best_epoch'] < 0:
                warnings.warn(
                    f"Invalid best_epoch={checkpoint_data['best_epoch']} in checkpoint."
                )

    def _is_same_filesystem(self, path1: str, path2: str) -> bool:
        try:
            s1 = os.stat(path1)
            s2 = os.stat(os.path.dirname(path2))
            return s1.st_dev == s2.st_dev
        except Exception:
            return False

    def _make_json_serializable(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        return obj


# ---------------------------------------------------------------------------
# Master state initialiser
# ---------------------------------------------------------------------------

def initialize_master_state() -> Dict[str, Any]:
    """Create initial master training state."""
    return {
        'phase': 'cross_validation',
        'current_iteration': 0,
        'current_fold': 0,
        'all_iterations_completed': False,
        'cv_fold_details': [],
        'final_training_completed': False,
    }
