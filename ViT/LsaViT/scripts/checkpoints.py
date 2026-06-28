"""
Checkpoint Management Module

Author: Rodrigo Sá
Date: 2025
"""

import os
import json
import shutil
import glob
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple
import tempfile
import warnings
import random


class CheckpointManager:
    """
    Manages checkpoints
    """
    
    def __init__(
        self, 
        local_checkpoint_dir: str, 
        drive_checkpoint_dir: str,
        n_folds: int = 5, 
        use_atomic_save: bool = True,
        auto_cleanup: bool = True
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
    
    def get_cv_checkpoint_path(self, iter_idx: int, fold_idx: int, use_local: bool = True) -> str:
        checkpoint_dir = self.local_checkpoint_dir if use_local else self.drive_checkpoint_dir
        return os.path.join(checkpoint_dir, f'cv_iter_{iter_idx}_fold_{fold_idx}_checkpoint.pth')
    
    def get_fold_history_path(self, iter_idx: int, fold_idx: int) -> str:
        return os.path.join(self.histories_dir, f'history_iter_{iter_idx}_fold_{fold_idx}.json')
    
    def get_final_checkpoint_path(self, use_local: bool = True) -> str:
        checkpoint_dir = self.local_checkpoint_dir if use_local else self.drive_checkpoint_dir
        return os.path.join(checkpoint_dir, 'final_model_checkpoint.pth')
    
    def get_master_state_path(self, use_local: bool = True) -> str:
        checkpoint_dir = self.local_checkpoint_dir if use_local else self.drive_checkpoint_dir
        return os.path.join(checkpoint_dir, 'master_training_state.json')
    
    def get_results_csv_path(self, experiment_name: str, use_local: bool = True) -> str:
        checkpoint_dir = self.local_checkpoint_dir if use_local else self.drive_checkpoint_dir
        return os.path.join(checkpoint_dir, f'cv_results_{experiment_name}.csv')
    
    def save_checkpoint(self, checkpoint_path: str, checkpoint_data: Dict[str, Any], verbose: bool = True) -> bool:
        """Save checkpoint with atomic write."""
        self._validate_checkpoint_data(checkpoint_data, checkpoint_path)
        
        try:
            checkpoint_dir = os.path.dirname(checkpoint_path)
            
            if self.use_atomic_save:
                can_atomic_rename = self._is_same_filesystem(checkpoint_dir, checkpoint_path)
                
                if can_atomic_rename:
                    temp_path = checkpoint_path + '.tmp'
                    torch.save(checkpoint_data, temp_path)
                    os.replace(temp_path, checkpoint_path)
                else:
                    with tempfile.NamedTemporaryFile(mode='wb', dir=checkpoint_dir, prefix='.tmp_checkpoint_', suffix='.pth', delete=False) as tmp_file:
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
                print(msg)
            return True
            
        except Exception as e:
            print(f"Failed to save checkpoint {os.path.basename(checkpoint_path)}: {e}")
            return False
    
    def load_checkpoint(self, checkpoint_path: str, verbose: bool = True, map_location: str = 'cpu') -> Optional[Dict[str, Any]]:
        """Load checkpoint"""
        if not os.path.exists(checkpoint_path):
            return None
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
            
            if verbose:
                msg = f"Checkpoint loaded: {os.path.basename(checkpoint_path)}"
                if 'epoch' in checkpoint:
                    msg += f" (epoch={checkpoint['epoch']})"
                print(msg)
            
            return checkpoint
            
        except Exception as e:
            print(f"Failed to load checkpoint {checkpoint_path}: {e}")
            return None
    
    def save_fold_history(self, iter_idx: int, fold_idx: int, history_data: Dict[str, Any], verbose: bool = True) -> bool:
        """Save lightweight fold history"""
        history_path = self.get_fold_history_path(iter_idx, fold_idx)
        
        try:
            serializable_data = self._make_json_serializable(history_data)
            
            with tempfile.NamedTemporaryFile(mode='w', dir=self.histories_dir, prefix='.tmp_history_', suffix='.json', delete=False) as tmp_file:
                temp_path = tmp_file.name
                json.dump(serializable_data, tmp_file, indent=2)
            
            shutil.move(temp_path, history_path)
            
            if verbose:
                print(f"  History saved: {os.path.basename(history_path)}")
            
            return True
            
        except Exception as e:
            print(f"Failed to save fold history: {e}")
            return False
    
    def load_fold_history(self, iter_idx: int, fold_idx: int, verbose: bool = False) -> Optional[Dict[str, Any]]:
        """Load lightweight fold history."""
        history_path = self.get_fold_history_path(iter_idx, fold_idx)
        
        if not os.path.exists(history_path):
            return None
        
        try:
            with open(history_path, 'r') as f:
                history_data = json.load(f)
            
            if verbose:
                print(f"  History loaded: {os.path.basename(history_path)}")
            
            return history_data
            
        except Exception as e:
            if verbose:
                print(f"Failed to load fold history: {e}")
            return None
    
    def load_all_training_histories(self, n_iterations: int, n_folds: int, verbose: bool = False) -> List[Dict[str, Any]]:
        """Load all fold histories for dynamics analysis."""
        histories = []
        
        for iter_idx in range(n_iterations):
            for fold_idx in range(n_folds):
                history = self.load_fold_history(iter_idx, fold_idx, verbose=False)
                
                if history is None:
                    if verbose:
                        print(f"  Warning: Could not load history for iter {iter_idx+1}, fold {fold_idx+1}")
                    continue
                
                if 'history' not in history or 'n_epochs' not in history:
                    if verbose:
                        print(f"  Warning: Incomplete history for iter {iter_idx+1}, fold {fold_idx+1}")
                    continue
                
                histories.append(history)
                
                if verbose:
                    print(f"  Loaded: Iter {iter_idx+1} Fold {fold_idx+1} ({history['n_epochs']} epochs)")
        
        if verbose:
            print(f"\nSuccessfully loaded {len(histories)} training histories")
        
        return histories
    
    def cleanup_previous_fold_checkpoint(self, current_iter: int, current_fold: int, verbose: bool = True) -> int:
        """Remove previous fold's checkpoint"""
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
            checkpoint_path = self.get_cv_checkpoint_path(prev_iter, prev_fold, use_local=use_local)
            if os.path.exists(checkpoint_path):
                try:
                    os.remove(checkpoint_path)
                    cleaned += 1
                    if verbose:
                        location = "local" if use_local else "Drive"
                        print(f"  Cleaned {location} checkpoint: iter_{prev_iter}_fold_{prev_fold}")
                except Exception as e:
                    if verbose:
                        print(f"  Failed to remove {checkpoint_path}: {e}")
        
        return cleaned
    
    def cleanup_all_fold_checkpoints(self, verbose: bool = True) -> int:
        """Remove ALL fold checkpoints after CV completes."""
        if not self.auto_cleanup:
            return 0
        
        cleaned = 0
        
        for checkpoint_dir in [self.local_checkpoint_dir, self.drive_checkpoint_dir]:
            pattern = os.path.join(checkpoint_dir, 'cv_iter_*_fold_*_checkpoint.pth')
            for checkpoint_path in glob.glob(pattern):
                try:
                    os.remove(checkpoint_path)
                    cleaned += 1
                except Exception as e:
                    if verbose:
                        print(f"  Failed to remove {checkpoint_path}: {e}")
        
        if verbose and cleaned > 0:
            print(f"\n Cleaned up {cleaned} CV checkpoint(s) after analysis completion")
        
        return cleaned
    
    def cleanup_fold_histories(self, verbose: bool = True) -> int:
        """Remove fold histories after evaluation completes."""
        if not self.auto_cleanup:
            return 0
        
        cleaned = 0
        history_pattern = os.path.join(self.histories_dir, 'history_iter_*_fold_*.json')
        
        for history_path in glob.glob(history_pattern):
            try:
                os.remove(history_path)
                cleaned += 1
            except Exception as e:
                if verbose:
                    print(f"  Failed to remove {history_path}: {e}")
        
        if verbose and cleaned > 0:
            print(f"\n Cleaned up {cleaned} fold history file(s) after evaluation")
        
        return cleaned
    
    def cleanup_temp_files(self, verbose: bool = True) -> int:
        """Clean orphaned temporary files."""
        cleaned = 0
        
        for location in [self.local_checkpoint_dir, self.drive_checkpoint_dir, self.histories_dir]:
            tmp_files = glob.glob(os.path.join(location, '*.tmp'))
            tmp_files.extend(glob.glob(os.path.join(location, '.tmp_*')))
            
            for tmp_file in tmp_files:
                try:
                    os.remove(tmp_file)
                    cleaned += 1
                    if verbose:
                        print(f"  Removed temp file: {os.path.basename(tmp_file)}")
                except Exception as e:
                    if verbose:
                        print(f"  Failed to remove {tmp_file}: {e}")
        
        if verbose and cleaned > 0:
            print(f"Cleaned up {cleaned} temporary file(s)")
        
        return cleaned
    
    def backup_to_drive(self, local_path: str, drive_path: Optional[str] = None, verbose: bool = True, verify: bool = True) -> bool:
        """Copy from local to Drive with verification."""
        if drive_path is None:
            drive_path = local_path.replace(self.local_checkpoint_dir, self.drive_checkpoint_dir)
        
        try:
            os.makedirs(os.path.dirname(drive_path), exist_ok=True)
            shutil.copy2(local_path, drive_path)
            
            if verify:
                local_size = os.path.getsize(local_path)
                drive_size = os.path.getsize(drive_path)
                if local_size != drive_size:
                    warnings.warn(f"Size mismatch: local={local_size}, drive={drive_size}")
                    return False
            
            if verbose:
                print(f"  Backed up to Drive: {os.path.basename(drive_path)}")
            return True
            
        except Exception as e:
            print(f"Drive backup failed: {e}")
            return False
    
    def sync_from_drive(self, verbose: bool = True) -> int:
        """Sync checkpoints and histories from Drive on startup."""
        sync_patterns = [
            (self.drive_checkpoint_dir, self.local_checkpoint_dir, '*.pth'),
            (self.drive_checkpoint_dir, self.local_checkpoint_dir, '*.json'),
            (self.drive_checkpoint_dir, self.local_checkpoint_dir, '*.csv'),
        ]
        
        synced_count = 0
        
        for source_dir, dest_dir, pattern in sync_patterns:
            source_files = [f for f in glob.glob(os.path.join(source_dir, pattern)) if not f.endswith('.tmp') and not os.path.basename(f).startswith('.tmp')]
            
            for source_file in source_files:
                dest_file = os.path.join(dest_dir, os.path.basename(source_file))
                should_copy = not os.path.exists(dest_file)
                
                if not should_copy:
                    try:
                        should_copy = os.path.getmtime(source_file) > os.path.getmtime(dest_file)
                    except OSError:
                        should_copy = True
                
                if should_copy:
                    try:
                        os.makedirs(dest_dir, exist_ok=True)
                        shutil.copy2(source_file, dest_file)
                        synced_count += 1
                        if verbose:
                            print(f"  Synced: {os.path.basename(source_file)}")
                    except Exception as e:
                        print(f"  Failed to sync {os.path.basename(source_file)}: {e}")
        
        if verbose and synced_count > 0:
            print(f"Sync complete: {synced_count} file(s) updated")
        
        return synced_count
    
    def save_master_state(self, state: Dict[str, Any], backup_to_drive: bool = True) -> bool:
        """Save master training state."""
        local_path = self.get_master_state_path(use_local=True)
        
        try:
            serializable_state = self._make_json_serializable(state)
            
            with tempfile.NamedTemporaryFile('w', dir=self.local_checkpoint_dir, delete=False) as f:
                json.dump(serializable_state, f, indent=2)
                temp_path = f.name
            
            os.replace(temp_path, local_path)
            
            if backup_to_drive:
                drive_path = self.get_master_state_path(use_local=False)
                shutil.copy2(local_path, drive_path)
            
            return True
            
        except Exception as e:
            print(f"Failed to save master state: {e}")
            return False
    
    def load_master_state(self, prefer_drive: bool = True) -> Optional[Dict[str, Any]]:
        """Load master training state."""
        local_path = self.get_master_state_path(use_local=True)
        drive_path = self.get_master_state_path(use_local=False)
        
        path_to_load = drive_path if prefer_drive and os.path.exists(drive_path) else (local_path if os.path.exists(local_path) else (drive_path if os.path.exists(drive_path) else None))
        
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
        best_val_loss: float = float('inf')
    ) -> Tuple[bool, bool]:
        """
        Create CV checkpoint
        """
        checkpoint_data = {
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
                'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                'numpy': np.random.get_state(),
                'random': random.getstate()
            }
        }
        
        local_checkpoint_path = self.get_cv_checkpoint_path(iter_idx, fold_idx, use_local=True)
        checkpoint_success = self.save_checkpoint(local_checkpoint_path, checkpoint_data, verbose=True)
        
        if checkpoint_success:
            drive_checkpoint_path = self.get_cv_checkpoint_path(iter_idx, fold_idx, use_local=False)
            self.backup_to_drive(local_checkpoint_path, drive_checkpoint_path, verbose=False)
        
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
        
        history_success = self.save_fold_history(iter_idx, fold_idx, history_data, verbose=False)
        
        return checkpoint_success, history_success
    
    def create_final_checkpoint(self, epoch: int, model, optimizer, scheduler, training_history: Dict, mean: list, std: list, scaler=None) -> bool:
        """Create final model checkpoint."""
        checkpoint_data = {
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
                'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                'numpy': np.random.get_state(),
                'random': random.getstate()
            }
        }
        
        local_path = self.get_final_checkpoint_path(use_local=True)
        success = self.save_checkpoint(local_path, checkpoint_data)
        
        if success:
            drive_path = self.get_final_checkpoint_path(use_local=False)
            self.backup_to_drive(local_path, drive_path, verbose=True)
        
        return success
    
    def restore_training_state(self, checkpoint: Dict, model, optimizer, scheduler=None, scaler=None) -> None:
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
        
        if 'torch_cuda' in rng_state and rng_state['torch_cuda'] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng_state['torch_cuda'])
        
        if 'numpy' in rng_state:
            np.random.set_state(rng_state['numpy'])
        
        if 'random' in rng_state:
            random.setstate(rng_state['random'])
        
        print("Model, optimizer, scheduler, scaler, and RNG states restored")
    
    def _validate_checkpoint_data(self, checkpoint_data: Dict[str, Any], checkpoint_path: str) -> None:
        """Validate checkpoint before saving."""
        if 'fold_idx' in checkpoint_data:
            required = ['best_epoch', 'epochs_no_improve']
            missing = [f for f in required if f not in checkpoint_data]
            
            if missing:
                raise ValueError(f"Checkpoint missing: {missing}")
            
            if checkpoint_data['best_epoch'] < 0:
                warnings.warn(f"Invalid best_epoch={checkpoint_data['best_epoch']}")
    
    def _is_same_filesystem(self, path1: str, path2: str) -> bool:
        """Check if paths are on same filesystem."""
        try:
            stat1 = os.stat(path1)
            stat2 = os.stat(os.path.dirname(path2))
            return stat1.st_dev == stat2.st_dev
        except:
            return False
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format."""
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