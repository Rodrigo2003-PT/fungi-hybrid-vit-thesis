"""
Author: Rodrigo Sá
Date: 2026
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


class CrossValidationStatistics:
    """
    Attributes:
        n_iterations: Number of independent iterations (default: 30)
        n_folds: Number of folds per iteration (default: 5)
        confidence_level: Confidence level for intervals (default: 0.95)
    """
    
    def __init__(
        self,
        n_iterations: int = 30,
        n_folds: int = 5,
        confidence_level: float = 0.95
    ):
        self.n_iterations = n_iterations
        self.n_folds = n_folds
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        
    def load_summary_csv(self, csv_path: str) -> pd.DataFrame:
        """
        Load and validate CV results summary CSV.
        
        Args:
            csv_path: Path to CSV file containing fold-level results
            
        Returns:
            Cleaned DataFrame with standardized column names
            
        Raises:
            FileNotFoundError: If CSV file does not exist
            ValueError: If required columns are missing
        """
        if not Path(csv_path).exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        drop_cols = [
            'ema_alpha', 'min_delta', 'epsilon_val_loss', 'epsilon_slide_ba', 
            'stopping_criterion', 'selection_criterion', 'training_time', 
            'iteration_seed_base', 'iteration_seed_effective', 'val_loss_at_best'
        ]
        df = df.drop(columns=drop_cols, errors='ignore')

        rename_map = {
            'val_slide_bal_acc_raw': 'val_slide_bal_acc',
            'val_slide_mcc_best': 'val_slide_mcc',
            'val_slide_f1_best': 'val_slide_f1',
            'val_slide_auc_best': 'val_slide_auc'
        }
        df = df.rename(columns=rename_map)

        required_keys = ['iteration', 'fold']
        required_patch = ['val_acc', 'val_bal_acc', 'val_mcc', 'val_f1', 'val_auc']
        required_slide = [
            'val_slide_bal_acc', 'val_slide_bal_acc_smoothed',
            'val_slide_mcc', 'val_slide_f1', 'val_slide_auc'
        ]
        required_loss = ['val_loss', 'train_loss']
        
        required_cols = required_keys + required_patch + required_slide + required_loss
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(
                f"CSV missing required columns: {missing_cols}\n"
                f"Available columns: {df.columns.tolist()}"
            )
        return df
    
    def compute_hierarchical_statistics(
        self,
        df: pd.DataFrame,
        metrics: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Args:
            df: DataFrame with columns ['iteration', 'fold', ...metrics]
            metrics: List of metric column names
            
        Returns:
            Dictionary mapping metric names to statistical summaries:
            {metric: {'mean', 'std', 'ci_lower', 'ci_upper', 'n_iterations'}}
        """
        results = {}
        
        for metric in metrics:
            if metric not in df.columns:
                warnings.warn(f"Metric '{metric}' not found in DataFrame, skipping")
                continue
            
            # Compute iteration-level means (average k folds)
            iteration_means = df.groupby('iteration')[metric].mean().values
            n_iter = len(iteration_means)
            
            if n_iter == 0:
                warnings.warn(f"No valid data for metric '{metric}'")
                continue
            
            # Compute statistics across N iteration means
            global_mean = np.mean(iteration_means)
            global_std = np.std(iteration_means, ddof=1)  # Between-iteration std
            
            # 95% Confidence Interval using t-distribution
            if n_iter > 1:
                t_critical = stats.t.ppf(1 - self.alpha/2, df=n_iter - 1)
                margin = t_critical * (global_std / np.sqrt(n_iter))
                ci_lower = global_mean - margin
                ci_upper = global_mean + margin
            else:
                ci_lower = global_mean
                ci_upper = global_mean
            
            results[metric] = {
                'mean': float(global_mean),
                'std': float(global_std),
                'ci_lower': float(ci_lower),
                'ci_upper': float(ci_upper),
                'n_iterations': int(n_iter)
            }
        
        return results
    
    def generate_latex_table(
        self,
        patch_stats: Dict[str, Dict[str, float]],
        slide_stats: Dict[str, Dict[str, float]],
        output_path: Optional[str] = None
    ) -> str:
        """  
        Args:
            patch_stats: Statistics for patch-level metrics
            slide_stats: Statistics for slide-level metrics
            output_path: Optional path to save .tex file
            
        Returns:
            LaTeX table string
        """
        latex_lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Performance Metrics (Mean ± Std, 95\% CI)}",
            r"\label{tab:cv_results}",
            r"\begin{tabular}{lcc}",
            r"\toprule",
            r"\textbf{Metric} & \textbf{Patch-Level} & \textbf{Slide-Level} \\",
            r"\midrule"
        ]
        
        # Metric display names
        metric_names = {
            'val_bal_acc': 'Balanced Accuracy',
            'val_slide_bal_acc': 'Balanced Accuracy',
            'val_mcc': 'MCC',
            'val_slide_mcc': 'MCC',
            'val_f1': 'F1-Score',
            'val_slide_f1': 'F1-Score',
            'val_auc': 'AUC',
            'val_slide_auc': 'AUC'
        }
        
        # Match patch and slide metrics
        metric_pairs = [
            ('val_bal_acc', 'val_slide_bal_acc'),
            ('val_mcc', 'val_slide_mcc'),
            ('val_f1', 'val_slide_f1'),
            ('val_auc', 'val_slide_auc')
        ]
        
        for patch_key, slide_key in metric_pairs:
            if patch_key in patch_stats and slide_key in slide_stats:
                p_stat = patch_stats[patch_key]
                s_stat = slide_stats[slide_key]
                
                name = metric_names.get(patch_key, patch_key)
                
                patch_str = (
                    f"{p_stat['mean']:.4f} $\\pm$ {p_stat['std']:.4f} "
                    f"[{p_stat['ci_lower']:.4f}, {p_stat['ci_upper']:.4f}]"
                )
                slide_str = (
                    f"{s_stat['mean']:.4f} $\\pm$ {s_stat['std']:.4f} "
                    f"[{s_stat['ci_lower']:.4f}, {s_stat['ci_upper']:.4f}]"
                )
                
                latex_lines.append(
                    f"{name} & {patch_str} & {slide_str} \\\\"
                )
        
        latex_lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}"
        ])
        
        latex_table = "\n".join(latex_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(latex_table)
        
        return latex_table
    
    def load_training_histories(
        self,
        history_dir: str
    ) -> List[Dict]:
        """
        Load all training history JSON files from directory.
        
        Args:
            history_dir: Directory containing history_iter_*.json files
            
        Returns:
            List of history dictionaries with structure:
            {
                'iteration': int,
                'fold': int,
                'history': dict with metrics,
                'n_epochs': int,
                ...
            }
            
        Raises:
            FileNotFoundError: If directory or no history files found
        """
        history_path = Path(history_dir)
        if not history_path.exists():
            raise FileNotFoundError(f"History directory not found: {history_dir}")
        
        history_files = sorted(history_path.glob("history_iter_*.json"))
        
        if not history_files:
            raise FileNotFoundError(
                f"No history_iter_*.json files found in {history_dir}"
            )
        
        histories = []
        failed_files = []
        
        for file_path in history_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                if 'history' not in data or 'n_epochs' not in data:
                    warnings.warn(f"Invalid structure in {file_path.name}, skipping")
                    failed_files.append(file_path.name)
                    continue
                
                # Parse iteration and fold from filename
                stem = file_path.stem  # e.g., "history_iter_0_fold_0"
                parts = stem.split('_')
                try:
                    iter_idx = int(parts[2])
                    fold_idx = int(parts[4])
                    data['iteration'] = iter_idx
                    data['fold'] = fold_idx
                except (IndexError, ValueError):
                    warnings.warn(f"Could not parse iteration/fold from {file_path.name}")
                    failed_files.append(file_path.name)
                    continue
                
                histories.append(data)
                
            except Exception as e:
                warnings.warn(f"Failed to load {file_path.name}: {e}")
                failed_files.append(file_path.name)
        
        print(f" Loaded {len(histories)} training histories")
        if failed_files:
            print(f"   Failed to load {len(failed_files)} files: {failed_files[:5]}...")
        
        return histories
    
    def analyze_convergence_dynamics(
        self,
        histories: List[Dict]
    ) -> Dict[str, np.ndarray]:
        """
        Args:
            histories: List of training history dictionaries with:
                - 'iteration': iteration index
                - 'fold': fold index
                - 'history': dict with 'train_loss', 'val_loss', etc.
                - 'n_epochs': number of epochs trained
                
        Returns:
            Dictionary containing:
                - epochs: Array [0, 1, ..., T_max-1]
                - train_loss_mean, train_loss_std (between-iteration)
                - val_loss_mean, val_loss_std (between-iteration)
                - generalization_gap_mean, generalization_gap_std
        """
        if not histories:
            raise ValueError("No training histories provided")
        
        iter_fold_pairs = [(h['iteration'], h['fold']) for h in histories]
        iterations = sorted(set(i for i, f in iter_fold_pairs))
        n_iterations = len(iterations)
        
        T_max = max(h['n_epochs'] for h in histories)
        
        # Aggregate fold-level curves within each iteration
        iteration_train_curves = {}
        iteration_val_curves = {}
        
        for iter_idx in iterations:
            iter_histories = [h for h in histories if h['iteration'] == iter_idx]
            k_folds = len(iter_histories)
            
            # Initialize arrays for this iteration
            iter_train = np.full((T_max, k_folds), np.nan)
            iter_val = np.full((T_max, k_folds), np.nan)
            
            for fold_idx, h in enumerate(iter_histories):
                n_epochs = h['n_epochs']
                hist = h['history']
                
                if 'train_loss' in hist and len(hist['train_loss']) == n_epochs:
                    iter_train[:n_epochs, fold_idx] = hist['train_loss']
                    if n_epochs < T_max:
                        last_val = hist['train_loss'][-1]
                        iter_train[n_epochs:, fold_idx] = last_val
                
                if 'val_loss' in hist and len(hist['val_loss']) == n_epochs:
                    iter_val[:n_epochs, fold_idx] = hist['val_loss']
                    if n_epochs < T_max:
                        last_val = hist['val_loss'][-1]
                        iter_val[n_epochs:, fold_idx] = last_val

            iteration_train_curves[iter_idx] = np.nanmean(iter_train, axis=1)
            iteration_val_curves[iter_idx] = np.nanmean(iter_val, axis=1)
        
        # Compute between-iteration statistics
        # Stack all iteration curves (N_iterations × T_max)
        train_iter_array = np.array([iteration_train_curves[i] for i in iterations])
        val_iter_array = np.array([iteration_val_curves[i] for i in iterations])
        
        # Compute mean and std across iterations (axis=0)
        train_loss_mean = np.mean(train_iter_array, axis=0)
        train_loss_std = np.std(train_iter_array, axis=0, ddof=1)
        
        val_loss_mean = np.mean(val_iter_array, axis=0)
        val_loss_std = np.std(val_iter_array, axis=0, ddof=1)
        
        # Generalization gap
        gap_iter_array = val_iter_array - train_iter_array
        gap_mean = np.mean(gap_iter_array, axis=0)
        gap_std = np.std(gap_iter_array, axis=0, ddof=1)
        
        return {
            'epochs': np.arange(T_max),
            'train_loss_mean': train_loss_mean,
            'train_loss_std': train_loss_std,
            'val_loss_mean': val_loss_mean,
            'val_loss_std': val_loss_std,
            'generalization_gap_mean': gap_mean,
            'generalization_gap_std': gap_std,
        }
    
    def plot_learning_curves(
        self,
        dynamics: Dict[str, np.ndarray],
        output_path: Optional[str] = None,
        show: bool = False
    ):
        """
        Plot training dynamics with LOCF-imputed trajectories.
        
        Args:
            dynamics: Output from analyze_convergence_dynamics()
            output_path: Optional path to save figure
            show: Whether to display plot
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        epochs = dynamics['epochs'] + 1
        
        # Subplot 1: Loss curves
        ax1.plot(epochs, dynamics['train_loss_mean'], 
                label='Training Loss', color='#2E86AB', linewidth=2)
        ax1.fill_between(
            epochs,
            dynamics['train_loss_mean'] - dynamics['train_loss_std'],
            dynamics['train_loss_mean'] + dynamics['train_loss_std'],
            alpha=0.2, color='#2E86AB', 
            label='±1 SD (between-iteration)'
        )
        
        ax1.plot(epochs, dynamics['val_loss_mean'], 
                label='Validation Loss', color='#A23B72', linewidth=2)
        ax1.fill_between(
            epochs,
            dynamics['val_loss_mean'] - dynamics['val_loss_std'],
            dynamics['val_loss_mean'] + dynamics['val_loss_std'],
            alpha=0.2, color='#A23B72',
            label='±1 SD (between-iteration)'
        )
        
        ax1.set_ylabel('Loss (Mean ± SD)', fontsize=12)
        ax1.set_title('Training Dynamics: Loss Curves (LOCF-Imputed)', 
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Generalization gap
        ax2.plot(epochs, dynamics['generalization_gap_mean'], 
                color='#F18F01', linewidth=2, label='Generalization Gap')
        ax2.fill_between(
            epochs,
            dynamics['generalization_gap_mean'] - dynamics['generalization_gap_std'],
            dynamics['generalization_gap_mean'] + dynamics['generalization_gap_std'],
            alpha=0.2, color='#F18F01',
            label='±1 SD (between-iteration)'
        )
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Gap: Val Loss - Train Loss', fontsize=12)
        ax2.set_title('Overfitting Dynamics (Generalization Gap)', 
                     fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_metric_distributions(
            self,
            df: pd.DataFrame,
            metric: str = 'val_slide_bal_acc',
            output_path: Optional[str] = None,
            show: bool = False
        ):
            """
            Visualize distribution of iteration-level metric scores.
            """
            if metric not in df.columns:
                raise ValueError(f"Metric '{metric}' not found in DataFrame")
            
            iteration_means = df.groupby('iteration')[metric].mean()
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Boxplot
            ax1.boxplot([iteration_means.values], vert=True, widths=0.5,
                    patch_artist=True,
                    boxprops=dict(facecolor='#A8DADC', alpha=0.7),
                    medianprops=dict(color='#E63946', linewidth=2))
            ax1.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
            ax1.set_title('Distribution Across Iterations', fontsize=13, fontweight='bold')
            ax1.set_xticks([1])
            ax1.set_xticklabels(['Iteration Means\n(k-fold averaged)'])
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Histogram with KDE
            ax2.hist(iteration_means.values, bins=15, alpha=0.6, color='#457B9D', 
                    edgecolor='black', density=True)
            
            if iteration_means.values.std() > 1e-8:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(iteration_means.values)
                x_range = np.linspace(iteration_means.min(), iteration_means.max(), 200)
                ax2.plot(x_range, kde(x_range), color='#E63946', linewidth=2, label='KDE')
            else:
                ax2.axvline(iteration_means.values[0], color='#E63946', linewidth=2, label='Constant Data (No KDE)')
            
            ax2.axvline(iteration_means.mean(), color='green', linestyle='--', 
                    linewidth=2, label=f'Mean: {iteration_means.mean():.4f}')
            
            ax2.set_xlabel(metric.replace('_', ' ').title(), fontsize=12)
            ax2.set_ylabel('Density', fontsize=12)
            ax2.set_title('Empirical Distribution', fontsize=13, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
            
            if show:
                plt.show()
            else:
                plt.close()


def main():
    CSV_PATH = "cv_results.csv"
    HISTORY_DIR = "checkpoints"
    OUTPUT_DIR = "output"
    
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    analyzer = CrossValidationStatistics(
        n_iterations=30,
        n_folds=5,
        confidence_level=0.95
    )
    
    df = analyzer.load_summary_csv(CSV_PATH)
    df.to_csv(f"{OUTPUT_DIR}/cv_results_cleaned.csv", index=False)
    
    patch_metrics = [
        'val_bal_acc', 'val_mcc', 'val_f1', 'val_auc'
    ]
    slide_metrics = [
        'val_slide_bal_acc', 'val_slide_mcc', 
        'val_slide_f1', 'val_slide_auc'
    ]
    
    patch_stats = analyzer.compute_hierarchical_statistics(df, patch_metrics)
    slide_stats = analyzer.compute_hierarchical_statistics(df, slide_metrics)
    
    print("PATCH-LEVEL METRICS")
    for metric, stats in patch_stats.items():
        print(f"\n{metric}:")
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Std (between-iteration): {stats['std']:.4f}")
        print(f"  95% CI: [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]")
    
    print("SLIDE-LEVEL METRICS")
    for metric, stats in slide_stats.items():
        print(f"\n{metric}:")
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Std (between-iteration): {stats['std']:.4f}")
        print(f"  95% CI: [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]")
    
    latex_table = analyzer.generate_latex_table(
        patch_stats, slide_stats,
        output_path=f"{OUTPUT_DIR}/results_table.tex"
    )
    
    histories = analyzer.load_training_histories(HISTORY_DIR)
    dynamics = analyzer.analyze_convergence_dynamics(histories)
    
    analyzer.plot_learning_curves(
        dynamics,
        output_path=f"{OUTPUT_DIR}/learning_curves.png",
        show=False
    )

    analyzer.plot_metric_distributions(
        df,
        metric='val_slide_bal_acc',
        output_path=f"{OUTPUT_DIR}/metric_distribution_slide_level.png",
        show=False
    )
    analyzer.plot_metric_distributions(
        df,
        metric='val_bal_acc',
        output_path=f"{OUTPUT_DIR}/metric_distribution_patch_level.png",
        show=False
    )

if __name__ == "__main__":
    main()
