"""
Plotting Utilities Module

Author: Rodrigo Sá
"""

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc


class TrainingPlotter:
    """
    Handles all plotting operations for the training pipeline.
    """

    def __init__(self, output_dir: str, dpi: int = 300, style: str = "default"):
        self.output_dir = output_dir
        self.dpi = dpi
        self.style = style

        os.makedirs(output_dir, exist_ok=True)
        if style != "default":
            plt.style.use(style)

        plt.rcParams["figure.facecolor"] = "white"
        plt.rcParams["axes.facecolor"] = "white"

    def plot_iteration_results(
        self,
        cv_results_df: pd.DataFrame,
        experiment_name: str = "",
        save_name: str = "cv_iteration_results.png",
        show: bool = True,
        prefer_slide_level: bool = True,
        metric_cols: Optional[Sequence[str]] = None,
    ) -> str:
        """
        Plot distribution of mean CV metrics across all iterations.
        """
        if cv_results_df is None or cv_results_df.empty:
            print("No CV results to plot.")
            return ""

        df = cv_results_df.copy()

        slide_default = [
            "val_slide_bal_acc_raw",
            "val_slide_mcc_best",
            "val_slide_f1_best",
            "val_slide_auc_best",
        ]
        patch_legacy = [
            "val_bal_acc",
            "val_mcc",
            "val_f1",
            "val_auc",
        ]

        if metric_cols is not None:
            metrics_to_plot = list(metric_cols)
        else:
            if prefer_slide_level and all(col in df.columns for col in slide_default):
                metrics_to_plot = slide_default
            elif all(col in df.columns for col in patch_legacy):
                metrics_to_plot = patch_legacy
            else:
                existing_slide = [c for c in slide_default if c in df.columns]
                existing_patch = [c for c in patch_legacy if c in df.columns]
                metrics_to_plot = existing_slide or existing_patch

        if "iteration" not in df.columns:
            raise ValueError("cv_results_df must contain an 'iteration' column.")

        if not metrics_to_plot:
            raise ValueError(
                "No known metric columns found to plot. Expected either slide-level "
                f"{slide_default} or legacy patch-level {patch_legacy}."
            )

        # Calculate mean metric per iteration
        iteration_means = df.groupby("iteration")[metrics_to_plot].mean().reset_index()

        # Melt for seaborn box/strip plots
        melted = iteration_means.melt(
            id_vars=["iteration"],
            value_vars=metrics_to_plot,
            var_name="Metric",
            value_name="Mean Score",
        )

        # Human-readable labels
        label_map = {
            "val_slide_acc_best": "Slide Acc (best epoch)",
            "val_slide_bal_acc_raw": "Slide Balanced Acc (best epoch)",
            "val_slide_mcc_best": "Slide MCC (best epoch)",
            "val_slide_f1_best": "Slide F1 (weighted, best epoch)",
            "val_slide_auc_best": "Slide AUC (best epoch)",
            "val_bal_acc": "Patch Balanced Acc",
            "val_mcc": "Patch MCC",
            "val_f1": "Patch F1 (weighted)",
            "val_auc": "Patch AUC",
        }
        melted["Metric"] = melted["Metric"].map(lambda x: label_map.get(x, x))

        fig, ax = plt.subplots(figsize=(12, 7))
        sns.boxplot(x="Metric", y="Mean Score", data=melted, palette="viridis", ax=ax)
        sns.stripplot(x="Metric", y="Mean Score", data=melted, color=".25", alpha=0.5, ax=ax)

        n_iterations = iteration_means["iteration"].nunique()
        title = f"CV Performance Distribution ({n_iterations} Iterations)"
        if experiment_name:
            title = f"{title} - {experiment_name}"
        if prefer_slide_level and all(col in df.columns for col in slide_default):
            title = f"{title} (slide-level CV)"
        ax.set_title(title, fontsize=16)

        ax.set_xlabel("Metric", fontsize=13)
        ax.set_ylabel("Mean score per iteration", fontsize=13)

        ymin = float(np.nanmin(melted["Mean Score"].values))
        ymax = float(np.nanmax(melted["Mean Score"].values))
        pad = 0.05 * (ymax - ymin + 1e-9)
        ax.set_ylim(ymin - pad, ymax + pad)

        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        print(f"CV iteration distribution plot saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return save_path

    def plot_training_curves(
        self,
        history: Dict[str, List[float]],
        title: str = "Training Curves",
        save_name: str = "training_curves.png",
        show: bool = True,
    ) -> str:
        """
        Plot training loss and accuracy curves for the final model.
        """
        if "train_loss" not in history or "train_acc" not in history:
            raise ValueError("history must contain 'train_loss' and 'train_acc' keys.")

        epochs = range(1, len(history["train_loss"]) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Loss
        axes[0].plot(epochs, history["train_loss"], "b-", linewidth=2, label="Training Loss")
        axes[0].set_xlabel("Epoch", fontsize=12)
        axes[0].set_ylabel("Loss", fontsize=12)
        axes[0].set_title("Training Loss", fontsize=14)
        axes[0].grid(alpha=0.3)
        axes[0].legend()

        # Accuracy
        axes[1].plot(epochs, history["train_acc"], "g-", linewidth=2, label="Train Accuracy (Hard)")

        if "train_acc_soft" in history and history["train_acc"] != history["train_acc_soft"]:
            axes[1].plot(
                epochs,
                history["train_acc_soft"],
                "g--",
                linewidth=2,
                label="Train Accuracy (Soft/MixUp)",
                alpha=0.7,
            )

        axes[1].set_xlabel("Epoch", fontsize=12)
        axes[1].set_ylabel("Accuracy", fontsize=12)
        axes[1].set_title("Training Accuracy", fontsize=14)
        axes[1].grid(alpha=0.3)
        axes[1].legend()

        fig.suptitle(title, fontsize=16, y=1.02)
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        print(f"Training curves saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return save_path

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str],
        title: str = "Confusion Matrix",
        save_name: str = "confusion_matrix.png",
        normalize: bool = False,
        show: bool = True,
    ) -> str:
        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-12)
            fmt = ".2f"
        else:
            fmt = "d"

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={"label": "Proportion" if normalize else "Count"},
            ax=ax,
        )

        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        print(f"Confusion matrix saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return save_path

    def plot_roc_auc_curves(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        title: str = "ROC Curve",
        save_name: str = "roc_auc_curve.png",
        show: bool = True,
    ) -> str:
        """
        ROC curve for binary classification.
        Expects y_proba shape [n_samples, 2].
        """
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], "k--", lw=2)

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=14)
        plt.ylabel("True Positive Rate", fontsize=14)
        plt.title(title, fontsize=16)
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        print(f"ROC/AUC curve plot saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return save_path
