"""
embeddings.py

Embedding Space Analysis
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

try:
    import umap

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


class EmbeddingAnalyzer:
    """
    Embedding space analyzer with slide-aware diagnostics.
    """

    def __init__(self, output_dir: str, dpi: int = 300, random_seed: int = 42):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.random_seed = random_seed
        self._configure_matplotlib()

    def _configure_matplotlib(self) -> None:
        plt.rcParams.update(
            {
                "font.size": 11,
                "figure.facecolor": "white",
                "savefig.dpi": self.dpi,
                "font.family": "sans-serif",
            }
        )

    def compute_tsne_projection(
        self,
        embeddings: np.ndarray,
        perplexity: int = 30,
        n_iter: int = 1000,
        learning_rate: float = 200.0,
    ) -> np.ndarray:
        n_samples = embeddings.shape[0]
        max_perplexity = (n_samples - 1) // 3
        if perplexity > max_perplexity:
            warnings.warn(f"Reducing perplexity from {perplexity} to {max_perplexity}")
            perplexity = max_perplexity

        print("\nComputing t-SNE projection:")
        print(f" Samples: {n_samples}")
        print(f" Perplexity: {perplexity}")

        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            n_iter=n_iter,
            learning_rate=learning_rate,
            random_state=self.random_seed,
            verbose=1,
        )
        projection = tsne.fit_transform(embeddings)
        print(f" Complete. KL divergence: {tsne.kl_divergence_:.4f}")
        return projection

    def compute_umap_projection(self, embeddings: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1) -> np.ndarray:
        if not UMAP_AVAILABLE:
            raise ImportError("UMAP not installed. Install with: pip install umap-learn")

        n_samples = embeddings.shape[0]
        if n_neighbors >= n_samples:
            n_neighbors = n_samples - 1

        print("\nComputing UMAP projection:")
        print(f" Samples: {n_samples}")
        print(f" Neighbors: {n_neighbors}")

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=self.random_seed,
            verbose=True,
        )
        projection = reducer.fit_transform(embeddings)
        print(" Complete")
        return projection

    # -----------------------------
    # Slide-aware summary metrics
    # -----------------------------
    def compute_slide_aware_metrics(
        self,
        embeddings: np.ndarray,
        projection: np.ndarray,
        labels: np.ndarray,
        predictions: np.ndarray,
        slide_ids: np.ndarray,
        n_permutations: int = 1000,
    ) -> Dict:
        """
        Calculates slide-level metrics with Class-Stratified Permutation Testing.
        
        Refactored for Rigor & Cleanliness:
        1. Statistics: Uses Class-Stratified Permutation (controls for species difficulty).
        2. Geometry: Uses original centroid-based definitions for Homogeneity.
        3. Schema: Returns clean dictionary (legacy 'intra' variance keys removed).
        """
        # 0. Setup Deterministic RNG
        rng = np.random.RandomState(self.random_seed)
        
        # 1. Prepare Data
        errors = (labels != predictions).astype(int)
        unique_slides = np.unique(slide_ids)
        unique_classes = np.unique(labels)
        
        # -------------------------------------------------------------------------
        # PART A: Geometric Metrics (Original Physics)
        # -------------------------------------------------------------------------
        slide_centroids = []
        all_dists = [] 
        obs_slide_rates = []
        
        for slide in unique_slides:
            mask = slide_ids == slide
            if np.sum(mask) == 0: continue
            
            # Geometric (Physical Clustering)
            slide_proj = projection[mask]
            centroid = np.mean(slide_proj, axis=0)
            slide_centroids.append(centroid)
            dists = np.linalg.norm(slide_proj - centroid, axis=1)
            all_dists.extend(dists)
            
            # Error Rate
            rate = np.mean(errors[mask])
            obs_slide_rates.append(rate)
            
        # A1. Intra-slide Variance (Geometric)
        intra_slide_variance = np.var(all_dists) if all_dists else 0.0
        
        # A2. Inter-slide Separation
        inter_dists = []
        if len(slide_centroids) > 1:
            centroids = np.stack(slide_centroids)
            from scipy.spatial.distance import pdist
            inter_dists = pdist(centroids)
            inter_sep_mean = float(np.mean(inter_dists))
        else:
            inter_sep_mean = 0.0
            
        # A3. Homogeneity Score (Geometric)
        homogeneity_score = inter_sep_mean / (intra_slide_variance + 1e-10)

        # -------------------------------------------------------------------------
        # PART B: Statistical Test for Slide Bias (Class-Stratified)
        # -------------------------------------------------------------------------
        obs_error_var = np.var(obs_slide_rates)
        null_error_vars = []
        
        slide_indices_map = {slide: np.where(slide_ids == slide)[0] for slide in unique_slides}
        class_indices_map = {c: np.where(labels == c)[0] for c in unique_classes}
        
        print(f"Running Class-Stratified Permutation Test ({n_permutations} iter)...")
        
        for _ in range(n_permutations):
            # Stratified Shuffle: Shuffle errors ONLY within their class
            perm_errors = errors.copy()
            for c in unique_classes:
                idxs = class_indices_map[c]
                perm_errors[idxs] = rng.permutation(perm_errors[idxs])
            
            # Compute Metric on Null Data
            perm_rates = []
            for slide in unique_slides:
                idxs = slide_indices_map[slide]
                perm_rates.append(np.mean(perm_errors[idxs]))
            null_error_vars.append(np.var(perm_rates))
            
        null_error_vars = np.array(null_error_vars)
        
        # Finite-sample p-value
        n_extreme = np.sum(null_error_vars >= obs_error_var)
        p_value_slide_variance = (n_extreme + 1) / (n_permutations + 1)
        
        # Classification
        if p_value_slide_variance < 0.05:
            classification = "Systematic (statistically significant)"
        else:
            classification = "Stochastic (random noise)"

        # -------------------------------------------------------------------------
        # PART C: Clean Schema Return
        # -------------------------------------------------------------------------
        return {
            "n_slides": int(len(unique_slides)),
            "intra_slide_variance": float(intra_slide_variance),
            "inter_slide_separation_mean": float(inter_sep_mean),
            "slide_error_rate_variance": float(obs_error_var),
            "homogeneity_score": float(homogeneity_score),
            
            # Cleaned 'statistical_classification' (Removed legacy 'intra' keys)
            "statistical_classification": {
                "classification": classification,
                "confidence": float(1.0 - p_value_slide_variance),
                "n_permutations": int(n_permutations),
                
                # Only keeping active statistics
                "obs_slide_error_var": float(obs_error_var),
                "null_slide_error_var_mean": float(np.mean(null_error_vars)),
                "null_slide_error_var_std": float(np.std(null_error_vars)),
                "p_value_slide_variance": float(p_value_slide_variance),
                
                "interpretation": (
                    f"Observed Error Var={obs_error_var:.4f} vs Null (Stratified)={np.mean(null_error_vars):.4f}. "
                    f"p={p_value_slide_variance:.4f}."
                )
            }
        }

    def compute_trustworthiness(self, embeddings: np.ndarray, projection: np.ndarray, n_neighbors: int = 5) -> float:
        from sklearn.manifold import trustworthiness as sklearn_trustworthiness

        n_samples = embeddings.shape[0]
        if n_neighbors >= n_samples:
            n_neighbors = min(5, n_samples - 1)
        return float(sklearn_trustworthiness(embeddings, projection, n_neighbors=n_neighbors))

    def compute_silhouette_score(self, projection: np.ndarray, labels: np.ndarray) -> float:
        return float(silhouette_score(projection, labels, metric="euclidean"))

    def plot_embedding_projection(
        self,
        projection: np.ndarray,
        labels: np.ndarray,
        predictions: np.ndarray,
        slide_ids: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None,
        method_name: str = "t-SNE",
        save_name: str = "embedding.png",
        show: bool = False,
        plot_errors_only: bool = False,
    ) -> str:
        projection = np.asarray(projection, dtype=np.float32)
        labels = np.asarray(labels).astype(int)
        predictions = np.asarray(predictions).astype(int)

        if class_names is None:
            uniq = np.unique(labels)
            class_names = [f"Class {i}" for i in range(int(uniq.max()) + 1)]

        correct = labels == predictions
        tp = (labels == 1) & (predictions == 1)
        tn = (labels == 0) & (predictions == 0)
        fp = (labels == 0) & (predictions == 1)
        fn = (labels == 1) & (predictions == 0)

        if plot_errors_only:
            fig, ax = plt.subplots(figsize=(10, 8))
            error_colors = {"TP": "#2ecc71", "TN": "#3498db", "FP": "#e74c3c", "FN": "#f39c12"}

            for mask, lab, col in [(tn, "TN", error_colors["TN"]), (tp, "TP", error_colors["TP"]), (fp, "FP", error_colors["FP"]), (fn, "FN", error_colors["FN"])]:
                if np.sum(mask) > 0:
                    ax.scatter(
                        projection[mask, 0],
                        projection[mask, 1],
                        c=col,
                        label=f"{lab} (n={int(np.sum(mask))})",
                        alpha=0.6,
                        s=50,
                        edgecolors="black",
                        linewidth=0.5,
                    )
            ax.set_xlabel(f"{method_name} Dimension 1", fontweight="bold")
            ax.set_ylabel(f"{method_name} Dimension 2", fontweight="bold")
            ax.set_title(f"{method_name} - Error Stratification", fontweight="bold", pad=15)
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)
        else:
            fig, axes = plt.subplots(2, 2, figsize=(16, 14))

            ax1 = axes[0, 0]
            for class_idx in np.unique(labels):
                m = labels == class_idx
                ax1.scatter(
                    projection[m, 0],
                    projection[m, 1],
                    label=f"{class_names[int(class_idx)]} (n={int(np.sum(m))})",
                    alpha=0.6,
                    s=50,
                )
            ax1.set_title("Ground Truth", fontweight="bold")
            ax1.set_xlabel(f"{method_name} Dim 1")
            ax1.set_ylabel(f"{method_name} Dim 2")
            ax1.legend(loc="best")
            ax1.grid(True, alpha=0.3)

            ax2 = axes[0, 1]
            for class_idx in np.unique(predictions):
                m = predictions == class_idx
                ax2.scatter(
                    projection[m, 0],
                    projection[m, 1],
                    label=f"{class_names[int(class_idx)]} (n={int(np.sum(m))})",
                    alpha=0.6,
                    s=50,
                )
            ax2.set_title("Predictions", fontweight="bold")
            ax2.set_xlabel(f"{method_name} Dim 1")
            ax2.set_ylabel(f"{method_name} Dim 2")
            ax2.legend(loc="best")
            ax2.grid(True, alpha=0.3)

            ax3 = axes[1, 0]
            ax3.scatter(projection[correct, 0], projection[correct, 1], c="green", label=f"Correct (n={int(np.sum(correct))})", alpha=0.6, s=50)
            ax3.scatter(projection[~correct, 0], projection[~correct, 1], c="red", label=f"Incorrect (n={int(np.sum(~correct))})", alpha=0.6, s=50)
            ax3.set_title("Performance", fontweight="bold")
            ax3.set_xlabel(f"{method_name} Dim 1")
            ax3.set_ylabel(f"{method_name} Dim 2")
            ax3.legend(loc="best")
            ax3.grid(True, alpha=0.3)

            ax4 = axes[1, 1]
            for mask, lab, col in [(tn, "TN", "#3498db"), (tp, "TP", "#2ecc71"), (fp, "FP", "#e74c3c"), (fn, "FN", "#f39c12")]:
                if np.sum(mask) > 0:
                    ax4.scatter(projection[mask, 0], projection[mask, 1], c=col, label=f"{lab} (n={int(np.sum(mask))})", alpha=0.6, s=50)
            ax4.set_title("Error Types", fontweight="bold")
            ax4.set_xlabel(f"{method_name} Dim 1")
            ax4.set_ylabel(f"{method_name} Dim 2")
            ax4.legend(loc="best")
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()

        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close()

        print(f"Saved: {save_path}")
        return str(save_path)

    def analyze_embedding_quality(
        self,
        embeddings: np.ndarray,
        projection: np.ndarray,
        labels: np.ndarray,
        predictions: np.ndarray,
        slide_ids: np.ndarray,
        method_name: str = "t-SNE",
    ) -> Dict:
        print(f"\nAnalyzing {method_name} embedding quality...")

        trust_k5 = self.compute_trustworthiness(embeddings, projection, n_neighbors=5)
        silhouette_gt = self.compute_silhouette_score(projection, labels)
        silhouette_pred = self.compute_silhouette_score(projection, predictions)
        accuracy = float(np.mean(np.asarray(labels) == np.asarray(predictions)))

        slide_metrics = self.compute_slide_aware_metrics(
            embeddings=embeddings,
            projection=projection,
            labels=labels,
            predictions=predictions,
            slide_ids=slide_ids,
        )

        metrics = {
            "method": method_name,
            "trustworthiness_k5": float(trust_k5),
            "silhouette_ground_truth": float(silhouette_gt),
            "silhouette_predictions": float(silhouette_pred),
            "classification_accuracy": float(accuracy),
            "slide_aware_metrics": slide_metrics,
        }

        print("\nStandard Metrics:")
        print(f" Trustworthiness (k=5): {trust_k5:.4f}")
        print(f" Silhouette (GT): {silhouette_gt:.4f}")
        print(f" Silhouette (Pred): {silhouette_pred:.4f}")
        print(f" Accuracy: {accuracy:.4f}")
        print(" Statistical classification complete (see above)")

        return metrics
