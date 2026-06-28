from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats


@dataclass
class StrataDefinition:
    """Definition of a sampling stratum."""
    stratum_id: str
    mask: np.ndarray  # Boolean mask identifying samples in this stratum
    n_samples: int
    target_samples: int  # Number of samples to draw from this stratum
    
    
@dataclass  
class StratifiedSample:
    """A stratified sample with provenance."""
    sample_indices: np.ndarray  # Indices into original dataset
    stratum_ids: List[str]  # Stratum ID for each sample
    weights: np.ndarray  # Sampling weights (for weighted statistics)
    

@dataclass
class SpatialMetricsSummary:
    """Summary statistics for spatial attention metrics."""
    mean: float
    std: float
    ci_lower: float  # 95% CI lower bound
    ci_upper: float  # 95% CI upper bound
    median: float
    q25: float  # 25th percentile
    q75: float  # 75th percentile
    n_samples: int
    
    def to_dict(self) -> Dict:
        return {
            "mean": float(self.mean),
            "std": float(self.std),
            "ci_95_lower": float(self.ci_lower),
            "ci_95_upper": float(self.ci_upper),
            "median": float(self.median),
            "q25": float(self.q25),
            "q75": float(self.q75),
            "n_samples": int(self.n_samples),
        }


@dataclass
class MMDCriticSelection:
    """Selected prototypes/criticisms with auditable per-sample metadata."""
    sample_indices: np.ndarray
    stratum_ids: List[str]
    roles: List[str]
    metadata: List[Dict]


class StratifiedMMDCriticSelector:
    """
    Stratified MMD-critic selector for qualitative XAI review panels.

    Scientific intent
    -----------------
    For each predefined outcome stratum (e.g. TP/TN/FP/FN), prototypes are
    selected by greedy minimisation of the empirical Maximum Mean Discrepancy
    (MMD) between the full stratum feature distribution and the selected set.
    Criticisms are then selected as high-witness-function samples: points that
    remain poorly represented by the prototype set. This follows the
    MMD-critic principle of Kim, Khanna & Koyejo (NeurIPS 2016): prototypes
    summarise the distribution, while criticisms expose systematic gaps.

    The class intentionally accepts a generic feature matrix so the caller can
    combine explanation morphology (Gini/entropy), model behaviour
    (confidence, faithfulness AUC), and representation geometry (PCA/embedding
    coordinates) without hard-coding domain assumptions in the sampler.
    """

    def __init__(self, random_seed: int = 42):
        self.random_seed = int(random_seed)
        self.rng = np.random.RandomState(self.random_seed)

    @staticmethod
    def _impute_and_standardise(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"features must be 2D, got shape {X.shape}")
        col_mean = np.nanmean(X, axis=0)
        col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
        inds = np.where(~np.isfinite(X))
        if inds[0].size:
            X = X.copy()
            X[inds] = np.take(col_mean, inds[1])
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        return ((X - mu) / sd).astype(np.float64)

    @staticmethod
    def _rbf_kernel(X: np.ndarray, gamma: Optional[float] = None) -> Tuple[np.ndarray, float]:
        X = np.asarray(X, dtype=np.float64)
        sq_norm = np.sum(X * X, axis=1, keepdims=True)
        d2 = np.maximum(sq_norm + sq_norm.T - 2.0 * (X @ X.T), 0.0)
        if gamma is None:
            tri = d2[np.triu_indices_from(d2, k=1)]
            positive = tri[tri > 1e-12]
            if positive.size == 0:
                gamma = 1.0
            else:
                median_d2 = float(np.median(positive))
                gamma = 1.0 / max(median_d2, 1e-12)
        K = np.exp(-float(gamma) * d2)
        return K.astype(np.float64), float(gamma)

    @staticmethod
    def _mmd2_for_subset(K: np.ndarray, subset: Sequence[int]) -> float:
        subset = list(map(int, subset))
        if len(subset) == 0:
            return float("inf")
        # Biased empirical MMD^2. The full-full term is constant for a given
        # stratum, but retained for numerical interpretability in metadata.
        k_xx = float(np.mean(K))
        k_xs = float(np.mean(K[:, subset]))
        k_ss = float(np.mean(K[np.ix_(subset, subset)]))
        return float(k_xx - 2.0 * k_xs + k_ss)

    def _choose_with_slide_preference(
        self,
        candidate_order: Sequence[int],
        selected_global: set,
        used_slides: set,
        slide_ids_group: Optional[np.ndarray],
        slide_aware: bool,
    ) -> Optional[int]:
        ordered = [int(i) for i in candidate_order if int(i) not in selected_global]
        if not ordered:
            return None
        if not slide_aware or slide_ids_group is None:
            return ordered[0]
        for local_i in ordered:
            sid = str(slide_ids_group[local_i])
            if sid not in used_slides:
                return local_i
        return ordered[0]

    def select_within_strata(
        self,
        strata: List[StrataDefinition],
        features: np.ndarray,
        quotas: Dict[str, int],
        slide_ids: Optional[np.ndarray] = None,
        prototype_fraction: float = 0.6,
        slide_aware: bool = True,
        rbf_gamma: Optional[float] = None,
    ) -> MMDCriticSelection:
        """
        Select prototypes and criticisms independently within each stratum.

        Parameters
        ----------
        strata : list of StrataDefinition
            Outcome strata. Masks index the rows of ``features``.
        features : (N, D) array
            Feature representation used for MMD: e.g. rollout Gini/entropy,
            confidence, faithfulness AUC, and embedding coordinates.
        quotas : dict
            Requested number of review samples per stratum.
        slide_ids : optional (N,) array
            Slide identifiers used as a soft diversity constraint.
        prototype_fraction : float
            Fraction of each stratum quota allocated to prototypes. Remaining
            slots are criticisms. At least one prototype is selected whenever a
            non-empty stratum has a positive quota.
        slide_aware : bool
            Prefer candidates from unused slides when possible. The constraint
            is relaxed if it would prevent filling the quota.
        rbf_gamma : optional float
            RBF kernel width parameter. If None, the median-distance heuristic
            is used separately within each stratum.
        """
        X_all = self._impute_and_standardise(features)
        slide_ids_arr = None if slide_ids is None else np.asarray(slide_ids, dtype=object)

        all_indices: List[int] = []
        all_strata: List[str] = []
        all_roles: List[str] = []
        all_meta: List[Dict] = []

        for s in strata:
            group = str(s.stratum_id)
            global_idx = np.where(np.asarray(s.mask, dtype=bool))[0]
            n_available = int(global_idx.size)
            quota = min(int(quotas.get(group, s.target_samples)), n_available)
            if n_available == 0 or quota <= 0:
                continue

            if n_available <= quota:
                for rank, gi in enumerate(global_idx):
                    all_indices.append(int(gi))
                    all_strata.append(group)
                    all_roles.append("complete_small_stratum")
                    all_meta.append({
                        "row_index": int(gi),
                        "group": group,
                        "selection_role": "complete_small_stratum",
                        "selection_reason": "include_all_available_mmd_critic_stratum",
                        "selection_metric": "mmd_critic_feature_space",
                        "selection_target": "all_available",
                        "mmd2_after_selection": 0.0,
                        "witness_score": 0.0,
                        "kernel_gamma": None,
                        "slide_diversity_applied": bool(slide_aware),
                        "within_group_rank": int(rank + 1),
                    })
                continue

            Xg = X_all[global_idx]
            K, gamma = self._rbf_kernel(Xg, gamma=rbf_gamma)
            slide_g = None if slide_ids_arr is None else slide_ids_arr[global_idx]
            n_prototypes = max(1, int(np.ceil(float(prototype_fraction) * quota)))
            n_prototypes = min(n_prototypes, quota)
            n_criticisms = max(0, quota - n_prototypes)

            selected_local: List[int] = []
            used_slides: set = set()

            # Greedy exact MMD minimisation for prototypes.
            for proto_rank in range(n_prototypes):
                candidates = [i for i in range(n_available) if i not in selected_local]
                scored = []
                for ci in candidates:
                    mmd2 = self._mmd2_for_subset(K, selected_local + [ci])
                    scored.append((mmd2, ci))
                scored.sort(key=lambda t: (t[0], t[1]))
                chosen = self._choose_with_slide_preference(
                    [ci for _, ci in scored],
                    selected_global=set(selected_local),
                    used_slides=used_slides,
                    slide_ids_group=slide_g,
                    slide_aware=slide_aware,
                )
                if chosen is None:
                    break
                selected_local.append(chosen)
                if slide_g is not None:
                    used_slides.add(str(slide_g[chosen]))
                all_indices.append(int(global_idx[chosen]))
                all_strata.append(group)
                all_roles.append("prototype")
                all_meta.append({
                    "row_index": int(global_idx[chosen]),
                    "group": group,
                    "selection_role": "prototype",
                    "selection_reason": "mmd_critic_prototype_minimise_mmd",
                    "selection_metric": "rbf_kernel_mmd",
                    "selection_target": "minimise_empirical_mmd2",
                    "mmd2_after_selection": self._mmd2_for_subset(K, selected_local),
                    "witness_score": float("nan"),
                    "kernel_gamma": float(gamma),
                    "slide_diversity_applied": bool(slide_aware),
                    "within_group_rank": int(proto_rank + 1),
                })

            # Criticisms: largest positive witness values under prototype set.
            # High positive witness means the full stratum density exceeds the
            # prototype-set density locally, i.e. the prototypes underrepresent
            # this sample. This is the MMD-critic criticism criterion.
            if n_criticisms > 0 and selected_local:
                mean_full = K.mean(axis=1)
                mean_proto = K[:, selected_local].mean(axis=1)
                witness = mean_full - mean_proto
                order = np.argsort(-witness, kind="mergesort")
                for crit_rank in range(n_criticisms):
                    chosen = self._choose_with_slide_preference(
                        order,
                        selected_global=set(selected_local),
                        used_slides=used_slides,
                        slide_ids_group=slide_g,
                        slide_aware=slide_aware,
                    )
                    if chosen is None:
                        break
                    selected_local.append(chosen)
                    if slide_g is not None:
                        used_slides.add(str(slide_g[chosen]))
                    all_indices.append(int(global_idx[chosen]))
                    all_strata.append(group)
                    all_roles.append("criticism")
                    all_meta.append({
                        "row_index": int(global_idx[chosen]),
                        "group": group,
                        "selection_role": "criticism",
                        "selection_reason": "mmd_critic_high_witness_underrepresented",
                        "selection_metric": "mmd_witness_function",
                        "selection_target": "maximise_positive_witness",
                        "mmd2_after_selection": self._mmd2_for_subset(K, selected_local),
                        "witness_score": float(witness[chosen]),
                        "kernel_gamma": float(gamma),
                        "slide_diversity_applied": bool(slide_aware),
                        "within_group_rank": int(crit_rank + 1),
                    })

        return MMDCriticSelection(
            sample_indices=np.asarray(all_indices, dtype=np.int64),
            stratum_ids=all_strata,
            roles=all_roles,
            metadata=all_meta,
        )

class SpatialMetricsAggregator:
    """
    Compute weighted statistics for stratified samples.
    """
    
    @staticmethod
    def compute_summary(
        values: np.ndarray,
        weights: Optional[np.ndarray] = None,
        confidence_level: float = 0.95,
    ) -> SpatialMetricsSummary:
        """
        Compute summary statistics with confidence intervals.
        
        Args:
            values: Metric values (N,)
            weights: Optional sampling weights (N,)
            confidence_level: CI confidence level (default: 0.95)
            
        Returns:
            Summary statistics
        """
        values = np.asarray(values, dtype=np.float32)
        n = len(values)
        
        if n == 0:
            return SpatialMetricsSummary(
                mean=np.nan, std=np.nan, ci_lower=np.nan, ci_upper=np.nan,
                median=np.nan, q25=np.nan, q75=np.nan, n_samples=0,
            )
        
        if weights is None:
            weights = np.ones(n, dtype=np.float32)
        else:
            weights = np.asarray(weights, dtype=np.float32)
        
        # Weighted mean and variance
        mean = np.average(values, weights=weights)
        variance = np.average((values - mean)**2, weights=weights)
        std = np.sqrt(variance)
        
        # Confidence interval (t-distribution for small samples)
        if n >= 2:
            se = std / np.sqrt(n)
            alpha = 1 - confidence_level
            t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
            ci_lower = mean - t_crit * se
            ci_upper = mean + t_crit * se
        else:
            ci_lower = ci_upper = mean
        
        # Percentiles (unweighted - more robust for small samples)
        median = float(np.median(values))
        q25 = float(np.percentile(values, 25))
        q75 = float(np.percentile(values, 75))
        
        return SpatialMetricsSummary(
            mean=float(mean),
            std=float(std),
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            median=median,
            q25=q25,
            q75=q75,
            n_samples=n,
        )
    
    @staticmethod
    def compare_strata(
        values_a: np.ndarray,
        values_b: np.ndarray,
        weights_a: Optional[np.ndarray] = None,
        weights_b: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Compare two strata using Welch's t-test.
        
        Returns:
            Dictionary with test statistics and p-value
        """
        values_a = np.asarray(values_a, dtype=np.float32)
        values_b = np.asarray(values_b, dtype=np.float32)
        
        if len(values_a) < 2 or len(values_b) < 2:
            return {
                "test": "welch_t_test",
                "statistic": np.nan,
                "p_value": np.nan,
                "significant": False,
                "note": "Insufficient samples for test",
            }
        
        # Use weighted means if weights provided
        if weights_a is not None:
            mean_a = np.average(values_a, weights=weights_a)
            var_a = np.average((values_a - mean_a)**2, weights=weights_a)
        else:
            mean_a = np.mean(values_a)
            var_a = np.var(values_a, ddof=1)
        
        if weights_b is not None:
            mean_b = np.average(values_b, weights=weights_b)
            var_b = np.average((values_b - mean_b)**2, weights=weights_b)
        else:
            mean_b = np.mean(values_b)
            var_b = np.var(values_b, ddof=1)
        
        # Welch's t-test
        n_a = len(values_a)
        n_b = len(values_b)
        
        se_diff = np.sqrt(var_a/n_a + var_b/n_b)
        if se_diff == 0:
            t_stat = np.nan
            p_val = np.nan
        else:
            t_stat = (mean_a - mean_b) / se_diff
            # Welch-Satterthwaite degrees of freedom
            df = (var_a/n_a + var_b/n_b)**2 / (
                (var_a/n_a)**2/(n_a-1) + (var_b/n_b)**2/(n_b-1)
            )
            p_val = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=df))
        
        return {
            "test": "welch_t_test",
            "statistic": float(t_stat),
            "p_value": float(p_val),
            "significant": bool(p_val < 0.05) if not np.isnan(p_val) else False,
            "df": float(df) if not np.isnan(p_val) else np.nan,
        }

class AttentionDistributionalMetrics:
    """
    Compute distributional metrics for attention rollout vectors.
    
    References:
    - Gini (1912): Variability and Mutability
    - Shannon (1948): A Mathematical Theory of Communication
    - Lorenz (1905): Methods of Measuring Concentration of Wealth
    """
    
    @staticmethod
    def compute_gini_coefficient(attention_vector: np.ndarray) -> float:
        """
        Compute Gini Coefficient for attention distribution.
        
        High Gini (→1.0) = Focal attention (model looks at specific patches)
        Low Gini (→0.0) = Diffuse attention (model looks everywhere equally)
        
        Args:
            attention_vector: (P,) attention weights (must sum to ≈1.0)
            
        Returns:
            Gini coefficient in [0, 1]
            
        Mathematical Definition:
            G = (Σᵢ Σⱼ |aᵢ - aⱼ|) / (2n Σᵢ aᵢ)
            
        Interpretation:
            - G ≈ 0: Uniform distribution (texture bias)
            - G ≈ 1: Peaked distribution (focal attention on cell/object)
        """
        attention_vector = np.asarray(attention_vector, dtype=np.float64).flatten()
        
        if len(attention_vector) == 0:
            return np.nan
        
        # Remove zeros (padding/masked patches)
        attention_vector = attention_vector[attention_vector > 0]
        
        if len(attention_vector) == 0:
            return np.nan
        
        # Sort ascending
        sorted_attention = np.sort(attention_vector)
        n = len(sorted_attention)
        
        # Compute Gini using Lorenz curve area
        # G = 1 - 2*B where B is area under Lorenz curve
        cumsum = np.cumsum(sorted_attention)
        
        # Normalize cumulative sum
        total = cumsum[-1]
        if total == 0:
            return np.nan
            
        # Area under Lorenz curve (using trapezoidal rule)
        # Lorenz curve: (i/n, cumsum[i]/total)
        heights = cumsum / total
        # Area = sum of trapezoids
        lorenz_area = np.sum((heights[:-1] + heights[1:]) / 2.0) / n
        
        gini = 1.0 - 2.0 * lorenz_area
        
        return float(np.clip(gini, 0.0, 1.0))
    
    @staticmethod
    def compute_spatial_entropy(attention_vector: np.ndarray) -> float:
        """
        Compute Shannon Entropy for attention distribution.
        
        High Entropy = Diffuse/uniform attention (texture bias indicator)
        Low Entropy = Concentrated attention (focal attention)
        
        Args:
            attention_vector: (P,) attention weights (normalized to sum ≈1.0)
            
        Returns:
            Entropy in nats (natural logarithm base)
            
        Mathematical Definition:
            H = -Σᵢ pᵢ log(pᵢ)
            
        Interpretation:
            - H ≈ log(P): Maximum entropy (uniform distribution)
            - H ≈ 0: Minimum entropy (delta function)
        """
        attention_vector = np.asarray(attention_vector, dtype=np.float64).flatten()
        
        if len(attention_vector) == 0:
            return np.nan
        
        # Ensure proper probability distribution
        attention_vector = attention_vector + 1e-10  # Numerical stability
        attention_vector = attention_vector / attention_vector.sum()
        
        # Remove near-zero probabilities (contribute nothing to entropy)
        attention_vector = attention_vector[attention_vector > 1e-10]
        
        if len(attention_vector) == 0:
            return np.nan
        
        # Shannon entropy (natural logarithm)
        entropy = -np.sum(attention_vector * np.log(attention_vector))
        
        return float(entropy)
    
    @staticmethod
    def compute_rollout_metrics_batch(
        rollout_vectors: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Compute Gini and Entropy for batch of rollout vectors.
        
        Args:
            rollout_vectors: (N, P) attention rollout for N samples
            
        Returns:
            Dictionary with:
                - gini_coefficients: (N,) array
                - spatial_entropies: (N,) array
        """
        n_samples = rollout_vectors.shape[0]
        
        gini_values = np.zeros(n_samples, dtype=np.float32)
        entropy_values = np.zeros(n_samples, dtype=np.float32)
        
        for i in range(n_samples):
            gini_values[i] = AttentionDistributionalMetrics.compute_gini_coefficient(
                rollout_vectors[i]
            )
            entropy_values[i] = AttentionDistributionalMetrics.compute_spatial_entropy(
                rollout_vectors[i]
            )
        
        return {
            "gini_coefficients": gini_values,
            "spatial_entropies": entropy_values,
        }


def compute_effect_size_cohens_d(
    group_a: np.ndarray,
    group_b: np.ndarray,
) -> float:
    """
    Compute Cohen's d effect size for two groups.
    
    |d| < 0.2: Negligible
    |d| ≈ 0.5: Medium
    |d| ≈ 0.8: Large
    |d| > 1.2: Very large
    
    Args:
        group_a: First group values
        group_b: Second group values
        
    Returns:
        Cohen's d
    """
    group_a = np.asarray(group_a, dtype=np.float32)
    group_b = np.asarray(group_b, dtype=np.float32)
    
    if len(group_a) < 2 or len(group_b) < 2:
        return np.nan
    
    mean_a = np.mean(group_a)
    mean_b = np.mean(group_b)
    
    var_a = np.var(group_a, ddof=1)
    var_b = np.var(group_b, ddof=1)
    
    n_a = len(group_a)
    n_b = len(group_b)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    
    if pooled_std == 0:
        return np.nan
    
    cohens_d = (mean_a - mean_b) / pooled_std
    
    return float(cohens_d)

def compare_groups_patch_level(values_a: np.ndarray, values_b: np.ndarray, confidencelevel: float = 0.95) -> Dict:
    """
    Patch-level comparison: mean difference, CI for diff, Welch t-test p-value, and Cohen's d.
    Returns descriptive inference (assumes independent samples).
    """
    values_a = np.asarray(values_a, dtype=np.float32)
    values_b = np.asarray(values_b, dtype=np.float32)

    if len(values_a) < 2 or len(values_b) < 2:
        return {
            "test": "welch_ttest",
            "note": "Insufficient samples for test",
            "n_a": int(len(values_a)),
            "n_b": int(len(values_b)),
            "mean_diff": np.nan,
            "ci_95_lower": np.nan,
            "ci_95_upper": np.nan,
            "p_value": np.nan,
            "cohens_d": np.nan,
        }

    mean_a = float(np.mean(values_a))
    mean_b = float(np.mean(values_b))
    var_a = float(np.var(values_a, ddof=1))
    var_b = float(np.var(values_b, ddof=1))
    n_a = len(values_a)
    n_b = len(values_b)

    mean_diff = mean_a - mean_b
    se_diff = np.sqrt(var_a / n_a + var_b / n_b)

    # Welch-Satterthwaite df
    df_num = (var_a / n_a + var_b / n_b) ** 2
    df_den = (var_a**2) / (n_a**2 * (n_a - 1)) + (var_b**2) / (n_b**2 * (n_b - 1))
    df = df_num / df_den if df_den > 0 else np.nan

    alpha = 1.0 - confidencelevel
    tcrit = stats.t.ppf(1.0 - alpha / 2.0, df=df) if not np.isnan(df) else np.nan
    ci_lower = mean_diff - tcrit * se_diff if not np.isnan(tcrit) else np.nan
    ci_upper = mean_diff + tcrit * se_diff if not np.isnan(tcrit) else np.nan

    # Welch p-value via existing aggregator utility
    ttest = SpatialMetricsAggregator.compare_strata(values_a, values_b)

    d = compute_effect_size_cohens_d(values_a, values_b)

    return {
        "test": "welch_ttest",
        "n_a": int(n_a),
        "n_b": int(n_b),
        "mean_a": mean_a,
        "mean_b": mean_b,
        "mean_diff": float(mean_diff),
        "ci_95_lower": float(ci_lower),
        "ci_95_upper": float(ci_upper),
        "df": float(ttest.get("df", np.nan)),
        "statistic": float(ttest.get("statistic", np.nan)),
        "p_value": float(ttest.get("p_value", np.nan)),
        "cohens_d": float(d),
    }
