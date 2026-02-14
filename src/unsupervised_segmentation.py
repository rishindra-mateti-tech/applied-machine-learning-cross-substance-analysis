"""
Unsupervised Behavioral Segmentation, Cluster Topology Optimization, and Polysubstance Synthesis
Author: Rishindra Mateti (Wright State University)

Implements:
1. Vector-optimized K-Means from scratch.
2. Vector-optimized Fuzzy C-Means (Bezdek's formulation) from scratch.
3. Multi-start stability sweeps over 20 random restarts computing both Davies-Bouldin and Silhouette metrics.
4. Non-parametric Kruskal-Wallis evaluation connecting latent clusters to polysubstance involvement.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.metrics import silhouette_score
from scipy import stats


class KMeansScratch:
    """K-Means clustering implemented from mathematical first principles."""
    def __init__(self, n_clusters: int = 4, max_iter: int = 300, tol: float = 1e-6, random_state: Optional[int] = 42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: float = 0.0

    def fit(self, X: np.ndarray) -> KMeansScratch:
        rng = np.random.default_rng(self.random_state)
        n_samples = X.shape[0]

        initial_indices = rng.choice(n_samples, size=self.n_clusters, replace=False)
        self.centroids_ = X[initial_indices].copy()

        for _ in range(self.max_iter):
            distances = np.linalg.norm(X[:, np.newaxis, :] - self.centroids_[np.newaxis, :, :], axis=2)
            self.labels_ = np.argmin(distances, axis=1)

            new_centroids = np.zeros_like(self.centroids_)
            for k in range(self.n_clusters):
                cluster_members = X[self.labels_ == k]
                if len(cluster_members) > 0:
                    new_centroids[k] = np.mean(cluster_members, axis=0)
                else:
                    new_centroids[k] = X[rng.choice(n_samples)]

            shift = np.linalg.norm(new_centroids - self.centroids_)
            self.centroids_ = new_centroids
            if shift < self.tol:
                break

        final_distances = np.linalg.norm(X - self.centroids_[self.labels_], axis=1)
        self.inertia_ = float(np.sum(final_distances ** 2))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(X[:, np.newaxis, :] - self.centroids_[np.newaxis, :, :], axis=2)
        return np.argmin(distances, axis=1)


class FuzzyCMeansScratch:
    """Fuzzy C-Means formulated via Bezdek's objective function."""
    def __init__(self, n_clusters: int = 4, m: float = 2.0, max_iter: int = 200, tol: float = 1e-5, random_state: Optional[int] = 42):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids_: Optional[np.ndarray] = None
        self.u_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> FuzzyCMeansScratch:
        rng = np.random.default_rng(self.random_state)
        n_samples = X.shape[0]

        raw_u = rng.uniform(0.01, 1.0, size=(n_samples, self.n_clusters))
        self.u_ = raw_u / np.sum(raw_u, axis=1, keepdims=True)

        for _ in range(self.max_iter):
            u_m = self.u_ ** self.m
            self.centroids_ = (u_m.T @ X) / np.sum(u_m, axis=0)[:, np.newaxis]

            distances = np.maximum(
                np.linalg.norm(X[:, np.newaxis, :] - self.centroids_[np.newaxis, :, :], axis=2),
                1e-12
            )

            power = 2.0 / (self.m - 1.0)
            ratio = (distances[:, :, np.newaxis] / distances[:, np.newaxis, :]) ** power
            new_u = 1.0 / np.sum(ratio, axis=2)

            diff = np.max(np.abs(new_u - self.u_))
            self.u_ = new_u
            if diff < self.tol:
                break

        return self

    def get_hard_labels(self) -> np.ndarray:
        return np.argmax(self.u_, axis=1)


def compute_davies_bouldin_index(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    """Computes Davies-Bouldin cluster separation index."""
    k = len(centroids)
    if k <= 1:
        return 0.0

    scatter = np.zeros(k)
    for i in range(k):
        members = X[labels == i]
        if len(members) > 0:
            scatter[i] = np.mean(np.linalg.norm(members - centroids[i], axis=1))
        else:
            scatter[i] = 0.0

    centroid_dists = np.linalg.norm(centroids[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
    np.fill_diagonal(centroid_dists, np.inf)

    similarity_matrix = (scatter[:, np.newaxis] + scatter[np.newaxis, :]) / centroid_dists
    max_similarity = np.max(similarity_matrix, axis=1)
    return float(np.mean(max_similarity))


def evaluate_cluster_topology_stability(
    X: np.ndarray,
    k_range: List[int] = list(range(2, 9)),
    n_restarts: int = 20,
    base_seed: int = 42
) -> Dict[str, np.ndarray]:
    """
    Evaluates topological stability across cluster count k over 20 random restarts.
    Computes both Davies-Bouldin and Silhouette score distributions.
    """
    db_means, db_stds, db_mins = [], [], []
    sil_means, sil_stds, sil_maxs = [], [], []

    for k in k_range:
        db_scores = []
        sil_scores = []

        for trial in range(n_restarts):
            seed = base_seed + trial * 13 + k * 29
            model = KMeansScratch(n_clusters=k, random_state=seed).fit(X)

            db = compute_davies_bouldin_index(X, model.labels_, model.centroids_)
            db_scores.append(db)

            sil = float(silhouette_score(X, model.labels_))
            sil_scores.append(sil)

        db_means.append(float(np.mean(db_scores)))
        db_stds.append(float(np.std(db_scores)))
        db_mins.append(float(np.min(db_scores)))

        sil_means.append(float(np.mean(sil_scores)))
        sil_stds.append(float(np.std(sil_scores)))
        sil_maxs.append(float(np.max(sil_scores)))

    return {
        'k_values': np.array(k_range),
        'db_mean': np.array(db_means),
        'db_std': np.array(db_stds),
        'db_min': np.array(db_mins),
        'sil_mean': np.array(sil_means),
        'sil_std': np.array(sil_stds),
        'sil_max': np.array(sil_maxs)
    }


def evaluate_cluster_polysubstance_association(
    labels: np.ndarray,
    polysubstance_scores: np.ndarray
) -> Dict[str, object]:
    """
    Evaluates association between discovered cluster memberships and polysubstance involvement.
    Computes cluster-specific mean scores and executes Kruskal-Wallis non-parametric test.
    """
    unique_clusters = np.unique(labels)
    cluster_scores = [polysubstance_scores[labels == c] for c in unique_clusters]

    means = [float(np.mean(s)) for s in cluster_scores]
    stds = [float(np.std(s)) for s in cluster_scores]
    sizes = [len(s) for s in cluster_scores]

    # Non-parametric ANOVA across cluster groups
    stat, p_val = stats.kruskal(*cluster_scores)

    return {
        'cluster_means': means,
        'cluster_stds': stds,
        'cluster_sizes': sizes,
        'kruskal_statistic': float(stat),
        'p_value': float(p_val)
    }
