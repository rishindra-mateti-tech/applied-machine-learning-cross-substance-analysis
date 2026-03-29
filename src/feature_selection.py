"""
Leakage-Free Feature Selection and Parsimonious Model Discovery
Author: Rishindra Mateti (Wright State University)

Implements greedy forward stepwise feature selection using nested 5-fold cross-validation
strictly within the training cohort, with fold-specific inner scaling to guarantee zero data leakage.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import numpy as np
from src.dataset_pipeline import LeakageFreeStandardScaler
from src.supervised_classification import RegularizedLogisticRegressionScratch


def kfold_cross_validate_logistic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_splits: int = 5,
    l2_lambda: float = 0.1,
    random_seed: int = 42
) -> float:
    """
    Computes mean validation accuracy across n_splits folds strictly inside training data.
    Scales features fold-by-fold using parameters fit only on inner training folds.
    """
    n_samples = len(y_train)
    rng = np.random.default_rng(random_seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[:n_samples % n_splits] += 1

    current = 0
    fold_accuracies = []

    for size in fold_sizes:
        val_idx = indices[current:current + size]
        train_idx = np.concatenate([indices[:current], indices[current + size:]])
        current += size

        X_tr_raw = X_train[train_idx]
        y_tr = y_train[train_idx]
        X_va_raw = X_train[val_idx]
        y_va = y_train[val_idx]

        # Inner-fold scaling: fit strictly on X_tr_raw
        scaler = LeakageFreeStandardScaler()
        X_tr = scaler.fit_transform(X_tr_raw)
        X_va = scaler.transform(X_va_raw)

        model = RegularizedLogisticRegressionScratch(
            learning_rate=0.1,
            n_iterations=600,
            l2_lambda=l2_lambda
        ).fit(X_tr, y_tr)

        preds = model.predict(X_va)
        acc = float(np.mean(preds == y_va))
        fold_accuracies.append(acc)

    return float(np.mean(fold_accuracies))


def leakage_free_forward_selection(
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    n_splits: int = 5,
    l2_lambda: float = 0.1
) -> Tuple[List[int], List[str], List[float]]:
    """
    Greedy forward stepwise feature selection performed STRICTLY on training data.
    Iteratively adds the predictor yielding the highest inner cross-validated performance gain.
    Inner scaling is applied fold-specifically.
    Stops when no remaining feature improves cross-validated accuracy.
    """
    n_features = X_train_raw.shape[1]
    remaining = list(range(n_features))
    selected_indices: List[int] = []
    selected_names: List[str] = []
    cv_score_history: List[float] = []

    best_overall_score = 0.0

    while remaining:
        best_candidate: Optional[int] = None
        best_candidate_score = -1.0

        for candidate in remaining:
            trial_features = selected_indices + [candidate]
            X_subset = X_train_raw[:, trial_features]

            score = kfold_cross_validate_logistic(
                X_subset, y_train, n_splits=n_splits, l2_lambda=l2_lambda
            )

            if score > best_candidate_score:
                best_candidate_score = score
                best_candidate = candidate

        # Check if candidate improves upon previous iteration score
        if best_candidate is not None and best_candidate_score > best_overall_score:
            selected_indices.append(best_candidate)
            selected_names.append(feature_names[best_candidate])
            remaining.remove(best_candidate)
            best_overall_score = best_candidate_score
            cv_score_history.append(best_candidate_score)
        else:
            break

    return selected_indices, selected_names, cv_score_history
