"""
Empirical Evaluation Metrics, PR-AUC, and Statistical Confidence Interval Utilities
Author: Rishindra Mateti (Wright State University)

Provides formal classification metric computation, confusion matrix generation,
ROC/AUC calculation, Precision-Recall AUC (PR-AUC), and non-parametric bootstrap confidence interval estimation.
"""

from __future__ import annotations

from typing import Dict, Tuple, List, Optional
import numpy as np


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Computes 2x2 confusion matrix:
    [[TN, FP],
     [FN, TP]]
    """
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return np.array([[tn, fp], [fn, tp]], dtype=int)


def compute_classification_metrics(cm: np.ndarray) -> Dict[str, float]:
    """Computes Accuracy, Precision, Recall, Specificity, and F1-Score."""
    tn, fp = cm[0, 0], cm[0, 1]
    fn, tp = cm[1, 0], cm[1, 1]

    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2.0 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'specificity': float(specificity),
        'f1_score': float(f1),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp)
    }


def compute_roc_auc(y_true: np.ndarray, y_scores: np.ndarray, n_thresholds: int = 200) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Computes True Positive Rate (TPR), False Positive Rate (FPR), and AUC via the trapezoidal rule.
    """
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    tpr_list = []
    fpr_list = []

    positives = np.sum(y_true == 1)
    negatives = np.sum(y_true == 0)

    for thresh in thresholds:
        preds = (y_scores >= thresh).astype(int)
        tp = np.sum((y_true == 1) & (preds == 1))
        fp = np.sum((y_true == 0) & (preds == 1))

        tpr = tp / positives if positives > 0 else 0.0
        fpr = fp / negatives if negatives > 0 else 0.0

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    fpr_arr = np.array(fpr_list)
    tpr_arr = np.array(tpr_list)

    sort_idx = np.lexsort((tpr_arr, fpr_arr))
    fpr_sorted = fpr_arr[sort_idx]
    tpr_sorted = tpr_arr[sort_idx]

    if fpr_sorted[0] > 0.0 or tpr_sorted[0] > 0.0:
        fpr_sorted = np.insert(fpr_sorted, 0, 0.0)
        tpr_sorted = np.insert(tpr_sorted, 0, 0.0)
    if fpr_sorted[-1] < 1.0 or tpr_sorted[-1] < 1.0:
        fpr_sorted = np.append(fpr_sorted, 1.0)
        tpr_sorted = np.append(tpr_sorted, 1.0)

    if hasattr(np, 'trapezoid'):
        auc = float(np.trapezoid(tpr_sorted, fpr_sorted))
    elif hasattr(np, 'trapz'):
        auc = float(np.trapz(tpr_sorted, fpr_sorted))
    else:
        auc = float(np.sum((fpr_sorted[1:] - fpr_sorted[:-1]) * (tpr_sorted[1:] + tpr_sorted[:-1])) / 2.0)
    auc = max(0.0, min(1.0, auc))
    return fpr_sorted, tpr_sorted, auc


def compute_pr_auc(y_true: np.ndarray, y_scores: np.ndarray, n_thresholds: int = 200) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Computes Precision, Recall, and Area Under the Precision-Recall Curve (PR-AUC / Average Precision).
    """
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    precision_list = []
    recall_list = []

    positives = np.sum(y_true == 1)

    for thresh in thresholds:
        preds = (y_scores >= thresh).astype(int)
        tp = np.sum((y_true == 1) & (preds == 1))
        fp = np.sum((y_true == 0) & (preds == 1))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / positives if positives > 0 else 0.0

        precision_list.append(prec)
        recall_list.append(rec)

    precision_arr = np.array(precision_list)
    recall_arr = np.array(recall_list)

    # Sort in ascending order of recall for monotonic integration
    sort_idx = np.argsort(recall_arr)
    recall_sorted = recall_arr[sort_idx]
    precision_sorted = precision_arr[sort_idx]

    # Prepend (0, 1) and append (1, baseline)
    if recall_sorted[0] > 0.0:
        recall_sorted = np.insert(recall_sorted, 0, 0.0)
        precision_sorted = np.insert(precision_sorted, 0, 1.0)

    if hasattr(np, 'trapezoid'):
        pr_auc = float(np.trapezoid(precision_sorted, recall_sorted))
    elif hasattr(np, 'trapz'):
        pr_auc = float(np.trapz(precision_sorted, recall_sorted))
    else:
        pr_auc = float(np.sum((recall_sorted[1:] - recall_sorted[:-1]) * (precision_sorted[1:] + precision_sorted[:-1])) / 2.0)

    pr_auc = max(0.0, min(1.0, pr_auc))
    return recall_sorted, precision_sorted, pr_auc


def compute_bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    n_bootstraps: int = 1000,
    random_seed: int = 42
) -> Dict[str, Tuple[float, float]]:
    """
    Estimates 95% non-parametric bootstrap confidence intervals [2.5%, 97.5%]
    for holdout Accuracy, F1-Score, and AUC-ROC.
    """
    rng = np.random.default_rng(random_seed)
    n = len(y_true)

    boot_acc = []
    boot_f1 = []
    boot_auc = []

    for _ in range(n_bootstraps):
        boot_idx = rng.choice(n, size=n, replace=True)
        y_t = y_true[boot_idx]
        y_p = y_pred[boot_idx]
        y_s = y_prob[boot_idx]

        if len(np.unique(y_t)) < 2:
            continue

        cm = compute_confusion_matrix(y_t, y_p)
        m = compute_classification_metrics(cm)
        _, _, a = compute_roc_auc(y_t, y_s)

        boot_acc.append(m['accuracy'])
        boot_f1.append(m['f1_score'])
        boot_auc.append(a)

    return {
        'accuracy_ci': (float(np.percentile(boot_acc, 2.5)), float(np.percentile(boot_acc, 97.5))),
        'f1_ci': (float(np.percentile(boot_f1, 2.5)), float(np.percentile(boot_f1, 97.5))),
        'auc_ci': (float(np.percentile(boot_auc, 2.5)), float(np.percentile(boot_auc, 97.5)))
    }
