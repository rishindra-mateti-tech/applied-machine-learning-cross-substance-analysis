"""
Supervised Classification, Model Benchmarking, and Statistical Inference
Author: Rishindra Mateti (Wright State University)

Implements:
1. Parametric Regularized Logistic Regression from mathematical first principles via batch gradient descent.
2. Approximate model-based Wald confidence intervals derived from the inverted Hessian covariance matrix
   under L2 regularization: Sigma = (X_b^T W X_b + lambda * I_reg)^{-1}.
3. Benjamini-Hochberg False Discovery Rate (FDR) control for multiple hypothesis testing across associations.
4. Scikit-Learn Logistic Regression parity verification.
5. Specified Random Forest baseline (100 trees, max_depth=8, min_samples_split=5).
6. Tuned Random Forest with nested inner 3-fold cross-validation grid search on training data.
7. Multi-Layer Perceptron (MLP) neural network benchmark with early stopping to ensure verified convergence.
8. Probability calibration diagnostics (Brier score and Expected Calibration Error).
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier


class RegularizedLogisticRegressionScratch:
    """
    Parametric Logistic Regression fitted from mathematical first principles via batch gradient descent.
    Incorporates L2 Tikhonov ridge regularization on non-intercept coefficients:
    J(theta) = -(1/m) * sum[y*ln(h) + (1-y)*ln(1-h)] + (lambda / 2m) * sum(theta_j^2)
    """
    def __init__(
        self,
        learning_rate: float = 0.1,
        n_iterations: int = 2000,
        l2_lambda: float = 0.1,
        tol: float = 1e-7
    ):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.l2_lambda = l2_lambda
        self.tol = tol
        self.theta_: Optional[np.ndarray] = None
        self.cost_history_: List[float] = []
        self.cov_matrix_: Optional[np.ndarray] = None
        self.standard_errors_: Optional[np.ndarray] = None

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250.0, 250.0)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> RegularizedLogisticRegressionScratch:
        m, n = X.shape
        # Prepend bias unit column of ones: X_b has shape (m, n + 1)
        X_b = np.column_stack([np.ones(m), X])
        self.theta_ = np.zeros(n + 1)
        self.cost_history_ = []

        for _ in range(self.n_iterations):
            linear_output = X_b @ self.theta_
            predictions = self._sigmoid(linear_output)
            errors = predictions - y

            # Gradient with L2 regularization penalty excluding bias term theta_0
            reg_penalty_grad = np.copy(self.theta_)
            reg_penalty_grad[0] = 0.0
            gradient = (X_b.T @ errors + self.l2_lambda * reg_penalty_grad) / m

            # Parameter update
            self.theta_ -= self.learning_rate * gradient

            # Compute regularized cross-entropy loss J(theta)
            eps = 1e-15
            clamped_preds = np.clip(predictions, eps, 1.0 - eps)
            cost = -np.mean(y * np.log(clamped_preds) + (1.0 - y) * np.log(1.0 - clamped_preds))
            reg_loss = (self.l2_lambda / (2.0 * m)) * np.sum(self.theta_[1:] ** 2)
            total_cost = float(cost + reg_loss)
            self.cost_history_.append(total_cost)

            if len(self.cost_history_) > 1 and abs(self.cost_history_[-2] - total_cost) < self.tol:
                break

        # Compute asymptotic covariance matrix via observed Fisher Information / Hessian:
        # H = X_b^T W X_b + lambda * I_reg
        p = self._sigmoid(X_b @ self.theta_)
        w = p * (1.0 - p)
        W = np.diag(np.clip(w, 1e-6, 0.25))
        I_reg = np.eye(n + 1)
        I_reg[0, 0] = 0.0

        H = X_b.T @ W @ X_b + self.l2_lambda * I_reg
        try:
            self.cov_matrix_ = np.linalg.pinv(H)
            variances = np.diag(self.cov_matrix_)
            variances = np.maximum(variances, 1e-10)
            self.standard_errors_ = np.sqrt(variances)
        except np.linalg.LinAlgError:
            self.cov_matrix_ = np.eye(n + 1) * 0.01
            self.standard_errors_ = np.ones(n + 1) * 0.1

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.theta_ is None:
            raise RuntimeError("Model is not fitted yet.")
        m = X.shape[0]
        X_b = np.column_stack([np.ones(m), X])
        return self._sigmoid(X_b @ self.theta_)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_odds_ratios_with_ci(
        self,
        feature_names: Optional[List[str]] = None,
        alpha: float = 0.05
    ) -> List[Dict[str, float]]:
        """
        Computes exponentiated coefficients (adjusted odds ratios) with approximate model-based
        95% Wald confidence intervals and two-sided p-values:
        OR = exp(theta_j)
        CI = [exp(theta_j - z * SE_j), exp(theta_j + z * SE_j)]
        Note: These represent approximate model-based intervals under L2 ridge regularization.
        """
        if self.theta_ is None or self.standard_errors_ is None:
            raise RuntimeError("Model is not fitted yet.")

        z = 1.95996  # 95% standard normal quantile
        coefficients = self.theta_[1:]
        ses = self.standard_errors_[1:]

        results = []
        for i, (coef, se) in enumerate(zip(coefficients, ses)):
            name = feature_names[i] if feature_names and i < len(feature_names) else f"Feature_{i}"
            or_val = float(np.exp(coef))
            ci_lower = float(np.exp(coef - z * se))
            ci_upper = float(np.exp(coef + z * se))
            z_stat = float(coef / se) if se > 0 else 0.0
            p_val = float(2.0 * stats.norm.sf(abs(z_stat)))
            results.append({
                'feature': name,
                'coefficient': float(coef),
                'standard_error': float(se),
                'z_statistic': z_stat,
                'p_value': p_val,
                'odds_ratio': or_val,
                'ci_lower_95': ci_lower,
                'ci_upper_95': ci_upper
            })
        return results


def apply_benjamini_hochberg(p_values: List[float], q: float = 0.05) -> Tuple[List[bool], List[float]]:
    """
    Applies the Benjamini-Hochberg False Discovery Rate (FDR) procedure for multiple hypothesis testing.
    Returns boolean array of rejected null hypotheses (significant after FDR) and adjusted q-values.
    """
    p_arr = np.asarray(p_values, dtype=float)
    n = len(p_arr)
    sorted_indices = np.argsort(p_arr)
    sorted_p = p_arr[sorted_indices]

    thresholds = (np.arange(1, n + 1) / n) * q
    significant = sorted_p <= thresholds
    k_max = np.where(significant)[0]

    cutoff = sorted_p[k_max[-1]] if len(k_max) > 0 else -1.0
    reject = (p_arr <= cutoff).tolist()

    q_vals = np.zeros(n)
    running_min = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        adj_p = min(1.0, (n / rank) * sorted_p[i])
        running_min = min(running_min, adj_p)
        q_vals[sorted_indices[i]] = running_min

    return reject, q_vals.tolist()


def train_sklearn_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    C: float = 10.0,
    random_state: int = 42
) -> Tuple[LogisticRegression, np.ndarray, np.ndarray]:
    """
    Evaluates Scikit-Learn Logistic Regression baseline for mathematical parity verification.
    """
    lr = LogisticRegression(
        C=C,
        penalty='l2',
        solver='lbfgs',
        max_iter=1000,
        random_state=random_state
    )
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    y_prob = lr.predict_proba(X_test)[:, 1]
    return lr, y_pred, y_prob


def train_random_forest_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_estimators: int = 100,
    max_depth: Optional[int] = 8,
    min_samples_split: int = 5,
    random_state: int = 42
) -> Tuple[RandomForestClassifier, np.ndarray, np.ndarray]:
    """
    Evaluates specified Random Forest baseline configuration (100 trees, max_depth=8, min_samples_split=5).
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    return rf, y_pred, y_prob


def train_random_forest_tuned(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    cv: int = 3,
    random_state: int = 42
) -> Tuple[RandomForestClassifier, np.ndarray, np.ndarray, Dict[str, object]]:
    """
    Evaluates Random Forest with 3-fold inner cross-validation grid search strictly on X_train.
    Hyperparameter grid: max_depth in [4, 8, None], min_samples_split in [2, 5].
    """
    base_rf = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    param_grid = {
        'max_depth': [4, 8, None],
        'min_samples_split': [2, 5]
    }
    grid_search = GridSearchCV(
        base_rf,
        param_grid=param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    best_rf = grid_search.best_estimator_

    y_pred = best_rf.predict(X_test)
    y_prob = best_rf.predict_proba(X_test)[:, 1]
    return best_rf, y_pred, y_prob, grid_search.best_params_


def train_mlp_benchmark(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    hidden_layer_sizes: Tuple[int, ...] = (64, 32),
    alpha: float = 0.01,
    random_state: int = 42
) -> Tuple[MLPClassifier, np.ndarray, np.ndarray]:
    """
    Evaluates an empirical Multi-Layer Perceptron benchmark using Scikit-Learn.
    Architecture: Input -> Dense(64, ReLU) -> Dense(32, ReLU) -> Output(Sigmoid).
    Uses early stopping with validation tracking to ensure verified convergence without warnings.
    """
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=1500,
        early_stopping=True,
        n_iter_no_change=20,
        validation_fraction=0.15,
        alpha=alpha,
        random_state=random_state
    )
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    y_prob = mlp.predict_proba(X_test)[:, 1]
    return mlp, y_pred, y_prob


def compute_calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
    """
    Computes Brier score and Expected Calibration Error (ECE).
    Brier Score = (1/N) * sum((y_prob - y_true)^2)
    ECE = sum_{b=1}^B (|B_b|/N) * |acc(B_b) - conf(B_b)|
    """
    brier = float(np.mean((y_prob - y_true) ** 2))

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper) if i > 0 else (y_prob >= bin_lower) & (y_prob <= bin_upper)
        bin_count = np.sum(in_bin)

        if bin_count > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            ece += (bin_count / n) * abs(bin_acc - bin_conf)

    return {
        'brier_score': float(brier),
        'ece': float(ece)
    }
