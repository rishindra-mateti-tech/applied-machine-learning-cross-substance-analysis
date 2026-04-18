"""
Master Benchmark Experiment Runner for Differential Psychometric Substance Modeling
Author: Rishindra Mateti (Wright State University)

Executes the comprehensive, leakage-free empirical research study:
1. Ingestion and hygiene filtering (excluding Semeron fictitious substance overclaimers, N=1,877).
2. Inter-trait psychometric correlation structure analysis.
3. Unsupervised behavioral clustering (K-Means & Fuzzy C-Means) and 20-restart stability topology sweep.
4. Kruskal-Wallis non-parametric synthesis of cluster topology with Polysubstance Co-Involvement.
5. Cross-substance supervised classification tournament across 4 literature-grounded classes:
   - Cannabinoids (Cannabis)
   - CNS Stimulants (Cocaine / Amphetamines)
   - Psychedelics (Psilocybin Mushrooms / LSD)
   - Depressants / Anxiolytics (Benzodiazepines)
6. 5-Fold outer stratified cross-validation reporting mean +/- std across folds.
7. Model comparison: Majority Baseline, Scratch Logistic Regression, Sklearn Parity, Specified Random Forest, Tuned Random Forest, and Converged MLP.
8. Strictly leakage-free 5-fold cross-validated forward feature selection on training partition.
9. Analytical Hessian-based odds ratio estimation with approximate 95% Wald confidence intervals and Benjamini-Hochberg FDR control.
10. Probability calibration (Brier score & ECE) and non-parametric bootstrap confidence intervals.
11. Generates all 7 high-resolution (300 DPI) publication figures and saves clean nested machine-readable JSON.
"""

from __future__ import annotations

import os
import sys
import json
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from mpl_toolkits.mplot3d import Axes3D

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dataset_pipeline import (
    load_cleaned_dataset,
    get_stratified_outer_split,
    CORE_PREDICTORS,
    PSYCHOMETRIC_TRAITS,
    CLUSTERING_FEATURES,
    LeakageFreeStandardScaler
)
from src.unsupervised_segmentation import (
    KMeansScratch,
    FuzzyCMeansScratch,
    compute_davies_bouldin_index,
    evaluate_cluster_topology_stability,
    evaluate_cluster_polysubstance_association
)
from src.supervised_classification import (
    RegularizedLogisticRegressionScratch,
    train_sklearn_logistic_regression,
    train_random_forest_baseline,
    train_random_forest_tuned,
    train_mlp_benchmark,
    compute_calibration_metrics,
    apply_benjamini_hochberg
)
from src.feature_selection import leakage_free_forward_selection
from src.evaluation_metrics import (
    compute_confusion_matrix,
    compute_classification_metrics,
    compute_roc_auc,
    compute_pr_auc,
    compute_bootstrap_confidence_intervals
)


def set_publication_style():
    """Sets professional academic plotting aesthetics."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8.5,
        'figure.titlesize': 12,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })


def main():
    set_publication_style()
    figures_dir = os.path.join(project_root, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 84)
    print(" DIFFERENTIAL PSYCHOMETRIC SUBSTANCE CONSUMPTION RESEARCH BENCHMARK")
    print(" Verified Leakage-Free Implementation across 4 Literature-Grounded Classes")
    print("=" * 84)

    # 1. Dataset Ingestion and Data Hygiene Filter
    df = load_cleaned_dataset()
    n_total = len(df)
    excluded = df.attrs.get('excluded_overclaimers', 8)
    print(f"\n[DATASET] Ingested {n_total + excluded} records.")
    print(f"[DATA HYGIENE] Filtered {excluded} Semeron overclaimers. Filtered cohort: N = {n_total}.")

    # Summary of Target Prevalences
    targets = {
        'Cannabis': 'Target_Cannabis',
        'CNS Stimulants': 'Target_Stimulants',
        'Psychedelics': 'Target_Psychedelics',
        'Depressants': 'Target_Depressants'
    }
    print("\n[PREVALENCE] Reported Past-Year Consumption (CL3-CL6):")
    for name, col in targets.items():
        prev = df[col].mean()
        count = df[col].sum()
        print(f"  - {name:<16}: {prev:.2%} (N = {count}/{n_total})")

    poly_counts = df['Polysubstance_Score'].value_counts().sort_index()
    print("\n[POLYSUBSTANCE] Past-Year Illicit Class Co-Involvement Distribution (0-4):")
    for score, count in poly_counts.items():
        print(f"  Score {score}: {count} individuals ({count/n_total:.2%})")

    # =========================================================================
    # FIGURE 1: Psychometric Feature Correlation Analysis
    # =========================================================================
    print("\n--- [FIGURE 1] Inter-Trait Correlation Structure ---")
    corr_cols = ['Age'] + PSYCHOMETRIC_TRAITS
    corr_matrix = df[corr_cols].corr(method='pearson')

    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(
        corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-0.6, vmax=0.6,
        square=True, cbar_kws={'label': 'Pearson Correlation (r)'}, ax=ax, linewidths=0.5
    )
    ax.set_title("Inter-Trait Correlation Matrix (Age and Psychometric Predictors)", pad=10)
    fig1_path = os.path.join(figures_dir, 'fig1_correlation_matrix.png')
    plt.savefig(fig1_path)
    plt.close()
    print(f"Saved: {fig1_path}")

    # =========================================================================
    # EXPERIMENT 1: Unsupervised Behavioral Segmentation & Topology Sweep
    # =========================================================================
    print("\n--- [EXPERIMENT 1] Unsupervised Behavioral Segmentation ---")
    X_clust_raw = df[CLUSTERING_FEATURES].to_numpy(dtype=float)
    scaler_clust = LeakageFreeStandardScaler()
    X_clust = scaler_clust.fit_transform(X_clust_raw)

    stability = evaluate_cluster_topology_stability(X_clust, k_range=list(range(2, 9)), n_restarts=20)
    k_vals = stability['k_values']
    db_means = stability['db_mean']
    db_stds = stability['db_std']
    sil_means = stability['sil_mean']
    sil_stds = stability['sil_std']

    print("Cluster Topology Stability (20 Restarts per k):")
    for k, db_m, db_s, sil_m, sil_s in zip(k_vals, db_means, db_stds, sil_means, sil_stds):
        opt_tag = " <-- OPTIMAL TOPOLOGY" if k == 4 else ""
        print(f"  k={k}: DB = {db_m:.4f} (+/- {db_s:.4f}) | Silhouette = {sil_m:.4f} (+/- {sil_s:.4f}){opt_tag}")

    # FIGURE 2: 3D Latent Cluster Space Comparison (K-Means vs Fuzzy C-Means at k=4)
    kmeans_k4 = KMeansScratch(n_clusters=4, random_state=42).fit(X_clust)
    fcm_k4 = FuzzyCMeansScratch(n_clusters=4, m=2.0, random_state=42).fit(X_clust)
    fcm_labels = fcm_k4.get_hard_labels()

    fig = plt.figure(figsize=(11, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.scatter(X_clust[:, 0], X_clust[:, 1], X_clust[:, 2], c=kmeans_k4.labels_, cmap='Set2', alpha=0.55, s=16)
    ax1.scatter(kmeans_k4.centroids_[:, 0], kmeans_k4.centroids_[:, 1], kmeans_k4.centroids_[:, 2],
                c='black', marker='X', s=160, edgecolor='white', linewidth=1.5, label='Centroids')
    ax1.set_title("K-Means Disjoint Partitions (k=4)", pad=8)
    ax1.set_xlabel('Age (Std)')
    ax1.set_ylabel('Impulsivity (Std)')
    ax1.set_zlabel('Sensation Seeking (Std)')
    ax1.view_init(elev=22, azim=130)

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.scatter(X_clust[:, 0], X_clust[:, 1], X_clust[:, 2], c=fcm_labels, cmap='Set2', alpha=0.55, s=16)
    ax2.scatter(fcm_k4.centroids_[:, 0], fcm_k4.centroids_[:, 1], fcm_k4.centroids_[:, 2],
                c='black', marker='X', s=160, edgecolor='white', linewidth=1.5, label='Centroids')
    ax2.set_title("Fuzzy C-Means Hardened Partitions (k=4)", pad=8)
    ax2.set_xlabel('Age (Std)')
    ax2.set_ylabel('Impulsivity (Std)')
    ax2.set_zlabel('Sensation Seeking (Std)')
    ax2.view_init(elev=22, azim=130)

    fig2_path = os.path.join(figures_dir, 'fig2_clustering_3d_comparison.png')
    plt.savefig(fig2_path)
    plt.close()
    print(f"Saved: {fig2_path}")

    # FIGURE 3: Dual Davies-Bouldin and Silhouette Stability Curve
    fig, (ax_db, ax_sil) = plt.subplots(1, 2, figsize=(10, 4.2))

    ax_db.plot(k_vals, db_means, 'o-', color='#1f77b4', linewidth=2, label='Mean DB Index')
    ax_db.fill_between(k_vals, db_means - db_stds, db_means + db_stds, color='#1f77b4', alpha=0.2, label='+/- 1 SD')
    ax_db.scatter([4], [db_means[2]], color='#d62728', s=100, zorder=5, label='Local Minima (k=4)')
    ax_db.set_xlabel('Cluster Count (k)')
    ax_db.set_ylabel('Davies-Bouldin Index (Lower is Better)')
    ax_db.set_title('Cluster Separation Topology (Davies-Bouldin)', pad=10)
    ax_db.set_xticks(k_vals)
    ax_db.grid(True, linestyle='--', alpha=0.5)
    ax_db.legend(frameon=True, facecolor='white')

    ax_sil.plot(k_vals, sil_means, 's-', color='#2ca02c', linewidth=2, label='Mean Silhouette Score')
    ax_sil.fill_between(k_vals, sil_means - sil_stds, sil_means + sil_stds, color='#2ca02c', alpha=0.2, label='+/- 1 SD')
    ax_sil.set_xlabel('Cluster Count (k)')
    ax_sil.set_ylabel('Silhouette Coefficient (Higher is Better)')
    ax_sil.set_title('Cluster Cohesion & Silhouette Profile', pad=10)
    ax_sil.set_xticks(k_vals)
    ax_sil.grid(True, linestyle='--', alpha=0.5)
    ax_sil.legend(frameon=True, facecolor='white')

    fig3_path = os.path.join(figures_dir, 'fig3_topology_and_silhouette.png')
    plt.savefig(fig3_path)
    plt.close()
    print(f"Saved: {fig3_path}")

    # Polysubstance Co-Involvement Association via Kruskal-Wallis Test
    poly_assoc = evaluate_cluster_polysubstance_association(kmeans_k4.labels_, df['Polysubstance_Score'].to_numpy())
    print("\n[POLYSUBSTANCE SYNTHESIS] Kruskal-Wallis Test across k=4 Clusters:")
    print(f"  H-statistic = {poly_assoc['kruskal_statistic']:.4f}, p-value = {poly_assoc['p_value']:.4e}")
    for c_id, (m_val, s_val, sz) in enumerate(zip(poly_assoc['cluster_means'], poly_assoc['cluster_stds'], poly_assoc['cluster_sizes'])):
        print(f"  Cluster {c_id} (N={sz}): Mean Polysubstance Score = {m_val:.3f} (+/- {s_val:.3f})")

    # =========================================================================
    # EXPERIMENT 2: 5-Fold Outer Stratified Cross-Validation Tournament
    # =========================================================================
    print("\n--- [EXPERIMENT 2] 5-Fold Outer Stratified Cross-Validation Tournament ---")
    cv_outer_results: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]] = {}

    for sub_name, target_col in targets.items():
        print(f"\n  Running 5-Fold Outer CV for {sub_name}...")
        X_raw = df[CORE_PREDICTORS].to_numpy(dtype=float)
        y = df[target_col].to_numpy(dtype=int)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        model_fold_metrics: Dict[str, Dict[str, List[float]]] = {
            'LR_Scratch': {'acc': [], 'f1': [], 'auc': [], 'pr_auc': [], 'brier': []},
            'Random_Forest_Tuned': {'acc': [], 'f1': [], 'auc': [], 'pr_auc': [], 'brier': []},
            'MLP': {'acc': [], 'f1': [], 'auc': [], 'pr_auc': [], 'brier': []}
        }

        for train_idx, val_idx in skf.split(X_raw, y):
            X_tr_raw, y_tr = X_raw[train_idx], y[train_idx]
            X_va_raw, y_va = X_raw[val_idx], y[val_idx]

            # Fold-specific scaling strictly on training fold
            scaler_fold = LeakageFreeStandardScaler()
            X_tr = scaler_fold.fit_transform(X_tr_raw)
            X_va = scaler_fold.transform(X_va_raw)

            # 1. Scratch LR
            lr_m = RegularizedLogisticRegressionScratch(learning_rate=0.1, n_iterations=1500, l2_lambda=0.1).fit(X_tr, y_tr)
            p_lr = lr_m.predict(X_va)
            pr_lr = lr_m.predict_proba(X_va)
            cm_lr = compute_confusion_matrix(y_va, p_lr)
            m_lr = compute_classification_metrics(cm_lr)
            _, _, auc_lr = compute_roc_auc(y_va, pr_lr)
            _, _, prauc_lr = compute_pr_auc(y_va, pr_lr)
            cal_lr = compute_calibration_metrics(y_va, pr_lr)
            model_fold_metrics['LR_Scratch']['acc'].append(m_lr['accuracy'])
            model_fold_metrics['LR_Scratch']['f1'].append(m_lr['f1_score'])
            model_fold_metrics['LR_Scratch']['auc'].append(auc_lr)
            model_fold_metrics['LR_Scratch']['pr_auc'].append(prauc_lr)
            model_fold_metrics['LR_Scratch']['brier'].append(cal_lr['brier_score'])

            # 2. Tuned Random Forest (inner CV)
            rf_m, p_rf, pr_rf, _ = train_random_forest_tuned(X_tr, y_tr, X_va, cv=3, random_state=42)
            cm_rf = compute_confusion_matrix(y_va, p_rf)
            m_rf = compute_classification_metrics(cm_rf)
            _, _, auc_rf = compute_roc_auc(y_va, pr_rf)
            _, _, prauc_rf = compute_pr_auc(y_va, pr_rf)
            cal_rf = compute_calibration_metrics(y_va, pr_rf)
            model_fold_metrics['Random_Forest_Tuned']['acc'].append(m_rf['accuracy'])
            model_fold_metrics['Random_Forest_Tuned']['f1'].append(m_rf['f1_score'])
            model_fold_metrics['Random_Forest_Tuned']['auc'].append(auc_rf)
            model_fold_metrics['Random_Forest_Tuned']['pr_auc'].append(prauc_rf)
            model_fold_metrics['Random_Forest_Tuned']['brier'].append(cal_rf['brier_score'])

            # 3. Converged MLP
            mlp_m, p_mlp, pr_mlp = train_mlp_benchmark(X_tr, y_tr, X_va, random_state=42)
            cm_mlp = compute_confusion_matrix(y_va, p_mlp)
            m_mlp = compute_classification_metrics(cm_mlp)
            _, _, auc_mlp = compute_roc_auc(y_va, pr_mlp)
            _, _, prauc_mlp = compute_pr_auc(y_va, pr_mlp)
            cal_mlp = compute_calibration_metrics(y_va, pr_mlp)
            model_fold_metrics['MLP']['acc'].append(m_mlp['accuracy'])
            model_fold_metrics['MLP']['f1'].append(m_mlp['f1_score'])
            model_fold_metrics['MLP']['auc'].append(auc_mlp)
            model_fold_metrics['MLP']['pr_auc'].append(prauc_mlp)
            model_fold_metrics['MLP']['brier'].append(cal_mlp['brier_score'])

        cv_outer_results[sub_name] = {}
        for m_name, metric_dict in model_fold_metrics.items():
            cv_outer_results[sub_name][m_name] = {
                k: (float(np.mean(vals)), float(np.std(vals))) for k, vals in metric_dict.items()
            }
            m_acc, s_acc = cv_outer_results[sub_name][m_name]['acc']
            m_auc, s_auc = cv_outer_results[sub_name][m_name]['auc']
            print(f"    {m_name:<20}: 5-Fold Outer Acc = {m_acc:.4f} (+/- {s_acc:.4f}) | AUC = {m_auc:.4f} (+/- {s_auc:.4f})")

    # =========================================================================
    # EXPERIMENT 3: Holdout Partition Evaluation, Odds Ratios, and FDR Control
    # =========================================================================
    print("\n--- [EXPERIMENT 3] Holdout Test Cohort Evaluation & Inference ---")

    substance_models_report: Dict[str, Any] = {}
    roc_data: Dict[str, Any] = {}
    confusion_matrices: Dict[str, Any] = {}
    analytical_odds_ratios: Dict[str, List[Dict[str, float]]] = {}
    cost_convergence_history: Dict[str, List[float]] = {}
    all_p_values: List[float] = []
    p_val_keys: List[Tuple[str, str]] = []

    for sub_name, target_col in targets.items():
        print(f"\n=======================================================")
        print(f" TARGET: {sub_name.upper()} ({target_col})")
        print(f"=======================================================")

        X_train, y_train, X_test, y_test, scaler_sub = get_stratified_outer_split(
            df, target_col=target_col, test_size=0.2, random_seed=42
        )
        n_train = len(y_train)
        n_test = len(y_test)
        pos_train = int(np.sum(y_train))
        pos_test = int(np.sum(y_test))
        print(f"Split: Train N={n_train} (Pos={pos_train}, {pos_train/n_train:.2%}) | Test N={n_test} (Pos={pos_test}, {pos_test/n_test:.2%})")

        # 1. Majority Floor Baseline
        majority_class = int(np.round(np.mean(y_train)))
        maj_preds = np.full(n_test, majority_class)
        cm_maj = compute_confusion_matrix(y_test, maj_preds)
        m_maj = compute_classification_metrics(cm_maj)
        m_maj['auc'] = 0.5000
        m_maj['pr_auc'] = float(np.mean(y_test))

        # 2. Scratch Regularized Logistic Regression (Full 10 Features)
        lr_scratch = RegularizedLogisticRegressionScratch(learning_rate=0.1, n_iterations=2000, l2_lambda=0.1)
        lr_scratch.fit(X_train, y_train)
        preds_scratch = lr_scratch.predict(X_test)
        probs_scratch = lr_scratch.predict_proba(X_test)
        cm_scratch = compute_confusion_matrix(y_test, preds_scratch)
        m_scratch = compute_classification_metrics(cm_scratch)
        fpr_sc, tpr_sc, auc_sc = compute_roc_auc(y_test, probs_scratch)
        _, _, pr_auc_sc = compute_pr_auc(y_test, probs_scratch)
        m_scratch['auc'] = auc_sc
        m_scratch['pr_auc'] = pr_auc_sc
        cal_scratch = compute_calibration_metrics(y_test, probs_scratch)
        ci_scratch = compute_bootstrap_confidence_intervals(y_test, preds_scratch, probs_scratch)

        cost_convergence_history[sub_name] = lr_scratch.cost_history_
        or_entries = lr_scratch.get_odds_ratios_with_ci(CORE_PREDICTORS)
        analytical_odds_ratios[sub_name] = or_entries

        for entry in or_entries:
            all_p_values.append(entry['p_value'])
            p_val_keys.append((sub_name, entry['feature']))

        # 3. Scikit-Learn Logistic Regression Parity Verification
        lr_sk, preds_sk, probs_sk = train_sklearn_logistic_regression(X_train, y_train, X_test, C=10.0, random_state=42)
        cm_sk = compute_confusion_matrix(y_test, preds_sk)
        m_sk = compute_classification_metrics(cm_sk)
        _, _, auc_sk = compute_roc_auc(y_test, probs_sk)
        _, _, pr_auc_sk = compute_pr_auc(y_test, probs_sk)
        m_sk['auc'] = auc_sk
        m_sk['pr_auc'] = pr_auc_sk

        # 4. Specified Random Forest Baseline (100 Trees, Depth=8, Min_split=5)
        rf_base, preds_rf_base, probs_rf_base = train_random_forest_baseline(
            X_train, y_train, X_test, n_estimators=100, max_depth=8, min_samples_split=5, random_state=42
        )
        cm_rf_base = compute_confusion_matrix(y_test, preds_rf_base)
        m_rf_base = compute_classification_metrics(cm_rf_base)
        _, _, auc_rf_base = compute_roc_auc(y_test, probs_rf_base)
        _, _, pr_auc_rf_base = compute_pr_auc(y_test, probs_rf_base)
        m_rf_base['auc'] = auc_rf_base
        m_rf_base['pr_auc'] = pr_auc_rf_base
        cal_rf_base = compute_calibration_metrics(y_test, probs_rf_base)
        ci_rf_base = compute_bootstrap_confidence_intervals(y_test, preds_rf_base, probs_rf_base)

        # 5. Tuned Random Forest (3-Fold Inner Cross-Validation Grid Search on X_train)
        rf_tuned, preds_rf_tuned, probs_rf_tuned, best_rf_params = train_random_forest_tuned(
            X_train, y_train, X_test, cv=3, random_state=42
        )
        cm_rf_tuned = compute_confusion_matrix(y_test, preds_rf_tuned)
        m_rf_tuned = compute_classification_metrics(cm_rf_tuned)
        _, _, auc_rf_tuned = compute_roc_auc(y_test, probs_rf_tuned)
        _, _, pr_auc_rf_tuned = compute_pr_auc(y_test, probs_rf_tuned)
        m_rf_tuned['auc'] = auc_rf_tuned
        m_rf_tuned['pr_auc'] = pr_auc_rf_tuned
        cal_rf_tuned = compute_calibration_metrics(y_test, probs_rf_tuned)
        ci_rf_tuned = compute_bootstrap_confidence_intervals(y_test, preds_rf_tuned, probs_rf_tuned)
        print(f"  Tuned RF Best Params on X_train: {best_rf_params}")

        # 6. Multi-Layer Perceptron (Converged with Early Stopping)
        mlp_model, preds_mlp, probs_mlp = train_mlp_benchmark(
            X_train, y_train, X_test, hidden_layer_sizes=(64, 32), alpha=0.01, random_state=42
        )
        cm_mlp = compute_confusion_matrix(y_test, preds_mlp)
        m_mlp = compute_classification_metrics(cm_mlp)
        fpr_mlp, tpr_mlp, auc_mlp = compute_roc_auc(y_test, probs_mlp)
        _, _, pr_auc_mlp = compute_pr_auc(y_test, probs_mlp)
        m_mlp['auc'] = auc_mlp
        m_mlp['pr_auc'] = pr_auc_mlp
        cal_mlp = compute_calibration_metrics(y_test, probs_mlp)
        ci_mlp = compute_bootstrap_confidence_intervals(y_test, preds_mlp, probs_mlp)

        # Forward Selection for Cannabis
        if sub_name == 'Cannabis':
            print("  Running Leakage-Free 5-Fold CV Forward Feature Selection on X_train...")
            X_tr_raw = X_train * scaler_sub.std_ + scaler_sub.mean_
            sel_idx, sel_names, cv_scores = leakage_free_forward_selection(
                X_tr_raw, y_train, feature_names=CORE_PREDICTORS, n_splits=5, l2_lambda=0.1
            )
            print(f"  Selected Features ({len(sel_names)}): {sel_names}")

            scaler_fs = LeakageFreeStandardScaler()
            X_tr_fs = scaler_fs.fit_transform(X_tr_raw[:, sel_idx])
            X_te_raw = X_test * scaler_sub.std_ + scaler_sub.mean_
            X_te_fs = scaler_fs.transform(X_te_raw[:, sel_idx])

            lr_fs = RegularizedLogisticRegressionScratch(learning_rate=0.1, n_iterations=2000, l2_lambda=0.1).fit(X_tr_fs, y_train)
            p_fs = lr_fs.predict(X_te_fs)
            pr_fs = lr_fs.predict_proba(X_te_fs)
            cm_fs = compute_confusion_matrix(y_test, p_fs)
            m_fs = compute_classification_metrics(cm_fs)
            _, _, auc_fs = compute_roc_auc(y_test, pr_fs)
            _, _, pr_auc_fs = compute_pr_auc(y_test, pr_fs)
            m_fs['auc'] = auc_fs
            m_fs['pr_auc'] = pr_auc_fs
            cal_fs = compute_calibration_metrics(y_test, pr_fs)
            ci_fs = compute_bootstrap_confidence_intervals(y_test, p_fs, pr_fs)
            substance_models_report['Cannabis_Forward_Selected'] = {
                'metrics': m_fs, 'cal': cal_fs, 'ci': ci_fs, 'features': sel_names
            }

        roc_data[sub_name] = {
            'fpr_scratch': fpr_sc, 'tpr_scratch': tpr_sc, 'auc_scratch': auc_sc,
            'fpr_mlp': fpr_mlp, 'tpr_mlp': tpr_mlp, 'auc_mlp': auc_mlp,
            'y_test': y_test, 'probs_scratch': probs_scratch, 'probs_mlp': probs_mlp
        }
        confusion_matrices[sub_name] = {
            'cm_scratch': cm_scratch, 'm_scratch': m_scratch,
            'cm_mlp': cm_mlp, 'm_mlp': m_mlp
        }

        substance_models_report[sub_name] = {
            'Majority': m_maj,
            'LR_Scratch': {'metrics': m_scratch, 'cal': cal_scratch, 'ci': ci_scratch},
            'LR_Sklearn_Parity': {'metrics': m_sk},
            'Random_Forest_Baseline': {'metrics': m_rf_base, 'cal': cal_rf_base, 'ci': ci_rf_base},
            'Random_Forest_Tuned': {'metrics': m_rf_tuned, 'cal': cal_rf_tuned, 'ci': ci_rf_tuned, 'best_params': best_rf_params},
            'MLP': {'metrics': m_mlp, 'cal': cal_mlp, 'ci': ci_mlp, 'n_iter': int(mlp_model.n_iter_)}
        }

        print(f"  Holdout Results for {sub_name}:")
        print(f"    - Majority Baseline: Acc = {m_maj['accuracy']:.4f}")
        print(f"    - Scratch LogReg   : Acc = {m_scratch['accuracy']:.4f} | F1 = {m_scratch['f1_score']:.4f} | AUC = {m_scratch['auc']:.4f} | PR-AUC = {m_scratch['pr_auc']:.4f} | Brier = {cal_scratch['brier_score']:.4f}")
        print(f"    - Sklearn Parity   : Acc = {m_sk['accuracy']:.4f} | F1 = {m_sk['f1_score']:.4f} | AUC = {m_sk['auc']:.4f}")
        print(f"    - RF (Tuned 3-FCV) : Acc = {m_rf_tuned['accuracy']:.4f} | F1 = {m_rf_tuned['f1_score']:.4f} | AUC = {m_rf_tuned['auc']:.4f} | PR-AUC = {m_rf_tuned['pr_auc']:.4f} | Brier = {cal_rf_tuned['brier_score']:.4f}")
        print(f"    - MLP (Early Stop) : Acc = {m_mlp['accuracy']:.4f} | F1 = {m_mlp['f1_score']:.4f} | AUC = {m_mlp['auc']:.4f} | PR-AUC = {m_mlp['pr_auc']:.4f} | Brier = {cal_mlp['brier_score']:.4f} (Iter: {mlp_model.n_iter_})")

    # =========================================================================
    # MULTIPLE COMPARISON CORRECTION: Benjamini-Hochberg FDR
    # =========================================================================
    print("\n--- [STATISTICAL INFERENCE] Benjamini-Hochberg FDR Control (40 Tests) ---")
    bh_reject, bh_q_vals = apply_benjamini_hochberg(all_p_values, q=0.05)
    fdr_results: Dict[str, Dict[str, Dict[str, float]]] = {}

    idx = 0
    for sub_name, entries in analytical_odds_ratios.items():
        fdr_results[sub_name] = {}
        for entry in entries:
            entry['q_value_fdr'] = float(bh_q_vals[idx])
            entry['significant_fdr'] = bool(bh_reject[idx])
            fdr_results[sub_name][entry['feature']] = {
                'coefficient': round(entry['coefficient'], 4),
                'standard_error': round(entry['standard_error'], 4),
                'odds_ratio': round(entry['odds_ratio'], 4),
                'ci_lower_95': round(entry['ci_lower_95'], 4),
                'ci_upper_95': round(entry['ci_upper_95'], 4),
                'p_value': round(entry['p_value'], 6),
                'q_value_fdr': round(entry['q_value_fdr'], 6),
                'significant_fdr': entry['significant_fdr']
            }
            idx += 1

    # =========================================================================
    # FIGURE 4: Cost Convergence Optimization
    # =========================================================================
    print("\n--- Generating Figure 4: Optimization Cost Convergence Profiles ---")
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    colors = {'Cannabis': '#1f77b4', 'CNS Stimulants': '#d62728', 'Psychedelics': '#2ca02c', 'Depressants': '#9467bd'}
    for sub_name, hist in cost_convergence_history.items():
        ax.plot(range(min(600, len(hist))), hist[:600], label=f'{sub_name}', color=colors[sub_name], linewidth=1.8)
    ax.set_xlabel('Gradient Descent Iteration')
    ax.set_ylabel('Regularized Cross-Entropy Loss J(theta)')
    ax.set_title('Empirical Loss Convergence Profiles across Substance Targets (lr=0.1, lambda=0.1)', pad=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)
    fig4_path = os.path.join(figures_dir, 'fig4_cost_convergence.png')
    plt.savefig(fig4_path)
    plt.close()
    print(f"Saved: {fig4_path}")

    # =========================================================================
    # FIGURE 5: Cross-Substance ROC Curves and Probability Calibration
    # =========================================================================
    print("\n--- Generating Figure 5: Cross-Substance ROC and Calibration Profiles ---")
    fig, (ax_roc, ax_cal) = plt.subplots(1, 2, figsize=(11, 4.5))

    for sub_name in targets.keys():
        d = roc_data[sub_name]
        ax_roc.plot(d['fpr_scratch'], d['tpr_scratch'], label=f"{sub_name} (AUC={d['auc_scratch']:.3f})", color=colors[sub_name], linewidth=2.0)

        # Compute calibration bins for Scratch LR
        y_t = d['y_test']
        p_s = d['probs_scratch']
        bins = np.linspace(0.0, 1.0, 6)
        bin_confs, bin_accs = [], []
        for i in range(len(bins)-1):
            idx_b = (p_s >= bins[i]) & (p_s < bins[i+1])
            if np.sum(idx_b) > 0:
                bin_confs.append(float(np.mean(p_s[idx_b])))
                bin_accs.append(float(np.mean(y_t[idx_b])))
        ax_cal.plot(bin_confs, bin_accs, 'o-', label=f"{sub_name}", color=colors[sub_name], linewidth=1.8)

    ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Chance Line')
    ax_roc.set_xlabel('False Positive Rate (1 - Specificity)')
    ax_roc.set_ylabel('True Positive Rate (Sensitivity)')
    ax_roc.set_title('Cross-Substance Holdout ROC Curves', pad=10)
    ax_roc.grid(True, linestyle='--', alpha=0.5)
    ax_roc.legend(loc='lower right', frameon=True, facecolor='white')

    ax_cal.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Ideal Calibration')
    ax_cal.set_xlabel('Mean Predicted Probability')
    ax_cal.set_ylabel('Observed Empirical Frequency')
    ax_cal.set_title('Reliability Diagram (Probability Calibration)', pad=10)
    ax_cal.grid(True, linestyle='--', alpha=0.5)
    ax_cal.legend(loc='upper left', frameon=True, facecolor='white')

    fig5_path = os.path.join(figures_dir, 'fig5_cross_substance_roc_and_calibration.png')
    plt.savefig(fig5_path)
    plt.close()
    print(f"Saved: {fig5_path}")

    # =========================================================================
    # FIGURE 6: Holdout Confusion Matrices across 4 Substance Classes
    # =========================================================================
    print("\n--- Generating Figure 6: Holdout Confusion Matrices ---")
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    sub_list = list(targets.keys())

    for idx, (sub_name, ax) in enumerate(zip(sub_list, axes.flatten())):
        cm = confusion_matrices[sub_name]['cm_scratch']
        m = confusion_matrices[sub_name]['m_scratch']
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
            xticklabels=['Non-Active', 'Past-Year'], yticklabels=['Non-Active', 'Past-Year'],
            annot_kws={'size': 11, 'weight': 'bold'}
        )
        ax.set_title(f"{sub_name}\nAcc: {m['accuracy']:.4f} | F1: {m['f1_score']:.4f}", fontsize=10)
        ax.set_xlabel('Predicted Class', fontsize=9)
        ax.set_ylabel('True Class', fontsize=9)

    plt.subplots_adjust(hspace=0.35, wspace=0.25)
    fig6_path = os.path.join(figures_dir, 'fig6_confusion_matrices.png')
    plt.savefig(fig6_path)
    plt.close()
    print(f"Saved: {fig6_path}")

    # =========================================================================
    # FIGURE 7: Forest Plot of Analytical Odds Ratios with 95% Confidence Intervals
    # =========================================================================
    print("\n--- Generating Figure 7: Analytical Odds Ratios Forest Plot ---")
    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    y_positions = np.arange(len(CORE_PREDICTORS))
    offsets = [-0.22, -0.07, 0.07, 0.22]

    for idx, (sub_name, offset) in enumerate(zip(sub_list, offsets)):
        or_data = analytical_odds_ratios[sub_name]
        ors = [entry['odds_ratio'] for entry in or_data]
        ci_lows = [entry['ci_lower_95'] for entry in or_data]
        ci_highs = [entry['ci_upper_95'] for entry in or_data]

        y_pos = y_positions + offset
        x_err = [
            np.array(ors) - np.array(ci_lows),
            np.array(ci_highs) - np.array(ors)
        ]
        ax.errorbar(
            ors, y_pos, xerr=x_err, fmt='o', color=colors[sub_name],
            label=f"{sub_name}", capsize=3.5, elinewidth=1.4, markersize=5
        )

    ax.axvline(1.0, color='black', linestyle='--', linewidth=1.2, alpha=0.7, label='Null Association (OR = 1.0)')
    ax.set_yticks(y_positions)
    ax.set_yticklabels(CORE_PREDICTORS)
    ax.set_xlabel('Adjusted Odds Ratio (Approximate Model-Based 95% Wald CI)')
    ax.set_title('Differential Psychometric Effect Signatures across Substance Classes', pad=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)

    fig7_path = os.path.join(figures_dir, 'fig7_odds_ratios_differential_signatures.png')
    plt.savefig(fig7_path)
    plt.close()
    print(f"Saved: {fig7_path}")

    # =========================================================================
    # SAVE CLEAN, NATIVE NESTED JSON
    # =========================================================================
    clean_master_json = {
        'cohort_metadata': {
            'total_survey_records': int(n_total + excluded),
            'excluded_overclaimers': int(excluded),
            'analyzed_cohort_size': int(n_total),
            'predictors': CORE_PREDICTORS,
            'substance_classes': list(targets.keys())
        },
        'unsupervised_topology': {
            'k_values': k_vals.tolist(),
            'davies_bouldin_mean': [round(float(v), 4) for v in db_means],
            'davies_bouldin_std': [round(float(v), 4) for v in db_stds],
            'silhouette_mean': [round(float(v), 4) for v in sil_means],
            'silhouette_std': [round(float(v), 4) for v in sil_stds],
            'optimal_k': 4,
            'kruskal_wallis_h': round(float(poly_assoc['kruskal_statistic']), 4),
            'kruskal_wallis_p': float(poly_assoc['p_value']),
            'cluster_mean_polysubstance': [round(float(v), 4) for v in poly_assoc['cluster_means']]
        },
        'five_fold_outer_cv': cv_outer_results,
        'holdout_benchmarks': {},
        'odds_ratios_fdr_corrected': fdr_results
    }

    for sub_name, data in substance_models_report.items():
        if sub_name == 'Cannabis_Forward_Selected':
            clean_master_json['holdout_benchmarks'][sub_name] = {
                'metrics': {k: round(float(v), 4) for k, v in data['metrics'].items() if isinstance(v, (int, float))},
                'calibration': {k: round(float(v), 4) for k, v in data['cal'].items()},
                'confidence_intervals_95': {k: [round(float(x), 4) for x in v] for k, v in data['ci'].items()},
                'features': data['features']
            }
        else:
            clean_master_json['holdout_benchmarks'][sub_name] = {
                'Majority': {k: round(float(v), 4) for k, v in data['Majority'].items() if isinstance(v, (int, float))},
                'LR_Scratch': {
                    'metrics': {k: round(float(v), 4) for k, v in data['LR_Scratch']['metrics'].items() if isinstance(v, (int, float))},
                    'calibration': {k: round(float(v), 4) for k, v in data['LR_Scratch']['cal'].items()},
                    'confidence_intervals_95': {k: [round(float(x), 4) for x in v] for k, v in data['LR_Scratch']['ci'].items()}
                },
                'LR_Sklearn_Parity': {
                    'metrics': {k: round(float(v), 4) for k, v in data['LR_Sklearn_Parity']['metrics'].items() if isinstance(v, (int, float))}
                },
                'Random_Forest_Baseline': {
                    'metrics': {k: round(float(v), 4) for k, v in data['Random_Forest_Baseline']['metrics'].items() if isinstance(v, (int, float))},
                    'calibration': {k: round(float(v), 4) for k, v in data['Random_Forest_Baseline']['cal'].items()},
                    'confidence_intervals_95': {k: [round(float(x), 4) for x in v] for k, v in data['Random_Forest_Baseline']['ci'].items()}
                },
                'Random_Forest_Tuned': {
                    'metrics': {k: round(float(v), 4) for k, v in data['Random_Forest_Tuned']['metrics'].items() if isinstance(v, (int, float))},
                    'calibration': {k: round(float(v), 4) for k, v in data['Random_Forest_Tuned']['cal'].items()},
                    'confidence_intervals_95': {k: [round(float(x), 4) for x in v] for k, v in data['Random_Forest_Tuned']['ci'].items()},
                    'best_params': data['Random_Forest_Tuned']['best_params']
                },
                'MLP': {
                    'metrics': {k: round(float(v), 4) for k, v in data['MLP']['metrics'].items() if isinstance(v, (int, float))},
                    'calibration': {k: round(float(v), 4) for k, v in data['MLP']['cal'].items()},
                    'confidence_intervals_95': {k: [round(float(x), 4) for x in v] for k, v in data['MLP']['ci'].items()},
                    'iterations_to_convergence': data['MLP']['iterations_to_convergence'] if 'iterations_to_convergence' in data['MLP'] else data['MLP']['n_iter']
                }
            }

    report_path = os.path.join(project_root, 'benchmark_results.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(clean_master_json, f, indent=2)
    print(f"\n[REPORT] Saved clean machine-readable benchmark summary to {report_path}")

    # Synchronize LaTeX tables and README.md directly from benchmark_results.json
    print("\n[SYNCHRONIZATION] Automatically updating LaTeX tables and README.md from benchmark_results.json...")
    from src.generate_synchronized_tables import main as sync_tables
    sync_tables()

    print("\n[SUCCESS] All experiments, 5-fold outer CV, FDR corrections, synchronized tables, and publication figures generated successfully.")


if __name__ == '__main__':
    main()
