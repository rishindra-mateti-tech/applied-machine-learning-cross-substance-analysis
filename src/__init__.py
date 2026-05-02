"""
Differential Psychometric Substance Consumption Research Package
Author: Rishindra Mateti (Wright State University)
"""

from src.dataset_pipeline import (
    load_cleaned_dataset,
    get_stratified_outer_split,
    LeakageFreeStandardScaler,
    CORE_PREDICTORS,
    PSYCHOMETRIC_TRAITS,
    DEMOGRAPHIC_FEATURES,
    CLUSTERING_FEATURES,
)

from src.unsupervised_segmentation import (
    KMeansScratch,
    FuzzyCMeansScratch,
    compute_davies_bouldin_index,
    evaluate_cluster_topology_stability,
    evaluate_cluster_polysubstance_association,
)

from src.supervised_classification import (
    RegularizedLogisticRegressionScratch,
    train_sklearn_logistic_regression,
    train_random_forest_baseline,
    train_random_forest_tuned,
    train_mlp_benchmark,
    compute_calibration_metrics,
    apply_benjamini_hochberg,
)

from src.feature_selection import (
    kfold_cross_validate_logistic,
    leakage_free_forward_selection,
)

from src.evaluation_metrics import (
    compute_confusion_matrix,
    compute_classification_metrics,
    compute_roc_auc,
    compute_pr_auc,
    compute_bootstrap_confidence_intervals,
)
