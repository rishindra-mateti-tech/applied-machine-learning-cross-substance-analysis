# Applied Machine Learning for Cross-Substance Analysis: Psychometric and Demographic Correlates of Consumption Patterns

**Author:** Rishindra Mateti  
**Affiliation:** Department of Computer Science, Wright State University, Fairborn, OH, USA  
**Contact:** mateti.7@wright.edu | research@rishindramateti.com  
**Primary Manuscript:** [Applied_Machine_Learning_for_Cross_Substance_Analysis.pdf](Applied_Machine_Learning_for_Cross_Substance_Analysis.pdf) (10-Page Native IEEE Format)  

---

## Executive Abstract

This repository provides an end-to-end, reproducible empirical machine learning benchmark evaluating the relationship between psychometric traits and self-reported substance consumption on the benchmark UCI Drug Consumption dataset. Operating under rigorous data hygiene protocols, we systematically eliminate fictitious drug (Semeron) overclaimers ($N=8$), yielding an analytically verified cohort of $N = 1,877$ respondents characterized across 10 continuous demographic and psychometric dimensions (NEO-FFI-R Five-Factor Model, BIS-11 Impulsivity, and ImpSS Sensation Seeking).

The investigation establishes two decoupled yet mutually reinforcing machine learning paradigms:
1. **Unsupervised Behavioral Segmentation:** Formulates vector-optimized K-Means and soft Fuzzy C-Means (FCM) from mathematical first principles. Evaluates cluster separation across resolutions $k \in [2, 8]$ over 20 random restarts, identifying an optimal four-cluster topology (Davies-Bouldin index $= 1.0893 \pm 0.0686$, Silhouette coefficient $= 0.2896 \pm 0.0082$). We note that the modest silhouette score reflects continuous, overlapping density distributions rather than discrete biological taxa. Non-parametric Kruskal-Wallis testing demonstrates an exploratory within-sample rank-order association between cluster assignments and the gradient of illicit polysubstance co-involvement ($H = 554.46, p = 7.51 \times 10^{-120}$).
2. **Cross-Substance Supervised Tournament:** Benchmarks six predictive model configurations across four literature-grounded illicit substance classes representing past-year consumption: Cannabinoids, CNS Stimulants, Psychedelics, and Depressants/Anxiolytics. Models include a Majority Floor Baseline, Scratch Regularized Logistic Regression with analytical Hessian covariance estimation (95% Wald confidence intervals), Scikit-Learn parity verification, a Specified Random Forest baseline (100 trees, depth 8), an Inner-CV Tuned Random Forest (optimizing depth and split constraints via 3-fold inner CV), and an Early-Stopped Multi-Layer Perceptron (MLP) benchmark.
3. **Statistical Inference and Multiple Testing Control:** Evaluates generalization across both 5-fold outer cross-validation and frozen holdouts. Controls the False Discovery Rate across all 40 hypothesis tests using the Benjamini-Hochberg procedure ($q = 0.05$). Evaluates probability calibration through reliability curves, Brier scores, and Expected Calibration Error (ECE). Calibration differences are target-dependent: regularized linear models provide stable, monotonic calibration across all classes, while early stopping enables the neural benchmark to achieve comparable or slightly superior calibration on specific targets such as Psychedelics and Depressants.

---

## 5-Fold Outer Cross-Validation Generalization

To ensure performance metrics do not reflect an artifact of a single data split, we report the mean and standard deviation across 5 stratified outer cross-validation folds. Within each fold, feature scaling is strictly isolated to training folds:

| Substance Target | Model Architecture | Outer CV Accuracy | Outer CV F1-Score | Outer CV AUC-ROC | Outer CV PR-AUC | Outer CV Brier Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---:
| **Cannabis**<br>(Prevalence: 52.80%) | Scratch Regularized LogReg | **0.7922 +/- 0.0260** | **0.8025 +/- 0.0241** | **0.8710 +/- 0.0176** | **0.8872 +/- 0.0183** | **0.1457 +/- 0.0109** |
|  | Tuned Random Forest (Inner CV) | 0.7848 +/- 0.0213 | 0.7938 +/- 0.0208 | 0.8665 +/- 0.0187 | 0.8857 +/- 0.0205 | 0.1522 +/- 0.0082 |
|  | Early-Stopped MLP Benchmark | 0.7890 +/- 0.0205 | 0.8010 +/- 0.0170 | 0.8691 +/- 0.0197 | 0.8858 +/- 0.0225 | 0.1473 +/- 0.0126 |
| **CNS Stimulants**<br>(Prevalence: 32.13%) | Scratch Regularized LogReg | **0.7406 +/- 0.0094** | **0.5527 +/- 0.0218** | **0.7911 +/- 0.0260** | **0.6176 +/- 0.0438** | **0.1698 +/- 0.0093** |
|  | Tuned Random Forest (Inner CV) | 0.7400 +/- 0.0091 | 0.5179 +/- 0.0285 | 0.7905 +/- 0.0273 | 0.6127 +/- 0.0521 | 0.1714 +/- 0.0072 |
|  | Early-Stopped MLP Benchmark | 0.7374 +/- 0.0124 | 0.5302 +/- 0.0277 | 0.7836 +/- 0.0228 | 0.6091 +/- 0.0387 | 0.1740 +/- 0.0086 |
| **Psychedelics**<br>(Prevalence: 27.76%) | Scratch Regularized LogReg | **0.8125 +/- 0.0203** | 0.6390 +/- 0.0479 | **0.8648 +/- 0.0217** | **0.6734 +/- 0.0423** | **0.1306 +/- 0.0093** |
|  | Tuned Random Forest (Inner CV) | 0.7933 +/- 0.0156 | 0.5703 +/- 0.0367 | 0.8569 +/- 0.0197 | 0.6598 +/- 0.0516 | 0.1363 +/- 0.0063 |
|  | Early-Stopped MLP Benchmark | 0.8119 +/- 0.0145 | **0.6497 +/- 0.0306** | 0.8624 +/- 0.0167 | 0.6621 +/- 0.0544 | 0.1316 +/- 0.0091 |
| **Depressants**<br>(Prevalence: 28.34%) | Scratch Regularized LogReg | 0.7294 +/- 0.0186 | **0.3590 +/- 0.0556** | **0.7396 +/- 0.0183** | 0.5043 +/- 0.0507 | **0.1752 +/- 0.0061** |
|  | Tuned Random Forest (Inner CV) | **0.7320 +/- 0.0171** | 0.2713 +/- 0.0328 | 0.7336 +/- 0.0312 | **0.5111 +/- 0.0490** | 0.1769 +/- 0.0070 |
|  | Early-Stopped MLP Benchmark | 0.7198 +/- 0.0040 | 0.2525 +/- 0.1057 | 0.6952 +/- 0.0538 | 0.4515 +/- 0.0389 | 0.1891 +/- 0.0150 |

---

## Frozen Stratified Holdout Benchmark Results

All metrics below reflect evaluation on frozen holdout test partitions ($N_{\text{test}} = 374\text{ to }375$, representing 20% stratified holdouts). Metrics match `benchmark_results.json` and the publication manuscript exactly:

### 1. Cannabinoids (Cannabis Past-Year Consumption, Prevalence: 52.80%)
| Model Architecture | Test Accuracy | Precision | Recall | Specificity | F1-Score | AUC-ROC (95% CI) | PR-AUC | Brier Score | ECE | Implementation Origin |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Majority Floor Baseline** | 0.5280 | 0.5280 | 1.0000 | 0.0000 | 0.6911 | 0.5000 | 0.5280 | 0.2492 | 0.0000 | Trivial Baseline |
| **Scratch Regularized LogReg** | 0.7600 | 0.8103 | **0.7121** | 0.8136 | 0.7581 | **0.8527 [0.8126, 0.8882]** | 0.8710 | 0.1584 | 0.0401 | First Principles (NumPy) |
| **Sklearn LogReg (Parity Check)** | 0.7600 | 0.8103 | **0.7121** | 0.8136 | 0.7581 | **0.8528** | **0.8713** | **0.1121** | **0.0000** | Scikit-Learn Parity Check |
| **Specified RF Baseline (Depth 8)** | 0.7520 | 0.8000 | 0.7071 | 0.8023 | 0.7507 | 0.8480 [0.8100, 0.8851] | 0.8614 | 0.1602 | 0.0401 | Specified Baseline |
| **Tuned RF (Inner 3-Fold CV)** | 0.7227 | 0.7640 | 0.6869 | 0.7627 | 0.7234 | 0.8381 [0.7965, 0.8754] | 0.8577 | 0.1675 | 0.0878 | Tuned Inner Grid Search |
| **Early-Stopped MLP Benchmark** | **0.7653** | **0.8274** | 0.7020 | **0.8362** | **0.7596** | 0.8519 [0.8125, 0.8878] | 0.8624 | 0.1579 | 0.0411 | Early Stopping |
| **Forward Selection LR (9 Feat.)** | 0.7493 | 0.7989 | 0.7020 | 0.8023 | 0.7473 | 0.8433 [0.8029, 0.8809] | 0.8614 | 0.1637 | 0.0452 | 5-Fold CV on $X_{\text{train}}$ |

### 2. CNS Stimulants (Cocaine / Amphetamines Past-Year Consumption, Prevalence: 32.13%)
| Model Architecture | Test Accuracy | Precision | Recall | Specificity | F1-Score | AUC-ROC (95% CI) | PR-AUC | Brier Score | ECE | Implementation Origin |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Majority Floor Baseline** | 0.6791 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.5000 | 0.3209 | 0.2179 | 0.0000 | Trivial Baseline |
| **Scratch Regularized LogReg** | 0.7433 | 0.6277 | **0.4917** | 0.8622 | **0.5514** | **0.7979 [0.7475, 0.8422]** | 0.5990 | 0.1687 | 0.0557 | First Principles (NumPy) |
| **Sklearn LogReg (Parity Check)** | 0.7433 | 0.6277 | **0.4917** | 0.8622 | **0.5514** | **0.7979** | 0.5992 | 0.2402 | **0.0000** | Scikit-Learn Parity Check |
| **Specified RF Baseline (Depth 8)** | 0.7380 | 0.6222 | 0.4667 | 0.8661 | 0.5333 | 0.7962 [0.7455, 0.8424] | **0.6106** | **0.1673** | 0.0355 | Specified Baseline |
| **Tuned RF (Inner 3-Fold CV)** | **0.7460** | **0.6582** | 0.4333 | 0.8937 | 0.5226 | 0.7952 [0.7468, 0.8401] | 0.5986 | 0.1702 | 0.0449 | Tuned Inner Grid Search |
| **Early-Stopped MLP Benchmark** | 0.7299 | 0.6338 | 0.3750 | **0.8976** | 0.4712 | 0.7879 [0.7372, 0.8335] | 0.6002 | 0.1719 | 0.0529 | Early Stopping |

### 3. Psychedelics (Psilocybin Mushrooms / LSD Past-Year Consumption, Prevalence: 27.76%)
| Model Architecture | Test Accuracy | Precision | Recall | Specificity | F1-Score | AUC-ROC (95% CI) | PR-AUC | Brier Score | ECE | Implementation Origin |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Majority Floor Baseline** | 0.7227 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.5000 | 0.2773 | 0.2004 | 0.0000 | Trivial Baseline |
| **Scratch Regularized LogReg** | **0.7893** | **0.6404** | **0.5481** | 0.8819 | **0.5907** | 0.8317 [0.7892, 0.8687] | 0.5902 | 0.1509 | 0.0578 | First Principles (NumPy) |
| **Sklearn LogReg (Parity Check)** | 0.7867 | 0.6333 | **0.5481** | 0.8782 | 0.5876 | 0.8318 | 0.5900 | 0.2419 | **0.0000** | Scikit-Learn Parity Check |
| **Specified RF Baseline (Depth 8)** | 0.7760 | 0.6111 | 0.5288 | 0.8708 | 0.5670 | 0.8258 [0.7793, 0.8634] | 0.6006 | 0.1492 | 0.0535 | Specified Baseline |
| **Tuned RF (Inner 3-Fold CV)** | 0.7733 | 0.6377 | 0.4231 | **0.9077** | 0.5087 | 0.8226 [0.7746, 0.8616] | 0.5965 | 0.1486 | 0.0352 | Tuned Inner Grid Search |
| **Early-Stopped MLP Benchmark** | 0.7840 | 0.6322 | 0.5288 | 0.8819 | 0.5759 | **0.8470 [0.8033, 0.8833]** | **0.6214** | **0.1421** | 0.0570 | Early Stopping |

### 4. Depressants / Anxiolytics (Benzodiazepines Past-Year Consumption, Prevalence: 28.34%)
| Model Architecture | Test Accuracy | Precision | Recall | Specificity | F1-Score | AUC-ROC (95% CI) | PR-AUC | Brier Score | ECE | Implementation Origin |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Majority Floor Baseline** | 0.7173 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.5000 | 0.2827 | 0.2028 | 0.0000 | Trivial Baseline |
| **Scratch Regularized LogReg** | 0.7093 | 0.4667 | 0.1981 | 0.9108 | 0.2781 | 0.7326 [0.6769, 0.7875] | 0.4590 | 0.1780 | 0.0692 | First Principles (NumPy) |
| **Sklearn LogReg (Parity Check)** | 0.7120 | 0.4773 | 0.1981 | 0.9145 | 0.2800 | 0.7313 | 0.4551 | 0.2480 | **0.0000** | Scikit-Learn Parity Check |
| **Specified RF Baseline (Depth 8)** | **0.7307** | **0.5641** | **0.2075** | 0.9368 | **0.3034** | 0.7289 [0.6743, 0.7853] | **0.4896** | **0.1767** | 0.0341 | Specified Baseline |
| **Tuned RF (Inner 3-Fold CV)** | 0.7227 | 0.5500 | 0.1038 | **0.9665** | 0.1746 | 0.7300 [0.6741, 0.7827] | 0.4835 | 0.1775 | 0.0563 | Tuned Inner Grid Search |
| **Early-Stopped MLP Benchmark** | 0.7120 | 0.4737 | 0.1698 | 0.9257 | 0.2500 | **0.7354 [0.6800, 0.7880]** | 0.4735 | 0.1775 | 0.0499 | Early Stopping |

---

## Analytical Odds Ratios and Benjamini-Hochberg FDR Control

Adjusted odds ratios are accompanied by analytical 95% Wald confidence intervals derived from the inverted Hessian covariance matrix $\Sigma = (X_b^T W X_b + \lambda I_{\text{reg}})^{-1}$. To safeguard against false discovery across all 40 hypothesis tests, the Benjamini-Hochberg procedure was applied at false discovery rate $q = 0.05$. Significant associations satisfying $q < 0.05$ are marked with an asterisk (*):

| Predictor Feature | Cannabis OR [95% CI] | Stimulants OR [95% CI] | Psychedelics OR [95% CI] | Depressants OR [95% CI] | Empirical Effect Pattern |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Age** | **0.422 [0.363, 0.491]\*** | **0.539 [0.466, 0.624]\*** | **0.369 [0.308, 0.443]\*** | **0.828 [0.723, 0.948]\*** | Consistent Negative Association with Age across All Targets |
| **Gender (Female)** | **0.683 [0.591, 0.789]\*** | **0.778 [0.680, 0.890]\*** | **0.631 [0.539, 0.739]\*** | **0.839 [0.734, 0.958]\*** | Lower Past-Year Prevalence in Females (except Depressants) |
| **Education** | **0.619 [0.536, 0.715]\*** | 0.930 [0.810, 1.067] | **0.753 [0.636, 0.891]\*** | **0.854 [0.747, 0.977]\*** | Negative Association with Higher Education for Cannabis |
| **Neuroticism (Nscore)** | **0.830 [0.701, 0.982]\*** | 1.152 [0.992, 1.336] | 0.837 [0.703, 0.996] | **1.574 [1.351, 1.834]\*** | Strongest Correlate for Depressants / Anxiolytics |
| **Impulsivity (BIS-11)** | 0.972 [0.812, 1.164] | 1.191 [1.009, 1.405] | 1.035 [0.851, 1.258] | 1.040 [0.883, 1.224] | Attenuated when Controlling for Sensation Seeking |
| **Agreeableness (Ascore)** | 0.986 [0.851, 1.143] | **0.829 [0.726, 0.948]\*** | 1.017 [0.872, 1.186] | **0.842 [0.738, 0.960]\*** | Weak Association across Target Classes |
| **Extraversion (Escore)** | **0.722 [0.606, 0.859]\*** | 1.028 [0.886, 1.193] | **0.786 [0.663, 0.931]\*** | 1.019 [0.880, 1.179] | Negative Association for Cannabis |
| **Openness (Oscore)** | **2.495 [2.108, 2.953]\*** | **1.267 [1.100, 1.460]\*** | **2.337 [1.953, 2.797]\*** | **1.369 [1.187, 1.578]\*** | Strongest Correlate for Hallucinogens and Cannabis |
| **Conscientiousness (Cscore)** | **0.776 [0.656, 0.917]\*** | **0.796 [0.686, 0.924]\*** | 1.068 [0.900, 1.267] | 0.936 [0.808, 1.083] | Consistent Negative Association across Substance Targets |
| **Sensation Seeking (SS)** | **1.808 [1.484, 2.202]\*** | **1.413 [1.186, 1.682]\*** | **1.703 [1.380, 2.101]\*** | **1.345 [1.128, 1.603]\*** | Universal Positive Correlate across All Substance Classes |

---

## Methodological Safeguards (Zero Data Leakage Protocols)

1. **Rigorous Cohort Hygiene:** Excludes respondents reporting consumption of the non-existent drug Semeron ($N=8$), avoiding noise injection from inattentive or unreliable survey respondents.
2. **Strict Scaling Isolation:** Z-score standardization parameters ($\mu, \sigma$) are computed strictly on $X_{\text{train}}$ via `LeakageFreeStandardScaler` and applied to $X_{\text{test}}$.
3. **Inner-Fold Scaling in Feature Selection:** Greedy forward stepwise selection applies fold-specific scaling within each internal training fold of the 5-fold cross-validation loop.
4. **Nested Grid Search for Ensembles:** The Random Forest is tuned via an inner 3-fold cross-validation grid search exclusively on training folds, avoiding test-set hyperparameter tuning.
5. **Outer 5-Fold Stratified Cross-Validation:** In addition to frozen holdout evaluation, models are benchmarked across 5 outer folds to confirm stability.
6. **Convergence Verification:** Multi-Layer Perceptron models use early stopping on an internal validation fraction (15%) with a high iteration ceiling (1,500 iterations), resolving non-convergence warnings.
7. **Multiple Testing Correction:** Significance testing across 40 hypothesis tests is adjusted using the Benjamini-Hochberg procedure at false discovery rate $q = 0.05$.
8. **Statistical Uncertainty:** Computes 1,000 bootstrap resamples on the holdout test set to estimate non-parametric 95% empirical confidence intervals.
9. **Non-Causal Academic Framing:** All analyses are presented strictly as empirical statistical associations on self-reported observational survey data.

---

## Repository Structure

```text
├── Applied_Machine_Learning_for_Cross_Substance_Analysis.tex  # 10-Page IEEE LaTeX Manuscript (Native)
├── Applied_Machine_Learning_for_Cross_Substance_Analysis.pdf  # Compiled Native IEEE Publication PDF (10 Pages)
├── IEEEtran.cls                                                   # Official IEEE Document Class
├── README.md                                                      # Benchmark Documentation & Results
├── requirements.txt                                               # Pinned Reproducible Environment Dependencies
├── run_benchmark_experiments.py                                   # Master Cross-Substance Benchmark Pipeline
├── run_experiments.py                                             # Compatibility Alias
├── benchmark_results.json                                         # Exact Serialized Empirical Metrics
├── data/
│   └── drug_consumption.data                                     # UCI Machine Learning Repository Data
├── figures/
│   ├── fig1_correlation_matrix.png                               # Inter-Trait Pearson Correlation Heatmap
│   ├── fig2_clustering_3d_comparison.png                         # 3D Latent Cluster Space (K-Means vs FCM)
│   ├── fig3_topology_and_silhouette.png                          # Dual Davies-Bouldin & Silhouette Stability Curves
│   ├── fig4_cost_convergence.png                                 # Loss J(theta) Descent Profiles across Targets
│   ├── fig5_cross_substance_roc_and_calibration.png              # Cross-Substance ROC & Reliability Diagrams
│   ├── fig6_confusion_matrices.png                               # 2x2 Grid of Holdout Confusion Matrices
│   └── fig7_odds_ratios_differential_signatures.png              # Forest Plot of Odds Ratios with 95% Wald CIs
└── src/
    ├── __init__.py                                                # Package Initialization & Unified Exports
    ├── dataset_pipeline.py                                        # Leakage-Free Loader, Scaler & Substance Definitions
    ├── unsupervised_segmentation.py                              # K-Means, FCM, Stability Sweeps & Kruskal-Wallis
    ├── supervised_classification.py                              # Scratch LogReg, Hessian Odds Ratios, RF, & MLP
    ├── feature_selection.py                                      # 5-Fold Cross-Validated Forward Selection
    └── evaluation_metrics.py                                      # Metrics, ROC/AUC, PR-AUC, & Bootstrap CIs
```

---

## Quick Start and Reproduction

### 1. Environment Installation
```bash
git clone https://github.com/rishindra-mateti-tech/Psychometric-Substance-Vulnerability-Analysis-Clustering-Classification.git
cd Psychometric-Substance-Vulnerability-Analysis-Clustering-Classification
pip install -r requirements.txt
```
*Tested with Python 3.10 to 3.13; minimum dependency requirements: `numpy>=1.24.0`, `pandas>=2.0.0`, `scipy>=1.10.0`, `scikit-learn>=1.3.0`, `matplotlib>=3.7.0`, `seaborn>=0.12.0`.*

### 2. Execute Complete Benchmark Suite (Single Command)
```bash
python run_benchmark_experiments.py
```
Or run via the compatibility alias:
```bash
python run_experiments.py
```

### 3. Compile Native IEEE Research Paper
```bash
pdflatex -interaction=nonstopmode -disable-installer Applied_Machine_Learning_for_Cross_Substance_Analysis.tex
pdflatex -interaction=nonstopmode -disable-installer Applied_Machine_Learning_for_Cross_Substance_Analysis.tex
```

---

## Citation and Attribution

```bibtex
@techreport{mateti2026applied,
  author      = {Rishindra Mateti},
  title       = {Applied Machine Learning for Cross-Substance Analysis: Psychometric and Demographic Correlates of Consumption Patterns},
  institution = {Department of Computer Science, Wright State University},
  year        = {2026},
  type        = {Technical Report},
  url         = {https://github.com/rishindra-mateti-tech/Psychometric-Substance-Vulnerability-Analysis-Clustering-Classification}
},
  title       = {Applied Machine Learning for Cross-Substance Analysis: Psychometric and Demographic Correlates of Consumption Patterns},
  institution = {Department of Computer Science, Wright State University},
  year        = {2026},
  type        = {Technical Report},
  url         = {https://github.com/rishindra-mateti-tech/Psychometric-Substance-Vulnerability-Analysis-Clustering-Classification}
}
```
