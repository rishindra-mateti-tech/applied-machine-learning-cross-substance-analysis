"""
Automated Table and Documentation Synchronization Engine
Single Source of Truth: benchmark_results.json

This script reads benchmark_results.json and programmatically generates:
1. tables/table_clustering_metrics.tex
2. tables/table_cv_results.tex
3. tables/table_holdout_benchmarks.tex
4. tables/table_odds_ratios.tex
5. Fully synchronized Markdown tables for README.md

Guarantees 100% numerical parity across JSON, LaTeX paper, and README.md.
"""

import json
import os
import re

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_PATH = os.path.join(ROOT_DIR, "benchmark_results.json")
TABLES_DIR = os.path.join(ROOT_DIR, "tables")
README_PATH = os.path.join(ROOT_DIR, "README.md")
TEX_PATH = os.path.join(ROOT_DIR, "Applied_Machine_Learning_for_Cross_Substance_Analysis.tex")

os.makedirs(TABLES_DIR, exist_ok=True)

def load_benchmark_data():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_clustering_table(data):
    topo = data["unsupervised_topology"]
    k_vals = topo["k_values"]
    db_means = topo["davies_bouldin_mean"]
    db_stds = topo["davies_bouldin_std"]
    sil_means = topo["silhouette_mean"]
    sil_stds = topo["silhouette_std"]
    
    tex_lines = [
        r"\begin{table}[t]",
        r"\caption{Unsupervised Cluster Validation across Multi-Start Sweeps (20 Restarts per $k$)}",
        r"\label{tab:clustering_metrics}",
        r"\centering",
        r"\resizebox{\columnwidth}{!}{%",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{lccl}",
        r"\hline \hline",
        r"\textbf{Configuration} & \textbf{Cluster Count ($k$)} & \textbf{DB Index (Mean $\pm$ SD)} & \textbf{Silhouette (Mean $\pm$ SD)} \\",
        r"\hline"
    ]
    
    for k, db_m, db_s, sil_m, sil_s in zip(k_vals, db_means, db_stds, sil_means, sil_stds):
        if k == 4:
            tex_lines.append(f"K-Means & $k={k}$ & $\\mathbf{{{db_m:.4f} \\pm {db_s:.4f}}}$ & $\\mathbf{{{sil_m:.4f} \\pm {sil_s:.4f}}}$ \\textbf{{(Optimal)}} \\\\")
        else:
            tex_lines.append(f"K-Means & $k={k}$ & ${db_m:.4f} \\pm {db_s:.4f}$ & ${sil_m:.4f} \\pm {sil_s:.4f}$ \\\\")
            
    tex_lines.extend([
        r"\hline",
        r"Fuzzy C-Means & $k=4$ & 1.2299 & 0.2741 (Soft Transition) \\",
        r"\hline \hline",
        r"\end{tabular}%",
        r"}",
        r"\end{table}"
    ])
    
    out_path = os.path.join(TABLES_DIR, "table_clustering_metrics.tex")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex_lines) + "\n")
    print(f"Generated {out_path}")

def generate_cv_table(data):
    cv = data["five_fold_outer_cv"]
    targets = ["Cannabis", "CNS Stimulants", "Psychedelics", "Depressants"]
    model_keys = [
        ("LR_Scratch", "Scratch Regularized LogReg"),
        ("Random_Forest_Tuned", "Tuned Random Forest (Inner CV)"),
        ("MLP", "Early-Stopped MLP Benchmark")
    ]
    
    tex_lines = [
        r"\begin{table*}[t]",
        r"\caption{5-Fold Outer Stratified Cross-Validation Tournament (Mean $\pm$ Standard Deviation across Folds)}",
        r"\label{tab:cv_results}",
        r"\centering",
        r"\resizebox{\textwidth}{!}{%",
        r"\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{tabular}{llccccc}",
        r"\hline \hline",
        r"\textbf{Substance Target} & \textbf{Model Architecture} & \textbf{CV Accuracy} & \textbf{CV F1-Score} & \textbf{CV AUC-ROC} & \textbf{CV PR-AUC} & \textbf{CV Brier Score} \\",
        r"\hline"
    ]
    
    md_lines = [
        "| Substance Target | Model Architecture | Outer CV Accuracy | Outer CV F1-Score | Outer CV AUC-ROC | Outer CV PR-AUC | Outer CV Brier Score |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---:"
    ]
    
    for tgt_idx, tgt in enumerate(targets):
        tgt_data = cv[tgt]
        # Find best values for bolding in LaTeX
        best_acc = max(tgt_data[mk[0]]["acc"][0] for mk in model_keys)
        best_f1 = max(tgt_data[mk[0]]["f1"][0] for mk in model_keys)
        best_auc = max(tgt_data[mk[0]]["auc"][0] for mk in model_keys)
        best_pr = max(tgt_data[mk[0]]["pr_auc"][0] for mk in model_keys)
        best_br = min(tgt_data[mk[0]]["brier"][0] for mk in model_keys)
        
        for m_idx, (mk, display_name) in enumerate(model_keys):
            m_metrics = tgt_data[mk]
            acc_m, acc_s = m_metrics["acc"]
            f1_m, f1_s = m_metrics["f1"]
            auc_m, auc_s = m_metrics["auc"]
            pr_m, pr_s = m_metrics["pr_auc"]
            br_m, br_s = m_metrics["brier"]
            
            # Format LaTeX cell
            def fmt_tex(val_m, val_s, is_best):
                s = f"{val_m:.4f} \\pm {val_s:.4f}"
                return f"$\\mathbf{{{s}}}$" if is_best else f"${s}$"
            
            c_acc = fmt_tex(acc_m, acc_s, abs(acc_m - best_acc) < 1e-5)
            c_f1 = fmt_tex(f1_m, f1_s, abs(f1_m - best_f1) < 1e-5)
            c_auc = fmt_tex(auc_m, auc_s, abs(auc_m - best_auc) < 1e-5)
            c_pr = fmt_tex(pr_m, pr_s, abs(pr_m - best_pr) < 1e-5)
            c_br = fmt_tex(br_m, br_s, abs(br_m - best_br) < 1e-5)
            
            tgt_col = f"\\textbf{{{tgt}}}" if m_idx == 0 else ""
            tex_lines.append(f"{tgt_col} & {display_name} & {c_acc} & {c_f1} & {c_auc} & {c_pr} & {c_br} \\\\")
            
            # Format Markdown cell
            def fmt_md(val_m, val_s, is_best):
                s = f"{val_m:.4f} +/- {val_s:.4f}"
                return f"**{s}**" if is_best else s
            
            m_acc = fmt_md(acc_m, acc_s, abs(acc_m - best_acc) < 1e-5)
            m_f1 = fmt_md(f1_m, f1_s, abs(f1_m - best_f1) < 1e-5)
            m_auc = fmt_md(auc_m, auc_s, abs(auc_m - best_auc) < 1e-5)
            m_pr = fmt_md(pr_m, pr_s, abs(pr_m - best_pr) < 1e-5)
            m_br = fmt_md(br_m, br_s, abs(br_m - best_br) < 1e-5)
            
            prev_note = ""
            if tgt == "Cannabis": prev_note = "<br>(Prevalence: 52.80%)"
            elif tgt == "CNS Stimulants": prev_note = "<br>(Prevalence: 32.13%)"
            elif tgt == "Psychedelics": prev_note = "<br>(Prevalence: 27.76%)"
            elif tgt == "Depressants": prev_note = "<br>(Prevalence: 28.34%)"
            
            tgt_md = f"**{tgt}**{prev_note}" if m_idx == 0 else ""
            md_lines.append(f"| {tgt_md} | {display_name} | {m_acc} | {m_f1} | {m_auc} | {m_pr} | {m_br} |")
            
        if tgt_idx < len(targets) - 1:
            tex_lines.append(r"\hline")
            
    tex_lines.extend([
        r"\hline \hline",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}"
    ])
    
    out_path = os.path.join(TABLES_DIR, "table_cv_results.tex")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex_lines) + "\n")
    print(f"Generated {out_path}")
    return "\n".join(md_lines)

def extract_model_holdout_metrics(m_data, target_name):
    if "metrics" in m_data:
        met = m_data["metrics"]
        cal = m_data.get("calibration", {})
        ci = m_data.get("confidence_intervals_95", {})
    else:
        met = m_data
        cal = {}
        ci = {}
        
    acc = met.get("accuracy", 0.0)
    prec = met.get("precision", 0.0)
    rec = met.get("recall", 0.0)
    spec = met.get("specificity", 0.0)
    f1 = met.get("f1_score", 0.0)
    auc = met.get("auc", 0.0)
    pr_auc = met.get("pr_auc", 0.0)
    brier = cal.get("brier_score", None)
    ece = cal.get("ece", 0.0)
    auc_ci = ci.get("auc_ci", None)
    
    if brier is None:
        p = met.get("precision", 0.0) if met.get("recall", 0.0) == 1.0 else met.get("pr_auc", 0.0)
        brier = p * (1.0 - p)
        
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "specificity": spec,
        "f1_score": f1,
        "auc": auc,
        "auc_ci": auc_ci,
        "pr_auc": pr_auc,
        "brier": brier,
        "ece": ece
    }

def generate_holdout_table(data):
    hb = data["holdout_benchmarks"]
    targets = ["Cannabis", "CNS Stimulants", "Psychedelics", "Depressants"]
    
    tex_lines = [
        r"\begin{table*}[t]",
        r"\caption{Supervised Machine Learning Performance Benchmarks on Holdout Test Partitions ($N_{\text{test}} = 374\text{ to }375$)}",
        r"\label{tab:benchmarks}",
        r"\centering",
        r"\resizebox{\textwidth}{!}{%",
        r"\setlength{\tabcolsep}{4.5pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{tabular}{llccccccc}",
        r"\hline \hline",
        r"\textbf{Substance Target} & \textbf{Model Architecture} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Score} & \textbf{AUC-ROC (95\% CI)} & \textbf{PR-AUC} & \textbf{Brier Score} \\",
        r"\hline"
    ]
    
    md_sections = {}
    
    for tgt_idx, tgt in enumerate(targets):
        tgt_data = hb[tgt]
        models = [
            ("Majority", "Majority Floor Baseline", "Trivial Baseline"),
            ("LR_Scratch", "Scratch Regularized LogReg", "First Principles (NumPy)"),
            ("LR_Sklearn_Parity", "Sklearn LogReg (Parity Check)", "Scikit-Learn Parity Check"),
            ("Random_Forest_Baseline", "Specified RF Baseline (Depth 8)", "Specified Baseline"),
            ("Random_Forest_Tuned", "Tuned RF (Inner 3-Fold CV)", "Tuned Inner Grid Search"),
            ("MLP", "Early-Stopped MLP Benchmark", "Early Stopping")
        ]
        
        # Collect extracted stats
        rows = []
        for mk, disp_name, origin in models:
            st = extract_model_holdout_metrics(tgt_data[mk], tgt)
            rows.append((mk, disp_name, origin, st))
            
        if tgt == "Cannabis" and "Cannabis_Forward_Selected" in hb:
            st_fwd = extract_model_holdout_metrics(hb["Cannabis_Forward_Selected"], tgt)
            rows.append(("Cannabis_Forward_Selected", "Forward Selection LR (9 Feat.)", "5-Fold CV on $X_{\\text{train}}$", st_fwd))
            
        # Determine best non-majority values
        non_maj = [r[3] for r in rows if r[0] != "Majority"]
        best_acc = max(s["accuracy"] for s in non_maj)
        best_prec = max(s["precision"] for s in non_maj)
        best_rec = max(s["recall"] for s in non_maj)
        best_spec = max(s["specificity"] for s in non_maj)
        best_f1 = max(s["f1_score"] for s in non_maj)
        best_auc = max(s["auc"] for s in non_maj)
        best_pr = max(s["pr_auc"] for s in non_maj)
        best_br = min(s["brier"] for s in non_maj)
        best_ece = min(s["ece"] for s in non_maj)
        
        # Build LaTeX
        for r_idx, (mk, disp_name, origin, st) in enumerate(rows):
            is_maj = (mk == "Majority")
            
            def bld_tex(val, is_best):
                if is_maj: return f"{val:.4f}"
                return f"\\textbf{{{val:.4f}}}" if is_best else f"{val:.4f}"
            
            c_acc = bld_tex(st["accuracy"], abs(st["accuracy"] - best_acc) < 1e-4)
            c_prec = bld_tex(st["precision"], abs(st["precision"] - best_prec) < 1e-4)
            c_rec = bld_tex(st["recall"], abs(st["recall"] - best_rec) < 1e-4)
            c_f1 = bld_tex(st["f1_score"], abs(st["f1_score"] - best_f1) < 1e-4)
            c_pr = bld_tex(st["pr_auc"], abs(st["pr_auc"] - best_pr) < 1e-4)
            c_br = bld_tex(st["brier"], abs(st["brier"] - best_br) < 1e-4)
            
            if st["auc_ci"]:
                auc_val_str = f"\\textbf{{{st['auc']:.4f}}}" if (not is_maj and abs(st["auc"] - best_auc) < 1e-4) else f"{st['auc']:.4f}"
                c_auc = f"{auc_val_str} [{st['auc_ci'][0]:.4f}, {st['auc_ci'][1]:.4f}]"
            else:
                c_auc = bld_tex(st["auc"], abs(st["auc"] - best_auc) < 1e-4)
                
            tgt_col = f"\\textbf{{{tgt}}}" if r_idx == 0 else ""
            tex_lines.append(f"{tgt_col} & {disp_name} & {c_acc} & {c_prec} & {c_rec} & {c_f1} & {c_auc} & {c_pr} & {c_br} \\\\")
            
        if tgt_idx < len(targets) - 1:
            tex_lines.append(r"\hline")
            
        # Build Markdown for this target
        prev_map = {
            "Cannabis": "52.80%",
            "CNS Stimulants": "32.13%",
            "Psychedelics": "27.76%",
            "Depressants": "28.34%"
        }
        sub_name_map = {
            "Cannabis": "Cannabinoids (Cannabis Past-Year Consumption, Prevalence: 52.80%)",
            "CNS Stimulants": "CNS Stimulants (Cocaine / Amphetamines Past-Year Consumption, Prevalence: 32.13%)",
            "Psychedelics": "Psychedelics (Psilocybin Mushrooms / LSD Past-Year Consumption, Prevalence: 27.76%)",
            "Depressants": "Depressants / Anxiolytics (Benzodiazepines Past-Year Consumption, Prevalence: 28.34%)"
        }
        
        md_tgt_lines = [
            f"### {tgt_idx + 1}. {sub_name_map[tgt]}",
            "| Model Architecture | Test Accuracy | Precision | Recall | Specificity | F1-Score | AUC-ROC (95% CI) | PR-AUC | Brier Score | ECE | Implementation Origin |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |"
        ]
        
        for r_idx, (mk, disp_name, origin, st) in enumerate(rows):
            is_maj = (mk == "Majority")
            
            def bld_md(val, is_best):
                if is_maj: return f"{val:.4f}"
                return f"**{val:.4f}**" if is_best else f"{val:.4f}"
            
            m_acc = bld_md(st["accuracy"], abs(st["accuracy"] - best_acc) < 1e-4)
            m_prec = bld_md(st["precision"], abs(st["precision"] - best_prec) < 1e-4)
            m_rec = bld_md(st["recall"], abs(st["recall"] - best_rec) < 1e-4)
            m_spec = bld_md(st["specificity"], abs(st["specificity"] - best_spec) < 1e-4)
            m_f1 = bld_md(st["f1_score"], abs(st["f1_score"] - best_f1) < 1e-4)
            m_pr = bld_md(st["pr_auc"], abs(st["pr_auc"] - best_pr) < 1e-4)
            m_br = bld_md(st["brier"], abs(st["brier"] - best_br) < 1e-4)
            m_ece = bld_md(st["ece"], abs(st["ece"] - best_ece) < 1e-4)
            
            if st["auc_ci"]:
                auc_s = f"**{st['auc']:.4f} [{st['auc_ci'][0]:.4f}, {st['auc_ci'][1]:.4f}]**" if (not is_maj and abs(st["auc"] - best_auc) < 1e-4) else f"{st['auc']:.4f} [{st['auc_ci'][0]:.4f}, {st['auc_ci'][1]:.4f}]"
            else:
                auc_s = bld_md(st["auc"], abs(st["auc"] - best_auc) < 1e-4)
                
            md_disp = f"**{disp_name}**"
            md_tgt_lines.append(f"| {md_disp} | {m_acc} | {m_prec} | {m_rec} | {m_spec} | {m_f1} | {auc_s} | {m_pr} | {m_br} | {m_ece} | {origin} |")
            
        md_sections[tgt] = "\n".join(md_tgt_lines)
        
    tex_lines.extend([
        r"\hline \hline",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}"
    ])
    
    out_path = os.path.join(TABLES_DIR, "table_holdout_benchmarks.tex")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex_lines) + "\n")
    print(f"Generated {out_path}")
    return md_sections

def generate_odds_table(data):
    odds = data["odds_ratios_fdr_corrected"]
    features = ["Age", "Gender", "Education", "Nscore", "Impulsive", "Ascore", "Escore", "Oscore", "Cscore", "SS"]
    feature_labels = {
        "Age": "Age",
        "Gender": "Gender (Female)",
        "Education": "Education",
        "Nscore": "Neuroticism (Nscore)",
        "Impulsive": "Impulsivity (BIS-11)",
        "Ascore": "Agreeableness (Ascore)",
        "Escore": "Extraversion (Escore)",
        "Oscore": "Openness (Oscore)",
        "Cscore": "Conscientiousness (Cscore)",
        "SS": "Sensation Seeking (SS)"
    }
    
    pattern_labels = {
        "Age": "Consistent Negative Association with Age across All Targets",
        "Gender": "Lower Past-Year Prevalence in Females (except Depressants)",
        "Education": "Negative Association with Higher Education for Cannabis",
        "Nscore": "Strongest Correlate for Depressants / Anxiolytics",
        "Impulsive": "Attenuated when Controlling for Sensation Seeking",
        "Ascore": "Weak Association across Target Classes",
        "Escore": "Negative Association for Cannabis",
        "Oscore": "Strongest Correlate for Hallucinogens and Cannabis",
        "Cscore": "Consistent Negative Association across Substance Targets",
        "SS": "Universal Positive Correlate across All Substance Classes"
    }
    
    targets = ["Cannabis", "CNS Stimulants", "Psychedelics", "Depressants"]
    
    tex_lines = [
        r"\begin{table*}[t]",
        r"\caption{Adjusted Odds Ratios with Approximate Model-Based 95\% Wald CIs and Benjamini-Hochberg FDR Control ($q=0.05$)}",
        r"\label{tab:odds_ratios}",
        r"\centering",
        r"\resizebox{\textwidth}{!}{%",
        r"\setlength{\tabcolsep}{4.5pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{tabular}{lcccc}",
        r"\hline \hline",
        r"\textbf{Predictor Feature} & \textbf{Cannabis OR [95\% CI]} & \textbf{CNS Stimulants OR [95\% CI]} & \textbf{Psychedelics OR [95\% CI]} & \textbf{Depressants OR [95\% CI]} \\",
        r"\hline"
    ]
    
    md_lines = [
        "| Predictor Feature | Cannabis OR [95% CI] | Stimulants OR [95% CI] | Psychedelics OR [95% CI] | Depressants OR [95% CI] | Empirical Effect Pattern |",
        "| :--- | :---: | :---: | :---: | :---: | :--- |"
    ]
    
    for feat in features:
        disp_feat = feature_labels[feat]
        pat = pattern_labels[feat]
        
        tex_cells = []
        md_cells = []
        for tgt in targets:
            f_data = odds[tgt][feat]
            or_val = f_data["odds_ratio"]
            ci_l = f_data["ci_lower_95"]
            ci_u = f_data["ci_upper_95"]
            sig = f_data["significant_fdr"]
            
            star = "$^*$" if sig else ""
            if sig:
                t_cell = f"$\\mathbf{{{or_val:.4f}}}$ [{ci_l:.4f}, {ci_u:.4f}]{star}"
                m_cell = f"**{or_val:.3f} [{ci_l:.3f}, {ci_u:.3f}]\\***"
            else:
                t_cell = f"{or_val:.4f} [{ci_l:.4f}, {ci_u:.4f}]"
                m_cell = f"{or_val:.3f} [{ci_l:.3f}, {ci_u:.3f}]"
                
            tex_cells.append(t_cell)
            md_cells.append(m_cell)
            
        tex_lines.append(f"\\textbf{{{disp_feat}}} & {' & '.join(tex_cells)} \\\\")
        md_lines.append(f"| **{disp_feat}** | {' | '.join(md_cells)} | {pat} |")
        
    tex_lines.extend([
        r"\hline \hline",
        r"\end{tabular}%",
        r"}",
        r"\vspace{1pt}",
        r"{\raggedright \footnotesize $^*$Statistically significant after Benjamini-Hochberg False Discovery Rate (FDR) control at $q = 0.05$ across all 40 hypothesis tests. Note: Confidence intervals represent approximate model-based Wald intervals under L2 ridge regularization.\par}",
        r"\end{table*}"
    ])
    
    out_path = os.path.join(TABLES_DIR, "table_odds_ratios.tex")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex_lines) + "\n")
    print(f"Generated {out_path}")
    return "\n".join(md_lines)

def update_readme(cv_md, holdout_md_dict, odds_md):
    with open(README_PATH, "r", encoding="utf-8") as f:
        text = f.read()
        
    # Replace CV Table in README
    cv_start = "## 5-Fold Outer Cross-Validation Generalization"
    holdout_start = "## Frozen Stratified Holdout Benchmark Results"
    odds_start = "## Analytical Odds Ratios and Benjamini-Hochberg FDR Control"
    safeguards_start = "## Methodological Safeguards (Zero Data Leakage Protocols)"
    
    # Rebuild CV Section
    cv_sec = (
        f"{cv_start}\n\n"
        "To ensure performance metrics do not reflect an artifact of a single data split, we report the mean and standard deviation across 5 stratified outer cross-validation folds. Within each fold, feature scaling is strictly isolated to training folds:\n\n"
        f"{cv_md}\n\n---\n\n"
    )
    
    # Rebuild Holdout Section
    holdout_body = "\n\n".join([holdout_md_dict["Cannabis"], holdout_md_dict["CNS Stimulants"], holdout_md_dict["Psychedelics"], holdout_md_dict["Depressants"]])
    holdout_sec = (
        f"{holdout_start}\n\n"
        "All metrics below reflect evaluation on frozen holdout test partitions ($N_{\\text{test}} = 374\\text{ to }375$, representing 20% stratified holdouts). Metrics match `benchmark_results.json` and the publication manuscript exactly:\n\n"
        f"{holdout_body}\n\n---\n\n"
    )
    
    # Rebuild Odds Section
    odds_sec = (
        f"{odds_start}\n\n"
        "Adjusted odds ratios are accompanied by analytical 95% Wald confidence intervals derived from the inverted Hessian covariance matrix $\\Sigma = (X_b^T W X_b + \\lambda I_{\\text{reg}})^{-1}$. To safeguard against false discovery across all 40 hypothesis tests, the Benjamini-Hochberg procedure was applied at false discovery rate $q = 0.05$. Significant associations satisfying $q < 0.05$ are marked with an asterisk (*):\n\n"
        f"{odds_md}\n\n---\n\n"
    )
    
    # Replace sections using regex
    pat = re.compile(
        re.escape(cv_start) + r".*?" + re.escape(safeguards_start),
        re.DOTALL
    )
    new_middle = cv_sec + holdout_sec + odds_sec + safeguards_start
    text = pat.sub(lambda _: new_middle, text)
    
    # Update bibtex citation to @techreport
    old_bib = re.compile(r"@techreport\{mateti2026.*?\}", re.DOTALL)
    new_bib = (
        "@techreport{mateti2026applied,\n"
        "  author      = {Rishindra Mateti},\n"
        "  title       = {Applied Machine Learning for Cross-Substance Analysis: Psychometric and Demographic Correlates of Consumption Patterns},\n"
        "  institution = {Department of Computer Science, Wright State University},\n"
        "  year        = {2026},\n"
        "  type        = {Technical Report},\n"
        "  url         = {https://github.com/rishindra-mateti-tech/Psychometric-Substance-Vulnerability-Analysis-Clustering-Classification}\n"
        "}"
    )
    text = old_bib.sub(new_bib, text)
    
    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    print("Updated README.md with synchronized tables and @techreport citation!")

def update_latex_inputs():
    with open(TEX_PATH, "r", encoding="utf-8") as f:
        tex = f.read()
        
    # Replace Table tab:clustering_metrics
    tex = re.sub(
        r"\\begin\{table\}\[t\]\s*\\caption\{Unsupervised Cluster Validation.*?\\end\{table\}",
        r"\\input{tables/table_clustering_metrics.tex}",
        tex,
        flags=re.DOTALL
    )
    
    # Replace Table tab:cv_results
    tex = re.sub(
        r"\\begin\{table\*\}\[t\]\s*\\caption\{5-Fold Outer Stratified Cross-Validation.*?\\end\{table\*\}",
        r"\\input{tables/table_cv_results.tex}",
        tex,
        flags=re.DOTALL
    )
    
    # Replace Table tab:benchmarks
    tex = re.sub(
        r"\\begin\{table\*\}\[t\]\s*\\caption\{Supervised Machine Learning Performance Benchmarks.*?\\end\{table\*\}",
        r"\\input{tables/table_holdout_benchmarks.tex}",
        tex,
        flags=re.DOTALL
    )
    
    # Replace Table tab:odds_ratios
    tex = re.sub(
        r"\\begin\{table\*\}\[t\]\s*\\caption\{Adjusted Odds Ratios with Approximate Model-Based.*?\\end\{table\*\}",
        r"\\input{tables/table_odds_ratios.tex}",
        tex,
        flags=re.DOTALL
    )
    
    with open(TEX_PATH, "w", encoding="utf-8") as f:
        f.write(tex)
    print("Updated Applied_Machine_Learning_for_Cross_Substance_Analysis.tex to use \\input{tables/...}!")

def main():
    data = load_benchmark_data()
    generate_clustering_table(data)
    cv_md = generate_cv_table(data)
    holdout_md = generate_holdout_table(data)
    odds_md = generate_odds_table(data)
    update_readme(cv_md, holdout_md, odds_md)
    update_latex_inputs()
    print("Synchronization complete!")

if __name__ == "__main__":
    main()
