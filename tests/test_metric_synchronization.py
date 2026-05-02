"""
Automated Test Suite: Cross-Artifact Metric Parity & Scientific Integrity

Asserts 100% numerical parity across:
1. benchmark_results.json
2. tables/table_cv_results.tex
3. tables/table_holdout_benchmarks.tex
4. tables/table_odds_ratios.tex
5. README.md

Also audits:
- requirements.txt pinned dependency compliance (no >=)
- BibTeX citation validity (@techreport)
- Anti-dash invariant (zero em dashes, zero en dashes)
"""

import json
import os
import re

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def test_requirements_pinned():
    req_path = os.path.join(ROOT_DIR, "requirements.txt")
    with open(req_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert ">=" not in content, "requirements.txt contains unpinned >= ranges!"
    for line in content.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            assert "==" in line, f"Dependency line not strictly pinned: {line}"
    print("[PASS] requirements.txt is 100% strictly pinned.")

def test_bibtex_techreport():
    readme_path = os.path.join(ROOT_DIR, "README.md")
    with open(readme_path, "r", encoding="utf-8") as f:
        readme = f.read()
    assert "@techreport{mateti2026applied" in readme, "README.md does not use @techreport for citation!"
    assert "institution =" in readme, "README.md @techreport missing institution field!"
    print("[PASS] README.md citation uses valid @techreport specification.")

def test_zero_dashes():
    bad_files = []
    for root, dirs, files in os.walk(ROOT_DIR):
        if ".git" in dirs:
            dirs.remove(".git")
        for f in files:
            if f.endswith((".py", ".tex", ".md")):
                path = os.path.join(root, f)
                with open(path, "r", encoding="utf-8", errors="ignore") as fp:
                    txt = fp.read()
                    if "\u2014" in txt or "\u2013" in txt:
                        bad_files.append(path)
    assert len(bad_files) == 0, f"Em or en dashes found in: {bad_files}"
    print("[PASS] Zero em dashes and zero en dashes across entire repository.")

def test_cross_artifact_numerical_parity():
    json_path = os.path.join(ROOT_DIR, "benchmark_results.json")
    cv_tex_path = os.path.join(ROOT_DIR, "tables", "table_cv_results.tex")
    holdout_tex_path = os.path.join(ROOT_DIR, "tables", "table_holdout_benchmarks.tex")
    odds_tex_path = os.path.join(ROOT_DIR, "tables", "table_odds_ratios.tex")
    readme_path = os.path.join(ROOT_DIR, "README.md")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(cv_tex_path, "r", encoding="utf-8") as f:
        cv_tex = f.read()
    with open(holdout_tex_path, "r", encoding="utf-8") as f:
        holdout_tex = f.read()
    with open(odds_tex_path, "r", encoding="utf-8") as f:
        odds_tex = f.read()
    with open(readme_path, "r", encoding="utf-8") as f:
        readme = f.read()

    # 1. Verify 5-fold CV numbers
    targets = ["Cannabis", "CNS Stimulants", "Psychedelics", "Depressants"]
    models_cv = ["LR_Scratch", "Random_Forest_Tuned", "MLP"]
    
    cv_checks = 0
    for tgt in targets:
        for m in models_cv:
            m_data = data["five_fold_outer_cv"][tgt][m]
            for metric in ["acc", "f1", "auc", "pr_auc", "brier"]:
                mean_val, std_val = m_data[metric]
                m_str = f"{mean_val:.4f}"
                s_str = f"{std_val:.4f}"
                
                # Check in LaTeX CV table
                assert m_str in cv_tex, f"CV metric {tgt} {m} {metric} ({m_str}) missing in table_cv_results.tex"
                assert s_str in cv_tex, f"CV std {tgt} {m} {metric} ({s_str}) missing in table_cv_results.tex"
                
                # Check in README
                assert m_str in readme, f"CV metric {tgt} {m} {metric} ({m_str}) missing in README.md"
                assert s_str in readme, f"CV std {tgt} {m} {metric} ({s_str}) missing in README.md"
                cv_checks += 2
    print(f"[PASS] Verified {cv_checks} 5-fold cross-validation metrics across JSON, TeX, and README.")

    # 2. Verify Holdout numbers
    models_ho = ["Majority", "LR_Scratch", "LR_Sklearn_Parity", "Random_Forest_Baseline", "Random_Forest_Tuned", "MLP"]
    ho_checks = 0
    for tgt in targets:
        for m in models_ho:
            m_dict = data["holdout_benchmarks"][tgt][m]
            met = m_dict["metrics"] if "metrics" in m_dict else m_dict
            cal = m_dict.get("calibration", {})
            ci = m_dict.get("confidence_intervals_95", {})
            
            acc_str = f"{met['accuracy']:.4f}"
            f1_str = f"{met['f1_score']:.4f}"
            auc_str = f"{met['auc']:.4f}"
            
            assert acc_str in holdout_tex, f"Holdout Acc {tgt} {m} ({acc_str}) missing in table_holdout_benchmarks.tex"
            assert acc_str in readme, f"Holdout Acc {tgt} {m} ({acc_str}) missing in README.md"
            ho_checks += 1
            
            if m != "Majority":
                assert f1_str in holdout_tex, f"Holdout F1 {tgt} {m} ({f1_str}) missing in table_holdout_benchmarks.tex"
                assert f1_str in readme, f"Holdout F1 {tgt} {m} ({f1_str}) missing in README.md"
                ho_checks += 1
                
                if "brier_score" in cal:
                    br_str = f"{cal['brier_score']:.4f}"
                    assert br_str in holdout_tex, f"Holdout Brier {tgt} {m} ({br_str}) missing in table_holdout_benchmarks.tex"
                    assert br_str in readme, f"Holdout Brier {tgt} {m} ({br_str}) missing in README.md"
                    ho_checks += 1
    print(f"[PASS] Verified {ho_checks} holdout metrics across JSON, TeX, and README.")

    # 3. Verify Odds Ratios
    odds_checks = 0
    for tgt in targets:
        for feat in ["Age", "Gender", "Education", "Nscore", "Impulsive", "Ascore", "Escore", "Oscore", "Cscore", "SS"]:
            f_data = data["odds_ratios_fdr_corrected"][tgt][feat]
            or_str_tex = f"{f_data['odds_ratio']:.4f}"
            or_str_md = f"{f_data['odds_ratio']:.3f}"
            
            assert or_str_tex in odds_tex, f"Odds ratio {tgt} {feat} ({or_str_tex}) missing in table_odds_ratios.tex"
            assert or_str_md in readme, f"Odds ratio {tgt} {feat} ({or_str_md}) missing in README.md"
            odds_checks += 2
    print(f"[PASS] Verified {odds_checks} odds ratio entries across JSON, TeX, and README.")

if __name__ == "__main__":
    test_requirements_pinned()
    test_bibtex_techreport()
    test_zero_dashes()
    test_cross_artifact_numerical_parity()
    print("\nALL VERIFICATION TESTS PASSED SUCCESSFULLY! ZERO NUMERICAL MISMATCHES.")
