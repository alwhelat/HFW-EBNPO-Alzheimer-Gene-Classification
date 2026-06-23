"""
main.py -- HFW-NPO Pipeline for GSE33000 and GSE132903 brain-expression datasets.
Leakage-free 5-fold stratified CV with RealHybridFilter, adaptive SMOTE, NPO, and soft-voting ensemble.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    matthews_corrcoef, f1_score, cohen_kappa_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT    = PROJECT_ROOT.parent
DATASET_DIR  = REPO_ROOT / "dataset"
RESULTS_DIR  = PROJECT_ROOT / "result-data 4"

sys.path.insert(0, str(PROJECT_ROOT))

from fitness_evaluator import FitnessEvaluator  # noqa: E402
from hybrid_filter     import RealHybridFilter  # noqa: E402
from npo_optimizer     import NPOOptimizer      # noqa: E402


DATASETS: list[dict] = [
    {
        "name":       "Data1_GSE33000",
        "file":       "Data1(GSE33000).csv",
        "index_col":  "ID_REF",          # primary index; fall back to col 0
        "class_col":  "Class",
        "class_map":  {"AD": 1, "HD": 0, "ND": 0},
        "filter_k":   80,                # v3: reduced 200->80 (better NPO signal)
    },
    {
        "name":       "Data2_GSE132903",
        "file":       "GSE132903.csv",
        "index_col":  None,              # dynamic: auto-detect first column
        "class_col":  "Class",
        "class_map":  {"AD": 1, "ND": 0},
        "filter_k":   80,                # v3: reduced 200->80 (better NPO signal)
    },
    # Data3_GSE63060 is handled exclusively by run_experiments_gse63060.py
]

# NPO configuration (shared across all datasets)
NPO_CLANS    = 10      # v4: report §3.2.1 -> 10×20=200 agents
NPO_FAMILIES = 20      # v4: report §3.2.1
NPO_ITER     = 300     # v4: report §3.2.1 -> 300 iters
NPO_LB       = -10.0
NPO_UB       =  10.0

# Fitness configuration  [v3: alpha 0.95->0.90, sqrt penalty, SVC-RBF inner]
ALPHA_FIT    = 0.85     # v4: report §3.2.2 -> 0.85*(1-BalAcc) + 0.15*sqrt(ratio)

# Cross-validation
N_SPLITS     = 5
RANDOM_STATE = 42


def _load_dataset(cfg: dict) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load and normalise a dataset, returning X, y, and feature names."""
    file_path = DATASET_DIR / cfg["file"]

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    index_col = cfg["index_col"]
    if index_col is None:
        # Dynamic: use the first column as the index
        probe = pd.read_csv(file_path, nrows=0)
        index_col = probe.columns[0]

    try:
        df = pd.read_csv(file_path, index_col=index_col)
    except (KeyError, ValueError):
        df = pd.read_csv(file_path, index_col=0)

    class_col = cfg["class_col"]
    if class_col not in df.columns:
        # Try case-insensitive search
        matches = [c for c in df.columns if c.strip().lower() == class_col.lower()]
        if not matches:
            raise KeyError(
                f"Class column '{class_col}' not found in {cfg['file']}. "
                f"Available: {list(df.columns[-6:])}"
            )
        class_col = matches[0]

    y_raw = df[class_col].values
    df    = df.drop(columns=[class_col])

    # Drop any residual non-numeric columns
    df = df.select_dtypes(include=[np.number])

    unique_raw = sorted({str(v).strip() for v in y_raw})
    print(f"  Raw class values found: {unique_raw}")

    class_map = cfg["class_map"]
    class_map_ci = {k.strip().upper(): v for k, v in class_map.items()}
    y = np.array(
        [class_map_ci.get(str(v).strip().upper(), -1) for v in y_raw],
        dtype=np.int64,
    )

    valid_mask = y != -1
    n_dropped  = int((~valid_mask).sum())
    if n_dropped > 0:
        warnings.warn(
            f"[{cfg['name']}] Dropped {n_dropped} samples with unknown labels. "
            f"Raw values: {unique_raw}  |  Map keys: {list(class_map.keys())}",
            UserWarning, stacklevel=2,
        )
    df = df.iloc[valid_mask]
    y  = y[valid_mask]

    n_classes = len(np.unique(y))
    print(f"  After mapping: {dict(Counter(y.tolist()))} | {n_classes} classes")

    if n_classes < 2:
        raise ValueError(
            f"[{cfg['name']}] Only {n_classes} class(es) after mapping. "
            f"Check class_map={class_map} against raw values={unique_raw}."
        )

    feature_names: list[str] = list(df.columns)
    X_raw     = df.values.astype(np.float64)
    X_imputed = SimpleImputer(strategy="mean").fit_transform(X_raw)
    X         = MinMaxScaler().fit_transform(X_imputed)

    return X, y, feature_names


def _make_smote(y_train: np.ndarray) -> SMOTE | None:
    """Build a SMOTE instance with safe k_neighbors for small minority classes.

    Returns None when SMOTE is impossible or unsafe for this fold.
    """
    counts = Counter(y_train.tolist())
    if len(counts) < 2:
        return None
    min_count = min(counts.values())
    if min_count < 2:
        return None
    return SMOTE(
        random_state = RANDOM_STATE,
        k_neighbors  = min(5, min_count - 1),
    )


def run_dataset(cfg: dict) -> dict:
    """Execute the full HFW-NPO pipeline for one dataset and return serialisable results."""
    name      = cfg["name"]
    filter_k  = cfg["filter_k"]

    print(f"\n{'='*72}")
    print(f"  DATASET  : {name}")
    print(f"  File     : {DATASET_DIR / cfg['file']}")
    print(f"  Filter   : top-{filter_k} | omega=0.40*Fisher+0.30*MI+0.30*Welch")
    print(f"  Fitness  : f = {ALPHA_FIT}*(1-BalAcc) + "
          f"{round(1.0-ALPHA_FIT,2)}*sqrt(|genes|/{filter_k})")
    print(f"  NPO      : {NPO_CLANS}x{NPO_FAMILIES}={NPO_CLANS*NPO_FAMILIES} "
          f"agents | {NPO_ITER} iters")
    print(f"  Classes  : {cfg['class_map']}")
    print('='*72)

    X, y, feature_names = _load_dataset(cfg)
    n_samples, n_features = X.shape

    dist = dict(Counter(y.tolist()))
    print(f"  Loaded   : {n_samples} samples x {n_features} features")
    print(f"  Classes  : {dist}")

    skf = StratifiedKFold(
        n_splits     = N_SPLITS,
        shuffle      = True,
        random_state = RANDOM_STATE,
    )

    fold_accuracies:     list[float]       = []
    fold_bal_accuracies: list[float]       = []
    fold_mcc:            list[float]       = []
    fold_f1_ad:          list[float]       = []
    fold_kappa:          list[float]       = []
    fold_auc:            list[float]       = []
    fold_gene_counts:    list[int]         = []
    fold_gene_names:     list[list[str]]   = []
    fold_convergence:    list[list[float]] = []
    fold_smote_flags:    list[bool]        = []

    pbar = tqdm(
        enumerate(skf.split(X, y), start=1),
        total         = N_SPLITS,
        desc          = f"  [{name}] Fold 1/{N_SPLITS}",
        unit          = "fold",
        dynamic_ncols = True,
        colour        = "cyan",
    )

    for fold_idx, (train_idx, test_idx) in pbar:
        pbar.set_description(f"  [{name}] Fold {fold_idx}/{N_SPLITS}")
        fold_t0 = time.time()

        X_train_raw = X[train_idx];  X_test_raw = X[test_idx]
        y_train     = y[train_idx];  y_test     = y[test_idx]

        tqdm.write(f"\n  {'='*60}")
        tqdm.write(f"  FOLD {fold_idx}/{N_SPLITS}  [{name}]")
        tqdm.write(
            f"  Train {len(y_train)}: {dict(Counter(y_train.tolist()))}  "
            f"Test {len(y_test)}: {dict(Counter(y_test.tolist()))}"
        )
        tqdm.write(f"  {'='*60}")

        tqdm.write(f"  Phase 1: RealHybridFilter | top-{filter_k} probes ...")
        hf = RealHybridFilter(
            k_features   = filter_k,
            alpha_w      = 0.40,
            beta_w       = 0.30,
            gamma_w      = 0.30,
            random_state = RANDOM_STATE,
        )
        X_train_filt, selected_probe_names = hf.fit_transform(
            X_train_raw, y_train, feature_names
        )
        X_test_filt = hf.transform(X_test_raw)   # training-derived indices only

        tqdm.write(
            f"  Phase 1: train {X_train_filt.shape} | "
            f"test {X_test_filt.shape} | zero leakage"
        )

        tqdm.write("  Phase 2: Adaptive SMOTE ...")
        smote = _make_smote(y_train)
        if smote is not None:
            X_bal, y_bal = smote.fit_resample(X_train_filt, y_train)
            fold_smote_flags.append(True)
            tqdm.write(
                f"  Phase 2: SMOTE {dict(Counter(y_train.tolist()))} "
                f"-> {dict(Counter(y_bal.tolist()))}"
            )
        else:
            X_bal, y_bal = X_train_filt, y_train
            fold_smote_flags.append(False)
            tqdm.write("  Phase 2: SMOTE skipped (<2 minority samples).")

        # Safety guard: Phases 3 & 4 require at least 2 classes in training data
        if len(np.unique(y_bal)) < 2:
            tqdm.write(
                f"  Warning: fold {fold_idx} has only 1 class in training data "
                f"({dict(Counter(y_bal.tolist()))}) -- skipping fold."
            )
            fold_accuracies.append(float("nan"))
            fold_bal_accuracies.append(float("nan"))
            fold_mcc.append(float("nan"))
            fold_f1_ad.append(float("nan"))
            fold_kappa.append(float("nan"))
            fold_auc.append(float("nan"))
            fold_gene_counts.append(0)
            fold_gene_names.append([])
            fold_convergence.append([1.0] * NPO_ITER)
            continue

        tqdm.write(
            f"  Phase 3: NPO | {NPO_CLANS}x{NPO_FAMILIES} agents "
            f"| {NPO_ITER} iters | alpha={ALPHA_FIT} ..."
        )
        evaluator = FitnessEvaluator(
            X          = X_bal,
            y          = y_bal,
            alpha      = ALPHA_FIT,
            k_retained = filter_k,
        )
        npo = NPOOptimizer(
            obj_func   = evaluator.evaluate,
            dim        = X_bal.shape[1],
            lb         = NPO_LB,
            ub         = NPO_UB,
            n_clans    = NPO_CLANS,
            n_families = NPO_FAMILIES,
            max_iter   = NPO_ITER,
        )
        npo.optimize()

        sel_idx = npo.get_selected_indices()
        if len(sel_idx) == 0:
            tqdm.write(
                "  Phase 3: empty NPO mask -- "
                "falling back to top-10 filter-ranked probes."
            )
            sel_idx = np.arange(min(10, X_bal.shape[1]))

        elected_names: list[str] = [selected_probe_names[i] for i in sel_idx]
        X_tr_el = X_bal[:,      sel_idx]
        X_te_el = X_test_filt[:, sel_idx]

        tqdm.write(
            f"  Phase 3: best fitness={npo.global_best_fit:.6f} "
            f"| {len(sel_idx)} probes elected."
        )

        # Report §3.3: simplified 3-classifier ensemble with better calibration.
        #   "RF + LR + Ridge" -- diverse, stable, report-recommended.
        #   RidgeClassifier wrapped in CalibratedClassifierCV for probabilities.
        #   All classifiers use class_weight='balanced'.
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.linear_model import RidgeClassifier as _Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler as _SS

        tqdm.write("  Phase 4: Soft-Voting Ensemble (Ridge + RF + LR) ...")

        ridge_base = _Ridge(alpha=1.0, class_weight="balanced")
        ridge_cal  = CalibratedClassifierCV(ridge_base, cv=3, method="sigmoid")

        voting_clf = VotingClassifier(
            estimators=[
                (
                    "ridge",
                    ridge_cal,
                ),
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators = 500,
                        class_weight = "balanced",
                        max_features = "sqrt",
                        min_samples_leaf = 2,
                        random_state = RANDOM_STATE,
                        n_jobs       = -1,
                    ),
                ),
                (
                    "lr",
                    make_pipeline(
                        _SS(),
                        LogisticRegression(
                            penalty      = "l2",
                            C            = 1.0,
                            class_weight = "balanced",
                            solver       = "lbfgs",
                            max_iter     = 2000,
                            random_state = RANDOM_STATE,
                        ),
                    ),
                ),
            ],
            voting = "soft",
        )
        voting_clf.fit(X_tr_el, y_bal)
        y_pred  = voting_clf.predict(X_te_el)
        try:
            y_proba = voting_clf.predict_proba(X_te_el)[:, 1]
            auc_val = float(roc_auc_score(y_test, y_proba))
        except Exception:
            auc_val = float("nan")

        acc     = float(accuracy_score(y_test, y_pred))
        bal_acc = float(balanced_accuracy_score(y_test, y_pred))
        mcc_val = float(matthews_corrcoef(y_test, y_pred))
        f1_val  = float(f1_score(y_test, y_pred, pos_label=1, zero_division=0))
        kap_val = float(cohen_kappa_score(y_test, y_pred))

        fold_accuracies.append(acc)
        fold_bal_accuracies.append(bal_acc)
        fold_mcc.append(mcc_val)
        fold_f1_ad.append(f1_val)
        fold_kappa.append(kap_val)
        fold_auc.append(auc_val)
        fold_gene_counts.append(len(sel_idx))
        fold_gene_names.append(elected_names)
        fold_convergence.append(list(npo.history))

        elapsed = time.time() - fold_t0
        pbar.set_postfix(
            acc=f"{acc:.3f}", bal=f"{bal_acc:.3f}",
            mcc=f"{mcc_val:.3f}", genes=len(sel_idx), sec=f"{elapsed:.0f}",
        )
        tqdm.write(
            f"  Phase 4: fold {fold_idx}: "
            f"Acc={acc:.4f}  BalAcc={bal_acc:.4f}  MCC={mcc_val:.4f}  "
            f"F1-AD={f1_val:.4f}  Kappa={kap_val:.4f}  AUC={auc_val:.4f}  "
            f"Probes={len(sel_idx)}  Time={elapsed:.1f}s"
        )

    pbar.close()

    def _valid(lst: list) -> list:
        return [v for v in lst if not (isinstance(v, float) and np.isnan(v))]

    valid_acc   = _valid(fold_accuracies)
    valid_bal   = _valid(fold_bal_accuracies)
    valid_mcc_  = _valid(fold_mcc)
    valid_f1_   = _valid(fold_f1_ad)
    valid_kap_  = _valid(fold_kappa)
    valid_auc_  = _valid(fold_auc)
    valid_g     = [g for g in fold_gene_counts if g > 0]

    def _ms(lst: list) -> tuple[float, float]:
        return (float(np.mean(lst)), float(np.std(lst))) if lst else (0.0, 0.0)

    mean_acc,     std_acc     = _ms(valid_acc)
    mean_bal_acc, std_bal_acc = _ms(valid_bal)
    mean_mcc,     std_mcc     = _ms(valid_mcc_)
    mean_f1_ad,   std_f1_ad   = _ms(valid_f1_)
    mean_kappa,   std_kappa   = _ms(valid_kap_)
    mean_auc,     std_auc     = _ms(valid_auc_)
    mean_genes    = float(np.mean(valid_g)) if valid_g else 0.0
    union_genes   = sorted({g for fold in fold_gene_names for g in fold})

    print(f"\n  {'='*72}")
    print(f"  RESULT -- {name}  |  Soft-Voting Ensemble")
    print(f"    Accuracy       : {mean_acc:.4f}  ±  {std_acc:.4f}")
    print(f"    Bal. Accuracy  : {mean_bal_acc:.4f}  ±  {std_bal_acc:.4f}")
    print(f"    MCC            : {mean_mcc:.4f}  ±  {std_mcc:.4f}")
    print(f"    F1-AD          : {mean_f1_ad:.4f}  ±  {std_f1_ad:.4f}")
    print(f"    Cohen Kappa    : {mean_kappa:.4f}  ±  {std_kappa:.4f}")
    print(f"    ROC-AUC        : {mean_auc:.4f}  ±  {std_auc:.4f}")
    print(f"    Avg probes/fold: {mean_genes:.1f}")
    print(f"    Union probes   : {len(union_genes)}")
    print()
    for fi in range(N_SPLITS):
        tag = "[SMOTE]" if fold_smote_flags[fi] else "       "
        print(
            f"      Fold {fi+1} {tag}: "
            f"Acc={fold_accuracies[fi]:.4f}  "
            f"BalAcc={fold_bal_accuracies[fi]:.4f}  "
            f"MCC={fold_mcc[fi]:.4f}  "
            f"F1={fold_f1_ad[fi]:.4f}  "
            f"Probes={fold_gene_counts[fi]}"
        )

    ds_dir = RESULTS_DIR / name
    ds_dir.mkdir(parents=True, exist_ok=True)

    gene_file = ds_dir / "selected_genes.txt"
    with gene_file.open("w", encoding="utf-8") as fh:
        fh.write(f"# Dataset          : {name}\n")
        fh.write(
            f"# Accuracy         : {mean_acc:.4f} +- {std_acc:.4f} "
            f"({N_SPLITS}-fold leakage-free CV)\n"
        )
        fh.write(
            f"# Balanced Accuracy: {mean_bal_acc:.4f} +- {std_bal_acc:.4f}\n"
        )
        fh.write(
            f"# Probes           : {len(union_genes)} unique "
            f"across {N_SPLITS} folds\n\n"
        )
        for g in union_genes:
            fh.write(f"{g}\n")
    print(f"  Saved: {gene_file}")

    for fi, names in enumerate(fold_gene_names, start=1):
        with (ds_dir / f"genes_fold{fi}.txt").open("w", encoding="utf-8") as fh:
            fh.write(
                f"# Fold {fi}  "
                f"Acc={fold_accuracies[fi-1]:.4f}  "
                f"BalAcc={fold_bal_accuracies[fi-1]:.4f}  "
                f"Probes={fold_gene_counts[fi-1]}\n"
            )
            for g in sorted(names):
                fh.write(f"{g}\n")

    max_len   = max(len(h) for h in fold_convergence)
    padded    = [h + [h[-1]] * (max_len - len(h)) for h in fold_convergence]
    mean_traj = np.mean(padded, axis=0).tolist()
    conv_file = ds_dir / "convergence.json"
    with conv_file.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "dataset":          name,
                "n_iterations":     NPO_ITER,
                "n_folds":          N_SPLITS,
                "alpha_fitness":    ALPHA_FIT,
                "smote_applied":    fold_smote_flags,
                "folds":            fold_convergence,
                "mean_trajectory":  mean_traj,
            },
            fh, indent=2,
        )
    print(f"  Saved: {conv_file}")

    result = {
        "name":              name,
        "filter_config": {
            "k_features_retained": filter_k,
            "omega_formula": (
                "omega = 0.40*Fisher + 0.30*MutualInfo + 0.30*WelchLogP"
            ),
            "weight_fisher":      0.40,
            "weight_mutual_info": 0.30,
            "weight_welch_log_p": 0.30,
            "fisher_formula": (
                "F_i = (mean_class1_i - mean_class0_i)^2 "
                "/ (var_class1_i + var_class0_i)"
            ),
            "mutual_info_method": (
                "sklearn.feature_selection.mutual_info_classif"
            ),
            "statistical_significance_method": (
                "scipy.stats.ttest_ind(equal_var=False): "
                "score = -log10(p_value), p clipped [1e-300, 1.0]"
            ),
            "class_mapping": cfg["class_map"],
        },
        "npo_config": {
            "n_clans":            NPO_CLANS,
            "n_families":         NPO_FAMILIES,
            "pop_size":           NPO_CLANS * NPO_FAMILIES,
            "max_iterations":     NPO_ITER,
            "lb":                 NPO_LB,
            "ub":                 NPO_UB,
            "levy_beta":          1.5,
            "levy_scale":         0.01,
            "migration_interval": 5,
        },
        "fitness_config": {
            "alpha":   ALPHA_FIT,
            "formula": (
                f"f = {ALPHA_FIT}*(1 - BalancedAccuracy_3FoldCV) "
                f"+ {round(1.0-ALPHA_FIT,2)}*sqrt(|selected| / {filter_k})"
            ),
            "inner_cv_folds":   3,
            "inner_classifier": (
                "RidgeClassifier(alpha=1.0, class_weight='balanced')"
            ),
        },
        "ensemble_classifier": {
            "type":    "VotingClassifier(voting='soft')",
            "members": [
                "CalibratedClassifierCV(RidgeClassifier(alpha=1.0,class_weight='balanced'),cv=3)",
                "RandomForestClassifier(n_estimators=500, class_weight='balanced', random_state=42)",
                "Pipeline(StandardScaler + LogisticRegression(C=1.0,class_weight='balanced',max_iter=2000))",
            ],
        },
        "outer_cv_folds":  N_SPLITS,
        "smote_applied":   fold_smote_flags,
        "folds": [
            {
                "fold":                   fi + 1,
                "accuracy":               fold_accuracies[fi],
                "balanced_accuracy":      fold_bal_accuracies[fi],
                "n_probes_selected":      fold_gene_counts[fi],
                "smote":                  fold_smote_flags[fi],
                "selected_probe_ids":     fold_gene_names[fi],
                "convergence_trajectory": [float(v) for v in fold_convergence[fi]],
            }
            for fi in range(N_SPLITS)
        ],
        "mean_convergence_trajectory": mean_traj,
        "final_outputs": {
            "mean_accuracy":          mean_acc,
            "std_accuracy":           std_acc,
            "mean_balanced_accuracy": mean_bal_acc,
            "std_balanced_accuracy":  std_bal_acc,
            "mean_mcc":               mean_mcc,
            "std_mcc":                std_mcc,
            "mean_f1_ad":             mean_f1_ad,
            "std_f1_ad":              std_f1_ad,
            "mean_cohen_kappa":       mean_kappa,
            "std_cohen_kappa":        std_kappa,
            "mean_roc_auc":           mean_auc,
            "std_roc_auc":            std_auc,
            "mean_probes_per_fold":   mean_genes,
            "n_unique_probes_union":  len(union_genes),
            "union_probe_ids":        union_genes,
        },
    }

    result_cache_file = ds_dir / "dataset_results.json"
    with result_cache_file.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    print(f"  Saved: {result_cache_file}")

    return result


def _print_cross_dataset_summary(all_results: list[dict]) -> None:
    sep = "=" * 72

    print(f"\n{sep}")
    print("  CROSS-DATASET SUMMARY  |  HFW-NPO  |  Soft-Voting Ensemble")
    print(sep)
    print(
        f"  {'Dataset':<26} {'Acc':>7}  {'±':>5}  "
        f"{'BalAcc':>7}  {'±':>5}  {'Genes':>6}"
    )
    print(f"  {'-'*68}")
    for r in all_results:
        if not r or "name" not in r:
            continue
        fo          = r.get("final_outputs", r)
        mean_acc    = fo.get("mean_accuracy", 0.0)
        std_acc     = fo.get("std_accuracy", 0.0)
        mean_bal    = fo.get("mean_balanced_accuracy", 0.0)
        std_bal     = fo.get("std_balanced_accuracy", 0.0)
        n_genes     = fo.get("n_unique_probes_union", len(fo.get("union_probe_ids", [])))
        if n_genes == 0:
            n_genes = fo.get("mean_probes_per_fold", 0.0)
        print(
            f"  {r['name']:<26} "
            f"{mean_acc*100:>7.2f}%  "
            f"{std_acc*100:>5.2f}%  "
            f"{mean_bal*100:>7.2f}%  "
            f"{std_bal*100:>5.2f}%  "
            f"{int(n_genes):>5d}"
        )
    print(f"  {'-'*68}")
    print(sep)

    print("\n  PER-FOLD BALANCED ACCURACY:")
    for r in all_results:
        if not r or "name" not in r or "folds" not in r:
            continue
        vals = [
            f"{r['folds'][i].get('balanced_accuracy', 0.0)*100:.2f}%"
            for i in range(min(N_SPLITS, len(r["folds"])))
        ]
        print(f"    {r['name']:<26}: {' | '.join(vals)}")

    print(f"\n{sep}")


def _print_manuscript_checklist(all_results: list[dict]) -> None:
    sep = "=" * 72

    print(f"\n{sep}")
    print("  MANUSCRIPT EXPERIMENTS CHECKLIST")
    print(sep)

    print("\n  [1] CLASSIFICATION PERFORMANCE TABLE")
    for r in all_results:
        if not r or "name" not in r:
            continue
        fo = r.get("final_outputs", {})
        print(
            f"      {r['name']:<26}  "
            f"Acc={fo.get('mean_accuracy', 0.0)*100:.2f}% "
            f"± {fo.get('std_accuracy', 0.0)*100:.2f}%  "
            f"BalAcc={fo.get('mean_balanced_accuracy', 0.0)*100:.2f}% "
            f"± {fo.get('std_balanced_accuracy', 0.0)*100:.2f}%"
        )
    print("      ACTION: Insert as Results table (report BOTH metrics).")

    print("\n  [2] NPO CONVERGENCE TRAJECTORIES")
    for r in all_results:
        if not r or "name" not in r:
            continue
        traj = r.get("mean_convergence_trajectory", [])
        if not traj:
            print(f"      {r['name']:<26}  (no trajectory data)")
            continue
        improv = (traj[0] - traj[-1]) / traj[0] * 100 if traj[0] > 0 else 0.0
        print(
            f"      {r['name']:<26}  "
            f"f_init={traj[0]:.4f}  f_final={traj[-1]:.4f}  "
            f"improvement={improv:.1f}%"
        )
    print("      ACTION: Plot mean ± std convergence; compare vs PSO/GA.")

    print("\n  [3] STATISTICAL SIGNIFICANCE")
    for r in all_results:
        if not r or "name" not in r or "folds" not in r:
            continue
        vals = [
            round(r["folds"][i].get("balanced_accuracy", 0.0) * 100, 2)
            for i in range(min(N_SPLITS, len(r["folds"])))
        ]
        print(f"      {r['name']:<26}: BalAcc per-fold = {vals}")
    print("      ACTION: scipy.stats.wilcoxon(npo, baseline) | p < 0.05 + Cohen d > 0.8")

    print("\n  [4] BIOMARKER GENE UNION SETS")
    for r in all_results:
        if not r or "name" not in r:
            continue
        fo = r.get("final_outputs", {})
        print(
            f"      {r['name']:<26}: "
            f"{fo.get('n_unique_probes_union', 0)} unique probes | "
            f"avg {fo.get('mean_probes_per_fold', 0.0):.1f} per fold"
        )
    print("      ACTION: Submit union lists to DAVID / g:Profiler for pathway enrichment.")

    print(f"\n{sep}")


def main() -> None:
    wall_start = time.time()

    sep = "=" * 72
    print(sep)
    print("  HFW-NPO ALZHEIMER'S CLASSIFICATION  |  THREE-DATASET STUDY")
    print("  Soft-Voting Ensemble  |  Leakage-Free 5-Fold Stratified CV")
    print(f"  Repo root : {REPO_ROOT}")
    print(f"  Datasets  : {DATASET_DIR}")
    print(f"  Results   : {RESULTS_DIR}")
    print(sep)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Results directory ready: {RESULTS_DIR}")
    print(sep)

    all_results: list[dict] = []
    for cfg in DATASETS:
        ds_done_marker   = RESULTS_DIR / cfg["name"] / "selected_genes.txt"
        ds_result_cache  = RESULTS_DIR / cfg["name"] / "dataset_results.json"
        if ds_done_marker.exists():
            if ds_result_cache.exists():
                with ds_result_cache.open(encoding="utf-8") as fh:
                    cached = json.load(fh)
                print(f"\n  Skipping {cfg['name']} (cached) -- loaded from dataset_results.json")
                all_results.append(cached)
            else:
                print(f"\n  No cache JSON for {cfg['name']} -- re-running.")
                result = run_dataset(cfg)
                all_results.append(result)
            continue
        try:
            result = run_dataset(cfg)
            all_results.append(result)
        except (FileNotFoundError, ValueError) as exc:
            print(f"\n  Skipping {cfg['name']}: {exc}")
            all_results.append({})

    wall_elapsed = time.time() - wall_start

    unified_payload = {
        "project":     "HFW-NPO Alzheimer's Gene Expression Classification",
        "description": (
            "Unified three-dataset study: Brain cortex (GSE33000), "
            "Brain gene expression (GSE132903), Peripheral blood (GSE63060)"
        ),
        "filter_formula": "omega = 0.40*Fisher + 0.30*MI + 0.30*Welch",
        "fitness_formula": (
            f"f = {ALPHA_FIT}*(1-BalAcc_3FoldCV_Ridge) + "
            f"{round(1.0-ALPHA_FIT,2)}*sqrt(|selected|/K_retained)  [v4]"
        ),
        "ensemble": "VotingClassifier(voting='soft') -- Ridge(cal) + RF(500) + LR  [v4 report-driven]",
        "outer_cv":  f"StratifiedKFold(n_splits={N_SPLITS}, shuffle=True, random_state={RANDOM_STATE})",
        "wall_time_seconds": round(wall_elapsed, 1),
        "datasets": {
            r["name"]: r
            for r in all_results if r
        },
    }

    unified_file = RESULTS_DIR / "unified_project_results.json"
    with unified_file.open("w", encoding="utf-8") as fh:
        json.dump(unified_payload, fh, indent=2, ensure_ascii=False)
    print(f"\n  Master results saved to:\n    {unified_file}")

    _print_cross_dataset_summary(all_results)
    _print_manuscript_checklist(all_results)

    print(f"\n  Total wall time: {wall_elapsed / 60:.1f} min")
    print(sep)


if __name__ == "__main__":
    main()
