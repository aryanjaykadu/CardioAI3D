"""
Evaluation Module for Clinical Classification and Multitask Learning Pipelines.

Features:
- Comprehensive clinical metrics: Accuracy, Precision, Recall/Sensitivity, Specificity, F1, ROC-AUC
- Calibration evaluation: Brier Score and Expected Calibration Error (ECE)
- Zero-leakage Stratified K-Fold Cross-Validation:
  * Strict training-fold isolation of ColumnTransformer preprocessing
  * Within-fold Recursive Feature Elimination (RFE) pruning to top 15 predictors
  * Class imbalance mitigation using Focal Loss or SMOTE
  * Output probability calibration via Platt Scaling or Isotonic Regression
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

from src.preprocessing import build_preprocessor, get_feature_names_from_preprocessor
from src.models import MultitaskCardiacModel, TARGET_TASKS


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Computes Expected Calibration Error (ECE) across probability bins.
    ECE measures the difference between confidence and empirical accuracy.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_proba, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    n_total = len(y_true)

    for b in range(n_bins):
        mask = (bin_indices == b)
        n_bin = np.sum(mask)
        if n_bin > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_proba[mask])
            ece += (n_bin / n_total) * np.abs(bin_acc - bin_conf)

    return float(ece)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    """
    Calculates primary clinical performance and calibration metrics on test predictions.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    y_proba = np.asarray(y_proba).ravel()

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Fallback if only 1 class present
        tn, fp, fn, tp = 0, 0, 0, 0

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    try:
        auc = roc_auc_score(y_true, y_proba)
    except (ValueError, IndexError):
        auc = np.nan

    try:
        brier = brier_score_loss(y_true, y_proba)
    except Exception:
        brier = np.nan

    ece = compute_expected_calibration_error(y_true, y_proba)

    metrics = {
        'Accuracy': float(accuracy_score(y_true, y_pred)),
        'Precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'Recall (Sensitivity)': float(recall_score(y_true, y_pred, zero_division=0)),
        'Specificity': float(specificity),
        'F1-Score': float(f1_score(y_true, y_pred, zero_division=0)),
        'ROC-AUC': float(auc),
        'Brier Score (Calibration)': float(brier),
        'ECE (Calibration Error)': float(ece),
        'True Positives': int(tp),
        'False Positives': int(fp),
        'True Negatives': int(tn),
        'False Negatives': int(fn)
    }
    return metrics


def run_rfe_feature_selection(
    X_train_trans: np.ndarray,
    y_train_cad: np.ndarray,
    feature_names: List[str],
    n_features_to_select: int = 15,
    random_state: int = 42
) -> Tuple[np.ndarray, List[str]]:
    """
    Runs Recursive Feature Elimination strictly on the training fold to select
    the top N critical clinical predictors with zero data leakage.
    """
    base_estimator = LogisticRegression(
        C=0.5,
        solver='liblinear',
        max_iter=1000,
        random_state=random_state
    )

    rfe = RFE(
        estimator=base_estimator,
        n_features_to_select=min(n_features_to_select, X_train_trans.shape[1]),
        step=1
    )
    rfe.fit(X_train_trans, y_train_cad)
    support_mask = rfe.support_

    selected_names = [feature_names[i] for i, sel in enumerate(support_mask) if sel]
    return support_mask, selected_names


def evaluate_multitask_pipeline_cv(
    X: pd.DataFrame,
    targets: Dict[str, pd.Series],
    n_splits: int = 5,
    n_features_to_select: int = 15,
    apply_smote: bool = False,
    calibration_method: str = 'platt',
    epochs: int = 100,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Executes a rigorous 5-fold Stratified Cross-Validation on the Multitask Learning Pipeline.

    Strict Leakage Prevention Guarantees:
    1. Preprocessing (StandardScaler + OneHotEncoder) is re-fitted ONLY inside training fold.
    2. RFE (pruning down to 15 clinical predictors) is executed ONLY on training fold.
    3. Severe class imbalance is counteracted using Focal Loss (or SMOTE).
    4. Platt Scaling / Isotonic Regression calibration is learned on an inner training-fold split.
    5. The validation fold is completely unpolluted and only evaluated at test time.

    Returns:
        Summary dict containing target metrics, fold details, and feature frequencies.
    """
    # Stratify by CAD status to ensure balanced fold splits
    strat_labels = targets['CAD'].values
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_metrics_by_target: Dict[str, List[Dict[str, float]]] = {t: [] for t in TARGET_TASKS}
    selected_features_per_fold: List[List[str]] = []
    feature_selection_counts: Dict[str, int] = {}

    print(f"\n[CV] Executing {n_splits}-Fold Multitask Cross-Validation (RFE={n_features_to_select}, Calibration={calibration_method.upper()}):")

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, strat_labels)):
        # 1. Isolate Raw Splits
        X_train_raw = X.iloc[train_idx].copy()
        X_val_raw = X.iloc[val_idx].copy()

        y_train_dict = {t: targets[t].iloc[train_idx].values for t in TARGET_TASKS}
        y_val_dict = {t: targets[t].iloc[val_idx].values for t in TARGET_TASKS}

        # 2. Strict Preprocessing Isolation
        preprocessor, _, _ = build_preprocessor(X_train_raw)
        X_train_trans = preprocessor.fit_transform(X_train_raw)
        X_val_trans = preprocessor.transform(X_val_raw)
        raw_feature_names = get_feature_names_from_preprocessor(preprocessor)

        # 3. Recursive Feature Elimination (RFE) on Training Fold Only
        rfe_mask, selected_names = run_rfe_feature_selection(
            X_train_trans=X_train_trans,
            y_train_cad=y_train_dict['CAD'],
            feature_names=raw_feature_names,
            n_features_to_select=n_features_to_select,
            random_state=random_state + fold_idx
        )
        selected_features_per_fold.append(selected_names)
        for fn in selected_names:
            feature_selection_counts[fn] = feature_selection_counts.get(fn, 0) + 1

        X_train_sel = X_train_trans[:, rfe_mask]
        X_val_sel = X_val_trans[:, rfe_mask]

        # 4. Optional SMOTE Resampling for Severe Class Imbalance
        if apply_smote:
            try:
                from imblearn.over_sampling import SMOTE
                smote = SMOTE(random_state=random_state + fold_idx, k_neighbors=3)
                # Resample on CAD target to maintain spatial consistency
                X_train_sel, y_cad_res = smote.fit_resample(X_train_sel, y_train_dict['CAD'])
                # Adjust training dictionary
                y_train_dict_adj = {'CAD': y_cad_res}
                for t in ['LAD', 'LCX', 'RCA']:
                    # Resample each target fold if needed
                    smote_t = SMOTE(random_state=random_state + fold_idx, k_neighbors=3)
                    _, y_t_res = smote_t.fit_resample(X_train_trans[:, rfe_mask], y_train_dict[t])
                    y_train_dict_adj[t] = y_t_res[:len(X_train_sel)]
                y_train_dict = y_train_dict_adj
            except Exception as e:
                # If SMOTE fails or single minority count is too small, fallback to Focal Loss
                pass

        # 5. Inner Calibration Split (ensures calibration parameters are fit out-of-training)
        n_train_samples = len(X_train_sel)
        inner_perm = np.random.RandomState(random_state + fold_idx).permutation(n_train_samples)
        calib_size = max(6, int(n_train_samples * 0.20))
        sub_calib_idx = inner_perm[:calib_size]
        sub_train_idx = inner_perm[calib_size:]

        X_sub_train = X_train_sel[sub_train_idx]
        y_sub_train = {t: y_train_dict[t][sub_train_idx] for t in TARGET_TASKS}

        X_sub_calib = X_train_sel[sub_calib_idx]
        y_sub_calib = {t: y_train_dict[t][sub_calib_idx] for t in TARGET_TASKS}

        # 6. Train PyTorch Multitask Neural Network with Focal Loss
        model = MultitaskCardiacModel(
            input_dim=n_features_to_select,
            epochs=epochs,
            lr=0.003,
            focal_gamma=2.0,
            focal_alpha=0.65,
            calibration_method=calibration_method,
            random_state=random_state + fold_idx
        )
        model.fit(
            X=X_sub_train,
            y=y_sub_train,
            eval_set=(X_sub_calib, y_sub_calib)
        )

        # 7. Predict Calibrated Probabilities on Validation Fold
        val_probs = model.predict_proba(X_val_sel)
        val_preds = model.predict(X_val_sel, threshold=0.5)

        # 8. Compute Clinical Metrics for Each Branch Head
        for task in TARGET_TASKS:
            m = calculate_metrics(
                y_true=y_val_dict[task],
                y_pred=val_preds[task],
                y_proba=val_probs[task]
            )
            fold_metrics_by_target[task].append(m)

        cad_f1 = fold_metrics_by_target['CAD'][-1]['F1-Score']
        cad_auc = fold_metrics_by_target['CAD'][-1]['ROC-AUC']
        print(f"  Fold {fold_idx + 1}/{n_splits} Complete | CAD F1: {cad_f1:.4f} | CAD ROC-AUC: {cad_auc:.4f}")

    # Aggregate Metrics Across Folds (Mean ± Std)
    summary_by_target = {}
    for task in TARGET_TASKS:
        metrics_list = fold_metrics_by_target[task]
        agg = {}
        for key in metrics_list[0].keys():
            vals = [fold[key] for fold in metrics_list if not np.isnan(fold[key])]
            if vals:
                agg[key] = (float(np.mean(vals)), float(np.std(vals)))
            else:
                agg[key] = (np.nan, np.nan)
        summary_by_target[task] = agg

    return {
        'summary_by_target': summary_by_target,
        'fold_metrics_by_target': fold_metrics_by_target,
        'selected_features_per_fold': selected_features_per_fold,
        'feature_selection_counts': feature_selection_counts
    }


def format_multitask_metrics_table(cv_summary: Dict[str, Any]) -> pd.DataFrame:
    """
    Formats the Multitask CV evaluation summary into a publication-ready comparison table.
    """
    rows = []
    summary_dict = cv_summary['summary_by_target']

    for task in TARGET_TASKS:
        task_metrics = summary_dict[task]
        row = {'Task / Head': task}
        for metric_name in ['Accuracy', 'Precision', 'Recall (Sensitivity)', 'Specificity', 'F1-Score', 'ROC-AUC', 'Brier Score (Calibration)', 'ECE (Calibration Error)']:
            if metric_name in task_metrics:
                mean_val, std_val = task_metrics[metric_name]
                row[metric_name] = f"{mean_val:.4f} ± {std_val:.4f}"
        rows.append(row)

    return pd.DataFrame(rows)


def evaluate_with_stratified_cv(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42
) -> Dict[str, Tuple[float, float]]:
    """
    Performs leakage-free Stratified K-Fold cross validation on full dataset for classical pipelines.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }

    scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, return_train_score=False)

    summary = {
        'Accuracy': (float(np.mean(scores['test_accuracy'])), float(np.std(scores['test_accuracy']))),
        'Precision': (float(np.mean(scores['test_precision'])), float(np.std(scores['test_precision']))),
        'Recall': (float(np.mean(scores['test_recall'])), float(np.std(scores['test_recall']))),
        'F1-Score': (float(np.mean(scores['test_f1'])), float(np.std(scores['test_f1']))),
        'ROC-AUC': (float(np.mean(scores['test_roc_auc'])), float(np.std(scores['test_roc_auc'])))
    }
    return summary


def format_metrics_table(results_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Converts nested evaluation results into a clean formatted comparison DataFrame.
    """
    rows = []
    for model_name, metrics in results_dict.items():
        row = {'Model': model_name}
        for k, v in metrics.items():
            if isinstance(v, tuple):
                row[f"{k} (CV Mean±Std)"] = f"{v[0]:.4f} ± {v[1]:.4f}"
            elif isinstance(v, float):
                row[k] = round(v, 4)
            else:
                row[k] = v
        rows.append(row)
    return pd.DataFrame(rows)
