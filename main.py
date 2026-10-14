"""
Main Machine Learning Pipeline for Coronary Artery Disease and Vessel Stenosis Prediction.
Dataset: Extension of Z-Alizadeh Sani Dataset (UCI ID 411).

Key Pipeline Features:
1. Multitask Learning (MTL) Neural Network using PyTorch:
   - Shared representation trunk branching into 4 heads: CAD, LAD, LCX, RCA.
2. Strict Zero Data Leakage:
   - ['LAD', 'LCX', 'RCA', 'Cath'] are explicitly excluded from feature matrix X.
   - Preprocessing is fitted ONLY on training folds.
3. Within-fold Recursive Feature Elimination (RFE):
   - Pruning inputs down to top 15 most critical clinical predictors (e.g. Region RWMA, Tinversion).
4. Class Imbalance Mitigation:
   - Dynamic Focal Loss with task-specific frequency weighting (and SMOTE option).
5. Output Probability Calibration:
   - Platt Scaling (logistic calibration) and Isotonic Regression for strictly calibrated stenosis risks.
"""

import os
import sys

# Ensure UTF-8 output encoding across Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.calibration import calibration_curve

from src.data_loader import load_dataset, prepare_features_and_targets, LEAKAGE_COLUMNS
from src.preprocessing import build_preprocessor, get_feature_names_from_preprocessor
from src.models import (
    MultitaskCardiacModel,
    MultitaskCardiacNN,
    get_candidate_models,
    TARGET_TASKS
)
from src.evaluation import (
    calculate_metrics,
    run_rfe_feature_selection,
    evaluate_multitask_pipeline_cv,
    format_multitask_metrics_table,
    evaluate_with_stratified_cv,
    format_metrics_table
)
from src.interpretability import (
    explain_with_lime,
    compute_cohort_lime_importance,
    explain_with_shap,
    get_model_feature_importance
)


def ensure_directories():
    os.makedirs("reports/figures", exist_ok=True)
    os.makedirs("reports/tables", exist_ok=True)


def plot_multitask_calibration_curves(
    y_test_dict: dict,
    uncalib_probs: dict,
    calib_probs: dict,
    output_dir: str = "reports/figures"
):
    """
    Plots Reliability Diagrams comparing raw uncalibrated probabilities against
    Platt Scaled / Isotonic calibrated probabilities for all 4 heads.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, task in enumerate(TARGET_TASKS):
        ax = axes[idx]
        y_true = y_test_dict[task]

        # Calibration curve for uncalibrated
        fop_uncal, mp_uncal = calibration_curve(y_true, uncalib_probs[task], n_bins=8, strategy='uniform')
        # Calibration curve for calibrated
        fop_cal, mp_cal = calibration_curve(y_true, calib_probs[task], n_bins=8, strategy='uniform')

        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', alpha=0.6)
        ax.plot(mp_uncal, fop_uncal, marker='o', lw=1.8, color='crimson', label='Raw Softmax/Sigmoid')
        ax.plot(mp_cal, fop_cal, marker='s', lw=2.2, color='royalblue', label='Platt Scaled (Calibrated)')

        ax.set_title(f"Reliability Curve: {task} Prediction Head", fontsize=12, fontweight='bold')
        ax.set_xlabel("Mean Predicted Probability", fontsize=10)
        ax.set_ylabel("Empirical True Fraction", fontsize=10)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "multitask_calibration_curves.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Saved calibration reliability curves: {save_path}")


def plot_multitask_roc_curves(
    y_test_dict: dict,
    probs_dict: dict,
    output_dir: str = "reports/figures"
):
    """
    Plots multi-head ROC curves for CAD, LAD, LCX, and RCA on the test holdout.
    """
    plt.figure(figsize=(8, 7))
    palette = {'CAD': '#1f77b4', 'LAD': '#d62728', 'LCX': '#ff7f0e', 'RCA': '#2ca02c'}

    for task in TARGET_TASKS:
        y_true = y_test_dict[task]
        y_prob = probs_dict[task]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        score = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2.2, color=palette[task], label=f'{task} Head (AUC = {score:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Baseline (AUC = 0.500)')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=11)
    plt.title('Multitask PyTorch Architecture: Multi-Head ROC Curves', fontsize=13, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "multitask_roc_curves.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Saved multitask ROC curves: {save_path}")


def plot_multitask_confusion_matrices(
    y_test_dict: dict,
    preds_dict: dict,
    output_dir: str = "reports/figures"
):
    """
    Plots 2x2 confusion matrix heatmaps for all 4 task heads.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for idx, task in enumerate(TARGET_TASKS):
        ax = axes[idx]
        cm = confusion_matrix(y_test_dict[task], preds_dict[task])
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            ax=ax,
            xticklabels=['Normal (0)', 'Disease/Stenotic (1)'],
            yticklabels=['Normal (0)', 'Disease/Stenotic (1)'],
            cbar=False
        )
        ax.set_title(f'Confusion Matrix: {task} Head', fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=9)
        ax.set_ylabel('True Label', fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "multitask_confusion_matrices.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Saved multitask confusion matrices: {save_path}")


def run_pipeline():
    print("=" * 80)
    print("CARDIOAI MULTITASK LEARNING (MTL) CLINICAL DECISION SUPPORT PIPELINE")
    print("Dataset: Extension of Z-Alizadeh Sani (303 patients, 59 attributes)")
    print("Architecture: PyTorch Shared Trunk -> 4 Branched Heads (CAD, LAD, LCX, RCA)")
    print("=" * 80)

    ensure_directories()

    # 1. Load Data
    print("\n[Step 1] Loading clinical dataset...")
    df = load_dataset()
    print(f"Loaded raw dataset with shape: {df.shape}")

    # 2. Strict Target Leakage Safeguard
    print("\n[Step 2] Applying Strict Target Leakage Guard...")
    print(f"Columns strictly dropped from predictor features X: {LEAKAGE_COLUMNS}")
    X, targets = prepare_features_and_targets(df)
    print(f"Feature matrix X shape: {X.shape} (55 predictor clinical features)")
    for col in LEAKAGE_COLUMNS:
        assert col not in X.columns, f"LEAKAGE DETECTED: {col} found in feature matrix X!"
    print("VERIFICATION CONFIRMED: Zero target leakage. LAD, LCX, RCA, and Cath fully excluded.")

    # Print Class Distribution
    print("\nTarget Class Distributions:")
    for t_name in TARGET_TASKS:
        t_series = targets[t_name]
        pos_cnt = int(t_series.sum())
        neg_cnt = int(len(t_series) - pos_cnt)
        pct = (pos_cnt / len(t_series)) * 100
        print(f"  - {t_name:4s}: Positive={pos_cnt:3d} ({pct:.1f}%), Negative={neg_cnt:3d} ({100-pct:.1f}%)")

    # 3. 5-Fold Zero-Leakage Multitask Cross-Validation Pipeline
    print("\n" + "=" * 80)
    print("[Step 3] EXECUTING 5-FOLD ZERO-LEAKAGE MULTITASK CROSS-VALIDATION")
    print("  * Preprocessor fitted ONLY on training folds")
    print("  * RFE pruning inputs down to top 15 clinical predictors inside each fold")
    print("  * Dynamic Focal Loss counteracting class imbalance")
    print("  * Platt Scaling probability calibration on inner validation splits")
    print("=" * 80)

    cv_results = evaluate_multitask_pipeline_cv(
        X=X,
        targets=targets,
        n_splits=5,
        n_features_to_select=15,
        apply_smote=False,
        calibration_method='platt',
        epochs=80,
        random_state=42
    )

    df_cv_table = format_multitask_metrics_table(cv_results)
    cv_table_path = "reports/tables/multitask_cv_metrics.csv"
    df_cv_table.to_csv(cv_table_path, index=False)
    print(f"\n5-Fold Multitask Cross-Validation Summary (Mean ± Std):")
    print(df_cv_table.to_string(index=False))
    print(f"Saved CV table: {cv_table_path}")

    # Top features selected by RFE across CV folds
    print("\nTop Predictors Selected by RFE across CV Folds:")
    rfe_freqs = sorted(cv_results['feature_selection_counts'].items(), key=lambda x: x[1], reverse=True)
    df_rfe = pd.DataFrame(rfe_freqs, columns=['Clinical_Feature', 'Fold_Selection_Count'])
    df_rfe.to_csv("reports/tables/top_15_rfe_features.csv", index=False)
    for feat, count in rfe_freqs[:15]:
        print(f"  - {feat:25s}: {count}/5 Folds")

    # 4. Holdout Train/Test Split Evaluation
    print("\n" + "=" * 80)
    print("[Step 4] HOLDOUT TRAIN/TEST SPLIT (80% Train, 20% Holdout Test)")
    print("=" * 80)

    X_train_raw, X_test_raw, y_train_cad, y_test_cad = train_test_split(
        X, targets['CAD'], test_size=0.20, stratify=targets['CAD'], random_state=42
    )
    y_train_dict = {t: targets[t].loc[X_train_raw.index].values for t in TARGET_TASKS}
    y_test_dict = {t: targets[t].loc[X_test_raw.index].values for t in TARGET_TASKS}

    # Fit preprocessor on X_train_raw only
    preprocessor, _, _ = build_preprocessor(X_train_raw)
    X_train_trans = preprocessor.fit_transform(X_train_raw)
    X_test_trans = preprocessor.transform(X_test_raw)
    raw_feature_names = get_feature_names_from_preprocessor(preprocessor)

    # RFE Pruning on X_train_trans only to top 15 predictors
    rfe_mask, selected_feature_names = run_rfe_feature_selection(
        X_train_trans=X_train_trans,
        y_train_cad=y_train_dict['CAD'],
        feature_names=raw_feature_names,
        n_features_to_select=15,
        random_state=42
    )
    print(f"Top 15 Clinical Predictors Selected for Final Model:")
    for i, fn in enumerate(selected_feature_names, 1):
        print(f"  {i:2d}. {fn}")

    X_train_sel = X_train_trans[:, rfe_mask]
    X_test_sel = X_test_trans[:, rfe_mask]

    # Train Multitask Cardiac Model with Platt Scaling
    mtl_model = MultitaskCardiacModel(
        input_dim=15,
        hidden_dim1=48,
        hidden_dim2=24,
        head_dim=16,
        epochs=100,
        lr=0.003,
        focal_gamma=2.0,
        calibration_method='platt',
        calib_split_ratio=0.20,
        random_state=42
    )
    mtl_model.fit(X_train_sel, y_train_dict)

    # Predictions: Calibrated Probabilities and Raw Logits
    calibrated_probs = mtl_model.predict_proba(X_test_sel)
    raw_logits = mtl_model.predict_raw_logits(X_test_sel)
    uncalibrated_probs = {k: 1.0 / (1.0 + np.exp(-np.clip(v, -15.0, 15.0))) for k, v in raw_logits.items()}
    predictions = mtl_model.predict(X_test_sel, threshold=0.5)

    # Holdout Metrics
    holdout_rows = []
    for task in TARGET_TASKS:
        m = calculate_metrics(
            y_true=y_test_dict[task],
            y_pred=predictions[task],
            y_proba=calibrated_probs[task]
        )
        row = {'Task / Head': task}
        row.update({k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()})
        holdout_rows.append(row)

    df_holdout = pd.DataFrame(holdout_rows)
    holdout_path = "reports/tables/multitask_holdout_test_metrics.csv"
    df_holdout.to_csv(holdout_path, index=False)
    print(f"\nHoldout Test Performance Across All 4 Heads (Calibrated):")
    print(df_holdout[['Task / Head', 'Accuracy', 'Precision', 'Recall (Sensitivity)', 'Specificity', 'F1-Score', 'ROC-AUC', 'Brier Score (Calibration)', 'ECE (Calibration Error)']].to_string(index=False))

    # Generate Visualization Figures
    print("\n[Step 5] Generating Multitask Visualization Figures...")
    plot_multitask_calibration_curves(y_test_dict, uncalibrated_probs, calibrated_probs)
    plot_multitask_roc_curves(y_test_dict, calibrated_probs)
    plot_multitask_confusion_matrices(y_test_dict, predictions)

    # 6. Benchmark Single-Task Baselines for Comparison
    print("\n" + "=" * 80)
    print("[Step 6] BENCHMARKING CLASSICAL SINGLE-TASK BASELINE MODELS")
    print("=" * 80)
    for target_name in TARGET_TASKS:
        y_train_single = targets[target_name].loc[X_train_raw.index]
        y_test_single = targets[target_name].loc[X_test_raw.index]
        candidate_models = get_candidate_models(random_state=42)

        test_results = {}
        for m_name, model in candidate_models.items():
            if m_name == 'Multitask Neural Network (MTL)':
                continue
            pipe = Pipeline([('preprocessor', preprocessor), ('clf', model)])
            pipe.fit(X_train_raw, y_train_single)
            y_pred = pipe.predict(X_test_raw)
            y_prob = pipe.predict_proba(X_test_raw)[:, 1] if hasattr(pipe, 'predict_proba') else y_pred
            test_results[m_name] = calculate_metrics(y_test_single.values, y_pred, y_prob)

        df_bench = pd.DataFrame(test_results).T
        df_bench.index.name = 'Model'
        df_bench.to_csv(f"reports/tables/{target_name}_test_metrics.csv")

    print("\n" + "=" * 80)
    print("ALL PIPELINE STAGES COMPLETED SUCCESSFULLY!")
    print("Multitask Learning Architecture is fully trained, cross-validated, calibrated, and documented.")
    print("=" * 80)

    return mtl_model, cv_results, df_holdout


if __name__ == "__main__":
    run_pipeline()
