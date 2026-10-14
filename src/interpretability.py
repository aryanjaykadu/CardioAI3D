"""
Clinical Interpretability Module (SHAP & LIME).

Provides local and global feature attribution for CAD status and vessel stenosis
predictions to facilitate clinical decision support and trust.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import List, Dict, Any, Optional


def explain_with_lime(
    model,
    X_train_transformed: np.ndarray,
    X_test_transformed: np.ndarray,
    feature_names: List[str],
    patient_idx: int = 0,
    class_names: List[str] = ['Normal', 'Disease/Stenotic'],
    output_dir: str = "reports/figures"
) -> Dict[str, Any]:
    """
    Computes LIME tabular explanation for a specific patient.
    
    Returns:
        Dictionary containing patient explanation weights and prediction probabilities.
    """
    os.makedirs(output_dir, exist_ok=True)
    try:
        import lime
        import lime.lime_tabular

        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train_transformed,
            feature_names=feature_names,
            class_names=class_names,
            mode='classification',
            random_state=42
        )

        patient_vector = X_test_transformed[patient_idx]
        exp = explainer.explain_instance(
            data_row=patient_vector,
            predict_fn=model.predict_proba,
            num_features=10
        )

        explanation_list = exp.as_list()
        pred_probs = model.predict_proba(patient_vector.reshape(1, -1))[0]

        # Save figure
        try:
            fig = exp.as_pyplot_figure(label=1)
            plt.title(f"LIME Patient Explanation (Patient #{patient_idx}) - {class_names[1]}", fontsize=11, fontweight='bold')
            plt.tight_layout()
            fig_path = os.path.join(output_dir, f"lime_patient_{patient_idx}_{class_names[1]}.png")
            plt.savefig(fig_path, dpi=300)
            plt.close()
        except Exception as e_plot:
            print(f"LIME plot note: {e_plot}")

        return {
            'success': True,
            'method': 'LIME',
            'patient_idx': patient_idx,
            'predicted_class': class_names[int(np.argmax(pred_probs))],
            'probabilities': {class_names[0]: float(pred_probs[0]), class_names[1]: float(pred_probs[1])},
            'top_features': explanation_list
        }
    except Exception as e:
        print(f"LIME explanation note: {e}")
        return {'success': False, 'error': str(e)}


def compute_cohort_lime_importance(
    model,
    X_train_transformed: np.ndarray,
    X_sample_transformed: np.ndarray,
    feature_names: List[str],
    class_names: List[str] = ['Normal', 'Disease/Stenotic'],
    sample_size: int = 25,
    output_dir: str = "reports/figures"
) -> pd.DataFrame:
    """
    Aggregates local LIME feature attributions across a cohort of patients
    to derive a global clinical importance profile.
    """
    os.makedirs(output_dir, exist_ok=True)
    try:
        import lime
        import lime.lime_tabular

        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train_transformed,
            feature_names=feature_names,
            class_names=class_names,
            mode='classification',
            random_state=42
        )

        importance_sums = {fn: 0.0 for fn in feature_names}
        counts = {fn: 0 for fn in feature_names}

        n_samples = min(sample_size, len(X_sample_transformed))
        for i in range(n_samples):
            exp = explainer.explain_instance(
                data_row=X_sample_transformed[i],
                predict_fn=model.predict_proba,
                num_features=15
            )
            for feat_expr, weight in exp.as_list():
                for orig_feat in feature_names:
                    if orig_feat in feat_expr:
                        importance_sums[orig_feat] += abs(weight)
                        counts[orig_feat] += 1
                        break

        avg_importances = []
        for feat in feature_names:
            c = counts[feat]
            avg_imp = importance_sums[feat] / c if c > 0 else 0.0
            avg_importances.append({'Feature': feat, 'Mean_Abs_LIME_Weight': avg_imp})

        df_imp = pd.DataFrame(avg_importances).sort_values(by='Mean_Abs_LIME_Weight', ascending=False)

        # Plot cohort LIME
        if not df_imp.empty:
            plt.figure(figsize=(10, 6))
            top_15 = df_imp.head(15).sort_values(by='Mean_Abs_LIME_Weight', ascending=True)
            plt.barh(top_15['Feature'], top_15['Mean_Abs_LIME_Weight'], color='#ff7f0e')
            plt.title(f"LIME Cohort Feature Importance - {class_names[1]}", fontsize=13, fontweight='bold')
            plt.xlabel("Mean Absolute LIME Weight (Impact on Odds)")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"lime_cohort_{class_names[1]}.png"), dpi=300)
            plt.close()

        return df_imp
    except Exception as e:
        print(f"Error computing cohort LIME importances: {e}")
        return pd.DataFrame()


def explain_with_shap(
    model,
    X_train_transformed: np.ndarray,
    X_test_transformed: np.ndarray,
    feature_names: List[str],
    target_name: str,
    output_dir: str = "reports/figures"
) -> Optional[pd.DataFrame]:
    """
    Computes TreeSHAP or KernelSHAP feature importance when available.
    """
    os.makedirs(output_dir, exist_ok=True)
    try:
        import shap
        # Check model type
        if hasattr(model, 'tree_') or hasattr(model, 'estimators_'):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(X_test_transformed) if hasattr(explainer, '__call__') else explainer.shap_values(X_test_transformed)
        elif hasattr(model, 'coef_'):
            # Analytical LinearExplainer for Logistic Regression
            explainer = shap.LinearExplainer(model, X_train_transformed)
            shap_values = explainer(X_test_transformed) if hasattr(explainer, '__call__') else explainer.shap_values(X_test_transformed)
        else:
            background = shap.sample(X_train_transformed, 20)
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values = explainer.shap_values(X_test_transformed[:10])

        # Handle Explanation object vs raw numpy array/list
        if hasattr(shap_values, 'values'):
            sv = shap_values.values
        else:
            sv = shap_values

        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]
        elif isinstance(sv, np.ndarray) and len(sv.shape) == 3:
            sv = sv[:, :, 1]

        mean_abs_shap = np.mean(np.abs(sv), axis=0)
        df_shap = pd.DataFrame({
            'Feature': feature_names,
            'Mean_Abs_SHAP': mean_abs_shap
        }).sort_values(by='Mean_Abs_SHAP', ascending=False)

        # Plot summary
        plt.figure(figsize=(10, 6))
        top_15 = df_shap.head(15).sort_values(by='Mean_Abs_SHAP', ascending=True)
        plt.barh(top_15['Feature'], top_15['Mean_Abs_SHAP'], color='#1f77b4')
        plt.title(f"SHAP Global Feature Importance - {target_name}", fontsize=13, fontweight='bold')
        plt.xlabel("Mean |SHAP Value| (Impact on Model Output)")
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"shap_importance_{target_name}.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()

        return df_shap
    except Exception as e:
        print(f"SHAP explanation note: {e}")
        return None


def get_model_feature_importance(
    model,
    feature_names: List[str],
    target_name: str,
    output_dir: str = "reports/figures"
) -> pd.DataFrame:
    """
    Extracts native tree feature importances or linear odds-ratio coefficients
    as an interpretable baseline.
    """
    os.makedirs(output_dir, exist_ok=True)
    importances = None
    metric_label = "Importance"

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        metric_label = "Gini Feature Importance"
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
        metric_label = "Absolute Logistic Coefficient |Beta|"

    if importances is not None:
        df_imp = pd.DataFrame({
            'Feature': feature_names,
            metric_label: importances
        }).sort_values(by=metric_label, ascending=False)

        # Plot top 15
        plt.figure(figsize=(10, 6))
        top_15 = df_imp.head(15).sort_values(by=metric_label, ascending=True)
        plt.barh(top_15['Feature'], top_15[metric_label], color='#2ca02c')
        plt.title(f"Top 15 Predictors for {target_name} ({metric_label})", fontsize=13, fontweight='bold')
        plt.xlabel(metric_label)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"top_features_{target_name}.png"), dpi=300)
        plt.close()

        return df_imp

    return pd.DataFrame()
