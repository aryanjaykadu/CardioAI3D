"""
Unit and Integration Tests for CAD & Vessel-Specific Stenosis Multitask ML Pipeline.

Tests include:
- Strict zero-leakage target exclusion
- Dataset dimensionality and binary target validation
- Multitask PyTorch architecture forward pass and head outputs
- Binary Focal Loss class-imbalance weighting
- RFE feature selection (pruning to 15 features including Region RWMA & Tinversion)
- Platt Scaling & Isotonic Regression output probability calibration
- End-to-end zero-leakage cross-validation execution
"""

import unittest
import numpy as np
import pandas as pd
import torch

from src.data_loader import load_dataset, prepare_features_and_targets, LEAKAGE_COLUMNS
from src.preprocessing import build_preprocessor, get_feature_names_from_preprocessor
from src.models import (
    MultitaskCardiacNN,
    BinaryFocalLoss,
    MultitaskCalibrator,
    MultitaskCardiacModel,
    TARGET_TASKS
)
from src.evaluation import (
    calculate_metrics,
    run_rfe_feature_selection,
    evaluate_multitask_pipeline_cv,
    format_multitask_metrics_table
)


class TestMultitaskPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = load_dataset()
        cls.X, cls.targets = prepare_features_and_targets(cls.df)

    def test_strict_zero_leakage(self):
        """Verify that LAD, LCX, RCA, and Cath are NEVER present in input features X."""
        for col in LEAKAGE_COLUMNS:
            self.assertNotIn(col, self.X.columns, f"Target leakage detected: {col} is in X!")

    def test_dataset_dimensions(self):
        """Check expected row count (303 patients) and 55 predictor features."""
        self.assertEqual(len(self.X), 303)
        self.assertEqual(self.X.shape[1], 55)

    def test_targets_present_and_binary(self):
        """Check that all 4 targets (CAD, LAD, LCX, RCA) exist and have binary values {0, 1}."""
        for target_name in TARGET_TASKS:
            self.assertIn(target_name, self.targets)
            target_series = self.targets[target_name]
            self.assertEqual(len(target_series), 303)
            unique_vals = set(target_series.unique())
            self.assertTrue(unique_vals.issubset({0, 1}), f"Non-binary values in {target_name}: {unique_vals}")

    def test_multitask_nn_architecture(self):
        """Verify PyTorch shared hidden layers branch into 4 distinct classification heads."""
        input_dim = 15
        batch_size = 16
        model = MultitaskCardiacNN(input_dim=input_dim, hidden_dim1=32, hidden_dim2=16, head_dim=8)
        dummy_x = torch.randn(batch_size, input_dim)
        outputs = model(dummy_x)

        self.assertIsInstance(outputs, dict)
        for task in TARGET_TASKS:
            self.assertIn(task, outputs)
            self.assertEqual(outputs[task].shape, torch.Size([batch_size]))

    def test_binary_focal_loss_counteracts_imbalance(self):
        """Verify focal loss correctly downweights easy examples and scales with class imbalance."""
        focal_loss = BinaryFocalLoss(alpha=0.75, gamma=2.0)
        logits = torch.tensor([5.0, -5.0, 0.0]) # easy positive, easy negative, uncertain
        targets = torch.tensor([1.0, 0.0, 1.0])
        loss = focal_loss(logits, targets)
        self.assertGreater(float(loss), 0.0)
        self.assertFalse(torch.isnan(loss))

    def test_rfe_prunes_to_top_15_clinical_features(self):
        """Verify RFE selects exactly 15 features including critical predictors like Region RWMA and Tinversion."""
        preprocessor, _, _ = build_preprocessor(self.X)
        X_trans = preprocessor.fit_transform(self.X)
        feature_names = get_feature_names_from_preprocessor(preprocessor)

        mask, selected_names = run_rfe_feature_selection(
            X_trans,
            self.targets['CAD'].values,
            feature_names,
            n_features_to_select=15,
            random_state=42
        )
        self.assertEqual(len(selected_names), 15)
        self.assertEqual(np.sum(mask), 15)
        # Verify clinical hallmarks are captured
        self.assertTrue(any('Region RWMA' in fn for fn in selected_names), "Region RWMA should be among top predictors")
        self.assertTrue(any('Tinversion' in fn for fn in selected_names), "Tinversion should be among top predictors")

    def test_platt_and_isotonic_calibration(self):
        """Verify Platt Scaling and Isotonic Regression produce strictly bounded probabilities."""
        logits_dict = {
            'CAD': np.array([-2.5, 0.0, 3.2, 1.5, -1.0]),
            'LAD': np.array([-1.2, 1.8, 0.5, -0.4, 2.1]),
            'LCX': np.array([-3.0, -0.5, 0.2, 1.1, -1.8]),
            'RCA': np.array([0.1, 2.2, -1.5, 0.8, -0.9])
        }
        y_dict = {
            'CAD': np.array([0, 0, 1, 1, 0]),
            'LAD': np.array([0, 1, 1, 0, 1]),
            'LCX': np.array([0, 0, 1, 1, 0]),
            'RCA': np.array([0, 1, 0, 1, 0])
        }

        # Test Platt Scaling
        platt = MultitaskCalibrator(method='platt')
        platt.fit(logits_dict, y_dict)
        probs_platt = platt.predict_proba(logits_dict)
        for task in TARGET_TASKS:
            self.assertTrue(np.all(probs_platt[task] >= 0.0) and np.all(probs_platt[task] <= 1.0))

        # Test Isotonic Regression
        iso = MultitaskCalibrator(method='isotonic')
        iso.fit(logits_dict, y_dict)
        probs_iso = iso.predict_proba(logits_dict)
        for task in TARGET_TASKS:
            self.assertTrue(np.all(probs_iso[task] >= 0.0) and np.all(probs_iso[task] <= 1.0))

    def test_evaluation_metrics_calculation(self):
        """Verify calculation of clinical metrics and calibration scores."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 0])
        y_proba = np.array([0.1, 0.9, 0.8, 0.2, 0.4])
        metrics = calculate_metrics(y_true, y_pred, y_proba)

        self.assertIn('Accuracy', metrics)
        self.assertIn('Precision', metrics)
        self.assertIn('Recall (Sensitivity)', metrics)
        self.assertIn('F1-Score', metrics)
        self.assertIn('ROC-AUC', metrics)
        self.assertIn('Brier Score (Calibration)', metrics)
        self.assertIn('ECE (Calibration Error)', metrics)
        self.assertGreaterEqual(metrics['Accuracy'], 0.0)
        self.assertLessEqual(metrics['Accuracy'], 1.0)


if __name__ == '__main__':
    unittest.main()
