"""
Multitask Learning (MTL) Neural Network and Model Factory Module.

Implements a shared representation PyTorch architecture branching into 4 distinct
classification heads for simultaneous coronary artery disease (CAD) and vessel-specific
stenosis (LAD, LCX, RCA) prediction.

Includes:
- MultitaskCardiacNN: PyTorch neural network with shared hidden layers and 4 task heads
- BinaryFocalLoss: Class-imbalance counteracting focal loss
- MultitaskCalibrator: Platt Scaling and Isotonic Regression output probability calibration
- MultitaskCardiacModel: Scikit-learn compatible estimator wrapper
- get_candidate_models: Classical baseline model factory for comparative benchmarking
"""

import os
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


TARGET_TASKS = ['CAD', 'LAD', 'LCX', 'RCA']


class MultitaskCardiacNN(nn.Module):
    """
    Multitask Learning (MTL) Neural Network for Cardiac Decision Support.

    Architecture:
    1. Shared Trunk:
       - Input Layer (15 clinical predictors pruned via RFE)
       - Dense(input_dim -> hidden_dim1) + BatchNorm1d + ReLU + Dropout
       - Dense(hidden_dim1 -> hidden_dim2) + BatchNorm1d + ReLU + Dropout
    2. Four Task-Specific Classification Heads:
       - CAD Head: Dense(hidden_dim2 -> 16) -> ReLU -> Dropout -> Dense(16 -> 1)
       - LAD Head: Dense(hidden_dim2 -> 16) -> ReLU -> Dropout -> Dense(16 -> 1)
       - LCX Head: Dense(hidden_dim2 -> 16) -> ReLU -> Dropout -> Dense(16 -> 1)
       - RCA Head: Dense(hidden_dim2 -> 16) -> ReLU -> Dropout -> Dense(16 -> 1)
    """

    def __init__(
        self,
        input_dim: int = 15,
        hidden_dim1: int = 48,
        hidden_dim2: int = 24,
        head_dim: int = 16,
        dropout_rate: float = 0.25
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.head_dim = head_dim
        self.dropout_rate = dropout_rate

        # Shared representation trunk (LayerNorm guarantees batch-size independence for single-sample inference)
        self.shared_trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.LayerNorm(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.LayerNorm(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # 4 Independent Branching Classification Heads
        self.head_cad = nn.Sequential(
            nn.Linear(hidden_dim2, head_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(head_dim, 1)
        )
        self.head_lad = nn.Sequential(
            nn.Linear(hidden_dim2, head_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(head_dim, 1)
        )
        self.head_lcx = nn.Sequential(
            nn.Linear(hidden_dim2, head_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(head_dim, 1)
        )
        self.head_rca = nn.Sequential(
            nn.Linear(hidden_dim2, head_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(head_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through shared trunk and all 4 classification heads.

        Returns:
            Dictionary of unnormalized logits for CAD, LAD, LCX, RCA (shape: (batch,))
        """
        features = self.shared_trunk(x)
        logits = {
            'CAD': self.head_cad(features).squeeze(-1),
            'LAD': self.head_lad(features).squeeze(-1),
            'LCX': self.head_lcx(features).squeeze(-1),
            'RCA': self.head_rca(features).squeeze(-1)
        }
        return logits


class BinaryFocalLoss(nn.Module):
    """
    Numerically stable Binary Focal Loss to counteract severe class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    where p_t is model probability for the true class.
    """

    def __init__(
        self,
        alpha: float = 0.65,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_weight = alpha_t * torch.pow((1.0 - p_t).clamp(min=1e-6), self.gamma)
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class MultitaskCalibrator:
    """
    Calibrates predicted logits or raw probabilities for each task head
    using either Platt Scaling (logistic calibration) or Isotonic Regression.
    """

    def __init__(self, method: str = 'platt'):
        if method not in ('platt', 'isotonic'):
            raise ValueError(f"Unsupported calibration method: {method}. Choose 'platt' or 'isotonic'.")
        self.method = method
        self.calibrators: Dict[str, Any] = {}

    def fit(self, logits_dict: Dict[str, np.ndarray], y_dict: Dict[str, np.ndarray]):
        """
        Fits calibration model on validation logits and true binary labels.
        """
        for task in TARGET_TASKS:
            if task not in logits_dict or task not in y_dict:
                continue
            z = logits_dict[task].reshape(-1, 1)
            y = np.asarray(y_dict[task]).ravel().astype(int)

            if len(np.unique(y)) < 2:
                # Degenerate case (only 1 class present in calibration split)
                self.calibrators[task] = None
                continue

            if self.method == 'platt':
                # Platt Scaling: Univariate logistic regression on uncalibrated logits
                lr = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
                lr.fit(z, y)
                # Enforce monotonicity: higher raw logit must correspond to higher probability
                if lr.coef_[0, 0] > 0:
                    self.calibrators[task] = lr
                else:
                    self.calibrators[task] = None
            elif self.method == 'isotonic':
                # Isotonic Regression on uncalibrated sigmoid probabilities
                raw_probs = 1.0 / (1.0 + np.exp(-np.clip(z.ravel(), -15.0, 15.0)))
                iso = IsotonicRegression(out_of_bounds='clip')
                iso.fit(raw_probs, y)
                self.calibrators[task] = iso

    def predict_proba(self, logits_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Transforms raw logits into calibrated posterior probabilities in range [0, 1].
        """
        calibrated_probs = {}
        for task in TARGET_TASKS:
            if task not in logits_dict:
                continue
            z = logits_dict[task].reshape(-1, 1)
            calibrator = self.calibrators.get(task, None)

            if calibrator is not None:
                if self.method == 'platt':
                    p = calibrator.predict_proba(z)[:, 1]
                else:
                    raw_p = 1.0 / (1.0 + np.exp(-np.clip(z.ravel(), -15.0, 15.0)))
                    p = calibrator.predict(raw_p)
            else:
                # Fallback to standard sigmoid
                p = 1.0 / (1.0 + np.exp(-np.clip(z.ravel(), -15.0, 15.0)))

            calibrated_probs[task] = np.clip(p, 1e-4, 1.0 - 1e-4)

        return calibrated_probs


class MultitaskCardiacModel(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible estimator wrapping the PyTorch Multitask Neural Network.

    Features:
    - Shared representation trunk + 4 distinct heads (CAD, LAD, LCX, RCA)
    - Focal Loss to counteract severe class imbalance
    - Integrated Platt Scaling / Isotonic Regression probability calibration
    - Standard fit, predict, predict_proba interfaces
    """

    def __init__(
        self,
        input_dim: int = 15,
        hidden_dim1: int = 48,
        hidden_dim2: int = 24,
        head_dim: int = 16,
        dropout_rate: float = 0.25,
        lr: float = 0.003,
        weight_decay: float = 1e-4,
        batch_size: int = 32,
        epochs: int = 120,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.65,
        calibration_method: str = 'platt',
        calib_split_ratio: float = 0.20,
        task_weights: Optional[Dict[str, float]] = None,
        random_state: int = 42,
        device: str = 'cpu'
    ):
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.head_dim = head_dim
        self.dropout_rate = dropout_rate
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.calibration_method = calibration_method
        self.calib_split_ratio = calib_split_ratio
        self.task_weights = task_weights or {'CAD': 1.0, 'LAD': 1.0, 'LCX': 1.0, 'RCA': 1.0}
        self.random_state = random_state
        self.device = device

        self.net: Optional[MultitaskCardiacNN] = None
        self.calibrator: Optional[MultitaskCalibrator] = None
        self.training_history: List[Dict[str, float]] = []

    def _init_network(self, input_dim: int):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        self.net = MultitaskCardiacNN(
            input_dim=input_dim,
            hidden_dim1=self.hidden_dim1,
            hidden_dim2=self.hidden_dim2,
            head_dim=self.head_dim,
            dropout_rate=self.dropout_rate
        ).to(self.device)

    def _prepare_targets(
        self,
        y: Union[pd.DataFrame, Dict[str, pd.Series], np.ndarray]
    ) -> Dict[str, np.ndarray]:
        if isinstance(y, dict):
            return {k: np.asarray(v, dtype=np.float32).ravel() for k, v in y.items()}
        elif isinstance(y, pd.DataFrame):
            return {col: y[col].values.astype(np.float32) for col in TARGET_TASKS if col in y.columns}
        elif isinstance(y, np.ndarray):
            if y.ndim == 2 and y.shape[1] == 4:
                return {task: y[:, i].astype(np.float32) for i, task in enumerate(TARGET_TASKS)}
            elif y.ndim == 1:
                return {'CAD': y.astype(np.float32)}
        raise ValueError("y must be a Dict, DataFrame with target tasks, or 2D ndarray of shape (N, 4).")

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[pd.DataFrame, Dict[str, pd.Series], np.ndarray],
        eval_set: Optional[Tuple[Any, Any]] = None
    ):
        """
        Trains the Multitask Learning Neural Network with Focal Loss and fits
        the output calibrator on a held-out calibration partition.
        """
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        if isinstance(X, pd.DataFrame):
            X_arr = X.values.astype(np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)

        y_dict = self._prepare_targets(y)
        n_samples, actual_input_dim = X_arr.shape

        self._init_network(actual_input_dim)

        # Split calibration set if not externally provided
        if eval_set is not None:
            X_train, y_train_dict = X_arr, y_dict
            X_calib, y_calib_raw = eval_set
            if isinstance(X_calib, pd.DataFrame):
                X_calib = X_calib.values.astype(np.float32)
            y_calib_dict = self._prepare_targets(y_calib_raw)
        elif self.calib_split_ratio > 0.0 and n_samples >= 20:
            np.random.seed(self.random_state)
            indices = np.random.permutation(n_samples)
            calib_size = max(4, int(n_samples * self.calib_split_ratio))
            calib_idx = indices[:calib_size]
            train_idx = indices[calib_size:]

            X_train = X_arr[train_idx]
            X_calib = X_arr[calib_idx]
            y_train_dict = {k: v[train_idx] for k, v in y_dict.items()}
            y_calib_dict = {k: v[calib_idx] for k, v in y_dict.items()}
        else:
            X_train = X_arr
            X_calib = X_arr
            y_train_dict = y_dict
            y_calib_dict = y_dict

        # PyTorch Tensors & DataLoader
        t_X_train = torch.tensor(X_train, dtype=torch.float32)
        t_y_cad = torch.tensor(y_train_dict.get('CAD', np.zeros(len(X_train))), dtype=torch.float32)
        t_y_lad = torch.tensor(y_train_dict.get('LAD', np.zeros(len(X_train))), dtype=torch.float32)
        t_y_lcx = torch.tensor(y_train_dict.get('LCX', np.zeros(len(X_train))), dtype=torch.float32)
        t_y_rca = torch.tensor(y_train_dict.get('RCA', np.zeros(len(X_train))), dtype=torch.float32)

        dataset = TensorDataset(t_X_train, t_y_cad, t_y_lad, t_y_lcx, t_y_rca)
        loader = DataLoader(dataset, batch_size=min(self.batch_size, len(dataset)), shuffle=True)

        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-5)
        # Dynamic Focal Loss configured with task-specific inverse class frequencies
        task_loss_fns = {}
        for task in TARGET_TASKS:
            if task in y_train_dict:
                pos_cnt = float(np.sum(y_train_dict[task] == 1))
                neg_cnt = float(np.sum(y_train_dict[task] == 0))
                alpha_val = float(np.clip(neg_cnt / max(1.0, pos_cnt + neg_cnt), 0.25, 0.85))
                task_loss_fns[task] = BinaryFocalLoss(alpha=alpha_val, gamma=self.focal_gamma)
            else:
                task_loss_fns[task] = BinaryFocalLoss(alpha=self.focal_alpha, gamma=self.focal_gamma)

        self.net.train()
        self.training_history = []

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_x, b_cad, b_lad, b_lcx, b_rca in loader:
                batch_x = batch_x.to(self.device)
                targets_batch = {
                    'CAD': b_cad.to(self.device),
                    'LAD': b_lad.to(self.device),
                    'LCX': b_lcx.to(self.device),
                    'RCA': b_rca.to(self.device)
                }

                optimizer.zero_grad()
                logits = self.net(batch_x)

                total_loss = 0.0
                for task in TARGET_TASKS:
                    if task in targets_batch:
                        l = task_loss_fns[task](logits[task], targets_batch[task])
                        total_loss += self.task_weights.get(task, 1.0) * l

                total_loss.backward()
                optimizer.step()
                epoch_loss += total_loss.item() * len(batch_x)

            scheduler.step()
            epoch_loss /= len(X_train)
            self.training_history.append({'epoch': epoch + 1, 'loss': epoch_loss})

        # Fit probability calibrator on calibration partition
        self.net.eval()
        with torch.no_grad():
            t_X_calib = torch.tensor(X_calib, dtype=torch.float32).to(self.device)
            raw_logits_calib = self.net(t_X_calib)
            logits_calib_dict = {k: v.cpu().numpy() for k, v in raw_logits_calib.items()}

        self.calibrator = MultitaskCalibrator(method=self.calibration_method)
        self.calibrator.fit(logits_calib_dict, y_calib_dict)
        return self

    def predict_raw_logits(self, X: Union[np.ndarray, pd.DataFrame]) -> Dict[str, np.ndarray]:
        """Returns raw uncalibrated network logits for all 4 heads."""
        if self.net is None:
            raise RuntimeError("Model has not been fitted yet.")
        if isinstance(X, pd.DataFrame):
            X_arr = X.values.astype(np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)

        self.net.eval()
        with torch.no_grad():
            t_X = torch.tensor(X_arr, dtype=torch.float32).to(self.device)
            logits = self.net(t_X)
            return {k: v.cpu().numpy() for k, v in logits.items()}

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> Dict[str, np.ndarray]:
        """
        Returns strictly calibrated stenosis probabilities [0, 1] for all 4 heads.
        """
        logits_dict = self.predict_raw_logits(X)
        if self.calibrator is not None:
            return self.calibrator.predict_proba(logits_dict)
        else:
            return {k: 1.0 / (1.0 + np.exp(-np.clip(v, -15.0, 15.0))) for k, v in logits_dict.items()}

    def predict_task_proba(self, X: Union[np.ndarray, pd.DataFrame], task: str) -> np.ndarray:
        """
        Returns calibrated binary class probability matrix (N, 2) for a specific task.
        Compatible with scikit-learn metrics and explainability tools.
        """
        all_probs = self.predict_proba(X)
        pos_probs = all_probs[task]
        neg_probs = 1.0 - pos_probs
        return np.column_stack([neg_probs, pos_probs])

    def predict(self, X: Union[np.ndarray, pd.DataFrame], threshold: float = 0.5) -> Dict[str, np.ndarray]:
        """Returns binary predictions {0, 1} for all 4 heads at decision threshold."""
        probs = self.predict_proba(X)
        return {k: (v >= threshold).astype(int) for k, v in probs.items()}


def get_candidate_models(random_state: int = 42) -> Dict[str, BaseEstimator]:
    """
    Returns a dictionary of candidate classification estimators.
    Retains classical baselines for benchmarking alongside the Multitask Architecture.
    """
    models = {
        'Multitask Neural Network (MTL)': MultitaskCardiacModel(
            random_state=random_state,
            calibration_method='platt'
        ),
        'Logistic Regression': LogisticRegression(
            C=0.5,
            solver='liblinear',
            max_iter=1000,
            random_state=random_state
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=150,
            max_depth=6,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=random_state
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            random_state=random_state
        ),
        'Support Vector Machine': SVC(
            C=1.0,
            kernel='rbf',
            probability=True,
            class_weight='balanced',
            random_state=random_state
        ),
        'K-Nearest Neighbors': KNeighborsClassifier(
            n_neighbors=7,
            weights='distance'
        ),
        'Multi-Layer Perceptron': MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            alpha=0.01,
            max_iter=500,
            early_stopping=True,
            random_state=random_state
        )
    }
    return models
