"""
Model zoo + leakage-free repeated cross-validation used by evaluate.py.

Every model is wrapped as `spec.fit(X_train, Y_train) -> predictor`, where the
predictor has `.predict(X) -> {task: P(positive)}`. All learned steps
(standardisation, feature selection, hyper-parameter search, calibration,
decision threshold) happen inside the training fold only.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.baselines import esc2019_ptp
from src.features import TASKS

CLINICAL_FEATURES = [
    "Age", "Sex", "Typical Chest Pain", "Atypical", "Nonanginal", "Dyspnea",
    "DM", "HTN", "Current Smoker", "FH", "DLP",
]
LR_CS = np.logspace(-3, 1, 9)


# --------------------------------------------------------------------------- models
class _SklearnPerTask:
    def __init__(self, models: Dict[str, object], cols: List[str]):
        self.models, self.cols = models, cols

    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        Xc = X[self.cols].values
        return {t: m.predict_proba(Xc)[:, 1] for t, m in self.models.items()}


def _tuned_lr(seed: int):
    return LogisticRegressionCV(
        Cs=LR_CS, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
        scoring="neg_log_loss", max_iter=5000, l1_ratios=(0.0,), use_legacy_attributes=False,
    )


def fit_lr(X, Y, seed, cols=None, tasks=TASKS):
    cols = cols or list(X.columns)
    models = {t: make_pipeline(StandardScaler(), _tuned_lr(seed)).fit(X[cols].values, Y[t]) for t in tasks}
    return _SklearnPerTask(models, cols)


class _RFEPerTask:
    def __init__(self, models, masks, cols):
        self.models, self.masks, self.cols = models, masks, cols

    def predict(self, X):
        Xc = X[self.cols].values
        return {t: m.predict_proba(Xc[:, self.masks[t]])[:, 1] for t, m in self.models.items()}


def rfe_select(X_scaled: np.ndarray, y: np.ndarray, k: int, seed: int) -> np.ndarray:
    est = LogisticRegression(C=0.5, solver="liblinear", max_iter=2000, random_state=seed)
    return RFE(est, n_features_to_select=k, step=1).fit(X_scaled, y).support_


def fit_lr_rfe(X, Y, seed, k=15, tasks=TASKS):
    cols = list(X.columns)
    Xs = StandardScaler().fit_transform(X.values)
    models, masks = {}, {}
    for t in tasks:
        masks[t] = rfe_select(Xs, Y[t], k, seed)
        models[t] = make_pipeline(StandardScaler(), _tuned_lr(seed)).fit(X.values[:, masks[t]], Y[t])
    return _RFEPerTask(models, masks, cols)


def fit_rf(X, Y, seed, tasks=TASKS):
    models = {t: RandomForestClassifier(
        n_estimators=500, min_samples_leaf=3, max_features="sqrt",
        class_weight="balanced", random_state=seed, n_jobs=1).fit(X.values, Y[t]) for t in tasks}
    return _SklearnPerTask(models, list(X.columns))


def fit_gb(X, Y, seed, tasks=TASKS):
    models = {t: GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.05, max_depth=2, subsample=0.8,
        random_state=seed).fit(X.values, Y[t]) for t in tasks}
    return _SklearnPerTask(models, list(X.columns))


class _ESC:
    def predict(self, X):
        return {"CAD": esc2019_ptp(X)}


def fit_esc(X, Y, seed, tasks=TASKS):
    return _ESC()


class _MTL:
    def __init__(self, scaler, mask, model, cols):
        self.scaler, self.mask, self.model, self.cols = scaler, mask, model, cols

    def predict(self, X):
        Z = self.scaler.transform(X[self.cols].values)[:, self.mask]
        return {t: v for t, v in self.model.predict_proba(Z.astype(np.float32)).items()}


def fit_mtl(X, Y, seed, k=15, tasks=TASKS, epochs=100):
    from src.models import MultitaskCardiacModel  # torch import only when needed
    cols = list(X.columns)
    scaler = StandardScaler().fit(X.values)
    Xs = scaler.transform(X.values)
    mask = rfe_select(Xs, Y["CAD"], k, seed)
    model = MultitaskCardiacModel(
        input_dim=k, hidden_dim1=48, hidden_dim2=24, head_dim=16, epochs=epochs,
        lr=0.003, focal_gamma=2.0, calibration_method="platt", calib_split_ratio=0.20,
        random_state=seed,
    ).fit(Xs[:, mask].astype(np.float32), {t: Y[t] for t in TASKS})
    return _MTL(scaler, mask, model, cols)


@dataclass
class ModelSpec:
    key: str
    name: str
    fit: Callable
    tasks: List[str]
    note: str


def model_zoo(include_mtl: bool = True) -> List[ModelSpec]:
    zoo = [
        ModelSpec("esc", "ESC 2019 pre-test probability (guideline)", fit_esc, ["CAD"],
                  "Age, sex, symptom type lookup table. No training."),
        ModelSpec("clin_lr", "Clinical logistic regression (bedside risk factors)",
                  lambda X, Y, s: fit_lr(X, Y, s, cols=CLINICAL_FEATURES), TASKS,
                  "11 history/symptom variables, L2, C tuned in-fold."),
        ModelSpec("lr", "Logistic regression (all 55 features)", fit_lr, TASKS,
                  "L2, C tuned by inner 5-fold CV on log-loss."),
        ModelSpec("lr_rfe", "Logistic regression (RFE top-15)", fit_lr_rfe, TASKS,
                  "Per-task RFE to 15 features inside the fold, then tuned L2 LR."),
        ModelSpec("rf", "Random forest", fit_rf, TASKS, "500 trees, min leaf 3, balanced."),
        ModelSpec("gb", "Gradient boosting", fit_gb, TASKS, "150 stumps-depth-2, lr 0.05."),
    ]
    if include_mtl:
        zoo.append(ModelSpec("mtl", "Multitask neural net (RFE-15, focal loss, Platt)", fit_mtl, TASKS,
                             "Shared trunk + 4 heads, PyTorch, 100 epochs."))
    return zoo


# --------------------------------------------------------------------------- metrics
def fast_auc(y: np.ndarray, p: np.ndarray) -> float:
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(p)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ece(y, p, n_bins=10):
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    return float(sum((bins == b).mean() * abs(y[bins == b].mean() - p[bins == b].mean())
                     for b in range(n_bins) if (bins == b).any()))


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def rule_out_threshold(y: np.ndarray, p: np.ndarray, target_sens: float) -> float:
    """Largest threshold t such that predicting positive for p >= t keeps sensitivity >= target."""
    pos = np.sort(p[y == 1])
    k = int(np.floor((1 - target_sens) * len(pos)))
    return float(pos[k]) if len(pos) else 0.5


def rule_out_stats(y: np.ndarray, flagged_low: np.ndarray) -> Dict[str, float]:
    """flagged_low = 1 means the model says 'low risk, could defer angiography'."""
    tn = int(((flagged_low == 1) & (y == 0)).sum())
    fn = int(((flagged_low == 1) & (y == 1)).sum())
    tp = int(((flagged_low == 0) & (y == 1)).sum())
    fp = int(((flagged_low == 0) & (y == 0)).sum())
    n = len(y)
    return {
        "sensitivity": tp / max(1, tp + fn),
        "specificity": tn / max(1, tn + fp),
        "npv": tn / (tn + fn) if (tn + fn) else np.nan,
        "ruled_out_pct": 100 * (tn + fn) / n,
        "normal_caths_avoided_per_100": 100 * tn / n,
        "cad_missed_per_100": 100 * fn / n,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


# --------------------------------------------------------------------------- CV engine
def _inner_oof_cad(spec: ModelSpec, X: pd.DataFrame, Y: Dict[str, np.ndarray], seed: int, folds: int = 3) -> np.ndarray:
    oof = np.zeros(len(X))
    for tr, va in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, Y["CAD"]):
        m = spec.fit(X.iloc[tr], {t: v[tr] for t, v in Y.items()}, seed)
        oof[va] = m.predict(X.iloc[va])["CAD"]
    return oof


def cross_validate_spec(
    spec: ModelSpec,
    X: pd.DataFrame,
    Y: Dict[str, np.ndarray],
    repeats: int,
    folds: int,
    seed: int,
    target_sens: float = 0.95,
    tune_threshold: bool = True,
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, np.ndarray]:
    """
    Returns out-of-fold predictions with shape (repeats, n) per task, and the
    rule-out decision (1 = flagged low risk) for CAD with shape (repeats, n).
    """
    n = len(X)
    oof = {t: np.full((repeats, n), np.nan) for t in spec.tasks}
    ruled_out = np.zeros((repeats, n), dtype=int)
    thresholds = []
    for r in range(repeats):
        rs = seed + 1000 * r
        for f, (tr, te) in enumerate(StratifiedKFold(folds, shuffle=True, random_state=rs).split(X, Y["CAD"])):
            Xtr, Xte = X.iloc[tr], X.iloc[te]
            Ytr = {t: v[tr] for t, v in Y.items()}
            model = spec.fit(Xtr, Ytr, rs + f)
            pred = model.predict(Xte)
            for t in spec.tasks:
                oof[t][r, te] = pred[t]
            if tune_threshold:
                inner = _inner_oof_cad(spec, Xtr, Ytr, rs + f)
                thr = rule_out_threshold(Ytr["CAD"], inner, target_sens)
                thresholds.append(thr)
                ruled_out[r, te] = (pred["CAD"] < thr).astype(int)
        if log:
            log(f"    {spec.key:8s} repeat {r + 1}/{repeats}: CAD AUROC {fast_auc(Y['CAD'], oof['CAD'][r]):.3f}")
    return {"oof": oof, "ruled_out": ruled_out, "thresholds": np.array(thresholds)}


def bootstrap(fn: Callable[[np.ndarray], float], n: int, B: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.array([fn(rng.integers(0, n, n)) for _ in range(B)])
