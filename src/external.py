"""
External validation of the BEDSIDE model: leave-one-site-out (LOSO) across 5 hospitals.

For each hospital H: impute / scale / tune C / choose the >=95%-sensitivity rule-out
threshold using ONLY the other four hospitals, then score every patient of H once.
This is the closest thing to "deploy it in a hospital it has never seen" that the
public data allow.
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src.benchmark import (_tuned_lr, bootstrap, brier, ece, fast_auc, rule_out_stats,
                           rule_out_threshold)
from src.harmonize import (BEDSIDE_FEATURES, SITES, bedside_matrix, esc_symptom_ptp,
                           training_medians)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class BedsideLR:
    name = "Bedside logistic regression (7 inputs)"

    def fit(self, df, seed):
        self.med = training_medians(df)
        self.m = make_pipeline(StandardScaler(), _tuned_lr(seed)).fit(bedside_matrix(df, self.med).values, df["cad"].values)
        return self

    def predict(self, df):
        return self.m.predict_proba(bedside_matrix(df, self.med).values)[:, 1]


class BedsideGB:
    name = "Bedside gradient boosting (7 inputs)"

    def fit(self, df, seed):
        self.med = training_medians(df)
        self.m = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=2, subsample=0.8,
                                            random_state=seed).fit(bedside_matrix(df, self.med).values, df["cad"].values)
        return self

    def predict(self, df):
        return self.m.predict_proba(bedside_matrix(df, self.med).values)[:, 1]


class ESCTable:
    name = "ESC 2019 pre-test probability (guideline)"

    def fit(self, df, seed):
        return self

    def predict(self, df):
        return esc_symptom_ptp(df)


MODELS = {"bedside_lr": BedsideLR, "bedside_gb": BedsideGB, "esc": ESCTable}


def calibration_slope_intercept(y, p):
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    slope = LogisticRegression(C=1e6, max_iter=1000).fit(lp.reshape(-1, 1), y).coef_[0, 0]
    return float(slope), float(np.mean(p) - np.mean(y))  # slope, calibration-in-the-large (pred - observed)


def _inner_threshold(cls, df, seed, target):
    oof = np.zeros(len(df))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(df, df["site"] + df["cad"].astype(str)):
        oof[va] = cls().fit(df.iloc[tr], seed).predict(df.iloc[va])
    return rule_out_threshold(df["cad"].values, oof, target)


def loso(df: pd.DataFrame, seed: int, target_sens: float, B: int, log=print) -> Dict:
    preds = {k: np.full(len(df), np.nan) for k in MODELS}
    ruled = {k: np.zeros(len(df), dtype=int) for k in MODELS}
    thresholds = {k: {} for k in MODELS}
    for site in SITES:
        te = (df["site"] == site).values
        train = df[~te].reset_index(drop=True)
        for key, cls in MODELS.items():
            model = cls().fit(train, seed)
            preds[key][te] = model.predict(df[te])
            thr = _inner_threshold(cls, train, seed, target_sens)
            thresholds[key][site] = thr
            ruled[key][te] = (preds[key][te] < thr).astype(int)
        log(f"    LOSO {site:11s} n={te.sum():3d}  bedside LR AUROC {fast_auc(df['cad'].values[te], preds['bedside_lr'][te]):.3f}"
            f"  ESC {fast_auc(df['cad'].values[te], preds['esc'][te]):.3f}")

    y = df["cad"].values
    per_site: List[Dict] = []
    for site in SITES + ["All sites (pooled)"]:
        idx = np.ones(len(df), bool) if site.startswith("All") else (df["site"] == site).values
        yy = y[idx]
        row = {"site": site, "n": int(idx.sum()), "prevalence": float(yy.mean()), "models": {}}
        for key in MODELS:
            pp = preds[key][idx]
            slope, citl = calibration_slope_intercept(yy, pp)
            f = lambda ii, pp=pp, yy=yy: fast_auc(yy[ii], pp[ii])
            ro = rule_out_stats(yy, ruled[key][idx])
            row["models"][key] = {
                "auroc": fast_auc(yy, pp), "auroc_ci": list(np.nanpercentile(bootstrap(f, len(yy), B, seed), [2.5, 97.5])),
                "brier": brier(yy, pp), "ece": ece(yy, pp), "cal_slope": slope, "cal_in_large": citl,
                "threshold": thresholds[key].get(site), **ro,
            }
        per_site.append(row)

    # paired ΔAUROC (bedside LR - ESC) per site
    for row in per_site:
        idx = np.ones(len(df), bool) if row["site"].startswith("All") else (df["site"] == row["site"]).values
        yy, a, b = y[idx], preds["bedside_lr"][idx], preds["esc"][idx]
        f = lambda ii: fast_auc(yy[ii], a[ii]) - fast_auc(yy[ii], b[ii])
        bs = bootstrap(f, len(yy), B, seed + 7)
        row["delta_lr_vs_esc"] = {"delta": f(np.arange(len(yy))), "ci": list(np.nanpercentile(bs, [2.5, 97.5]))}
    return {"per_site": per_site, "preds": preds, "ruled": ruled,
            "model_names": {k: v.name for k, v in MODELS.items()}}


def pooled_cv(df: pd.DataFrame, seed: int, repeats: int, folds: int) -> Dict:
    """Pooled 5-site CV (stratified by site x outcome) for the bedside LR, for comparison with LOSO."""
    y, strata = df["cad"].values, (df["site"] + df["cad"].astype(str)).values
    aucs, P = [], np.zeros((repeats, len(df)))
    for r in range(repeats):
        for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed + 1000 * r).split(df, strata):
            P[r, te] = BedsideLR().fit(df.iloc[tr], seed).predict(df.iloc[te])
        aucs.append(fast_auc(y, P[r]))
    return {"auroc": float(np.mean(aucs)), "auroc_per_repeat": aucs}


def local_adaptation(df: pd.DataFrame, seed: int, ks=(0, 25, 50, 100), draws: int = 20) -> List[Dict]:
    """
    'How many local patients does a new hospital need?'  For each held-out site, add k
    randomly drawn local patients (with labels) to the other sites' training data, refit
    the bedside LR, and score the remaining local patients. Repeated `draws` times.
    """
    rows = []
    for site in SITES:
        te_all = np.where(df["site"].values == site)[0]
        other = df[df["site"] != site]
        for k in ks:
            neg_left = (len(te_all) - k) * (1 - df['cad'].values[te_all].mean())
            if neg_left < 10:  # too few non-CAD patients left to estimate AUROC
                continue
            aucs = []
            for d in range(1 if k == 0 else draws):
                rng = np.random.default_rng(seed + 31 * d + k)
                loc = rng.choice(te_all, size=k, replace=False) if k else np.array([], int)
                rest = np.setdiff1d(te_all, loc)
                train = pd.concat([other, df.iloc[loc]], ignore_index=True)
                y = df["cad"].values[rest]
                if len(np.unique(y)) < 2:
                    continue
                aucs.append(fast_auc(y, BedsideLR().fit(train, seed).predict(df.iloc[rest])))
            rows.append({"site": site, "k_local": int(k), "auroc_mean": float(np.mean(aucs)),
                         "auroc_p10": float(np.percentile(aucs, 10)), "auroc_p90": float(np.percentile(aucs, 90)), "draws": len(aucs)})
    return rows
