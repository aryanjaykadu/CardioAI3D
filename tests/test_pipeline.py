"""Fast checks for the CardioAI pipeline.   Run:  python -m pytest -q"""

import json
import os

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.baselines import esc2019_ptp
from src.benchmark import (ModelSpec, cross_validate_spec, fast_auc, fit_lr,
                           rule_out_stats, rule_out_threshold)
from src.features import ALL_FEATURES, TARGET_COLUMNS, encode_features, encode_targets

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def data():
    df = pd.read_csv(os.path.join(ROOT, "data", "z_alizadeh_sani_extended.csv"))
    return df, encode_features(df), encode_targets(df)


def test_shapes_and_no_target_columns(data):
    df, X, Y = data
    assert X.shape == (303, 55)
    assert list(X.columns) == ALL_FEATURES
    for c in TARGET_COLUMNS:
        assert c not in X.columns
    assert not X.isnull().any().any()
    assert all(set(np.unique(v)) == {0, 1} for v in Y.values())


def test_cad_label_matches_vessels(data):
    _, _, Y = data
    any_vessel = Y["LAD"] | Y["LCX"] | Y["RCA"]
    assert (Y["CAD"] == any_vessel).mean() > 0.99  # one record in the source data disagrees


def test_esc_table_spot_values(data):
    rows = pd.DataFrame([
        {"Age": 55, "Sex": 1, "Typical Chest Pain": 1, "Atypical": 0, "Nonanginal": 0, "Dyspnea": 0},
        {"Age": 65, "Sex": 0, "Typical Chest Pain": 0, "Atypical": 1, "Nonanginal": 0, "Dyspnea": 0},
        {"Age": 72, "Sex": 1, "Typical Chest Pain": 0, "Atypical": 0, "Nonanginal": 0, "Dyspnea": 1},
    ])
    assert np.allclose(esc2019_ptp(rows), [0.32, 0.11, 0.32])


def test_fast_auc_matches_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = np.round(rng.random(200), 2)  # include ties
    assert abs(fast_auc(y, p) - roc_auc_score(y, p)) < 1e-12


def test_rule_out_threshold_reaches_target_sensitivity():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    p = np.clip(0.3 * y + rng.random(500) * 0.7, 0, 1)
    t = rule_out_threshold(y, p, 0.95)
    assert rule_out_stats(y, (p < t).astype(int))["sensitivity"] >= 0.95


def test_cv_signal_and_negative_control(data):
    _, X, Y = data
    spec = ModelSpec("t", "lr", lambda Xa, Ya, s: fit_lr(Xa, Ya, s, tasks=["CAD"]), ["CAD"], "")
    real = cross_validate_spec(spec, X, Y, 1, 5, 42, tune_threshold=False)
    assert fast_auc(Y["CAD"], real["oof"]["CAD"][0]) > 0.85
    perm = np.random.default_rng(42).permutation(len(X))
    Yp = {k: v[perm] for k, v in Y.items()}
    null = cross_validate_spec(spec, X, Yp, 1, 5, 42, tune_threshold=False)
    assert abs(fast_auc(Yp["CAD"], null["oof"]["CAD"][0]) - 0.5) < 0.12


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "dashboard", "data", "model.json")), reason="run export_model.py first")
def test_exported_model_reproduces_sklearn(data):
    _, X, Y = data
    m = json.load(open(os.path.join(ROOT, "dashboard", "data", "model.json")))
    t = m["tasks"]["CAD"]
    z = t["intercept"] + ((X.values - t["mean"]) / t["scale"]) @ np.array(t["coef"])
    ref = fit_lr(X, Y, 42, tasks=["CAD"]).predict(X)["CAD"]
    assert np.allclose(1 / (1 + np.exp(-z)), ref, atol=1e-6)


def test_dashboard_assets_exist():
    for f in ["index.html", "app.js", "styles.css", "lib/three.module.js", "lib/three.core.js",
              "lib/OrbitControls.js", "assets/logo.svg"]:
        assert os.path.exists(os.path.join(ROOT, "dashboard", f)), f


# ---------------------------------------------------------------- multi-centre bedside model
from src.harmonize import BEDSIDE_FEATURES, SITES, bedside_matrix, load_harmonised, training_medians


def test_harmonised_sites_and_labels():
    df = load_harmonised()
    assert list(df["site"].unique()) == SITES
    assert df.groupby("site").size().to_dict() == {"Tehran": 303, "Cleveland": 303, "Budapest": 294, "Zurich": 123, "Long Beach": 200}
    assert set(df["cad"].unique()) == {0, 1}
    # chest-pain categories are mutually exclusive
    assert ((df["cp_typical"] + df["cp_atypical"] + df["cp_nonanginal"]) <= 1).all()


def test_bedside_matrix_imputes_with_training_medians_only():
    df = load_harmonised()
    train, test = df[df["site"] != "Long Beach"], df[df["site"] == "Long Beach"]
    med = training_medians(train)
    Xt = bedside_matrix(test, med)
    assert list(Xt.columns) == BEDSIDE_FEATURES and not Xt.isnull().any().any()
    miss = test["sbp"].isna().values
    assert miss.sum() > 0 and (Xt["sbp"].values[miss] == med["sbp"]).all() and (Xt["sbp_missing"].values[miss] == 1).all()


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "dashboard", "data", "bedside.json")), reason="run export_model.py first")
def test_exported_bedside_model_matches_python():
    from src.external import BedsideLR
    df = load_harmonised()
    b = json.load(open(os.path.join(ROOT, "dashboard", "data", "bedside.json")))
    X = bedside_matrix(df, b["medians"]).values
    z = b["intercept"] + ((X - b["mean"]) / b["scale"]) @ np.array(b["coef"])
    assert np.allclose(1 / (1 + np.exp(-z)), BedsideLR().fit(df, 42).predict(df), atol=1e-6)
