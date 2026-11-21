"""
Export the deployed model and the honest per-patient predictions for the dashboard.

    python evaluate.py      # first: produces reports/eval/{results.json, oof_predictions.csv}
    python export_model.py  # then:  writes dashboard/data/{model.json, patients.json, evidence.json}

Deployed model: standardised L2 logistic regression on all 55 encoded features,
one per task (CAD, LAD, LCX, RCA), C chosen by 5-fold CV on log-loss, trained
on all 303 patients. Because it is linear in log-odds the dashboard can run it
exactly in the browser and show exact per-feature contributions.

Patient predictions shown in the dashboard are OUT-OF-FOLD (each patient scored
by models that never saw them, averaged over CV repeats), so the case browser
shows the error rate a new patient would see, including the misses.
"""

import json
import os
import shutil

import numpy as np
import pandas as pd

from src.baselines import esc2019_ptp
from src.benchmark import fit_lr
from src.features import (ALL_FEATURES, FEATURE_GROUPS, FEATURE_LABELS, TASKS,
                          encode_features, encode_targets)

DATA = os.path.join("data", "z_alizadeh_sani_extended.csv")
EVAL = os.path.join("reports", "eval")
OUT = os.path.join("dashboard", "data")
SEED = 42


def bootstrap_coefs(Xs, y, C, B, seed):
    """Refit the L2 LR (fixed C, fixed scaling) on B bootstrap resamples -> [intercept, *coef] rows."""
    from sklearn.linear_model import LogisticRegression
    rng, rows = np.random.default_rng(seed), []
    while len(rows) < B:
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        m = LogisticRegression(C=C, max_iter=5000).fit(Xs[i], y[i])
        rows.append([round(float(m.intercept_[0]), 4)] + [round(float(c), 4) for c in m.coef_[0]])
    return rows


def group_of(f):
    return next(g for g, cols in FEATURE_GROUPS.items() if f in cols)


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(DATA)
    X, Y = encode_features(df), encode_targets(df)
    results = json.load(open(os.path.join(EVAL, "results.json"), encoding="utf-8"))
    oof = pd.read_csv(os.path.join(EVAL, "oof_predictions.csv"))
    dep = results["config"]["deployed"]
    assert dep == "lr", "dashboard runs a logistic regression in the browser; deploy 'lr'"

    model = fit_lr(X, Y, SEED)
    feats = []
    for f in ALL_FEATURES:
        v = X[f]
        kind = "binary" if set(v.unique()) <= {0.0, 1.0} else ("ordinal" if f in ("Function Class", "Region RWMA", "VHD") else "continuous")
        feats.append({"key": f, "label": FEATURE_LABELS[f], "group": group_of(f), "kind": kind,
                      "min": float(v.min()), "max": float(v.max()), "median": float(v.median()),
                      "p01": float(v.quantile(0.01)), "p99": float(v.quantile(0.99))})
    tasks = {}
    for t in TASKS:
        pipe = model.models[t]
        sc, lr = pipe.named_steps["standardscaler"], pipe.named_steps["logisticregressioncv"]
        tasks[t] = {"intercept": float(lr.intercept_[0]), "coef": [float(c) for c in lr.coef_[0]],
                    "mean": [float(m) for m in sc.mean_], "scale": [float(s) for s in sc.scale_],
                    "C": float(lr.C_)}
        Xs = (X.values - sc.mean_) / sc.scale_
        tasks[t]["boot"] = bootstrap_coefs(Xs, Y[t], float(lr.C_), 200, SEED)
    dep_res = next(m for m in results["models"] if m["key"] == dep)
    threshold = dep_res["rule_out"]["threshold_median"]

    # Sanity check: JS formula == sklearn
    z = tasks["CAD"]["intercept"] + ((X.values - tasks["CAD"]["mean"]) / tasks["CAD"]["scale"]) @ np.array(tasks["CAD"]["coef"])
    assert np.allclose(1 / (1 + np.exp(-z)), model.predict(X)["CAD"], atol=1e-9)

    json.dump({"name": "CardioAI logistic regression", "features": feats, "tasks": tasks,
               "rule_out_threshold": threshold, "target_sensitivity": results["config"]["target_sens"],
               "trained_on": int(len(X))}, open(os.path.join(OUT, "model.json"), "w"), indent=1)

    esc = esc2019_ptp(X)
    patients = []
    for i in range(len(X)):
        patients.append({
            "row": i,
            "x": [round(float(v), 3) for v in X.iloc[i].values],
            "truth": {t: int(Y[t][i]) for t in TASKS},
            "oof": {t: round(float(oof.loc[i, f"{dep}_{t}"]), 4) for t in TASKS},
            "ruled_out_frac": round(float(oof.loc[i, f"{dep}_ruled_out_frac"]), 2),
            "esc": round(float(esc[i]), 3),
        })

    # Featured cases: chosen by rule from out-of-fold results (including failures).
    def pick(cond, key, reverse=False):
        c = [p for p in patients if cond(p)]
        c.sort(key=key, reverse=reverse)
        return c[0]["row"] if c else None
    featured = [
        ("Three-vessel disease, correctly flagged", pick(lambda p: p["truth"]["LAD"] and p["truth"]["LCX"] and p["truth"]["RCA"], lambda p: p["oof"]["CAD"], True)),
        ("Normal angiogram, correctly ruled out", pick(lambda p: p["truth"]["CAD"] == 0 and p["oof"]["CAD"] < threshold, lambda p: p["oof"]["CAD"])),
        ("Isolated LAD stenosis", pick(lambda p: p["truth"]["LAD"] and not p["truth"]["LCX"] and not p["truth"]["RCA"], lambda p: abs(p["oof"]["CAD"] - 0.8))),
        ("MISSED: CAD the model called low risk", pick(lambda p: p["truth"]["CAD"] == 1 and p["oof"]["CAD"] < threshold, lambda p: p["oof"]["CAD"])),
        ("False alarm: normal angiogram, high score", pick(lambda p: p["truth"]["CAD"] == 0, lambda p: p["oof"]["CAD"], True)),
    ]
    json.dump({"featured": [{"title": t, "row": r} for t, r in featured if r is not None],
               "patients": patients}, open(os.path.join(OUT, "patients.json"), "w"), separators=(",", ":"))

    # Threshold trade-off curve from out-of-fold predictions (for the interactive chart)
    y, p = Y["CAD"], oof[f"{dep}_CAD"].values
    pe = esc
    curve = []
    for thr in np.round(np.linspace(0.0, 1.0, 101), 2):
        for name, pp in (("model", p), ("esc", pe)):
            low = pp < thr
            curve.append({"src": name, "t": float(thr),
                          "sens": float(((~low) & (y == 1)).sum() / (y == 1).sum()),
                          "spec": float((low & (y == 0)).sum() / (y == 0).sum()),
                          "avoided": float(100 * (low & (y == 0)).sum() / len(y)),
                          "missed": float(100 * (low & (y == 1)).sum() / len(y))})
    evidence = {k: results[k] for k in ("config", "cohort", "models", "ablation", "subgroups", "negative_control", "external")}
    evidence["curve"] = curve
    json.dump(evidence, open(os.path.join(OUT, "evidence.json"), "w"), indent=1)
    shutil.copytree(os.path.join(EVAL, "figures"), os.path.join("dashboard", "figures"), dirs_exist_ok=True)
    print(f"model.json, patients.json ({len(patients)} patients, {len(featured)} featured), evidence.json -> {OUT}")
    print(f"Rule-out threshold (median of training-fold thresholds): {threshold:.3f}")
    for t in TASKS:
        top = sorted(zip(ALL_FEATURES, tasks[t]["coef"]), key=lambda a: -abs(a[1]))[:6]
        print(f"  {t}: C={tasks[t]['C']:.3g}; top: " + ", ".join(f"{f} {c:+.2f}" for f, c in top))


def export_bedside(results):
    """Bedside model (7 harmonised inputs, 5 hospitals): coefficients, bootstrap, and LOSO predictions."""
    from src.external import BedsideLR, _inner_threshold
    from src.harmonize import BEDSIDE_FEATURES, BEDSIDE_LABELS, SITES, bedside_matrix, load_harmonised, training_medians
    hdf = load_harmonised()
    model = BedsideLR().fit(hdf, SEED)
    sc, lr = model.m.named_steps["standardscaler"], model.m.named_steps["logisticregressioncv"]
    Xb = bedside_matrix(hdf, model.med)
    Xs = (Xb.values - sc.mean_) / sc.scale_
    z = lr.intercept_[0] + Xs @ lr.coef_[0]
    assert np.allclose(1 / (1 + np.exp(-z)), model.predict(hdf), atol=1e-9)
    thr = _inner_threshold(BedsideLR, hdf, SEED, results["config"]["target_sens"])
    ext = pd.read_csv(os.path.join(EVAL, "external_predictions.csv"))
    site_thr = {r["site"]: r["models"]["bedside_lr"]["threshold"] for r in results["external"]["per_site"] if r["site"] in SITES}
    cp = (hdf["cp_typical"] * 1 + hdf["cp_atypical"] * 2 + hdf["cp_nonanginal"] * 3).astype(int)
    pts = []
    for i, r in hdf.reset_index(drop=True).iterrows():
        loso = float(ext.loc[i, "loso_bedside_lr"])
        pts.append({"site": r["site"], "age": float(r["age"]), "male": int(r["male"]), "cp": int(cp[i]),
                    "sbp": None if pd.isna(r["sbp"]) else float(r["sbp"]), "fbs": None if pd.isna(r["fbs"]) else int(r["fbs"]),
                    "stt": int(r["ecg_stt"]), "lvh": int(r["ecg_lvh"]), "cad": int(r["cad"]),
                    "loso": round(loso, 4), "ruled": int(loso < site_thr[r["site"]]), "esc": round(float(ext.loc[i, "loso_esc"]), 3)})
    out = {"features": BEDSIDE_FEATURES, "labels": BEDSIDE_LABELS, "medians": model.med,
           "intercept": float(lr.intercept_[0]), "coef": [float(c) for c in lr.coef_[0]],
           "mean": [float(m) for m in sc.mean_], "scale": [float(v) for v in sc.scale_], "C": float(lr.C_),
           "boot": bootstrap_coefs(Xs, hdf["cad"].values, float(lr.C_), 200, SEED),
           "rule_out_threshold": thr, "site_thresholds": site_thr, "trained_on": int(len(hdf)), "sites": SITES, "patients": pts}
    json.dump(out, open(os.path.join(OUT, "bedside.json"), "w"), separators=(",", ":"))
    print(f"bedside.json: {len(pts)} patients from {len(SITES)} hospitals, threshold {thr:.3f}, C={lr.C_:.3g}")
    print("  coef: " + ", ".join(f"{f} {c:+.2f}" for f, c in zip(BEDSIDE_FEATURES, lr.coef_[0])))


if __name__ == "__main__":
    main()
    export_bedside(json.load(open(os.path.join(EVAL, "results.json"), encoding="utf-8")))
