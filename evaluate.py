"""
CardioAI evaluation: reproducible, leakage-free benchmark.
    python evaluate.py                 # full run (5 x 5-fold CV, all models), ~8-10 min on a laptop CPU
    python evaluate.py --quick         # 1 x 5-fold, no neural net, ~1 min (smoke test)
Question being answered
-----------------------
Among patients already referred for invasive coronary angiography, can routinely
collected clinical data identify people whose angiogram will be normal (no
vessel with >=50% stenosis), so the invasive test could be deferred, while
missing as few CAD patients as possible? And can it say WHICH vessel is narrowed?
Protocol
--------
* Repeated stratified K-fold CV (default 5 repeats x 5 folds, seed 42).
* Every learned step (scaling, RFE, C search, Platt calibration, decision threshold)
  is fit on the training fold only. The rule-out threshold is picked on inner
  3-fold out-of-fold predictions of the training fold to give >=95% sensitivity,
  then frozen and applied to the untouched test fold.
* Reference: ESC 2019 guideline pre-test probability table (age, sex, symptoms).
* 95% CIs: patient-level bootstrap (B=2000) over the pooled out-of-fold predictions.
Outputs: reports/eval/{results.json, summary.md, *.csv, figures/*.png}
"""
import argparse
import json
import os
import platform
import sys
import time
import numpy as np
import pandas as pd
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from src.benchmark import (ModelSpec, bootstrap, brier, cross_validate_spec, ece, fast_auc,
                           fit_lr, model_zoo, rule_out_stats)
from src.external import local_adaptation, loso, pooled_cv
from src.features import FEATURE_SETS, TASKS, encode_features, encode_targets
from src.harmonize import SITES, load_harmonised, site_summary
OUT = os.path.join("reports", "eval")
DATA = os.path.join("data", "z_alizadeh_sani_extended.csv")

def ci(a):
    a = a[~np.isnan(a)]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

def mean_over_repeats(fn, arr2d, idx=None):
    """Apply fn(y_idx, p_idx) to each repeat row, return the mean."""
    return float(np.nanmean([fn(idx, row) for row in arr2d]))

def summarise(spec, res, Y, B, seed):
    n = len(Y["CAD"])
    out = {"key": spec.key, "name": spec.name, "note": spec.note, "tasks": {}}
    for t in spec.tasks:
        P, y = res["oof"][t], Y[t]
        auc_fn = lambda idx: float(np.mean([fast_auc(y[idx], row[idx]) for row in P]))
        out["tasks"][t] = {
            "auroc": auc_fn(np.arange(n)),
            "auroc_ci": ci(bootstrap(auc_fn, n, B, seed)),
            "auroc_per_repeat": [fast_auc(y, row) for row in P],
            "brier": float(np.mean([brier(y, row) for row in P])),
            "ece": float(np.mean([ece(y, row) for row in P])),
        }
    # rule-out (CAD only)
    R, y = res["ruled_out"], Y["CAD"]
    per_rep = [rule_out_stats(y, row) for row in R]
    ro = {k: float(np.mean([d[k] for d in per_rep])) for k in per_rep[0]}
    for k in ["sensitivity", "specificity", "npv", "normal_caths_avoided_per_100", "cad_missed_per_100"]:
        ro[k + "_ci"] = ci(bootstrap(lambda idx: float(np.nanmean([rule_out_stats(y[idx], row[idx])[k] for row in R])), n, B // 4, seed + 1))
    ro["threshold_median"] = float(np.median(res["thresholds"])) if len(res["thresholds"]) else None
    out["rule_out"] = ro
    return out

def paired_delta(resA, resB, y, B, seed):
    """AUROC(A) - AUROC(B) on CAD with a paired bootstrap CI."""
    PA, PB = resA["oof"]["CAD"], resB["oof"]["CAD"]
    f = lambda idx: float(np.mean([fast_auc(y[idx], a[idx]) - fast_auc(y[idx], b[idx]) for a, b in zip(PA, PB)]))
    d = f(np.arange(len(y)))
    boots = bootstrap(f, len(y), B, seed)
    return {"delta": d, "ci": ci(boots), "p_le_0": float(np.mean(boots <= 0))}

def plot_figures(results, oofs, Y, X, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    os.makedirs(out_dir, exist_ok=True)
    colors = {"esc": "#8a8a8a", "clin_lr": "#c9a227", "lr": "#2a6fdb", "lr_rfe": "#0f9d8a",
              "rf": "#7b5cd6", "gb": "#d9534f", "mtl": "#e07b39"}
    y = Y["CAD"]
    # ROC (repeat 1)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for m in results["models"]:
        p = oofs[m["key"]]["oof"]["CAD"][0]
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, lw=2 if m["key"] in ("esc", "lr") else 1.3, color=colors[m["key"]],
                label=f"{m['name'].split(' (')[0]}  {m['tasks']['CAD']['auroc']:.3f}")
    ax.plot([0, 1], [0, 1], ls=":", color="#bbb", lw=1)
    ax.set_xlabel("1 − specificity"); ax.set_ylabel("Sensitivity")
    ax.set_title("CAD detection, out-of-fold ROC (AUROC = mean of 5 repeats)", fontsize=10)
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "roc_cad.png"), dpi=160); plt.close(fig)
    # Calibration
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for key in ("esc", "lr", "gb", "mtl"):
        if key not in oofs: continue
        p = oofs[key]["oof"]["CAD"].mean(0)
        q = np.quantile(p, np.linspace(0, 1, 8)); b = np.clip(np.searchsorted(q, p, side="right") - 1, 0, 6)
        xs = [p[b == i].mean() for i in range(7) if (b == i).any()]; ys = [y[b == i].mean() for i in range(7) if (b == i).any()]
        ax.plot(xs, ys, marker="o", lw=1.6, color=colors[key], label=next(m["name"].split(" (")[0] for m in results["models"] if m["key"] == key))
    ax.plot([0, 1], [0, 1], ls=":", color="#bbb")
    ax.set_xlabel("Predicted probability of CAD"); ax.set_ylabel("Observed fraction with CAD")
    ax.set_title("Calibration (septiles of out-of-fold predictions)", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "calibration_cad.png"), dpi=160); plt.close(fig)
    # Rule-out bars
    fig, ax = plt.subplots(figsize=(7, 4))
    names = [m["name"].split(" (")[0] for m in results["models"]]
    av = [m["rule_out"]["normal_caths_avoided_per_100"] for m in results["models"]]
    ms = [m["rule_out"]["cad_missed_per_100"] for m in results["models"]]
    yy = np.arange(len(names))
    ax.barh(yy - 0.18, av, 0.36, color="#0f9d8a", label="Normal angiograms avoided / 100 patients")
    ax.barh(yy + 0.18, ms, 0.36, color="#d9534f", label="CAD patients missed / 100 patients")
    ax.set_yticks(yy, names, fontsize=8); ax.invert_yaxis()
    ax.set_title("Rule-out at a threshold set on training folds for ≥95% sensitivity", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "rule_out.png"), dpi=160); plt.close(fig)

def write_markdown(res, path):
    NL = chr(10)
    L = []
    c, cfg = res["cohort"], res["config"]
    L.append("# CardioAI evaluation summary" + NL)
    L.append(f"Cohort: n={c['n']} cath-referred patients (Tehran); prevalence CAD {c['prevalence']['CAD']:.0%}, "
             f"LAD {c['prevalence']['LAD']:.0%}, LCX {c['prevalence']['LCX']:.0%}, RCA {c['prevalence']['RCA']:.0%}.  ")
    L.append(f"Protocol: {cfg['repeats']}x{cfg['folds']}-fold stratified CV, seed {cfg['seed']}, bootstrap B={cfg['bootstrap']}. "
             f"Runtime {cfg['runtime_s']:.0f}s." + NL)
    L.append("## Full model: CAD (any vessel >=50% stenosis at angiography)" + NL)
    L.append("| Model | AUROC (95% CI) | dAUROC vs ESC 2019 (95% CI) | Brier | ECE |")
    L.append("|---|---|---|---|---|")
    for m in res["models"]:
        t = m["tasks"]["CAD"]; d = m.get("delta_vs_esc")
        ds = f"{d['delta']:+.3f} ({d['ci'][0]:+.3f}, {d['ci'][1]:+.3f})" if d else "-"
        L.append(f"| {m['name']} | {t['auroc']:.3f} ({t['auroc_ci'][0]:.3f}-{t['auroc_ci'][1]:.3f}) | {ds} | {t['brier']:.3f} | {t['ece']:.3f} |")
    L.append(NL + "## Rule-out operating point (threshold chosen on training folds for >=95% sensitivity)" + NL)
    L.append("| Model | Sensitivity | Specificity | NPV | Normal angiograms avoided /100 | CAD missed /100 |")
    L.append("|---|---|---|---|---|---|")
    for m in res["models"]:
        r = m["rule_out"]
        L.append(f"| {m['name']} | {r['sensitivity']:.3f} ({r['sensitivity_ci'][0]:.2f}-{r['sensitivity_ci'][1]:.2f}) | "
                 f"{r['specificity']:.3f} ({r['specificity_ci'][0]:.2f}-{r['specificity_ci'][1]:.2f}) | "
                 f"{r['npv']:.3f} ({r['npv_ci'][0]:.2f}-{r['npv_ci'][1]:.2f}) | "
                 f"{r['normal_caths_avoided_per_100']:.1f} ({r['normal_caths_avoided_per_100_ci'][0]:.1f}-{r['normal_caths_avoided_per_100_ci'][1]:.1f}) | "
                 f"{r['cad_missed_per_100']:.1f} ({r['cad_missed_per_100_ci'][0]:.1f}-{r['cad_missed_per_100_ci'][1]:.1f}) |")
    L.append(NL + "## Vessel-level stenosis (AUROC, 95% CI)" + NL)
    L.append("| Model | LAD | LCX | RCA |")
    L.append("|---|---|---|---|")
    for m in res["models"]:
        if "LAD" not in m["tasks"]:
            continue
        cells = [f"{m['tasks'][v]['auroc']:.3f} ({m['tasks'][v]['auroc_ci'][0]:.2f}-{m['tasks'][v]['auroc_ci'][1]:.2f})" for v in ("LAD", "LCX", "RCA")]
        L.append(f"| {m['name']} | " + " | ".join(cells) + " |")
    L.append(NL + "## Which tests does it depend on? (logistic regression, CAD)" + NL)
    L.append("| Inputs available | # features | AUROC (95% CI) | Normal angiograms avoided /100 | CAD missed /100 |")
    L.append("|---|---|---|---|---|")
    for a in res["ablation"]:
        L.append(f"| {a['set']} | {a['n_features']} | {a['auroc']:.3f} ({a['auroc_ci'][0]:.3f}-{a['auroc_ci'][1]:.3f}) | "
                 f"{a['normal_caths_avoided_per_100']:.1f} | {a['cad_missed_per_100']:.1f} |")
    L.append(NL + "## Subgroups (CAD AUROC, out-of-fold)" + NL)
    L.append("| Subgroup | n | CAD prevalence | Deployed model | ESC 2019 |")
    L.append("|---|---|---|---|---|")
    for s in res["subgroups"]:
        L.append(f"| {s['group']} | {s['n']} | {s['prevalence']:.0%} | {s['model_auroc']:.3f} ({s['model_auroc_ci'][0]:.2f}-{s['model_auroc_ci'][1]:.2f}) | {s['esc_auroc']:.3f} |")
    ex = res["external"]
    L.append(NL + "## External validation: bedside model, leave-one-hospital-out (5 hospitals, n=1,223)" + NL)
    L.append("Each hospital is scored by a model trained only on the other four (imputation, scaling, C and threshold all from the other four)." + NL)
    L.append("| Held-out hospital | n | CAD prev. | Bedside LR AUROC (95% CI) | ESC 2019 AUROC | dAUROC (95% CI) | Cal. slope | Pred - obs | Sens. | Avoided /100 | Missed /100 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in ex["per_site"]:
        m, e, d = r["models"]["bedside_lr"], r["models"]["esc"], r["delta_lr_vs_esc"]
        L.append(f"| {r['site']} | {r['n']} | {r['prevalence']:.0%} | {m['auroc']:.3f} ({m['auroc_ci'][0]:.2f}-{m['auroc_ci'][1]:.2f}) | {e['auroc']:.3f} | "
                 f"{d['delta']:+.3f} ({d['ci'][0]:+.2f}, {d['ci'][1]:+.2f}) | {m['cal_slope']:.2f} | {m['cal_in_large']:+.2f} | {m['sensitivity']:.2f} | "
                 f"{m['normal_caths_avoided_per_100']:.1f} | {m['cad_missed_per_100']:.1f} |")
    L.append(NL + f"Pooled 5-site cross-validation (sites mixed in training and test): AUROC {ex['pooled_cv_auroc']:.3f}. "
             "The drop to leave-one-hospital-out is the transfer gap." + NL)
    L.append("### How many local patients does a new hospital need?" + NL)
    L.append("Bedside LR AUROC on the remaining local patients after adding k labelled local patients to training (mean, 10th-90th percentile over random draws)." + NL)
    L.append("| Hospital | +0 local | +25 local | +50 local | +100 local |")
    L.append("|---|---|---|---|---|")
    for site in [s["site"] for s in ex["sites"]]:
        cells = []
        for k in (0, 25, 50, 100):
            a = next((a for a in ex["adaptation"] if a["site"] == site and a["k_local"] == k), None)
            cells.append(f"{a['auroc_mean']:.3f} ({a['auroc_p10']:.2f}-{a['auroc_p90']:.2f})" if a else "n/a")
        L.append(f"| {site} | " + " | ".join(cells) + " |")
    nc = res["negative_control"]
    L.append(NL + "## Negative control" + NL + f"Labels permuted, same pipeline: CAD AUROC {nc['auroc']:.3f} (expected ~0.5)." + NL)
    with open(path, "w", encoding="utf-8") as f:
        f.write(NL.join(L) + NL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--target-sens", type=float, default=0.95)
    ap.add_argument("--deployed", default="lr", help="model key reported in subgroups / used by the dashboard")
    ap.add_argument("--quick", action="store_true", help="1 repeat, no neural net, B=300")
    args = ap.parse_args()
    if args.quick:
        args.repeats, args.bootstrap = 1, 300
    try:
        import torch
        torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    except ImportError:
        pass
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(DATA)
    X, Y = encode_features(df), encode_targets(df)
    log = print
    log(f"Cohort n={len(X)}, features={X.shape[1]}; CAD prevalence {Y['CAD'].mean():.3f}")
    zoo = model_zoo(include_mtl=not args.quick)
    oofs, summaries = {}, []
    for spec in zoo:
        log(f"\n[{spec.key}] {spec.name}")
        oofs[spec.key] = cross_validate_spec(spec, X, Y, args.repeats, args.folds, args.seed, args.target_sens, log=log)
        summaries.append(summarise(spec, oofs[spec.key], Y, args.bootstrap, args.seed))
    for s in summaries:
        if s["key"] != "esc":
            s["delta_vs_esc"] = paired_delta(oofs[s["key"]], oofs["esc"], Y["CAD"], args.bootstrap, args.seed)
            s["delta_vs_clinical_lr"] = paired_delta(oofs[s["key"]], oofs["clin_lr"], Y["CAD"], args.bootstrap, args.seed) if s["key"] != "clin_lr" else None
    # ---- feature-availability ablation (logistic regression)
    log("\n[ablation] which inputs matter")
    ablation = []
    for name, cols in FEATURE_SETS.items():
        spec = ModelSpec("abl", name, lambda Xa, Ya, s, c=cols: fit_lr(Xa, Ya, s, cols=c, tasks=["CAD"]), ["CAD"], "")
        r = cross_validate_spec(spec, X, Y, args.repeats, args.folds, args.seed, args.target_sens)
        sm = summarise(spec, r, Y, args.bootstrap // 2, args.seed)
        ablation.append({"set": name, "n_features": len(cols), "auroc": sm["tasks"]["CAD"]["auroc"],
                         "auroc_ci": sm["tasks"]["CAD"]["auroc_ci"], **{k: sm["rule_out"][k] for k in
                         ("sensitivity", "specificity", "normal_caths_avoided_per_100", "cad_missed_per_100")}})
        log(f"    {name:40s} AUROC {ablation[-1]['auroc']:.3f}")
    # ---- subgroups for the deployed model
    dep = oofs[args.deployed]["oof"]["CAD"]; esc = oofs["esc"]["oof"]["CAD"]; y = Y["CAD"]
    groups = {
        "All patients": np.ones(len(y), bool),
        "Women": X["Sex"].values == 0, "Men": X["Sex"].values == 1,
        "Age < 60": X["Age"].values < 60, "Age ≥ 60": X["Age"].values >= 60,
        "Diabetes": X["DM"].values == 1, "No diabetes": X["DM"].values == 0,
        "Typical chest pain": X["Typical Chest Pain"].values == 1,
        "No typical chest pain": X["Typical Chest Pain"].values == 0,
        "Normal echo wall motion (RWMA = 0)": X["Region RWMA"].values == 0,
    }
    subgroups = []
    for g, mask in groups.items():
        ii = np.where(mask)[0]
        f = lambda idx, P=dep: float(np.mean([fast_auc(y[ii][idx], row[ii][idx]) for row in P]))
        subgroups.append({"group": g, "n": int(mask.sum()), "prevalence": float(y[mask].mean()),
                          "model_auroc": f(np.arange(len(ii))), "model_auroc_ci": ci(bootstrap(f, len(ii), args.bootstrap // 2, args.seed)),
                          "esc_auroc": float(np.mean([fast_auc(y[ii], row[ii]) for row in esc]))})
    # ---- negative control: permuted labels
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(y))
    Yp = {t: v[perm] for t, v in Y.items()}
    nc_spec = ModelSpec("nc", "permuted", lambda Xa, Ya, s: fit_lr(Xa, Ya, s, tasks=["CAD"]), ["CAD"], "")
    nc = cross_validate_spec(nc_spec, X, Yp, 1, args.folds, args.seed, tune_threshold=False)
    negative_control = {"auroc": fast_auc(Yp["CAD"], nc["oof"]["CAD"][0])}
    # ---- external validation of the bedside model across 5 hospitals
    log("[external] leave-one-hospital-out, bedside model (7 harmonised inputs)")
    hdf = load_harmonised()
    ext = loso(hdf, args.seed, args.target_sens, args.bootstrap // 2, log=log)
    pooled = pooled_cv(hdf, args.seed, args.repeats, args.folds)
    adapt = local_adaptation(hdf, args.seed, draws=5 if args.quick else 20)
    ss = site_summary(hdf)
    ext_df = hdf[["site", "cad"]].copy()
    for k, v in ext["preds"].items():
        ext_df[f"loso_{k}"] = v
    ext_df.to_csv(os.path.join(OUT, "external_predictions.csv"), index=False)
    external = {"sites": [{"site": s, **{k: float(v) for k, v in ss.loc[s].items()}} for s in SITES],
                "per_site": ext["per_site"], "model_names": ext["model_names"],
                "pooled_cv_auroc": pooled["auroc"], "adaptation": adapt}
    results = {
        "config": {**vars(args), "runtime_s": time.time() - t0, "python": platform.python_version(),
                   "numpy": np.__version__, "pandas": pd.__version__,
                   "sklearn": __import__("sklearn").__version__},
        "cohort": {"n": int(len(y)), "n_features": int(X.shape[1]),
                   "prevalence": {t: float(Y[t].mean()) for t in TASKS}},
        "models": summaries, "ablation": ablation, "subgroups": subgroups,
        "negative_control": negative_control,
        "external": external,
    }
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    # out-of-fold predictions (mean over repeats) for every patient and model
    oof_df = pd.DataFrame({"row": np.arange(len(y)), **{f"true_{t}": Y[t] for t in TASKS}})
    for k, r in oofs.items():
        for t, P in r["oof"].items():
            oof_df[f"{k}_{t}"] = P.mean(0)
        oof_df[f"{k}_ruled_out_frac"] = r["ruled_out"].mean(0)
    oof_df.to_csv(os.path.join(OUT, "oof_predictions.csv"), index=False)
    plot_figures(results, oofs, Y, X, os.path.join(OUT, "figures"))
    write_markdown(results, os.path.join(OUT, "summary.md"))
    log(f"\nDone in {time.time() - t0:.0f}s -> {OUT}/summary.md")
    print(open(os.path.join(OUT, "summary.md"), encoding="utf-8").read())

if __name__ == "__main__":
    main()
