"""
Harmonised multi-centre dataset for the BEDSIDE model (5 hospitals, 4 countries).

Sources (all public, all labelled by invasive coronary angiography, CAD = any major
vessel with >=50% diameter narrowing):

    site          n    source
    Tehran       303   Z-Alizadeh Sani extension (UCI #411)
    Cleveland    303   UCI Heart Disease (#45), processed.cleveland.data
    Budapest     294   UCI Heart Disease (#45), processed.hungarian.data
    Zurich       123   UCI Heart Disease (#45), processed.switzerland.data
    Long Beach   200   UCI Heart Disease (#45), processed.va.data

Only variables that exist (with the same meaning) at every site are kept:

    age, male sex, chest-pain type (typical / atypical / non-anginal / none),
    resting systolic BP, fasting glucose > 120 mg/dL, resting-ECG ST-T abnormality,
    resting-ECG left-ventricular hypertrophy

Mapping notes (documented limitations):
* UCI restecg is a single category (0 normal, 1 ST-T abnormality, 2 LVH by Estes);
  Tehran records ST elevation / ST depression / T inversion / LVH separately. We map
  Tehran ST-T = any of the three, LVH = LVH; UCI 1 -> ST-T, 2 -> LVH.
* Tehran "none" = no typical, atypical or non-anginal pain (may still have dyspnoea);
  UCI cp = 4 is "asymptomatic".
* Resting BP 0 in the UCI files is a missing-value code. Missing BP and glucose are
  imputed (training-set median) with a missingness indicator, all fit in-fold.
"""

import os
from typing import Tuple

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UCI_COLS = "age sex cp trestbps chol fbs restecg thalach exang oldpeak slope ca thal num".split()
UCI_SITES = {"Cleveland": "cleveland", "Budapest": "hungarian", "Zurich": "switzerland", "Long Beach": "va"}
SITES = ["Tehran", "Cleveland", "Budapest", "Zurich", "Long Beach"]

BEDSIDE_FEATURES = ["age", "male", "cp_typical", "cp_atypical", "cp_nonanginal",
                    "sbp", "sbp_missing", "fbs_high", "fbs_missing", "ecg_stt", "ecg_lvh"]
BEDSIDE_LABELS = {
    "age": "Age (years)", "male": "Male sex", "cp_typical": "Typical angina",
    "cp_atypical": "Atypical angina", "cp_nonanginal": "Non-anginal chest pain",
    "sbp": "Resting systolic BP (mmHg)", "sbp_missing": "BP not recorded",
    "fbs_high": "Fasting glucose > 120 mg/dL", "fbs_missing": "Glucose not recorded",
    "ecg_stt": "Resting ECG: ST-T abnormality", "ecg_lvh": "Resting ECG: LVH",
}


def _from_tehran(path=None) -> pd.DataFrame:
    df = pd.read_csv(path or os.path.join(ROOT, "data", "z_alizadeh_sani_extended.csv"))
    yn = lambda s: s.astype(str).str.strip().isin(["Y", "1"]).astype(float)
    typ, aty, non = df["Typical Chest Pain"].astype(float), yn(df["Atypical"]), yn(df["Nonanginal"])
    return pd.DataFrame({
        "site": "Tehran",
        "age": df["Age"].astype(float),
        "male": (df["Sex"].astype(str).str.strip() == "Male").astype(float),
        # precedence typical > atypical > non-anginal, so the categories are exclusive
        "cp_typical": typ,
        "cp_atypical": ((typ == 0) & (aty == 1)).astype(float),
        "cp_nonanginal": ((typ == 0) & (aty == 0) & (non == 1)).astype(float),
        "sbp": df["BP"].astype(float),
        "fbs": (df["FBS"].astype(float) > 120).astype(float),
        "ecg_stt": ((df["St Elevation"] + df["St Depression"] + df["Tinversion"]) > 0).astype(float),
        "ecg_lvh": yn(df["LVH"]),
        "cad": (df["Cath"].astype(str).str.strip() == "CAD").astype(int),
    })


def _from_uci(site: str, stem: str) -> pd.DataFrame:
    d = pd.read_csv(os.path.join(ROOT, "data", "uci_heart", f"processed.{stem}.data"),
                    header=None, names=UCI_COLS, na_values="?")
    sbp = d["trestbps"].astype(float).where(d["trestbps"] > 0)
    rest = d["restecg"]
    return pd.DataFrame({
        "site": site,
        "age": d["age"].astype(float),
        "male": d["sex"].astype(float),
        "cp_typical": (d["cp"] == 1).astype(float),
        "cp_atypical": (d["cp"] == 2).astype(float),
        "cp_nonanginal": (d["cp"] == 3).astype(float),
        "sbp": sbp,
        "fbs": d["fbs"].astype(float),
        "ecg_stt": (rest == 1).astype(float).where(rest.notna()),
        "ecg_lvh": (rest == 2).astype(float).where(rest.notna()),
        "cad": (d["num"] > 0).astype(int),
    })


def load_harmonised() -> pd.DataFrame:
    parts = [_from_tehran()] + [_from_uci(s, stem) for s, stem in UCI_SITES.items()]
    df = pd.concat(parts, ignore_index=True)
    # Resting-ECG missing in 2 UCI rows: treat as normal (documented; affects <0.2%).
    df[["ecg_stt", "ecg_lvh"]] = df[["ecg_stt", "ecg_lvh"]].fillna(0.0)
    return df


def bedside_matrix(df: pd.DataFrame, medians: dict) -> pd.DataFrame:
    """Raw harmonised rows -> BEDSIDE_FEATURES, imputing with the given (training) medians."""
    out = df[["age", "male", "cp_typical", "cp_atypical", "cp_nonanginal", "ecg_stt", "ecg_lvh"]].copy()
    out["sbp_missing"] = df["sbp"].isna().astype(float)
    out["sbp"] = df["sbp"].fillna(medians["sbp"])
    out["fbs_missing"] = df["fbs"].isna().astype(float)
    out["fbs_high"] = df["fbs"].fillna(0.0)  # missing glucose carries its own indicator
    return out[BEDSIDE_FEATURES]


def training_medians(df: pd.DataFrame) -> dict:
    return {"sbp": float(df["sbp"].median())}


def site_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("site", sort=False)
    return pd.DataFrame({
        "n": g.size(), "cad_prevalence": g["cad"].mean(), "age_mean": g["age"].mean(),
        "male_frac": g["male"].mean(), "typical_frac": g["cp_typical"].mean(),
        "none_frac": g.apply(lambda s: float(((s.cp_typical + s.cp_atypical + s.cp_nonanginal) == 0).mean()), include_groups=False),
        "sbp_missing": g["sbp"].apply(lambda s: float(s.isna().mean())),
        "fbs_missing": g["fbs"].apply(lambda s: float(s.isna().mean())),
    }).reindex(SITES)


def esc_symptom_ptp(df: pd.DataFrame) -> np.ndarray:
    """ESC 2019 table from harmonised columns (no-pain rows use the lowest, non-anginal row)."""
    from src.baselines import _AGE_BANDS, _ESC_2019
    out = np.empty(len(df))
    for i, r in enumerate(df.itertuples(index=False)):
        sym = "typical" if r.cp_typical else "atypical" if r.cp_atypical else "nonanginal"
        band = next(j for j, (lo, hi) in enumerate(_AGE_BANDS) if lo <= r.age < hi)
        out[i] = _ESC_2019[(sym, int(r.male))][band] / 100.0
    return out
