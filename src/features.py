"""
Deterministic feature encoding for the Z-Alizadeh Sani (extension) dataset.

Every raw column is mapped to a number with a fixed, hand-written rule (no
statistics are learned here), so the encoding cannot leak information across
folds and can be re-implemented exactly in the browser (dashboard/app.js).

Learned preprocessing (standardisation) happens later, inside each training fold.
"""

from typing import Dict, List

import numpy as np
import pandas as pd

TARGET_COLUMNS = ["Cath", "LAD", "LCX", "RCA"]
TASKS = ["CAD", "LAD", "LCX", "RCA"]

# Columns with a single value in the whole cohort (carry no information).
CONSTANT_COLUMNS = ["Exertional CP"]

# Feature groups, ordered by how hard they are to obtain. Used for the
# "what if this test is not available?" ablation in evaluate.py.
FEATURE_GROUPS: Dict[str, List[str]] = {
    "history": [
        "Age", "Weight", "Length", "Sex", "BMI", "DM", "HTN", "Current Smoker",
        "EX-Smoker", "FH", "Obesity", "CRF", "CVA", "Airway disease",
        "Thyroid Disease", "CHF", "DLP",
    ],
    "symptoms": [
        "Typical Chest Pain", "Dyspnea", "Function Class", "Atypical",
        "Nonanginal", "LowTH Ang",
    ],
    "exam": [
        "BP", "PR", "Edema", "Weak Peripheral Pulse", "Lung rales",
        "Systolic Murmur", "Diastolic Murmur",
    ],
    "ecg": [
        "Q Wave", "St Elevation", "St Depression", "Tinversion", "LVH",
        "Poor R Progression", "BBB_LBBB", "BBB_RBBB",
    ],
    "labs": [
        "FBS", "CR", "TG", "LDL", "HDL", "BUN", "ESR", "HB", "K", "Na", "WBC",
        "Lymph", "Neut", "PLT",
    ],
    "echo": ["EF-TTE", "Region RWMA", "VHD"],
}

ALL_FEATURES: List[str] = [f for g in FEATURE_GROUPS.values() for f in g]

# Feature sets for the availability ablation (cumulative).
FEATURE_SETS: Dict[str, List[str]] = {
    "Bedside (history + symptoms + exam)": FEATURE_GROUPS["history"] + FEATURE_GROUPS["symptoms"] + FEATURE_GROUPS["exam"],
    "+ ECG": FEATURE_GROUPS["history"] + FEATURE_GROUPS["symptoms"] + FEATURE_GROUPS["exam"] + FEATURE_GROUPS["ecg"],
    "+ ECG + labs": FEATURE_GROUPS["history"] + FEATURE_GROUPS["symptoms"] + FEATURE_GROUPS["exam"] + FEATURE_GROUPS["ecg"] + FEATURE_GROUPS["labs"],
    "+ ECG + labs + echo (all)": ALL_FEATURES,
}

VHD_ORDER = {"N": 0, "mild": 1, "Moderate": 2, "Severe": 3}

# Human-readable labels and units used by the dashboard.
FEATURE_LABELS: Dict[str, str] = {
    "Age": "Age (years)", "Weight": "Weight (kg)", "Length": "Height (cm)",
    "Sex": "Male sex", "BMI": "BMI (kg/m²)", "DM": "Diabetes", "HTN": "Hypertension",
    "Current Smoker": "Current smoker", "EX-Smoker": "Ex-smoker",
    "FH": "Family history of CAD", "Obesity": "Obesity", "CRF": "Chronic renal failure",
    "CVA": "Prior stroke", "Airway disease": "Airway disease",
    "Thyroid Disease": "Thyroid disease", "CHF": "Heart failure", "DLP": "Dyslipidaemia",
    "BP": "Blood pressure (mmHg)", "PR": "Pulse rate (bpm)", "Edema": "Oedema",
    "Weak Peripheral Pulse": "Weak peripheral pulse", "Lung rales": "Lung rales",
    "Systolic Murmur": "Systolic murmur", "Diastolic Murmur": "Diastolic murmur",
    "Typical Chest Pain": "Typical chest pain", "Dyspnea": "Dyspnoea",
    "Function Class": "Functional class (NYHA)", "Atypical": "Atypical chest pain",
    "Nonanginal": "Non-anginal chest pain", "LowTH Ang": "Low-threshold angina",
    "Q Wave": "ECG: Q wave", "St Elevation": "ECG: ST elevation",
    "St Depression": "ECG: ST depression", "Tinversion": "ECG: T inversion",
    "LVH": "ECG: LV hypertrophy", "Poor R Progression": "ECG: poor R progression",
    "BBB_LBBB": "ECG: LBBB", "BBB_RBBB": "ECG: RBBB",
    "FBS": "Fasting glucose (mg/dL)", "CR": "Creatinine (mg/dL)",
    "TG": "Triglycerides (mg/dL)", "LDL": "LDL (mg/dL)", "HDL": "HDL (mg/dL)",
    "BUN": "BUN (mg/dL)", "ESR": "ESR (mm/h)", "HB": "Haemoglobin (g/dL)",
    "K": "Potassium (mEq/L)", "Na": "Sodium (mEq/L)", "WBC": "WBC (/µL)",
    "Lymph": "Lymphocytes (%)", "Neut": "Neutrophils (%)", "PLT": "Platelets (10³/µL)",
    "EF-TTE": "Echo: ejection fraction (%)", "Region RWMA": "Echo: wall-motion abnormal regions",
    "VHD": "Echo: valve disease grade (0–3)",
}


def _yn(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    out = s.map({"Y": 1, "N": 0, "1": 1, "0": 0})
    if out.isnull().any():
        raise ValueError(f"Unexpected values in {series.name}: {sorted(s.unique())}")
    return out.astype(float)


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw dataframe -> numeric dataframe with columns ALL_FEATURES (fixed order)."""
    out = pd.DataFrame(index=df.index)
    for col in ALL_FEATURES:
        if col == "Sex":
            s = df["Sex"].astype(str).str.strip()
            out[col] = s.map({"Male": 1.0, "Fmale": 0.0, "Female": 0.0})
        elif col == "BBB_LBBB":
            out[col] = (df["BBB"].astype(str).str.strip() == "LBBB").astype(float)
        elif col == "BBB_RBBB":
            out[col] = (df["BBB"].astype(str).str.strip() == "RBBB").astype(float)
        elif col == "VHD":
            out[col] = df["VHD"].astype(str).str.strip().map(VHD_ORDER).astype(float)
        elif not pd.api.types.is_numeric_dtype(df[col]):
            out[col] = _yn(df[col])
        else:
            out[col] = df[col].astype(float)
    if out.isnull().any().any():
        bad = out.columns[out.isnull().any()].tolist()
        raise ValueError(f"Encoding produced missing values in: {bad}")
    return out


def encode_targets(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Cath -> CAD (1 = CAD), vessels -> 1 = stenotic (>=50% narrowing at angiography)."""
    t = {
        "CAD": df["Cath"].astype(str).str.strip().map({"CAD": 1, "Normal": 0}),
        "LAD": df["LAD"].astype(str).str.strip().map({"Stenotic": 1, "Normal": 0}),
        "LCX": df["LCX"].astype(str).str.strip().map({"Stenotic": 1, "Normal": 0}),
        "RCA": df["RCA"].astype(str).str.strip().map({"Stenotic": 1, "Normal": 0}),
    }
    for k, v in t.items():
        if v.isnull().any():
            raise ValueError(f"Unmapped target values in {k}")
    return {k: v.values.astype(int) for k, v in t.items()}
