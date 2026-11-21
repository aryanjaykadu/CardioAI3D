"""
Guideline reference: ESC 2019 pre-test probability (PTP) of obstructive CAD.

Source: Knuuti J, et al. 2019 ESC Guidelines for the diagnosis and management of
chronic coronary syndromes. Eur Heart J 2020;41:407-477, Table 5 (derived from
Juarez-Orozco LE, et al. Eur Heart J Cardiovasc Imaging 2019).

Values (in %) are transcribed from that table. The table needs only age, sex and
symptom type, so it is the "what a clinician would estimate at the bedside"
reference. It was derived in populations with much lower CAD prevalence than
this cath-referred cohort (71%), so its absolute values are expected to be
mis-calibrated here; we compare on ranking (AUROC) and on a rule-out operating
point whose threshold is chosen on training folds, exactly like the ML models.
"""

import numpy as np
import pandas as pd

# rows: age band; columns: (symptom, sex) -> PTP %
_AGE_BANDS = [(0, 40), (40, 50), (50, 60), (60, 70), (70, 200)]
_ESC_2019 = {
    #            30-39 40-49 50-59 60-69 70+
    ("typical", 1):    [3, 22, 32, 44, 52],
    ("typical", 0):    [5, 10, 13, 16, 27],
    ("atypical", 1):   [4, 10, 17, 26, 34],
    ("atypical", 0):   [3, 6, 6, 11, 19],
    ("nonanginal", 1): [1, 3, 11, 22, 24],
    ("nonanginal", 0): [1, 2, 3, 6, 10],
    ("dyspnoea", 1):   [0, 12, 20, 27, 32],
    ("dyspnoea", 0):   [3, 3, 9, 14, 12],
}


def _symptom(row: pd.Series) -> str:
    if float(row["Typical Chest Pain"]) == 1:
        return "typical"
    if float(row["Atypical"]) == 1:
        return "atypical"
    if float(row["Nonanginal"]) == 1:
        return "nonanginal"
    if float(row["Dyspnea"]) == 1:
        return "dyspnoea"
    # No chest pain and no dyspnoea: lowest category in the table.
    return "nonanginal"


def esc2019_ptp(X_encoded: pd.DataFrame) -> np.ndarray:
    """Return ESC 2019 PTP as a probability in [0, 1] for each row of encoded features."""
    out = np.empty(len(X_encoded))
    for i, (_, row) in enumerate(X_encoded.iterrows()):
        band = next(j for j, (lo, hi) in enumerate(_AGE_BANDS) if lo <= row["Age"] < hi)
        out[i] = _ESC_2019[(_symptom(row), int(row["Sex"]))][band] / 100.0
    # Tiny deterministic jitter-free tie handling is left to the metrics (ties are averaged).
    return out
