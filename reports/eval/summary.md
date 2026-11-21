# CardioAI evaluation summary

Cohort: n=303 cath-referred patients (Tehran); prevalence CAD 71%, LAD 58%, LCX 39%, RCA 38%.  
Protocol: 5x5-fold stratified CV, seed 42, bootstrap B=2000. Runtime 675s.

## Full model: CAD (any vessel >=50% stenosis at angiography)

| Model | AUROC (95% CI) | dAUROC vs ESC 2019 (95% CI) | Brier | ECE |
|---|---|---|---|---|
| ESC 2019 pre-test probability (guideline) | 0.822 (0.767-0.873) | - | 0.405 | 0.488 |
| Clinical logistic regression (bedside risk factors) | 0.915 (0.873-0.950) | +0.093 (+0.052, +0.135) | 0.096 | 0.051 |
| Logistic regression (all 55 features) | 0.922 (0.889-0.950) | +0.100 (+0.059, +0.145) | 0.099 | 0.042 |
| Logistic regression (RFE top-15) | 0.924 (0.893-0.949) | +0.102 (+0.057, +0.151) | 0.100 | 0.056 |
| Random forest | 0.920 (0.884-0.949) | +0.098 (+0.052, +0.145) | 0.120 | 0.148 |
| Gradient boosting | 0.918 (0.882-0.946) | +0.096 (+0.053, +0.142) | 0.102 | 0.032 |
| Multitask neural net (RFE-15, focal loss, Platt) | 0.874 (0.836-0.907) | +0.052 (+0.004, +0.103) | 0.126 | 0.061 |

## Rule-out operating point (threshold chosen on training folds for >=95% sensitivity)

| Model | Sensitivity | Specificity | NPV | Normal angiograms avoided /100 | CAD missed /100 |
|---|---|---|---|---|---|
| ESC 2019 pre-test probability (guideline) | 0.963 (0.94-0.99) | 0.368 (0.27-0.47) | 0.800 (0.68-0.92) | 10.6 (7.6-14.2) | 2.6 (1.0-4.6) |
| Clinical logistic regression (bedside risk factors) | 0.951 (0.92-0.97) | 0.639 (0.55-0.73) | 0.840 (0.75-0.91) | 18.3 (13.9-22.8) | 3.5 (1.8-5.5) |
| Logistic regression (all 55 features) | 0.944 (0.92-0.97) | 0.563 (0.48-0.66) | 0.805 (0.71-0.89) | 16.2 (12.5-20.6) | 4.0 (2.1-6.0) |
| Logistic regression (RFE top-15) | 0.963 (0.94-0.98) | 0.487 (0.41-0.57) | 0.846 (0.76-0.92) | 14.0 (10.8-17.5) | 2.6 (1.4-4.3) |
| Random forest | 0.944 (0.91-0.97) | 0.680 (0.60-0.77) | 0.829 (0.75-0.92) | 19.5 (15.4-24.3) | 4.0 (1.9-6.4) |
| Gradient boosting | 0.946 (0.91-0.97) | 0.602 (0.52-0.69) | 0.819 (0.73-0.91) | 17.3 (13.3-21.6) | 3.8 (2.0-6.1) |
| Multitask neural net (RFE-15, focal loss, Platt) | 0.960 (0.94-0.98) | 0.317 (0.26-0.37) | 0.790 (0.69-0.89) | 9.1 (6.9-11.6) | 2.8 (1.6-4.1) |

## Vessel-level stenosis (AUROC, 95% CI)

| Model | LAD | LCX | RCA |
|---|---|---|---|
| Clinical logistic regression (bedside risk factors) | 0.789 (0.73-0.84) | 0.736 (0.68-0.79) | 0.740 (0.68-0.80) |
| Logistic regression (all 55 features) | 0.826 (0.78-0.87) | 0.680 (0.62-0.74) | 0.718 (0.66-0.77) |
| Logistic regression (RFE top-15) | 0.804 (0.76-0.85) | 0.673 (0.62-0.73) | 0.725 (0.67-0.78) |
| Random forest | 0.846 (0.80-0.89) | 0.738 (0.68-0.79) | 0.713 (0.66-0.77) |
| Gradient boosting | 0.827 (0.78-0.87) | 0.732 (0.68-0.78) | 0.691 (0.64-0.74) |
| Multitask neural net (RFE-15, focal loss, Platt) | 0.797 (0.75-0.84) | 0.665 (0.62-0.71) | 0.652 (0.61-0.70) |

## Which tests does it depend on? (logistic regression, CAD)

| Inputs available | # features | AUROC (95% CI) | Normal angiograms avoided /100 | CAD missed /100 |
|---|---|---|---|---|
| Bedside (history + symptoms + exam) | 30 | 0.909 (0.869-0.943) | 15.9 | 3.6 |
| + ECG | 38 | 0.922 (0.889-0.952) | 16.7 | 3.0 |
| + ECG + labs | 52 | 0.915 (0.882-0.946) | 14.9 | 3.3 |
| + ECG + labs + echo (all) | 55 | 0.922 (0.891-0.949) | 16.2 | 4.0 |

## Subgroups (CAD AUROC, out-of-fold)

| Subgroup | n | CAD prevalence | Deployed model | ESC 2019 |
|---|---|---|---|---|
| All patients | 303 | 71% | 0.922 (0.89-0.95) | 0.822 |
| Women | 127 | 68% | 0.956 (0.92-0.99) | 0.890 |
| Men | 176 | 74% | 0.891 (0.84-0.94) | 0.859 |
| Age < 60 | 162 | 58% | 0.930 (0.89-0.96) | 0.796 |
| Age ≥ 60 | 141 | 87% | 0.855 (0.77-0.93) | 0.753 |
| Diabetes | 90 | 89% | 0.964 (0.92-1.00) | 0.928 |
| No diabetes | 213 | 64% | 0.899 (0.86-0.94) | 0.834 |
| Typical chest pain | 164 | 94% | 0.838 (0.71-0.94) | 0.577 |
| No typical chest pain | 139 | 45% | 0.850 (0.79-0.91) | 0.755 |
| Normal echo wall motion (RWMA = 0) | 217 | 62% | 0.896 (0.85-0.93) | 0.807 |

## External validation: bedside model, leave-one-hospital-out (5 hospitals, n=1,223)

Each hospital is scored by a model trained only on the other four (imputation, scaling, C and threshold all from the other four).

| Held-out hospital | n | CAD prev. | Bedside LR AUROC (95% CI) | ESC 2019 AUROC | dAUROC (95% CI) | Cal. slope | Pred - obs | Sens. | Avoided /100 | Missed /100 |
|---|---|---|---|---|---|---|---|---|---|---|
| Tehran | 303 | 71% | 0.770 (0.70-0.84) | 0.831 | -0.061 (-0.13, +0.00) | 0.89 | -0.35 | 0.72 | 21.5 | 19.8 |
| Cleveland | 303 | 46% | 0.779 (0.73-0.83) | 0.626 | +0.153 (+0.10, +0.21) | 0.84 | +0.09 | 0.96 | 22.4 | 1.7 |
| Budapest | 294 | 36% | 0.824 (0.77-0.87) | 0.504 | +0.320 (+0.25, +0.39) | 1.11 | +0.11 | 0.87 | 38.8 | 4.8 |
| Zurich | 123 | 93% | 0.753 (0.50-0.93) | 0.475 | +0.278 (+0.08, +0.48) | 0.84 | -0.21 | 0.97 | 0.8 | 3.3 |
| Long Beach | 200 | 74% | 0.676 (0.59-0.77) | 0.560 | +0.116 (+0.00, +0.22) | 0.50 | +0.02 | 1.00 | 2.0 | 0.0 |
| All sites (pooled) | 1223 | 59% | 0.745 (0.72-0.77) | 0.695 | +0.050 (+0.02, +0.08) | 0.65 | -0.05 | 0.89 | 20.6 | 6.8 |

Pooled 5-site cross-validation (sites mixed in training and test): AUROC 0.841. The drop to leave-one-hospital-out is the transfer gap.

### How many local patients does a new hospital need?

Bedside LR AUROC on the remaining local patients after adding k labelled local patients to training (mean, 10th-90th percentile over random draws).

| Hospital | +0 local | +25 local | +50 local | +100 local |
|---|---|---|---|---|
| Tehran | 0.770 (0.77-0.77) | 0.811 (0.79-0.83) | 0.831 (0.81-0.85) | 0.858 (0.84-0.87) |
| Cleveland | 0.779 (0.78-0.78) | 0.779 (0.77-0.79) | 0.780 (0.77-0.79) | 0.780 (0.76-0.80) |
| Budapest | 0.824 (0.82-0.82) | 0.825 (0.82-0.83) | 0.828 (0.81-0.84) | 0.833 (0.82-0.85) |
| Zurich | n/a | n/a | n/a | n/a |
| Long Beach | 0.676 (0.68-0.68) | 0.668 (0.64-0.69) | 0.677 (0.65-0.71) | 0.674 (0.63-0.71) |

## Negative control
Labels permuted, same pipeline: CAD AUROC 0.492 (expected ~0.5).

