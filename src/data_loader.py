"""
Data Loader Module for Extension of Z-Alizadeh Sani Dataset.

Ensures strict separation of features and targets, explicitly preventing
target leakage by removing ['LAD', 'LCX', 'RCA', 'Cath'] from predictor features.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List


LEAKAGE_COLUMNS = ['LAD', 'LCX', 'RCA', 'Cath']

TARGET_MAPPINGS = {
    'Cath': {'CAD': 1, 'Normal': 0},
    'CAD': {'CAD': 1, 'Normal': 0},
    'LAD': {'Stenotic': 1, 'Normal': 0},
    'LCX': {'Stenotic': 1, 'Normal': 0},
    'RCA': {'Stenotic': 1, 'Normal': 0}
}


def load_dataset(
    csv_path: str = "data/z_alizadeh_sani_extended.csv",
    use_uci_api: bool = False
) -> pd.DataFrame:
    """
    Loads the Extension of Z-Alizadeh Sani dataset.
    Can load via ucimlrepo if specified and available, otherwise loads local cleaned CSV.
    """
    if use_uci_api:
        try:
            from ucimlrepo import fetch_ucirepo
            dataset = fetch_ucirepo(id=411)
            # Combine features and targets
            df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
            return df
        except Exception as e:
            print(f"Warning: Failed to fetch via ucimlrepo ({e}). Falling back to local file.")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found at {csv_path}")

    df = pd.read_csv(csv_path)
    return df


def prepare_features_and_targets(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    """
    Prepares predictor features X and target series dict.
    
    CRITICAL: Explicitly drops ['LAD', 'LCX', 'RCA', 'Cath'] from X
    to strictly eliminate data leakage across overall CAD and individual vessel stenosis.
    
    Returns:
        X: DataFrame of predictor features (without LAD, LCX, RCA, Cath)
        targets: Dictionary containing binary Series for:
                 - 'CAD' (from Cath)
                 - 'LAD'
                 - 'LCX'
                 - 'RCA'
    """
    df_clean = df.copy()

    # Verify all leakage columns are present in dataframe
    for col in LEAKAGE_COLUMNS:
        if col not in df_clean.columns:
            raise ValueError(f"Expected column '{col}' not found in dataset columns: {list(df_clean.columns)}")

    # Extract target series
    targets = {}
    
    # 1. Overall CAD status from 'Cath' column
    # Cath: 'CAD' -> 1, 'Normal' -> 0
    cath_series = df_clean['Cath'].astype(str).str.strip()
    targets['CAD'] = cath_series.map(TARGET_MAPPINGS['Cath'])
    if targets['CAD'].isnull().any():
        raise ValueError(f"Unmapped values found in Cath: {cath_series.unique()}")

    # 2. Vessel-specific stenosis targets
    for vessel in ['LAD', 'LCX', 'RCA']:
        vessel_series = df_clean[vessel].astype(str).str.strip()
        targets[vessel] = vessel_series.map(TARGET_MAPPINGS[vessel])
        if targets[vessel].isnull().any():
            raise ValueError(f"Unmapped values found in {vessel}: {vessel_series.unique()}")

    # 3. Model input features X: EXPLICITLY DROP ALL 4 TARGET LEAKAGE COLUMNS
    X = df_clean.drop(columns=LEAKAGE_COLUMNS)

    # Double-check assertion for target leakage
    for col in LEAKAGE_COLUMNS:
        assert col not in X.columns, f"TARGET LEAKAGE DETECTED: {col} is still present in feature set X!"

    return X, targets
