"""
Preprocessing Module for Clinical Features.

Constructs leakage-free scikit-learn transformers for numeric scaling,
categorical encoding, and zero-variance feature elimination.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline


class ZeroVarianceDropper(BaseEstimator, TransformerMixin):
    """Drops columns that have constant values across all training samples."""
    def __init__(self):
        self.non_constant_cols_ = []

    def fit(self, X: pd.DataFrame, y=None):
        if isinstance(X, np.ndarray):
            # Assume 2D array
            stds = np.nanstd(X, axis=0)
            self.non_constant_cols_ = [i for i, s in enumerate(stds) if s > 1e-7]
        else:
            self.non_constant_cols_ = [col for col in X.columns if X[col].nunique() > 1]
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.non_constant_cols_]
        elif isinstance(X, np.ndarray):
            return X[:, self.non_constant_cols_]
        return X


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """
    Builds a robust ColumnTransformer for numerical and categorical features.
    
    Returns:
        preprocessor: ColumnTransformer instance
        numeric_cols: List of numeric column names
        categorical_cols: List of categorical column names
    """
    # Identify column types
    numeric_cols = []
    categorical_cols = []

    for col in X.columns:
        # Check if column is numeric or can be cast to float
        try:
            pd.to_numeric(X[col])
            numeric_cols.append(col)
        except (ValueError, TypeError):
            categorical_cols.append(col)

    # Preprocessing pipelines
    numeric_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('onehot', OneHotEncoder(drop='if_binary', sparse_output=False, handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_cols),
            ('cat', categorical_pipeline, categorical_cols)
        ],
        remainder='drop'
    )

    return preprocessor, numeric_cols, categorical_cols


def get_feature_names_from_preprocessor(preprocessor: ColumnTransformer) -> List[str]:
    """
    Extracts human-readable feature names post-transformation.
    Useful for SHAP, LIME, and coefficient interpretability.
    """
    feature_names = []
    try:
        feature_names = list(preprocessor.get_feature_names_out())
        # Clean prefix names e.g. "num__Age" -> "Age", "cat__Sex_Male" -> "Sex_Male"
        clean_names = []
        for name in feature_names:
            if name.startswith("num__"):
                clean_names.append(name.replace("num__", ""))
            elif name.startswith("cat__"):
                clean_names.append(name.replace("cat__", ""))
            else:
                clean_names.append(name)
        return clean_names
    except Exception:
        # Fallback if get_feature_names_out fails
        return [f"Feature_{i}" for i in range(100)]
