"""
Dataset Ingestion, Cleaning, and Leakage-Free Preprocessing Pipeline
Author: Rishindra Mateti (Wright State University)

Implements rigorous data hygiene for the UCI Drug Consumption benchmark:
1. Excludes Semeron fictitious drug overclaimers (N=8), yielding N=1,877 verified records.
2. Isolates dietary commodities (Caffeine, Chocolate) and legal substances (Alcohol, Nicotine).
3. Defines four literature-grounded illicit substance classes for past-year consumption (CL3-CL6):
   - Cannabinoids: Cannabis
   - Central Nervous System Stimulants: Cocaine or Amphetamines
   - Psychedelics: Psilocybin Mushrooms or LSD
   - Depressants / Anxiolytics: Benzodiazepines
4. Formulates a past-year illicit Polysubstance Co-Involvement Score (0 to 4 classes).
5. Provides a strictly leakage-free standard scaler with fold-specific parameter fitting.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd


COLUMN_NAMES = [
    'ID', 'Age', 'Gender', 'Education', 'Country', 'Ethnicity',
    'Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS',
    'Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 'Choc',
    'Coke', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 'Legalh', 'LSD',
    'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA'
]

PSYCHOMETRIC_TRAITS = ['Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS']
DEMOGRAPHIC_FEATURES = ['Age', 'Gender', 'Education', 'Country', 'Ethnicity']
CORE_PREDICTORS = ['Age', 'Gender', 'Education', 'Nscore', 'Impulsive', 'Ascore', 'Escore', 'Oscore', 'Cscore', 'SS']
CLUSTERING_FEATURES = ['Age', 'Impulsive', 'SS']

ALL_DRUG_COLS = [
    'Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 'Choc',
    'Coke', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 'Legalh', 'LSD',
    'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA'
]


def get_default_data_path() -> str:
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data',
        'drug_consumption.data'
    )


def load_cleaned_dataset(data_path: Optional[str] = None) -> pd.DataFrame:
    """
    Ingests and cleans UCI Drug Consumption dataset.
    Removes respondents who reported using the fictitious drug 'Semeron' (Semer != 'CL0').
    Parses past-year consumption targets (CL3-CL6: used within last day, week, month, or year).
    """
    path = data_path or get_default_data_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")

    df = pd.read_csv(path, header=None, names=COLUMN_NAMES)
    df = df.drop(columns=['ID'])

    # Data Hygiene Filter: Remove Semeron fictitious drug overclaimers
    initial_count = len(df)
    df = df[df['Semer'] == 'CL0'].copy()
    cleaned_count = len(df)
    excluded_count = initial_count - cleaned_count

    # Convert all substance levels CL0 - CL6 to integer indices 0 - 6
    for drug in ALL_DRUG_COLS:
        df[f'{drug}_level'] = df[drug].str.replace('CL', '').astype(int)

    # Define Past-Year Consumption Targets (CL3 to CL6)
    # 1. Cannabinoids: Cannabis
    df['Target_Cannabis'] = (df['Cannabis_level'] >= 3).astype(int)

    # 2. CNS Stimulants: Cocaine or Amphetamines
    df['Target_Stimulants'] = ((df['Coke_level'] >= 3) | (df['Amphet_level'] >= 3)).astype(int)

    # 3. Psychedelics: Psilocybin Mushrooms or LSD
    df['Target_Psychedelics'] = ((df['Mushrooms_level'] >= 3) | (df['LSD_level'] >= 3)).astype(int)

    # 4. Depressants / Anxiolytics: Benzodiazepines
    df['Target_Depressants'] = (df['Benzos_level'] >= 3).astype(int)

    # Polysubstance Co-Involvement Score: count of distinct past-year illicit classes (0 to 4)
    df['Polysubstance_Score'] = (
        df['Target_Cannabis'] +
        df['Target_Stimulants'] +
        df['Target_Psychedelics'] +
        df['Target_Depressants']
    )

    df.attrs['excluded_overclaimers'] = excluded_count
    return df


class LeakageFreeStandardScaler:
    """
    Fits mean and standard deviation strictly on training partitions to prevent data leakage.
    """
    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> LeakageFreeStandardScaler:
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ == 0.0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler has not been fitted.")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def get_stratified_outer_split(
    df: pd.DataFrame,
    target_col: str = 'Target_Cannabis',
    test_size: float = 0.2,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LeakageFreeStandardScaler]:
    """
    Partitions dataset into outer training (80%) and holdout test (20%) sets.
    Features are standardized strictly using parameters learned on the training partition.
    """
    X_raw = df[CORE_PREDICTORS].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=int)

    rng = np.random.default_rng(random_seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    pos_test = int(len(pos_idx) * test_size)
    neg_test = int(len(neg_idx) * test_size)

    test_idx = np.concatenate([pos_idx[:pos_test], neg_idx[:neg_test]])
    train_idx = np.concatenate([pos_idx[pos_test:], neg_idx[neg_test:]])

    rng.shuffle(test_idx)
    rng.shuffle(train_idx)

    X_train_raw = X_raw[train_idx]
    y_train = y[train_idx]
    X_test_raw = X_raw[test_idx]
    y_test = y[test_idx]

    scaler = LeakageFreeStandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    return X_train, y_train, X_test, y_test, scaler
