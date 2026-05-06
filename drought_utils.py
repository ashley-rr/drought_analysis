"""
Drought Data Processing Functions
Extracted from cs152drought.ipynb for use in Streamlit dashboard
"""

import numpy as np
import pandas as pd
from feature_engine.outliers import Winsorizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ── Constants ──────────────────────────────────────────────────────────────────
SCALED_DROUGHT_LABELS = {
    0: "None",
    1: "D0 Abnormal",
    2: "D1 Moderate",
    3: "D2 Severe"
}

DROUGHT_COLORS = {
    0: "#4a9e6c",
    1: "#f0c040",
    2: "#c03020",
    3: "#800010"
}

# ── Data Loading & Preprocessing ───────────────────────────────────────────────
def load_and_preprocess_data(filepath):
    """
    Load CSV and perform initial preprocessing.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with initial preprocessing applied
    """
    df = pd.read_csv(filepath)
    
    # Drop null scores
    df = df[df['score'].notnull()].reset_index(drop=True)
    
    # Extract date features
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    
    # Round scores to integers
    df['score'] = df['score'].round().astype(int)
    df['score'] = df['score'].clip(0, 3)
    
    # Standardize column names
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    
    return df


def apply_outlier_treatment(df, measures_list=None):
    """
    Apply Winsorizer to handle outliers in numeric features.
    
    Args:
        df: Input DataFrame
        measures_list: List of columns to treat for outliers (if None, auto-detect)
        
    Returns:
        DataFrame with outliers treated
    """
    if measures_list is None:
        # Auto-detect numeric columns excluding metadata and target
        measures_list = [x for x in df.select_dtypes(include=["int64", "float64", "int32"]).columns]
        remove_list = ["fips", "score", "year", "month", "day"]
        measures_list = [i for i in measures_list if i not in remove_list]
    
    outlier = Winsorizer(
        capping_method="gaussian",
        tail="both",
        fold=3,
        variables=measures_list,
        missing_values="ignore"
    )
    
    outlier.fit(df)
    df_treated = outlier.transform(df)
    
    return df_treated


def prepare_train_test_split(df, test_size=0.2, include_soil=True, soil_df=None):
    """
    Prepare train/test split with optional soil data merge.
    
    Args:
        df: Main DataFrame
        test_size: Proportion for test set
        include_soil: Whether to merge soil data
        soil_df: Soil DataFrame (required if include_soil=True)
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    # Drop date and fips from features (keep for merging if needed)
    feature_cols = [col for col in df.columns if col not in ["score", "date"]]
    
    X = df[feature_cols].copy()
    y = df["score"].copy()
    
    # Merge soil data if provided
    if include_soil and soil_df is not None and 'fips' in X.columns:
        X = X.merge(soil_df, on='fips', how='left')
    
    # Drop fips after merging
    if 'fips' in X.columns:
        X = X.drop(columns=['fips'])
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test


# ── Stratified Sampling ────────────────────────────────────────────────────────
def stratified_sample(df, total_rows, random_state=42):
    """
    Stratified sampling to maintain class distribution.
    
    Args:
        df: Input DataFrame with 'score' column
        total_rows: Target number of rows
        random_state: Random seed
        
    Returns:
        Sampled DataFrame
    """
    if len(df) <= total_rows:
        return df
    
    score_counts = df['score'].value_counts().sort_index()
    sampled_dfs = []
    
    for score in score_counts.index:
        score_data = df[df['score'] == score]
        proportion = len(score_data) / len(df)
        target_n = int(total_rows * proportion)
        
        if len(score_data) >= target_n:
            sampled = score_data.sample(n=target_n, random_state=random_state)
        else:
            sampled = score_data.sample(n=target_n, random_state=random_state, replace=True)
        
        sampled_dfs.append(sampled)
    
    result = pd.concat(sampled_dfs, ignore_index=True)
    result = result.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Ensure exact size
    if len(result) > total_rows:
        result = result.iloc[:total_rows]
    elif len(result) < total_rows:
        deficit = total_rows - len(result)
        extra = df.sample(n=deficit, random_state=random_state)
        result = pd.concat([result, extra], ignore_index=True)
    
    return result


# ── Visualization Helpers ──────────────────────────────────────────────────────
def get_class_distribution(y):
    """
    Get class distribution statistics.
    
    Args:
        y: Target series
        
    Returns:
        Dictionary with counts and percentages
    """
    counts = y.value_counts().sort_index()
    total = len(y)
    
    return {
        'counts': counts.to_dict(),
        'percentages': {k: (v / total * 100) for k, v in counts.items()},
        'total': total
    }


def get_feature_stats(df, exclude_cols=None):
    """
    Get summary statistics for numeric features.
    
    Args:
        df: DataFrame
        exclude_cols: Columns to exclude
        
    Returns:
        DataFrame with statistics
    """
    if exclude_cols is None:
        exclude_cols = ['score', 'fips', 'year', 'month', 'day']
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    return df[feature_cols].describe().T


def get_correlation_matrix(df, exclude_cols=None):
    """
    Calculate correlation matrix for numeric features.
    
    Args:
        df: DataFrame
        exclude_cols: Columns to exclude
        
    Returns:
        Correlation matrix DataFrame
    """
    if exclude_cols is None:
        exclude_cols = ['fips', 'year', 'month', 'day']
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    return df[feature_cols].corr()


# ── Complete Pipeline ──────────────────────────────────────────────────────────
def load_complete_dataset(data_path, soil_path=None, 
                         apply_outlier_treatment_flag=True,
                         sample_size=None):
    """
    Complete data loading pipeline (single dataset).
    
    Args:
        data_path: Path to CSV file (train or test)
        soil_path: Path to soil data CSV (optional)
        apply_outlier_treatment_flag: Whether to apply outlier treatment
        sample_size: Target size for dataset (None = use all)
        
    Returns:
        Dictionary with df, soil_df, and metadata
    """
    # Load data
    df = load_and_preprocess_data(data_path)
    
    # Apply outlier treatment
    if apply_outlier_treatment_flag:
        df = apply_outlier_treatment(df)
    
    # Sample if requested
    if sample_size is not None:
        df = stratified_sample(df, sample_size)
    
    # Load soil data if provided
    soil_df = None
    if soil_path is not None:
        soil_df = pd.read_csv(soil_path)
    
    return {
        'df': df,
        'soil_df': soil_df,
        'class_dist': get_class_distribution(df['score']),
        'is_demo': False
    }


# ── Demo Data Generator ────────────────────────────────────────────────────────
def generate_demo_data(n_samples=5000):
    """
    Generate synthetic demo data when real CSVs aren't available.
    Uses real FIPS codes with proper US coordinates.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        Dictionary with df, soil_df, and metadata
    """
    import sys
    import os
    
    # Try to import fips_coordinates
    try:
        from fips_coordinates import FIPS_COORDS
        real_fips_list = list(FIPS_COORDS.keys())
    except ImportError:
        # Fallback to major US counties if fips_coordinates not available
        real_fips_list = [
            "06037", "17031", "48201", "04013", "06073", "36047", "06059",
            "12086", "36081", "48113", "29095", "06065", "39035", "42101",
            "26163", "32003", "53033", "25025", "24510", "41051"
        ]
    
    rng = np.random.default_rng(42)
    
    # Use real FIPS codes
    fips = rng.choice(real_fips_list, size=n_samples, replace=True)
    dates = pd.date_range("2000-01-01", periods=n_samples, freq="D")
    
    df = pd.DataFrame({
        'fips': fips,
        'date': dates,
        'prectot': np.abs(rng.normal(3, 2, n_samples)),
        'ps': rng.normal(101, 0.5, n_samples),
        'qv2m': np.abs(rng.normal(8, 3, n_samples)),
        't2m': rng.normal(15, 8, n_samples),
        't2mdew': rng.normal(10, 8, n_samples),
        'ws10m': np.abs(rng.normal(4, 1.5, n_samples)),
        'score': rng.integers(0, 4, n_samples),
    })
    
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    
    # Add coordinates using fips_coordinates module
    try:
        from fips_coordinates import add_coordinates_to_dataframe
        df = add_coordinates_to_dataframe(df)
    except ImportError:
        # Fallback: approximate coords
        df['lat'] = rng.uniform(25, 49, n_samples)
        df['lon'] = rng.uniform(-125, -67, n_samples)
    
    # Generate soil data
    unique_fips = list(set(df['fips'].unique()))
    soil_df = pd.DataFrame({
        'fips': unique_fips,
        'clay': rng.uniform(10, 60, len(unique_fips)),
        'sand': rng.uniform(10, 70, len(unique_fips)),
    })
    
    return {
        'df': df,
        'soil_df': soil_df,
        'class_dist': get_class_distribution(df['score']),
        'is_demo': True
    }
