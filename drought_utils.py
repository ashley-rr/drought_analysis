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
from dashboard import fips_to_name 
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
                         sample_size=None):
    """
    Complete data loading pipeline (single dataset).
    
    Args:
        data_path: Path to CSV file (train or test)
        soil_path: Path to soil data CSV (optional)
        sample_size: Target size for dataset (None = use all)
        
    Returns:
        Dictionary with df, soil_df, and metadata
    """
    # Load data
    df = load_and_preprocess_data(data_path)
 
    return {
        'df': df,
        'class_dist': get_class_distribution(df['score']),
    }


def sample_data(csv_path="train.csv", n_samples=5000):
    
    df = pd.read_csv(csv_path)

    # Randomly sample rows
    df = df.sample(n=min(n_samples, len(df)), random_state=42)

    # Ensure datetime
    df['date'] = pd.to_datetime(df['date'])

    # Add time columns
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day

    # Add coordinates if possible
    try:
        from fips_coordinates import add_coordinates_to_dataframe
        df = add_coordinates_to_dataframe(df)
    except ImportError:
        pass

    # Soil data (optional)
    soil_df = None

    return {
        'df': df,
        'soil_df': soil_df,
        'class_dist': get_class_distribution(df['score']),
        'is_demo': True
    }

def add_state_column(df):
    """
    Add a 'state' column to dataframe based on FIPS codes.
    
    Args:
        df: DataFrame with 'fips' column
        
    Returns:
        DataFrame with added 'state' column
    """
    df = df.copy()
    if 'fips' in df.columns:
        df['state'] = df['fips'].apply(fips_to_name)
    return df
