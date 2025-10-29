import os
import pandas as pd
from datetime import datetime

def date_slug(d: str) -> str:
    return d

def save_df(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def load_csv_if_exists(path: str):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def cache_path(prefix: str, date: str, suffix: str = ".csv") -> str:
    slug = date_slug(date)
    return os.path.join("data", "cache", f"{prefix}_{slug}{suffix}")
