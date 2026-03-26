from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd


REQUIRED_COLUMNS = {
    "restaurant_id",
    "name",
    "address",
    "location",
    "cuisines",
    "rating",
    "votes",
    "cost_for_two",
    "online_order",
    "book_table",
    "url",
}


def load_restaurants_parquet(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load Phase 1 cleaned restaurants parquet.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Parquet not found: {p}")
    df = pd.read_parquet(p)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in parquet: {sorted(missing)}")
    return df

