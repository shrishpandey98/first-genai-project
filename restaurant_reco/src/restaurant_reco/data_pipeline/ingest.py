from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from datasets import load_dataset


@dataclass(frozen=True)
class DatasetSource:
    hf_dataset: str = "ManikaSaini/zomato-restaurant-recommendation"
    hf_split: str = "train"


def load_raw_zomato_dataset(source: DatasetSource = DatasetSource()) -> pd.DataFrame:
    """
    Load the Zomato restaurant dataset from Hugging Face into a pandas DataFrame.
    """
    ds = load_dataset(source.hf_dataset, split=source.hf_split)
    return ds.to_pandas()

