from pathlib import Path

import pandas as pd

from restaurant_reco.data_pipeline.build import build
from restaurant_reco.data_pipeline.ingest import DatasetSource


def test_build_writes_parquet_and_data_dict(tmp_path: Path, monkeypatch):
    # Avoid network calls by monkeypatching the dataset loader.
    from restaurant_reco.data_pipeline import ingest as ingest_mod

    def fake_load_raw(_source: DatasetSource):
        return pd.DataFrame(
            [
                {
                    "name": "B",
                    "address": "Addr2",
                    "location": "Loc2",
                    "cuisines": "Italian",
                    "rate": "3.8/5",
                    "approx_cost(for two people)": "1,200",
                    "votes": 10,
                    "online_order": "Yes",
                    "book_table": "Yes",
                    "url": "http://example.com/b",
                }
            ]
        )

    monkeypatch.setattr(ingest_mod, "load_raw_zomato_dataset", fake_load_raw)

    out_path = tmp_path / "restaurants.parquet"
    dict_path = tmp_path / "DATA_DICTIONARY.md"

    build(out_path=out_path, data_dict_path=dict_path, source=DatasetSource())

    assert out_path.exists()
    assert dict_path.exists()

    df = pd.read_parquet(out_path)
    assert len(df) == 1
    assert df.iloc[0]["name"] == "B"

    md = dict_path.read_text(encoding="utf-8")
    assert "Data Dictionary" in md
    assert "`restaurant_id`" in md

