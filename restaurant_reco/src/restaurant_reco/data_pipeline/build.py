from __future__ import annotations

import argparse
from pathlib import Path

from restaurant_reco.data_pipeline.cleaning import clean_restaurants
from restaurant_reco.data_pipeline.data_dictionary import render_data_dictionary_markdown
from restaurant_reco.data_pipeline import ingest
from restaurant_reco.data_pipeline.ingest import DatasetSource


def build(out_path: Path, data_dict_path: Path, source: DatasetSource) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data_dict_path.parent.mkdir(parents=True, exist_ok=True)

    # Refer via module to make testing/monkeypatching straightforward.
    raw = ingest.load_raw_zomato_dataset(source)
    cleaned, report = clean_restaurants(raw)
    cleaned.to_parquet(out_path, index=False)

    data_dict_path.write_text(render_data_dictionary_markdown(), encoding="utf-8")

    print(
        f"Wrote {len(cleaned)} records to {out_path} "
        f"(input_rows={report.input_rows}, dropped_duplicates={report.dropped_duplicates})."
    )
    print(f"Wrote data dictionary to {data_dict_path}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cleaned restaurant dataset artifacts.")
    parser.add_argument("--out", required=True, help="Output parquet path.")
    parser.add_argument("--data-dict", required=True, help="Output data dictionary markdown path.")
    parser.add_argument(
        "--hf-dataset",
        default="ManikaSaini/zomato-restaurant-recommendation",
        help="Hugging Face dataset name.",
    )
    parser.add_argument("--hf-split", default="train", help="Dataset split.")
    args = parser.parse_args()

    build(
        out_path=Path(args.out),
        data_dict_path=Path(args.data_dict),
        source=DatasetSource(hf_dataset=args.hf_dataset, hf_split=args.hf_split),
    )


if __name__ == "__main__":
    main()

