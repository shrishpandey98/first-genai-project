from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

import pandas as pd


_RATING_RE = re.compile(r"^\s*(?P<val>\d+(\.\d+)?)\s*/\s*5\s*$")


def parse_rating(rate: object) -> float | None:
    """
    Parse dataset `rate` values such as '4.1/5'.
    Returns None for missing/non-numeric ratings (e.g. 'NEW', '-', NaN).
    """
    if rate is None:
        return None
    if isinstance(rate, float) and math.isnan(rate):
        return None
    s = str(rate).strip()
    if not s or s.upper() == "NEW" or s == "-":
        return None
    m = _RATING_RE.match(s)
    if not m:
        return None
    return float(m.group("val"))


def parse_cost_for_two(cost: object) -> int | None:
    """
    Parse `approx_cost(for two people)` values such as '800', '1,200'.
    Returns None for missing/unparseable values.
    """
    if cost is None:
        return None
    if isinstance(cost, float) and math.isnan(cost):
        return None
    s = str(cost).strip()
    if not s or s == "-" or s.lower() == "nan":
        return None
    s = s.replace(",", "")
    digits = re.sub(r"[^\d]", "", s)
    if not digits:
        return None
    return int(digits)


def parse_yes_no(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    s = str(value).strip().lower()
    if not s:
        return None
    if s in {"yes", "y", "true", "1"}:
        return True
    if s in {"no", "n", "false", "0"}:
        return False
    return None


def parse_cuisines(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return []
    parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]


def stable_restaurant_id(name: str, address: str, location: str) -> str:
    raw = f"{name}||{address}||{location}".encode("utf-8", errors="ignore")
    return hashlib.sha1(raw).hexdigest()


@dataclass(frozen=True)
class CleaningReport:
    input_rows: int
    output_rows: int
    dropped_duplicates: int


def clean_restaurants(raw: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """
    Normalize and clean the Hugging Face Zomato dataset into a canonical form.
    Expected raw columns (best-effort): name, address, location, cuisines, rate,
    approx_cost(for two people), votes, online_order, book_table, url.
    """
    df = raw.copy()

    # Canonical columns (best effort if missing)
    for col in ["name", "address", "location", "cuisines", "rate", "approx_cost(for two people)", "votes", "online_order", "book_table", "url"]:
        if col not in df.columns:
            df[col] = None

    df["rating"] = df["rate"].map(parse_rating)
    df["cost_for_two"] = df["approx_cost(for two people)"].map(parse_cost_for_two)
    df["online_order_bool"] = df["online_order"].map(parse_yes_no)
    df["book_table_bool"] = df["book_table"].map(parse_yes_no)
    df["cuisines_list"] = df["cuisines"].map(parse_cuisines)

    # Normalize strings for ID + display
    df["name"] = df["name"].astype(str).fillna("").map(lambda x: x.strip())
    df["address"] = df["address"].astype(str).fillna("").map(lambda x: x.strip())
    df["location"] = df["location"].astype(str).fillna("").map(lambda x: x.strip())

    df["restaurant_id"] = df.apply(
        lambda r: stable_restaurant_id(r["name"], r["address"], r["location"]),
        axis=1,
    )

    before = len(df)
    df = df.drop_duplicates(subset=["restaurant_id"], keep="first")
    after = len(df)

    out = pd.DataFrame(
        {
            "restaurant_id": df["restaurant_id"],
            "name": df["name"],
            "address": df["address"],
            "location": df["location"],
            "cuisines": df["cuisines_list"],
            "rating": df["rating"],
            "votes": pd.to_numeric(df["votes"], errors="coerce").astype("Int64"),
            "cost_for_two": df["cost_for_two"].astype("Int64"),
            "online_order": df["online_order_bool"].astype("boolean"),
            "book_table": df["book_table_bool"].astype("boolean"),
            "url": df["url"].astype(str).fillna("").map(lambda x: x.strip()),
        }
    )

    report = CleaningReport(
        input_rows=before,
        output_rows=len(out),
        dropped_duplicates=(before - after),
    )
    return out, report

