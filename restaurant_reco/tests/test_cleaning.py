import pandas as pd

from restaurant_reco.data_pipeline.cleaning import (
    clean_restaurants,
    parse_cost_for_two,
    parse_cuisines,
    parse_rating,
    parse_yes_no,
)


def test_parse_rating():
    assert parse_rating("4.1/5") == 4.1
    assert parse_rating(" 3.0 / 5 ") == 3.0
    assert parse_rating("NEW") is None
    assert parse_rating("-") is None
    assert parse_rating(None) is None


def test_parse_cost_for_two():
    assert parse_cost_for_two("800") == 800
    assert parse_cost_for_two("1,200") == 1200
    assert parse_cost_for_two("-") is None
    assert parse_cost_for_two(None) is None


def test_parse_yes_no():
    assert parse_yes_no("Yes") is True
    assert parse_yes_no("No") is False
    assert parse_yes_no("true") is True
    assert parse_yes_no("0") is False
    assert parse_yes_no("maybe") is None


def test_parse_cuisines():
    assert parse_cuisines("North Indian, Chinese") == ["North Indian", "Chinese"]
    assert parse_cuisines("") == []
    assert parse_cuisines(None) == []


def test_clean_restaurants_dedup_and_schema():
    raw = pd.DataFrame(
        [
            {
                "name": "A",
                "address": "Addr1",
                "location": "Loc1",
                "cuisines": "North Indian, Chinese",
                "rate": "4.1/5",
                "approx_cost(for two people)": "800",
                "votes": 100,
                "online_order": "Yes",
                "book_table": "No",
                "url": "http://example.com/a",
            },
            # Duplicate by name+address+location → same restaurant_id → should be dropped
            {
                "name": "A",
                "address": "Addr1",
                "location": "Loc1",
                "cuisines": "North Indian",
                "rate": "4.0/5",
                "approx_cost(for two people)": "900",
                "votes": 50,
                "online_order": "No",
                "book_table": "No",
                "url": "http://example.com/a2",
            },
        ]
    )

    cleaned, report = clean_restaurants(raw)
    assert report.input_rows == 2
    assert report.output_rows == 1
    assert report.dropped_duplicates == 1

    row = cleaned.iloc[0].to_dict()
    assert set(cleaned.columns) == {
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
    assert row["name"] == "A"
    assert row["rating"] == 4.1
    assert row["cost_for_two"] == 800
    assert row["online_order"] is True
    assert row["book_table"] is False
    assert row["cuisines"] == ["North Indian", "Chinese"]

