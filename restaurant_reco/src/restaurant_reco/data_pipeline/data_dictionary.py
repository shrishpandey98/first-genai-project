from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FieldDef:
    name: str
    type: str
    description: str


CANONICAL_FIELDS: list[FieldDef] = [
    FieldDef("restaurant_id", "string", "Stable ID derived from name+address+location (sha1)."),
    FieldDef("name", "string", "Restaurant name (trimmed)."),
    FieldDef("address", "string", "Address (trimmed)."),
    FieldDef("location", "string", "Area/neighborhood (trimmed)."),
    FieldDef("cuisines", "list[string]", "List of cuisines parsed from comma-separated string."),
    FieldDef("rating", "float | null", "Numeric rating parsed from 'rate' like '4.1/5'."),
    FieldDef("votes", "int | null", "Vote count (nullable integer)."),
    FieldDef("cost_for_two", "int | null", "Approx cost for two, parsed from 'approx_cost(for two people)'."),
    FieldDef("online_order", "bool | null", "Parsed from Yes/No (nullable)."),
    FieldDef("book_table", "bool | null", "Parsed from Yes/No (nullable)."),
    FieldDef("url", "string", "Restaurant URL (trimmed)."),
]


def render_data_dictionary_markdown() -> str:
    lines: list[str] = []
    lines.append("## Data Dictionary (Processed Restaurants)\n")
    lines.append("This schema is produced by the Phase 1 pipeline from the Hugging Face Zomato dataset.\n")
    lines.append("### Canonical fields\n")
    lines.append("| Field | Type | Description |")
    lines.append("| --- | --- | --- |")
    for f in CANONICAL_FIELDS:
        lines.append(f"| `{f.name}` | {f.type} | {f.description} |")
    lines.append("\n### Cleaning/normalization rules\n")
    lines.append("- **rating**: parsed from strings like `4.1/5`; values like `NEW`, `-`, empty → `null`.")
    lines.append("- **cost_for_two**: digits-only parse from values like `1,200`; missing/unparseable → `null`.")
    lines.append("- **cuisines**: split by comma, trimmed; missing → empty list.")
    lines.append("- **online_order/book_table**: `Yes/No` → boolean; unknown → `null`.")
    lines.append("- **deduplication**: drop duplicates by `restaurant_id`.")
    lines.append("")
    return "\n".join(lines)

