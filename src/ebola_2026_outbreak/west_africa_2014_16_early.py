"""Confirmed West Africa 2014-16 cases before the WHO/HDX series begins.

The WHO data in `west_africa_2014_16.py` starts on 29 August 2014. For the
months before, this uses OCHA ROWCA's manually compiled national confirmed
counts (ministry sitreps and WHO reports), keeping only dates before the WHO
series starts so the two never overlap.

Source: HDX, "Sub-national time series data on Ebola cases and deaths in
Guinea, Liberia, Sierra Leone, Nigeria, Senegal and Mali since March 2014"
(https://data.humdata.org/dataset/rowca-ebola-cases).

The compilation has data-entry errors, such as Liberia at 342 then 22 on
28 August 2014. A value far below the country's running maximum breaks its
cumulative sequence, so it is dropped and logged.
Ministry figures run 2-4 days ahead of WHO's, so the joined series steps
down slightly at the join, and the size of that step is logged.
"""

import logging
from datetime import date
from pathlib import Path

import polars as pl

from ebola_2026_outbreak.aggregate import forward_fill_cumulative
from ebola_2026_outbreak.fetch import get
from ebola_2026_outbreak.west_africa_2014_16 import (
    FINAL_INDICATOR,
    FINAL_PATH,
    OUTBREAK,
    PROCESSED_DIR,
    ROOT,
)

logger = logging.getLogger(__name__)

RAW_PATH = ROOT / "data" / "raw" / f"{OUTBREAK}_rowca_ebola_cases.xlsx"
SOURCE_URL = (
    "https://data.humdata.org/dataset/f326cf9d-8e37-437d-9b85-5ea3d36e3d39/"
    "resource/9b68ab69-e0b3-4ff5-b2d8-90bf446acd52/download/data-ebola-public.xlsx"
)
OUTPUT_PATH = PROCESSED_DIR / f"{OUTBREAK}_early_confirmed_cases.csv"

SHEET = "ROWCA Ebola All Sec Review"
CATEGORY = "Confirmed cases"
NATIONAL = "National"
SOURCE_LABEL = "OCHA ROWCA"
# A value more than 10% below the country's running maximum is an entry
# error; small dips such as Guinea's 208 -> 207 are reclassifications.
MAX_DIP_SHARE = 0.10


def load_rowca(path: Path, *, before: date) -> pl.DataFrame:
    """Load ROWCA national confirmed counts dated before `before`.

    Downloads the workbook if absent. Later rows are dropped first, so entry
    problems outside the window (nulls in November 2014, conflicting
    duplicates in 2015) do not affect this series.

    Args:
        path: Local workbook path.
        before: First date of the WHO series.

    Returns:
        `country`, `date`, `value` rows.

    Raises:
        ValueError: If expected columns are missing or values are null.
    """
    if not path.exists():
        path.write_bytes(get(SOURCE_URL).content)
        logger.info("Downloaded %s", path.relative_to(ROOT))

    raw = pl.read_excel(path, sheet_name=SHEET)
    missing = {"Country", "Localite", "Category", "Value", "Date"} - set(raw.columns)
    if missing:
        raise ValueError(f"ROWCA sheet missing columns: {missing}")

    df = (
        raw.filter(pl.col("Category") == CATEGORY, pl.col("Localite") == NATIONAL)
        .select(
            pl.col("Country").alias("country"),
            pl.col("Date").cast(pl.Date).alias("date"),
            pl.col("Value").alias("value"),
        )
        .filter(pl.col("date") < before)
    )
    if df.null_count().sum_horizontal().item():
        raise ValueError(f"Nulls in ROWCA confirmed rows: {df.null_count()}")
    return df


def deduplicate(df: pl.DataFrame) -> pl.DataFrame:
    """Collapse repeated (country, date) rows, raising if their values differ."""
    conflicts = (
        df.group_by("country", "date")
        .agg(pl.col("value").n_unique().alias("n"))
        .filter(pl.col("n") > 1)
    )
    if conflicts.height:
        raise ValueError(f"Conflicting ROWCA values: {conflicts.rows()}")
    return df.unique(subset=["country", "date"])


def drop_entry_errors(df: pl.DataFrame) -> pl.DataFrame:
    """Drop values far below the country's running maximum, logging each one."""
    kept: list[dict[str, object]] = []
    for (country,), group in df.sort("date").group_by("country", maintain_order=True):
        running_max = 0.0
        for row in group.iter_rows(named=True):
            if row["value"] < running_max * (1 - MAX_DIP_SHARE):
                logger.warning(
                    "Dropping %s %s: %s after a maximum of %s",
                    country,
                    row["date"],
                    row["value"],
                    running_max,
                )
                continue
            running_max = max(running_max, row["value"])
            kept.append(row)
    return pl.DataFrame(kept, schema=df.schema)


def early_series(df: pl.DataFrame) -> pl.DataFrame:
    """Sum cleaned country counts by date.

    Each country's last value is carried forward first, as in the WHO series.
    """
    filled = forward_fill_cumulative(
        df.with_columns(indicator=pl.lit(CATEGORY)).select(
            "date", "country", "indicator", "value"
        ),
        entity="country",
        cumulative_prefix=CATEGORY,
    )
    return (
        filled.group_by("date")
        .agg(pl.col("value").sum())
        .sort("date")
        .select(
            pl.lit(OUTBREAK).alias("outbreak"),
            "date",
            pl.lit(FINAL_INDICATOR).alias("indicator"),
            pl.col("value"),
            pl.lit(SOURCE_LABEL).alias("source"),
        )
    )


def main() -> None:
    """Write ROWCA confirmed counts for the dates before the WHO series."""
    who = pl.read_csv(FINAL_PATH, try_parse_dates=True).sort("date")
    who_start: date = who["date"].to_list()[0]

    rowca = drop_entry_errors(deduplicate(load_rowca(RAW_PATH, before=who_start)))
    for country, first in (
        rowca.group_by("country").agg(pl.col("date").min()).sort("date").rows()
    ):
        logger.info("%s: first ROWCA confirmed count on %s", country, first)

    early = early_series(rowca)
    rowca_last: float = early["value"].to_list()[-1]
    who_first: float = who["value"].to_list()[0]
    logger.info(
        "Join on %s: ROWCA %.0f on %s, WHO %.0f (step %+.0f)",
        who_start,
        rowca_last,
        early["date"].to_list()[-1],
        who_first,
        who_first - rowca_last,
    )
    early.write_csv(OUTPUT_PATH)
    logger.info(
        "Wrote %d rows (%s to %s) to %s",
        early.height,
        early["date"].to_list()[0],
        early["date"].to_list()[-1],
        OUTPUT_PATH.name,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
