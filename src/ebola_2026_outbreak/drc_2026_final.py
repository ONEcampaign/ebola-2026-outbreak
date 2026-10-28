"""Build the national 2026 DRC series from the sitrep province counts.

A reference and cross-check output that no chart reads. The outbreak
comparison chart sums WHO's figures (see `chart_tables`). The files keep the
West Africa 2014-16 shape and final indicator label:

    - Intermediate: one row per (date, indicator), value summed across provinces.
    - Final: cumulative confirmed cases only, one row per date.

Dates with no parsed sitrep are backfilled with INRB's digitised national
banner figures (see `drc_2026_inrb`), and the `source` column of the
intermediate records which rows came from where. On shared dates the parsed
sitrep wins and INRB is only a cross-check.

The national totals are cross-checked against the WHO table, whose "Data as
of" date matches the sitrep date.
"""

import logging

import polars as pl

from ebola_2026_outbreak.aggregate import forward_fill_cumulative
from ebola_2026_outbreak.drc_2026_inrb import OUTPUT_PATH as INRB_PATH
from ebola_2026_outbreak.drc_2026_inrb import SCHEMA as INRB_SCHEMA
from ebola_2026_outbreak.drc_2026_sitreps_parse import (
    INDICATORS,
    OUTBREAK,
    ROOT,
)
from ebola_2026_outbreak.drc_2026_sitreps_parse import (
    OUTPUT_PATH as PROVINCE_PATH,
)
from ebola_2026_outbreak.drc_2026_sitreps_parse import (
    SCHEMA as PROVINCE_SCHEMA,
)
from ebola_2026_outbreak.drc_2026_who import CONFIRMED_CASES as WHO_CASES
from ebola_2026_outbreak.drc_2026_who import DRC_COUNTRY as WHO_COUNTRY
from ebola_2026_outbreak.drc_2026_who import OUTPUT_PATH as WHO_PATH
from ebola_2026_outbreak.drc_2026_who import SCHEMA as WHO_SCHEMA

logger = logging.getLogger(__name__)

PROCESSED_DIR = ROOT / "data" / "processed"
INTERMEDIATE_PATH = PROCESSED_DIR / f"{OUTBREAK}_indicators_by_date.csv"
FINAL_PATH = PROCESSED_DIR / f"{OUTBREAK}_cumulative_confirmed_cases.csv"

CUMULATIVE_PREFIX = "Cumulative"
SITREP_CASES = INDICATORS["cases"]
# Matches the West Africa final output so both series share one label.
FINAL_INDICATOR = "Cumulative number of confirmed Ebola cases"
SOURCE_SITREP, SOURCE_INRB = "sitrep_pdf", "inrb"


def sum_by_date_indicator(df: pl.DataFrame) -> pl.DataFrame:
    """Sum values across provinces for each date and indicator.

    A null new-case count ("ND" in the sitrep) makes that date's national
    new-case sum null instead of undercounting it.
    """
    return (
        df.group_by("date", "indicator")
        .agg(pl.when(pl.col("value").null_count() == 0).then(pl.col("value").sum()))
        .sort("date", "indicator")
        .select(
            pl.lit(OUTBREAK).alias("outbreak"),
            "date",
            "indicator",
            "value",
            pl.lit(SOURCE_SITREP).alias("source"),
        )
    )


def log_incomplete_dates(rows: pl.DataFrame, *, expected: set[str]) -> None:
    """Warn for each date in `rows` with no value for an `expected` indicator."""
    present = (
        rows.group_by("date")
        .agg(pl.col("indicator").filter(pl.col("value").is_not_null()))
        .sort("date")
    )
    for day, indicators in present.iter_rows():
        absent = sorted(expected - set(indicators))
        if absent:
            logger.warning("INRB backfill on %s has no %s", day, absent)


def backfill_from_inrb(
    *, intermediate: pl.DataFrame, inrb: pl.DataFrame
) -> pl.DataFrame:
    """Add INRB national rows for dates with no parsed sitrep.

    Args:
        intermediate: Sitrep national rows from `sum_by_date_indicator`.
        inrb: INRB national rows (see `drc_2026_inrb.SCHEMA`).

    Returns:
        `intermediate` plus the INRB rows on its missing dates, sorted.

    Raises:
        ValueError: If a (date, indicator) pair appears more than once after
            the merge.
    """
    missing = inrb.filter(~pl.col("date").is_in(intermediate["date"].implode()))
    added = missing.drop_nulls("value").select(
        "outbreak", "date", "indicator", "value", pl.lit(SOURCE_INRB).alias("source")
    )
    log_incomplete_dates(missing, expected=set(inrb["indicator"]))

    merged = pl.concat([intermediate, added]).sort("date", "indicator")
    duplicated = merged.filter(pl.struct("date", "indicator").is_duplicated())
    if duplicated.height:
        pairs = duplicated.select("date", "indicator").unique().sort("date").rows()
        raise ValueError(f"Duplicate (date, indicator) after INRB backfill: {pairs}")
    if added.height:
        logger.info(
            "Backfilled %d rows on %d dates (%s to %s) from INRB",
            added.height,
            added["date"].n_unique(),
            added["date"].min(),
            added["date"].max(),
        )
    return merged


def cross_check_inrb(*, intermediate: pl.DataFrame, inrb: pl.DataFrame) -> None:
    """Log where parsed sitrep totals differ from INRB's on shared dates."""
    compared = intermediate.filter(pl.col("source") == SOURCE_SITREP).join(
        inrb.select("date", "indicator", pl.col("value").alias("inrb_value")),
        on=["date", "indicator"],
        how="inner",
    )
    mismatches = compared.filter(pl.col("value") != pl.col("inrb_value"))
    logger.info(
        "INRB cross-check: %d shared rows, %d match",
        compared.height,
        compared.height - mismatches.height,
    )
    for row in mismatches.iter_rows(named=True):
        logger.warning(
            "%s %s: sitrep %d, INRB %d",
            row["date"],
            row["indicator"].lower(),
            row["value"],
            row["inrb_value"],
        )


def build_final(intermediate: pl.DataFrame) -> pl.DataFrame:
    """Keep national cumulative confirmed cases, relabelled to match West Africa."""
    final = (
        intermediate.filter(pl.col("indicator") == SITREP_CASES)
        .with_columns(indicator=pl.lit(FINAL_INDICATOR))
        .select("outbreak", "date", "indicator", "value")
    )
    if final.is_empty():
        raise ValueError(f"No rows for indicator {SITREP_CASES!r}")
    if final["date"].is_duplicated().any():
        raise ValueError("Duplicate dates in final series")

    dips = final.filter(pl.col("value").diff() < 0)
    for row in dips.iter_rows(named=True):
        logger.warning("National cumulative cases fall on %s", row["date"])
    return final


def cross_check_who(final: pl.DataFrame, who: pl.DataFrame) -> None:
    """Log where the sitrep national total differs from WHO's DRC figure.

    Args:
        final: Final sitrep series.
        who: WHO table rows (all countries and indicators).

    Raises:
        ValueError: If `who` has no DRC confirmed-case rows, which means the
            labels do not match.
    """
    who_cases = who.filter(
        pl.col("country") == WHO_COUNTRY, pl.col("indicator") == WHO_CASES
    ).select("date", pl.col("value").alias("who_value"))
    if who_cases.is_empty():
        raise ValueError(f"No WHO rows for {WHO_COUNTRY!r} / {WHO_CASES!r}")
    compared = final.join(who_cases, on="date", how="inner")
    mismatches = compared.filter(pl.col("value") != pl.col("who_value"))
    logger.info(
        "WHO cross-check: %d shared dates, %d match",
        compared.height,
        compared.height - mismatches.height,
    )
    for row in mismatches.iter_rows(named=True):
        logger.warning(
            "%s: sitrep total %d, WHO %s", row["date"], row["value"], row["who_value"]
        )


def main() -> None:
    """Build the DRC intermediate and final outputs."""
    provinces = pl.read_csv(PROVINCE_PATH, schema=PROVINCE_SCHEMA)
    filled = forward_fill_cumulative(
        provinces.select("date", "province", "indicator", "value"),
        entity="province",
        cumulative_prefix=CUMULATIVE_PREFIX,
    )

    intermediate = sum_by_date_indicator(filled)
    if INRB_PATH.exists():
        inrb = pl.read_csv(INRB_PATH, schema=INRB_SCHEMA)
        cross_check_inrb(intermediate=intermediate, inrb=inrb)
        intermediate = backfill_from_inrb(intermediate=intermediate, inrb=inrb)
    else:
        logger.warning("No INRB series at %s; skipping backfill", INRB_PATH.name)
    intermediate.write_csv(INTERMEDIATE_PATH)
    logger.info("Wrote %d rows to %s", intermediate.height, INTERMEDIATE_PATH.name)

    final = build_final(intermediate)
    final.write_csv(FINAL_PATH)
    logger.info("Wrote %d rows to %s", final.height, FINAL_PATH.name)

    if WHO_PATH.exists():
        cross_check_who(final, pl.read_csv(WHO_PATH, schema=WHO_SCHEMA))
    else:
        logger.warning("No WHO table at %s; skipping cross-check", WHO_PATH.name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
