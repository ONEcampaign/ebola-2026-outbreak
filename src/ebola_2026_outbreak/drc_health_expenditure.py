"""Health expenditure by source for DR Congo, from WHO's GHED.

Follows `create_expenditure_by_source()` in ONEcampaign/topic_health_financing
(scripts/analysis/create_data.py) for one country, sourcing GHED through
`bblocks.data_importers.GHED`. Output is one row per year from 2013 to 2023
with total current health expenditure in millions of constant 2023 USD and
each source's share of it, in percent: domestic government, external, other private and
out-of-pocket.
"""

import logging
from pathlib import Path

import polars as pl
from bblocks.data_importers import GHED

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = ROOT / "data" / "output" / "drc_expenditure_by_source.csv"

ISO3 = "COD"
# The original writes country_converter's short name.
COUNTRY_NAME = "DR Congo"
YEAR, COUNTRY = "Year", "Country"

GOV, EXT, PVT, OOP = "Domestic government", "External", "Other private", "Out-of-pocket"
SOURCE_ORDER = [GOV, EXT, PVT, OOP]
# Shares of current health expenditure published by GHED.
SHARE_CODES = {GOV: "gghed_che", EXT: "ext_che", OOP: "hf3_che"}
# Private spending excluding out-of-pocket: FS4 + FS5 + FS6 + FSNEC - FS6.1.
PVT_ADD = ["fs4", "fs5", "fs6", "fsnec"]
PVT_SUBTRACT = "fs61"
# Current health expenditure by financing scheme.
CHE_CODES = ["hf1", "hf2", "hf3", "hf4", "hfnec"]

# Current health expenditure in millions of constant 2023 USD, as GHED
# publishes it.
CHE_USD_CODE = "che_usd2023"
TOTAL = "Total health expenditure (million constant 2023 USD)"
GHED_USD_UNIT = "Millions"
FIRST_YEAR, LAST_YEAR = 2013, 2023

# Components of the sums may be absent and count as 0, as in the original; if
# all are absent, the share check fails instead.
REQUIRED_CODES = frozenset([*SHARE_CODES.values(), CHE_USD_CODE])
SHARE_SUM_TOLERANCE = 1.0


def load_ghed(iso3: str = ISO3) -> pl.DataFrame:
    """Download GHED and keep one country's `year, indicator_code, value`.

    Raises:
        ValueError: If an indicator the output needs is missing, as after GHED
            rebases constant USD to a new year, or the USD total is not in
            millions.
    """
    data = GHED().get_data()
    df = pl.from_pandas(data.loc[data["iso3_code"] == iso3]).select(
        pl.col("year").cast(pl.Int64), "indicator_code", "value", "unit"
    )
    missing = REQUIRED_CODES - set(df["indicator_code"].unique())
    if missing:
        raise ValueError(f"GHED has no {sorted(missing)} for {iso3}")

    usd_units = set(df.filter(pl.col("indicator_code") == CHE_USD_CODE)["unit"])
    if usd_units != {GHED_USD_UNIT}:
        raise ValueError(f"Expected {CHE_USD_CODE} in {GHED_USD_UNIT}, got {usd_units}")
    return df.drop("unit")


def indicator(ghed: pl.DataFrame, code: str) -> pl.DataFrame:
    """One indicator's `year, value` rows, dropping and logging nulls."""
    rows = ghed.filter(pl.col("indicator_code") == code).select("year", "value")
    nulls = rows.filter(pl.col("value").is_null())
    if nulls.height:
        logger.warning("%s is null for %s", code, nulls["year"].to_list())
    return rows.drop_nulls("value")


def sum_components(
    ghed: pl.DataFrame, *, add: list[str], subtract: str | None = None
) -> pl.DataFrame:
    """Sum components per year, as the original's `dropna(how="all").fillna(0)`.

    Years where every component is missing are dropped; otherwise missing
    components count as 0, and each fill is logged.

    Args:
        ghed: Long GHED rows for one country.
        add: Indicator codes to add.
        subtract: Indicator code to subtract, if any.

    Returns:
        `year` and `value` rows.
    """
    codes = [*add, *([subtract] if subtract else [])]
    wide = (
        ghed.filter(pl.col("indicator_code").is_in(codes))
        .pivot(on="indicator_code", index="year", values="value")
        .sort("year")
    )
    for code in codes:
        if code not in wide.columns:
            wide = wide.with_columns(pl.lit(None, dtype=pl.Float64).alias(code))
    wide = wide.filter(~pl.all_horizontal(pl.col(codes).is_null()))

    for code in codes:
        filled = wide.filter(pl.col(code).is_null())["year"].to_list()
        if filled:
            logger.info("%s missing for %s, counted as 0", code, filled)

    total = pl.sum_horizontal(pl.col(add).fill_null(0))
    if subtract:
        total = total - pl.col(subtract).fill_null(0)
    return wide.select("year", total.alias("value"))


def other_private_share(ghed: pl.DataFrame) -> pl.DataFrame:
    """Private spending excluding out-of-pocket as a percent of CHE."""
    pvt = sum_components(ghed, add=PVT_ADD, subtract=PVT_SUBTRACT)
    che = sum_components(ghed, add=CHE_CODES).rename({"value": "che"})
    return pvt.join(che, on="year", how="left").select(
        "year", (pl.col("value") / pl.col("che") * 100).alias("value")
    )


def validate_shares(df: pl.DataFrame) -> None:
    """Check every year has all four shares and they sum to about 100.

    Raises:
        ValueError: If a year is missing a source or sums outside tolerance.
    """
    incomplete = df.filter(pl.any_horizontal(pl.col(SOURCE_ORDER).is_null()))
    if incomplete.height:
        raise ValueError(f"Years missing a source share: {incomplete[YEAR].to_list()}")

    totals = df.select(YEAR, pl.sum_horizontal(SOURCE_ORDER).alias("total"))
    off = totals.filter((pl.col("total") - 100).abs() > SHARE_SUM_TOLERANCE)
    if off.height:
        raise ValueError(f"Shares do not sum to 100: {off.rows()}")
    logger.info(
        "Source shares sum to %.2f-%.2f across %d years",
        totals["total"].min(),
        totals["total"].max(),
        totals.height,
    )


def expenditure_by_source(
    ghed: pl.DataFrame, *, first_year: int = FIRST_YEAR, last_year: int = LAST_YEAR
) -> pl.DataFrame:
    """Total health expenditure and each source's share of it, by year.

    Args:
        ghed: Long GHED rows for one country.
        first_year: First year to keep.
        last_year: Last year to keep.

    Returns:
        `Year`, `Country`, `TOTAL` in millions of USD, and one percent column per source
        in `SOURCE_ORDER`, for every year from `first_year` to `last_year`.

    Raises:
        ValueError: If a year in the range is missing, or has no total or an
            incomplete set of shares.
    """
    shares = {source: indicator(ghed, code) for source, code in SHARE_CODES.items()}
    shares[PVT] = other_private_share(ghed)
    total = indicator(ghed, CHE_USD_CODE).select("year", pl.col("value").alias(TOTAL))

    years = pl.DataFrame(
        {"year": range(first_year, last_year + 1)}, schema={"year": pl.Int64}
    )
    df = years.join(total, on="year", how="left")
    for source in SOURCE_ORDER:
        df = df.join(shares[source].rename({"value": source}), on="year", how="left")
    df = df.rename({"year": YEAR}).sort(YEAR)

    no_total = df.filter(pl.col(TOTAL).is_null())[YEAR].to_list()
    if no_total:
        raise ValueError(f"No {CHE_USD_CODE} for {no_total}")
    validate_shares(df)
    return df.select(YEAR, pl.lit(COUNTRY_NAME).alias(COUNTRY), TOTAL, *SOURCE_ORDER)


def main() -> None:
    """Download GHED and write DR Congo's expenditure by source."""
    df = expenditure_by_source(load_ghed())
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(OUTPUT_PATH)
    logger.info(
        "Wrote %d years (%d-%d) to %s",
        df.height,
        df[YEAR].min(),
        df[YEAR].max(),
        OUTPUT_PATH.name,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
