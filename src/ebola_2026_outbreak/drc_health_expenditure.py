"""Health expenditure by source for DR Congo, from WHO's GHED.

Replicates `create_expenditure_by_source()` in ONEcampaign/topic_health_financing
(scripts/analysis/create_data.py) for one country, sourcing GHED through
`bblocks.data_importers.GHED`. Four sources (domestic government, external,
other private, out-of-pocket) are given as a share of current health
expenditure and in constant USD. The current GHED release prices constant USD
in 2023 dollars; the original used 2022.
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
ENTITY_NAME = "DR Congo"
USD_SUFFIX = "usd2023"
SHARE_UNIT = "percent of health expenditure"
USD_UNIT = f"constant USD ({USD_SUFFIX.removeprefix('usd')})"

GOV, EXT, PVT, OOP = "Domestic government", "External", "Other private", "Out-of-pocket"
# Direct share and USD indicators per source; other private is calculated.
SHARE_CODES = {GOV: "gghed_che", EXT: "ext_che", OOP: "hf3_che"}
USD_CODES = {
    GOV: f"gghed_{USD_SUFFIX}",
    EXT: f"ext_{USD_SUFFIX}",
    OOP: f"hf3_{USD_SUFFIX}",
}
# Private spending excluding out-of-pocket: FS4 + FS5 + FS6 + FSNEC - FS6.1.
PVT_ADD = ["fs4", "fs5", "fs6", "fsnec"]
PVT_SUBTRACT = "fs61"
# Current health expenditure by financing scheme.
CHE_CODES = ["hf1", "hf2", "hf3", "hf4", "hfnec"]
SOURCE_ORDER = [GOV, EXT, PVT, OOP]

# Indicators used directly. Components of the sums may be absent and count as
# 0, as in the original; if all are absent, the share check fails instead.
REQUIRED_CODES = frozenset([*SHARE_CODES.values(), *USD_CODES.values()])
SHARE_SUM_TOLERANCE = 1.0
# GHED publishes USD indicators in millions; the original output is in USD.
GHED_USD_UNIT = "Millions"
USD_PER_GHED_UNIT = 1_000_000
OUTPUT_COLUMNS = ["iso3_code", "year", "value", "unit", "source", "entity_name"]


def load_ghed(iso3: str = ISO3) -> pl.DataFrame:
    """Download GHED and keep one country's `year, indicator_code, value`.

    Raises:
        ValueError: If an indicator the calculation needs is missing, as after
            GHED rebases constant USD to a new year, or USD indicators are not
            in millions.
    """
    data = GHED().get_data()
    df = pl.from_pandas(data.loc[data["iso3_code"] == iso3]).select(
        pl.col("year").cast(pl.Int64), "indicator_code", "value", "unit"
    )
    missing = REQUIRED_CODES - set(df["indicator_code"].unique())
    if missing:
        raise ValueError(f"GHED has no {sorted(missing)} for {iso3}")

    usd_units = set(
        df.filter(pl.col("indicator_code").str.ends_with(USD_SUFFIX))["unit"].unique()
    )
    if usd_units != {GHED_USD_UNIT}:
        raise ValueError(f"Expected USD indicators in {GHED_USD_UNIT}, got {usd_units}")
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


def other_private_usd(ghed: pl.DataFrame) -> pl.DataFrame:
    """Private spending excluding out-of-pocket in constant USD."""
    return sum_components(
        ghed,
        add=[f"{code}_{USD_SUFFIX}" for code in PVT_ADD],
        subtract=f"{PVT_SUBTRACT}_{USD_SUFFIX}",
    )


def _labelled(series: pl.DataFrame, *, unit: str, source: str) -> pl.DataFrame:
    """Attach unit and source labels to a `year, value` series."""
    return series.with_columns(unit=pl.lit(unit), source=pl.lit(source))


def validate_shares(df: pl.DataFrame) -> None:
    """Check the four source shares sum to about 100 in every year.

    Raises:
        ValueError: If a year is missing a source or sums outside tolerance.
    """
    sums = (
        df.filter(pl.col("unit") == SHARE_UNIT)
        .group_by("year")
        .agg(pl.col("value").sum().alias("total"), pl.len().alias("n"))
        .sort("year")
    )
    incomplete = sums.filter(pl.col("n") != len(SOURCE_ORDER))["year"].to_list()
    if incomplete:
        raise ValueError(f"Years missing a source share: {incomplete}")
    off = sums.filter((pl.col("total") - 100).abs() > SHARE_SUM_TOLERANCE)
    if off.height:
        raise ValueError(
            f"Shares do not sum to 100: {off.select('year', 'total').rows()}"
        )
    logger.info(
        "Source shares sum to %.2f-%.2f across %d years",
        sums["total"].min(),
        sums["total"].max(),
        sums.height,
    )


def expenditure_by_source(ghed: pl.DataFrame) -> pl.DataFrame:
    """Build the eight source series in the original output's layout.

    Args:
        ghed: Long GHED rows for one country.

    Returns:
        Columns `OUTPUT_COLUMNS`: shares first, then constant USD, each in
        `SOURCE_ORDER`.
    """
    shares = {source: indicator(ghed, code) for source, code in SHARE_CODES.items()}
    shares[PVT] = other_private_share(ghed)
    usd_millions = {source: indicator(ghed, code) for source, code in USD_CODES.items()}
    usd_millions[PVT] = other_private_usd(ghed)
    usd = {
        source: series.with_columns(pl.col("value") * USD_PER_GHED_UNIT)
        for source, series in usd_millions.items()
    }

    df = pl.concat(
        [_labelled(shares[s], unit=SHARE_UNIT, source=s) for s in SOURCE_ORDER]
        + [_labelled(usd[s], unit=USD_UNIT, source=s) for s in SOURCE_ORDER]
    ).with_columns(iso3_code=pl.lit(ISO3), entity_name=pl.lit(ENTITY_NAME))

    if df.select("year", "unit", "source").is_duplicated().any():
        raise ValueError("Duplicate (year, unit, source) rows")
    validate_shares(df)
    return df.select(OUTPUT_COLUMNS)


def main() -> None:
    """Download GHED and write DR Congo's expenditure by source."""
    df = expenditure_by_source(load_ghed())
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(OUTPUT_PATH)
    logger.info(
        "Wrote %d rows (%d-%d) to %s",
        df.height,
        df["year"].min(),
        df["year"].max(),
        OUTPUT_PATH.name,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
