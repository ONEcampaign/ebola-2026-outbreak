"""Build chart-ready tables from the processed outbreak data.

Outputs, in data/output/, with display column names for charting tools:
    - 2026_outbreak_country_summary.csv: latest WHO figures per country,
      plus Total.
    - 2026_outbreak_drc_cases_by_province.csv: latest sitrep confirmed cases
      per province, keyed to the map's OBJECTIDs.
    - outbreak_comparison_weekly.csv: cumulative confirmed cases by week since
      each outbreak's first confirmed case.
"""

import logging
import unicodedata
from datetime import date
from pathlib import Path

import polars as pl

from ebola_2026_outbreak.aggregate import forward_fill_cumulative
from ebola_2026_outbreak.drc_2026_sitreps_parse import INDICATORS as SITREP_INDICATORS
from ebola_2026_outbreak.drc_2026_sitreps_parse import OUTPUT_PATH as PROVINCE_PATH
from ebola_2026_outbreak.drc_2026_sitreps_parse import SCHEMA as PROVINCE_SCHEMA
from ebola_2026_outbreak.drc_2026_who import OUTPUT_PATH as WHO_PATH
from ebola_2026_outbreak.drc_2026_who import SCHEMA as WHO_SCHEMA
from ebola_2026_outbreak.west_africa_2014_16 import FINAL_PATH as WEST_AFRICA_PATH

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "data" / "output"
OBJECTID_PATH = ROOT / "data" / "reference" / "drc_provinces_objectid.csv"
COUNTRY_SUMMARY_PATH = OUTPUT_DIR / "2026_outbreak_country_summary.csv"
PROVINCE_MAP_PATH = OUTPUT_DIR / "2026_outbreak_drc_cases_by_province.csv"
COMPARISON_PATH = OUTPUT_DIR / "outbreak_comparison_weekly.csv"

# Week 0 for each outbreak. West Africa: Institut Pasteur (Lyon) confirmed
# Ebola by PCR in Guinean samples, reported by WHO on 23 March 2014 with
# figures as of 22 March (Disease Outbreak News, 23 March 2014). DRC: start of
# the reporting period in WHO's Bundibugyo table title ("15 May-20 October").
WEST_AFRICA_FIRST_CONFIRMED = date(2014, 3, 22)
DRC_FIRST_CONFIRMED = date(2026, 5, 15)
DAYS_PER_WEEK = 7

WHO_CASES, WHO_DEATHS, WHO_RECOVERED = (
    "Confirmed cases",
    "Confirmed deaths",
    "Recovered",
)
TOTAL_LABEL = "Total"
CFR_DECIMALS = 1

COUNTRY, CASES, DEATHS, CFR, RECOVERED = (
    "Country",
    "Confirmed cases",
    "Confirmed deaths",
    "Case Fatality Rate",
    "Recovered",
)
NAME, OBJECTID = "Name", "OBJECTID"
WEEK = "Weeks since the first confirmed case"
WEST_AFRICA, DRC = "West Africa 2014-16 outbreak", "DRC 2026 outbreak"


def _cfr(deaths: pl.Expr, cases: pl.Expr) -> pl.Expr:
    """Case fatality rate in percent, null when there are no cases."""
    return pl.when(cases > 0).then(deaths / cases * 100).round(CFR_DECIMALS).alias(CFR)


def country_summary(who: pl.DataFrame) -> pl.DataFrame:
    """Latest WHO confirmed cases, deaths, CFR and recoveries per country.

    Each country uses its own latest "Data as of" date. The Total row sums
    countries and skips missing values, as WHO's own Total row does.

    Args:
        who: Long WHO table rows (see `drc_2026_who.SCHEMA`).

    Returns:
        One row per country plus Total, with display column names.
    """
    latest = who.filter(pl.col("date") == pl.col("date").max().over("country"))
    for country, as_of in (
        latest.select("country", "date").unique().sort("country").rows()
    ):
        logger.info("%s: data as of %s", country, as_of)

    wide = latest.pivot(on="indicator", index="country", values="value").select(
        pl.col("country").alias(COUNTRY),
        pl.col(WHO_CASES).cast(pl.Int64).alias(CASES),
        pl.col(WHO_DEATHS).cast(pl.Int64).alias(DEATHS),
        pl.col(WHO_RECOVERED).cast(pl.Int64).alias(RECOVERED),
    )
    for column in (CASES, DEATHS, RECOVERED):
        missing = wide.filter(pl.col(column).is_null())[COUNTRY].to_list()
        if missing:
            logger.warning("%s not reported by WHO for: %s", column, missing)

    countries = wide.sort(CASES, descending=True)
    total = countries.select(
        pl.lit(TOTAL_LABEL).alias(COUNTRY), pl.col(CASES, DEATHS, RECOVERED).sum()
    )
    table = pl.concat([countries, total])
    return table.select(
        COUNTRY, CASES, DEATHS, _cfr(pl.col(DEATHS), pl.col(CASES)), RECOVERED
    )


def _match_key(name: str) -> str:
    """Accent- and case-insensitive key, so "Haut-Uélé" matches "Haut-Uele"."""
    decomposed = unicodedata.normalize("NFKD", name)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()


def province_map(provinces: pl.DataFrame, lookup: pl.DataFrame) -> pl.DataFrame:
    """Latest cumulative confirmed cases for provinces with cases, with OBJECTIDs.

    Args:
        provinces: Sitrep province rows (see `drc_2026_sitreps_parse.SCHEMA`).
        lookup: Map provinces with `Name` and `OBJECTID`.

    Returns:
        `Name` (lookup spelling), `OBJECTID` and `Confirmed cases`.

    Raises:
        ValueError: If a province with cases has no OBJECTID.
    """
    cases = provinces.filter(
        pl.col("indicator") == SITREP_INDICATORS["cases"],
        pl.col("date") == pl.col("date").max(),
        pl.col("value") > 0,
    )
    logger.info("Province cases as of %s", cases["date"].max())

    key = pl.col("province").map_elements(_match_key, return_dtype=pl.String)
    lookup_key = pl.col(NAME).map_elements(_match_key, return_dtype=pl.String)
    joined = cases.with_columns(key=key).join(
        lookup.with_columns(key=lookup_key), on="key", how="left"
    )
    unmatched = joined.filter(pl.col(OBJECTID).is_null())["province"].to_list()
    if unmatched:
        raise ValueError(f"No OBJECTID for provinces: {unmatched}")
    return joined.select(
        NAME, OBJECTID, pl.col("value").cast(pl.Int64).alias(CASES)
    ).sort(CASES, descending=True)


def drc_outbreak_series(who: pl.DataFrame) -> pl.DataFrame:
    """WHO confirmed cases summed across all affected countries by date.

    Each country's last value is carried onto later dates first, because
    countries are reported on different dates.
    """
    confirmed = who.filter(pl.col("indicator") == WHO_CASES).select(
        "date", "country", "indicator", "value"
    )
    filled = forward_fill_cumulative(
        confirmed, entity="country", cumulative_prefix=WHO_CASES
    )
    return filled.group_by("date").agg(pl.col("value").sum()).sort("date")


def weekly(series: pl.DataFrame, *, start: date, name: str) -> pl.DataFrame:
    """Last cumulative value in each week since `start`.

    Args:
        series: `date` and `value` columns.
        start: Week 0's first day.
        name: Output value column name.

    Returns:
        `WEEK` and `name` columns, one row per week with an observation.

    Raises:
        ValueError: If any observation predates `start`.
    """
    if series.filter(pl.col("date") < start).height:
        raise ValueError(f"{name} has data before its week 0 ({start})")
    return (
        series.with_columns(
            ((pl.col("date") - pl.lit(start)).dt.total_days() // DAYS_PER_WEEK)
            .cast(pl.Int64)
            .alias(WEEK)
        )
        .sort("date")
        .group_by(WEEK)
        .agg(pl.col("value").last().cast(pl.Int64).alias(name))
    )


def weekly_comparison(west_africa: pl.DataFrame, drc: pl.DataFrame) -> pl.DataFrame:
    """Cumulative confirmed cases per week since each outbreak's first case.

    Weeks without an observation stay null rather than being interpolated.

    Args:
        west_africa: West Africa `date` and `value` series.
        drc: DRC 2026 `date` and `value` series.

    Returns:
        `WEEK`, `WEST_AFRICA` and `DRC` columns from week 0 to the last week
        either series reaches.
    """
    wa = weekly(west_africa, start=WEST_AFRICA_FIRST_CONFIRMED, name=WEST_AFRICA)
    dr = weekly(drc, start=DRC_FIRST_CONFIRMED, name=DRC)
    last_week = max([*wa[WEEK].to_list(), *dr[WEEK].to_list()])
    weeks = pl.DataFrame({WEEK: range(last_week + 1)}, schema={WEEK: pl.Int64})
    table = weeks.join(wa, on=WEEK, how="left").join(dr, on=WEEK, how="left")
    for column in (WEST_AFRICA, DRC):
        logger.info(
            "%s: %d of %d weeks have no observation",
            column,
            table[column].null_count(),
            table.height,
        )
    return table


def main() -> None:
    """Write the three chart tables."""
    who = pl.read_csv(WHO_PATH, schema=WHO_SCHEMA)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = country_summary(who)
    summary.write_csv(COUNTRY_SUMMARY_PATH)
    logger.info("Wrote %d rows to %s", summary.height, COUNTRY_SUMMARY_PATH.name)

    provinces = pl.read_csv(PROVINCE_PATH, schema=PROVINCE_SCHEMA)
    lookup = pl.read_csv(OBJECTID_PATH, schema={NAME: pl.String, OBJECTID: pl.Int64})
    mapped = province_map(provinces, lookup)
    mapped.write_csv(PROVINCE_MAP_PATH)
    logger.info("Wrote %d rows to %s", mapped.height, PROVINCE_MAP_PATH.name)

    west_africa = pl.read_csv(WEST_AFRICA_PATH, try_parse_dates=True)
    comparison = weekly_comparison(
        west_africa.select("date", "value"), drc_outbreak_series(who)
    )
    comparison.write_csv(COMPARISON_PATH)
    logger.info("Wrote %d rows to %s", comparison.height, COMPARISON_PATH.name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
