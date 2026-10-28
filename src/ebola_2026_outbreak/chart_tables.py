"""Build chart-ready tables from the processed outbreak data.

Outputs, in data/output/, with display column names for charting tools:
    - 2026_outbreak_country_summary.csv: latest WHO figures per country,
      plus Total. When the latest DRC sitrep is newer than WHO's DRC figures,
      the DRC row takes the sitrep's confirmed cases and deaths.
    - 2026_outbreak_drc_cases_by_province.csv: latest sitrep confirmed cases
      for every DRC province, empty where the sitrep reports none, keyed to
      the map's OBJECTIDs.
    - 2026_outbreak_drc_cases_by_zone.csv: latest sitrep confirmed cases for
      every DRC health zone, empty where the sitrep reports none, keyed to the
      COD-AB P-codes of drc_health_zones_geometry.csv (see
      `drc_health_zone_geometry`).
    - outbreak_comparison_weekly.csv: cumulative confirmed cases by week since
      each outbreak's first confirmed case. West Africa uses OCHA ROWCA counts
      before 29 August 2014 and WHO/HDX counts from then on. The 2026 series
      sums WHO's figures across countries, with DRC's from a newer sitrep as
      in the country summary.
    - 2026_outbreak_headline_figures.json: outbreak-wide confirmed cases,
      confirmed deaths and case fatality rate (percent) from the country
      summary's Total row, dated by the DRC row's date (DRC holds over 99% of
      cases), and the number of DRC provinces with confirmed cases from the
      latest sitrep, dated by that sitrep. Dates are in prose ("18 October").
"""

import json
import logging
import unicodedata
from datetime import date
from pathlib import Path
from typing import TypedDict

import polars as pl

from ebola_2026_outbreak.aggregate import forward_fill_cumulative
from ebola_2026_outbreak.drc_2026_sitreps_extract import BY_ZONE_PATH, BY_ZONE_SCHEMA
from ebola_2026_outbreak.drc_2026_sitreps_parse import INDICATORS as SITREP_INDICATORS
from ebola_2026_outbreak.drc_2026_sitreps_parse import OUTPUT_PATH as PROVINCE_PATH
from ebola_2026_outbreak.drc_2026_sitreps_parse import SCHEMA as PROVINCE_SCHEMA
from ebola_2026_outbreak.drc_2026_who import CONFIRMED_CASES as WHO_CASES
from ebola_2026_outbreak.drc_2026_who import CONFIRMED_DEATHS as WHO_DEATHS
from ebola_2026_outbreak.drc_2026_who import DRC_COUNTRY as WHO_DRC
from ebola_2026_outbreak.drc_2026_who import OUTPUT_PATH as WHO_PATH
from ebola_2026_outbreak.drc_2026_who import RECOVERED as WHO_RECOVERED
from ebola_2026_outbreak.drc_2026_who import SCHEMA as WHO_SCHEMA
from ebola_2026_outbreak.drc_health_zone_geometry import CODES_PATH, GEOMETRY_PATH
from ebola_2026_outbreak.west_africa_2014_16 import FINAL_PATH as WEST_AFRICA_PATH
from ebola_2026_outbreak.west_africa_2014_16_early import (
    OUTPUT_PATH as WEST_AFRICA_EARLY_PATH,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "data" / "output"
OBJECTID_PATH = ROOT / "data" / "reference" / "drc_provinces_objectid.csv"
COUNTRY_SUMMARY_PATH = OUTPUT_DIR / "2026_outbreak_country_summary.csv"
PROVINCE_MAP_PATH = OUTPUT_DIR / "2026_outbreak_drc_cases_by_province.csv"
ZONE_MAP_PATH = OUTPUT_DIR / "2026_outbreak_drc_cases_by_zone.csv"
COMPARISON_PATH = OUTPUT_DIR / "outbreak_comparison_weekly.csv"
HEADLINE_PATH = OUTPUT_DIR / "2026_outbreak_headline_figures.json"

# Week 0 for each outbreak. West Africa: Institut Pasteur (Lyon) confirmed
# Ebola by PCR in Guinean samples, reported by WHO on 23 March 2014 with
# figures as of 22 March (Disease Outbreak News, 23 March 2014). DRC: start of
# the reporting period in WHO's Bundibugyo table title ("15 May-20 October").
WEST_AFRICA_FIRST_CONFIRMED = date(2014, 3, 22)
DRC_FIRST_CONFIRMED = date(2026, 5, 15)
DAYS_PER_WEEK = 7

TOTAL_LABEL = "Total"
CFR_DECIMALS = 1

COUNTRY, CASES, DEATHS, CFR, RECOVERED = (
    "Country",
    "Confirmed cases",
    "Confirmed deaths",
    "Case Fatality Rate",
    "Recovered",
)
NAME, OBJECTID, CODE = "Name", "OBJECTID", "Code"
WEEK = "Weeks since the first confirmed case"
WEST_AFRICA, DRC = "West Africa 2014-16 outbreak", "DRC 2026 outbreak"


class HeadlineFigure(TypedDict):
    """One headline number and the date its source reports, e.g. "18 October"."""

    value: int | float
    update_date: str


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
    """Latest cumulative confirmed cases for every map province, with OBJECTIDs.

    Sitreps list only provinces with cases, so every other province in
    `lookup` gets an empty (null) count and still appears on the map.

    Args:
        provinces: Sitrep province rows (see `drc_2026_sitreps_parse.SCHEMA`).
        lookup: Map provinces with `Name` and `OBJECTID`.

    Returns:
        `Name` (lookup spelling), `OBJECTID` and `Confirmed cases`, one row per
        `lookup` province, sorted by cases then name, provinces without
        cases last.

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
    with_cases = joined.select(OBJECTID, pl.col("value").cast(pl.Int64).alias(CASES))
    return (
        lookup.join(with_cases, on=OBJECTID, how="left")
        .select(NAME, OBJECTID, CASES)
        .sort([CASES, NAME], descending=[True, False], nulls_last=True)
    )


def zone_map(
    by_zone: pl.DataFrame, codes: pl.DataFrame, shapes: pl.DataFrame
) -> pl.DataFrame:
    """Latest cumulative confirmed cases for every map zone, by P-code.

    Zones the sitrep does not list, or lists with 0 cases, get an empty
    count. Rows sharing a P-code (a health area the sitrep lists as a zone,
    and the zone holding it) are summed. A zone with cases but no P-code (not
    a mapped zone, such as "A ventiler") is logged with its cases, which the
    map then leaves out. So is
    a zone the gazetteer gained after `drc_health_zone_geometry` last wrote
    the codes, which re-running that script settles.

    Args:
        by_zone: Sitrep zone rows (see `drc_2026_sitreps_extract.BY_ZONE_SCHEMA`).
        codes: `province`, `zone` and `code` per sitrep zone.
        shapes: Map zones with `Code` and `Name`.

    Returns:
        `Code`, `Name` and `Confirmed cases`, one row per `shapes` zone,
        sorted by cases then name, zones without cases last.

    Raises:
        ValueError: If a code is not a map zone.
    """
    cases = by_zone.filter(
        pl.col("level") == "zone",
        pl.col("indicator") == SITREP_INDICATORS["cases"],
        pl.col("date") == pl.col("date").max(),
    )
    logger.info("Zone cases as of %s", cases["date"].max())

    joined = cases.join(codes, on=["province", "zone"], how="left", coalesce=True)
    unknown = joined.join(codes, on=["province", "zone"], how="anti")
    for province, zone in unknown.select("province", "zone").rows():
        logger.warning("%s / %s is missing from %s", province, zone, CODES_PATH.name)
    stray = set(codes["code"].drop_nulls()) - set(shapes[CODE])
    if stray:
        raise ValueError(f"Zone codes with no map shape: {sorted(stray)}")
    unmapped = joined.filter(pl.col("code").is_null(), pl.col("value") > 0)
    for province, zone, value in unmapped.select("province", "zone", "value").rows():
        logger.warning("%s / %s: %d cases with no map zone", province, zone, value)

    # Summed by code, since a health area listed as a zone shares its zone's.
    with_cases = (
        joined.filter(pl.col("code").is_not_null(), pl.col("value") > 0)
        .group_by(pl.col("code").alias(CODE))
        .agg(pl.col("value").sum().alias(CASES))
    )
    return (
        shapes.select(CODE, NAME)
        .join(with_cases, on=CODE, how="left")
        .sort([CASES, NAME], descending=[True, False], nulls_last=True)
    )


def latest_date(dates: pl.Series) -> date:
    """Latest date in a series.

    Raises:
        ValueError: If the series is empty or holds no dates.
    """
    latest = dates.max()
    if not isinstance(latest, date):
        raise ValueError(f"No dates in {dates.name!r}")
    return latest


def summary_date(latest: pl.DataFrame) -> date:
    """Date the country summary's totals by the DRC row's date.

    DRC holds over 99% of cases, so a newer date for a small country (Kenya)
    must not date the totals. Countries on other dates are logged.

    Args:
        latest: Rows the summary was built from (output of `with_newer_sitrep`).

    Returns:
        The latest DRC date in `latest`.

    Raises:
        ValueError: If `latest` has no DRC rows.
    """
    drc = latest.filter(pl.col("country") == WHO_DRC)
    if drc.is_empty():
        raise ValueError(f"No {WHO_DRC!r} rows to date the totals by")
    drc_as_of = latest_date(drc["date"])
    as_of = latest.group_by("country").agg(pl.col("date").max()).sort("country")
    for country, day in as_of.filter(pl.col("date") != drc_as_of).rows():
        logger.info("Totals dated by DRC (%s); %s is as of %s", drc_as_of, country, day)
    return drc_as_of


def prose_date(day: date) -> str:
    """Format a date as day and month name without the year, e.g. "18 October"."""
    return f"{day.day} {day:%B}"


def with_newer_sitrep(who: pl.DataFrame, provinces: pl.DataFrame) -> pl.DataFrame:
    """Add the latest sitrep's DRC totals to the WHO table when WHO is older.

    WHO's DRC figures lag the DRC sitreps by a day or more. When the latest
    sitrep is newer than WHO's latest DRC date, its national confirmed cases
    and deaths (summed across provinces) are added as a DRC observation on the
    sitrep date. Sitreps do not report recoveries, so WHO's latest DRC value is
    carried onto that date.

    Args:
        who: Long WHO table rows (see `drc_2026_who.SCHEMA`).
        provinces: Sitrep province rows (see `drc_2026_sitreps_parse.SCHEMA`).

    Returns:
        `who`, plus DRC cases, deaths and recoveries on the sitrep date when
        the sitrep is newer.

    Raises:
        ValueError: If the latest sitrep lacks national cases or deaths.
    """
    drc = who.filter(pl.col("country") == WHO_DRC)
    who_as_of = latest_date(drc["date"])
    cumulative = provinces.filter(
        pl.col("indicator").is_in(
            [SITREP_INDICATORS["cases"], SITREP_INDICATORS["deaths"]]
        )
    )
    sitrep_as_of = latest_date(cumulative["date"])
    if sitrep_as_of <= who_as_of:
        logger.info(
            "DRC: WHO (%s) is as recent as the latest sitrep (%s); using WHO",
            who_as_of,
            sitrep_as_of,
        )
        return who

    totals = dict(
        cumulative.filter(pl.col("date") == sitrep_as_of)
        .group_by("indicator")
        .agg(pl.col("value").sum())
        .iter_rows()
    )
    missing = [
        label
        for label in (SITREP_INDICATORS["cases"], SITREP_INDICATORS["deaths"])
        if label not in totals
    ]
    if missing:
        raise ValueError(f"Sitrep of {sitrep_as_of} has no national {missing}")
    recovered = drc.filter(
        pl.col("date") == who_as_of, pl.col("indicator") == WHO_RECOVERED
    )["value"]
    logger.info(
        "DRC: sitrep (%s) is newer than WHO (%s); using the sitrep, with WHO's "
        "recoveries",
        sitrep_as_of,
        who_as_of,
    )
    indicators = [WHO_CASES, WHO_DEATHS, WHO_RECOVERED]
    sitrep_rows = pl.DataFrame(
        {
            "date": [sitrep_as_of] * len(indicators),
            "country": [WHO_DRC] * len(indicators),
            "indicator": indicators,
            "value": [
                float(totals[SITREP_INDICATORS["cases"]]),
                float(totals[SITREP_INDICATORS["deaths"]]),
                recovered.item() if recovered.len() == 1 else None,
            ],
        },
        schema={k: who.schema[k] for k in ("date", "country", "indicator", "value")},
    )
    return pl.concat([who, sitrep_rows], how="diagonal")


def headline_figures(
    *,
    summary: pl.DataFrame,
    summary_as_of: date,
    mapped: pl.DataFrame,
    sitrep_as_of: date,
) -> dict[str, HeadlineFigure]:
    """Headline outbreak figures with their update dates.

    Args:
        summary: Output of `country_summary`, including its Total row.
        summary_as_of: Date for the Total row's figures (see `summary_date`).
        mapped: Output of `province_map`, one row per map province.
        sitrep_as_of: Date of the sitrep `mapped` comes from.

    Returns:
        `total_confirmed_cases`, `total_deaths` (confirmed), `death_rate`
        (confirmed case fatality rate in percent) and `provinces_affected`.

    Raises:
        ValueError: If the Total row is missing or any of its figures is null.
    """
    total = summary.filter(pl.col(COUNTRY) == TOTAL_LABEL)
    if total.height != 1:
        raise ValueError(f"Expected one {TOTAL_LABEL!r} row, found {total.height}")
    row = total.row(0, named=True)
    missing = [column for column in (CASES, DEATHS, CFR) if row[column] is None]
    if missing:
        raise ValueError(f"{TOTAL_LABEL} row has no value for {missing}")

    summary_date, sitrep_date = prose_date(summary_as_of), prose_date(sitrep_as_of)
    return {
        "total_confirmed_cases": {"value": row[CASES], "update_date": summary_date},
        "total_deaths": {"value": row[DEATHS], "update_date": summary_date},
        "death_rate": {"value": row[CFR], "update_date": summary_date},
        "provinces_affected": {
            "value": mapped.filter(pl.col(CASES) > 0).height,
            "update_date": sitrep_date,
        },
    }


def drc_outbreak_series(who: pl.DataFrame) -> pl.DataFrame:
    """Confirmed cases summed across all affected countries by date.

    Takes WHO rows, or the output of `with_newer_sitrep`. Each country's last
    value is carried onto later dates first, because countries are reported on
    different dates.
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


def west_africa_series(early: pl.DataFrame, who: pl.DataFrame) -> pl.DataFrame:
    """Join ROWCA counts before the WHO series to the WHO series itself.

    Args:
        early: ROWCA `date` and `value` rows, all before the WHO series.
        who: WHO/HDX `date` and `value` rows.

    Returns:
        `date` and `value`, sorted by date.

    Raises:
        ValueError: If the two series overlap in time.
    """
    who_start = who.select(pl.col("date").min()).item()
    if early.filter(pl.col("date") >= who_start).height:
        raise ValueError(f"Early series overlaps the WHO series from {who_start}")
    return pl.concat([early, who], how="vertical_relaxed").sort("date")


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
    """Write the chart tables and headline figures."""
    who = pl.read_csv(WHO_PATH, schema=WHO_SCHEMA)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    provinces = pl.read_csv(PROVINCE_PATH, schema=PROVINCE_SCHEMA)
    latest = with_newer_sitrep(who, provinces)
    summary = country_summary(latest)
    summary.write_csv(COUNTRY_SUMMARY_PATH)
    logger.info("Wrote %d rows to %s", summary.height, COUNTRY_SUMMARY_PATH.name)

    lookup = pl.read_csv(OBJECTID_PATH, schema={NAME: pl.String, OBJECTID: pl.Int64})
    mapped = province_map(provinces, lookup)
    mapped.write_csv(PROVINCE_MAP_PATH)
    logger.info("Wrote %d rows to %s", mapped.height, PROVINCE_MAP_PATH.name)

    zones = zone_map(
        pl.read_csv(BY_ZONE_PATH, schema=BY_ZONE_SCHEMA),
        pl.read_csv(CODES_PATH, infer_schema=False),
        pl.read_csv(GEOMETRY_PATH, columns=[CODE, NAME], infer_schema=False),
    )
    zones.write_csv(ZONE_MAP_PATH)
    logger.info("Wrote %d rows to %s", zones.height, ZONE_MAP_PATH.name)

    west_africa = west_africa_series(
        pl.read_csv(WEST_AFRICA_EARLY_PATH, try_parse_dates=True).select(
            "date", "value"
        ),
        pl.read_csv(WEST_AFRICA_PATH, try_parse_dates=True).select("date", "value"),
    )
    comparison = weekly_comparison(west_africa, drc_outbreak_series(latest))
    comparison.write_csv(COMPARISON_PATH)
    logger.info("Wrote %d rows to %s", comparison.height, COMPARISON_PATH.name)

    sitrep_cases = provinces.filter(pl.col("indicator") == SITREP_INDICATORS["cases"])
    headline = headline_figures(
        summary=summary,
        summary_as_of=summary_date(latest),
        mapped=mapped,
        sitrep_as_of=latest_date(sitrep_cases["date"]),
    )
    HEADLINE_PATH.write_text(json.dumps(headline, indent=2) + "\n")
    logger.info("Wrote %d figures to %s", len(headline), HEADLINE_PATH.name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
