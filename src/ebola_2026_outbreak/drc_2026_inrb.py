"""Fetch INRB's national DRC sitrep series to backfill and cross-check ours.

INRB (with the Kraemer Lab, Oxford) hand-digitises the INSP sitrep banner
figures in INRB-UMIE/BDBV2026-Data. Their series covers dates the
sante.gouv.cd media API does not list (mid-June to early August 2026).

The source is pinned to one commit so upstream edits to past data change our
history only when `INRB_REF` is bumped. Run by hand; not part of the daily
scrape.
"""

import io
import logging
from itertools import pairwise

import polars as pl

from ebola_2026_outbreak.drc_2026_sitreps_parse import INDICATORS, OUTBREAK, ROOT
from ebola_2026_outbreak.fetch import get

logger = logging.getLogger(__name__)

INRB_REPO = "INRB-UMIE/BDBV2026-Data"
INRB_REF = "6f76a2179569451f245b8e3100bf7134b38a5cf4"  # build-2026-10-21-db4aee1
RAW_URL = (
    f"https://raw.githubusercontent.com/{INRB_REPO}/{INRB_REF}"
    "/data/insp_sitrep/processed/{file}"
)
OUTPUT_PATH = ROOT / "data" / "processed" / f"{OUTBREAK}_inrb_national.csv"

NATIONAL_NOM = "DRC"
MISSING = "ND"
# Our indicator label -> INRB metric (file insp_sitrep__<metric>__daily.csv).
METRICS = {
    INDICATORS["cases"]: "national_cumulative_confirmed_cases",
    INDICATORS["deaths"]: "national_cumulative_confirmed_deaths",
}

SCHEMA = pl.Schema(
    {
        "outbreak": pl.String,
        "date": pl.Date,
        "indicator": pl.String,
        "value": pl.Int64,
        "source_ref": pl.String,
    }
)


def fetch_csv(*, metric: str) -> pl.DataFrame:
    """Download one INRB processed CSV with every column read as a string."""
    url = RAW_URL.format(file=f"insp_sitrep__{metric}__daily.csv")
    response = get(url)
    return pl.read_csv(io.BytesIO(response.content), infer_schema=False)


def parse_national(*, raw: pl.DataFrame, metric: str, indicator: str) -> pl.DataFrame:
    """Validate one INRB national file and convert it to `SCHEMA`.

    Args:
        raw: INRB CSV with `nom`, `date` and `metric` string columns.
        metric: INRB metric name, also its value column.
        indicator: Our indicator label for the output rows.

    Returns:
        Rows in `SCHEMA`, sorted by date. "ND" values become null.

    Raises:
        ValueError: On unexpected columns, a non-national row, an invalid or
            duplicate date, or a non-integer value.
    """
    expected = ["nom", "date", metric]
    if raw.columns != expected:
        raise ValueError(f"{metric}: columns {raw.columns}, expected {expected}")

    other_noms = raw.filter(pl.col("nom") != NATIONAL_NOM)["nom"].unique().to_list()
    if other_noms:
        raise ValueError(f"{metric}: non-national rows {other_noms}")

    dates = raw["date"].str.to_date("%Y-%m-%d", strict=False)
    bad_dates = raw.filter(dates.is_null())["date"].to_list()
    if bad_dates:
        raise ValueError(f"{metric}: invalid dates {bad_dates}")
    duplicates = dates.filter(dates.is_duplicated()).unique().to_list()
    if duplicates:
        raise ValueError(f"{metric}: duplicate dates {duplicates}")

    values = raw[metric].str.strip_chars()
    missing = values == MISSING
    numbers = values.cast(pl.Int64, strict=False)
    bad_values = raw.filter(numbers.is_null() & ~missing)[metric].to_list()
    if bad_values:
        raise ValueError(f"{metric}: non-integer values {bad_values}")
    if missing.any():
        logger.warning("%s: %d %s values read as null", metric, missing.sum(), MISSING)

    return pl.DataFrame(
        {
            "outbreak": OUTBREAK,
            "date": dates,
            "indicator": indicator,
            "value": numbers,
            "source_ref": INRB_REF,
        },
        schema=SCHEMA,
    ).sort("date")


def check_monotonic(df: pl.DataFrame) -> None:
    """Log any fall in a cumulative national count between dates."""
    for indicator, group in df.drop_nulls("value").group_by(
        "indicator", maintain_order=True
    ):
        for (d0, v0), (d1, v1) in pairwise(group.select("date", "value").iter_rows()):
            if v1 < v0:
                logger.warning(
                    "INRB %s falls from %d (%s) to %d (%s)",
                    indicator[0].lower(),
                    v0,
                    d0,
                    v1,
                    d1,
                )


def main() -> None:
    """Fetch, validate and write the INRB national series."""
    frames = [
        parse_national(raw=fetch_csv(metric=metric), metric=metric, indicator=label)
        for label, metric in METRICS.items()
    ]
    national = pl.concat(frames).sort("date", "indicator")
    check_monotonic(national)
    national.write_csv(OUTPUT_PATH)
    logger.info(
        "Wrote %d rows (%s to %s) to %s",
        national.height,
        national["date"].min(),
        national["date"].max(),
        OUTPUT_PATH.name,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
