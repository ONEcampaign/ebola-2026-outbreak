"""Aggregate 2014-16 West Africa Ebola outbreak indicators across countries by date.

Historical comparison data, separate from the 2026 DRC outbreak.

Source: WHO situation reports via HDX, "Number of Ebola Cases and Deaths in
Affected Countries" (https://data.humdata.org/dataset/ebola-cases-2014),
resource ebola_data_db_format.csv, saved locally with an outbreak prefix.

"Guinea 2" and "Liberia 2" hold cases reported after each country was first
declared Ebola-free (HDX metadata), so they add to "Guinea" and "Liberia".

Outputs:
    - Intermediate: one row per (date, indicator), value summed across countries.
    - Final: cumulative confirmed cases only, one row per date.
"""

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
OUTBREAK = "west_africa_2014_16"
RAW_PATH = ROOT / "data" / "raw" / f"{OUTBREAK}_ebola_data_db_format.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
INTERMEDIATE_PATH = PROCESSED_DIR / f"{OUTBREAK}_indicators_by_date.csv"
FINAL_PATH = PROCESSED_DIR / f"{OUTBREAK}_cumulative_confirmed_cases.csv"

FINAL_INDICATOR = "Cumulative number of confirmed Ebola cases"
RAW_COLUMNS = {"Indicator": "indicator", "Country": "country", "Date": "date"}
# Rates and shares are not additive across countries.
NON_ADDITIVE_PREFIXES = ("Case fatality rate", "Proportion of")
CUMULATIVE_PREFIX = "Cumulative number of"


def load_raw(path: Path) -> pl.DataFrame:
    """Load the raw CSV and validate schema, nulls and duplicate keys.

    Args:
        path: Path to the raw CSV.

    Returns:
        DataFrame with snake_case columns and a parsed date.
    """
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found: {path}")

    df = pl.read_csv(path).rename(RAW_COLUMNS)
    missing = {*RAW_COLUMNS.values(), "value"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.with_columns(
        pl.col("indicator").str.strip_chars(),
        pl.col("date").str.to_date("%Y-%m-%d"),
    )

    null_counts = {c: n for c, n in df.null_count().row(0, named=True).items() if n}
    if null_counts:
        raise ValueError(f"Nulls in raw data: {null_counts}")

    n_dupes = df.select("indicator", "country", "date").is_duplicated().sum()
    if n_dupes:
        raise ValueError(f"{n_dupes} duplicate (indicator, country, date) rows")

    logger.info("Loaded %d rows from %s", df.height, path.name)
    return df


def drop_non_additive(df: pl.DataFrame) -> pl.DataFrame:
    """Remove rate/proportion indicators that cannot be summed across countries."""
    is_non_additive = pl.any_horizontal(
        pl.col("indicator").str.starts_with(p) for p in NON_ADDITIVE_PREFIXES
    )

    dropped = df.filter(is_non_additive)
    logger.warning(
        "Dropping %d rows across %d non-additive indicators",
        dropped.height,
        dropped["indicator"].n_unique(),
    )
    return df.filter(~is_non_additive)


def forward_fill_cumulative(df: pl.DataFrame) -> pl.DataFrame:
    """Carry each country's last reported cumulative value onto later report dates.

    Only cumulative indicators are filled: a stale "last 21 days" count is not
    valid on a later date. The grid uses dates on which any country reported the
    indicator. Dates before a country's first report stay empty, so that country
    adds nothing on those dates.
    """
    is_cumulative = pl.col("indicator").str.starts_with(CUMULATIVE_PREFIX)
    cumulative, other = df.filter(is_cumulative), df.filter(~is_cumulative)

    grid = (
        cumulative.select("indicator", "country")
        .unique()
        .join(cumulative.select("indicator", "date").unique(), on="indicator")
    )
    filled = (
        grid.join(cumulative, on=["indicator", "country", "date"], how="left")
        .sort("indicator", "country", "date")
        .with_columns(pl.col("value").forward_fill().over("indicator", "country"))
        .drop_nulls("value")
    )
    logger.info(
        "Forward-filled %d cumulative rows (%d -> %d)",
        filled.height - cumulative.height,
        cumulative.height,
        filled.height,
    )
    return pl.concat([filled.select(df.columns), other])


def sum_by_date_indicator(df: pl.DataFrame) -> pl.DataFrame:
    """Sum values across countries for each date and indicator, tagged by outbreak."""
    return (
        df.group_by("date", "indicator")
        .agg(pl.col("value").sum())
        .sort("date", "indicator")
        .select(pl.lit(OUTBREAK).alias("outbreak"), "date", "indicator", "value")
    )


def select_indicator(df: pl.DataFrame, indicator: str) -> pl.DataFrame:
    """Filter to a single indicator and check dates are unique."""
    out = df.filter(pl.col("indicator") == indicator)
    if out.is_empty():
        raise ValueError(f"Indicator not found: {indicator}")
    if out["date"].is_duplicated().any():
        raise ValueError(f"Duplicate dates for indicator: {indicator}")

    n_decreases = (out["value"].diff() < 0).sum()
    if n_decreases:
        logger.warning(
            "%d dates where summed cumulative value decreases",
            n_decreases,
        )
    return out


def main() -> None:
    """Build the intermediate and final outputs."""
    raw = load_raw(RAW_PATH)
    additive = forward_fill_cumulative(drop_non_additive(raw))

    intermediate = sum_by_date_indicator(additive)
    intermediate.write_csv(INTERMEDIATE_PATH)
    logger.info("Wrote %d rows to %s", intermediate.height, INTERMEDIATE_PATH.name)

    final = select_indicator(intermediate, FINAL_INDICATOR)
    final.write_csv(FINAL_PATH)
    logger.info("Wrote %d rows to %s", final.height, FINAL_PATH.name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
