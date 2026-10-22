"""DR Congo's external debt service against government health spending, by year.

Debt service is the sum of `value_outflows` across every creditor and
instrument in `data/raw/net_flows_agg.parquet` (ONE's net flows dataset,
gitignored). Outflows are stored as negative USD; the output reports them as
positive. Health spending is domestic general government health expenditure
(GGHE-D) from WHO's GHED. Both series are in millions of current USD.
"""

import logging

import polars as pl
from bblocks.data_importers import GHED

from ebola_2026_outbreak.drc_health_expenditure import (
    COUNTRY,
    COUNTRY_NAME,
    GHED_USD_UNIT,
    ISO3,
    ROOT,
    YEAR,
    indicator,
)

logger = logging.getLogger(__name__)

NET_FLOWS_PATH = ROOT / "data" / "raw" / "net_flows_agg.parquet"
OUTPUT_PATH = ROOT / "data" / "output" / "drc_debt_service_health_spending.csv"

ENTITY_NAME = "Democratic Republic of the Congo"
NET_FLOWS_COLUMNS = ["year", "entity_name", "prices", "value_outflows"]
NET_FLOWS_PRICES = "current"
USD_PER_MILLION = 1_000_000

# Domestic general government health expenditure, millions of current USD.
GGHED_USD_CODE = "gghed_usd"

DEBT, HEALTH = "Debt servicing", "Health spending"
DECIMALS = 2
FIRST_YEAR = 2013


def debt_service(net_flows: pl.DataFrame, *, entity: str = ENTITY_NAME) -> pl.DataFrame:
    """Total debt service outflows per year for one entity.

    Args:
        net_flows: Rows with `NET_FLOWS_COLUMNS`, one per year, creditor and
            instrument.
        entity: `entity_name` to keep.

    Returns:
        `year` and `value` in millions of current USD, positive.

    Raises:
        ValueError: On missing columns, no rows for `entity`, prices other than
            current, or a null or positive outflow.
    """
    missing = set(NET_FLOWS_COLUMNS) - set(net_flows.columns)
    if missing:
        raise ValueError(f"Net flows data has no {sorted(missing)} column")

    rows = net_flows.filter(pl.col("entity_name") == entity)
    if rows.is_empty():
        raise ValueError(f"No net flows rows for {entity!r}")

    prices = set(rows["prices"].unique())
    if prices != {NET_FLOWS_PRICES}:
        raise ValueError(f"Expected {NET_FLOWS_PRICES} prices, got {prices}")

    nulls = rows["value_outflows"].null_count()
    if nulls:
        raise ValueError(f"{nulls} null outflows for {entity!r}")
    positive = rows.filter(pl.col("value_outflows") > 0).height
    if positive:
        raise ValueError(f"{positive} positive outflows for {entity!r}; expected <= 0")

    series = (
        rows.group_by("year")
        .agg((pl.col("value_outflows").sum().abs() / USD_PER_MILLION).alias("value"))
        .sort("year")
    )
    return series


def load_gov_health(iso3: str = ISO3) -> pl.DataFrame:
    """Download GHED and return one country's GGHE-D `year, value` rows.

    Raises:
        ValueError: If GGHE-D is missing for the country or not in millions.
    """
    data = GHED().get_data()
    ghed = pl.from_pandas(
        data.loc[
            (data["iso3_code"] == iso3) & (data["indicator_code"] == GGHED_USD_CODE)
        ]
    ).select(pl.col("year").cast(pl.Int64), "indicator_code", "value", "unit")
    if ghed.is_empty():
        raise ValueError(f"GHED has no {GGHED_USD_CODE} for {iso3}")

    units = set(ghed["unit"])
    if units != {GHED_USD_UNIT}:
        raise ValueError(f"Expected {GGHED_USD_CODE} in {GHED_USD_UNIT}, got {units}")
    return indicator(ghed.drop("unit"), GGHED_USD_CODE)


def combine(
    *, debt: pl.DataFrame, health: pl.DataFrame, first_year: int = FIRST_YEAR
) -> pl.DataFrame:
    """Join the two series on year from `first_year`, keeping years either covers.

    Years only one source covers keep a null in the other column, logged, as
    are years with no debt service recorded.

    Args:
        debt: Debt service `year, value` rows.
        health: Government health spending `year, value` rows.
        first_year: First year to keep.

    Returns:
        `Country`, `Year`, `DEBT` and `HEALTH`, sorted by year.

    Raises:
        ValueError: If either input repeats a year.
    """
    for name, df in ((DEBT, debt), (HEALTH, health)):
        if df["year"].is_duplicated().any():
            raise ValueError(f"{name} has duplicate years")
    debt = debt.filter(pl.col("year") >= first_year)
    health = health.filter(pl.col("year") >= first_year)

    joined = debt.rename({"value": DEBT}).join(
        health.rename({"value": HEALTH}), on="year", how="full", coalesce=True
    )
    expected_years = set(debt["year"]) | set(health["year"])
    if joined.height != len(expected_years):
        raise ValueError(
            f"Join gave {joined.height} rows, expected {len(expected_years)} years"
        )
    for column in (DEBT, HEALTH):
        gaps = joined.filter(pl.col(column).is_null())["year"].sort().to_list()
        if gaps:
            logger.warning("%s missing for %s", column, gaps)
    zero_years = joined.filter(pl.col(DEBT) == 0)["year"].sort().to_list()
    if zero_years:
        logger.warning("No debt service recorded for %s", zero_years)

    return joined.sort("year").select(
        pl.lit(COUNTRY_NAME).alias(COUNTRY),
        pl.col("year").alias(YEAR),
        pl.col(DEBT).round(DECIMALS),
        pl.col(HEALTH).round(DECIMALS),
    )


def main() -> None:
    """Write DR Congo's debt service and government health spending by year."""
    if not NET_FLOWS_PATH.exists():
        raise FileNotFoundError(f"Net flows data not found at {NET_FLOWS_PATH}")
    net_flows = pl.read_parquet(NET_FLOWS_PATH, columns=NET_FLOWS_COLUMNS)

    df = combine(debt=debt_service(net_flows), health=load_gov_health())
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
