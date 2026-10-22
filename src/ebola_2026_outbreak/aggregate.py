"""Aggregation helpers shared by the outbreak pipelines."""

import logging

import polars as pl

logger = logging.getLogger(__name__)


def forward_fill_cumulative(
    df: pl.DataFrame, *, entity: str, cumulative_prefix: str
) -> pl.DataFrame:
    """Carry each entity's last reported cumulative value onto later report dates.

    Only cumulative indicators are filled: a stale "last 21 days" count is not
    valid on a later date. The grid uses dates on which any entity reported the
    indicator. Dates before an entity's first report stay empty, so that entity
    adds nothing on those dates. Other columns are null on filled rows.

    Args:
        df: Long data with `date`, `indicator`, `value` and the `entity` column.
        entity: Column identifying the reporting unit, e.g. "country".
        cumulative_prefix: Indicator prefix marking cumulative indicators.

    Returns:
        `df` with filled cumulative rows added.
    """
    is_cumulative = pl.col("indicator").str.starts_with(cumulative_prefix)
    cumulative, other = df.filter(is_cumulative), df.filter(~is_cumulative)

    grid = (
        cumulative.select("indicator", entity)
        .unique()
        .join(cumulative.select("indicator", "date").unique(), on="indicator")
    )
    filled = (
        grid.join(cumulative, on=["indicator", entity, "date"], how="left")
        .sort("indicator", entity, "date")
        .with_columns(pl.col("value").forward_fill().over("indicator", entity))
        .drop_nulls("value")
    )
    logger.info(
        "Forward-filled %d cumulative rows (%d -> %d)",
        filled.height - cumulative.height,
        cumulative.height,
        filled.height,
    )
    return pl.concat([filled.select(df.columns), other])
