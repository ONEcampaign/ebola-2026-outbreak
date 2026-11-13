"""DR Congo health zone shapes and P-codes for a zone-level map, from OCHA's COD-AB.

COD-AB's admin level 3 holds the 519 health zones (zones de santé), with
boundaries from September 2019. The shapes are simplified as one coverage, so
neighbouring zones keep a shared border, and written with each zone's P-code
as `Code` to `data/output/drc_health_zones_geometry.csv`. Each zone's province,
name and P-code go to `data/reference/cod_ab_health_zones.csv`. Run it by hand
only to refresh COD-AB.

`zone_codes` matches the sitrep gazetteer's zones to those P-codes by province
and name when the chart tables are built. Zones COD-AB spells differently are
listed in `drc_2026_sitrep_schema.COD_SPELLINGS` or as gazetteer aliases,
health areas the sitreps report as zones take their zone's code in
`drc_2026_sitrep_schema.HEALTH_AREA_ZONES`, and rows that are not zones are in
`NOT_MAPPED`. The schema module keeps those tables, so the gazetteer matches
COD-AB the same way without the shapely dependency.
"""

import io
import json
import logging
import zipfile
from pathlib import Path

import polars as pl
import shapely
from shapely.geometry import mapping, shape

from ebola_2026_outbreak.drc_2026_sitrep_schema import (
    COD_AB_PATH,
    GAZETTEER_PATH,
    HEALTH_AREA_ZONES,
    cod_ab_names,
    load_gazetteer,
    name_key,
)
from ebola_2026_outbreak.fetch import get

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
GEOMETRY_PATH = ROOT / "data" / "output" / "drc_health_zones_geometry.csv"

COD_AB_URL = (
    "https://data.humdata.org/dataset/f42132b9-8cc6-4201-b020-9259c56e8868"
    "/resource/97260e2b-65b1-41e3-aef2-fb0e6874e406/download"
    "/cod_admin_boundaries.geojson.zip"
)
ADMIN3_FILE = "cod_admin3.geojson"
EXPECTED_ZONES = 519
# Degrees; about 550 m at the equator. Keeps all 519 zones at about 1.3 MB.
SIMPLIFY_TOLERANCE = 0.005
# Degrees; four decimals is about 11 m.
COORDINATE_GRID = 1e-4
BYTES_PER_MB = 1e6
POLYGON_TYPES = frozenset({"Polygon", "MultiPolygon"})

# Gazetteer (province, zone) rows that are not zones, so have no map shape.
# "A ventiler" holds cases not yet assigned to a zone.
NOT_MAPPED = frozenset({("Ituri", "A ventiler")})

GEOMETRY, CODE, NAME = "Geometry", "Code", "Name"


def load_admin3(url: str = COD_AB_URL) -> list[dict]:
    """Download COD-AB and return the admin level 3 GeoJSON features."""
    archive = zipfile.ZipFile(io.BytesIO(get(url).content))
    features = json.loads(archive.read(ADMIN3_FILE))["features"]
    if len(features) != EXPECTED_ZONES:
        raise ValueError(f"Expected {EXPECTED_ZONES} zones, found {len(features)}")
    return features


def polygonal(geometry: shapely.Geometry) -> shapely.Geometry:
    """Repair a shape and keep only its polygon parts.

    Raises:
        ValueError: If nothing polygonal is left.
    """
    valid = shapely.make_valid(geometry)
    parts = [
        g
        for g in getattr(valid, "geoms", [valid])
        if g.geom_type in POLYGON_TYPES and not g.is_empty
    ]
    if not parts:
        raise ValueError(f"No polygon left after repair: {valid.geom_type}")
    merged = shapely.union_all(parts)
    return merged


def geometry_table(features: list[dict]) -> pl.DataFrame:
    """One row per zone: simplified GeoJSON geometry, P-code and name.

    Raises:
        ValueError: If P-codes repeat, or a simplified shape is empty,
            invalid or not a polygon.
    """
    codes = [f["properties"]["adm3_pcode"] for f in features]
    if len(set(codes)) != len(codes):
        raise ValueError("COD-AB admin 3 P-codes are not unique")

    repaired = [polygonal(shape(f["geometry"])) for f in features]
    simplified = shapely.coverage_simplify(repaired, tolerance=SIMPLIFY_TOLERANCE)
    gridded = shapely.set_precision(simplified, COORDINATE_GRID)
    bad = [
        code
        for code, g in zip(codes, gridded, strict=True)
        if g.is_empty or not g.is_valid or g.geom_type not in POLYGON_TYPES
    ]
    if bad:
        raise ValueError(f"Simplified shapes are empty, invalid or not polygons: {bad}")

    return pl.DataFrame(
        {
            GEOMETRY: [json.dumps(mapping(g), separators=(",", ":")) for g in gridded],
            CODE: codes,
            NAME: [f["properties"]["adm3_name"] for f in features],
        }
    ).sort(CODE)


def cod_ab_table(features: list[dict]) -> pl.DataFrame:
    """One row per COD-AB zone: `province`, `zone` and P-code `code`."""
    return pl.DataFrame(
        [
            {
                "province": p["adm1_name"],
                "zone": p["adm3_name"],
                "code": p["adm3_pcode"],
            }
            for p in (f["properties"] for f in features)
        ],
        schema={"province": pl.String, "zone": pl.String, "code": pl.String},
    ).sort("code")


def gazetteer_table(path: Path = GAZETTEER_PATH) -> pl.DataFrame:
    """Every gazetteer zone as `province`, `zone` and `aliases`.

    The curated rows of `path`, then the zones only COD-AB lists
    (`Gazetteer.unseen`), so a zone released before it is curated still gets
    its P-code.
    """
    curated = pl.read_csv(path, infer_schema=False)
    unseen = pl.DataFrame(
        sorted(load_gazetteer(path).unseen),
        schema={"province": pl.String, "zone": pl.String},
        orient="row",
    ).with_columns(aliases=pl.lit(None, dtype=pl.String))
    return pl.concat([curated, unseen])


def zone_codes(cod_ab: pl.DataFrame, gazetteer: pl.DataFrame) -> pl.DataFrame:
    """The COD-AB P-code of each gazetteer zone, matched by province and name.

    A zone matches on its `HEALTH_AREA_ZONES` entry, else on its
    `cod_ab_names` (its `COD_SPELLINGS` entry or own name, then its aliases),
    all by `name_key`.

    Args:
        cod_ab: COD-AB zones with `province`, `zone` and `code` (see
            `cod_ab_table`).
        gazetteer: Sitrep zones with `province`, `zone` and `aliases`.

    Returns:
        `province`, `zone` and `code`, one row per gazetteer zone, null where
        COD-AB has no such zone. Such zones not in `NOT_MAPPED` are logged.
    """
    cod = {
        (name_key(province), name_key(zone)): code
        for province, zone, code in cod_ab.select("province", "zone", "code").rows()
    }
    codes = []
    for province, zone, aliases in gazetteer.select(
        "province", "zone", "aliases"
    ).iter_rows():
        names = cod_ab_names(zone, aliases)
        if holder := HEALTH_AREA_ZONES.get(zone):
            names[0] = holder
        code = next(
            (
                cod[k]
                for n in names
                if n and (k := (name_key(province), name_key(n))) in cod
            ),
            None,
        )
        if code is None and (province, zone) not in NOT_MAPPED:
            logger.warning("No COD-AB zone for %s / %s", province, zone)
        codes.append(code)
    return gazetteer.select("province", "zone").with_columns(
        code=pl.Series(codes, dtype=pl.String)
    )


def main() -> None:
    """Write the zone shapes and the COD-AB zone table."""
    features = load_admin3()

    shapes = geometry_table(features)
    shapes.write_csv(GEOMETRY_PATH)
    logger.info(
        "Wrote %d zones (%.1f MB) to %s",
        shapes.height,
        GEOMETRY_PATH.stat().st_size / BYTES_PER_MB,
        GEOMETRY_PATH.name,
    )

    zones = cod_ab_table(features)
    zones.write_csv(COD_AB_PATH)
    logger.info("Wrote %d zones to %s", zones.height, COD_AB_PATH.name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
