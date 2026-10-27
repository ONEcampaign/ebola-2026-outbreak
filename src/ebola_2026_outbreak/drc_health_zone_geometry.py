"""DR Congo health zone shapes for a zone-level map, from OCHA's COD-AB.

COD-AB's admin level 3 holds the 519 health zones (zones de santé), with
boundaries from September 2019. The shapes are simplified as one coverage, so
neighbouring zones keep a shared border, and written with each zone's P-code
as `Code`. The sitrep gazetteer's zones are matched to those codes by province
and name; zones COD-AB spells differently are listed in `COD_SPELLINGS`.
A sitrep zone with no shape keeps an empty code. Run by hand; the boundaries
do not change with the outbreak data.
"""

import io
import json
import logging
import zipfile
from pathlib import Path

import polars as pl
import shapely
from shapely.geometry import mapping, shape

from ebola_2026_outbreak.drc_2026_sitrep_schema import GAZETTEER_PATH, name_key
from ebola_2026_outbreak.fetch import get

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
GEOMETRY_PATH = ROOT / "data" / "output" / "drc_health_zones_geometry.csv"
CODES_PATH = ROOT / "data" / "reference" / "drc_2026_health_zone_codes.csv"

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
POLYGON_TYPES = frozenset({"Polygon", "MultiPolygon"})

# Sitrep zone name -> COD-AB name, for the same zone spelled differently.
COD_SPELLINGS = {"Mongbwalu": "Mongbalu", "Nyankunde": "Nyakunde"}

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


def zone_codes(features: list[dict], gazetteer: pl.DataFrame) -> pl.DataFrame:
    """The COD-AB P-code of each gazetteer zone, matched by province and name.

    Args:
        features: COD-AB admin 3 features.
        gazetteer: Sitrep zones with `province`, `zone` and `aliases`.

    Returns:
        `province`, `zone` and `code`, null where COD-AB has no such zone.
    """
    cod = {
        (name_key(p["adm1_name"]), name_key(p["adm3_name"])): p["adm3_pcode"]
        for p in (f["properties"] for f in features)
    }
    codes = []
    for province, zone, aliases in gazetteer.select(
        "province", "zone", "aliases"
    ).iter_rows():
        names = [COD_SPELLINGS.get(zone, zone), *(aliases or "").split("|")]
        code = next(
            (
                cod[k]
                for n in names
                if n and (k := (name_key(province), name_key(n))) in cod
            ),
            None,
        )
        if code is None:
            logger.warning("No COD-AB zone for %s / %s", province, zone)
        codes.append(code)
    return gazetteer.select("province", "zone").with_columns(
        code=pl.Series(codes, dtype=pl.String)
    )


def main() -> None:
    """Write the zone shapes and the gazetteer's zone codes."""
    features = load_admin3()

    shapes = geometry_table(features)
    shapes.write_csv(GEOMETRY_PATH)
    logger.info(
        "Wrote %d zones (%.1f MB) to %s",
        shapes.height,
        GEOMETRY_PATH.stat().st_size / 1e6,
        GEOMETRY_PATH.name,
    )

    gazetteer = pl.read_csv(GAZETTEER_PATH, infer_schema=False)
    codes = zone_codes(features, gazetteer)
    codes.write_csv(CODES_PATH)
    logger.info(
        "Wrote %d zones, %d with a code, to %s",
        codes.height,
        codes["code"].is_not_null().sum(),
        CODES_PATH.name,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
