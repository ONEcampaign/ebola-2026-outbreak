"""Tests for the DR Congo health zone shapes and codes."""

import json

import polars as pl
import pytest

from ebola_2026_outbreak.drc_2026_sitrep_schema import GAZETTEER_PATH
from ebola_2026_outbreak.drc_health_zone_geometry import (
    COD_AB_PATH,
    EXPECTED_ZONES,
    NOT_MAPPED,
    cod_ab_table,
    geometry_table,
    zone_codes,
)

SQUARE = [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]
NEXT_SQUARE = [[[1, 0], [2, 0], [2, 1], [1, 1], [1, 0]]]


def feature(code: str, name: str, province: str, rings: list) -> dict:
    """A COD-AB admin 3 feature stub."""
    return {
        "type": "Feature",
        "properties": {"adm3_pcode": code, "adm3_name": name, "adm1_name": province},
        "geometry": {"type": "Polygon", "coordinates": rings},
    }


def test_geometry_table_writes_geojson_polygons() -> None:
    out = geometry_table(
        [
            feature("CD2", "Aru", "Ituri", NEXT_SQUARE),
            feature("CD1", "Bunia", "Ituri", SQUARE),
        ]
    )
    assert out.columns == ["Geometry", "Code", "Name"]
    assert out["Code"].to_list() == ["CD1", "CD2"]
    assert json.loads(out["Geometry"][0])["type"] == "Polygon"


def test_geometry_table_rejects_repeated_codes() -> None:
    with pytest.raises(ValueError, match="not unique"):
        geometry_table(
            [
                feature("CD1", "A", "Ituri", SQUARE),
                feature("CD1", "B", "Ituri", NEXT_SQUARE),
            ]
        )


def test_cod_ab_table_keeps_province_zone_and_code() -> None:
    out = cod_ab_table(
        [
            feature("CD2", "Aru", "Ituri", NEXT_SQUARE),
            feature("CD1", "Bunia", "Ituri", SQUARE),
        ]
    )
    assert out.rows() == [("Ituri", "Bunia", "CD1"), ("Ituri", "Aru", "CD2")]


def cod_ab(rows: list[tuple[str, str, str]]) -> pl.DataFrame:
    """COD-AB zone table from (province, zone, code) tuples."""
    return pl.DataFrame(rows, schema=["province", "zone", "code"], orient="row")


def test_zone_codes_matches_spellings_and_accents() -> None:
    zones = cod_ab([("Ituri", "Mongbalu", "CD1"), ("Bas-Uele", "Buta", "CD2")])
    gazetteer = pl.DataFrame(
        {
            "province": ["Ituri", "Bas-Uélé", "Bas-Uélé"],
            "zone": ["Mongbwalu", "Buta", "Dakwa"],
            "aliases": [None, None, None],
        },
        schema={"province": pl.String, "zone": pl.String, "aliases": pl.String},
    )
    out = zone_codes(zones, gazetteer)
    assert out["code"].to_list() == ["CD1", "CD2", None]


def test_zone_codes_matches_a_gazetteer_alias() -> None:
    gazetteer = pl.DataFrame(
        {"province": ["Ituri"], "zone": ["Bunya"], "aliases": ["Foo|Bunia"]}
    )
    out = zone_codes(cod_ab([("Ituri", "Bunia", "CD1")]), gazetteer)
    assert out["code"].to_list() == ["CD1"]


def test_zone_codes_maps_a_health_area_to_its_zone() -> None:
    gazetteer = pl.DataFrame(
        {"province": ["Bas-Uélé"], "zone": ["Dakwa"], "aliases": [None]},
        schema={"province": pl.String, "zone": pl.String, "aliases": pl.String},
    )
    zones = cod_ab([("Bas-Uele", "Ango", "CD3")])
    assert zone_codes(zones, gazetteer)["code"].to_list() == ["CD3"]


def test_committed_cod_ab_table_has_every_zone_once() -> None:
    zones = pl.read_csv(COD_AB_PATH, infer_schema=False)
    assert zones.height == EXPECTED_ZONES
    assert zones["code"].n_unique() == EXPECTED_ZONES


def test_every_gazetteer_zone_has_a_code() -> None:
    # Fails when the gazetteer gains a zone that COD-AB spells differently.
    # Add the COD-AB spelling as a gazetteer alias.
    codes = zone_codes(
        pl.read_csv(COD_AB_PATH, infer_schema=False),
        pl.read_csv(GAZETTEER_PATH, infer_schema=False),
    )
    unresolved = codes.filter(pl.col("code").is_null()).select("province", "zone")
    assert set(unresolved.rows()) == NOT_MAPPED
