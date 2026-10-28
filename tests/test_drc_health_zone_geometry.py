"""Tests for the DR Congo health zone shapes and codes."""

import json

import polars as pl
import pytest

from ebola_2026_outbreak.drc_health_zone_geometry import geometry_table, zone_codes

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


def test_zone_codes_matches_spellings_and_accents() -> None:
    features = [
        feature("CD1", "Mongbalu", "Ituri", SQUARE),
        feature("CD2", "Buta", "Bas-Uele", NEXT_SQUARE),
    ]
    gazetteer = pl.DataFrame(
        {
            "province": ["Ituri", "Bas-Uélé", "Bas-Uélé"],
            "zone": ["Mongbwalu", "Buta", "Dakwa"],
            "aliases": [None, None, None],
        },
        schema={"province": pl.String, "zone": pl.String, "aliases": pl.String},
    )
    out = zone_codes(features, gazetteer)
    assert out["code"].to_list() == ["CD1", "CD2", None]


def test_zone_codes_maps_a_health_area_to_its_zone() -> None:
    features = [feature("CD3", "Ango", "Bas-Uele", SQUARE)]
    gazetteer = pl.DataFrame(
        {"province": ["Bas-Uélé"], "zone": ["Dakwa"], "aliases": [None]},
        schema={"province": pl.String, "zone": pl.String, "aliases": pl.String},
    )
    assert zone_codes(features, gazetteer)["code"].to_list() == ["CD3"]
