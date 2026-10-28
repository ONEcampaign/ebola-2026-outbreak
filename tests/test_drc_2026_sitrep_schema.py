"""Tests for the sitrep extraction contracts."""

from pathlib import Path
from typing import Any

import pytest

from ebola_2026_outbreak.drc_2026_sitrep_schema import (
    FIELDS,
    JSON_SCHEMA,
    SECTIONS,
    Flag,
    HealthZone,
    ModelResult,
    RowKey,
    canonical_key,
    is_total,
    load_gazetteer,
    name_key,
    normalise,
    read_model_result,
    read_words,
    write_model_result,
    write_words,
)

GAZETTEER = load_gazetteer()


def zone_row(province: str, zone: str, cases: Any = 12) -> dict[str, Any]:
    return {
        "province": province,
        "zone": zone,
        "cum_confirmed": cases,
        "cum_deaths": 6,
        "cfr_pct": 50,
        "new_confirmed": None,
        "community_deaths": 0,
        "cte_deaths": None,
        "total_deaths_day": 0,
    }


def raw(**sections: Any) -> dict[str, Any]:
    empty = {
        "kpis": None,
        "provinces": [],
        "zone_subtotals": [],
        "zone_total": None,
        "zones": [],
    }
    return {"sitrep_no": 157, "report_date": "2026-10-18", **empty, **sections}


def walk_objects(schema: dict[str, Any]) -> list[dict[str, Any]]:
    found = [schema] if "properties" in schema else []
    children = [*schema.get("properties", {}).values()]
    if "items" in schema:
        children.append(schema["items"])
    for child in children:
        found.extend(walk_objects(child))
    return found


def test_json_schema_is_strict_and_matches_fields() -> None:
    objects = walk_objects(JSON_SCHEMA)
    assert len(objects) == 6
    for obj in objects:
        assert obj["required"] == list(obj["properties"])
        assert obj["additionalProperties"] is False

    props = JSON_SCHEMA["properties"]
    assert set(props) == {"sitrep_no", "report_date", *SECTIONS}
    for section in SECTIONS:
        node = props[section].get("items", props[section])["properties"]
        numeric = [f for f in node if f not in ("province", "zone")]
        assert tuple(numeric) == FIELDS[section]
        assert all("null" in node[f]["type"] for f in numeric)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("Makiso-Kisangani", "MakisoKisangani"),
        ("Makiso-Kisangani", "MAKISO  KISANGANI"),
        ("Haut-Uélé", "haut uele"),
        ("A ventiler", "A ventiler **"),
        ("Nia-Nia", "Nia\N{NON-BREAKING HYPHEN}Nia"),
    ],
)
def test_name_key_matches_variants(a: str, b: str) -> None:
    assert name_key(a) == name_key(b)


def test_gazetteer_covers_every_seed_zone() -> None:
    zones = {z for entries in GAZETTEER.values() for z in entries}
    assert len(zones) == 67
    assert GAZETTEER[name_key("Boma Mangbetu")] == (
        HealthZone("Haut-Uélé", "Boma Mangbetu"),
    )


@pytest.mark.parametrize(
    "csv",
    [
        "province,zone,aliases\nBas Uélé,Buta,\n",
        "province,zone,aliases\nIturi,Bunia,\nIturi,Bunia,\n",
        "province,zone,aliases\nIturi,Bunia,\nIturi,Nyankunde,Bunia\n",
    ],
    ids=["province-not-canonical", "zone-twice", "alias-names-two-zones"],
)
def test_load_gazetteer_refuses(tmp_path: Path, csv: str) -> None:
    path = tmp_path / "zones.csv"
    path.write_text(csv)
    with pytest.raises(ValueError):
        load_gazetteer(path)


def test_load_gazetteer_reads_aliases(tmp_path: Path) -> None:
    path = tmp_path / "zones.csv"
    path.write_text("province,zone,aliases\nSud-Kivu,Miti-Murhesa,Mili Murhesa|Miti\n")
    gazetteer = load_gazetteer(path)
    assert gazetteer[name_key("Mili-Murhesa")] == (
        HealthZone("Sud-Kivu", "Miti-Murhesa"),
    )


def test_normalise_keys_rows_on_canonical_names() -> None:
    result = normalise(
        raw(
            kpis=dict.fromkeys(FIELDS["kpis"], 1),
            provinces=[
                {"province": "Bas Uélé**", **dict.fromkeys(FIELDS["provinces"], 3)},
                {"province": "Total", **dict.fromkeys(FIELDS["provinces"], 9)},
            ],
            zones=[zone_row("Bas Uélé", "BUTA"), zone_row("Ituri", "A ventiler **")],
        ),
        GAZETTEER,
    )

    assert result.name_flags == []
    rows = result.extraction.rows
    assert set(rows) == {
        RowKey("kpis", "", ""),
        RowKey("provinces", "Bas-Uélé", ""),
        RowKey("zones", "Bas-Uélé", "Buta"),
        RowKey("zones", "Ituri", "A ventiler"),
    }
    zone = rows[RowKey("zones", "Bas-Uélé", "Buta")]
    assert zone["cfr_pct"] == 50.0
    assert isinstance(zone["cfr_pct"], float)
    assert zone["new_confirmed"] is None
    assert zone["community_deaths"] == 0
    assert rows[RowKey("kpis", "", "")]["contact_followup_pct"] == 1.0


@pytest.mark.parametrize(
    ("zones", "key", "reason"),
    [
        (
            [zone_row("Tshopo", "Bunia")],
            RowKey("zones", "Ituri", "Bunia"),
            "wrong_province",
        ),
        (
            [zone_row("Ituri", "Bunya")],
            RowKey("zones", "Ituri", "Bunya"),
            "unknown_name",
        ),
        (
            [zone_row("Ituri", "Bunia"), zone_row("Ituri", "BUNIA", cases=99)],
            RowKey("zones", "Ituri", "Bunia"),
            "duplicate_row",
        ),
    ],
    ids=["wrong-province", "unknown-name", "duplicate"],
)
def test_normalise_flags_every_field_of_a_doubtful_row(
    zones: list[dict[str, Any]], key: RowKey, reason: str
) -> None:
    result = normalise(raw(zones=zones), GAZETTEER)
    assert result.name_flags == [Flag(key, f, reason) for f in FIELDS["zones"]]
    assert result.extraction.rows[key]["cum_confirmed"] == 12


def test_normalise_unknown_province() -> None:
    row = {"province": "Kinshasa", **dict.fromkeys(FIELDS["zone_subtotals"], 0)}
    result = normalise(raw(zone_subtotals=[row]), GAZETTEER)
    key = RowKey("zone_subtotals", "Kinshasa", "")
    assert {f.key for f in result.name_flags} == {key}
    assert {f.reason for f in result.name_flags} == {"unknown_name"}


@pytest.mark.parametrize(
    ("cases", "expected"),
    [(12.0, 12), ("12", ValueError), (12.5, ValueError), (True, ValueError)],
)
def test_normalise_type_checks_counts(cases: Any, expected: Any) -> None:
    extraction = raw(zones=[zone_row("Ituri", "Bunia", cases=cases)])
    if expected is ValueError:
        with pytest.raises(ValueError):
            normalise(extraction, GAZETTEER)
        return
    rows = normalise(extraction, GAZETTEER).extraction.rows
    value = rows[RowKey("zones", "Ituri", "Bunia")]["cum_confirmed"]
    assert value == expected
    assert isinstance(value, int)


@pytest.mark.parametrize(
    ("extraction", "message"),
    [
        (
            raw(
                zones=[
                    {
                        k: v
                        for k, v in zone_row("Ituri", "Bunia").items()
                        if k != "cum_confirmed"
                    }
                ]
            ),
            "zones.cum_confirmed is missing",
        ),
        (
            {k: v for k, v in raw().items() if k != "zones"},
            "extraction.zones is missing",
        ),
        (raw(zones=None), "zones is not a list"),
    ],
    ids=["missing-field", "missing-section", "null-section"],
)
def test_normalise_names_what_breaks_the_schema(
    extraction: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        normalise(extraction, GAZETTEER)


def test_cache_round_trip(tmp_path: Path) -> None:
    sha = "ab" * 32
    assert read_model_result(sha, "gemini", root=tmp_path) is None
    assert read_words(sha, root=tmp_path) is None

    result = ModelResult(
        extraction_raw=raw(),
        model_id_requested="gemini-3.8-flash",
        model_id_returned=None,
        usage={"totalTokenCount": 10},
        requested_at="2026-10-22T12:00:00Z",
        response={"candidates": []},
    )
    write_model_result(sha, "gemini", result, root=tmp_path)
    assert read_model_result(sha, "gemini", root=tmp_path) == result

    dump = Path(__file__).parent / "fixtures" / "sitrep_words" / "SitRep_096.json"
    path = write_words(sha, dump, root=tmp_path)
    assert read_words(sha, root=tmp_path) == path
    assert path.read_bytes() == dump.read_bytes()
    assert sorted(p.name for p in (tmp_path / sha).iterdir()) == [
        "gemini.json",
        "words.json",
    ]


@pytest.mark.parametrize(
    "label", ["Total", "TOTAL RDC", "Total général", "Total **", "Total Ituri"]
)
def test_total_labels_share_one_rule(label: str) -> None:
    assert is_total(name_key(label))
    assert canonical_key(RowKey("provinces", label, ""), GAZETTEER) is None
    assert canonical_key(RowKey("zones", "Ituri", label), GAZETTEER) is None


def test_total_prefix_rule_leaves_totaux_unknown() -> None:
    printed = RowKey("zone_subtotals", "Totaux", "")
    assert not is_total(name_key("Totaux"))
    assert canonical_key(printed, GAZETTEER) == (printed, "unknown_name")
