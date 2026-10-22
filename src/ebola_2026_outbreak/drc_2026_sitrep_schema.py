"""Shared contracts for extracting DRC sitreps with two models.

Each sitrep from SitRep 96 (report date 2026-08-19) onwards is read by Gemini
and by Mistral OCR, both asked for the same JSON object (`JSON_SCHEMA`) with
the same instructions (`PROMPT`). The object has five sections: the headline
indicators of page 1 (`kpis`), the province summary table (`provinces`), and
the zone table's zone rows (`zones`), province subtotal rows
(`zone_subtotals`) and Total row (`zone_total`). A blank cell is null and a
printed 0 is 0, so the two stay distinct all the way to the outputs.

`normalise` turns a model's raw object into rows keyed on
(section, province, zone). Provinces are matched with
`drc_2026_sitreps_parse.canonical_province`. Zone names are matched on a key
that drops case, accents, spaces, hyphens and punctuation, first against the
zone names and then the aliases of the gazetteer
(`data/reference/drc_2026_health_zones.csv`). The gazetteer also fixes each
zone's province, so a zone filed under another province, or a name it does
not know, flags every field of that row for review. An unknown name is
resolved by adding an alias to the gazetteer.

Raw model responses and the words of the PDF's table pages are cached in
`data/interim/drc_2026_sitrep_extractions/<sha256>/`, keyed on the PDF's
sha256 so a re-upload never overwrites the previous version's record. Parsed
extractions and text-layer rows are not cached: release normalises again
from the raw responses and reads the rows again from the words, so a
gazetteer change takes effect on the next run.
"""

import copy
import hashlib
import json
import logging
import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any, Literal, NamedTuple

import polars as pl

from ebola_2026_outbreak.drc_2026_sitreps_download import OUTBREAK, ROOT
from ebola_2026_outbreak.drc_2026_sitreps_parse import TOTAL, canonical_province

logger = logging.getLogger(__name__)

# First report in the current layout; zone-level extraction starts here.
ZONE_START = date(2026, 8, 19)

GAZETTEER_PATH = ROOT / "data" / "reference" / f"{OUTBREAK}_health_zones.csv"
CACHE_DIR = ROOT / "data" / "interim" / f"{OUTBREAK}_sitrep_extractions"
REVIEW_DIR = ROOT / "data" / "review" / OUTBREAK

type Section = Literal["kpis", "provinces", "zone_subtotals", "zone_total", "zones"]
type Extractor = Literal["gemini", "mistral"]
type FlagReason = Literal[
    "models_disagree",
    "text_mismatch",
    "row_not_located",
    "unknown_name",
    "wrong_province",
    "duplicate_row",
    "sum_mismatch",
    "section_missing",
]
type Status = Literal["released", "held", "pending", "superseded"]
type Number = int | float
type Values = dict[str, Number | None]

SECTIONS: tuple[Section, ...] = (
    "kpis",
    "provinces",
    "zone_subtotals",
    "zone_total",
    "zones",
)
EXTRACTORS: tuple[Extractor, ...] = ("gemini", "mistral")

KPI_FIELDS = (
    "cum_confirmed",
    "cum_deaths",
    "cfr_pct",
    "in_isolation",
    "cum_recovered",
    "contact_followup_pct",
)
PROVINCE_FIELDS = (
    "new_confirmed",
    "cum_confirmed",
    "cum_deaths",
    "cfr_pct",
    "zs_affected",
    "zs_total",
)
# Columns of the zone table, shared by its zone, subtotal and Total rows.
ZONE_TABLE_FIELDS = (
    "cum_confirmed",
    "cum_deaths",
    "cfr_pct",
    "new_confirmed",
    "community_deaths",
    "cte_deaths",
    "total_deaths_day",
)
# Numeric fields of each section, in schema order. Names are not fields.
FIELDS: dict[Section, tuple[str, ...]] = {
    "kpis": KPI_FIELDS,
    "provinces": PROVINCE_FIELDS,
    "zone_subtotals": ZONE_TABLE_FIELDS,
    "zone_total": ZONE_TABLE_FIELDS,
    "zones": ZONE_TABLE_FIELDS,
}
PERCENT_FIELDS = frozenset({"cfr_pct", "contact_followup_pct"})
# Zone-table counts that add up: zones to subtotals, subtotals to Total.
COUNT_FIELDS = tuple(f for f in ZONE_TABLE_FIELDS if f not in PERCENT_FIELDS)

# `level` column of the by-zone output for each zone-table section.
LEVELS: dict[Section, str] = {
    "zones": "zone",
    "zone_subtotals": "province",
    "zone_total": "national",
}
# Output indicator labels. The first three match drc_2026_sitreps_parse.
ZONE_INDICATORS = {
    "cum_confirmed": "Cumulative confirmed cases",
    "cum_deaths": "Cumulative confirmed deaths",
    "new_confirmed": "New confirmed cases (24h)",
    "community_deaths": "Community deaths (24h)",
    "cte_deaths": "CTE deaths (24h)",
    "total_deaths_day": "Total deaths (24h)",
}
HEADLINE_INDICATORS = {
    "cum_confirmed": "Cumulative confirmed cases",
    "cum_deaths": "Cumulative confirmed deaths",
    "cfr_pct": "Case fatality rate (%)",
    "in_isolation": "Patients in isolation",
    "cum_recovered": "Cumulative recoveries",
    "contact_followup_pct": "Contacts followed up (%)",
}


class RowKey(NamedTuple):
    """One table row. `province` and `zone` are "" where they do not apply.

    `kpis` and `zone_total` rows have both empty, `provinces` and
    `zone_subtotals` rows have `zone` empty. Names are canonical, except in a
    row flagged `unknown_name`, which keeps the name as the model printed it.
    """

    section: Section
    province: str
    zone: str


class Flag(NamedTuple):
    """One value held for review.

    `field` is None only for `section_missing`, where no row exists. `page`
    is the 1-based PDF page of the row, when known.
    """

    key: RowKey
    field: str | None
    reason: FlagReason
    page: int | None = None


class Discrepancy(NamedTuple):
    """A failed sum check whose operands were all verified by a reviewer.

    The report is released as printed and the discrepancy is recorded.
    `key` is the row holding the printed figure the operands should equal.
    """

    key: RowKey
    field: str
    printed: Number | None
    computed: Number


class ModelResult(NamedTuple):
    """One model's answer for one PDF, as cached.

    Attributes:
        extraction_raw: The JSON object the model returned, before
            normalisation.
        model_id_requested: Model id sent in the request (the pin).
        model_id_returned: Model id the response reports, if any.
        usage: Token or page counts as the API reports them.
        requested_at: UTC timestamp of the request, ISO 8601.
        response: The full response body, kept for audit.
    """

    extraction_raw: dict[str, Any]
    model_id_requested: str
    model_id_returned: str | None
    usage: dict[str, Any]
    requested_at: str
    response: dict[str, Any]


class TextRow(NamedTuple):
    """A table row as printed in the PDF text layer.

    `printed` is the row's tokens after the name, joined by single spaces
    ("12 6 50,0 1 0 0 0"), or None when the row was not located.
    """

    key: RowKey
    printed: str | None
    located: bool
    page: int | None


class Extraction(NamedTuple):
    """A model's extraction with every row under its canonical key."""

    sitrep_no: int | None
    report_date: str | None
    rows: dict[RowKey, Values]


class Normalised(NamedTuple):
    """Result of `normalise`: the rows, and the flags its names raised."""

    extraction: Extraction
    name_flags: list[Flag]


class Reconciled(NamedTuple):
    """Values accepted from the two models, and the values held for review.

    `accepted` holds only accepted fields, so a row with one flagged field
    lacks that field.
    """

    accepted: dict[RowKey, Values]
    flags: list[Flag]


class HealthZone(NamedTuple):
    """A gazetteer entry: canonical province and zone names."""

    province: str
    zone: str


# Name key -> every zone it names. More than one only when provinces share a
# zone name, and the model's province then picks one.
type Gazetteer = dict[str, tuple[HealthZone, ...]]


# Extraction schema


def _nullable(kind: str) -> dict[str, Any]:
    return {"type": [kind, "null"]}


def _object(properties: dict[str, Any], *, nullable: bool = False) -> dict[str, Any]:
    """A strict object: every property required, no others allowed."""
    return {
        "type": ["object", "null"] if nullable else "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _fields(fields: tuple[str, ...]) -> dict[str, Any]:
    return {
        f: _nullable("number" if f in PERCENT_FIELDS else "integer") for f in fields
    }


_NAME = {"type": "string"}

JSON_SCHEMA = _object(
    {
        "sitrep_no": _nullable("integer"),
        "report_date": _nullable("string"),
        "kpis": _object(_fields(KPI_FIELDS), nullable=True),
        "provinces": {
            "type": "array",
            "items": _object({"province": _NAME, **_fields(PROVINCE_FIELDS)}),
        },
        "zone_subtotals": {
            "type": "array",
            "items": _object({"province": _NAME, **_fields(ZONE_TABLE_FIELDS)}),
        },
        "zone_total": _object(_fields(ZONE_TABLE_FIELDS), nullable=True),
        "zones": {
            "type": "array",
            "items": _object(
                {"province": _NAME, "zone": _NAME, **_fields(ZONE_TABLE_FIELDS)}
            ),
        },
    }
)


def schema_for(extractor: Extractor) -> dict[str, Any]:
    """Return the response schema to send to one extractor.

    Every vendor-specific change to the schema belongs here. Both extractors
    take `JSON_SCHEMA` unchanged.
    """
    return copy.deepcopy(JSON_SCHEMA)


PROMPT = """\
You extract structured data from a situation report (SitRep) on an Ebola virus \
disease outbreak in the Democratic Republic of the Congo. The report is in \
French. Return one JSON object that follows the supplied JSON schema. Return \
only the JSON.

Fields:

- `sitrep_no`: the report number as an integer.
- `report_date`: the date the report covers, as `YYYY-MM-DD`.
- `kpis`: the headline indicators of the first page: cumulative confirmed \
cases, cumulative deaths, case fatality rate (CFR), patients in isolation, \
cumulative recovered, and the percentage of contacts followed. Use `null` for \
any indicator the report does not show.
- `provinces`: one row per province from the province summary table (usually \
"Tableau 1" or its equivalent). Leave out the Total row. `zs_affected` and \
`zs_total` come from the "ZS historiquement touchées" column, written like \
`28/36`, which gives 28 and 36.
- `zones`: one row per health zone (zone de santé, ZS) from the zone-level \
table (usually "Tableau 2"). Leave out province subtotal rows and the Total \
row. Each row carries its parent province. Keep rows such as `A ventiler`. \
Copy zone names as printed, joining names that wrap across lines (`Makiso-` \
followed by `Kisangani` becomes `Makiso-Kisangani`), and drop footnote marks \
such as `*` or `**` after a name.
- `zone_subtotals`: one row per province subtotal row of the same zone-level \
table, with the values printed on that subtotal row.
- `zone_total`: the Total row of the zone-level table.
- If the report has no province table or no zone table, return an empty array \
for that section; if it has no headline indicators, return `null` for `kpis`; \
if the zone table has no Total row, return `null` for `zone_total`.

Value rules:

- Integers have no thousands separator: `9 277` and `9.277` become 9277.
- Percentages are floats without the percent sign, with a point as the \
decimal separator: `48,3 %` becomes 48.3.
- A blank cell is `null`. A printed `0` is 0. `NA` or `N/A` is `null`.
- Copy numbers exactly as printed. Do not compute, correct, or infer any \
value, and do not fill a blank from another table or another row.
- Read each value from the column whose header names it. Tables may have \
blank cells, so keep each value under its own column.
"""


# Names


def name_key(name: str) -> str:
    """Match key for a place name.

    Drops case, accents, spaces, hyphens and punctuation, so
    "Makiso-Kisangani", "MakisoKisangani" and "MAKISO KISANGANI" share a key,
    and footnote marks ("A ventiler **") fall away.
    """
    decomposed = unicodedata.normalize("NFKD", name).casefold()
    letters = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[\W_]+", "", letters)


def _province(label: str) -> str | None:
    """Canonical province for a label, "Total", or None if unknown."""
    cleaned = " ".join(re.sub(r"[^\w\s-]", " ", label).split())
    try:
        return canonical_province(cleaned)
    except ValueError:
        return None


def load_gazetteer(path: Path = GAZETTEER_PATH) -> Gazetteer:
    """Read the health-zone gazetteer.

    Args:
        path: CSV with columns `province, zone, aliases`. Aliases are
            `|`-separated.

    Returns:
        Name key of every zone name and alias -> the zones it names.

    Raises:
        ValueError: If a province is not canonical, a zone is listed twice,
            or one name key names two zones of the same province.
    """
    df = pl.read_csv(
        path, schema={"province": pl.String, "zone": pl.String, "aliases": pl.String}
    )
    names: defaultdict[str, set[HealthZone]] = defaultdict(set)
    seen: set[HealthZone] = set()
    for province, zone, aliases in df.iter_rows():
        if _province(province) != province:
            raise ValueError(f"Province not canonical in gazetteer: {province!r}")
        entry = HealthZone(province, zone)
        if entry in seen:
            raise ValueError(f"Zone listed twice in gazetteer: {entry}")
        seen.add(entry)
        for name in [zone, *(aliases.split("|") if aliases else [])]:
            names[name_key(name)].add(entry)

    for key, zones in names.items():
        provinces = [z.province for z in zones]
        if not key or len(set(provinces)) != len(provinces):
            raise ValueError(f"Gazetteer name {key!r} is ambiguous: {sorted(zones)}")
    return {key: tuple(sorted(zones)) for key, zones in names.items()}


# Normalisation


def _get(obj: object, where: str, field: str) -> Any:
    """`obj[field]`, where `where` names `obj` in an error.

    Raises:
        ValueError: If `obj` is not an object or lacks `field`.
    """
    if not isinstance(obj, Mapping):
        raise ValueError(f"{where} is not an object: {obj!r}")
    if field not in obj:
        raise ValueError(f"{where}.{field} is missing")
    return obj[field]


def _list(raw: object, section: Section) -> list[Any]:
    rows = _get(raw, "extraction", section)
    if not isinstance(rows, list):
        raise ValueError(f"{section} is not a list: {rows!r}")
    return rows


def _name(row: object, section: Section, field: str) -> str:
    name = _get(row, section, field)
    if not isinstance(name, str):
        raise ValueError(f"{section}.{field} is not a string: {name!r}")
    return name


def _values(row: object, section: Section) -> Values:
    """Read and type-check a row's numeric fields.

    Raises:
        ValueError: On a row that is not an object, a missing field, or a
            count that is not a whole number.
    """
    values: Values = {}
    for field in FIELDS[section]:
        value = _get(row, section, field)
        if value is None:
            values[field] = None
        elif isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{section}.{field} is not a number: {value!r}")
        elif field in PERCENT_FIELDS:
            values[field] = float(value)
        elif isinstance(value, float) and not value.is_integer():
            raise ValueError(f"{section}.{field} is not a whole number: {value!r}")
        else:
            values[field] = int(value)
    return values


def normalise(raw: Mapping[str, Any], gazetteer: Gazetteer) -> Normalised:
    """Key a model's raw extraction on canonical names.

    A province or zone label that is unknown, a zone filed under a province
    other than the gazetteer's, and a row repeated under one key each flag
    every field of the row. The first of repeated rows is kept. A Total row
    filed among provinces, subtotals or zones is dropped, since the prompt
    excludes it and `zone_total` carries the zone table's Total.

    Args:
        raw: The object the model returned, shaped by `JSON_SCHEMA`.
        gazetteer: From `load_gazetteer`.

    Returns:
        The rows under canonical keys, and the name flags.

    Raises:
        ValueError: If `raw` breaks the schema (a missing field or a value of
            the wrong type); the message names the section and field.
    """
    rows: dict[RowKey, Values] = {}
    flags: list[Flag] = []

    def add(key: RowKey, values: Values, reason: FlagReason | None = None) -> None:
        if key in rows:
            reason = "duplicate_row"
        else:
            rows[key] = values
        if reason is not None:
            flags.extend(Flag(key, field, reason) for field in FIELDS[key.section])

    for section in ("kpis", "zone_total"):
        if (row := _get(raw, "extraction", section)) is not None:
            add(RowKey(section, "", ""), _values(row, section))

    for section in ("provinces", "zone_subtotals"):
        for row in _list(raw, section):
            label = _name(row, section, "province")
            province = _province(label)
            values = _values(row, section)
            if province == TOTAL:
                logger.info("Dropping a Total row filed in %s", section)
            elif province is None:
                add(RowKey(section, label.strip(), ""), values, "unknown_name")
            else:
                add(RowKey(section, province, ""), values)

    for row in _list(raw, "zones"):
        label, zone = _name(row, "zones", "province"), _name(row, "zones", "zone")
        province = _province(label)
        values = _values(row, "zones")
        if name_key(zone) == name_key(TOTAL) or province == TOTAL:
            logger.info("Dropping a Total row filed in zones")
            continue
        candidates = gazetteer.get(name_key(zone), ())
        match = next((z for z in candidates if z.province == province), None)
        if match is not None:
            add(RowKey("zones", *match), values)
        elif len(candidates) == 1:
            add(RowKey("zones", *candidates[0]), values, "wrong_province")
        else:
            printed = RowKey("zones", province or label.strip(), zone.strip())
            add(printed, values, "wrong_province" if candidates else "unknown_name")

    extraction = Extraction(
        _get(raw, "extraction", "sitrep_no"),
        _get(raw, "extraction", "report_date"),
        rows,
    )
    return Normalised(extraction, flags)


# Cache


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file, the cache and review key of a PDF."""
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def cache_path(
    sha256: str, name: Extractor | Literal["words"], *, root: Path = CACHE_DIR
) -> Path:
    """Path of one cached record for a PDF."""
    return root / sha256 / f"{name}.json"


def _write_json(path: Path, data: object) -> None:
    """Write JSON through a temporary file, so a crash leaves no partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def write_model_result(
    sha256: str, extractor: Extractor, result: ModelResult, *, root: Path = CACHE_DIR
) -> Path:
    """Cache a model's answer as `<root>/<sha256>/<extractor>.json`.

    The record is a JSON object with the `ModelResult` fields as keys.
    """
    path = cache_path(sha256, extractor, root=root)
    _write_json(path, result._asdict())
    return path


def read_model_result(
    sha256: str, extractor: Extractor, *, root: Path = CACHE_DIR
) -> ModelResult | None:
    """Read a cached model answer, or None if the PDF has none yet."""
    path = cache_path(sha256, extractor, root=root)
    if not path.exists():
        return None
    return ModelResult(**json.loads(path.read_text()))


def write_words(sha256: str, dump: Path, *, root: Path = CACHE_DIR) -> Path:
    """Cache a word dump as `<root>/<sha256>/words.json`.

    Args:
        sha256: The PDF's sha256.
        dump: The words of the PDF's table pages, as
            `drc_2026_sitrep_textlayer.dump_words` writes them.
        root: The cache directory.
    """
    path = cache_path(sha256, "words", root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_bytes(dump.read_bytes())
    tmp.replace(path)
    return path


def read_words(sha256: str, *, root: Path = CACHE_DIR) -> Path | None:
    """Path of a PDF's cached word dump, or None if the PDF has none yet."""
    path = cache_path(sha256, "words", root=root)
    return path if path.exists() else None
