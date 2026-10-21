# Ebola 2026 outbreak

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install dependencies
uv sync
```

## Project Structure

- `src/ebola_2026_outbreak/`: Reusable analysis code (importable package)
- `data/raw/`: Input data files (gitignored — never commit source data)
- `data/processed/`: Derived/processed data (tracked in git)

Files are prefixed by outbreak so historical and current data stay apart:

- `west_africa_2014_16_*`: 2014-16 West Africa outbreak (WHO via
  [HDX](https://data.humdata.org/dataset/ebola-cases-2014)), used for comparison.
  Built by `uv run python src/ebola_2026_outbreak/west_africa_2014_16.py`.
  WHO's series starts on 29 August 2014; `west_africa_2014_16_early.py` fills
  the months before from OCHA ROWCA's national confirmed counts
  ([HDX](https://data.humdata.org/dataset/rowca-ebola-cases)) into
  `west_africa_2014_16_early_confirmed_cases.csv`.
- `drc_2026_*`: 2026 DRC Bundibugyo virus disease outbreak.
  - `drc_2026_who_by_country.csv`: the WHO country table from the
    [alert-and-response page](https://www.who.int/emergencies/alert-and-response).
    WHO shows only the latest figures, so each run of
    `uv run python src/ebola_2026_outbreak/drc_2026_who.py` adds to the history.
    `drc_2026_who_wayback.py` backfills it from Wayback Machine captures.
    Until early September WHO published the table as an image;
    `drc_2026_who_ocr.py` reads those images from the cached captures (run
    `uv sync --group ocr` first).
  - `drc_2026_sitrep_manifest.csv`: every DRC Ministry of Health sitrep PDF
    with its report date, built by `drc_2026_sitreps_download.py`. PDFs are
    saved to `data/raw/drc_2026/sitreps/`.
  - `drc_2026_sitrep_by_province.csv`: new, cumulative confirmed cases and
    cumulative confirmed deaths by province from each sitrep's province
    table, built by `drc_2026_sitreps_parse.py`.
  - `drc_2026_indicators_by_date.csv` and
    `drc_2026_cumulative_confirmed_cases.csv`: national totals summed across
    provinces, in the same shape and with the same final indicator label as
    the West Africa files. Built by `drc_2026_final.py`.

### Chart tables

`uv run python src/ebola_2026_outbreak/chart_tables.py` writes display-ready
tables to `data/output/`, refreshed by the daily scrape:

- `2026_outbreak_country_summary.csv`: latest WHO confirmed cases, deaths,
  case fatality rate and recoveries per country, with a Total row.
- `2026_outbreak_drc_cases_by_province.csv`: latest sitrep confirmed cases for
  each province with cases, keyed to the map OBJECTIDs in
  `data/reference/drc_provinces_objectid.csv`.
- `outbreak_comparison_weekly.csv`: cumulative confirmed cases by week since
  each outbreak's first confirmed case (22 March 2014 and 15 May 2026). West
  Africa uses ROWCA counts before 29 August 2014 and WHO counts after.

### Tests

```bash
uv run pytest
```

### Code Quality

```bash
# Run linter
uv run ruff check .

# Run formatter
uv run ruff format .

# Run type checker
uv run ty check src/ebola_2026_outbreak
```

## Pre-commit Hooks

Pre-commit hooks are configured to run automatically. To manually run:

```bash
pre-commit run --all-files
```

## License

This project is licensed under the mit License - see the LICENSE file for details.
