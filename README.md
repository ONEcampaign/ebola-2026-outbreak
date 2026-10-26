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
    `uv run python src/ebola_2026_outbreak/drc_2026_who.py` adds to the history;
    the `who_scrape.yml` workflow runs it daily at 12:00 UTC.
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
  - `drc_2026_inrb_national.csv`: national cumulative confirmed cases and
    deaths from INSP sitreps, as digitised by INRB and the Kraemer Lab
    (Oxford) in [BDBV2026-Data](https://github.com/INRB-UMIE/BDBV2026-Data).
    Pinned to one commit (`INRB_REF`); rerun
    `drc_2026_inrb.py` by hand after bumping it. INRB's metadata asks users to
    confirm distribution terms with INSP before republishing externally.
  - `drc_2026_indicators_by_date.csv` and
    `drc_2026_cumulative_confirmed_cases.csv`: national totals summed across
    provinces, in the same shape and with the same final indicator label as
    the West Africa files. Built by `drc_2026_final.py`. Dates with no parsed
    sitrep, including most of 16 June to 5 August, which the Ministry's media
    API does not list, come from the INRB series. The intermediate's
    `source` column says which.

### Zone-level sitreps

From SitRep 96 (report date 19 August 2026), the sitrep's health-zone table is
read by two models, Gemini and Mistral OCR, and checked against the PDF's text
layer and the table's own sums. `drc_2026_sitreps_extract.py` runs in the daily
scrape at 12:30 and 18:00 UTC (`extract`, then `release`) and writes:

- `drc_2026_sitrep_by_zone.csv`: one row per date, zone and indicator (cumulative
  confirmed cases and deaths, new cases, community, CTE and total deaths in the
  last 24 hours). The `level` column is `zone`, `province` (the table's
  province subtotals) or `national` (its Total row), so sum one level only.
  A blank `value` is a cell the sitrep leaves empty, and 0 is a printed zero.
- `drc_2026_sitrep_headline.csv`: the six headline indicators (cumulative cases
  and deaths, case fatality rate, people in isolation, recoveries, contacts
  followed up).
- `drc_2026_sitrep_extraction_status.csv`: one row per uploaded PDF, with its
  status (`released`, `held`, `pending` or `superseded`) and the number of cells
  flagged.

A report is released when every value is accepted or typed by a person. A
table row is accepted when each field matches the PDF text layer, whether both
models agree on it or only one model matches. A headline figure is accepted
when both models agree. Otherwise the report is held, and the run opens one
issue that tags the reviewers. To review a held report:

1. Open its file in `data/review/drc_2026/`. Each row is a cell to check, with
   the PDF page to open in `pdf_url`.
1. Type the printed value in `reviewed_value`, either a number or `blank` for
   an empty cell. Type `absent` where a flagged zone name is a misreading and the
   table has no such row. If a zone name is unknown, add it or its spelling to
   `aliases` in `data/reference/drc_2026_health_zones.csv`.
1. Open a pull request that says `Refs #N` for the issue. The review check runs
   `drc_2026_sitreps_extract.py check` and passes when the report would be
   released. The check runs the base branch's code against the pull request's
   data. Merging runs the release, and the issue closes once all its reports
   are out.

The extraction needs two repository secrets, `GEMINI_API_KEY` and
`MISTRAL_API_KEY`. Only the extract step and the weekly golden check read them.
Raw model answers are cached in `data/interim/drc_2026_sitrep_extractions/`, so
`release` needs no keys and no PDFs.

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
- `2026_outbreak_headline_figures.json`: four figures, each with `value` and
  `update_date`, written as day and month ("18 October").
  `total_confirmed_cases`, `total_deaths` (confirmed) and
  `death_rate` (confirmed case fatality rate, in percent) come from WHO's
  Total row across all affected countries, dated by the latest WHO "Data as
  of" date. `provinces_affected` counts DRC provinces with confirmed cases in
  the latest sitrep, dated by that sitrep.

### Health expenditure

`uv run python src/ebola_2026_outbreak/drc_health_expenditure.py` writes
`data/output/drc_expenditure_by_source.csv`: one row per year from 2013 to 2023
with DR Congo's total current health expenditure in millions of constant 2023
USD and each source's share of it in percent (domestic government, external,
other private, out-of-pocket). Data come from WHO's GHED via `bblocks.data_importers`, using
the method of `expenditure_by_source.csv` in
[topic_health_financing](https://github.com/ONEcampaign/topic_health_financing).

`uv run python src/ebola_2026_outbreak/drc_debt_vs_health.py` writes
`data/output/drc_debt_service_health_spending.csv`: one row per year from 2013 with DR
Congo's external debt service and domestic general government health
expenditure (GHED's GGHE-D), both in millions of current USD. Debt service sums
the outflows to every creditor in ONE's net flows dataset, which must be saved
first at `data/raw/net_flows_agg.parquet`.

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
