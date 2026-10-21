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
- `drc_2026_*`: 2026 DRC Bundibugyo virus disease outbreak.
  - `drc_2026_who_by_country.csv`: the WHO country table from the
    [alert-and-response page](https://www.who.int/emergencies/alert-and-response).
    WHO shows only the latest figures, so each run of
    `uv run python src/ebola_2026_outbreak/drc_2026_who.py` adds to the history.
    `drc_2026_who_wayback.py` backfills it from Wayback Machine captures.
  - `drc_2026_sitrep_manifest.csv`: every DRC Ministry of Health sitrep PDF
    with its report date, built by `drc_2026_sitreps_download.py`. PDFs are
    saved to `data/raw/drc_2026/sitreps/`.

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
