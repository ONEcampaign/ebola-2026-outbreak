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

