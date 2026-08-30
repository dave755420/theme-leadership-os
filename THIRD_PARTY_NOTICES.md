# Third-party notices

Theme Leadership OS keeps its declared runtime dependency set small: NumPy and
pandas provide the numeric and tabular primitives used by the signal and
validation modules. PyYAML, yfinance, Typer, and Streamlit are optional imports
for catalog parsing, personal research, and presentation; they are not required
for the local CSV/core path. All listed packages use permissive licenses
compatible with Apache-2.0 distribution.

The NumPy, pandas, and development-tool version ranges below are declared in
`pyproject.toml`. The optional package entries are intentionally not required
by the core package; when one is installed, the environment lock is the
authoritative version record for that run.

| Package | Use | License | Project |
| --- | --- | --- | --- |
| numpy | Numeric arrays used by the signal/validation implementations | BSD-3-Clause | <https://github.com/numpy/numpy> |
| pandas | Tabular time-series operations used by the signal/validation implementations | BSD-3-Clause | <https://github.com/pandas-dev/pandas> |
| PyYAML | Optional YAML parser for the versioned catalog; a constrained fallback parser is bundled | MIT | <https://github.com/yaml/pyyaml> |
| yfinance | Optional personal-research price adapter, explicitly disabled by default | Apache-2.0 | <https://github.com/ranaroussi/yfinance> |
| Typer | Optional CLI command framework; argparse fallback remains available | MIT | <https://github.com/fastapi/typer> |
| Streamlit | Optional presentation/dashboard runtime | Apache-2.0 | <https://github.com/streamlit/streamlit> |
| setuptools | Build backend | MIT | <https://github.com/pypa/setuptools> |
| pytest | Test runner | MIT | <https://github.com/pytest-dev/pytest> |
| ruff | Linting and formatting checks | MIT | <https://github.com/astral-sh/ruff> |
| mypy | Optional static type checking | MIT | <https://github.com/python/mypy> |

Optional packages are not a data license. The yfinance adapter is personal
research only, marks its observations non-redistributable and not PIT-complete,
and never stores or returns the provider's raw response frame. It must not be
used as an automatic catalog source. Source-specific terms, attribution
requirements, and retention decisions must be recorded in the source registry
and followed under [`docs/DATA_POLICY.md`](docs/DATA_POLICY.md). In particular,
raw market data, vendor payloads, and restricted snapshots must not be
redistributed from this repository.
