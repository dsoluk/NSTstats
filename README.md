# NSTstats

High-level documentation for the project now lives under the `docs/` folder.

- Start here: [docs/PROCESS.md](docs/PROCESS.md)
  - End-to-end workflow (ingest → merge → schedule → forecast → compare → analyze)
  - Expanded Data Quality (DQ) section with checks, common mismatches, remediation, and a suggested DQ workflow
  - Mermaid flowchart of the pipeline (renders in PyCharm)

## Quick start

Common CLI commands (see details in `docs/PROCESS.md`):

```bash
# End-to-end input refresh, then merge
python -m app.cli all
python -m app.cli merge

# Build or refresh schedule lookup
python -m app.cli --schedule-lookup

# Forecasts
python -m app.cli forecast --current-week 12

# Compare forecasts to external projections
python -m app.cli compare --current-week 12

# Analyze season-total metrics
python -m app.cli analyze

# Distribution diagnostics (data quality)
python -m app.cli dq --windows szn l7 --metrics G A PPP SOG FOW HIT BLK PIM
```

Note: On Windows/PowerShell, the above commands work the same; ensure your virtual environment is activated.

## Viewing diagrams in PyCharm

Open `docs/PROCESS.md` and enable the Markdown preview. PyCharm (2023.2+) can render Mermaid diagrams directly in the editor, so the workflow flowchart will appear inline.

## Contributing docs

We expect to add more documentation over time. Please place new docs under `docs/` and link them here. Consider adding:
- `docs/DQ.md` for deeper data-quality specs and gate criteria
- `docs/FORECASTING.md` for methodology details and formulas
- `docs/CLI.md` for exhaustive command/flag reference
