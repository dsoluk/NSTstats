# NSTstats

High-level documentation for the project now lives under the `docs/` folder.

- Start here: [docs/PROCESS.md](docs/PROCESS.md)
  - End-to-end workflow (ingest → merge → schedule → forecast → compare → analyze)
  - Expanded Data Quality (DQ) section with checks, common mismatches, remediation, and a suggested DQ workflow
  - Mermaid flowchart of the pipeline (renders in PyCharm)

## Junie Pro Guidelines

- Project-specific guidance: [docs/JUNIE_GUIDELINES.md](docs/JUNIE_GUIDELINES.md) — adds NSTstats preferences on top of the org-wide guide.
- Canonical org-wide guide: [.junie/guidelines.md](.junie/guidelines.md) — the single source of truth used by Junie across projects.

## Quick start

The project has been migrated to Django. You can now use the interactive menu or standard Django management commands.

### Interactive Menu
```bash
python manage.py menu
```

### Django Management Commands
The original CLI is still available via the `nststats` management command:

```bash
# End-to-end input refresh, then merge
python manage.py nststats daily
python manage.py nststats weekly

# Run original CLI commands
python manage.py nststats forecast --current-week 12
```

### Maintenance Interface (Django Admin)
You can maintain parameters and view data via the Django Admin interface:
1. Start the server: `python manage.py runserver`
2. Open `http://127.0.0.1:8000/admin/`
3. Login with `admin` / `admin123`

### Common CLI commands (Legacy)
The old way of running commands still works if `.env` is populated, but using `python manage.py nststats` is preferred as it uses the database for parameters.

Note: On Windows/PowerShell, the above commands work the same; ensure your virtual environment is activated.

## Viewing diagrams in PyCharm

Open `docs/PROCESS.md` and enable the Markdown preview. PyCharm (2023.2+) can render Mermaid diagrams directly in the editor, so the workflow flowchart will appear inline.

## Contributing docs

We expect to add more documentation over time. Please place new docs under `docs/` and link them here. Consider adding:
- `docs/DQ.md` for deeper data-quality specs and gate criteria
- `docs/FORECASTING.md` for methodology details and formulas
- `docs/CLI.md` for exhaustive command/flag reference
