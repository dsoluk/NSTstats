### avg-compare, fetch-daily-gp and backfill-gp — quick reference

This page summarizes three CLI processes used for league analytics and data backfilling. For broader context on the end‑to‑end workflow, see [process.md](PROCESS.md).

#### avg-compare
Purpose:
- Compute average stat comparisons for a given team vs a specific opponent and vs the league average, using data from weeks strictly prior to the provided `current-week`.
- Produces two grids (per position group: Skaters, Goalies):
  - Base totals grid (totals to-date before current week)
  - Per‑GP grid (totals normalized by GP; special handling for `GAA` and `SV%`)

Inputs (required flags):
- `--league-key` — Yahoo league key (e.g., `435.l.12345`).
- `--current-week` — the current fantasy week number; averages include weeks `< current-week`.
- `--team-id` — your team id (numeric id or full `team_key`).
- `--opp-team-id` — opponent team id (numeric id or full `team_key`).

Behavior notes:
- Excludes `SA`, `SV` from the display grids (informational only).
- Inverse xWin for goalie rate stats where lower is better: `GA`, `GAA`.
- Per‑GP specifics:
  - `GAA` computed as total `GA` divided by goalie GP.
  - `SV%` computed as total `SV` divided by total `SA`.

Output:
- A CSV named like `avg_compare_<league>_w<current>_t<team>_vs_<opp>.csv` written to the default downloads folder used by the app.
- Console prints summarize the computation and the output path.

Example:
```
python -m app.cli avg-compare \
  --league-key nhl.p.2526 \
  --current-week 9 \
  --team-id 4 \
  --opp-team-id 8
```

#### daily (Composite)
Purpose:
- Run the daily pipeline: `sync-rosters`, `all` (Yahoo, NST, Merge), and `schedule-lookup`.
- Prompts for `current-week`.

#### weekly (Composite)
Purpose:
- Finalize the prior week: `fetch-daily-gp`, `backfill-gp`, and `avg-compare`.
- Automatically processes the prior week based on the prompted `current-week`.
- Prompts for `current-week`, `team-id`, and `opp-team-id`.

#### sync-rosters
Purpose:
- Synchronize current Yahoo league rosters to the local database `CurrentRoster` table.
- This creates a local snapshot of ownership, which is used by the `merge` command and `avg-compare`.
- Reduces reliance on repetitive Yahoo API calls and `all_rosters.csv`.

Inputs:
- `--league-key` (required) — Yahoo league key (e.g., `nhl.p.2526`).

Example:
```
python -m app.cli sync-rosters --league-key nhl.p.2526
```

#### init-weeks
Purpose:
- Populates the `Week` table with start and end dates for a given league and season.
- This is a prerequisite for `fetch-daily-gp`, `backfill-gp`, and any command that needs to resolve week date windows.

Inputs:
- `--league-key` (required)
- `--season` (optional, defaults to .env)
- `--start-week` / `--end-week` (optional defaults)

Example:
```
python -m app.cli init-weeks --league-key nhl.p.2526 --season 2025
```

#### backfill-gp
Purpose:
- Aggregate per‑date roster appearances (`RosterSlotDaily.gp`) into weekly totals per player (`WeeklyPlayerGP`) for a week range, optionally marking weeks closed.
- This is a DB‑only operation; it does not call Yahoo. Use `fetch-daily-gp` beforehand to populate daily rows if needed.

Inputs:
- `--league-key` (required) — Yahoo league key (e.g., `465.l.12345`).
- `--start-week` (default: 1) — first week to process, inclusive.
- `--end-week` (default: 8) — last week to process, inclusive.
- `--team-keys` (optional) — comma‑separated list of `team_key`s to restrict scope.
- `--batch-size` (default: 3) — teams processed per batch before pausing.
- `--sleep-sec` (default: 2.0) — seconds to sleep between batches.
- `--force` — overwrite existing `WeeklyPlayerGP` rows (otherwise existing rows are skipped).
- `--no-close` — do not mark processed weeks as closed.
- `--dry-run` — print planned upserts without writing to the DB.

Behavior summary:
- For each week in range, sums `RosterSlotDaily.gp` by `player_key` within the week’s date window for each team in scope.
- Writes/updates `WeeklyPlayerGP` rows per player/week/team; commits per team.
- Respects `--force` (overwrite vs skip existing) and `--dry-run` (no writes).
- Unless `--no-close` or `--dry-run` is set, marks each processed week closed.

Example:
```
python -m app.cli backfill-gp \
  --league-key nhl.p.2526 \
  --start-week 1 \
  --end-week 8 \
  --batch-size 4 \
  --sleep-sec 1.5 \
  --team-keys nhl.p.2526.t.1,nhl.p.2526.t.4
```

#### fetch-daily-gp
Purpose:
- Populate `RosterSlotDaily` by fetching team rosters and per‑date player stats from Yahoo for a given week range (rate‑limit friendly).
- Typically run before `backfill-gp` so weekly GP can be aggregated from daily rows.

Inputs:
- `--league-key` (required) — Yahoo league key (e.g., `nhl.p.2526`).
- `--start-week` (required) — first week to fetch, inclusive.
- `--end-week` (required) — last week to fetch, inclusive.
- `--team-keys` (optional) — comma‑separated list of `team_key`s to restrict scope.
- `--sleep-sec` (default: 1.0) — seconds to sleep between team calls.
- `--dry-run` — print planned actions; do not write to DB.

Behavior summary:
- Loads Yahoo credentials from environment (`.env` supported) and initializes OAuth.
- Resolves teams from the DB (filtered by `--team-keys` when provided).
- Resolves week date windows from the DB; warns and skips weeks missing dates.
- For each team and date in the selected weeks, fetches roster and per‑date player stats from Yahoo and upserts into `RosterSlotDaily` (and ensures `Team` rows exist).
- Respects `--sleep-sec` between team calls and `--dry-run` for no‑write mode.

Example:
```
python -m app.cli fetch-daily-gp \
  --league-key nhl.p.2526 \
  --start-week 1 \
  --end-week 8 \
  --sleep-sec 1.0 \
  --team-keys nhl.p.2526.t.1,nhl.p.2526.t.4
```

See also: [process.md](PROCESS.md)
