# Merge process overview (registry removed)

This document reflects the simplified merge without the player registry. Normalization (names, teams, positions) now lives in helpers/normalization.py. For the end‑to‑end process and environment variables, see PROCESS.md. For how players are scored and how those scores flow into merge/forecasting, see SCORING.md.

See also: [PROCESS.md](../PROCESS.md), [SCORING.md](../docs/SCORING.md)

- Primary implementation: app/merger.py
- Orchestration entry points: app/orchestrator.py
- Normalization utilities: helpers/normalization.py

## What changed
- Removed the persistent player registry and the run_registry_update step.
- app/merger.py no longer imports registries.*; it uses helpers.normalization instead.
- Team/position source of truth is NST data for skaters and Yahoo API for goalies; Yahoo contributes ownership and eligible positions for all.
- Goalies now use Yahoo as their exclusive data source for both stats and identity; the merge step remains necessary to attribute fantasy team ownership (`team_name`) to these goalies.
- Output files are suffixed by league ID (e.g., `merged_skaters_105618.csv`) to support multiple leagues.
- If skater data (from NST) lists multiple teams or positions (comma‑separated due to trades/changes), the current value is taken as the last token.
- Merge keys are simplified to cname + position, with fallbacks to cname + coarse F/D/G and then cname. For Yahoo-sourced goalies, the `player_key` is used as the primary join key where available.

## Inputs
- Scored CSVs produced by the pipelines (see run_nst in app/orchestrator.py):
  - `data/skaters_scored.csv` (skaters already scored with percentile T_scores and indexes)
  - `data/goalies_scored.csv` (goalies scored on GA, SV%, GAA, W, SV, SHO with T_scores and Goalie_Index)
  - League-specific versions: `data/skaters_scored_<league_id>.csv`, `data/goalies_scored_<league_id>.csv`
  - Optionally prior‑season variants:
    - data/skaters_scored_prior.csv
    - data/goalies_scored_prior.csv
- Yahoo Roster Data:
  - Local SQLite database `data/app.db` (exclusive source for `CurrentRoster` and `Player` metadata)

## Flow
1) Load and normalize Yahoo rosters
   - Load from DB (primary source of truth).
   - Build per‑player rows with:
     - cname: normalized player key (helpers.normalize_name)
     - ypos: normalized primary position code in {C,L,R,D,G} (helpers.normalize_position)
     - fpos: coarse fallback position where C/L/R → F (helpers.to_fpos)
     - yahoo_positions: eligible Yahoo positions with UTIL removed (semicolon‑joined)
     - yahoo_name, team_name (fantasy team), player_key (Yahoo ID)

2) Load Scored CSVs and normalize
   - Skaters: input is `data/skaters_scored[_<id>].csv` (produced by PlayerIndexPipeline)
   - Goalies: input is `data/goalies_scored[_<id>].csv` (produced by GoalieIndexPipeline)
   - Compute cname from Player, pos_code from Position (forcing G for goalies), team_code from Team.
   - For skaters, if NST fields contain comma‑separated history (e.g., CHI, VAN), select the last token (VAN) as current.

3) Merge
   - If both inputs have `player_key`, join on `player_key` first.
   - Fallback join on (cname, pos_code) → (cname, ypos).
   - Further fallback on (cname, fpos).
   - Final fallback on cname only.
   - Populate `team_name` (fantasy owner) and `Elig_Pos` (Yahoo eligible positions).

## Outputs
- data/merged_skaters.csv / data/merged_skaters_<league_id>.csv
- data/merged_goalies.csv / data/merged_goalies_<league_id>.csv
- If prior season files exist, prior merged outputs are produced as well.

## Notes
- helpers/normalization.py centralizes normalization for names (accents/suffixes removed), teams (Team2TM.xlsx + aliases), and positions (mapped to {C,L,R,D,G}).
- Trades/position changes: the “last token wins” rule ensures current team/position are used when input fields include history.
- **Data Integrity**: The merge process now relies exclusively on the local database for Yahoo roster information. `data/all_rosters.csv` is no longer used, simplifying the pipeline and ensuring that the most recently synced database state is always used.
- **Yahoo API Efficiency**: By using `sync-rosters` to populate the database and caching those results daily, we significantly reduce the number of direct Yahoo API calls required during subsequent `nst` or `merge` runs.
