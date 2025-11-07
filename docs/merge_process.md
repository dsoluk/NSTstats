# Merge process overview (registry removed)

This document reflects the simplified merge without the player registry. Normalization (names, teams, positions) now lives in helpers/normalization.py. For the end‑to‑end process and environment variables, see PROCESS.md.

See also: [PROCESS.md](../PROCESS.md)

- Primary implementation: app/merger.py
- Orchestration entry points: app/orchestrator.py
- Normalization utilities: helpers/normalization.py

## What changed
- Removed the persistent player registry and the run_registry_update step.
- app/merger.py no longer imports registries.*; it uses helpers.normalization instead.
- Team/position source of truth is NST data; Yahoo contributes ownership and eligible positions.
- If NST lists multiple teams or positions (comma‑separated due to trades/changes), the current value is taken as the last token.
- Merge keys are simplified to cname + position, with fallbacks to cname + coarse F/D/G and then cname.

## Inputs
- NST CSVs produced by the pipelines (see run_nst in app/orchestrator.py):
  - data/skaters.csv (used to compute data/skaters_scored.csv)
  - data/goalies.csv
  - Optionally prior‑season variants (data/skaters_prior.csv, data/goalies_prior.csv) and scored skaters (data/skaters_scored_prior.csv)
- Yahoo All Rosters export:
  - data/all_rosters.csv

## Flow
1) Load and normalize Yahoo rosters
   - Build per‑player rows with:
     - cname: normalized player key (helpers.normalize_name)
     - ypos: normalized primary position code in {C,L,R,D,G} (helpers.normalize_position)
     - fpos: coarse fallback position where C/L/R → F (helpers.to_fpos)
     - yahoo_positions: eligible Yahoo positions with UTIL removed (semicolon‑joined)
     - yahoo_name, team_name (fantasy team)

2) Load NST CSVs and normalize
   - Skaters: input is data/skaters_scored.csv (produced by PlayerIndexPipeline); Goalies: input is data/goalies.csv.
   - Compute cname from Player, pos_code from Position (forcing G for goalies), team_code from Team.
   - If NST fields contain comma‑separated history (e.g., CHI, VAN), select the last token (VAN) as current.

3) Merge
   - Join on (cname, pos_code) → (cname, ypos).
   - Fallback on (cname, fpos).
   - Final fallback on cname only.
   - Populate Elig_Pos from Yahoo eligible positions.

## Outputs
- data/merged_skaters.csv (from skaters_scored.csv)
- data/merged_goalies.csv (from goalies.csv)
- If prior season files exist (skaters_scored_prior.csv, goalies_prior.csv), prior merged outputs are produced as well.

## Notes
- helpers/normalization.py centralizes normalization for names (accents/suffixes removed), teams (Team2TM.xlsx + aliases), and positions (mapped to {C,L,R,D,G}).
- Trades/position changes: the “last token wins” rule ensures current team/position are used when input fields include history.
