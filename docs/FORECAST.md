# Forecast Step

This page documents the Forecast stage, which produces per-player stat forecasts across requested horizons (ROW, Next Week, ROS).

Summary
- Inputs:
  - data/merged_skaters.csv (merged NST per-60 rates and Yahoo ownership)
  - data/merged_skaters_prior.csv (PRIOR season merged NST per-60; auto-used to supply ly_* by default when present)
  - data/lookup_table.csv (team schedule with Games, GamesRestOfWeek, GamesROS, and SOS)
- Method:
  - Convert per-60 rates to per-game expectations using TOI/GP
    - PPP uses PP TOI when available, otherwise falls back to all-situations TOI
    - Other stats use all-situations TOI
  - Blend Season-to-date, Last-7, and Last-Year components:
    - Last-7 default weight is held at 0.2 (configurable)
    - Sliding-scale (auto-weights) splits the remaining mass (1 - last7_weight) between Season and Last-Year based on week progression across the season (default 24 weeks):
      - week_index = clamp((current_week - 1) / (weeks_in_season - 1), 0..1)
      - remaining = 1 - last7_weight
      - last_year_weight = remaining * (1 - week_index)
      - season_weight   = remaining * week_index
      - At week 1: season≈0.0, last_year≈0.8; mid-season: ≈0.4/0.4; final week: season≈0.8, last_year≈0.0
    - Prior-season ly_* fields are auto-populated from data/merged_skaters_prior.csv when present
    - xPG = weighted mean of components present (weights normalized over available components)
    - Missing components are ignored; if all components are missing, xPG falls back to 0.0
  - Scale per-game expectations by horizon games (ROW, NW, ROS)
  - Apply schedule-strength multiplier: multiplier = 1 + sos_weight * (0.5 - SOS)
- Outputs:
  - data/forecasts.csv with:
    - Identity: Player, Team, Position, team_name
    - Per-game expectations for transparency: xPG_<STAT>
    - Horizon totals: ROW_<STAT>, NW_<STAT>, ROS_<STAT>

CLI
python -m app.cli forecast --current-week <N> \
  --season-weight 0.8 --last7-weight 0.2 \
  [--last-year-weight 0.3] \
  [--sos-weight 0.3] \
  [--horizons row next ros] \
  [--skaters-csv data/merged_skaters.csv] \
  [--lookup-csv data/lookup_table.csv] \
  --out-csv data/forecasts.csv

Stats
- G, A, PPP, SOG, FOW, HIT, BLK, PIM

Details
- Last-Year blending in Forecast:
  - The forecast step optionally uses last-year columns when present in merged_skaters.csv or auto-merged from data/merged_skaters_prior.csv:
    - ly_<STAT> per-60 rates
    - ly_TOI/GP_all and ly_TOI/GP_pp for TOI
  - If these columns are absent or NaN, last-year contributes 0 weight to the blend. Keep --last-year-weight at 0.0 unless ly_* columns exist.
- Prior-season per-82 baseline in Analyze (last_year column):
  - The Analyze step now computes a prior-season baseline assuming an 82-game season for each player and stat:
    - LY82_<STAT> = (szn_<STAT> per-60) × (TOI/GP for that situation) × 82 / 60
    - PPP uses PP TOI/GP; all other stats use all-situations TOI/GP
  - These LY82 totals are merged by player name and appear in season_total_eval.xlsx as the "last_year" column next to forecast and projection.
  - For Points/Bangers/Overall sheets, LY82 group totals are computed using the same weights as Overall.
  - All three values (forecast, projection, last_year) are rounded up to integers in the Excel workbook for readability.
- Robustness to missing inputs:
  - TOI parsing accepts formats like "0:20:12" or "0 days 00:20:53.2" and numeric minutes
  - PPP TOI falls back to all-situations TOI when PP TOI is missing
  - Rate and TOI values are coerced to numeric; xPG falls back to 0.0 if all components are unavailable

Troubleshooting
- forecasts.csv has all nulls or zeros:
  - Ensure data/merged_skaters.csv contains columns: szn_<STAT>, l7_<STAT>, szn_TOI/GP_all, l7_TOI/GP_all; for PPP also szn_TOI/GP_pp and l7_TOI/GP_pp
  - Ensure data/lookup_table.csv has TM, Week, Games, GamesRestOfWeek, GamesROS, SOS
  - Verify you passed --current-week matching the lookup table weeks
  - If using --last-year-weight, confirm ly_<STAT> and ly_TOI/GP_* exist; otherwise set it to 0.0
- No rows in forecasts.csv:
  - The forecast step filters to owned players (team_name not empty). Re-run merge or pass an input that includes owned players.

Notes
- Schedule-strength impact is modest by default (sos_weight=0.3). Tune per league preferences.
- The per-game columns (xPG_*) are provided to support audits and calibration.
