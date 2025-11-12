# Compare Step

This page documents the Compare stage, which aligns our forecasts with an external projection source and optional last-year benchmarks, and produces a unified CSV used by Analyze.

Summary
- Input forecasts (application/forecast.py output)
- External projections (e.g., NatePts.xlsx)
- Optional last-year per-60 inputs (NST rate=y) for all-situations and PP
- Team schedule lookup for horizon games
- Output: data/compare.csv with aligned horizons (ROW, NW, ROS) for Forecast, Proj, and LY plus deltas

Command
python -m app.cli compare --current-week <N> \
  --forecast-csv data/forecasts.csv \
  --lookup-csv data/lookup_table.csv \
  --proj-xlsx NatePts.xlsx --proj-sheet NatePts \
  [--ly-sit-s <csv>] [--ly-sit-pp <csv>] \
  [--horizons row next ros] \
  [--all-players] \
  --out-csv data/compare.csv

Key Columns
- Player, Team, Position, team_name
- Forecast horizons: ROW_<STAT>, NW_<STAT>, ROS_<STAT>
- Projection horizons: Proj_ROW_<STAT>, Proj_NW_<STAT>, Proj_ROS_<STAT>
- Last-year horizons (when LY inputs provided): LY_ROW_<STAT>, LY_NW_<STAT>, LY_ROS_<STAT>
- Deltas: Δ_Fcst_vs_Proj_<H>_<STAT>, Δ_Fcst_vs_LY_<H>_<STAT>

Stats
- G, A, PPP, SOG, FOW, HIT, BLK, PIM

Notes
- Horizons default to row next ros; current_week used to map schedules
- Compare uses per-game values from projections (82-GP normalized) and LY per-60+TOI to scale into horizons
- Unmatched players from projections/LY are saved to data/eval/unmatched_in_compare.csv for debugging
