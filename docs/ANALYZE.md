# Analyze Step

This page documents the Analyze stage, which evaluates season-long totals from our forecasts against projections (and supports stat-by-stat analysis).

Summary
- Input: data/compare.csv (from the Compare step)
- Build season totals (ROW + NW + ROS) for each stat for Forecast and Proj, plus Last-Year if present
- Compute group scores (Points, Bangers, Overall) and per-GP counterparts
- Generate metrics overall, by position group (F/D), and by forecast deciles
- Export plots and an Excel workbook for pivoting

Command
python -m app.cli analyze \
  --compare-csv data/compare.csv \
  --out-dir data/eval

Outputs
- data/eval/metrics_summary.csv
- data/eval/metrics_by_position.csv
- data/eval/metrics_by_decile.csv
- data/eval/_plots/* (Overall and STAT_<STAT> scatter + residuals-by-decile)
- data/eval/season_total_eval.xlsx containing:
  - Overall, Points, Bangers pair sheets (forecast vs projection)
  - STAT_<STAT> sheets with a new 'stat' column to identify the stat being evaluated
  - STAT_All sheet consolidating all STAT_* with explicit 'stat' column for easy filtering/pivots
  - metrics_overall, metrics_by_position, metrics_by_decile sheets

Notes
- Column headings are written to all CSVs and Excel sheets (index=False) for clarity
- The 'stat' column enables stat-by-stat pivots and filters across all players and sheets
- Deciles are defined by the Overall forecast and joined to stat-level sheets for residual plots
