# Scoring and Indexing Details

This section describes how skaters and goalies are scored and how these scores feed forecasts. For the full workflow, see [PROCESS.md](./PROCESS.md).

## Skaters: Percentile T_ scores by window and segment
- Windows: season‑to‑date (Szn) and last 7 games (L7).
- Segments: Forwards (F) vs Defense (D), derived from `Position`.
- For each metric (e.g., `G, A, PPP, SOG, FOW, HIT, BLK, PIM`) and window, the scoring pipeline:
  1. Selects a distribution family per segment (from `data/dq/.../best_fit_comparison.csv` if available; else defaults such as `Normal(log1p)`).
  2. Computes the percentile `p = F(x)` within the segment using fitted parameters.
  3. Converts to an integer `T_` score: `T = ceil(99 * p)`, using `50` for missing values.
- Output columns include `T_szn_G`, `T_l7_SOG`, etc.

Implementation: `pipelines.py` → `PlayerIndexPipeline` (`_load_best_fit_mapping`, `_percentile_from_family`, `_score_cols_for_window`, `calculate_indexes_per_window`).

## Skater Indexes per window
- Offensive Index: weighted average of relevant `T_` scores (default uniform weights unless configured).
- Banger Index: weighted average of physical stats `T_` scores.
- Composite Index: weighted combination (default `0.7*Offensive + 0.3*Banger`).
- Output: `Offensive_Index_szn`, `Banger_Index_szn`, `Composite_Index_szn` (and `_l7`).

## Goalies
- Metrics include `GA, SV%, GAA`. The pipeline computes percentile `T_` scores per window and a `Goalie_Index_<window>`.
- Implementation: `pipelines.py` → `GoalieIndexPipeline`.

## Prior Season
- When prior‑season files exist, current season and prior season are processed separately and merged outputs are available (e.g., `merged_skaters_prior.csv`). These enable prior‑season (`ly_*`) features during forecasting.

## Forecasting blend (where prior/current are combined)
- The forecasting module blends Szn, L7, and prior season (LY) to produce per‑game expectations by stat, using PP TOI for PPP and all‑situations TOI otherwise.
- With `auto_weights=True`, weights slide over a `weeks_in_season` timeline (default 24), gradually shifting emphasis from LY to Szn while holding L7 fixed.
- Schedule strength (SOS) provides per‑horizon multipliers using `data/lookup_table.csv`.
- Implementation: `application/forecast.py` (`_expected_per_game`, `_lookup_games_and_sos`, `_sos_multiplier`, `forecast`).

## Legacy/simple pipeline
- `domain/metrics.py` contains an older/simple z‑score‑style pipeline that may be useful for experiments but is superseded by the percentile‑based approach above.

## Suggested improvements
- Robust dispersion (median/MAD) when fitting families to reduce outlier sensitivity.
- Empirical Bayes shrinkage for low‑TOI or small‑sample windows.
- Optional Gamma/log‑Gamma family for strictly positive skewed rates.
- Finer position segmentation (e.g., split F into C/LW/RW where sample sizes allow).
- Cross‑validated per‑stat weights for Offensive/Banger indexes and position‑specific composite weights.
- Goalie enhancements (xGA/GSAx integration, shot‑volume normalization).
- Forecast calibration with backtests and prediction intervals; per‑metric SOS weights.
- Pre‑forecast DQ gate and better identity via stable player IDs (e.g., NHL IDs).
