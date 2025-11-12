Data Quality (DQ) – Skaters, Goalies, and Prior Season

This document describes the Data Quality (DQ) checks and artifacts produced by the dq step. It supports skaters and goalies for the current season, and it also produces separate artifacts for the prior season when prior inputs are available.

What it does
- Fits statistical distributions to per-player metrics to sanity-check shapes and outliers.
- Produces histograms with fitted PDFs and QQ plots for normality checks (raw and log1p).
- Writes a machine-readable summary (CSV + JSON) of fit statistics for automation and tracking.

Inputs (current season)
- Skaters: data/skaters_scored.csv
- Goalies: data/merged_goalies.csv

Optional inputs (prior season)
- Skaters (prior): data/merged_skaters_prior.csv
- Goalies (prior): data/merged_goalies_prior.csv

Both inputs are expected to contain the windows and metrics described below.

Windows
- szn (season-to-date)
- l7 (last 7 games)

Notes:
- Current-season diagnostics honor whatever windows you pass (default in CLI is szn; you can add l7).
- Prior-season diagnostics default to szn only. l7 is skipped for prior unless you explicitly request it programmatically.

Metrics
- Skaters (default): G, A, PPP, SOG, FOW, HIT, BLK, PIM
- Goalies (default): GA, SV%, GAA

Columns are referenced as <window>_<metric>, e.g., szn_G, l7_SOG, szn_SV%, l7_GAA.

Segments
- Skaters are segmented by position group: F (forwards) and D (defensemen).
  - If a pos_group column exists in the input, it is used directly (normalized to uppercase); otherwise the position is derived from [Position|position|POS|pos] with robust detection for defense encodings (D, LD, RD, F/D, D/F, etc.).
- Goalies are treated as a single segment: G.

Outputs – current season (under data/dq/<timestamp>/)
- Plots organized per metric (PNG):
  - <window>_<segment>_hist.png – Histogram with fitted PDFs overlaid
  - <window>_<segment>_qq_raw.png – QQ plot of raw values vs Normal
  - <window>_<segment>_qq_log1p.png – QQ plot of log1p-transformed values vs Normal
- summary.csv – Aggregated fit stats for all analyzed metrics, windows, and segments (both skaters and goalies are combined here)
- summary.json – Same information grouped by window|segment|metric

Outputs – prior season (under data/dq/prior/<same_timestamp>/)
- Same set of plots per metric and segment as above, created under a sibling "prior" directory sharing the same timestamp as the current run.
- summary.csv – Aggregated fit stats for prior-season inputs.
- summary.json – Prior-season fit details grouped by window|segment|metric.

Advanced:
- If you call diagnostics.runner.run_dq with prior_input_csv programmatically (not via the CLI), a best_fit_comparison.csv may be written in the current-season directory comparing best-fit families between current and provided prior inputs. The standard CLI path keeps prior outputs in data/dq/prior/<timestamp>/ and does not write comparison files by default.

How to run
Run via CLI:
python -m app.cli dq --windows szn l7 --metrics G A PPP SOG FOW HIT BLK PIM --min-n 25
This will:
1) Run skater diagnostics (F/D) for the requested windows/metrics and write to data/dq/<timestamp>/.
2) Automatically append goalie diagnostics (G) for GA, SV%, GAA into the same timestamped directory and summary files.
3) If prior-season inputs exist, also run prior diagnostics and write to data/dq/prior/<same_timestamp>/:
   - Skaters prior: data/merged_skaters_prior.csv (defaults to window szn only)
   - Goalies prior: data/merged_goalies_prior.csv (defaults to window szn only)

Overrides:
- You can point the skater prior file elsewhere with --prior-csv <path>. If omitted, the CLI auto-detects data/merged_skaters_prior.csv when present.

Notes
- You can restrict windows (e.g., only szn) or metrics as desired for the current season. Prior-season defaults to szn only unless customized programmatically.
- The skater segmentation uses pos_group when available in inputs (current and prior).

Fit details
We evaluate several families and record AIC/BIC, log-likelihood, and Kolmogorov–Smirnov p-values:
- Normal
- LogNormal (for positive values)
- Gamma (for positive values)
- Normal on log1p(x)

The chosen (best) family is based on AIC.

Troubleshooting
- If a column is missing (e.g., l7_GAA), that metric/window is skipped.
- Very small sample sizes are skipped (configurable via --min-n, default 25).
- For goalies, ensure data/merged_goalies.csv exists and includes szn_ and l7_ columns for GA, SV%, and GAA.
- For prior, ensure data/merged_skaters_prior.csv and/or data/merged_goalies_prior.csv exist; outputs will appear under data/dq/prior/<timestamp>/.

See also: docs/PROCESS.md