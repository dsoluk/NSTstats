import argparse
import os
import sys
from app.orchestrator import run_all, run_yahoo, run_nst, run_merge
from diagnostics.runner import run_dq as run_dq_diag, run_dq_prior as run_dq_prior_diag
from application.forecast import forecast as run_forecast
from application.compare import compare as run_compare
from application.analyze import evaluate as run_analyze

# If you store canonical inputs in known NSTstats paths, set sensible defaults here
DEFAULT_OUT_CSV = "data/lookup_table.csv"


def cmd_schedule_refresh(argv=None):
    p = argparse.ArgumentParser(
        prog="nststats schedule-lookup",
        description="Build NHL schedule lookup using NHLschedule defaults; output saved to data/lookup_table.csv"
    )
    p.add_argument("--out-csv", default=DEFAULT_OUT_CSV,
                   help="Output CSV path (default: data/lookup_table.csv)")
    p.add_argument("--refresh-cache", action="store_true",
                   help="Bypass cache and refetch all team tables (proxied to nhl_schedule)")
    p.add_argument(
        "--nhl-schedule-path",
        dest="nhl_schedule_path",
        default=None,
        help=(
            "Optional path to your local NHLschedule project. If provided (or if the NHL_SCHEDULE_PATH env var is set),\n"
            "we will import nhl_schedule from there so recent local edits are used without reinstalling."
        ),
    )

    args = p.parse_args(argv)

    # Ensure output directory exists
    out_csv = args.out_csv or DEFAULT_OUT_CSV
    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # If user passed a local path (or env var is set), insert it into sys.path for this invocation
    nhl_schedule_path = args.nhl_schedule_path or os.environ.get("NHL_SCHEDULE_PATH")
    if nhl_schedule_path and os.path.isdir(nhl_schedule_path):
        if nhl_schedule_path not in sys.path:
            sys.path.insert(0, nhl_schedule_path)

    # Import here so that sys.path injection (if any) takes effect
    try:
        from nhl_schedule.build_lookup import build
    except Exception as e:
        raise RuntimeError(
            "Failed to import nhl_schedule. If you are developing NHLschedule in a separate project, "
            "either install it (pip install -e <path>) or pass --nhl-schedule-path / set NHL_SCHEDULE_PATH."
        ) from e

    # Use NHLschedule defaults for schedule path and sheet/table by passing None
    out_path = build(
        schedule_path=None,
        sheet_or_table=None,
        out_csv=out_csv,
        out_xlsx=None,
        refresh_cache=args.refresh_cache,
    )
    print(f"Lookup table written to: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="NSTstats CLI")
    # Top-level flag alias so users can run: python -m app.cli --schedule-lookup
    parser.add_argument("--schedule-lookup", dest="schedule_lookup", action="store_true", help="Run schedule lookup (alias for the 'schedule-lookup' subcommand)")

    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("all", help="Run Yahoo and NST pipelines")
    subparsers.add_parser("yahoo", help="Run Yahoo Fantasy helper")
    subparsers.add_parser("nst", help="Run NST pipelines (skaters, goalies)")
    subparsers.add_parser("merge", help="Merge NST with Yahoo ownership to CSVs")

    # Schedule lookup command (nhl_schedule integration)
    sched = subparsers.add_parser("schedule-lookup", help="Build or update the weekly schedule lookup table from inputs")
    # sched.add_argument("--matchups", default=DEFAULT_MATCHUPS, help="Path to matchups parquet/csv with team vs opponent per date")
    # sched.add_argument("--opp-ease", dest="opp_ease", default=DEFAULT_OPP_EASE, help="Path to opponent ease parquet/csv with OppDefenseScore0to100 by team")
    # sched.add_argument("--out-csv", dest="out_csv", default=DEFAULT_OUT_CSV, help="Output CSV path for the lookup table")
    # sched.add_argument("--out-xlsx", dest="out_xlsx", default=DEFAULT_OUT_XLSX, help="Optional Excel output path")

    # Diagnostics command
    dq = subparsers.add_parser("dq", help="Run distribution diagnostics on skaters and goalies")
    dq.add_argument("--windows", nargs="*", default=["szn"], help="Windows to analyze (e.g., szn l7)")
    dq.add_argument("--metrics", nargs="*", default=["G","A","PPP","SOG","FOW","HIT","BLK","PIM"], help="Metrics to analyze")
    dq.add_argument("--min-n", type=int, default=25, help="Minimum sample size per group")
    dq.add_argument("--out", default=None, help="Output directory root (default data/dq)")
    dq.add_argument("--prior-csv", dest="prior_csv", default=None, help="Optional prior-season CSV to compare (e.g., data/merged_skaters_prior.csv)")

    # Forecast command
    fc = subparsers.add_parser("forecast", help="Compute per-player stat forecasts for ROW, Next Week, and ROS")
    fc.add_argument("--current-week", dest="current_week", type=int, required=True, help="Current fantasy week number (1..24)")
    fc.add_argument("--season-weight", dest="season_weight", type=float, default=0.5, help="Weight for season-to-date rates (manual override; ignored if auto-weights is enabled)")
    fc.add_argument("--last7-weight", dest="last7_weight", type=float, default=0.2, help="Weight for last-7 games rates (default 0.2; held constant when auto-weights is enabled)")
    fc.add_argument("--last-year-weight", dest="last_year_weight", type=float, default=0.3, help="Weight for last-year per-60 based per-game expectation (manual override; ignored if auto-weights is enabled)")
    fc.add_argument("--sos-weight", dest="sos_weight", type=float, default=0.3, help="Schedule strength impact weight; positive means hard schedule lowers output (default 0.3)")
    fc.add_argument("--horizons", nargs="*", default=["row","next","ros"], help="Horizons to compute: row next ros")
    fc.add_argument("--skaters-csv", dest="skaters_csv", default=os.path.join("data","merged_skaters.csv"), help="Input merged skaters CSV (default data/merged_skaters.csv)")
    fc.add_argument("--lookup-csv", dest="lookup_csv", default=os.path.join("data","lookup_table.csv"), help="Schedule lookup CSV (default data/lookup_table.csv)")
    fc.add_argument("--out-csv", dest="out_csv", default=os.path.join("data","forecasts.csv"), help="Output CSV path (default data/forecasts.csv)")
    fc.add_argument("--auto-weights", dest="auto_weights", action="store_true", default=True, help="Enable sliding-scale weights between last-year and season-to-date (default on)")
    fc.add_argument("--no-auto-weights", dest="auto_weights", action="store_false", help="Disable sliding-scale weights; use manual season/last7/last-year weights")
    fc.add_argument("--weeks-in-season", dest="weeks_in_season", type=int, default=24, help="Total number of fantasy weeks (default 24)")

    # Compare command
    cp = subparsers.add_parser("compare", help="Compare our forecasts vs NatePts projections and Last-Year benchmarks")
    cp.add_argument("--current-week", dest="current_week", type=int, required=True, help="Current fantasy week number (1..24)")
    cp.add_argument("--forecast-csv", dest="forecast_csv", default=os.path.join("data","forecasts.csv"), help="Forecast CSV input (default data/forecasts.csv)")
    cp.add_argument("--lookup-csv", dest="lookup_csv", default=os.path.join("data","lookup_table.csv"), help="Schedule lookup CSV (default data/lookup_table.csv)")
    cp.add_argument("--proj-xlsx", dest="proj_xlsx", default=os.path.join("NatePts.xlsx"), help="Path to NatePts.xlsx (default NatePts.xlsx in project root)")
    cp.add_argument("--proj-sheet", dest="proj_sheet", default="NatePts", help="Sheet/Table name in NatePts.xlsx (default NatePts)")
    cp.add_argument("--ly-sit-s", dest="ly_sit_s_csv", default=None, help="Optional CSV of NST Last-Year all-situations per-60 (rate=y)")
    cp.add_argument("--ly-sit-pp", dest="ly_sit_pp_csv", default=None, help="Optional CSV of NST Last-Year power-play per-60 (rate=y)")
    cp.add_argument("--horizons", nargs="*", default=["row","next","ros"], help="Horizons to compute: row next ros")
    cp.add_argument("--all-players", dest="all_players", action="store_true", help="Include all players (default: owned only)")
    cp.add_argument("--out-csv", dest="out_csv", default=os.path.join("data","compare.csv"), help="Output CSV path (default data/compare.csv)")

    # Analyze command
    az = subparsers.add_parser("analyze", help="Evaluate season-total forecasts vs projections and export metrics/Excel")
    az.add_argument("--compare-csv", dest="compare_csv", default=os.path.join("data","compare.csv"), help="Input compare CSV (default data/compare.csv)")
    az.add_argument("--out-dir", dest="out_dir", default=os.path.join("data","eval"), help="Output directory for metrics/Excel (default data/eval)")

    args = parser.parse_args()

    # Determine command, honoring the top-level flag alias if provided
    if getattr(args, "schedule_lookup", False) and not getattr(args, "command", None):
        cmd = "schedule-lookup"
    else:
        cmd = args.command or "all"

    if cmd == "yahoo":
        run_yahoo()
    elif cmd == "nst":
        run_nst()
    elif cmd == "merge":
        run_merge()
    elif cmd == "dq":
        windows = getattr(args, "windows", ["szn"]) or ["szn"]
        metrics = getattr(args, "metrics", ["G","A","PPP","SOG","FOW","HIT","BLK","PIM"]) or ["G","A","PPP","SOG","FOW","HIT","BLK","PIM"]
        min_n = getattr(args, "min_n", 25)
        out = getattr(args, "out", None)
        out_dir = out if out else None
        prior_csv = getattr(args, "prior_csv", None)
        # 1) Skaters (F/D)
        out_path = run_dq_diag(
            input_csv=os.path.join("data", "skaters_scored.csv"),
            out_dir=out_dir or os.path.join("data", "dq"),
            metrics=metrics,
            windows=windows,
            min_n=min_n,
            prior_input_csv=None,  # keep current season separate from prior outputs
            group_by_position=True,
        )
        # 2) Goalies (G) — append to same summary
        try:
            g_metrics = ["GA", "SV%", "GAA"]
            g_csv = os.path.join("data", "merged_goalies.csv")
            if os.path.exists(g_csv):
                run_dq_diag(
                    input_csv=g_csv,
                    out_dir=out_dir or os.path.join("data", "dq"),
                    metrics=g_metrics,
                    windows=windows,
                    min_n=min_n,
                    prior_input_csv=None,
                    group_by_position=False,
                    segments=["G"],
                    existing_root=out_path,
                )
        except Exception as _e:
            print(f"[Warn] Goalie diagnostics skipped due to error: {_e}")
        # 3) Prior season runs in a separate dq/prior/<ts> directory
        try:
            ts = os.path.basename(out_path.rstrip(os.sep))
            # Prior Skaters
            prior_skaters_path = prior_csv or os.path.join("data", "merged_skaters_prior.csv")
            if os.path.exists(prior_skaters_path):
                run_dq_prior_diag(
                    input_csv=prior_skaters_path,
                    out_dir=os.path.join("data", "dq", "prior"),
                    metrics=metrics,
                    windows=None,  # default to ["szn"] for prior
                    min_n=min_n,
                    group_by_position=True,
                    segments=None,
                    existing_ts=ts,
                )
            # Prior Goalies
            prior_goalies_path = os.path.join("data", "merged_goalies_prior.csv")
            if os.path.exists(prior_goalies_path):
                run_dq_prior_diag(
                    input_csv=prior_goalies_path,
                    out_dir=os.path.join("data", "dq", "prior"),
                    metrics=g_metrics,
                    windows=None,  # default to ["szn"] for prior
                    min_n=min_n,
                    group_by_position=False,
                    segments=["G"],
                    existing_ts=ts,
                )
        except Exception as _e:
            print(f"[Warn] Prior-season diagnostics skipped due to error: {_e}")
        print(f"Diagnostics written to: {out_path}")
    elif cmd == "forecast":
        current_week = getattr(args, "current_week")
        season_weight = getattr(args, "season_weight", 0.8)
        last7_weight = getattr(args, "last7_weight", 0.2)
        sos_weight = getattr(args, "sos_weight", 0.3)
        horizons = tuple(getattr(args, "horizons", ["row","next","ros"]))
        skaters_csv = getattr(args, "skaters_csv", os.path.join("data","merged_skaters.csv"))
        lookup_csv = getattr(args, "lookup_csv", os.path.join("data","lookup_table.csv"))
        out_csv = getattr(args, "out_csv", os.path.join("data","forecasts.csv"))
        out_path = run_forecast(
            skaters_csv=skaters_csv,
            lookup_csv=lookup_csv,
            out_csv=out_csv,
            current_week=current_week,
            season_weight=season_weight,
            last7_weight=last7_weight,
            last_year_weight=getattr(args, "last_year_weight", 0.0),
            sos_weight=sos_weight,
            horizons=horizons,
            auto_weights=getattr(args, "auto_weights", True),
            weeks_in_season=getattr(args, "weeks_in_season", 24),
        )
        print(f"Forecasts written to: {out_path}")
    elif cmd == "schedule-lookup":
        # Pass an empty argv so the inner parser uses defaults and doesn't try to parse the outer Namespace
        cmd_schedule_refresh(argv=[])
    elif cmd == "compare":
        current_week = getattr(args, "current_week")
        forecast_csv = getattr(args, "forecast_csv", os.path.join("data","forecasts.csv"))
        lookup_csv = getattr(args, "lookup_csv", os.path.join("data","lookup_table.csv"))
        proj_xlsx = getattr(args, "proj_xlsx", os.path.join("NatePts.xlsx"))
        proj_sheet = getattr(args, "proj_sheet", "NatePts")
        ly_sit_s_csv = getattr(args, "ly_sit_s_csv", None)
        ly_sit_pp_csv = getattr(args, "ly_sit_pp_csv", None)
        horizons = tuple(getattr(args, "horizons", ["row","next","ros"]))
        out_csv = getattr(args, "out_csv", os.path.join("data","compare.csv"))
        all_players = bool(getattr(args, "all_players", False))
        out_path = run_compare(
            current_week=current_week,
            forecast_csv=forecast_csv,
            lookup_csv=lookup_csv,
            proj_xlsx=proj_xlsx,
            proj_sheet=proj_sheet,
            ly_sit_s_csv=ly_sit_s_csv,
            ly_sit_pp_csv=ly_sit_pp_csv,
            horizons=horizons,
            out_csv=out_csv,
            all_players=all_players,
        )
        print(f"Comparison written to: {out_path}")
    elif cmd == "analyze":
        compare_csv = getattr(args, "compare_csv", os.path.join("data", "compare.csv"))
        out_dir = getattr(args, "out_dir", os.path.join("data", "eval"))
        out_path = run_analyze(compare_csv=compare_csv, out_dir=out_dir)
        print(f"Season total evaluation written to: {out_path}\nPlots and CSVs in: {out_dir}")
    else:
        run_all()


if __name__ == "__main__":
    main()
