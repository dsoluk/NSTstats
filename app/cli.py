import argparse
import os
from nhl_schedule.build_lookup import build
from app.orchestrator import run_all, run_yahoo, run_registry_update, run_nst, run_merge
from diagnostics.runner import run_dq as run_dq_diag

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

    args = p.parse_args(argv)

    # Ensure output directory exists
    out_csv = args.out_csv or DEFAULT_OUT_CSV
    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

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

    subparsers.add_parser("all", help="Run Yahoo, registry update, and NST pipelines")
    subparsers.add_parser("yahoo", help="Run Yahoo Fantasy helper")
    subparsers.add_parser("registry", help="Update registry from NST player list")
    subparsers.add_parser("nst", help="Run NST pipelines (skaters, goalies)")
    subparsers.add_parser("merge", help="Merge NST+Registry with Yahoo ownership to CSVs")

    # Schedule lookup command (nhl_schedule integration)
    sched = subparsers.add_parser("schedule-lookup", help="Build or update the weekly schedule lookup table from inputs")
    # sched.add_argument("--matchups", default=DEFAULT_MATCHUPS, help="Path to matchups parquet/csv with team vs opponent per date")
    # sched.add_argument("--opp-ease", dest="opp_ease", default=DEFAULT_OPP_EASE, help="Path to opponent ease parquet/csv with OppDefenseScore0to100 by team")
    # sched.add_argument("--out-csv", dest="out_csv", default=DEFAULT_OUT_CSV, help="Output CSV path for the lookup table")
    # sched.add_argument("--out-xlsx", dest="out_xlsx", default=DEFAULT_OUT_XLSX, help="Optional Excel output path")

    # Diagnostics command
    dq = subparsers.add_parser("dq", help="Run distribution diagnostics on skaters_scored.csv")
    dq.add_argument("--windows", nargs="*", default=["szn"], help="Windows to analyze (e.g., szn l7)")
    dq.add_argument("--metrics", nargs="*", default=["G","A","PPP","SOG","FOW","HIT","BLK","PIM"], help="Metrics to analyze")
    dq.add_argument("--min-n", type=int, default=25, help="Minimum sample size per group")
    dq.add_argument("--out", default=None, help="Output directory root (default data/dq)")

    args = parser.parse_args()

    # Determine command, honoring the top-level flag alias if provided
    if getattr(args, "schedule_lookup", False) and not getattr(args, "command", None):
        cmd = "schedule-lookup"
    else:
        cmd = args.command or "all"

    if cmd == "yahoo":
        run_yahoo()
    elif cmd == "registry":
        run_registry_update()
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
        out_path = run_dq_diag(
            input_csv=os.path.join("data", "skaters_scored.csv"),
            out_dir=out_dir or os.path.join("data", "dq"),
            metrics=metrics,
            windows=windows,
            min_n=min_n,
        )
        print(f"Diagnostics written to: {out_path}")
    elif cmd == "schedule-lookup":
        # Pass an empty argv so the inner parser uses defaults and doesn't try to parse the outer Namespace
        cmd_schedule_refresh(argv=[])
    else:
        run_all()


if __name__ == "__main__":
    main()
