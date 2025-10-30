import argparse
import os
from app.orchestrator import run_all, run_yahoo, run_registry_update, run_nst, run_merge
from diagnostics.runner import run_dq as run_dq_diag

def main():
    parser = argparse.ArgumentParser(description="NSTstats CLI")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("all", help="Run Yahoo, registry update, and NST pipelines")
    subparsers.add_parser("yahoo", help="Run Yahoo Fantasy helper")
    subparsers.add_parser("registry", help="Update registry from NST player list")
    subparsers.add_parser("nst", help="Run NST pipelines (skaters, goalies)")
    subparsers.add_parser("merge", help="Merge NST+Registry with Yahoo ownership to CSVs")

    # Diagnostics command
    dq = subparsers.add_parser("dq", help="Run distribution diagnostics on skaters_scored.csv")
    dq.add_argument("--windows", nargs="*", default=["szn"], help="Windows to analyze (e.g., szn l7)")
    dq.add_argument("--metrics", nargs="*", default=["G","A","PPP","SOG","FOW","HIT","BLK","PIM"], help="Metrics to analyze")
    dq.add_argument("--min-n", type=int, default=25, help="Minimum sample size per group")
    dq.add_argument("--out", default=None, help="Output directory root (default data/dq)")

    args = parser.parse_args()

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
    else:
        run_all()

if __name__ == "__main__":
    main()
