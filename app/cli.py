import argparse
from app.orchestrator import run_all, run_yahoo, run_registry_update, run_nst, run_merge

def main():
    parser = argparse.ArgumentParser(description="NSTstats CLI")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("all", help="Run Yahoo, registry update, and NST pipelines")
    subparsers.add_parser("yahoo", help="Run Yahoo Fantasy helper")
    subparsers.add_parser("registry", help="Update registry from NST player list")
    subparsers.add_parser("nst", help="Run NST pipelines (skaters, goalies)")
    subparsers.add_parser("merge", help="Merge NST+Registry with Yahoo ownership to CSVs")

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
    else:
        run_all()

if __name__ == "__main__":
    main()
