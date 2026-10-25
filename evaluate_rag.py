"""Run repeatable offline (default) or explicitly confirmed live RAG evaluation."""
from __future__ import annotations

import argparse
from pathlib import Path

from app.evaluation.dataset import load_dataset
from app.evaluation.runner import run_evaluation, save_report


ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("offline", "live"), default="offline")
    parser.add_argument("--dataset", default=str(ROOT / "evals" / "dataset.json"))
    parser.add_argument("--output-dir", default=str(ROOT / "evals" / "reports"))
    parser.add_argument("--confirm-live", action="store_true", help="Required to call configured providers and read configured Pinecone index")
    args = parser.parse_args()
    if args.mode == "live" and not args.confirm_live:
        parser.error("--mode live requires --confirm-live; live evaluation calls configured providers")
    dataset = load_dataset(args.dataset)
    report = run_evaluation(dataset, mode=args.mode)
    path = save_report(report, args.output_dir)
    print(f"Mode: {args.mode} | cases: {len(report['cases'])} | report: {path}")
    print("Summary:")
    for key, value in report["summary"].items():
        print(f"  {key}: {'n/a' if value is None else f'{value:.3f}'}")
    failures = report["failed_case_ids"]
    print("Failed cases: " + (", ".join(failures) if failures else "none"))
    if args.mode == "offline":
        print("Offline fixtures validate evaluator plumbing; they do not measure live model quality.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
