#!/usr/bin/env python3
"""Collect AMD distributed-inference (DI) CI data from Buildkite.

Usage:
    export BUILDKITE_TOKEN="bkua_..."
    python scripts/collect_di_ci.py --dry-run --days 30   # what builds exist?
    python scripts/collect_di_ci.py --days 30
    python scripts/collect_di_ci.py --days 30 --no-logs   # skip verdict lines

Writes under data/vllm/di/ only. Nothing here touches data/vllm/ci/, which
belongs to the nightly collector.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vllm.build_di_grid import write_grid
from vllm.ci.analyzer import compute_all_test_health
from vllm.di_collect import collect, configure, load_di_results, load_job_records
from vllm.di_pipelines import DI_KEY, DI_PIPELINES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "data" / "vllm" / "di"

configure()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=30, help="Days of history to fetch")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--dry-run", action="store_true",
                        help="List the builds that would be collected, fetch nothing else")
    parser.add_argument("--no-logs", action="store_true",
                        help="Skip job logs. Faster, but loses the SLURM driver's "
                             "phase/reason verdict — the infra-vs-workload distinction")
    parser.add_argument("--grid-only", action="store_true",
                        help="Rebuild grid.json from the existing jobs.jsonl, no API calls")
    args = parser.parse_args()

    output_dir = Path(args.output)

    if args.grid_only:
        records = load_job_records(output_dir / "jobs.jsonl")
        log.info("Rebuilding grid from %d cached records", len(records))
        log.info("Wrote %s", write_grid(records, output_dir))
        return

    builds, _ = collect(
        days=args.days,
        output_dir=output_dir,
        dry_run=args.dry_run,
        fetch_logs=not args.no_logs,
    )

    if args.dry_run:
        log.info("Dry run complete: %d builds in the last %d days", len(builds), args.days)
        return

    # Rebuild the grid from everything on disk, not just this pass, so the
    # history strip survives an incremental run.
    records = load_job_records(output_dir / "jobs.jsonl")
    log.info("Wrote %s", write_grid(records, output_dir))

    # Per-cell health from the framework's analyzer, over the persisted
    # job-level TestResult rows.
    by_build = load_di_results(output_dir / "test_results")
    if by_build:
        health = compute_all_test_health(by_build)
        health_path = output_dir / "di_health.json"
        health_path.write_text(json.dumps({
            "pipeline": DI_PIPELINES[DI_KEY],
            "builds_analyzed": len(by_build),
            "tests": [h.to_dict() for h in health],
        }, indent=2) + "\n")
        log.info("Wrote %s (%d cells, %d builds)", health_path, len(health), len(by_build))

    _print_summary(records)


def _print_summary(records: list[dict]):
    if not records:
        print("\nNo DI job records.\n")
        return
    latest = max(r.get("build_number") or 0 for r in records)
    current = [r for r in records if r.get("build_number") == latest]
    counts: dict[str, int] = {}
    for r in current:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    print("\n" + "=" * 60)
    print(f"DI PIPELINE — build #{latest}")
    print("=" * 60)
    print(f"  steps: {len(current)}")
    for verdict, n in sorted(counts.items()):
        print(f"    {verdict}: {n}")
    unclassified = [r for r in current if not r.get("label_ok")]
    if unclassified:
        print(f"  UNCLASSIFIED LABELS: {len(unclassified)}")
        for r in unclassified[:5]:
            print(f"    {r.get('label')!r}")
    waits = [r["queue_wait_s"] for r in current if isinstance(r.get("queue_wait_s"), (int, float))]
    if waits:
        print(f"  queue wait: max {max(waits) / 60:.1f} min, median "
              f"{sorted(waits)[len(waits) // 2] / 60:.1f} min")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
