"""Fold DI job records into the grid the dashboard renders.

The framework's analysis treats each step as one opaque test, which answers
"is this cell red?" but not "is 2P2D worse than 1P1D?" or "is vllm-router
flakier than proxy?" — the questions this pipeline exists to answer. Those
need the label decomposed, which is what this module does.

Reads ``data/vllm/di/jobs.jsonl``, writes ``data/vllm/di/grid.json``.
"""

from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from .di_pipelines import (
    DI_PIPELINES,
    DI_KEY,
    GRIDS,
    MODELS,
    STEP_TIMEOUT_MINS,
    TRANSPORT,
)

SCHEMA_VERSION = 2

# Verdicts that represent a completed attempt. Everything else (waiting,
# running, canceled, blocked) is excluded from pass rates: a step that never
# ran is not evidence about the cell, and counting it either way lies.
TERMINAL_VERDICTS = frozenset({"passed", "failed", "soft_failed", "timed_out", "broken"})
FAILED_VERDICTS = frozenset({"failed", "timed_out", "broken"})

# Below this many completed attempts, report the count and withhold the
# percentage. These are 120-minute jobs on a scarce allocation, so three
# samples is a normal amount of history — and "67%" from three samples is a
# more confident claim than the data supports.
MIN_SAMPLES_FOR_RATE = 5

# Cap the per-cell outcome strip. Matches the analyzer's flaky window.
HISTORY_LIMIT = 10

# A cell absent from this many of the newest builds is retired rather than
# newly renamed. Wide enough to tolerate one in-flight or partial build,
# narrow enough that a step renamed today still reads as an alert.
RECENT_BUILD_WINDOW = 3


def _cell_id(model: str, shape: str, mode: str, router: str) -> str:
    return f"{model}|{shape}|{mode}|{TRANSPORT}|{router}"


def expected_cells() -> list[dict]:
    """Enumerate every grid as defined in pipeline-disagg.yaml.

    Enumerated rather than derived from observed records so that a cell which
    never ran renders as ``never_run`` instead of vanishing — which is the only
    way the wide-EP 2P2D gap is visible at all.
    """
    cells = []
    for grid in GRIDS:
        for model in MODELS:
            for shape in grid["shapes"]:
                for router in grid["routers"]:
                    cells.append({
                        "cell_id": _cell_id(model, shape, grid["mode"], router),
                        "grid": grid["key"],
                        "model": model,
                        "shape": shape,
                        "mode": grid["mode"],
                        "transport": TRANSPORT,
                        "router": router,
                    })
    return cells


def _attempt(record: dict) -> dict:
    """The per-build slice of a cell's history."""
    return {
        "build_number": record.get("build_number"),
        "date": record.get("date", ""),
        "verdict": record.get("verdict", ""),
        "state": record.get("state", ""),
        "runtime_s": record.get("runtime_s"),
        "queue_wait_s": record.get("queue_wait_s"),
        "agent_name": record.get("agent_name", ""),
        "job_url": record.get("job_url", ""),
        "phase": record.get("phase", ""),
        "slurm_state": record.get("slurm_state", ""),
        "failure_class": record.get("failure_class", ""),
        "reason": record.get("reason", ""),
    }


def _median(values: Iterable[Optional[float]]) -> Optional[float]:
    nums = [v for v in values if isinstance(v, (int, float))]
    return round(statistics.median(nums), 1) if nums else None


def _count_flips(verdicts: list[str]) -> int:
    """Transitions between pass and fail across consecutive completed attempts.

    A cell that alternates is a different problem from one that is steadily
    red, and the steady one is usually already known.
    """
    binary = [v == "passed" for v in verdicts if v in TERMINAL_VERDICTS]
    return sum(1 for a, b in zip(binary, binary[1:]) if a != b)


def summarize_cell(cell: dict, records: list[dict]) -> dict:
    """Attach outcome history and rates to one enumerated cell."""
    ordered = sorted(records, key=lambda r: r.get("build_number") or 0)
    attempts = [_attempt(r) for r in ordered]
    verdicts = [a["verdict"] for a in attempts]
    completed = [v for v in verdicts if v in TERMINAL_VERDICTS]
    passed = sum(1 for v in completed if v == "passed")

    out = dict(cell)
    out["history"] = attempts[-HISTORY_LIMIT:][::-1]  # newest first
    out["attempts"] = len(attempts)
    out["completed"] = len(completed)
    out["passed"] = passed
    out["failed"] = sum(1 for v in completed if v in FAILED_VERDICTS)
    # None, not 0.0, when there is nothing to divide — the renderer must be
    # able to tell "no data" from "never passes".
    out["pass_rate"] = round(passed / len(completed), 4) if completed else None
    out["rate_is_reportable"] = len(completed) >= MIN_SAMPLES_FOR_RATE
    out["flips"] = _count_flips(verdicts)
    out["last_verdict"] = attempts[-1]["verdict"] if attempts else "never_run"
    out["last_build"] = attempts[-1]["build_number"] if attempts else None
    out["median_runtime_s"] = _median(a["runtime_s"] for a in attempts)
    out["median_queue_wait_s"] = _median(a["queue_wait_s"] for a in attempts)
    return out


def agent_attribution(records: list[dict]) -> list[dict]:
    """Failures grouped by agent, to expose a single sick MI350 box."""
    by_agent: dict[str, dict] = {}
    for r in records:
        if r.get("verdict") not in TERMINAL_VERDICTS:
            continue
        name = r.get("agent_name") or "unknown"
        row = by_agent.setdefault(name, {"agent_name": name, "completed": 0, "failed": 0})
        row["completed"] += 1
        if r.get("verdict") in FAILED_VERDICTS:
            row["failed"] += 1
    for row in by_agent.values():
        row["failure_rate"] = round(row["failed"] / row["completed"], 4)
    return sorted(by_agent.values(), key=lambda r: (-r["failed"], r["agent_name"]))


def failure_classes(records: list[dict]) -> dict[str, int]:
    """Counts of the SLURM driver's own infra/bringup/workload classification.

    Empty until logs are collected; the driver's verdict line is the only
    place this distinction exists today.
    """
    counts: dict[str, int] = {}
    for r in records:
        if r.get("verdict") not in FAILED_VERDICTS:
            continue
        cls = r.get("failure_class") or "unknown"
        counts[cls] = counts.get(cls, 0) + 1
    return dict(sorted(counts.items()))


def build_rollup(records: list[dict]) -> list[dict]:
    """Per-build, per-model pass counts — the "which build broke what" view.

    Derived here rather than in the renderer because the per-cell ``history``
    is capped at HISTORY_LIMIT to match the flaky window, so folding it back up
    would silently drop the middle builds.
    """
    builds: dict[int, dict] = {}
    for r in records:
        number = r.get("build_number")
        if number is None or not r.get("label_ok"):
            continue
        build = builds.setdefault(number, {
            "build_number": number,
            "date": r.get("date", ""),
            "models": {},
        })
        model = build["models"].setdefault(
            r.get("model", ""), {"passed": 0, "completed": 0, "runs": []}
        )
        verdict = r.get("verdict", "")
        model["runs"].append({
            "shape": r.get("shape", ""),
            "router": r.get("router", ""),
            "mode": r.get("mode", ""),
            "verdict": verdict,
            "runtime_s": r.get("runtime_s"),
            "job_url": r.get("job_url", ""),
            "failure_class": r.get("failure_class", ""),
            "slurm_state": r.get("slurm_state", ""),
            "reason": r.get("reason", ""),
        })
        if verdict in TERMINAL_VERDICTS:
            model["completed"] += 1
            if verdict == "passed":
                model["passed"] += 1
    return sorted(builds.values(), key=lambda b: b["build_number"], reverse=True)


def build_grid(records: list[dict]) -> dict:
    """Assemble the full grid payload from job records."""
    by_cell: dict[str, list[dict]] = {}
    unclassified: list[dict] = []
    for r in records:
        if not r.get("label_ok"):
            unclassified.append(_attempt(r) | {"label": r.get("label", "")})
            continue
        by_cell.setdefault(r.get("cell_id", ""), []).append(r)

    cells = [summarize_cell(c, by_cell.get(c["cell_id"], [])) for c in expected_cells()]

    # A cell observed in the data but absent from the enumerated grid means a
    # step was added or renamed upstream. Surface it rather than dropping it —
    # but a step renamed today and one dead for months want opposite volumes,
    # so split them on whether the cell still appears in the recent builds.
    recent = {b for b in sorted(
        {r.get("build_number") for r in records if r.get("build_number") is not None},
        reverse=True,
    )[:RECENT_BUILD_WINDOW]}

    known = {c["cell_id"] for c in cells}
    for cell_id, cell_records in sorted(by_cell.items()):
        if cell_id in known:
            continue
        first = cell_records[0]
        live = any(r.get("build_number") in recent for r in cell_records)
        cells.append(summarize_cell({
            "cell_id": cell_id,
            "grid": None,
            "model": first.get("model", ""),
            "shape": first.get("shape", ""),
            "mode": first.get("mode", ""),
            "transport": first.get("transport", ""),
            "router": first.get("router", ""),
            "unexpected": live,
            "retired": not live,
        }, cell_records))

    builds: dict[int, dict] = {}
    for r in records:
        number = r.get("build_number")
        if number is not None and number not in builds:
            builds[number] = {
                "build_number": number,
                "build_url": r.get("build_url", ""),
                "state": r.get("build_state", ""),
                "branch": r.get("branch", ""),
                "commit": r.get("commit", ""),
                "created_at": r.get("created_at", ""),
            }

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pipeline": DI_PIPELINES[DI_KEY],
        "axes": {
            "models": list(MODELS),
            "transport": TRANSPORT,
            # One entry per rendered grid. Mode is carried per grid rather than
            # globally: a wide-EP cell shares model, shape and router with a
            # TP8 cell, so a renderer keying without mode collapses the two.
            "grids": [
                {
                    "key": g["key"],
                    "title": g["title"],
                    "mode": g["mode"],
                    "mode_label": g["mode_label"],
                    "shapes": list(g["shapes"]),
                    "routers": list(g["routers"]),
                    "note": g["note"],
                }
                for g in GRIDS
            ],
        },
        "step_timeout_mins": STEP_TIMEOUT_MINS,
        "min_samples_for_rate": MIN_SAMPLES_FOR_RATE,
        "builds": sorted(builds.values(), key=lambda b: b["build_number"], reverse=True),
        "build_rollup": build_rollup(records),
        "cells": cells,
        "unclassified": unclassified,
        "agents": agent_attribution(records),
        "failure_classes": failure_classes(records),
    }


def write_grid(records: list[dict], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "grid.json"
    path.write_text(json.dumps(build_grid(records), indent=2) + "\n")
    return path
