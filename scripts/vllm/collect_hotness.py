#!/usr/bin/env python3
"""Windowed hotness aggregator for vLLM AMD CI workload.

Unlike analyzer.py which only examines the latest nightly build, this walks
EVERY build that finished in the last ``max(HOTNESS_WINDOWS_HOURS)`` hours
across the AMD pipelines, then emits a separate aggregation for each window
in ``HOTNESS_WINDOWS_HOURS`` (1h / 3h / 24h / 72h by default). The dashboard
switches between windows client-side.

Outputs ``data/vllm/ci/hotness.json`` with:
  - ``generated_at``: ISO8601 UTC
  - ``window_hours``: default window size in hours (backward-compat)
  - ``builds_examined``: number of builds walked
  - ``test_groups`` / ``branches`` / ``queues``: default-window rows
  - ``windows``: ``{"1h": {...}, "3h": {...}, "24h": {...}, "72h": {...}}``
    where each value has ``test_groups`` / ``branches`` / ``queues`` and
    ``builds_examined`` for that cutoff.

Designed to run as a GitHub Actions hourly cron; cheap enough to re-run
frequently since the widest window caps the Buildkite API cost.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vllm.constants import (  # noqa: E402
    AMD_PIPELINES,
    BK_API_BASE,
    BK_ORG,
    HOTNESS_WINDOW_HOURS,
    HOTNESS_WINDOWS_HOURS,
    is_amd_queue,
    is_excluded_queue,
)
from vllm.ci.utils import (  # noqa: E402
    classify_workload,
    hardware_from_job_name,
    parse_iso,
    percentile,
    queue_from_rules,
)
from vllm.ci import ratelimit  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

OUTPUT = Path(__file__).resolve().parent.parent.parent / "data" / "vllm" / "ci" / "hotness.json"

# Normalize job names to a stable group key.
#
# ``_STRIP_HW_PREFIX`` drops the agent-pool prefix (``mi325_4:``) so runs from
# different pools aggregate into one row. ``_STRIP_SHARD`` drops Buildkite's
# auto-shard suffix (``1/3``) since shards of the same test group share a
# single YAML label. ``_STRIP_TAIL`` is narrow on purpose — it only strips
# GPU-count decorations (``(4 GPUs)``) because those are cosmetic. We do NOT
# strip meaningful trailing parentheticals like ``(Extended Pooling)``,
# ``(Standard)``, ``(CPU)`` — those are distinct test groups in the YAML and
# aggregating them produced a phantom ``Multi-Modal Models`` row in the
# workload trajectory view that had no corresponding YAML label.
_STRIP_TAIL = re.compile(r"\s*\(\d+\s*GPUs?\)\s*$", re.IGNORECASE)
_STRIP_SHARD = re.compile(r"\s+(\d+)/(\d+)\s*$")
_STRIP_HW_PREFIX = re.compile(r"^mi\d+[a-zA-Z]?_\d+\s*:\s*", re.IGNORECASE)


def _bk_get(path: str, token: str, params: dict | None = None):
    headers = {"Authorization": f"Bearer {token}"}
    ratelimit.acquire()
    resp = requests.get(f"{BK_API_BASE}{path}", headers=headers, params=params, timeout=30)
    ratelimit.observe(resp.headers)
    if resp.status_code == 429:
        log.warning("Rate limited")
        return []
    resp.raise_for_status()
    return resp.json()


def _paginate(path: str, token: str, params: dict | None = None, max_pages: int = 20):
    params = dict(params or {})
    params.setdefault("per_page", 100)
    out: list = []
    for page in range(1, max_pages + 1):
        params["page"] = page
        items = _bk_get(path, token, params)
        if not isinstance(items, list) or not items:
            break
        out.extend(items)
        if len(items) < params["per_page"]:
            break
    return out


def _stats(values: list[float]) -> dict:
    """Summary block for hotness rows (``_min`` suffix on percentile keys)."""
    if not values:
        return {"count": 0, "avg_min": 0.0, "p50_min": 0.0, "p90_min": 0.0, "max_min": 0.0}
    s = sorted(values)
    return {
        "count": len(s),
        "avg_min": round(sum(s) / len(s), 1),
        "p50_min": round(percentile(s, 50), 1),
        "p90_min": round(percentile(s, 90), 1),
        "max_min": round(max(s), 1),
    }


def _normalize_group(job_name: str) -> str:
    """Collapse shard / hardware / GPU-count noise in a job name.

    ``mi325_4: V1 e2e (4 GPUs) 1/3`` and ``mi250_1: V1 e2e`` both land on
    ``V1 e2e``. Collapses are deliberately narrow: only drop the agent-pool
    prefix, the ``N/M`` shard suffix, and ``(N GPUs)`` decoration. Other
    trailing parentheticals like ``(Extended Pooling)`` or ``(Standard)`` are
    preserved — they identify distinct YAML test groups, and stripping them
    produced phantom rows in the workload trajectory view that had no
    corresponding label in the pipeline config.
    """
    name = (job_name or "").strip()
    name = _STRIP_HW_PREFIX.sub("", name)
    name = _STRIP_SHARD.sub("", name)
    name = _STRIP_TAIL.sub("", name)
    return name.strip() or job_name or "unknown"


def _build_url(slug: str, build: dict) -> tuple[str | None, str]:
    web_url = str(build.get("web_url") or "")
    if web_url:
        return web_url, "build.web_url"
    number = build.get("number")
    if number:
        return f"https://buildkite.com/{BK_ORG}/{slug}/builds/{number}", "pipeline_build_number"
    return None, "unavailable"


def _job_url(build_url: str | None, job: dict) -> tuple[str | None, str]:
    web_url = str(job.get("web_url") or "")
    if web_url:
        return web_url, "job.web_url"
    job_id = str(job.get("id") or "")
    if build_url and job_id:
        return f"{build_url}/steps/canvas?jid={job_id}&tab=output", "job.id"
    return None, "unavailable"


def _record_order(record: dict) -> tuple:
    return (
        record["finished_at"],
        int(record.get("build_number") or 0),
        str(record.get("job_id") or ""),
    )


def _latest_record(current: dict | None, candidate: dict) -> dict:
    if current is None or _record_order(candidate) > _record_order(current):
        return candidate
    return current


def _latest_evidence(record: dict) -> dict:
    return {
        "observed_at": record["finished_at"].strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pipeline": record["slug"],
        "build_number": record["build_number"],
        "build_url": record.get("build_url"),
        "build_url_source": record.get("build_url_source") or "unavailable",
        "job_name": record["job_name"],
        "job_id": record.get("job_id") or None,
        "job_url": record.get("job_url"),
        "job_url_source": record.get("job_url_source") or "unavailable",
        "state": record["state"],
    }


def _collect_job_records(token: str, max_window_hours: int) -> tuple[list[dict], int]:
    """Walk every build in the widest window; return (job_records, builds_examined).

    A job record is a flat dict with everything aggregators need. This is split
    out so we fetch from Buildkite exactly once and then run N aggregations
    over the same record set, one per window.
    """
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=max_window_hours)

    records: list[dict] = []
    builds_examined = 0

    for slug in AMD_PIPELINES:
        params = {"created_from": cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")}
        builds = _paginate(
            f"/organizations/{BK_ORG}/pipelines/{slug}/builds", token, params, max_pages=20
        )
        log.info("Pipeline %s: %d builds in window", slug, len(builds))

        for build in builds:
            builds_examined += 1
            branch = build.get("branch", "") or ""
            commit = (build.get("commit", "") or "")[:12]
            pr = build.get("pull_request") or {}
            fork_url = pr.get("repository") or ""
            source = build.get("source", "") or ""
            workload = classify_workload(slug, branch)
            build_number = build.get("number", 0)
            build_url, build_url_source = _build_url(slug, build)

            for job in build.get("jobs") or []:
                if job.get("type") != "script":
                    continue
                queue = queue_from_rules(job.get("agent_query_rules"))
                if not queue or is_excluded_queue(queue) or not is_amd_queue(queue):
                    continue
                started = parse_iso(job.get("started_at", ""))
                finished = parse_iso(job.get("finished_at", ""))
                if not started or not finished or finished <= started:
                    continue
                dur_min = (finished - started).total_seconds() / 60
                if dur_min <= 0 or dur_min > 24 * 60:
                    continue  # zombie / stuck job

                job_name = job.get("name", "") or ""
                job_url, job_url_source = _job_url(build_url, job)
                records.append(
                    {
                        "group": _normalize_group(job_name),
                        "hw": hardware_from_job_name(job_name, queue),
                        "workload": workload,
                        "queue": queue,
                        "duration_min": dur_min,
                        "state": job.get("state", "") or "",
                        "finished_at": finished,
                        "branch": branch,
                        "commit": commit,
                        "fork_url": fork_url,
                        "source": source,
                        "slug": slug,
                        "build_number": build_number,
                        "build_url": build_url,
                        "build_url_source": build_url_source,
                        "job_name": job_name,
                        "job_id": str(job.get("id") or ""),
                        "job_url": job_url,
                        "job_url_source": job_url_source,
                    }
                )

    return records, builds_examined


def _aggregate(records: list[dict]) -> dict:
    """Aggregate a pre-filtered list of job records into group/branch/queue rows."""
    group_durations: dict[tuple[str, str], list[float]] = defaultdict(list)
    group_failures: dict[tuple[str, str], int] = defaultdict(int)
    group_latest: dict[tuple[str, str], dict] = {}

    queue_durations: dict[str, list[float]] = defaultdict(list)
    queue_latest: dict[str, dict] = {}

    branch_durations: dict[str, list[float]] = defaultdict(list)
    branch_latest: dict[str, dict] = {}
    branch_builds: dict[str, set] = defaultdict(set)

    for j in records:
        key = (j["group"], j["hw"])
        group_durations[key].append(j["duration_min"])
        if j["state"] in ("failed", "timed_out", "broken"):
            group_failures[key] += 1
        group_latest[key] = _latest_record(group_latest.get(key), j)

        queue_durations[j["queue"]].append(j["duration_min"])
        queue_latest[j["queue"]] = _latest_record(queue_latest.get(j["queue"]), j)
        branch_durations[j["branch"]].append(j["duration_min"])
        branch_builds[j["branch"]].add(f"{j['slug']}#{j['build_number']}")
        branch_latest[j["branch"]] = _latest_record(branch_latest.get(j["branch"]), j)

    group_rows = []
    for (group, hw), durs in group_durations.items():
        stats = _stats(durs)
        fails = group_failures.get((group, hw), 0)
        fail_rate = fails / stats["count"] if stats["count"] else 0.0
        latest = group_latest[(group, hw)]
        stats.update(
            {
                "group": group,
                "hw": hw,
                "workload": latest["workload"],
                "fail_rate_pct": round(fail_rate * 100, 1),
                # Fraction retained for the legacy ci-hotness.js consumer.
                "fail_rate": round(fail_rate, 3),
                "failures": fails,
                "last_seen": latest["finished_at"].strftime("%Y-%m-%dT%H:%M:%SZ"),
                "latest_evidence": _latest_evidence(latest),
            }
        )
        group_rows.append(stats)
    group_rows.sort(key=lambda r: (-r["count"], -r["p90_min"]))

    branch_rows = []
    for branch, durs in branch_durations.items():
        stats = _stats(durs)
        latest = branch_latest[branch]
        stats.update(
            {
                "branch": branch,
                "commit": latest["commit"],
                "fork_url": latest["fork_url"],
                "source": latest["source"],
                "workload": latest["workload"],
                "last_seen": latest["finished_at"].strftime("%Y-%m-%dT%H:%M:%SZ"),
                "builds": len(branch_builds.get(branch, set())),
                "latest_evidence": _latest_evidence(latest),
            }
        )
        branch_rows.append(stats)
    branch_rows.sort(key=lambda r: (-r["builds"], -r["count"]))

    queue_rows = []
    for queue, durs in queue_durations.items():
        stats = _stats(durs)
        stats["queue"] = queue
        stats["last_seen"] = queue_latest[queue]["finished_at"].strftime("%Y-%m-%dT%H:%M:%SZ")
        stats["latest_evidence"] = _latest_evidence(queue_latest[queue])
        queue_rows.append(stats)
    queue_rows.sort(key=lambda r: -r["count"])

    return {
        "test_groups": group_rows,
        "branches": branch_rows,
        "queues": queue_rows,
    }


def collect_hotness(token: str) -> dict:
    now = datetime.now(timezone.utc)
    windows_hours = tuple(sorted(set(HOTNESS_WINDOWS_HOURS) | {HOTNESS_WINDOW_HOURS}))
    max_window = max(windows_hours)

    records, builds_examined = _collect_job_records(token, max_window)

    windows: dict[str, dict] = {}
    for w in windows_hours:
        cutoff = now - timedelta(hours=w)
        scoped = [r for r in records if r["finished_at"] >= cutoff]
        agg = _aggregate(scoped)
        agg["window_hours"] = w
        agg["jobs_in_window"] = len(scoped)
        windows[f"{w}h"] = agg

    default_key = f"{HOTNESS_WINDOW_HOURS}h"
    default_window = windows[default_key]

    return {
        "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "window_hours": HOTNESS_WINDOW_HOURS,
        "builds_examined": builds_examined,
        "test_groups": default_window["test_groups"],
        "branches": default_window["branches"],
        "queues": default_window["queues"],
        "windows": windows,
    }


def main():
    token = os.getenv("BUILDKITE_TOKEN")
    if not token:
        log.error("BUILDKITE_TOKEN not set")
        sys.exit(1)

    log.info("Collecting hotness windows=%s pipelines=%s", HOTNESS_WINDOWS_HOURS, AMD_PIPELINES)
    data = collect_hotness(token)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(data, indent=2))
    log.info(
        "Wrote hotness: %d groups, %d branches, %d queues -> %s",
        len(data["test_groups"]),
        len(data["branches"]),
        len(data["queues"]),
        OUTPUT,
    )


if __name__ == "__main__":
    main()
