#!/usr/bin/env python3
"""Collect privacy-safe AMD queue mappings for vLLM Omni and vLLM.

Queue snapshots measure occupancy, not how many unique jobs were assigned to
AMD hardware.  This collector queries the explicitly configured Buildkite
pipelines, deduplicates command-job UUIDs in memory, and publishes aggregates
only.  UUIDs, build payloads, and other raw job records are never written.

Schema v2 keeps 90 UTC calendar days of daily buckets and at least seven days
of hourly buckets.  Every bucket includes queue and pipeline breakdowns.
Build queries are split into bounded UTC-day slices so a high-volume range
cannot silently exhaust one global pagination cap or retain a 90-day raw
response in memory.  Incremental runs refresh recent buckets, preserve older
committed buckets, and backfill missing or incomplete hourly/daily coverage.

The REST builds endpoint filters parent-build creation time, while mappings
are bucketed by job creation time.  A configurable parent-build lookback makes
that source cohort conservative; the output explicitly records that jobs
added to arbitrarily older builds are not provably exhaustive.

The current hour/day is explicitly ``open`` and has an ``observed_through``
timestamp.  Its interval is partial, but ``collection_complete`` independently
reports whether the API query itself completed; an open interval is therefore
not mislabeled as a collection failure.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time as time_module
from collections import defaultdict
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vllm.constants import BK_API_BASE, BK_ORG  # noqa: E402
from vllm.ci.utils import parse_iso, queue_from_rules  # noqa: E402
from vllm.ci import ratelimit  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT / "config" / "vllm_amd_queue_capacity.json"
OUTPUT = ROOT / "data" / "vllm" / "ci" / "workload_mapping.json"

DEFAULT_BOOTSTRAP_DAYS = 90
DEFAULT_REFRESH_DAYS = 2
DEFAULT_RETENTION_DAYS = 90
DEFAULT_HOURLY_RETENTION_DAYS = 7
DEFAULT_PARENT_BUILD_LOOKBACK_DAYS = 3
DEFAULT_WINDOW_DAYS = 14
DEFAULT_MAX_PAGES = 50
PER_PAGE = 100
REQUEST_ATTEMPTS = 6
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

REPOSITORY_LABELS = {
    "omni": "vllm-project/vllm-omni",
    "main": "vllm-project/vllm",
}
STAT_FIELDS = (
    "mapped_jobs",
    "started_jobs",
    "finished_jobs",
    "mapped_gpu_slots",
)

log = logging.getLogger(__name__)


def _utc_iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _day_start(value: datetime) -> datetime:
    value = value.astimezone(timezone.utc)
    return datetime.combine(value.date(), time.min, tzinfo=timezone.utc)


def _hour_start(value: datetime) -> datetime:
    value = value.astimezone(timezone.utc)
    return value.replace(minute=0, second=0, microsecond=0)


def _date_range(start: date, end: date) -> list[date]:
    if end < start:
        return []
    return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]


def _hour_range(start: datetime, end: datetime) -> Iterable[datetime]:
    cursor = _hour_start(start)
    last = _hour_start(end)
    while cursor <= last:
        yield cursor
        cursor += timedelta(hours=1)


def _bounded_utc_slices(
    start: datetime,
    end: datetime,
) -> Iterable[tuple[datetime, datetime]]:
    """Yield at most one-UTC-day half-open slices spanning ``[start, end)``."""
    cursor = start.astimezone(timezone.utc)
    end = end.astimezone(timezone.utc)
    while cursor < end:
        next_midnight = _day_start(cursor) + timedelta(days=1)
        boundary = min(end, next_midnight)
        yield cursor, boundary
        cursor = boundary


def load_config(path: Path = CONFIG_PATH) -> dict:
    data = json.loads(path.read_text())
    if data.get("schema_version") != 1:
        raise ValueError(f"Unsupported AMD queue config schema in {path}")
    if not isinstance(data.get("queues"), list):
        raise ValueError(f"AMD queue config has no queues list: {path}")
    pipelines = data.get("workload_pipelines")
    if not isinstance(pipelines, dict) or not all(
        isinstance(pipelines.get(name), list) and pipelines[name] for name in ("omni", "main")
    ):
        raise ValueError(f"AMD queue config has incomplete workload_pipelines: {path}")
    return data


def monitored_queues(config: dict) -> dict[str, dict]:
    """Return the exact public AMD queue allowlist keyed by Buildkite queue ID."""
    rows: dict[str, dict] = {}
    for raw in config.get("queues") or []:
        if not isinstance(raw, dict) or raw.get("monitored") is not True:
            continue
        queue_id = str(raw.get("id") or "").strip()
        if not queue_id or "perf_eval" in queue_id.casefold():
            continue
        try:
            gpus_per_job = int(raw.get("gpus_per_job"))
        except (TypeError, ValueError):
            continue
        if gpus_per_job not in (1, 2, 4, 8):
            continue
        rows[queue_id] = {
            "id": queue_id,
            "label": raw.get("label") or queue_id.removeprefix("amd_"),
            "family": raw.get("family") or "unknown",
            "gpus_per_job": gpus_per_job,
            "lifecycle": raw.get("lifecycle") or "unknown",
        }
    if not rows:
        raise ValueError("AMD queue config produced an empty monitored queue allowlist")
    return rows


def _job_queue(job: dict) -> str:
    queue = queue_from_rules(job.get("agent_query_rules"))
    if queue:
        return queue
    cluster_queue = job.get("cluster_queue")
    if isinstance(cluster_queue, dict):
        return str(cluster_queue.get("key") or "")
    return ""


def _job_mapped_at(job: dict, build: dict) -> datetime | None:
    return (
        parse_iso(job.get("created_at"))
        or parse_iso(job.get("runnable_at"))
        or parse_iso(build.get("created_at"))
    )


def _job_gpu_hours(job: dict, gpus_per_job: int) -> float | None:
    started = parse_iso(job.get("started_at"))
    finished = parse_iso(job.get("finished_at"))
    if started is None or finished is None or finished <= started:
        return None
    duration_hours = (finished - started).total_seconds() / 3600
    # Treat records longer than a day as stale rather than publishing a
    # misleading resource-consumption spike.
    if duration_hours > 24:
        return None
    return duration_hours * gpus_per_job


def _empty_stats() -> dict:
    return {
        "mapped_jobs": 0,
        "started_jobs": 0,
        "finished_jobs": 0,
        "mapped_gpu_slots": 0,
        "gpu_hours": 0.0,
    }


def _empty_workload() -> dict:
    return {
        **_empty_stats(),
        "by_queue": {},
        "by_pipeline": {},
    }


def _empty_day(day: str) -> dict:
    """Compatibility helper used by tests and schema-v1 migration."""
    return {
        "date": day,
        "state": "closed",
        "open": False,
        "partial": False,
        "complete": True,
        "collection_complete": True,
        "lower_bound": False,
        "workloads": {
            "omni": _empty_workload(),
            "main": _empty_workload(),
        },
    }


def _request_build_page(
    path: str,
    token: str,
    params: dict[str, Any],
) -> list[dict]:
    response = None
    for attempt in range(1, REQUEST_ATTEMPTS + 1):
        try:
            ratelimit.acquire()
            response = requests.get(
                f"{BK_API_BASE}{path}",
                headers={"Authorization": f"Bearer {token}"},
                params=params,
                timeout=90,
            )
            ratelimit.observe(response.headers)
            if response.status_code not in RETRYABLE_STATUS_CODES:
                response.raise_for_status()
                break
            response.raise_for_status()
        except requests.RequestException as exc:
            retryable = (
                response is None
                or response.status_code in RETRYABLE_STATUS_CODES
            )
            if not retryable:
                raise
            if attempt >= REQUEST_ATTEMPTS:
                raise
            retry_after = 0
            if response is not None:
                for header in (
                    "Retry-After",
                    "RateLimit-Reset",
                    "RateLimit-User-Reset",
                ):
                    try:
                        retry_after = max(
                            retry_after,
                            int(float(response.headers.get(header) or 0)),
                        )
                    except (TypeError, ValueError):
                        continue
            delay = min(60, max(1, retry_after, 2 ** (attempt - 1)))
            log.warning(
                "Buildkite request retry %d/%d after %s; waiting %ds",
                attempt,
                REQUEST_ATTEMPTS,
                type(exc).__name__,
                delay,
            )
            time_module.sleep(delay)
    if response is None:
        raise RuntimeError("Buildkite request did not produce a response")
    payload = response.json()
    if not isinstance(payload, list):
        raise TypeError(f"Buildkite returned non-list payload for {path}")
    return payload


def _fetch_pipeline_slice(
    path: str,
    token: str,
    pipeline: str,
    start: datetime,
    end: datetime,
    *,
    max_pages: int,
    page_fetcher: Callable[[str, str, dict[str, Any]], list[dict]],
) -> tuple[list[dict], dict]:
    base_params: dict[str, Any] = {
        "created_from": _utc_iso(start),
        "created_to": _utc_iso(end),
        "include_retried_jobs": "true",
        "exclude_pipeline": "true",
        "per_page": PER_PAGE,
    }
    builds: list[dict] = []
    complete = False
    error_type: str | None = None
    pages = 0
    for page in range(1, max_pages + 1):
        try:
            rows = page_fetcher(path, token, {**base_params, "page": page})
        except Exception as exc:  # retain lower-bound metadata, never raw responses
            error_type = type(exc).__name__
            log.warning(
                "Failed %s %s page %d after retries (%s); publishing a lower bound",
                pipeline,
                start.date().isoformat(),
                page,
                error_type,
            )
            break
        pages = page
        log.info(
            "Fetched %s %s page %d (%d builds)",
            pipeline,
            start.date().isoformat(),
            page,
            len(rows),
        )
        if not rows:
            complete = True
            break
        builds.extend(row for row in rows if isinstance(row, dict))
        if len(rows) < PER_PAGE:
            complete = True
            break
    return builds, {
        "start": _utc_iso(start),
        "end_exclusive": _utc_iso(end),
        "pages_fetched": pages,
        "builds_fetched": len(builds),
        "complete": complete,
        "truncated": not complete and error_type is None and pages >= max_pages,
        "error_type": error_type,
    }


def _iter_pipeline_build_slices(
    token: str,
    pipeline: str,
    start: datetime,
    end: datetime,
    *,
    max_pages: int = DEFAULT_MAX_PAGES,
    page_fetcher: Callable[[str, str, dict[str, Any]], list[dict]] = _request_build_page,
) -> Iterable[tuple[list[dict], dict]]:
    """Yield one bounded slice at a time so callers can release raw builds."""
    path = f"/organizations/{BK_ORG}/pipelines/{pipeline}/builds"
    for slice_start, slice_end in _bounded_utc_slices(start, end):
        yield _fetch_pipeline_slice(
            path,
            token,
            pipeline,
            slice_start,
            slice_end,
            max_pages=max_pages,
            page_fetcher=page_fetcher,
        )


def _summarize_pipeline_slices(
    pipeline: str,
    start: datetime,
    end: datetime,
    slices: list[dict],
) -> dict:
    complete = bool(slices) and all(row["complete"] for row in slices)
    error_types = sorted({str(row["error_type"]) for row in slices if row.get("error_type")})
    return {
        "pipeline": pipeline,
        "start": _utc_iso(start),
        "end_exclusive": _utc_iso(end),
        "bounded_slice": "UTC day",
        "slice_count": len(slices),
        "pages_fetched": sum(int(row["pages_fetched"]) for row in slices),
        "builds_fetched": sum(int(row["builds_fetched"]) for row in slices),
        "complete": complete,
        "truncated": any(row["truncated"] for row in slices),
        "error_types": error_types,
        "slices": slices,
    }


def fetch_pipeline_builds(
    token: str,
    pipeline: str,
    start: datetime,
    end: datetime,
    *,
    max_pages: int = DEFAULT_MAX_PAGES,
    page_fetcher: Callable[[str, str, dict[str, Any]], list[dict]] = _request_build_page,
) -> tuple[list[dict], dict]:
    """Compatibility helper that materializes independently paginated slices.

    The production collector consumes ``_iter_pipeline_build_slices`` directly
    and therefore never holds an entire multi-day raw-build range in memory.
    """
    builds: list[dict] = []
    slices: list[dict] = []
    for rows, source in _iter_pipeline_build_slices(
        token,
        pipeline,
        start,
        end,
        max_pages=max_pages,
        page_fetcher=page_fetcher,
    ):
        builds.extend(rows)
        slices.append(source)
    return builds, _summarize_pipeline_slices(pipeline, start, end, slices)


def _events_from_builds(
    builds: list[dict],
    *,
    workload: str,
    pipeline: str,
    queue_catalog: dict[str, dict],
    start: datetime,
    end: datetime,
    seen_job_ids: set[str] | None = None,
) -> tuple[list[dict], dict]:
    events: list[dict] = []
    seen = seen_job_ids if seen_job_ids is not None else set()
    seen_before = len(seen)
    missing_job_ids = 0
    duplicate_job_ids = 0
    missing_mapped_at: list[datetime] = []
    for build in builds:
        build_pipeline = str((build.get("pipeline") or {}).get("slug") or pipeline)
        if build_pipeline != pipeline:
            continue
        for job in build.get("jobs") or []:
            if not isinstance(job, dict) or job.get("type") not in {"script", "command"}:
                continue
            queue = _job_queue(job)
            if queue not in queue_catalog:
                continue
            mapped_at = _job_mapped_at(job, build)
            if mapped_at is None or mapped_at < start or mapped_at >= end:
                continue
            job_id = str(job.get("id") or "").strip()
            if not job_id:
                missing_job_ids += 1
                missing_mapped_at.append(mapped_at)
                continue
            if job_id in seen:
                duplicate_job_ids += 1
                continue
            seen.add(job_id)
            queue_row = queue_catalog[queue]
            events.append(
                {
                    # Kept only in memory for deduplication.  Aggregation removes it.
                    "job_id": job_id,
                    "mapped_at": mapped_at,
                    "workload": workload,
                    "pipeline": pipeline,
                    "queue": queue,
                    "gpus_per_job": queue_row["gpus_per_job"],
                    "started": parse_iso(job.get("started_at")) is not None,
                    "finished": parse_iso(job.get("finished_at")) is not None,
                    "gpu_hours": _job_gpu_hours(job, queue_row["gpus_per_job"]),
                }
            )
    return events, {
        "mapped_job_ids": len(seen) - seen_before,
        "missing_job_ids": missing_job_ids,
        "duplicate_job_ids": duplicate_job_ids,
        # Internal only.  The caller removes these timestamps before publishing.
        "_missing_mapped_at": missing_mapped_at,
    }


def _add_event(bucket: dict, event: dict) -> None:
    bucket["mapped_jobs"] += 1
    bucket["started_jobs"] += int(event["started"])
    bucket["finished_jobs"] += int(event["finished"])
    bucket["mapped_gpu_slots"] += int(event["gpus_per_job"])
    if event["gpu_hours"] is not None:
        bucket["gpu_hours"] += float(event["gpu_hours"])

    for dimension, value in (
        ("by_queue", event["queue"]),
        ("by_pipeline", event["pipeline"]),
    ):
        dimension_bucket = bucket[dimension].setdefault(value, _empty_stats())
        dimension_bucket["mapped_jobs"] += 1
        dimension_bucket["started_jobs"] += int(event["started"])
        dimension_bucket["finished_jobs"] += int(event["finished"])
        dimension_bucket["mapped_gpu_slots"] += int(event["gpus_per_job"])
        if event["gpu_hours"] is not None:
            dimension_bucket["gpu_hours"] += float(event["gpu_hours"])


def _round_workload(workload: dict) -> None:
    workload["gpu_hours"] = round(float(workload["gpu_hours"]), 2)
    for dimension in ("by_queue", "by_pipeline"):
        workload[dimension] = {
            key: {
                **stats,
                "gpu_hours": round(float(stats["gpu_hours"]), 2),
            }
            for key, stats in sorted(workload[dimension].items())
        }


def _source_covers_interval(
    source: dict,
    start: datetime,
    observed_end: datetime,
) -> bool:
    """Return API completeness for the build-created range relevant to a bucket."""
    lookback_days = max(1, int(source.get("parent_build_lookback_days") or 1))
    required_start = start - timedelta(days=lookback_days)
    source_start = parse_iso(source.get("start"))
    source_end = parse_iso(source.get("end_exclusive"))
    if (
        source_start is None
        or source_end is None
        or source_start > required_start
        or source_end < observed_end
    ):
        return False
    relevant = []
    for row in source.get("slices") or []:
        slice_start = parse_iso(row.get("start"))
        slice_end = parse_iso(row.get("end_exclusive"))
        if slice_start is None or slice_end is None:
            return False
        if slice_start < observed_end and slice_end > required_start:
            relevant.append(row)
    return bool(relevant) and all(bool(row.get("complete")) for row in relevant)


def _collection_complete_for_bucket(
    sources: list[dict],
    missing_by_workload: dict[str, list[datetime]],
    start: datetime,
    observed_end: datetime,
    *,
    workload: str | None = None,
) -> bool:
    relevant_sources = [
        source
        for source in sources
        if workload is None or source.get("workload") == workload
    ]
    if not relevant_sources or not all(
        _source_covers_interval(source, start, observed_end)
        for source in relevant_sources
    ):
        return False
    return not any(
        start <= missing_at < observed_end
        for workload_name, missing_times in missing_by_workload.items()
        if workload is None or workload_name == workload
        for missing_at in missing_times
    )


def _bucket_status(
    start: datetime,
    nominal_end: datetime,
    now: datetime,
    collection_complete: bool,
) -> dict:
    observed_end = min(nominal_end, now)
    is_open = nominal_end > now
    state = "open" if is_open else ("closed" if collection_complete else "partial")
    return {
        "end_exclusive": _utc_iso(nominal_end),
        "observed_through": _utc_iso(observed_end),
        "state": state,
        "open": is_open,
        "partial": is_open or not collection_complete,
        # Backward-compatible "complete" means a closed interval whose query
        # completed.  Query truth remains independently available below.
        "complete": not is_open and collection_complete,
        "collection_complete": collection_complete,
        "lower_bound": not collection_complete,
    }


def _accumulate_event(
    buckets: dict[str, dict],
    event: dict,
    *,
    key: str,
) -> None:
    mapped_at = event["mapped_at"]
    bucket_key = (
        _utc_iso(_hour_start(mapped_at))
        if key == "hour"
        else mapped_at.astimezone(timezone.utc).date().isoformat()
    )
    workloads = buckets.setdefault(
        bucket_key,
        {
            "omni": _empty_workload(),
            "main": _empty_workload(),
        },
    )
    _add_event(workloads[event["workload"]], event)


def _finalize_intervals(
    buckets: dict[str, dict],
    starts: Iterable[datetime],
    *,
    interval: timedelta,
    key: str,
    now: datetime,
    sources: list[dict],
    missing_by_workload: dict[str, list[datetime]],
) -> list[dict]:
    rows: list[dict] = []
    for start in starts:
        nominal_end = start + interval
        observed_end = min(nominal_end, now)
        bucket_key = _utc_iso(start) if key == "hour" else start.date().isoformat()
        row = {
            key: bucket_key,
            "workloads": buckets.get(bucket_key)
            or {
                "omni": _empty_workload(),
                "main": _empty_workload(),
            },
        }
        complete_by_workload = {
            workload: _collection_complete_for_bucket(
                sources,
                missing_by_workload,
                start,
                observed_end,
                workload=workload,
            )
            for workload in ("omni", "main")
        }
        collection_complete = all(complete_by_workload.values())
        row["collection_complete_by_workload"] = complete_by_workload
        row.update(_bucket_status(start, nominal_end, now, collection_complete))
        for workload in row["workloads"].values():
            _round_workload(workload)
        rows.append(row)
    return rows


def _normalize_existing_workload(raw: Any) -> dict:
    result = _empty_workload()
    if not isinstance(raw, dict):
        return result
    for field in (*STAT_FIELDS, "gpu_hours"):
        value = raw.get(field)
        if field == "gpu_hours":
            result[field] = round(float(value or 0), 2)
        else:
            result[field] = int(value or 0)
    for dimension in ("by_queue", "by_pipeline"):
        if not isinstance(raw.get(dimension), dict):
            continue
        for name, stats in raw[dimension].items():
            if not isinstance(stats, dict):
                continue
            normalized = _empty_stats()
            for field in (*STAT_FIELDS, "gpu_hours"):
                if field == "gpu_hours":
                    normalized[field] = round(float(stats.get(field) or 0), 2)
                else:
                    normalized[field] = int(stats.get(field) or 0)
            result[dimension][str(name)] = normalized
    return result


def _normalize_existing_row(
    raw: dict,
    *,
    key: str,
    now: datetime,
) -> dict | None:
    raw_key = str(raw.get(key) or "")
    start = parse_iso(raw_key) if key == "hour" else parse_iso(f"{raw_key}T00:00:00Z")
    if start is None:
        return None
    nominal_end = start + (timedelta(hours=1) if key == "hour" else timedelta(days=1))
    collection_complete = bool(raw.get("collection_complete", not bool(raw.get("lower_bound"))))
    raw_by_workload = raw.get("collection_complete_by_workload")
    complete_by_workload = {
        workload: bool(
            raw_by_workload.get(workload, collection_complete)
            if isinstance(raw_by_workload, dict)
            else collection_complete
        )
        for workload in ("omni", "main")
    }
    collection_complete = all(complete_by_workload.values())
    row = {
        key: _utc_iso(start) if key == "hour" else start.date().isoformat(),
        "collection_complete_by_workload": complete_by_workload,
        "workloads": {
            workload: _normalize_existing_workload((raw.get("workloads") or {}).get(workload))
            for workload in ("omni", "main")
        },
    }
    row.update(_bucket_status(start, nominal_end, now, collection_complete))
    return row


def _merge_buckets(
    existing: dict,
    replacements: list[dict],
    *,
    collection: str,
    key: str,
    cutoff: str,
    ceiling: str,
    now: datetime,
) -> list[dict]:
    merged: dict[str, dict] = {}
    for raw in existing.get(collection) or []:
        if not isinstance(raw, dict):
            continue
        row = _normalize_existing_row(raw, key=key, now=now)
        if row is not None:
            merged[row[key]] = row
    merged.update({row[key]: row for row in replacements})
    return [
        merged[bucket]
        for bucket in sorted(merged)
        if cutoff <= bucket <= ceiling
    ]


def _sum_workloads(rows: list[dict]) -> dict:
    totals = {"omni": _empty_workload(), "main": _empty_workload()}
    for row in rows:
        for workload_name, raw_bucket in (row.get("workloads") or {}).items():
            if workload_name not in totals or not isinstance(raw_bucket, dict):
                continue
            bucket = _normalize_existing_workload(raw_bucket)
            target = totals[workload_name]
            for field in STAT_FIELDS:
                target[field] += bucket[field]
            target["gpu_hours"] += bucket["gpu_hours"]
            for dimension in ("by_queue", "by_pipeline"):
                for name, dimension_bucket in bucket[dimension].items():
                    aggregate = target[dimension].setdefault(name, _empty_stats())
                    for field in STAT_FIELDS:
                        aggregate[field] += dimension_bucket[field]
                    aggregate["gpu_hours"] += dimension_bucket["gpu_hours"]
    for bucket in totals.values():
        _round_workload(bucket)
    return totals


def _coverage(
    rows: list[dict],
    *,
    key: str,
    resolution: str,
    retention_days: int,
    now: datetime,
    expected_start: datetime,
    expected_end: datetime,
) -> dict:
    bucket_seconds = 3600 if key == "hour" else 86400
    expected_bucket_count = int(
        (expected_end - expected_start).total_seconds() / bucket_seconds
    )
    if not rows:
        return {
            "resolution": resolution,
            "retention_days": retention_days,
            "start": None,
            "expected_start": _utc_iso(expected_start),
            "end_exclusive": None,
            "observed_through": _utc_iso(now),
            "bucket_count": 0,
            "expected_bucket_count": expected_bucket_count,
            "missing_bucket_count": expected_bucket_count,
            "contiguous": False,
            "collection_complete": False,
            "job_created_range_exhaustive": False,
            "has_open_bucket": False,
        }
    first = rows[0]
    last = rows[-1]
    start = first[key] if key == "hour" else f"{first[key]}T00:00:00Z"
    missing_bucket_count = max(0, expected_bucket_count - len(rows))
    return {
        "resolution": resolution,
        "retention_days": retention_days,
        "start": start,
        "expected_start": _utc_iso(expected_start),
        "end_exclusive": last["end_exclusive"],
        "observed_through": last["observed_through"],
        "bucket_count": len(rows),
        "expected_bucket_count": expected_bucket_count,
        "missing_bucket_count": missing_bucket_count,
        "contiguous": missing_bucket_count == 0,
        "collection_complete": (
            missing_bucket_count == 0 and all(row["collection_complete"] for row in rows)
        ),
        "job_created_range_exhaustive": False,
        "has_open_bucket": any(row["open"] for row in rows),
    }


def _earliest_missing_daily(
    existing: dict,
    start: date,
    end: date,
) -> date | None:
    present = {
        str(row.get("date"))
        for row in existing.get("daily") or []
        if isinstance(row, dict) and row.get("date")
    }
    for day in _date_range(start, end):
        if day.isoformat() not in present:
            return day
    return None


def _earliest_missing_hourly(
    existing: dict,
    start: datetime,
    end: datetime,
) -> datetime | None:
    present = {
        str(row.get("hour"))
        for row in existing.get("hourly") or []
        if isinstance(row, dict) and row.get("hour")
    }
    for hour in _hour_range(start, end):
        if _utc_iso(hour) not in present:
            return hour
    return None


def _row_needs_refresh(row: dict) -> bool:
    if row.get("collection_complete") is not True:
        return True
    workloads = row.get("workloads")
    if not isinstance(workloads, dict):
        return True
    return any(
        not isinstance(workloads.get(workload), dict)
        or not isinstance(workloads[workload].get("by_queue"), dict)
        or not isinstance(workloads[workload].get("by_pipeline"), dict)
        for workload in ("omni", "main")
    )


def _earliest_incomplete_daily(
    existing: dict,
    start: date,
    end: date,
) -> date | None:
    candidates = []
    for row in existing.get("daily") or []:
        if not isinstance(row, dict) or not _row_needs_refresh(row):
            continue
        value = str(row.get("date") or "")
        try:
            day = date.fromisoformat(value)
        except ValueError:
            continue
        if start <= day <= end:
            candidates.append(day)
    return min(candidates) if candidates else None


def _earliest_incomplete_hourly(
    existing: dict,
    start: datetime,
    end: datetime,
) -> datetime | None:
    candidates = []
    for row in existing.get("hourly") or []:
        if not isinstance(row, dict) or not _row_needs_refresh(row):
            continue
        hour = parse_iso(row.get("hour"))
        if hour is not None and start <= hour <= end:
            candidates.append(hour)
    return min(candidates) if candidates else None


def collect_workload_mapping(
    token: str,
    config: dict,
    *,
    existing: dict | None = None,
    now: datetime | None = None,
    bootstrap_days: int = DEFAULT_BOOTSTRAP_DAYS,
    refresh_days: int = DEFAULT_REFRESH_DAYS,
    retention_days: int = DEFAULT_RETENTION_DAYS,
    hourly_retention_days: int = DEFAULT_HOURLY_RETENTION_DAYS,
    parent_build_lookback_days: int = DEFAULT_PARENT_BUILD_LOOKBACK_DAYS,
    window_days: int = DEFAULT_WINDOW_DAYS,
    max_pages: int = DEFAULT_MAX_PAGES,
    page_fetcher: Callable[[str, str, dict[str, Any]], list[dict]] = _request_build_page,
    force_days: int | None = None,
) -> dict:
    # Published source boundaries are serialized to whole seconds. Normalize
    # the in-memory boundary too, otherwise an open bucket can be mislabeled as
    # incomplete solely because source_end lost ``now``'s microseconds.
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).replace(
        microsecond=0
    )
    existing = existing if isinstance(existing, dict) else {}
    retention_days = max(90, retention_days)
    hourly_retention_days = max(7, hourly_retention_days)
    parent_build_lookback_days = max(1, parent_build_lookback_days)
    current_hour = _hour_start(now)
    daily_retention_start = now.date() - timedelta(days=retention_days - 1)
    hourly_retention_start = current_hour - timedelta(days=hourly_retention_days)

    if force_days is not None:
        query_start = _day_start(now) - timedelta(days=max(1, force_days) - 1)
    else:
        refresh_start = _day_start(now) - timedelta(days=max(1, refresh_days) - 1)
        candidates = [refresh_start]

        if not existing.get("daily"):
            candidates.append(_day_start(now) - timedelta(days=max(1, bootstrap_days) - 1))
        missing_day = _earliest_missing_daily(
            existing,
            daily_retention_start,
            now.date(),
        )
        if missing_day is not None:
            candidates.append(datetime.combine(missing_day, time.min, tzinfo=timezone.utc))
        incomplete_day = _earliest_incomplete_daily(
            existing,
            daily_retention_start,
            now.date(),
        )
        if incomplete_day is not None:
            candidates.append(
                datetime.combine(incomplete_day, time.min, tzinfo=timezone.utc)
            )

        missing_hour = _earliest_missing_hourly(
            existing,
            hourly_retention_start,
            current_hour,
        )
        if missing_hour is not None:
            candidates.append(missing_hour)
        incomplete_hour = _earliest_incomplete_hourly(
            existing,
            hourly_retention_start,
            current_hour,
        )
        if incomplete_hour is not None:
            candidates.append(incomplete_hour)

        # A schema-v1 aggregate has no pipeline dimensions.  Re-query retained
        # history so v2 totals never pretend that an empty by_pipeline map is
        # complete evidence.
        if existing and existing.get("schema_version") != 2:
            candidates.append(
                datetime.combine(
                    daily_retention_start,
                    time.min,
                    tzinfo=timezone.utc,
                )
            )
        query_start = _day_start(min(candidates))

    query_end = now
    build_query_start = query_start - timedelta(days=parent_build_lookback_days)
    queue_catalog = monitored_queues(config)

    daily_buckets: dict[str, dict] = {}
    hourly_buckets: dict[str, dict] = {}
    global_seen_job_ids: set[str] = set()
    sources: list[dict] = []
    missing_by_workload: dict[str, list[datetime]] = defaultdict(list)
    diagnostics: dict[str, dict] = {}
    for workload in ("omni", "main"):
        workload_sources = []
        missing_ids = 0
        duplicates = 0
        cross_pipeline_duplicates = 0
        for pipeline in config["workload_pipelines"][workload]:
            pipeline_seen_job_ids: set[str] = set()
            pipeline_slices: list[dict] = []
            pipeline_missing_ids = 0
            pipeline_duplicates = 0
            pipeline_missing_times: list[datetime] = []
            for builds, slice_source in _iter_pipeline_build_slices(
                token,
                pipeline,
                build_query_start,
                query_end,
                max_pages=max_pages,
                page_fetcher=page_fetcher,
            ):
                pipeline_slices.append(slice_source)
                extracted, event_meta = _events_from_builds(
                    builds,
                    workload=workload,
                    pipeline=pipeline,
                    queue_catalog=queue_catalog,
                    start=query_start,
                    end=query_end,
                    seen_job_ids=pipeline_seen_job_ids,
                )
                pipeline_missing_ids += event_meta["missing_job_ids"]
                pipeline_duplicates += event_meta["duplicate_job_ids"]
                pipeline_missing_times.extend(event_meta["_missing_mapped_at"])
                for event in extracted:
                    job_id = event["job_id"]
                    if job_id in global_seen_job_ids:
                        cross_pipeline_duplicates += 1
                        continue
                    global_seen_job_ids.add(job_id)
                    _accumulate_event(daily_buckets, event, key="date")
                    if event["mapped_at"] >= hourly_retention_start:
                        _accumulate_event(hourly_buckets, event, key="hour")
                # Bound peak memory to one source day.  The aggregate contains
                # no job UUIDs and survives after these raw objects are released.
                del builds, extracted

            missing_by_workload[workload].extend(pipeline_missing_times)
            source = _summarize_pipeline_slices(
                pipeline,
                build_query_start,
                query_end,
                pipeline_slices,
            )
            source.update(
                {
                    "workload": workload,
                    "repository": REPOSITORY_LABELS[workload],
                    "parent_build_lookback_days": parent_build_lookback_days,
                    "mapped_jobs_in_query": len(pipeline_seen_job_ids),
                    "mapped_job_ids": len(pipeline_seen_job_ids),
                    "missing_job_ids": pipeline_missing_ids,
                    "duplicate_job_ids": pipeline_duplicates,
                }
            )
            sources.append(source)
            workload_sources.append(source)
            missing_ids += pipeline_missing_ids
            duplicates += pipeline_duplicates
        diagnostics[workload] = {
            "repository": REPOSITORY_LABELS[workload],
            "pipelines": len(workload_sources),
            "missing_job_ids": missing_ids,
            "duplicate_job_ids": duplicates,
            "cross_pipeline_duplicate_job_ids": cross_pipeline_duplicates,
        }

    daily_replacements = _finalize_intervals(
        daily_buckets,
        (
            datetime.combine(day, time.min, tzinfo=timezone.utc)
            for day in _date_range(query_start.date(), now.date())
        ),
        interval=timedelta(days=1),
        key="date",
        now=now,
        sources=sources,
        missing_by_workload=missing_by_workload,
    )
    hourly_replacements = _finalize_intervals(
        hourly_buckets,
        _hour_range(max(query_start, hourly_retention_start), current_hour),
        interval=timedelta(hours=1),
        key="hour",
        now=now,
        sources=sources,
        missing_by_workload=missing_by_workload,
    )

    daily = _merge_buckets(
        existing,
        daily_replacements,
        collection="daily",
        key="date",
        cutoff=daily_retention_start.isoformat(),
        ceiling=now.date().isoformat(),
        now=now,
    )
    hourly = _merge_buckets(
        existing,
        hourly_replacements,
        collection="hourly",
        key="hour",
        cutoff=_utc_iso(hourly_retention_start),
        ceiling=_utc_iso(current_hour),
        now=now,
    )

    window_start = now.date() - timedelta(days=max(1, window_days) - 1)
    window_rows = [row for row in daily if row["date"] >= window_start.isoformat()]
    window_collection_complete = len(window_rows) == window_days and all(
        row["collection_complete"] for row in window_rows
    )
    collection_start = daily[0]["date"] if daily else now.date().isoformat()
    repositories = {
        workload: {
            "label": REPOSITORY_LABELS[workload],
            "pipelines": list(config["workload_pipelines"][workload]),
        }
        for workload in ("omni", "main")
    }

    return {
        "schema_version": 2,
        "generated_at": _utc_iso(now),
        "collection_start": collection_start,
        "timezone": "UTC",
        "repositories": repositories,
        "retention": {
            "hourly_days": hourly_retention_days,
            "daily_days": retention_days,
        },
        "coverage": {
            "hourly": _coverage(
                hourly,
                key="hour",
                resolution="UTC hour",
                retention_days=hourly_retention_days,
                now=now,
                expected_start=hourly_retention_start,
                expected_end=current_hour + timedelta(hours=1),
            ),
            "daily": _coverage(
                daily,
                key="date",
                resolution="UTC calendar day",
                retention_days=retention_days,
                now=now,
                expected_start=datetime.combine(
                    daily_retention_start,
                    time.min,
                    tzinfo=timezone.utc,
                ),
                expected_end=_day_start(now) + timedelta(days=1),
            ),
        },
        "window": {
            "days": window_days,
            "start_date": window_start.isoformat(),
            "end_date": now.date().isoformat(),
            "start": f"{window_start.isoformat()}T00:00:00Z",
            "observed_through": _utc_iso(now),
            "state": "open",
            "complete": window_collection_complete,
            "collection_complete": window_collection_complete,
            "job_created_range_exhaustive": False,
            "lower_bound": not window_collection_complete,
        },
        "scope": {
            "queues": sorted(queue_catalog),
            "excluded_queue_classes": list(
                (config.get("scope") or {}).get("excluded_queue_classes") or []
            ),
            "workload_pipelines": config["workload_pipelines"],
            "repositories": repositories,
            "attribution": {
                "mapping_timestamp": "job.created_at",
                "source_filter_timestamp": "parent build.created_at",
                "parent_build_lookback_days": parent_build_lookback_days,
                "job_created_range_exhaustive": False,
                "exact_within_declared_source_window": True,
                "limitation": (
                    "Jobs added to a parent build more than the configured "
                    "lookback after that build was created are outside the "
                    "REST source window and can be absent from these aggregates."
                ),
            },
        },
        "semantics": {
            "mapped_jobs": (
                "Unique Buildkite command-job UUIDs whose explicit queue rule maps "
                "to a monitored AMD queue; retry attempts are distinct UUIDs."
            ),
            "hourly_bucket": (
                "UTC hour containing the job mapping timestamp. The open hour is "
                "observed only through observed_through."
            ),
            "daily_bucket": (
                "UTC calendar day containing the job mapping timestamp. The open "
                "day is observed only through observed_through."
            ),
            "collection_complete": (
                "Whether all relevant bounded API slices completed and every "
                "mapped record had a UUID inside the declared parent-build "
                "source window; independent of whether a bucket is open. It "
                "does not claim exhaustive job-created coverage for arbitrarily "
                "old parent builds."
            ),
            "lower_bound": (
                "Signals incomplete API/UUID collection inside the declared "
                "source window. Consult job_created_range_exhaustive separately "
                "when interpreting the count as all jobs created in the interval."
            ),
            "started_jobs": "Mapped jobs with a Buildkite started_at timestamp.",
            "mapped_gpu_slots": "Sum of configured GPUs per mapped job; not GPU-hours.",
            "gpu_hours": (
                "Sum of started-to-finished wall hours multiplied by configured GPUs "
                "per job; unfinished and stale >24h records are excluded."
            ),
            "privacy": (
                "Only hourly/daily aggregates and query diagnostics are published; "
                "raw builds, raw jobs, and UUIDs are not retained."
            ),
        },
        "query": {
            "start": _utc_iso(query_start),
            "end_exclusive": _utc_iso(query_end),
            "build_created_start": _utc_iso(build_query_start),
            "parent_build_lookback_days": parent_build_lookback_days,
            "job_created_range_exhaustive": False,
            "bounded_slice": "UTC day",
            "bootstrap_days": bootstrap_days,
            "refresh_days": refresh_days,
            "forced_days": force_days,
            "pipeline_sources": sources,
            "diagnostics": diagnostics,
        },
        "totals": _sum_workloads(window_rows),
        "hourly": hourly,
        "daily": daily,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect vLLM Omni/vLLM AMD job mappings",
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--bootstrap-days", type=int, default=DEFAULT_BOOTSTRAP_DAYS)
    parser.add_argument("--refresh-days", type=int, default=DEFAULT_REFRESH_DAYS)
    parser.add_argument("--retention-days", type=int, default=DEFAULT_RETENTION_DAYS)
    parser.add_argument(
        "--hourly-retention-days",
        type=int,
        default=DEFAULT_HOURLY_RETENTION_DAYS,
    )
    parser.add_argument(
        "--parent-build-lookback-days",
        type=int,
        default=DEFAULT_PARENT_BUILD_LOOKBACK_DAYS,
        help=(
            "Include parent builds created this many days before the first "
            "job-created bucket (REST cannot query jobs directly by created_at)."
        ),
    )
    parser.add_argument("--window-days", type=int, default=DEFAULT_WINDOW_DAYS)
    parser.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES)
    parser.add_argument(
        "--force-days",
        type=int,
        default=None,
        help="Ignore incremental refresh and replace this many UTC calendar days.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    token = os.getenv("BUILDKITE_TOKEN", "").strip()
    if not token:
        raise SystemExit("BUILDKITE_TOKEN not set")
    config = load_config(args.config)
    existing = {}
    if args.output.exists():
        try:
            existing = json.loads(args.output.read_text())
        except (OSError, json.JSONDecodeError):
            log.warning("Ignoring unreadable existing workload mapping at %s", args.output)
    payload = collect_workload_mapping(
        token,
        config,
        existing=existing,
        bootstrap_days=args.bootstrap_days,
        refresh_days=args.refresh_days,
        retention_days=args.retention_days,
        hourly_retention_days=args.hourly_retention_days,
        parent_build_lookback_days=args.parent_build_lookback_days,
        window_days=args.window_days,
        max_pages=args.max_pages,
        force_days=args.force_days,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    log.info(
        "Wrote %s: Omni=%d vLLM=%d mapped jobs in the %d-day window (%s)",
        args.output,
        payload["totals"]["omni"]["mapped_jobs"],
        payload["totals"]["main"]["mapped_jobs"],
        payload["window"]["days"],
        ("query complete" if payload["window"]["collection_complete"] else "lower bound"),
    )


if __name__ == "__main__":
    main()
