#!/usr/bin/env python3
"""Buildkite queue snapshot collector for dashboard queue monitoring.

Appends one JSON line per snapshot to ``data/vllm/ci/queue_timeseries.jsonl``.

The collector prefers Buildkite's queue-native cluster metrics for queue
counts and wait-time percentiles. Active jobs are still collected for job
detail, workload splits, zombie filtering, and as a fallback when queue-native
metrics are unavailable.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import logging
import os
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

# Add scripts/ to sys.path so the ``vllm`` package resolves when this file is
# executed as ``python scripts/vllm/collect_queue_snapshot.py``.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vllm.constants import (  # noqa: E402
    AMD_METRIC_TARGET_QUEUES,
    BK_API_BASE,
    BK_CLUSTER_UUID,
    BK_GRAPHQL_URL,
    BK_ORG,
    QUEUE_HISTORY_ARCHIVE_BUCKET_MINUTES,
    QUEUE_HISTORY_HIGH_RES_HOURS,
    QUEUE_HISTORY_RETENTION_DAYS,
    QUEUE_ZOMBIE_THRESHOLD_MIN,
    TRACKED_QUEUES,
    is_amd_queue,
    is_excluded_queue,
    queue_history_reset_datetime,
)
from vllm.ci.utils import classify_workload, parse_iso, percentile, queue_from_rules  # noqa: E402
from vllm.ci import ratelimit  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

OUTPUT = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "vllm"
    / "ci"
    / "queue_timeseries.jsonl"
)
HISTORY_REPO_PATH = "data/vllm/ci/queue_timeseries.jsonl"

# Buildkite URL rewrite: the jobs endpoint returns hash-anchored URLs that
# 404 in the step canvas; re-point them so dashboard links land on the output tab.
_JOB_URL_REWRITE = re.compile(r"^(https://buildkite\.com/vllm/[a-z\-]+/builds/\d+)#([0-9a-f\-]+)$")

GRAPHQL_QUEUE_METRICS_Q = """
query QueueMetrics($org: ID!, $cluster: ID!, $first: Int!, $after: String) {
  organization(slug: $org) {
    cluster(id: $cluster) {
      queues(first: $first, after: $after) {
        edges {
          node {
            id
            key
            uuid
            dispatchPaused
            metrics {
              timestamp
              connectedAgentsCount
              waitingJobsCount
              runningJobsCount
              jobsPassedCount
              jobsFailedCount
              waitTimeSec {
                min
                p50
                p95
                max
              }
            }
          }
        }
        pageInfo {
          hasNextPage
          endCursor
        }
      }
    }
  }
}
"""

GRAPHQL_ACTIVE_JOBS_Q = """
query ActiveJobs($org: ID!, $states: [JobStates!], $first: Int!, $after: String) {
  organization(slug: $org) {
    jobs(
      first: $first,
      after: $after,
      clustered: true,
      type: [COMMAND],
      state: $states
    ) {
      edges {
        node {
          ... on JobTypeCommand {
            uuid
            state
            label
            runnableAt
            scheduledAt
            createdAt
            startedAt
            agentQueryRules
            clusterQueue {
              key
            }
            build {
              number
              branch
              commit
              url
            }
            pipeline {
              slug
            }
          }
        }
      }
      pageInfo {
        hasNextPage
        endCursor
      }
    }
  }
}
"""

GRAPHQL_QUEUE_JOBS_Q = """
query QueueJobs($org: ID!, $queue: [ID!]!, $states: [JobStates!], $first: Int!, $after: String) {
  organization(slug: $org) {
    jobs(
      first: $first,
      after: $after,
      clusterQueue: $queue,
      type: [COMMAND],
      state: $states
    ) {
      edges {
        node {
          ... on JobTypeCommand {
            uuid
            state
            label
            runnableAt
            scheduledAt
            createdAt
            startedAt
            agentQueryRules
            clusterQueue {
              key
            }
            build {
              number
              branch
              commit
              url
            }
            pipeline {
              slug
            }
          }
        }
      }
      pageInfo {
        hasNextPage
        endCursor
      }
    }
  }
}
"""

GRAPHQL_PAGE_SIZE = 100
GRAPHQL_WAITING_STATES = frozenset({"SCHEDULED"})
GRAPHQL_RUNNING_STATES = frozenset({"ASSIGNED", "ACCEPTED", "RUNNING", "CANCELING", "TIMING_OUT"})
GRAPHQL_ACTIVE_STATES = tuple(sorted(GRAPHQL_WAITING_STATES | GRAPHQL_RUNNING_STATES))

# Legacy REST build scan states. These are intentionally aligned with
# Buildkite's queue metrics docs rather than the older dashboard behavior:
# only ``scheduled`` jobs are "waiting", while assigned/accepted jobs count
# as already dispatched / running. Concurrency-limited jobs are excluded
# because they are not part of queue-page waiting-job metrics.
LEGACY_WAITING_STATES = frozenset({"scheduled"})
LEGACY_RUNNING_STATES = frozenset({"assigned", "accepted", "running", "canceling", "timing_out"})
# At 100 records per page this permits 10,000 active builds per state while
# still bounding a broken/repeating pagination response. Hitting the bound is
# an incomplete observation and must fail rather than publish partial counts.
REST_PAGINATION_SAFETY_CAP = 100


def bk_get(path: str, token: str, params: dict | None = None):
    headers = {"Authorization": f"Bearer {token}"}
    ratelimit.acquire()
    resp = requests.get(f"{BK_API_BASE}{path}", headers=headers, params=params, timeout=30)
    ratelimit.observe(resp.headers)
    if resp.status_code == 429:
        raise RuntimeError(f"Buildkite REST API rate limited on {path}")
    resp.raise_for_status()
    return resp.json()


def bk_get_paginated(
    path: str,
    token: str,
    params: dict | None = None,
    max_pages: int = REST_PAGINATION_SAFETY_CAP,
):
    """Fetch every REST page, or fail if exhaustion cannot be proven."""
    if max_pages < 1:
        raise ValueError("max_pages must be positive")
    params = dict(params or {})
    params.setdefault("per_page", 100)
    all_items: list = []
    for page in range(1, max_pages + 1):
        params["page"] = page
        items = bk_get(path, token, params)
        if not isinstance(items, list):
            raise RuntimeError(f"Buildkite REST API returned a non-list page for {path}")
        if not items:
            break
        all_items.extend(items)
        if len(items) < params["per_page"]:
            break
        if page == max_pages:
            raise RuntimeError(
                f"Buildkite REST pagination safety cap reached for {path} "
                f"after {max_pages} full pages"
            )
    return all_items


def bk_graphql(query: str, token: str, variables: dict | None = None) -> dict:
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    ratelimit.acquire()
    resp = requests.post(
        BK_GRAPHQL_URL,
        headers=headers,
        json={"query": query, "variables": variables or {}},
        timeout=30,
    )
    ratelimit.observe(resp.headers)
    if resp.status_code == 429:
        raise RuntimeError("Buildkite GraphQL rate limited")
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("errors"):
        raise RuntimeError(
            f"Buildkite GraphQL error: {payload['errors'][0].get('message', 'unknown')}"
        )
    return payload.get("data") or {}


def _rewrite_job_url(web_url: str) -> str:
    m = _JOB_URL_REWRITE.match(web_url or "")
    if m:
        return f"{m.group(1)}/steps/canvas?jid={m.group(2)}&tab=output"
    return web_url


def _queue_web_url(queue_uuid: str | None) -> str:
    if not queue_uuid:
        return ""
    return f"https://buildkite.com/organizations/{BK_ORG}/clusters/{BK_CLUSTER_UUID}/queues/{queue_uuid}"


def _queue_row() -> dict:
    return {
        "waiting": 0,
        "running": 0,
        "scheduled": 0,
        "total": 0,
        "connected_agents": None,
        "connected_agents_source": None,
        "jobs_passed": None,
        "jobs_passed_source": None,
        "jobs_failed": None,
        "jobs_failed_source": None,
        "zombie_waiting": 0,
        "zombie_running": 0,
        "wait_times": [],
        "count_source": "active_job_scan",
    }


def _history_cutoff(now: datetime) -> datetime:
    retention_cutoff = now - timedelta(days=QUEUE_HISTORY_RETENTION_DAYS)
    return max(retention_cutoff, queue_history_reset_datetime())


def _queue_row_has_current_schema(row: dict) -> bool:
    if not isinstance(row, dict):
        return False
    return (
        isinstance(row.get("official_wait"), dict)
        and isinstance(row.get("sample_wait"), dict)
        and isinstance(row.get("current_wait"), dict)
        and "p50_wait_source" in row
        and "p95_wait_source" in row
        and "p99_wait_source" in row
    )


def _snapshot_has_current_schema(snapshot: dict) -> bool:
    if not isinstance(snapshot, dict):
        return False
    sources = snapshot.get("sources")
    if not isinstance(sources, dict) or not isinstance(sources.get("wait_fields"), dict):
        return False
    queues = snapshot.get("queues")
    if not isinstance(queues, dict):
        return False
    for row in queues.values():
        if not _queue_row_has_current_schema(row):
            return False
    return True


_SAMPLE_WAIT_METRICS = ("p50", "p75", "p90", "p95", "p99", "max", "avg")


def _as_count(value, default: int = 0) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def _as_optional_count(value) -> int | None:
    if value is None:
        return None
    return _as_count(value)


def _as_optional_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _empty_official_wait() -> dict:
    return {"min": None, "p50": None, "p95": None, "max": None}


def _empty_sample_wait(*, available: bool, count: int | None) -> dict:
    return {
        "available": available,
        "count": count,
        **{metric: None for metric in _SAMPLE_WAIT_METRICS},
    }


def _reconcile_wait_sample(row: dict, sampled: dict) -> tuple[int, bool, dict]:
    """Compare a scheduled-job scan with counts, never with native wait values.

    Queue-native metrics and job reads are independent observations.  Exhaustive
    pagination proves that the job query ended, but an equal count cannot prove
    that both reads observed the same job membership.  The compatibility
    ``wait_sample_complete`` flag therefore means count coverage only; the
    structured reconciliation makes that limitation explicit.
    """
    waiting = _as_count(row.get("waiting"))
    zombie_waiting = _as_count(row.get("zombie_waiting"))
    sample_available = bool(sampled.get("available"))
    sample_count = _as_optional_count(sampled.get("count"))
    count_source = str(row.get("count_source") or "unknown")

    if count_source == "cluster_metrics":
        expected_non_zombie = max(0, waiting - zombie_waiting)
        reference_kind = "queue_native_waiting_jobs_including_observed_zombies"
        reference_count = waiting
        observed_count = (
            sample_count + zombie_waiting if sample_available and sample_count is not None else None
        )
    else:
        # Active-job counts already exclude jobs classified as zombies from the
        # row's waiting total, so compare the same non-zombie population.
        expected_non_zombie = waiting
        reference_kind = "published_non_zombie_waiting_jobs"
        reference_count = waiting
        observed_count = sample_count if sample_available else None

    if not sample_available:
        status = "not_sampled"
        reason = "scheduled_job_scan_not_performed_for_queue"
        count_delta = None
        counts_match = None
    elif sample_count is None:
        status = "invalid"
        reason = "scheduled_job_scan_missing_count"
        count_delta = None
        counts_match = False
    else:
        count_delta = observed_count - reference_count
        counts_match = count_delta == 0
        if counts_match:
            status = "count_match"
            reason = None
        elif count_delta < 0:
            status = "count_mismatch"
            reason = "scheduled_job_scan_below_reference_count"
        else:
            status = "count_mismatch"
            reason = "scheduled_job_scan_above_reference_count"

    count_complete = counts_match is True
    details = {
        "status": status,
        "reason": reason,
        "reference_kind": reference_kind,
        "reference_count": reference_count,
        "observed_count": observed_count,
        "count_delta": count_delta,
        "membership_verified": False,
        "native_wait_values_used": False,
    }
    return expected_non_zombie, count_complete, details


def _apply_wait_contract(
    row: dict,
    official_wait: dict,
    sample_wait: dict,
    *,
    emit_reconciliation: bool = True,
) -> dict:
    """Attach typed wait fields and their provenance to one queue row."""
    official = {
        metric: _as_optional_float(official_wait.get(metric))
        for metric in ("min", "p50", "p95", "max")
    }
    sample_count = _as_optional_count(sample_wait.get("count"))
    sample_available = bool(sample_wait.get("available"))
    sampled = _empty_sample_wait(available=sample_available, count=sample_count)
    if sample_available and sample_count:
        for metric in _SAMPLE_WAIT_METRICS:
            sampled[metric] = _as_optional_float(sample_wait.get(metric))
    if sample_wait.get("source"):
        sampled["source"] = sample_wait["source"]

    row["official_wait"] = official
    row["sample_wait"] = sampled
    row["wait_sample_count"] = sampled["count"]

    expected_count, sample_complete, reconciliation = _reconcile_wait_sample(row, sampled)
    row["wait_sample_expected_count"] = expected_count
    row["wait_sample_complete"] = sample_complete
    row.pop("wait_sample_promotable", None)
    row.pop("wait_sample_reconciliation", None)
    if emit_reconciliation and (
        expected_count > 0
        or (sampled["count"] or 0) > 0
        or reconciliation["status"] == "count_mismatch"
    ):
        row["wait_sample_reconciliation"] = reconciliation

    row["min_wait"] = official["min"]
    row["min_wait_source"] = "official_wait" if official["min"] is not None else None

    for metric in ("p50", "p95"):
        if official[metric] is not None:
            value, source = official[metric], "official_wait"
        elif sample_complete and sampled[metric] is not None:
            value, source = sampled[metric], "sample_wait"
        else:
            value, source = None, None
        row[f"{metric}_wait"] = value
        row[f"{metric}_wait_source"] = source

    row["p99_wait"] = sampled["p99"] if sample_complete else None
    row["p99_wait_source"] = "sample_wait" if row["p99_wait"] is not None else None
    row["current_wait"] = {
        metric: {
            "value": row[f"{metric}_wait"],
            "source": row[f"{metric}_wait_source"],
        }
        for metric in ("p50", "p95", "p99")
    }

    for metric in ("p75", "p90", "avg"):
        row[f"{metric}_wait"] = sampled[metric] if sample_complete else None
        row[f"{metric}_wait_source"] = "sample_wait" if row[f"{metric}_wait"] is not None else None
    official_max = official["max"]
    sampled_max = sampled["max"] if sample_complete else None
    if sampled_max is not None and (official_max is None or sampled_max > official_max):
        row["max_wait"] = sampled_max
        row["max_wait_source"] = "sample_wait"
    else:
        row["max_wait"] = official_max
        row["max_wait_source"] = "official_wait" if official_max is not None else None

    row["wait_source"] = {
        "official_wait": "cluster_metrics",
        "sample_wait": "scheduled_jobs",
    }.get(row["p95_wait_source"], "none")
    return row


def _apply_metric_sources(row: dict) -> dict:
    """Attach compact per-field source labels to queue metrics in a live row."""
    count_source = str(row.get("count_source") or "unknown")
    if count_source == "cluster_metrics":
        count_provider = "queue_native_metrics"
    elif count_source == "active_job_scan":
        count_provider = "active_job_scan"
    else:
        count_provider = count_source if count_source != "unknown" else None

    row["waiting_source"] = count_provider
    row["running_source"] = count_provider
    row["jobs_passed"] = _as_optional_count(row.get("jobs_passed"))
    row["jobs_failed"] = _as_optional_count(row.get("jobs_failed"))
    if row["jobs_passed"] is not None and not row.get("jobs_passed_source"):
        row["jobs_passed_source"] = (
            "queue_native_metrics" if count_source == "cluster_metrics" else None
        )
    if row["jobs_failed"] is not None and not row.get("jobs_failed_source"):
        row["jobs_failed_source"] = (
            "queue_native_metrics" if count_source == "cluster_metrics" else None
        )

    # Provider paths are invariant and live once in sources.metric_fields.
    # Repeating them for every queue more than doubles the 48-hour payload.
    row.pop("official_wait_field_sources", None)
    row.pop("field_provenance", None)
    return row


def _normalize_workload_splits(row: dict, workload_source: str) -> dict:
    """Validate job-scan workload splits against authoritative queue totals.

    Queue-native metrics and active-job scans are separate observations and can
    be captured a few seconds apart. A split that exceeds its queue total is
    therefore evidence of timing drift, not permission to increase or scale the
    queue count. Preserve that evidence in provenance while making the split
    unavailable to workload-history consumers.
    """
    for split_key, total_key in (
        ("waiting_by_workload", "waiting"),
        ("running_by_workload", "running"),
    ):
        provenance_key = f"{split_key}_provenance"
        existing_provenance = row.get(provenance_key)
        split = row.get(split_key)
        if split is None:
            if isinstance(existing_provenance, dict):
                row[provenance_key] = dict(existing_provenance)
            continue

        source = str(
            row.get(f"{split_key}_source")
            or (existing_provenance.get("source") if isinstance(existing_provenance, dict) else "")
            or workload_source
            or "active_job_scan"
        )
        if not isinstance(split, dict):
            row[split_key] = None
            row[provenance_key] = {
                "available": False,
                "status": "invalid",
                "source": source,
                "reason": "workload_split_is_not_an_object",
                "queue_total": row[total_key],
            }
            continue

        normalized_split: dict[str, int] = {}
        invalid_value = False
        for workload, value in split.items():
            try:
                count = int(value)
            except (TypeError, ValueError):
                invalid_value = True
                break
            if count < 0:
                invalid_value = True
                break
            normalized_split[str(workload)] = count

        if invalid_value:
            row[split_key] = None
            row[provenance_key] = {
                "available": False,
                "status": "invalid",
                "source": source,
                "reason": "workload_split_contains_invalid_count",
                "queue_total": row[total_key],
                "observed_split": split,
            }
            continue

        split_total = sum(normalized_split.values())
        queue_total = row[total_key]
        if split_total > queue_total:
            row[split_key] = None
            row[provenance_key] = {
                "available": False,
                "status": "inconsistent",
                "source": source,
                "reason": "observed_split_exceeds_queue_total",
                "queue_total": queue_total,
                "observed_split_total": split_total,
                "observed_split": normalized_split,
            }
            continue

        row[split_key] = normalized_split
        row[provenance_key] = {
            "available": True,
            "status": "complete" if split_total == queue_total else "partial",
            "source": source,
            "queue_total": queue_total,
            "observed_split_total": split_total,
        }
    return row


def _normalize_queue_row(
    row: dict,
    snapshot_count_source: str,
    snapshot_active_jobs_source: str,
    *,
    legacy: bool,
) -> dict:
    source_row = row if isinstance(row, dict) else {}
    # Detailed evidence is intentionally forward-only. Backfilling it into
    # every retained JSONL row roughly doubles the history file, while adding
    # no evidence that was captured at the time. Newly collected rows already
    # carry these markers before they enter normalization, so re-normalization
    # remains idempotent without inflating older snapshots.
    retain_native_activity = not legacy and any(
        field in source_row
        for field in ("jobs_passed", "jobs_passed_source", "jobs_failed", "jobs_failed_source")
    )
    source_official_wait = source_row.get("official_wait")
    retain_native_min = not legacy and (
        (isinstance(source_official_wait, dict) and "min" in source_official_wait)
        or "min_wait" in source_row
    )
    retain_wait_reconciliation = not legacy and isinstance(
        source_row.get("wait_sample_reconciliation"), dict
    )
    retain_metric_sources = not legacy and (
        "waiting_source" in source_row or "running_source" in source_row
    )

    normalized = dict(source_row)
    normalized.pop("wait_times", None)
    normalized["waiting"] = _as_count(source_row.get("waiting"))
    normalized["running"] = _as_count(source_row.get("running"))
    normalized["scheduled"] = normalized["waiting"]
    normalized["total"] = normalized["waiting"] + normalized["running"]
    normalized["zombie_waiting"] = _as_count(source_row.get("zombie_waiting"))
    normalized["zombie_running"] = _as_count(source_row.get("zombie_running"))
    if retain_native_activity:
        normalized["jobs_passed"] = _as_optional_count(source_row.get("jobs_passed"))
        normalized["jobs_failed"] = _as_optional_count(source_row.get("jobs_failed"))
    else:
        for field in ("jobs_passed", "jobs_passed_source", "jobs_failed", "jobs_failed_source"):
            normalized.pop(field, None)

    original_count_source = str(
        source_row.get("count_source") or snapshot_count_source or "unknown"
    )
    if legacy:
        normalized["count_source"] = "historical_counts"
        normalized["count_provenance"] = {
            "kind": "legacy_snapshot",
            "original_source": original_count_source,
            "preserved_fields": ["waiting", "running"],
        }
        normalized["connected_agents"] = None
        normalized["connected_agents_source"] = None
        official_wait = _empty_official_wait()
        sample_count = _as_count(source_row.get("wait_sample_count"))
        sample_wait = _empty_sample_wait(
            available=sample_count > 0,
            count=sample_count,
        )
        if sample_count > 0:
            sample_wait.update(
                {
                    metric: _as_optional_float(source_row.get(f"{metric}_wait"))
                    for metric in _SAMPLE_WAIT_METRICS
                }
            )
            sample_wait["source"] = "historical_scheduled_job_sample"
        normalized["official_wait_source"] = None
        normalized["sample_wait_source"] = (
            "historical_scheduled_job_sample" if sample_count > 0 else None
        )
    else:
        count_source = original_count_source
        if count_source == "active_jobs":
            count_source = "active_job_scan"
        normalized["count_source"] = count_source
        agent_source = source_row.get("connected_agents_source")
        if (
            not agent_source
            and count_source == "cluster_metrics"
            and "connected_agents" in source_row
            and source_row.get("connected_agents") is not None
        ):
            agent_source = "queue_native_metrics"
        normalized["connected_agents_source"] = agent_source or None
        normalized["connected_agents"] = (
            _as_count(source_row.get("connected_agents")) if agent_source else None
        )
        if retain_native_activity:
            normalized["jobs_passed_source"] = source_row.get("jobs_passed_source") or (
                "queue_native_metrics"
                if count_source == "cluster_metrics" and normalized["jobs_passed"] is not None
                else None
            )
            normalized["jobs_failed_source"] = source_row.get("jobs_failed_source") or (
                "queue_native_metrics"
                if count_source == "cluster_metrics" and normalized["jobs_failed"] is not None
                else None
            )
        official_wait = source_row.get("official_wait") or _empty_official_wait()
        sample_wait = source_row.get("sample_wait") or _empty_sample_wait(
            available=False,
            count=None,
        )
        has_official = any(
            _as_optional_float(official_wait.get(metric)) is not None
            for metric in ("min", "p50", "p95", "max")
        )
        normalized["official_wait_source"] = source_row.get("official_wait_source") or (
            "queue_native_metrics" if has_official else None
        )
        normalized["sample_wait_source"] = (
            source_row.get("sample_wait_source")
            or sample_wait.get("source")
            or ("active_job_scan" if sample_wait.get("available") else None)
        )

    normalized = _apply_wait_contract(
        normalized,
        official_wait,
        sample_wait,
        emit_reconciliation=retain_wait_reconciliation,
    )
    if not retain_native_min:
        normalized["official_wait"].pop("min", None)
        normalized.pop("min_wait", None)
        normalized.pop("min_wait_source", None)
    normalized.pop("official_wait_field_sources", None)
    normalized.pop("field_provenance", None)
    if retain_metric_sources:
        normalized = _apply_metric_sources(normalized)
    return _normalize_workload_splits(normalized, snapshot_active_jobs_source)


def _scope_totals(queues: dict[str, dict]) -> dict:
    count_sources = sorted({str(row.get("count_source") or "unknown") for row in queues.values()})
    if not count_sources:
        source = "unavailable"
    elif len(count_sources) == 1:
        source = count_sources[0]
    else:
        source = "mixed"
    return {
        "waiting": sum(row["waiting"] for row in queues.values()),
        "running": sum(row["running"] for row in queues.values()),
        "count_source": source,
        "count_sources": count_sources,
        "queue_count": len(queues),
    }


def _target_queue_scope(queues: dict[str, dict]) -> dict:
    """Describe the canonical AMD metric cohort without filtering other queues."""
    queue_ids = list(AMD_METRIC_TARGET_QUEUES)
    present = [queue for queue in queue_ids if queue in queues]
    return {
        "id": "amd_mi250_mi300_mi355",
        "families": ["MI250", "MI300", "MI355"],
        "gpu_widths": [1, 2, 4, 8],
        "queue_ids": queue_ids,
        "queue_count": len(queue_ids),
        "rows_present": present,
        "rows_missing": [queue for queue in queue_ids if queue not in queues],
        "all_rows_present": len(present) == len(queue_ids),
        "native_count_queue_ids": [
            queue
            for queue in queue_ids
            if (queues.get(queue) or {}).get("count_source") == "cluster_metrics"
        ],
        "native_activity_queue_ids": [
            queue
            for queue in queue_ids
            if (queues.get(queue) or {}).get("jobs_passed_source") == "queue_native_metrics"
            and (queues.get(queue) or {}).get("jobs_failed_source") == "queue_native_metrics"
        ],
        "monitoring_scope": "annotation_only_general_queue_monitoring_is_retained",
    }


def _selected_waits_source(queues: dict[str, dict]) -> str:
    selected = {
        row.get(f"{metric}_wait_source")
        for row in queues.values()
        for metric in ("p50", "p95")
        if row.get(f"{metric}_wait") is not None
    }
    selected.discard(None)
    if not selected:
        return "none"
    if selected == {"sample_wait"}:
        return "scheduled_jobs"
    if selected == {"official_wait"}:
        return "cluster_metrics"
    return "mixed"


def _wait_field_descriptions() -> dict:
    return {
        "official_wait": (
            "Buildkite queue-native waitTimeSec converted to minutes; contains min, p50, p95, and max."
        ),
        "sample_wait": (
            "Exact statistics in minutes from fetched, currently SCHEDULED, non-zombie jobs; "
            "available records whether that queue's jobs were fetched, and count is null when they were not."
        ),
        "current_wait": "Displayed p50, p95, and p99 values paired with their per-field source labels.",
        "min_wait": "official_wait.min when Buildkite reports it, otherwise null.",
        "p50_wait": (
            "official_wait.p50 when available, otherwise a count-reconciled sample_wait.p50, "
            "otherwise null."
        ),
        "p95_wait": (
            "official_wait.p95 when available, otherwise a count-reconciled sample_wait.p95, "
            "otherwise null."
        ),
        "p99_wait": "sample_wait.p99 only when scheduled-job counts reconcile, otherwise null.",
        "p75_wait_p90_wait_avg_wait": (
            "Sample-only compatibility fields; null unless scheduled-job counts reconcile."
        ),
        "max_wait": (
            "Greater of official_wait.max and a count-reconciled sample_wait.max, otherwise null."
        ),
        "field_source_labels": (
            "Each root wait field has a matching *_wait_source value of official_wait, sample_wait, or null."
        ),
        "sample_reconciliation": (
            "wait_sample_reconciliation compares scheduled-job counts only; native wait values "
            "never participate. wait_sample_complete means count coverage reconciled, not that "
            "independent reads had identical job membership. Reconstructed root wait fields are "
            "promoted only when that count reconciliation succeeds. Detailed reconciliation is "
            "stored only for a nonzero reference/sample or a mismatch."
        ),
    }


def _metric_field_descriptions() -> dict:
    return {
        "provider": "Buildkite GraphQL ClusterQueue.metrics",
        "observed_at_field": "metrics_ts",
        "provider_fields": {
            "waiting": "ClusterQueue.metrics.waitingJobsCount",
            "running": "ClusterQueue.metrics.runningJobsCount",
            "connected_agents": "ClusterQueue.metrics.connectedAgentsCount",
            "jobs_passed": "ClusterQueue.metrics.jobsPassedCount",
            "jobs_failed": "ClusterQueue.metrics.jobsFailedCount",
            "official_wait.min": "ClusterQueue.metrics.waitTimeSec.min",
            "official_wait.p50": "ClusterQueue.metrics.waitTimeSec.p50",
            "official_wait.p95": "ClusterQueue.metrics.waitTimeSec.p95",
            "official_wait.max": "ClusterQueue.metrics.waitTimeSec.max",
        },
        "source_fields": {
            "waiting": "waiting_source",
            "running": "running_source",
            "connected_agents": "connected_agents_source",
            "jobs_passed": "jobs_passed_source",
            "jobs_failed": "jobs_failed_source",
            "official_wait.*": "official_wait_source",
        },
        "notes": (
            "waiting/running may instead use a fully paginated active-job scan, as named by "
            "their source fields. Activity counts are the provider-defined latest metrics point, "
            "not reconstructed lifecycle totals. Null values remain null rather than becoming zero."
        ),
    }


def normalize_history_snapshot(snapshot: dict) -> dict | None:
    """Migrate one queue snapshot without inventing unavailable measurements."""
    if not isinstance(snapshot, dict) or parse_iso(snapshot.get("ts") or "") is None:
        return None
    queues = snapshot.get("queues")
    if not isinstance(queues, dict):
        return None

    original_sources = snapshot.get("sources") if isinstance(snapshot.get("sources"), dict) else {}
    retain_metric_contract = isinstance(original_sources.get("metric_fields"), dict)
    retain_target_contract = "target_queue_scope" in original_sources or isinstance(
        snapshot.get("target_queue_scope"), dict
    )
    retain_native_activity_source = "native_activity" in original_sources
    history_provenance = (
        original_sources.get("history_provenance")
        if isinstance(original_sources.get("history_provenance"), dict)
        else {}
    )
    already_migrated = history_provenance.get("migration") == "legacy_queue_snapshot_v1_to_v2"
    row_is_legacy = {queue: not _queue_row_has_current_schema(row) for queue, row in queues.items()}
    # A single sparse/archive-only queue must not downgrade every otherwise
    # typed row in the snapshot to the legacy migration path.
    legacy = bool(row_is_legacy) and all(row_is_legacy.values())
    historical = legacy or already_migrated
    snapshot_count_source = str(original_sources.get("counts") or "unknown")
    snapshot_active_jobs_source = str(original_sources.get("active_jobs") or "active_job_scan")
    normalized_queues = {
        queue: _normalize_queue_row(
            row,
            snapshot_count_source,
            snapshot_active_jobs_source,
            legacy=row_is_legacy.get(queue, True),
        )
        for queue, row in sorted(queues.items())
        if not is_excluded_queue(queue)
    }

    normalized = dict(snapshot)
    normalized["schema_version"] = 2
    normalized["queues"] = normalized_queues
    normalized["total_waiting"] = sum(row["waiting"] for row in normalized_queues.values())
    normalized["total_running"] = sum(row["running"] for row in normalized_queues.values())
    normalized["total_zombie_waiting"] = sum(
        row["zombie_waiting"] for row in normalized_queues.values()
    )
    normalized["total_zombie_running"] = sum(
        row["zombie_running"] for row in normalized_queues.values()
    )
    scope_totals = {
        "all": _scope_totals(normalized_queues),
        "amd": _scope_totals(
            {queue: row for queue, row in normalized_queues.items() if is_amd_queue(queue)}
        ),
    }
    if retain_target_contract:
        scope_totals["target"] = _scope_totals(
            {
                queue: normalized_queues[queue]
                for queue in AMD_METRIC_TARGET_QUEUES
                if queue in normalized_queues
            }
        )
        normalized["target_queue_scope"] = _target_queue_scope(normalized_queues)
    else:
        normalized.pop("target_queue_scope", None)
    normalized["scope_totals"] = scope_totals

    sources = dict(original_sources)
    sources["wait_fields"] = _wait_field_descriptions()
    if retain_metric_contract:
        sources["metric_fields"] = _metric_field_descriptions()
    if retain_target_contract:
        sources["target_queue_scope"] = (
            "Canonical MI250/MI300/MI355 queues at widths 1/2/4/8; annotations and target totals "
            "do not remove the dashboard's general queue monitoring."
        )
    sources["workload_split_fields"] = {
        "source": "Fetched active jobs, independent of queue-native metric timing.",
        "rule": (
            "Retain observed workload counts only when their sum does not exceed "
            "the authoritative queue total. Partial splits remain partial; no "
            "remainder is assigned. Over-limit splits are null and their raw "
            "evidence is preserved in the matching *_provenance field."
        ),
    }
    sources["history_reset_ts"] = queue_history_reset_datetime().strftime("%Y-%m-%dT%H:%M:%SZ")
    sources["zombie_threshold_min"] = QUEUE_ZOMBIE_THRESHOLD_MIN
    if legacy:
        original_count_sources = sorted(
            {
                row.get("count_provenance", {}).get("original_source", "unknown")
                for row in normalized_queues.values()
            }
        )
        has_samples = any(
            (row.get("sample_wait") or {}).get("count", 0) > 0 for row in normalized_queues.values()
        )
        sources.update(
            {
                "counts": "historical_counts",
                "agents": "unavailable",
                "official_wait": "unavailable",
                "sampled_wait": (
                    "historical_scheduled_job_sample" if has_samples else "unavailable"
                ),
                "waits": "sampled_historical_jobs" if has_samples else "none",
                "history_provenance": {
                    "migration": "legacy_queue_snapshot_v1_to_v2",
                    "counts": "Preserved running/waiting counts only.",
                    "original_count_sources": original_count_sources,
                    "agents": "Unavailable in the migrated contract.",
                    "official_wait": "Unavailable; legacy zero/default values were not retained.",
                    "sampled_wait": "Retained only where wait_sample_count was greater than zero.",
                },
            }
        )
        if retain_native_activity_source:
            sources["native_activity"] = "unavailable"
    elif historical:
        has_samples = any(
            (row.get("sample_wait") or {}).get("count", 0) > 0 for row in normalized_queues.values()
        )
        sources.update(
            {
                "counts": "historical_counts",
                "agents": "unavailable",
                "official_wait": "unavailable",
                "sampled_wait": (
                    "historical_scheduled_job_sample" if has_samples else "unavailable"
                ),
                "waits": "sampled_historical_jobs" if has_samples else "none",
            }
        )
        if retain_native_activity_source:
            sources["native_activity"] = "unavailable"
    else:
        has_agents = any(row.get("connected_agents_source") for row in normalized_queues.values())
        has_native_activity = any(
            row.get("jobs_passed_source") or row.get("jobs_failed_source")
            for row in normalized_queues.values()
        )
        has_official = any(row.get("official_wait_source") for row in normalized_queues.values())
        has_sample_scan = any(row.get("sample_wait_source") for row in normalized_queues.values())
        sources["agents"] = "queue_native_metrics" if has_agents else "unavailable"
        if retain_native_activity_source:
            sources["native_activity"] = (
                "queue_native_metrics" if has_native_activity else "unavailable"
            )
        sources["official_wait"] = "queue_native_metrics" if has_official else "unavailable"
        sources["sampled_wait"] = (
            str(sources.get("active_jobs") or "active_job_scan")
            if has_sample_scan
            else "unavailable"
        )
        sources["waits"] = _selected_waits_source(normalized_queues)
    normalized["sources"] = sources
    return normalized


def _read_history_text(text: str) -> tuple[int, list[dict]]:
    total = 0
    snapshots: list[dict] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        total += 1
        try:
            snapshot = json.loads(line)
        except json.JSONDecodeError:
            continue
        normalized = normalize_history_snapshot(snapshot)
        if normalized is not None:
            snapshots.append(normalized)
    return total, snapshots


def _read_history_file(path: Path) -> tuple[int, list[dict]]:
    if not path.exists():
        return 0, []
    return _read_history_text(path.read_text())


def normalize_history_rows(rows: list[dict]) -> list[dict]:
    """Normalize, de-duplicate by timestamp, and sort snapshots deterministically."""
    by_timestamp: dict[str, dict] = {}
    for snapshot in rows:
        normalized = normalize_history_snapshot(snapshot)
        if normalized is not None:
            previous = by_timestamp.get(normalized["ts"])
            by_timestamp[normalized["ts"]] = (
                _merge_same_timestamp_snapshots(previous, normalized)
                if previous is not None
                else normalized
            )
    return [by_timestamp[ts] for ts in sorted(by_timestamp)]


def _merge_same_timestamp_snapshots(previous: dict, current: dict) -> dict:
    """Let current values win while retaining richer hourly peak evidence."""
    merged = deepcopy(current)
    if (
        previous.get("history_mode") == "hourly_queue_wait_peaks"
        or current.get("history_mode") == "hourly_queue_wait_peaks"
    ):
        merged["history_mode"] = "hourly_queue_wait_peaks"
        merged["archive_bucket_start"] = current.get("archive_bucket_start") or previous.get(
            "archive_bucket_start"
        )

    merged_queues = merged.setdefault("queues", {})
    queue_names = set((previous.get("queues") or {})) | set((current.get("queues") or {}))
    for name in queue_names:
        for peak_field in ("archive_wait_peaks", "archive_sample_wait_peaks"):
            peak_candidates: dict[str, list[dict]] = {}
            for snapshot in (previous, current):
                row = (snapshot.get("queues") or {}).get(name)
                for metric, peak in ((row or {}).get(peak_field) or {}).items():
                    if isinstance(peak, dict) and _as_optional_float(peak.get("value")) is not None:
                        peak_candidates.setdefault(metric, []).append(peak)
            if not peak_candidates:
                continue
            target = merged_queues.get(name)
            if not isinstance(target, dict):
                target = deepcopy((previous.get("queues") or {}).get(name) or {})
                merged_queues[name] = target
            peaks = dict(target.get(peak_field) or {})
            for metric, candidates in peak_candidates.items():
                peaks[metric] = deepcopy(
                    max(
                        candidates,
                        key=lambda peak: (
                            _as_optional_float(peak.get("value")),
                            str(peak.get("observed_at") or ""),
                        ),
                    )
                )
            target[peak_field] = peaks

    previous_resolution = (previous.get("sources") or {}).get("history_resolution")
    if previous_resolution and not (merged.get("sources") or {}).get("history_resolution"):
        sources = dict(merged.get("sources") or {})
        sources["history_resolution"] = previous_resolution
        merged["sources"] = sources
    return merged


def _snapshot_peak_wait(snapshot: dict) -> float:
    values = [
        _as_optional_float(row.get("p95_wait"))
        for row in (snapshot.get("queues") or {}).values()
        if isinstance(row, dict)
    ]
    return max((value for value in values if value is not None), default=-1.0)


def _archive_bucket_snapshot(snapshots: list[dict]) -> dict:
    """Retain one coherent snapshot plus each queue's exact hourly wait peaks."""
    representative = max(
        snapshots,
        key=lambda snapshot: (
            _snapshot_peak_wait(snapshot),
            _as_count(snapshot.get("total_waiting")),
            snapshot["ts"],
        ),
    )
    archived = deepcopy(representative)
    archived["history_mode"] = "hourly_queue_wait_peaks"
    archived["archive_bucket_start"] = (
        parse_iso(representative["ts"])
        .replace(minute=0, second=0, microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )

    queue_names = sorted(
        {name for snapshot in snapshots for name in (snapshot.get("queues") or {})}
    )
    archived_queues = archived.setdefault("queues", {})
    for name in queue_names:
        peaks = {}
        sample_peaks = {}
        for metric in ("p50", "p95", "p99"):
            candidates = []
            sample_candidates = []
            for snapshot in snapshots:
                row = (snapshot.get("queues") or {}).get(name)
                if not isinstance(row, dict):
                    continue
                existing_peak = (row.get("archive_wait_peaks") or {}).get(metric)
                if isinstance(existing_peak, dict):
                    existing_value = _as_optional_float(existing_peak.get("value"))
                    if existing_value is not None:
                        candidates.append(
                            (
                                existing_value,
                                str(existing_peak.get("observed_at") or snapshot["ts"]),
                                existing_peak.get("source"),
                                existing_peak.get("provider"),
                                existing_peak.get("sample_count"),
                                existing_peak.get("sample_expected"),
                                existing_peak.get("sample_complete"),
                            )
                        )
                        if existing_peak.get("source") == "sample_wait":
                            sample_candidates.append(candidates[-1])
                existing_sample_peak = (row.get("archive_sample_wait_peaks") or {}).get(metric)
                if isinstance(existing_sample_peak, dict):
                    existing_sample_value = _as_optional_float(existing_sample_peak.get("value"))
                    if existing_sample_value is not None:
                        sample_candidates.append(
                            (
                                existing_sample_value,
                                str(existing_sample_peak.get("observed_at") or snapshot["ts"]),
                                "sample_wait",
                                existing_sample_peak.get("provider")
                                or existing_sample_peak.get("source"),
                                existing_sample_peak.get("sample_count"),
                                existing_sample_peak.get("sample_expected"),
                                existing_sample_peak.get("sample_complete"),
                            )
                        )
                value = _as_optional_float(row.get(f"{metric}_wait"))
                if value is None:
                    source = None
                else:
                    source = row.get(f"{metric}_wait_source")
                    provider = (
                        row.get("official_wait_source")
                        if source == "official_wait"
                        else row.get("sample_wait_source")
                        if source == "sample_wait"
                        else None
                    )
                    candidates.append(
                        (
                            value,
                            snapshot["ts"],
                            source,
                            provider,
                            row.get("wait_sample_count") if source == "sample_wait" else None,
                            row.get("wait_sample_expected_count")
                            if source == "sample_wait"
                            else None,
                            row.get("wait_sample_complete") if source == "sample_wait" else None,
                        )
                    )
                sample_value = _as_optional_float((row.get("sample_wait") or {}).get(metric))
                if sample_value is not None:
                    sample_candidates.append(
                        (
                            sample_value,
                            snapshot["ts"],
                            "sample_wait",
                            row.get("sample_wait_source"),
                            row.get("wait_sample_count"),
                            row.get("wait_sample_expected_count"),
                            row.get("wait_sample_complete"),
                        )
                    )
            if candidates:
                (
                    value,
                    observed_at,
                    source,
                    provider,
                    sample_count,
                    sample_expected,
                    sample_complete,
                ) = max(candidates, key=lambda item: (item[0], item[1]))
                peaks[metric] = {
                    "value": value,
                    "observed_at": observed_at,
                    "source": source,
                    "provider": provider,
                    "sample_count": sample_count,
                    "sample_expected": sample_expected,
                    "sample_complete": sample_complete,
                }
            if sample_candidates:
                (
                    sample_value,
                    sample_observed_at,
                    _,
                    sample_provider,
                    sample_count,
                    sample_expected,
                    sample_complete,
                ) = max(sample_candidates, key=lambda item: (item[0], item[1]))
                sample_peaks[metric] = {
                    "value": sample_value,
                    "observed_at": sample_observed_at,
                    "source": "sample_wait",
                    "provider": sample_provider,
                    "sample_count": sample_count,
                    "sample_expected": sample_expected,
                    "sample_complete": sample_complete,
                }
        if not peaks and not sample_peaks:
            continue
        target = archived_queues.get(name)
        if not isinstance(target, dict):
            target = {
                "waiting": 0,
                "running": 0,
                "count_source": "unavailable",
                "history_observation_only": True,
            }
            archived_queues[name] = target
        if peaks:
            target["archive_wait_peaks"] = peaks
        if sample_peaks:
            target["archive_sample_wait_peaks"] = sample_peaks

    sources = dict(archived.get("sources") or {})
    sources["history_resolution"] = (
        "One actual snapshot per UTC hour plus per-queue primary and reconstructed "
        "sample p50/p95/p99 peaks with their exact observation timestamps."
    )
    archived["sources"] = sources
    return archived


def compact_history_resolution(rows: list[dict], now: datetime) -> list[dict]:
    """Keep recent polls and per-queue wait peaks in older hourly envelopes."""
    normalized = normalize_history_rows(rows)
    high_resolution_cutoff = now - timedelta(hours=QUEUE_HISTORY_HIGH_RES_HOURS)
    recent: list[dict] = []
    archive_buckets: dict[datetime, list[dict]] = {}

    for snapshot in normalized:
        observed_at = parse_iso(snapshot.get("ts") or "")
        if observed_at is None:
            continue
        if observed_at >= high_resolution_cutoff:
            recent.append(snapshot)
            continue
        minute = (
            observed_at.minute // QUEUE_HISTORY_ARCHIVE_BUCKET_MINUTES
        ) * QUEUE_HISTORY_ARCHIVE_BUCKET_MINUTES
        bucket = observed_at.replace(minute=minute, second=0, microsecond=0)
        archive_buckets.setdefault(bucket, []).append(snapshot)

    archive = [_archive_bucket_snapshot(bucket) for bucket in archive_buckets.values()]
    return normalize_history_rows([*archive, *recent])


def write_history_file(path: Path, rows: list[dict]) -> None:
    normalized = normalize_history_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in normalized
    )
    path.write_text(text)


def merge_history_rows(path: Path, incoming_rows: list[dict]) -> tuple[int, int]:
    """Merge incoming history with local rows; local rows win equal timestamps."""
    _, existing_rows = _read_history_file(path)
    merged = normalize_history_rows([*incoming_rows, *existing_rows])
    write_history_file(path, merged)
    return len(incoming_rows), len(merged)


def merge_history_from_git_ref(path: Path, git_ref: str) -> tuple[int, int]:
    """Merge queue history from a git ref without line-count replacement."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    result = subprocess.run(
        ["git", "show", f"{git_ref}:{HISTORY_REPO_PATH}"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        log.warning("No queue history available at %s", git_ref)
        _, existing = _read_history_file(path)
        return 0, len(existing)
    if any(marker in result.stdout for marker in ("<<<<<<<", "=======", ">>>>>>>")):
        log.warning("Queue history at %s contains conflict markers; ignoring it", git_ref)
        _, existing = _read_history_file(path)
        return 0, len(existing)

    incoming_total, incoming_rows = _read_history_text(result.stdout)
    incoming_count, merged_count = merge_history_rows(path, incoming_rows)
    log.info(
        "Merged queue history from %s: %d parsed of %d lines, %d total rows",
        git_ref,
        incoming_count,
        incoming_total,
        merged_count,
    )
    return incoming_count, merged_count


def prune_history_file(path: Path, now: datetime | None = None) -> tuple[int, int]:
    """Migrate, retain recent detail, and preserve hourly historical spikes."""
    total, snapshots = _read_history_file(path)
    if total == 0 and not path.exists():
        return 0, 0

    current_time = now or datetime.now(timezone.utc)
    cutoff = _history_cutoff(current_time)
    kept = [snapshot for snapshot in snapshots if parse_iso(snapshot["ts"]) >= cutoff]
    compacted = compact_history_resolution(kept, current_time)
    write_history_file(path, compacted)
    return total, len(compacted)


def _wait_summary(times: list[float]) -> dict:
    """Return exact observed-sample statistics in minutes."""
    if not times:
        return {
            "p50": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
            "avg": None,
        }
    ordered = sorted(times)
    return {
        "p50": round(percentile(ordered, 50), 1),
        "p75": round(percentile(ordered, 75), 1),
        "p90": round(percentile(ordered, 90), 1),
        "p95": round(percentile(ordered, 95), 1),
        "p99": round(percentile(ordered, 99), 1),
        "max": round(max(ordered), 1),
        "avg": round(sum(ordered) / len(ordered), 1),
    }


def _minutes_from_seconds(value) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value) / 60.0, 1)
    except (TypeError, ValueError):
        return None


def _wait_summary_from_queue_metrics(wait_time_sec: dict | None) -> dict | None:
    """Return only wait statistics Buildkite reports natively."""
    if not isinstance(wait_time_sec, dict):
        return None
    min_wait = _minutes_from_seconds(wait_time_sec.get("min"))
    p50 = _minutes_from_seconds(wait_time_sec.get("p50"))
    p95 = _minutes_from_seconds(wait_time_sec.get("p95"))
    max_wait = _minutes_from_seconds(wait_time_sec.get("max"))
    if min_wait is None and p50 is None and p95 is None and max_wait is None:
        return None

    return {
        "min": min_wait,
        "p50": p50,
        "p95": p95,
        "max": max_wait,
    }


def _make_canvas_job_url(build_url: str, job_uuid: str, fallback_url: str = "") -> str:
    if build_url and job_uuid:
        return f"{build_url}/steps/canvas?jid={job_uuid}&tab=output"
    return _rewrite_job_url(fallback_url)


def _wait_minutes(
    now: datetime, runnable_at: str | None, scheduled_at: str | None, created_at: str | None
) -> float:
    anchor = parse_iso(runnable_at) or parse_iso(scheduled_at) or parse_iso(created_at)
    if anchor is None:
        return 0.0
    return (now - anchor).total_seconds() / 60


def _started_wait_minutes(
    runnable_at: str | None,
    scheduled_at: str | None,
    created_at: str | None,
    started_at: str | None,
) -> float | None:
    anchor = parse_iso(runnable_at) or parse_iso(scheduled_at) or parse_iso(created_at)
    started = parse_iso(started_at)
    if anchor is None or started is None:
        return None
    return round((started - anchor).total_seconds() / 60, 1)


def _run_minutes(now: datetime, started_at: str | None) -> float | None:
    started = parse_iso(started_at)
    if started is None:
        return None
    return round((now - started).total_seconds() / 60, 1)


def fetch_cluster_queue_metrics(token: str) -> dict[str, dict]:
    """Fetch queue-native counts from Buildkite cluster metrics."""
    metrics: dict[str, dict] = {}
    after = None
    while True:
        data = bk_graphql(
            GRAPHQL_QUEUE_METRICS_Q,
            token,
            {"org": BK_ORG, "cluster": BK_CLUSTER_UUID, "first": GRAPHQL_PAGE_SIZE, "after": after},
        )
        cluster = (data.get("organization") or {}).get("cluster") or {}
        queues = cluster.get("queues") or {}
        for edge in queues.get("edges") or []:
            node = edge.get("node") or {}
            key = node.get("key") or ""
            if not key or is_excluded_queue(key):
                continue
            latest = node.get("metrics") or {}
            waiting_count = latest.get("waitingJobsCount")
            running_count = latest.get("runningJobsCount")
            connected_agents = latest.get("connectedAgentsCount")
            jobs_passed = latest.get("jobsPassedCount")
            jobs_failed = latest.get("jobsFailedCount")
            metrics[key] = {
                "graphql_id": node.get("id") or "",
                "counts_available": waiting_count is not None and running_count is not None,
                "waiting": _as_count(waiting_count),
                "running": _as_count(running_count),
                "connected_agents": (
                    _as_count(connected_agents) if connected_agents is not None else None
                ),
                "jobs_passed": _as_optional_count(jobs_passed),
                "jobs_failed": _as_optional_count(jobs_failed),
                "official_wait": _wait_summary_from_queue_metrics(latest.get("waitTimeSec")),
                "metrics_ts": latest.get("timestamp") or "",
                "queue_url": _queue_web_url(node.get("uuid")),
                "dispatch_paused": bool(node.get("dispatchPaused")),
            }
        page = queues.get("pageInfo") or {}
        if not page.get("hasNextPage"):
            return metrics
        after = page.get("endCursor")


def _graphql_job_record(node: dict, fallback_queue: str = "") -> dict | None:
    state = node.get("state") or ""
    queue = (
        ((node.get("clusterQueue") or {}).get("key"))
        or fallback_queue
        or queue_from_rules(node.get("agentQueryRules"))
    )
    if not queue or is_excluded_queue(queue):
        return None
    build = node.get("build") or {}
    pipeline = node.get("pipeline") or {}
    return {
        "queue": queue,
        "state": state,
        "name": node.get("label") or "",
        "job_uuid": node.get("uuid") or "",
        "build_url": build.get("url") or "",
        "pipeline": pipeline.get("slug") or "",
        "build": build.get("number") or 0,
        "branch": build.get("branch") or "",
        "commit": (build.get("commit") or "")[:12],
        "workload": classify_workload(pipeline.get("slug") or "", build.get("branch") or "", queue),
        "fork_url": "",
        "source": "",
        "runnable_at": node.get("runnableAt"),
        "scheduled_at": node.get("scheduledAt"),
        "created_at": node.get("createdAt"),
        "started_at": node.get("startedAt"),
    }


def _deduplicate_active_jobs(records: list[dict]) -> list[dict]:
    """Deduplicate overlapping scans by job UUID, preferring the later state.

    REST build-state scans are sequential, so the same build can appear in
    both responses while it transitions. GraphQL supplemental reads can also
    overlap a queue-scoped read. Only a stable job UUID is safe to collapse;
    unidentified records remain separate rather than risking a false merge.
    """
    deduplicated: list[dict] = []
    positions: dict[str, int] = {}

    def state_rank(record: dict) -> int:
        state = str(record.get("state") or "").upper()
        if state in GRAPHQL_RUNNING_STATES or state.lower() in LEGACY_RUNNING_STATES:
            return 2
        if state in GRAPHQL_WAITING_STATES or state.lower() in LEGACY_WAITING_STATES:
            return 1
        return 0

    for record in records:
        job_uuid = str(record.get("job_uuid") or "").strip()
        if not job_uuid:
            deduplicated.append(record)
            continue
        position = positions.get(job_uuid)
        if position is None:
            positions[job_uuid] = len(deduplicated)
            deduplicated.append(record)
            continue
        if state_rank(record) >= state_rank(deduplicated[position]):
            deduplicated[position] = record
    return deduplicated


def _fetch_graphql_jobs(
    token: str,
    *,
    query: str,
    variables: dict,
    fallback_queue: str = "",
) -> list[dict]:
    jobs: list[dict] = []
    after = None
    while True:
        page_vars = dict(variables)
        page_vars["after"] = after
        data = bk_graphql(
            query,
            token,
            page_vars,
        )
        conn = (data.get("organization") or {}).get("jobs") or {}
        for edge in conn.get("edges") or []:
            node = edge.get("node") or {}
            record = _graphql_job_record(node, fallback_queue)
            if record:
                jobs.append(record)
        page = conn.get("pageInfo") or {}
        if not page.get("hasNextPage"):
            return jobs
        after = page.get("endCursor")


def fetch_active_cluster_jobs(
    token: str, queue_ids_by_key: dict[str, str] | None = None
) -> list[dict]:
    """Fetch active command jobs via GraphQL.

    Buildkite's queue metrics API accepts a cluster UUID, but the jobs API
    expects a GraphQL cluster-queue ID. Querying active jobs per queue keeps
    wait samples aligned with the queue-native backlog counts.
    """
    if queue_ids_by_key:
        jobs: list[dict] = []
        for queue, queue_id in sorted(queue_ids_by_key.items()):
            if not queue_id or is_excluded_queue(queue):
                continue
            jobs.extend(
                _fetch_graphql_jobs(
                    token,
                    query=GRAPHQL_QUEUE_JOBS_Q,
                    variables={
                        "org": BK_ORG,
                        "queue": [queue_id],
                        "states": list(GRAPHQL_ACTIVE_STATES),
                        "first": GRAPHQL_PAGE_SIZE,
                    },
                    fallback_queue=queue,
                )
            )
        return jobs

    return _fetch_graphql_jobs(
        token,
        query=GRAPHQL_ACTIVE_JOBS_Q,
        variables={
            "org": BK_ORG,
            "states": list(GRAPHQL_ACTIVE_STATES),
            "first": GRAPHQL_PAGE_SIZE,
        },
    )


def _collect_legacy_active_jobs(token: str) -> list[dict]:
    """Legacy fallback that scans active builds from the REST API."""
    records: list[dict] = []
    for state in ("running", "scheduled"):
        builds = bk_get_paginated(f"/organizations/{BK_ORG}/builds", token, {"state": state})
        log.info("Fetched %d %s builds", len(builds), state)

        for build in builds:
            build_branch = build.get("branch", "") or ""
            build_commit = (build.get("commit", "") or "")[:12]
            build_source = build.get("source", "") or ""
            pr = build.get("pull_request") or {}
            fork_url = pr.get("repository") or ""
            pipeline_slug = (build.get("pipeline") or {}).get("slug", "")
            build_url = build.get("web_url", "") or ""

            for job in build.get("jobs", []):
                if job.get("type") != "script":
                    continue
                queue = queue_from_rules(job.get("agent_query_rules"))
                if not queue or is_excluded_queue(queue):
                    continue

                job_state = (job.get("state", "") or "").lower()
                if (
                    job_state not in LEGACY_WAITING_STATES
                    and job_state not in LEGACY_RUNNING_STATES
                ):
                    continue

                records.append(
                    {
                        "queue": queue,
                        "state": job_state.upper(),
                        "name": job.get("name", "") or "",
                        "job_uuid": job.get("id", "") or "",
                        "build_url": build_url,
                        "pipeline": pipeline_slug,
                        "build": build.get("number", 0),
                        "branch": build_branch,
                        "commit": build_commit,
                        "workload": classify_workload(pipeline_slug, build_branch, queue),
                        "fork_url": fork_url,
                        "source": build_source,
                        "runnable_at": job.get("runnable_at"),
                        "scheduled_at": job.get("scheduled_at"),
                        "created_at": job.get("created_at"),
                        "started_at": job.get("started_at"),
                        "fallback_url": job.get("web_url", "") or "",
                    }
                )
    return _deduplicate_active_jobs(records)


def _seed_queue_metrics(queue_stats: dict, metrics_by_queue: dict[str, dict]) -> None:
    for queue, meta in metrics_by_queue.items():
        if is_excluded_queue(queue):
            continue
        stats = queue_stats[queue]
        if meta.get("counts_available", True):
            stats["waiting"] = _as_count(meta.get("waiting"))
            stats["running"] = _as_count(meta.get("running"))
            stats["scheduled"] = stats["waiting"]
            stats["total"] = stats["waiting"] + stats["running"]
            stats["count_source"] = "cluster_metrics"
        if meta.get("connected_agents") is not None:
            stats["connected_agents"] = _as_count(meta.get("connected_agents"))
            stats["connected_agents_source"] = "queue_native_metrics"
        if meta.get("jobs_passed") is not None:
            stats["jobs_passed"] = _as_count(meta.get("jobs_passed"))
            stats["jobs_passed_source"] = "queue_native_metrics"
        if meta.get("jobs_failed") is not None:
            stats["jobs_failed"] = _as_count(meta.get("jobs_failed"))
            stats["jobs_failed_source"] = "queue_native_metrics"
        if meta.get("queue_url"):
            stats["queue_url"] = meta["queue_url"]
        if meta.get("metrics_ts"):
            stats["metrics_ts"] = meta["metrics_ts"]
        if meta.get("dispatch_paused"):
            stats["dispatch_paused"] = True
        if meta.get("official_wait"):
            stats["official_wait"] = dict(meta["official_wait"])


def _apply_active_jobs(
    now: datetime,
    queue_stats: dict,
    active_jobs: list[dict],
    trusted_count_queues: set[str],
) -> tuple[list[dict], list[dict]]:
    pending_jobs: list[dict] = []
    running_jobs: list[dict] = []

    for job in active_jobs:
        queue = job.get("queue") or ""
        if not queue or is_excluded_queue(queue):
            continue

        stats = queue_stats[queue]
        trust_counts = queue in trusted_count_queues
        state = job.get("state") or ""
        is_waiting = state in GRAPHQL_WAITING_STATES
        is_running = state in GRAPHQL_RUNNING_STATES or state.lower() in LEGACY_RUNNING_STATES
        if not is_waiting and not is_running:
            continue

        workload = job.get("workload") or "vllm"
        build_url = job.get("build_url") or ""
        web_url = _make_canvas_job_url(
            build_url, job.get("job_uuid") or "", job.get("fallback_url", "")
        )
        queue_wait_before_start = _started_wait_minutes(
            job.get("runnable_at"),
            job.get("scheduled_at"),
            job.get("created_at"),
            job.get("started_at"),
        )

        if is_waiting:
            wait_mins = round(
                _wait_minutes(
                    now, job.get("runnable_at"), job.get("scheduled_at"), job.get("created_at")
                ),
                1,
            )
            is_zombie = wait_mins >= QUEUE_ZOMBIE_THRESHOLD_MIN
            if is_zombie:
                stats["zombie_waiting"] = int(stats.get("zombie_waiting") or 0) + 1
            else:
                if not trust_counts:
                    stats["waiting"] += 1
                    stats["scheduled"] += 1
                    stats["total"] += 1
                stats.setdefault("waiting_by_workload", {"vllm": 0, "omni": 0})
                stats["waiting_by_workload"][workload] += 1
                stats["wait_times"].append(wait_mins)

            pending_jobs.append(
                {
                    "name": job.get("name") or "",
                    "queue": queue,
                    "state": "scheduled",
                    "wait_min": wait_mins,
                    "analysis_excluded": is_zombie,
                    "exclusion_reason": "zombie_wait" if is_zombie else "",
                    "zombie_threshold_min": QUEUE_ZOMBIE_THRESHOLD_MIN,
                    "url": web_url,
                    "pipeline": job.get("pipeline") or "",
                    "build": job.get("build") or 0,
                    "branch": job.get("branch") or "",
                    "commit": job.get("commit") or "",
                    "workload": workload,
                    "fork_url": job.get("fork_url") or "",
                    "source": job.get("source") or "",
                    "queue_url": stats.get("queue_url") or "",
                }
            )
            continue

        run_mins = _run_minutes(now, job.get("started_at"))
        is_zombie = (run_mins or 0) >= QUEUE_ZOMBIE_THRESHOLD_MIN
        if is_zombie:
            stats["zombie_running"] = int(stats.get("zombie_running") or 0) + 1
        else:
            if not trust_counts:
                stats["running"] += 1
                stats["total"] += 1
            stats.setdefault("running_by_workload", {"vllm": 0, "omni": 0})
            stats["running_by_workload"][workload] += 1
        running_jobs.append(
            {
                "name": job.get("name") or "",
                "queue": queue,
                "state": "running",
                "analysis_excluded": is_zombie,
                "exclusion_reason": "zombie_running" if is_zombie else "",
                "zombie_threshold_min": QUEUE_ZOMBIE_THRESHOLD_MIN,
                "url": web_url,
                "pipeline": job.get("pipeline") or "",
                "build": job.get("build") or 0,
                "branch": job.get("branch") or "",
                "commit": job.get("commit") or "",
                "workload": workload,
                "fork_url": job.get("fork_url") or "",
                "source": job.get("source") or "",
                "queue_wait_before_start_min": queue_wait_before_start,
                "run_min": run_mins,
                "queue_url": stats.get("queue_url") or "",
            }
        )

    return pending_jobs, running_jobs


def collect_snapshot(token: str) -> dict:
    """Collect the latest queue state using queue-native metrics when possible."""
    now = datetime.now(timezone.utc)
    prune_history_file(OUTPUT, now)
    queue_stats: dict = defaultdict(_queue_row)
    for queue in TRACKED_QUEUES:
        queue_stats[queue]

    metrics_by_queue: dict[str, dict] = {}
    counts_source = "active_job_scan"
    active_jobs_source = "legacy_build_scan"
    sampled_queues: set[str] | None = None

    try:
        metrics_by_queue = fetch_cluster_queue_metrics(token)
        _seed_queue_metrics(queue_stats, metrics_by_queue)
        if any(meta.get("counts_available", True) for meta in metrics_by_queue.values()):
            counts_source = "cluster_metrics"
    except Exception as exc:
        log.warning(
            "Buildkite cluster metrics unavailable, falling back to active job counts: %s", exc
        )

    active_queue_ids = {
        queue: str(meta.get("graphql_id") or "")
        for queue, meta in metrics_by_queue.items()
        if (
            not is_excluded_queue(queue)
            and meta.get("graphql_id")
            and (
                not meta.get("counts_available", True)
                or _as_count(meta.get("waiting"))
                or _as_count(meta.get("running"))
            )
        )
    }
    metrics_queue_keys = {
        queue.casefold() for queue in metrics_by_queue if not is_excluded_queue(queue)
    }
    missing_metrics_queues = {
        queue
        for queue in TRACKED_QUEUES
        if not is_excluded_queue(queue) and queue.casefold() not in metrics_queue_keys
    }

    try:
        if active_queue_ids and missing_metrics_queues:
            active_jobs = fetch_active_cluster_jobs(token, active_queue_ids)
            organization_jobs = fetch_active_cluster_jobs(token, None)
            missing_by_key = {queue.casefold(): queue for queue in missing_metrics_queues}
            for job in organization_jobs:
                canonical_queue = missing_by_key.get(str(job.get("queue") or "").casefold())
                if canonical_queue:
                    active_jobs.append({**job, "queue": canonical_queue})
            active_jobs_source = "cluster_queue_and_organization_jobs_graphql"
            sampled_queues = set(active_queue_ids) | missing_metrics_queues
        else:
            active_jobs = fetch_active_cluster_jobs(token, active_queue_ids or None)
            active_jobs_source = (
                "cluster_queue_graphql" if active_queue_ids else "organization_jobs_graphql"
            )
            sampled_queues = set(active_queue_ids) if active_queue_ids else None
    except Exception as exc:
        log.warning(
            "Buildkite GraphQL active jobs unavailable, falling back to build scan: %s", exc
        )
        active_jobs = _collect_legacy_active_jobs(token)
    active_jobs = _deduplicate_active_jobs(active_jobs)

    trusted_count_queues = {
        queue
        for queue, meta in metrics_by_queue.items()
        if not is_excluded_queue(queue) and meta.get("counts_available", True)
    }
    pending_jobs, running_jobs = _apply_active_jobs(
        now,
        queue_stats,
        active_jobs,
        trusted_count_queues,
    )

    queues = {}
    has_official_wait = False
    has_sample_wait = False
    has_agent_metrics = False
    has_native_activity = False
    for queue, stats in sorted(queue_stats.items()):
        if is_excluded_queue(queue):
            continue
        if queue not in TRACKED_QUEUES and not stats["waiting"] and not stats["running"]:
            continue
        row = {k: v for k, v in stats.items() if k not in {"wait_times", "official_wait"}}
        official_wait = stats.get("official_wait") or _empty_official_wait()
        sample_summary = _wait_summary(stats["wait_times"])
        sample_available = sampled_queues is None or queue in sampled_queues
        sample_wait = {
            "available": sample_available,
            "count": len(stats["wait_times"]) if sample_available else None,
            **sample_summary,
        }
        row["official_wait_source"] = (
            "queue_native_metrics"
            if any(value is not None for value in official_wait.values())
            else None
        )
        row["sample_wait_source"] = active_jobs_source if sample_available else None
        _apply_wait_contract(row, official_wait, sample_wait)
        _apply_metric_sources(row)
        has_official_wait = has_official_wait or any(
            value is not None for value in official_wait.values()
        )
        has_sample_wait = has_sample_wait or bool(sample_wait["count"])
        has_agent_metrics = has_agent_metrics or bool(row.get("connected_agents_source"))
        has_native_activity = has_native_activity or bool(
            row.get("jobs_passed_source") or row.get("jobs_failed_source")
        )
        queues[queue] = row

    queue_count_sources = {row["count_source"] for row in queues.values()}
    if len(queue_count_sources) == 1:
        counts_source = next(iter(queue_count_sources))
    elif queue_count_sources:
        counts_source = "mixed_queue_native_and_active_job_scan"

    snapshot = {
        "ts": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "queues": queues,
        "total_waiting": sum(int(s.get("waiting") or 0) for s in queues.values()),
        "total_running": sum(int(s.get("running") or 0) for s in queues.values()),
        "total_zombie_waiting": sum(int(s.get("zombie_waiting") or 0) for s in queues.values()),
        "total_zombie_running": sum(int(s.get("zombie_running") or 0) for s in queues.values()),
        "sources": {
            "counts": counts_source,
            "waits": _selected_waits_source(queues),
            "active_jobs": active_jobs_source,
            "agents": "queue_native_metrics" if has_agent_metrics else "unavailable",
            "native_activity": ("queue_native_metrics" if has_native_activity else "unavailable"),
            "official_wait": ("queue_native_metrics" if has_official_wait else "unavailable"),
            "sampled_wait": active_jobs_source if has_sample_wait else "unavailable",
            "count_fields": {
                "waiting_running_scheduled_total": (
                    "Each queue row uses Buildkite cluster metrics when count_source is cluster_metrics; "
                    "otherwise counts are derived from fetched active jobs. Queue-native counts include zombies."
                ),
                "zombie_waiting_zombie_running": (
                    "Derived from fetched active jobs and reported separately from queue-native counts."
                ),
            },
            "wait_fields": _wait_field_descriptions(),
            "metric_fields": _metric_field_descriptions(),
            "target_queue_scope": (
                "Canonical MI250/MI300/MI355 queues at widths 1/2/4/8; annotations and target "
                "totals do not remove the dashboard's general queue monitoring."
            ),
            "history_reset_ts": queue_history_reset_datetime().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "zombie_threshold_min": QUEUE_ZOMBIE_THRESHOLD_MIN,
        },
    }

    run_id = os.getenv("GITHUB_RUN_ID", "")
    if run_id:
        snapshot["run_id"] = run_id

    snapshot = normalize_history_snapshot(snapshot)
    if snapshot is None:
        raise RuntimeError("Generated queue snapshot failed schema normalization")

    jobs_data = {
        "ts": snapshot["ts"],
        "zombie_threshold_min": QUEUE_ZOMBIE_THRESHOLD_MIN,
        "pending": sorted(pending_jobs, key=lambda job: job.get("wait_min", 0), reverse=True),
        "running": running_jobs,
    }
    jobs_path = OUTPUT.parent / "queue_jobs.json"
    jobs_path.write_text(json.dumps(jobs_data, indent=2))
    log.info(
        "Wrote %d pending + %d running jobs to %s",
        len(pending_jobs),
        len(running_jobs),
        jobs_path,
    )

    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prune-only", action="store_true")
    parser.add_argument(
        "--merge-history-git-ref",
        metavar="REF",
        help="Merge queue history from REF by timestamp, then exit.",
    )
    args = parser.parse_args()

    if args.merge_history_git_ref:
        merge_history_from_git_ref(OUTPUT, args.merge_history_git_ref)
        return

    if args.prune_only:
        before, kept = prune_history_file(OUTPUT)
        log.info("Pruned queue history: %d -> %d rows", before, kept)
        return

    token = os.getenv("BUILDKITE_TOKEN")
    if not token:
        log.error("BUILDKITE_TOKEN not set")
        sys.exit(1)

    log.info("Collecting queue snapshot...")
    snapshot = collect_snapshot(token)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("a", encoding="utf-8") as history_file:
        history_file.write(json.dumps(snapshot, separators=(",", ":")) + "\n")
    prune_history_file(OUTPUT)

    log.info(
        "Snapshot: %d queues, %d waiting, %d running -> %s",
        len(snapshot["queues"]),
        snapshot["total_waiting"],
        snapshot["total_running"],
        OUTPUT,
    )

    for queue, stats in sorted(
        snapshot["queues"].items(), key=lambda item: item[1]["waiting"], reverse=True
    ):
        if stats["waiting"] > 0 or stats["running"] > 0:
            print(f"  {queue:30s} waiting={stats['waiting']:3d} running={stats['running']:3d}")


if __name__ == "__main__":
    main()
