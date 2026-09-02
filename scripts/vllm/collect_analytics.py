#!/usr/bin/env python3
"""Collect per-build, per-job analytics from Buildkite for the rich CI dashboard.

Produces:
- data/vllm/ci/builds_analytics.json — per-build summary with job matrix
- data/vllm/ci/jobs_analytics.json — per-job failure/duration rankings

Usage:
    export BUILDKITE_TOKEN="bkua_..."
    python scripts/vllm/collect_analytics.py --days 30
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vllm.constants import BK_API_BASE, BK_ORG  # noqa: E402
from vllm.ci.utils import (  # noqa: E402
    duration_mins,
    parse_iso as parse_ts,
    percentile,
    queue_from_rules as _queue_from_rules,
)
from vllm.ci.reliability_history import (  # noqa: E402
    BUILD_FETCH_MAX_PAGES,
    BUILD_FETCH_PAGE_SIZE,
    OBSERVATION_LIMIT,
    buildkite_job_url_matches,
    build_all_main_reliability,
    compact_main_builds,
    compute_nightly_change_history,
    filter_reliability_builds,
    validate_all_main_reliability,
)
from vllm.pipelines import NIGHTLY_NAME_PATTERNS_BY_SLUG  # noqa: E402
from vllm.ci import ratelimit  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

PIPELINES = {"amd-ci": "AMD CI", "ci": "Upstream CI"}
ANALYTICS_WINDOWS_DAYS = (1, 3, 7, 14, 30)
DEFAULT_ANALYTICS_WINDOW_DAYS = 30
ANALYTICS_BUILD_LIMIT = 120
ANALYTICS_NIGHTLY_LIMIT = 30
ANALYTICS_WINDOW_BUILD_LIMIT = 50
ANALYTICS_WINDOW_NIGHTLY_LIMIT = 30
GATING_NIGHTLY_LIMIT = 30
# The AMD all-main ledger exists for the hourly live alert, not long-range
# browser analytics. Bounding it keeps analytics.json from growing needlessly.
AMD_MAIN_OBSERVATION_LIMIT = 24
BK_GET_MAX_ATTEMPTS = 5
BK_GET_BACKOFF_SECONDS = 2
BK_GET_MAX_BACKOFF_SECONDS = 60
BK_GET_CONNECT_TIMEOUT_SECONDS = 10
BK_GET_INITIAL_READ_TIMEOUT_SECONDS = 30
BK_GET_READ_TIMEOUT_STEP_SECONDS = 15
BK_GET_MAX_READ_TIMEOUT_SECONDS = 60
BK_GET_RETRY_STATUS_CODES = frozenset({500, 502, 503, 504, 520, 522, 524})

ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT = ROOT / "data" / "vllm" / "ci"

RESULT_SUFFIX = {"amd-ci": "amd", "ci": "upstream"}
# Current vLLM nightly slots in UTC. Actual Buildkite ``created_at`` values win
# whenever they are available; these hours are only for JSONL-only fallbacks.
FALLBACK_CREATED_HOUR_UTC = {"amd-ci": 9, "ci": 6}
RETRY_FIELDS = (
    "retried",
    "retried_in_job_id",
    "retries_count",
    "retry_source",
    "retry_type",
    "step_key",
)
FAILED_JOB_STATES = {"failed", "soft_fail", "soft_failed", "timed_out", "broken", "canceled", "expired"}


def buildkite_job_url(pipeline_slug: str, build_number: int, job_id: str = "", step_id: str = "") -> str:
    """Return the most specific Buildkite URL we can construct for a job."""
    if not build_number:
        return ""
    base = f"https://buildkite.com/{BK_ORG}/{pipeline_slug}/builds/{build_number}"
    if job_id:
        return f"{base}/steps/canvas?jid={job_id}&tab=output"
    if step_id:
        return f"{base}/steps/canvas?sid={step_id}&tab=output"
    return base


def _iso_from_nightly_date(date_str: str, pipeline_slug: str) -> str:
    """Best-effort timestamp for JSONL-only builds.

    The analytics UI needs a ``created_at`` value for window filtering. When a
    Buildkite list response is partial, the parsed test-result JSONL still has
    the nightly date and build number, so synthesize the current schedule hour.
    """
    if not date_str:
        return ""
    hour = FALLBACK_CREATED_HOUR_UTC.get(pipeline_slug, 12)
    return f"{date_str}T{hour:02d}:00:00Z"


def _result_count(row: dict) -> int:
    """Extract collapsed pytest count from rows like ``__passed__ (136)``."""
    name = str(row.get("name") or "")
    m = re.search(r"\((\d+)\)\s*$", name)
    return int(m.group(1)) if m else 1


def _result_status_to_job_state(statuses: list[str]) -> str:
    """Collapse one job's parsed test rows into a single analytics state."""
    lowered = {str(s or "").lower() for s in statuses}
    if lowered & {"soft_fail", "soft_failed"}:
        return "soft_fail"
    if lowered & {"failed", "error", "timed_out", "broken", "canceled"}:
        return "failed"
    if lowered & {"passed", "xpassed"}:
        return "passed"
    if lowered & {"skipped", "xfailed"}:
        return "skipped"
    return "unknown"


def nightly_date(iso_str):
    """Convert a UTC timestamp to the 'nightly date'.

    Boundary at 12:00 UTC so both pipelines align in the same column. Current
    scheduled runs are before noon UTC (upstream at ~06:00, AMD at ~09:00), so
    they keep the same calendar day. Older upstream runs after noon still map
    to the following nightly date.
    """
    if not iso_str:
        return ""
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        if dt.hour >= 12:
            dt += timedelta(days=1)
        return dt.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return iso_str[:10] if iso_str else ""


def _rate_limit_wait_seconds(headers, attempt):
    """Return a retry delay that clears Buildkite rate-limit windows."""
    reset_waits = []
    for name in ("RateLimit-Reset", "RateLimit-User-Reset"):
        try:
            reset_waits.append(max(0, int(float(headers.get(name, "")))) + 1)
        except (TypeError, ValueError):
            continue
    if reset_waits:
        return max(reset_waits)
    try:
        return max(0, int(float(headers.get("Retry-After", ""))))
    except (TypeError, ValueError):
        return 5 * (attempt + 1)


def _request_retry_wait_seconds(attempt):
    """Return a capped exponential delay for a zero-based request attempt."""
    return min(BK_GET_BACKOFF_SECONDS * (2**attempt), BK_GET_MAX_BACKOFF_SECONDS)


def _request_timeout(attempt):
    """Bound connect time while allowing a slow response more time on retry."""
    read_timeout = min(
        BK_GET_INITIAL_READ_TIMEOUT_SECONDS + BK_GET_READ_TIMEOUT_STEP_SECONDS * attempt,
        BK_GET_MAX_READ_TIMEOUT_SECONDS,
    )
    return BK_GET_CONNECT_TIMEOUT_SECONDS, read_timeout


def bk_get(path, token, params=None):
    """Fetch one Buildkite REST page with bounded transient-error retries."""
    headers = {"Authorization": f"Bearer {token}"}
    url = f"{BK_API_BASE}{path}"
    p = dict(params or {})
    for attempt in range(BK_GET_MAX_ATTEMPTS):
        try:
            ratelimit.acquire()
            resp = requests.get(
                url,
                headers=headers,
                params=p,
                timeout=_request_timeout(attempt),
            )
            ratelimit.observe(resp.headers)
        except (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
        ) as exc:
            if attempt == BK_GET_MAX_ATTEMPTS - 1:
                raise
            wait = _request_retry_wait_seconds(attempt)
            log.warning(
                "Buildkite request %s page %s failed (%s), retry %d/%d in %ds",
                path,
                p.get("page", 1),
                type(exc).__name__,
                attempt + 1,
                BK_GET_MAX_ATTEMPTS,
                wait,
            )
            time.sleep(wait)
            continue

        if resp.status_code == 429:
            if attempt == BK_GET_MAX_ATTEMPTS - 1:
                resp.raise_for_status()
            wait = _rate_limit_wait_seconds(resp.headers, attempt)
            log.warning(
                "Buildkite request %s page %s rate limited, retry %d/%d in %ds",
                path,
                p.get("page", 1),
                attempt + 1,
                BK_GET_MAX_ATTEMPTS,
                wait,
            )
            time.sleep(wait)
            continue
        if resp.status_code in BK_GET_RETRY_STATUS_CODES:
            if attempt == BK_GET_MAX_ATTEMPTS - 1:
                resp.raise_for_status()
            wait = _request_retry_wait_seconds(attempt)
            log.warning(
                "Buildkite request %s page %s returned HTTP %d, retry %d/%d in %ds",
                path,
                p.get("page", 1),
                resp.status_code,
                attempt + 1,
                BK_GET_MAX_ATTEMPTS,
                wait,
            )
            time.sleep(wait)
            continue

        resp.raise_for_status()
        payload = resp.json()
        return payload if isinstance(payload, list) else [payload]
    return []


def queue_from_rules(rules):
    """Analytics wants ``"unknown"`` when no queue rule is present (keeps
    the job-stats queue column non-null)."""
    return _queue_from_rules(rules) or "unknown"


def normalize_job(name):
    """Strip hardware prefix for cross-build comparison."""
    name = re.sub(r'^(mi\d+_\d+|gpu_\d+|amd_\w+):\s*', '', name, flags=re.IGNORECASE)
    return name.strip()


def queue_from_result_job_name(name):
    """Derive an AMD queue from a parsed JSONL job name when metadata is absent."""
    match = re.match(r"^(mi\d+_\d+):\s*", name or "", flags=re.IGNORECASE)
    if match:
        return "amd_" + match.group(1).lower()
    match = re.match(r"^(amd[-_\w]+):\s*", name or "", flags=re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return None


def job_metadata_keys(job):
    """Return identity keys from most-specific to most-general.

    ``name`` is normalized for cross-build rankings, but parsed JSONL can have
    the same normalized title on several hardware pools in one build. Keeping
    ``raw_name`` first prevents an MI300 failure from being attached to the
    MI355 row in the AMD hardware matrix.
    """
    raw = (job.get("raw_name") or job.get("job_name") or job.get("full_name") or "").strip()
    name = (job.get("name") or "").strip()
    keys = []
    for key in (raw, name, normalize_job(raw), normalize_job(name)):
        if key and key not in keys:
            keys.append(key)
    return keys


def _build_job_metadata(builds: list[dict]) -> dict[int, dict[str, dict[str, dict]]]:
    """Index existing per-job metadata by build number, exact ID, and name.

    Buildkite retains superseded attempts when ``include_retried_jobs`` is
    enabled.  Those attempts commonly share a name, so the ID index is the
    authoritative join for parsed JSONL rows that carry a ``job_id``.  The
    name index remains a fallback for historical rows without job identity.
    """
    meta: dict[int, dict[str, dict[str, dict]]] = {}
    for build in builds:
        index = meta.setdefault(
            int(build.get("number") or 0),
            {"by_job_id": {}, "by_name": {}},
        )
        for job in build.get("jobs") or []:
            payload = {
                k: job[k]
                for k in (
                    "wait", "q", "state", "soft_failed", "job_id", "step_id", "url",
                    "started_at", "finished_at", "runnable_at", "wall_completion_mins",
                    "queue_wait_mins", "end_to_end_mins", "duration_source",
                )
                if k in job and job[k] is not None
            }
            # Historical parsed-result payloads used ``dur`` for summed pytest
            # time. Do not silently recycle that value as Buildkite wall time.
            duration_is_wall = (
                job.get("duration_source") == "buildkite_wall"
                or build.get("source") != "test_results"
            )
            if duration_is_wall and isinstance(job.get("dur"), (int, float)):
                payload["dur"] = job["dur"]
                payload.setdefault("wall_completion_mins", job["dur"])
                payload.setdefault("duration_source", "buildkite_wall")
            payload.update({k: job[k] for k in RETRY_FIELDS if k in job})
            job_id = str(job.get("job_id") or "")
            if job_id:
                index["by_job_id"][job_id] = payload
            for key in job_metadata_keys(job):
                index["by_name"][key] = payload
    return meta


def _merge_job_metadata(
    base: dict[int, dict[str, dict[str, dict]]],
    fresh: dict[int, dict[str, dict[str, dict]]],
) -> dict[int, dict[str, dict[str, dict]]]:
    """Merge metadata indexes, letting a fresh Buildkite read win by key."""
    for build_number, incoming in fresh.items():
        current = base.setdefault(build_number, {"by_job_id": {}, "by_name": {}})
        current["by_job_id"].update(incoming.get("by_job_id") or {})
        current["by_name"].update(incoming.get("by_name") or {})
    return base


def _job_metadata_for_result(
    index: dict[str, dict[str, dict]],
    result_job: dict,
) -> dict:
    """Resolve metadata for one parsed-result job without crossing attempts."""
    job_id = str(result_job.get("job_id") or "")
    if job_id:
        # An exact identity is authoritative. Falling back to a shared name
        # here can attach a manual retry's timestamps to the original attempt.
        return (index.get("by_job_id") or {}).get(job_id) or {}

    by_name = index.get("by_name") or {}
    for key in job_metadata_keys(result_job):
        if key in by_name:
            return by_name[key]
    return {}


def _build_metadata(builds: list[dict]) -> dict[int, dict]:
    """Build-level metadata we can carry over when using parsed JSONL state."""
    return {int(b.get("number") or 0): b for b in builds if b.get("number") is not None}


def load_test_result_builds(output: Path, pipeline_slug: str, days: int, buildkite_builds: list[dict] | None = None,
                            previous_builds: list[dict] | None = None) -> list[dict]:
    """Build analytics rows from parsed CI test-result JSONL files.

    ``collect_ci.py`` runs immediately before this script in the scheduled
    workflow. Those JSONL files are the same parsed test source used by CI
    Health, so they are a better source for AMD failure/pass-rate analytics than
    Buildkite's soft-failed job state. Buildkite data, when present, is still
    used for wall-clock, queue, wait, and exact URLs.
    """
    suffix = RESULT_SUFFIX.get(pipeline_slug)
    if not suffix:
        return []

    results_dir = output / "test_results"
    if not results_dir.exists():
        return []

    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    paths = sorted(results_dir.glob(f"*_{suffix}.jsonl"))
    paths = [p for p in paths if p.name.rsplit("_", 1)[0] >= cutoff]
    if not paths:
        return []

    bk_meta = _build_metadata(buildkite_builds or [])
    prev_meta = _build_metadata(previous_builds or [])
    job_meta = _build_job_metadata(previous_builds or [])
    _merge_job_metadata(job_meta, _build_job_metadata(buildkite_builds or []))

    grouped: dict[int, dict] = {}
    for path in paths:
        fallback_date = path.name.rsplit("_", 1)[0]
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                log.warning("Skipping malformed analytics test-result row in %s", path)
                continue
            if row.get("pipeline") and row.get("pipeline") != pipeline_slug:
                continue
            build_number = int(row.get("build_number") or 0)
            if not build_number:
                continue
            raw_job_name = str(row.get("job_name") or row.get("classname") or "unknown").strip()
            job_name = normalize_job(raw_job_name)
            if not raw_job_name or not job_name:
                continue
            bucket = grouped.setdefault(build_number, {
                "date": row.get("date") or fallback_date,
                "jobs": {},
            })
            job = bucket["jobs"].setdefault(raw_job_name, {
                "name": job_name,
                "raw_name": raw_job_name,
                "job_id": str(row.get("job_id") or ""),
                "step_id": str(row.get("step_id") or ""),
                "statuses": [],
                "dur": 0.0,
                "tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "skipped_tests": 0,
            })
            if not job.get("job_id") and row.get("job_id"):
                job["job_id"] = str(row.get("job_id") or "")
            if not job.get("step_id") and row.get("step_id"):
                job["step_id"] = str(row.get("step_id") or "")
            status = str(row.get("status") or "unknown").lower()
            count = _result_count(row)
            job["statuses"].append(status)
            job["dur"] += float(row.get("duration_secs") or 0.0) / 60.0
            job["tests"] += count
            if status in ("passed", "xpassed"):
                job["passed_tests"] += count
            elif status in ("failed", "error", "timed_out", "broken", "canceled"):
                job["failed_tests"] += count
            elif status in ("skipped", "xfailed"):
                job["skipped_tests"] += count

    builds = []
    for build_number, bucket in grouped.items():
        meta = bk_meta.get(build_number) or prev_meta.get(build_number) or {}
        jobs = []
        passed = failed = soft = skipped = 0
        for raw_name, raw_job in sorted(bucket["jobs"].items()):
            metadata = _job_metadata_for_result(
                job_meta.get(build_number, {}),
                raw_job,
            )
            state = _result_status_to_job_state(raw_job["statuses"])
            if metadata.get("state") == "soft_fail" or metadata.get("soft_failed"):
                state = "soft_fail"
            elif state == "unknown" and metadata.get("state"):
                state = metadata["state"]

            if state == "passed":
                passed += 1
            elif state == "failed":
                failed += 1
            elif state == "soft_fail":
                soft += 1
            elif state == "skipped":
                skipped += 1

            entry = {
                "name": raw_job["name"],
                "raw_name": raw_job["raw_name"],
                "state": state,
                "test_duration_mins": round(raw_job["dur"], 1),
                "tests": raw_job["tests"],
                "passed_tests": raw_job["passed_tests"],
                "failed_tests": raw_job["failed_tests"],
                "skipped_tests": raw_job["skipped_tests"],
            }
            job_id = str(raw_job.get("job_id") or metadata.get("job_id") or "")
            step_id = str(raw_job.get("step_id") or metadata.get("step_id") or "")
            job_url = buildkite_job_url(
                pipeline_slug,
                build_number,
                job_id,
                step_id,
            ) or str(metadata.get("url") or "")
            if job_url:
                entry["url"] = job_url
            if job_id:
                entry["job_id"] = job_id
            if step_id:
                entry["step_id"] = step_id
            queue = queue_from_result_job_name(raw_job["raw_name"])
            if queue:
                entry["q"] = queue
            for k, v in metadata.items():
                if k == "q" and entry.get("q"):
                    continue
                if k in ("state", "soft_failed", "job_id", "step_id", "url"):
                    continue
                entry[k] = v
            jobs.append(entry)

        created = meta.get("created_at") or _iso_from_nightly_date(bucket["date"], pipeline_slug)
        build_state = meta.get("state") or ("failed" if failed else "passed")
        builds.append({
            "number": build_number,
            "state": build_state,
            "created_at": created,
            "date": bucket["date"] or nightly_date(created),
            "message": meta.get("message") or "nightly",
            "branch": meta.get("branch") or "main",
            "commit": meta.get("commit") or "",
            "author": meta.get("author") or "",
            "wall_mins": meta.get("wall_mins"),
            "passed": passed,
            "failed": failed,
            "soft_failed": soft,
            "skipped": skipped,
            "total_jobs": len(jobs),
            "jobs": jobs,
            "web_url": meta.get("web_url") or f"https://buildkite.com/{BK_ORG}/{pipeline_slug}/builds/{build_number}",
            "source": "test_results",
        })

    builds.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return builds


def choose_analytics_builds(buildkite_builds: list[dict], result_builds: list[dict],
                            previous_builds: list[dict] | None = None, pipeline_slug: str = "") -> list[dict]:
    """Prefer parsed test-result builds, with guards against empty overwrites."""
    if result_builds:
        if buildkite_builds and len(result_builds) < max(2, len(buildkite_builds) // 2):
            log.warning(
                "%s has only %d parsed-result builds versus %d Buildkite builds; keeping Buildkite analytics",
                pipeline_slug, len(result_builds), len(buildkite_builds),
            )
            return buildkite_builds
        if len(result_builds) > len(buildkite_builds):
            log.info("  using %d parsed test-result builds for %s analytics", len(result_builds), pipeline_slug)
        return result_builds

    if previous_builds and not buildkite_builds:
        log.warning("  preserving previous %s analytics: fresh collection returned no builds", pipeline_slug)
        return previous_builds

    return buildkite_builds


def _retry_group_key(job: dict) -> tuple[str, str]:
    step = str(job.get("step_key") or job.get("step_id") or "")
    name = str(job.get("raw_name") or job.get("name") or "unknown")
    return step or name, name


def _retry_attempt_summary(build: dict, job: dict) -> dict:
    observed_at = (
        job.get("finished_at")
        or job.get("started_at")
        or build.get("finished_at")
        or build.get("created_at")
        or ""
    )
    out = {
        "build_number": build.get("number"),
        "step": str(job.get("step_key") or job.get("step_id") or ""),
        "name": str(job.get("raw_name") or job.get("name") or "unknown"),
        "state": job.get("state") or "unknown",
        "observed_at": observed_at,
    }
    for key in ("job_id", "url", "retries_count", "retry_source", "retry_type"):
        if job.get(key) not in (None, ""):
            out[key] = job[key]
    return out


def compute_retry_analysis(builds: list[dict]) -> dict:
    """Summarize retry attempts and failed-then-passed job recoveries.

    Buildkite's explicit ``retried_in_job_id`` edge is authoritative when it
    is present. The step/name grouping also handles payloads where Buildkite
    retained retry counters but omitted the edge from a compact prior row.
    """
    retry_attempts: list[dict] = []
    recoveries: list[dict] = []
    builds_with_retries: set[int] = set()
    seen_attempts: set[tuple] = set()
    seen_recoveries: set[tuple] = set()

    for build in builds:
        build_number = int(build.get("number") or 0)
        jobs = list(build.get("jobs") or [])
        by_id = {str(job.get("job_id")): job for job in jobs if job.get("job_id")}
        retry_targets = {
            str(job.get("retried_in_job_id"))
            for job in jobs
            if job.get("retried_in_job_id")
        }
        grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for job in jobs:
            grouped[_retry_group_key(job)].append(job)

        for job in jobs:
            job_id = str(job.get("job_id") or "")
            is_attempt = (
                job_id in retry_targets
                or bool(job.get("retry_source"))
                or bool(job.get("retry_type"))
                or int(job.get("retries_count") or 0) > 0
            )
            if not is_attempt:
                continue
            attempt_key = (build_number, job_id or _retry_group_key(job))
            if attempt_key in seen_attempts:
                continue
            seen_attempts.add(attempt_key)
            retry_attempts.append(_retry_attempt_summary(build, job))
            builds_with_retries.add(build_number)

        for failed_job in jobs:
            target_id = str(failed_job.get("retried_in_job_id") or "")
            passed_job = by_id.get(target_id)
            if not passed_job:
                continue
            if failed_job.get("state") not in FAILED_JOB_STATES or passed_job.get("state") != "passed":
                continue
            recovery_key = (build_number, target_id, _retry_group_key(failed_job))
            if recovery_key in seen_recoveries:
                continue
            seen_recoveries.add(recovery_key)
            recoveries.append({
                "build_number": build_number,
                "step": str(failed_job.get("step_key") or failed_job.get("step_id") or ""),
                "name": str(failed_job.get("raw_name") or failed_job.get("name") or "unknown"),
                "observed_at": (
                    passed_job.get("finished_at")
                    or passed_job.get("started_at")
                    or failed_job.get("finished_at")
                    or build.get("finished_at")
                    or build.get("created_at")
                    or ""
                ),
                "failed_job_id": failed_job.get("job_id") or "",
                "passed_job_id": passed_job.get("job_id") or "",
                "failed_url": failed_job.get("url") or "",
                "passed_url": passed_job.get("url") or "",
            })

        for group_key, attempts in grouped.items():
            failed_jobs = [job for job in attempts if job.get("state") in FAILED_JOB_STATES]
            passed_retries = [
                job for job in attempts
                if job.get("state") == "passed"
                and (
                    str(job.get("job_id") or "") in retry_targets
                    or bool(job.get("retry_source"))
                    or bool(job.get("retry_type"))
                    or int(job.get("retries_count") or 0) > 0
                )
            ]
            if not failed_jobs:
                continue
            failed_job = failed_jobs[-1]
            for passed_job in passed_retries:
                passed_id = str(passed_job.get("job_id") or "")
                recovery_key = (build_number, passed_id, group_key)
                if recovery_key in seen_recoveries:
                    continue
                seen_recoveries.add(recovery_key)
                recoveries.append({
                    "build_number": build_number,
                    "step": group_key[0],
                    "name": group_key[1],
                    "observed_at": (
                        passed_job.get("finished_at")
                        or passed_job.get("started_at")
                        or failed_job.get("finished_at")
                        or build.get("finished_at")
                        or build.get("created_at")
                        or ""
                    ),
                    "failed_job_id": failed_job.get("job_id") or "",
                    "passed_job_id": passed_job.get("job_id") or "",
                    "failed_url": failed_job.get("url") or "",
                    "passed_url": passed_job.get("url") or "",
                })

    retry_attempts.sort(key=lambda row: (row.get("build_number") or 0, row.get("step", ""), row["name"]), reverse=True)
    recoveries.sort(key=lambda row: (row.get("build_number") or 0, row.get("step", ""), row["name"]), reverse=True)
    return {
        "summary": {
            "builds_evaluated": len(builds),
            "builds_with_retries": len(builds_with_retries),
            "retry_attempt_count": len(retry_attempts),
            "failed_then_passed_recovery_count": len(recoveries),
        },
        "retry_attempts": retry_attempts,
        "failed_then_passed_recoveries": recoveries,
    }


def attach_main_reliability(
    pipeline_data: dict,
    reliability: dict,
    retry_builds: list[dict] | None = None,
    retry_analysis: dict | None = None,
) -> None:
    """Attach the bounded all-main cohort and its compatibility stream."""
    pipeline_slug = str((reliability.get("cohort") or {}).get("pipeline") or "")
    if not pipeline_slug or not validate_all_main_reliability(reliability, pipeline_slug):
        raise ValueError("all-main reliability payload lacks strict exhaustive provenance")
    main_builds = compact_main_builds(reliability)
    retained = sum(len(build.get("jobs") or []) for build in main_builds)
    cohort = reliability.get("cohort") or {}
    provenance = reliability.get("provenance") or {}
    denominator = reliability.get("denominator") or {}
    pipeline_data["all_main_reliability"] = reliability
    pipeline_data["main_builds"] = main_builds
    pipeline_data["main_builds_provenance"] = {
        "schema_version": reliability.get("schema_version"),
        "cohort": cohort,
        "window": {
            key: cohort.get(key)
            for key in ("window_days", "requested_from", "observed_from", "observed_to")
        },
        "denominator": denominator,
        "source": provenance,
        "retention": {
            "eligible_observations_in_denominator": denominator.get("eligible_observations", 0),
            "eligible_observations_in_main_builds": retained,
            "observation_limit_per_group": provenance.get("observation_limit_per_group"),
        },
        "authoritative_evidence_key": "all_main_reliability",
    }
    eligible_numbers = {
        int(build.get("number") or 0)
        for build in reliability.get("builds") or []
        if int(build.get("number") or 0)
    }
    if retry_analysis is not None and validate_retry_analysis(
        retry_analysis,
        pipeline_slug,
        eligible_numbers,
    ):
        pipeline_data["main_retry_analysis"] = retry_analysis
        return
    if retry_builds is None:
        pipeline_data["main_retry_analysis"] = {
            "available": False,
            "summary": {
                "builds_evaluated": len(eligible_numbers),
                "builds_with_retries": 0,
                "retry_attempt_count": 0,
                "failed_then_passed_recovery_count": 0,
            },
            "retry_attempts": [],
            "failed_then_passed_recoveries": [],
            "provenance": {
                "source_pipeline": pipeline_slug,
                "complete": False,
                "reason": "complete raw retry attempts were unavailable; compacted history was not substituted",
            },
        }
        return
    complete_retry_builds = [
        build
        for build in retry_builds or []
        if int(build.get("number") or 0) in eligible_numbers
    ]
    analysis = compute_retry_analysis(complete_retry_builds)
    analysis["available"] = True
    analysis["provenance"] = {
        "source_pipeline": pipeline_slug,
        "complete": True,
        "scope": "same completed branch=main builds and test-job queue scope as all-main reliability",
        "cohort_build_numbers": sorted(eligible_numbers),
    }
    pipeline_data["main_retry_analysis"] = analysis


def validate_retry_analysis(
    payload: Any,
    pipeline_slug: str,
    cohort_build_numbers: set[int],
) -> bool:
    if not isinstance(payload, dict):
        return False
    provenance = payload.get("provenance")
    attempts = payload.get("retry_attempts")
    recoveries = payload.get("failed_then_passed_recoveries")
    provenance_builds = (
        provenance.get("cohort_build_numbers")
        if isinstance(provenance, dict)
        else None
    )
    if (
        payload.get("available") is not True
        or not isinstance(provenance, dict)
        or provenance.get("source_pipeline") != pipeline_slug
        or provenance.get("complete") is not True
        or not isinstance(provenance_builds, list)
        or any(not isinstance(number, int) or isinstance(number, bool) for number in provenance_builds)
        or set(provenance_builds) != cohort_build_numbers
        or not isinstance(attempts, list)
        or not isinstance(recoveries, list)
    ):
        return False
    for row in attempts:
        if not isinstance(row, dict):
            return False
        try:
            number = int(row.get("build_number") or 0)
        except (TypeError, ValueError):
            return False
        url = row.get("job_url") or row.get("url")
        if (
            number not in cohort_build_numbers
            or row.get("source_pipeline") not in (None, "", pipeline_slug)
            or not buildkite_job_url_matches(url, pipeline_slug, number)
        ):
            return False
    for row in recoveries:
        if not isinstance(row, dict):
            return False
        try:
            number = int(row.get("build_number") or 0)
        except (TypeError, ValueError):
            return False
        if (
            number not in cohort_build_numbers
            or row.get("source_pipeline") not in (None, "", pipeline_slug)
            or not buildkite_job_url_matches(
                row.get("failed_url") or row.get("failed_job_url"),
                pipeline_slug,
                number,
            )
            or not buildkite_job_url_matches(
                row.get("passed_url") or row.get("passed_job_url"),
                pipeline_slug,
                number,
            )
        ):
            return False
    return True


def _safe_build_number(build: Any) -> int:
    if not isinstance(build, dict):
        return 0
    try:
        return int(build.get("number") or 0)
    except (TypeError, ValueError):
        return 0


def _fetched_build_rank(build: dict) -> tuple:
    state = str(build.get("state") or "").lower()
    return (
        state in {"passed", "failed"} and bool(build.get("finished_at")),
        len(build.get("jobs") or []),
        str(build.get("finished_at") or ""),
        str(build.get("created_at") or ""),
    )


def fetch_pipeline_builds(pipeline_slug, token, days, max_pages=None):
    """Fetch Buildkite ``main`` builds exhaustively within the safety cap."""
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    log.info("Fetching %s builds (last %d days)...", pipeline_slug, days)
    path = f"/organizations/{BK_ORG}/pipelines/{pipeline_slug}/builds"
    page_limit = max_pages if max_pages is not None else BUILD_FETCH_MAX_PAGES
    by_number: dict[int, dict] = {}
    termination_reason = "max_pages"
    exhaustive = False
    pages_fetched = 0
    for page in range(1, page_limit + 1):
        pages_fetched = page
        rows = bk_get(path, token, {
            "branch": "main",
            "created_from": since,
            "per_page": BUILD_FETCH_PAGE_SIZE,
            "page": page,
            "include_retried_jobs": "true",
        })
        if not rows:
            termination_reason = "empty_page"
            exhaustive = True
            break
        novel_numbers = 0
        for build in rows:
            if not isinstance(build, dict):
                continue
            number = _safe_build_number(build)
            if number:
                existing = by_number.get(number)
                if existing is None:
                    by_number[number] = build
                    novel_numbers += 1
                elif _fetched_build_rank(build) > _fetched_build_rank(existing):
                    by_number[number] = build
        if len(rows) < BUILD_FETCH_PAGE_SIZE:
            termination_reason = "short_page"
            exhaustive = True
            break
        if not novel_numbers:
            termination_reason = "duplicate_page"
            log.warning("  stopping pagination at page %d: no new build numbers", page)
            break
    if not exhaustive:
        log.warning(
            "  incomplete Buildkite pagination (%s after %d pages, %d builds)",
            termination_reason,
            pages_fetched,
            len(by_number),
        )
    builds_raw = sorted(
        by_number.values(),
        key=lambda build: (
            str(build.get("created_at") or ""),
            _safe_build_number(build),
        ),
        reverse=True,
    )
    log.info("  %d unique builds fetched", len(builds_raw))
    return builds_raw, {
        "created_from": since,
        "page_size": BUILD_FETCH_PAGE_SIZE,
        "max_pages": page_limit,
        "pages_fetched": pages_fetched,
        "termination_reason": termination_reason,
        "exhaustive": exhaustive,
    }


def summarize_pipeline_builds(pipeline_slug, builds_raw, nightly_only=False, name_pattern=None):
    """Normalize Buildkite builds while retaining per-attempt provenance."""
    builds_raw = list(builds_raw or [])

    # Filter to nightly if requested
    if nightly_only and name_pattern:
        pat = re.compile(name_pattern, re.IGNORECASE)
        builds_raw = [b for b in builds_raw if pat.search(b.get("message", "") or "")]
        log.info("  %d nightly builds after filter", len(builds_raw))

    builds = []

    for b in builds_raw:
        build_num = b.get("number", 0)
        build_state = b.get("state", "")
        created = b.get("created_at", "")
        finished = b.get("finished_at", "")
        wall_mins = duration_mins(created, finished)
        message = (b.get("message") or "")[:100]
        author = (b.get("creator") or {}).get("name", "") or (b.get("author") or {}).get("name", "")

        jobs = [j for j in b.get("jobs", []) if j.get("type") == "script"]

        job_summaries = []
        passed = failed = soft = 0

        for j in jobs:
            name = j.get("name", "unknown")
            norm = normalize_job(name)
            state = j.get("state", "")
            sf = j.get("soft_failed", False)
            queue = queue_from_rules(j.get("agent_query_rules"))

            dur = duration_mins(j.get("started_at"), j.get("finished_at"))
            wait = duration_mins(j.get("runnable_at"), j.get("started_at"))
            end_to_end = duration_mins(j.get("runnable_at"), j.get("finished_at"))

            if state == "passed":
                passed += 1
            elif sf:
                soft += 1
            elif state in ("failed", "timed_out", "broken"):
                failed += 1

            job_id = str(j.get("id") or "")
            step_id = str((j.get("step") or {}).get("id") or "")
            job_entry = {
                "name": norm,
                "raw_name": name,
                "state": "soft_fail" if sf else state,
                "dur": dur,
                "wall_completion_mins": dur,
                "queue_wait_mins": wait,
                "end_to_end_mins": end_to_end,
                "duration_source": "buildkite_wall",
                "started_at": j.get("started_at") or "",
                "finished_at": j.get("finished_at") or "",
                "runnable_at": j.get("runnable_at") or "",
            }
            for key in RETRY_FIELDS:
                value = j.get(key, (j.get("step") or {}).get("key") if key == "step_key" else None)
                job_entry[key] = value
            if job_id:
                job_entry["job_id"] = job_id
            if step_id:
                job_entry["step_id"] = step_id
            job_url = buildkite_job_url(
                pipeline_slug,
                build_num,
                job_id,
                step_id,
            ) or j.get("web_url", "")
            if job_url:
                job_entry["url"] = job_url
            if wait is not None: job_entry["wait"] = round(wait, 1)
            if queue: job_entry["q"] = queue
            job_summaries.append(job_entry)

        builds.append({
            "number": build_num,
            "state": build_state,
            "created_at": created,
            "finished_at": finished,
            "date": nightly_date(created),
            "message": message,
            "branch": b.get("branch") or "",
            "commit": b.get("commit") or "",
            "author": author,
            "wall_mins": wall_mins,
            "passed": passed,
            "failed": failed,
            "soft_failed": soft,
            "total_jobs": len(jobs),
            "jobs": job_summaries,
            "web_url": b.get("web_url", ""),
        })

    # Sort builds newest first
    builds.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return builds


def collect_pipeline(pipeline_slug, token, days, nightly_only=False, name_pattern=None, builds_raw=None):
    """Fetch and normalize builds, preserving the historical public API."""
    if builds_raw is None:
        builds_raw, _ = fetch_pipeline_builds(pipeline_slug, token, days)
    return summarize_pipeline_builds(pipeline_slug, builds_raw, nightly_only, name_pattern)


def compute_job_rankings(builds):
    """Aggregate per-job rankings from the provided build slice."""
    job_stats = defaultdict(lambda: {"runs": 0, "passed": 0, "failed": 0, "soft_failed": 0,
                                     "durations": [], "wait_times": [], "queues": set()})

    for build in builds:
        for job in build.get("jobs", []):
            name = job.get("name", "unknown")
            state = job.get("state", "")
            queue = job.get("q")
            dur = job.get("dur")
            wait = job.get("wait")

            if state == "passed":
                job_stats[name]["passed"] += 1
            elif state == "soft_fail":
                job_stats[name]["soft_failed"] += 1
            elif state in ("failed", "timed_out", "broken"):
                job_stats[name]["failed"] += 1

            job_stats[name]["runs"] += 1
            if dur is not None:
                job_stats[name]["durations"].append(dur)
            if wait is not None:
                job_stats[name]["wait_times"].append(wait)
            if queue:
                job_stats[name]["queues"].add(queue)

    job_rankings = []
    for name, s in sorted(job_stats.items()):
        total = s["runs"]
        if total == 0:
            continue
        durs = sorted(s["durations"])
        waits = sorted(s["wait_times"])
        fail_rate = round((s["failed"] + s["soft_failed"]) / total * 100, 1)
        job_rankings.append({
            "name": name,
            "runs": total,
            "passed": s["passed"],
            "failed": s["failed"],
            "soft_failed": s["soft_failed"],
            "fail_rate": fail_rate,
            "is_soft_fail": s["failed"] == 0 and s["soft_failed"] > 0,
            "median_dur": round(median(durs), 1) if durs else None,
            "p90_dur": round(percentile(durs, 90), 1) if durs else None,
            "avg_dur": round(mean(durs), 1) if durs else None,
            "max_dur": round(max(durs), 1) if durs else None,
            "median_wait": round(median(waits), 1) if waits else None,
            "p90_wait": round(percentile(waits, 90), 1) if waits else None,
            "avg_wait": round(mean(waits), 1) if waits else None,
            "max_wait": round(max(waits), 1) if waits else None,
            "queues": sorted(s["queues"]),
        })
    return job_rankings


def compute_daily_stats(builds):
    """Aggregate pass/fail per day for stacked bar chart."""
    by_date = defaultdict(lambda: {"passed": 0, "failed": 0, "total": 0})
    for b in builds:
        d = b.get("date", "")
        if not d: continue
        if b["state"] in ("passed",):
            by_date[d]["passed"] += 1
        elif b["state"] in ("failed", "failing"):
            by_date[d]["failed"] += 1
        by_date[d]["total"] += 1
    return [{"date": k, **v} for k, v in sorted(by_date.items())]


def compute_queue_stats(job_rankings):
    """Aggregate wait times by queue."""
    by_queue = defaultdict(lambda: {"jobs": 0, "waits": []})
    for j in job_rankings:
        for q in j.get("queues", []):
            by_queue[q]["jobs"] += j["runs"]
            if j.get("median_wait") is not None:
                by_queue[q]["waits"].extend([j["median_wait"]] * j["runs"])

    queue_stats = []
    for q, d in sorted(by_queue.items()):
        waits = d["waits"]
        queue_stats.append({
            "queue": q,
            "jobs": d["jobs"],
            "median_wait": round(median(waits), 1) if waits else None,
            "p90_wait": round(sorted(waits)[int(len(waits) * 0.9)], 1) if len(waits) > 1 else None,
            "avg_wait": round(mean(waits), 1) if waits else None,
            "max_wait": round(max(waits), 1) if waits else None,
        })
    queue_stats.sort(key=lambda x: x.get("median_wait") or 0, reverse=True)
    return queue_stats


def compute_summary(builds, job_rankings):
    total_builds = len(builds)
    passed_builds = sum(1 for b in builds if b["state"] == "passed")
    failed_builds = sum(1 for b in builds if b["state"] in ("failed", "failing"))
    hard_failed_jobs = sum(1 for j in job_rankings if j["failed"] > 0)
    soft_failed_jobs = sum(1 for j in job_rankings if j["failed"] == 0 and j["soft_failed"] > 0)
    return {
        "total_builds": total_builds,
        "passed": passed_builds,
        "failed": failed_builds,
        "pass_rate": round(passed_builds / total_builds * 100, 1) if total_builds else 0,
        "total_jobs_tracked": len(job_rankings),
        "jobs_with_failures": hard_failed_jobs + soft_failed_jobs,
        "jobs_with_hard_failures": hard_failed_jobs,
        "jobs_with_soft_failures": soft_failed_jobs,
    }


def filter_builds_for_window(builds, window_days, now=None):
    if window_days <= 0:
        return []
    ref_now = now or datetime.now(timezone.utc)
    cutoff = ref_now - timedelta(days=window_days)
    return [
        build for build in builds
        if (parse_ts(build.get("created_at")) or cutoff) >= cutoff
    ]


def build_window_block(builds, window_days):
    job_rankings = compute_job_rankings(builds)
    failure_ranking = sorted(job_rankings, key=lambda x: x["fail_rate"], reverse=True)
    duration_ranking = sorted(job_rankings, key=lambda x: x.get("median_dur") or 0, reverse=True)
    return {
        "window_days": window_days,
        "build_count": len(builds),
        "summary": compute_summary(builds, job_rankings),
        "daily_stats": compute_daily_stats(builds),
        "builds": [chart_build_summary(build) for build in builds[:ANALYTICS_WINDOW_BUILD_LIMIT]],
        "nightly_builds": [chart_build_summary(build) for build in builds[:ANALYTICS_WINDOW_NIGHTLY_LIMIT]],
        "failure_ranking": [j for j in failure_ranking if j["failed"] > 0 or j["soft_failed"] > 0],
        "duration_ranking": duration_ranking,
        "queue_stats": compute_queue_stats(job_rankings),
    }


def compute_window_blocks(builds, max_days, now=None):
    window_days = sorted({d for d in ANALYTICS_WINDOWS_DAYS if d <= max_days} | {max_days})
    return {
        f"{days}d": build_window_block(filter_builds_for_window(builds, days, now=now), days)
        for days in window_days
    }


def chart_build_summary(build):
    """Return the per-build fields chart widgets need, without duplicating jobs."""
    return {key: value for key, value in build.items() if key != "jobs"}


def _buildkite_url_ids(url: str) -> dict[str, str]:
    """Extract compact Buildkite identifiers from an exact step URL."""
    if not url:
        return {}
    match = re.search(r"[?&](jid|sid)=([0-9a-fA-F-]+)", str(url))
    if not match:
        return {}
    key = "job_id" if match.group(1) == "jid" else "step_id"
    return {key: match.group(2)}


def gating_job_summary(job):
    """Return only fields needed by the AMD gating executive view."""
    keep = ("name", "raw_name", "state", "q", "job_id", "step_id")
    out = {key: job[key] for key in keep if key in job and job[key] not in (None, "")}
    if not out.get("job_id") and not out.get("step_id"):
        out.update(_buildkite_url_ids(str(job.get("url") or job.get("web_url") or "")))
    if not out.get("job_id") and not out.get("step_id") and (job.get("url") or job.get("web_url")):
        out["url"] = job.get("url") or job.get("web_url")
    return out


def gating_build_summary(build):
    """Slim nightly build payload for CI Health gating matching."""
    keep = ("number", "state", "created_at", "date", "message", "web_url")
    out = {key: build[key] for key in keep if key in build and build[key] not in (None, "")}
    out["jobs"] = [gating_job_summary(job) for job in build.get("jobs") or []]
    return out


def write_gating_nightlies(output: Path, all_data: dict[str, dict[str, Any]], generated_at: str) -> None:
    payload = {
        "generated_at": generated_at,
        "source": "scripts/vllm/collect_analytics.py",
    }
    for slug in ("ci", "amd-ci"):
        block = all_data.get(slug) or {}
        payload[slug] = {
            "pipeline": slug,
            "display_name": block.get("display_name") or PIPELINES.get(slug, slug),
            "builds": [
                gating_build_summary(build)
                for build in (block.get("builds") or [])[:GATING_NIGHTLY_LIMIT]
            ],
        }
    out_path = output / "gating_nightlies.json"
    out_path.write_text(json.dumps(payload, separators=(",", ":"), default=str) + "\n")
    log.info("Wrote %s", out_path)


def write_analytics(out_path: Path, payload: dict) -> None:
    """Write the large analytics artifact without whitespace amplification."""
    out_path.write_text(
        json.dumps(payload, separators=(",", ":"), default=str) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Collect CI analytics for rich dashboard")
    parser.add_argument("--days", type=int, default=90, help="Days of history (default: 90)")
    parser.add_argument("--pipeline", choices=["amd-ci", "ci", "both"], default="both")
    parser.add_argument("--output", type=str, default=str(OUTPUT))
    args = parser.parse_args()

    token = os.getenv("BUILDKITE_TOKEN")
    if not token:
        log.warning("BUILDKITE_TOKEN not set; using parsed test_results and previous metadata only")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    previous_data = {}
    previous_path = output / "analytics.json"
    if previous_path.exists():
        try:
            previous_data = json.loads(previous_path.read_text())
        except json.JSONDecodeError:
            log.warning("Ignoring malformed previous analytics at %s", previous_path)

    pipelines = ["amd-ci", "ci"] if args.pipeline == "both" else [args.pipeline]
    # A targeted refresh must not erase the other pipeline's analytics and
    # reliability history.
    all_data = {
        slug: block
        for slug, block in previous_data.items()
        if slug not in pipelines and isinstance(block, dict)
    }
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ref_now = datetime.now(timezone.utc)

    for slug in pipelines:
        log.info("=== %s ===", PIPELINES.get(slug, slug))

        # Fetch branch=main once. Nightly regression streams remain pipeline
        # specific; strict test-group reliability is published for both pipelines.
        # Upstream CI remains the only source for flake and retry analysis.
        previous_builds = (previous_data.get(slug) or {}).get("builds") or []
        raw_builds = []
        collection_provenance = {}
        if token:
            raw_builds, collection_provenance = fetch_pipeline_builds(
                slug,
                token,
                args.days,
            )
        buildkite_builds = (
            collect_pipeline(
                slug,
                token,
                args.days,
                nightly_only=True,
                name_pattern=NIGHTLY_NAME_PATTERNS_BY_SLUG.get(slug),
                builds_raw=raw_builds,
            )
            if token
            else []
        )
        result_builds = load_test_result_builds(output, slug, args.days, buildkite_builds, previous_builds)
        builds = choose_analytics_builds(buildkite_builds, result_builds, previous_builds, slug)
        job_rankings = compute_job_rankings(builds)
        windows = compute_window_blocks(builds, args.days, now=ref_now)
        default_window_days = min(DEFAULT_ANALYTICS_WINDOW_DAYS, args.days)
        default_window_key = f"{default_window_days}d"
        if default_window_key not in windows:
            default_window_key = sorted(windows.keys(), key=lambda k: int(k[:-1]))[-1]

        daily = compute_daily_stats(builds)
        queues = compute_queue_stats(job_rankings)

        # Sort rankings
        failure_ranking = sorted(job_rankings, key=lambda x: x["fail_rate"], reverse=True)
        duration_ranking = sorted(job_rankings, key=lambda x: x.get("median_dur") or 0, reverse=True)

        nightly_change_history = compute_nightly_change_history(builds)
        all_data[slug] = {
            "pipeline": slug,
            "display_name": PIPELINES.get(slug, slug),
            "days": args.days,
            "generated_at": generated_at,
            "cohort": {
                "name": "canonical message-matched nightlies",
                "pipeline": slug,
                "branch": "main",
                "window_days": args.days,
                "build_count": len(builds),
                "name_pattern": NIGHTLY_NAME_PATTERNS_BY_SLUG.get(slug) or "",
            },
            "transition_basis": (
                "canonical nightly job variants; fixed requires a current observed pass, "
                "and absence is reported as not_observed"
            ),
            "nightly_change_history": nightly_change_history,
            "summary": compute_summary(builds, job_rankings),
            "daily_stats": daily,
            "builds": builds[:ANALYTICS_BUILD_LIMIT],  # Long enough for 3-month trend views
            "nightly_builds": [chart_build_summary(build) for build in builds[:ANALYTICS_NIGHTLY_LIMIT]],
            "failure_ranking": [j for j in failure_ranking if j["failed"] > 0 or j["soft_failed"] > 0],
            "duration_ranking": duration_ranking,
            "queue_stats": queues,
            "default_window": default_window_key,
            "windows": windows,
        }
        previous_pipeline_data = previous_data.get(slug) or {}
        previous_all_main = previous_pipeline_data.get("all_main_reliability")
        previous_retry = previous_pipeline_data.get("main_retry_analysis")
        preserved_retry_analysis = None
        all_main_reliability = None
        complete_retry_builds = None
        if token and collection_provenance.get("exhaustive") is True:
            all_main_reliability = build_all_main_reliability(
                raw_builds,
                pipeline_slug=slug,
                window_days=args.days,
                generated_at=generated_at,
                nightly_pattern=NIGHTLY_NAME_PATTERNS_BY_SLUG.get(slug) or "",
                test_result_builds=result_builds,
                observation_limit=(
                    AMD_MAIN_OBSERVATION_LIMIT
                    if slug == "amd-ci"
                    else OBSERVATION_LIMIT
                ),
                collection_provenance=collection_provenance,
            )
            if slug == "ci":
                complete_retry_builds = summarize_pipeline_builds(
                    slug,
                    filter_reliability_builds(raw_builds),
                )
        elif validate_all_main_reliability(previous_all_main, slug):
            reason = (
                "Buildkite pagination was incomplete"
                if token
                else "BUILDKITE_TOKEN is unavailable"
            )
            log.warning("  preserving previous %s all-main reliability: %s", slug, reason)
            all_main_reliability = previous_all_main
            if slug == "ci":
                preserved_retry_analysis = previous_retry
        else:
            log.error("  strict %s all-main reliability is unavailable; refusing fallback data", slug)
        if all_main_reliability:
            if slug == "ci":
                attach_main_reliability(
                    all_data[slug],
                    all_main_reliability,
                    retry_builds=complete_retry_builds,
                    retry_analysis=preserved_retry_analysis,
                )
            else:
                # AMD reliability powers the live main-failure automation. Keep
                # upstream-only retry semantics out of the AMD block.
                all_data[slug]["all_main_reliability"] = all_main_reliability

        log.info("  %d builds, %d jobs tracked, %d with failures",
                 len(builds), len(job_rankings),
                 sum(1 for j in job_rankings if j["failed"] > 0))

    # Write output
    out_path = output / "analytics.json"
    write_analytics(out_path, all_data)
    log.info("Wrote %s", out_path)
    write_gating_nightlies(output, all_data, generated_at)

    # Print summary
    for slug, d in all_data.items():
        s = d["summary"]
        print(f"\n{d['display_name']}: {s['total_builds']} builds, {s['pass_rate']}% pass rate, "
              f"{s['jobs_with_failures']} jobs with failures, {s['total_jobs_tracked']} jobs tracked")


if __name__ == "__main__":
    main()
