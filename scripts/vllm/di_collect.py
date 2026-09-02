"""Collection logic for the AMD distributed-inference (DI) pipeline.

Kept separate from ``scripts/collect_di_ci.py`` (a thin CLI) so the parts
worth testing are importable without running a collection.

Two artifacts come out of a pass, because they answer different questions:

``jobs.jsonl``
    One rich record per (build, job): grid coordinates, queue wait, runtime,
    agent, and the SLURM driver's own ``phase``/``reason`` verdict. This is
    what the DI grid renders from.

``test_results/*.jsonl``
    ``TestResult`` rows in the framework's schema, one per job, so the
    existing health/flakiness/trend analysis applies to DI cells unmodified.

Note the deliberate ``log_text=""`` when calling ``parse_job_results``. DI
steps are SLURM launchers and emit no pytest output, so the parser's
job-level fallback is the path we want — forcing it makes the outcome
deterministic (no chance of a stray ``FAILED``-looking line in a 100 MB
SLURM log inventing test rows) and skips 20 large log downloads per build.
Logs are still fetched separately, but only to read the driver's verdict
line, and only once per job — a terminal job's log never changes, so
``jobs.jsonl`` doubles as the cache that keeps it from being re-downloaded.
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from .ci import config as cfg
from .ci.buildkite_client import _paginate, _scrub_pii, fetch_build_detail
from .ci.log_parser import fetch_job_log_result, parse_job_results
from .ci.models import TestResult
from .ci.utils import parse_iso
from .di_labels import parse_label
from .di_pipelines import DI_KEY, DI_PIPELINES, SKIP_JOB_PATTERNS

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# The SLURM driver's verdict
# ---------------------------------------------------------------------------

# run-slurm-disagg-test.sh:337 prints exactly one line to stderr before
# exiting, and then throws the classification away:
#
#   [slurm-submit] job 4821 finished: state=workload-failed phase=workload \
#       exit=1 reason=scontrol JobState=FAILED phase=workload
#
# ``state`` is the field that distinguishes a cluster problem from a real
# accuracy regression, which is the difference between hitting retry and
# filing a bug. Buildkite's red/green discards it.
VERDICT_RE = re.compile(
    r"\[slurm-submit\]\s+job\s+(?P<slurm_job_id>\S+)\s+finished:\s+"
    r"state=(?P<state>\S*)\s+"
    r"phase=(?P<phase>\S*)\s+"
    r"exit=(?P<exit>\S*)\s*"
    r"(?:reason=(?P<reason>.*))?$",
    re.MULTILINE,
)

# Which layer failed. Derived from the driver's STATE values (see the
# assignments at run-slurm-disagg-test.sh:162-327).
_INFRA_STATES = ("infra-", "preflight-rejected", "deadline")
_BRINGUP_STATES = ("server-failed", "bringup-timeout")
_WORKLOAD_STATES = ("workload-failed", "workload-timeout", "completed-no-verdict")


def classify_state(state: str) -> str:
    """Bucket a driver STATE into infra / bringup / workload / ok / unknown."""
    s = (state or "").strip().lower()
    if not s:
        return "unknown"
    if s == "completed":
        return "ok"
    if any(s.startswith(p) for p in _INFRA_STATES):
        return "infra"
    if s in _BRINGUP_STATES:
        return "bringup"
    if s in _WORKLOAD_STATES:
        return "workload"
    if s == "failed":
        # Set by the sentinel/gate path — the workload ran and returned a verdict.
        return "workload"
    return "unknown"


def extract_verdict(log_text: Optional[str]) -> dict:
    """Return the driver's verdict fields from a job log, or {} if absent.

    Takes the last match: the driver prints the line once, but a retried
    attempt writing into the same log would leave the earlier one behind.
    """
    if not log_text:
        return {}
    match = None
    for match in VERDICT_RE.finditer(log_text):
        pass
    if match is None:
        return {}
    state = (match.group("state") or "").strip()
    return {
        "slurm_job_id": (match.group("slurm_job_id") or "").strip(),
        "slurm_state": state,
        "phase": (match.group("phase") or "").strip(),
        "reason": (match.group("reason") or "").strip(),
        "failure_class": classify_state(state),
    }


# ---------------------------------------------------------------------------
# Job records
# ---------------------------------------------------------------------------

def verdict_for(job: dict) -> str:
    """Collapse a Buildkite job state into the outcome to render.

    Applies the one correction the nightly collector already pays for: a
    ``timed_out`` job that exited 0 actually succeeded.
    """
    state = job.get("state") or "unknown"
    exit_status = job.get("exit_status")
    if state == "timed_out" and exit_status == 0:
        return "passed"
    if state in cfg.BLOCKED_JOB_STATES:
        return "blocked"
    if state in cfg.WAITING_STATES:
        return "waiting"
    if state in cfg.RUNNING_STATES:
        return "running"
    if state in ("failed", "broken"):
        return "soft_failed" if job.get("soft_failed") else "failed"
    return state


def queue_from_job(job: dict) -> str:
    """Return the agent queue this job requested, or ""."""
    for rule in job.get("agent_query_rules") or ():
        if isinstance(rule, str) and rule.startswith("queue="):
            return rule.split("=", 1)[1]
    return ""


def _secs_between(start: Optional[str], end: Optional[str]) -> Optional[float]:
    s, e = parse_iso(start), parse_iso(end)
    if s is None or e is None:
        return None
    return round((e - s).total_seconds(), 1)


def job_record(build: dict, job: dict, verdict_fields: Optional[dict] = None) -> dict:
    """Build one rich JSONL record for a (build, job) pair."""
    label = job.get("name") or ""
    cell = parse_label(label)
    record = {
        "build_number": build.get("number"),
        "build_url": build.get("web_url", ""),
        "build_state": build.get("state", ""),
        "branch": build.get("branch", ""),
        "commit": (build.get("commit") or "")[:12],
        "created_at": build.get("created_at", ""),
        "date": (build.get("created_at") or "")[:10],
        "job_id": job.get("id", ""),
        "job_url": job.get("web_url", ""),
        "label": label,
        "state": job.get("state", ""),
        "exit_status": job.get("exit_status"),
        "soft_failed": bool(job.get("soft_failed")),
        "verdict": verdict_for(job),
        "agent_name": ((job.get("agent") or {}).get("name") or ""),
        "queue": queue_from_job(job),
        # Buildkite-side wait only, and in practice sub-second: the step is a
        # SLURM launcher that starts immediately and then blocks on sbatch, so
        # the allocation wait happens inside the job's runtime, not here.
        # Measuring the real queue wait needs the driver to report it.
        "queue_wait_s": _secs_between(job.get("runnable_at"), job.get("started_at")),
        "runtime_s": _secs_between(job.get("started_at"), job.get("finished_at")),
    }
    record.update(cell.as_dict())
    record["label_ok"] = record.pop("ok")
    record.pop("raw", None)
    record.update(verdict_fields or {})
    return record


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

def fetch_di_builds(days: int = 30, cache_dir: Optional[Path] = None) -> list[dict]:
    """Fetch DI builds newest-first, back ``days``.

    Unlike ``fetch_nightly_builds`` there is no name filter — the pipeline is
    dedicated, so every build counts — and ``branch`` is optional.
    """
    pipeline = cfg.PIPELINES[DI_KEY]
    slug = pipeline["slug"]
    created_from = datetime.now(timezone.utc) - timedelta(days=days)

    params = {
        "created_from": created_from.isoformat(),
        "per_page": 100,
        "include_retried_jobs": "true",
    }
    if pipeline.get("branch"):
        params["branch"] = pipeline["branch"]

    url = f"{cfg.BK_API_BASE}/organizations/{cfg.BK_ORG}/pipelines/{slug}/builds"
    builds = _paginate(url, params)
    builds.sort(key=lambda b: b.get("created_at", ""), reverse=True)
    log.info("Fetched %d DI builds from %s/%s", len(builds), cfg.BK_ORG, slug)

    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Scrub before writing: the Buildkite build payload carries author
        # emails and avatar URLs, and this repo is public.
        scrubbed = [_scrub_pii(json.loads(json.dumps(b))) for b in builds]
        (cache_dir / "builds_di.json").write_text(json.dumps(scrubbed, indent=2))
    return builds


def di_jobs(build: dict) -> list[dict]:
    """Return the DI test steps of a build, newest attempt only.

    Keeps non-terminal jobs: a step still queued behind ``concurrency: 2`` is
    the thing the queue-wait panel exists to show. Drops superseded retries
    (Buildkite sets ``retried_in_job_id`` on the old attempt) so a retried
    step appears once, with its final outcome.
    """
    out = []
    for job in build.get("jobs") or ():
        if job.get("type") != "script":
            continue
        if job.get("retried_in_job_id"):
            continue
        name = (job.get("name") or "").lower()
        if any(skip in name for skip in SKIP_JOB_PATTERNS):
            continue
        out.append(job)
    return out


# Fields ``extract_verdict`` produces. Only these are carried forward from a
# cached record — everything else on it (state, runtime_s, agent) is re-derived
# from the API each pass and must not be resurrected from disk.
_VERDICT_KEYS = ("slurm_job_id", "slurm_state", "phase", "reason", "failure_class")

# Buildkite marks a job terminal when the agent exits, but the last log chunks
# can land afterwards — and the driver's verdict is the final line printed, so
# it is exactly what a partial flush drops. Don't call a verdict-less log
# settled until the job has been finished this long.
_LOG_SETTLE_SECONDS = 900


def _already_scanned(cached: Optional[dict]) -> bool:
    """Has this job's log already been read to a conclusion?

    ``log_scanned`` is the marker going forward. The ``failure_class`` fallback
    backfills records written before it existed: those with a verdict are
    settled by definition, and those without get re-read once and then marked.
    """
    if not cached:
        return False
    return bool(cached.get("log_scanned")) or "failure_class" in cached


def _log_has_settled(job: dict) -> bool:
    finished = parse_iso(job.get("finished_at"))
    if finished is None:
        return False
    age = (datetime.now(timezone.utc) - finished).total_seconds()
    return age >= _LOG_SETTLE_SECONDS


def _verdicts_for_jobs(
    jobs: list[dict],
    cached_by_key: Optional[dict[tuple, dict]] = None,
    build_number: Optional[int] = None,
    workers: int = 4,
) -> dict[str, dict]:
    """Extract each job's driver verdict, downloading only logs we haven't read.

    A terminal job's log never changes, so re-downloading it every three hours
    is pure waste — and at ~440 jobs per pass it was the whole of this repo's
    Buildkite traffic. ``scripts/collect_ci.py`` already caches on the same
    principle.
    """
    cached_by_key = cached_by_key or {}

    def one(job: dict) -> tuple[str, dict]:
        job_id = job.get("id", "")
        if job.get("state") not in cfg.TERMINAL_STATES:
            return job_id, {}

        cached = cached_by_key.get((build_number, job_id))
        if _already_scanned(cached):
            fields = {k: cached[k] for k in _VERDICT_KEYS if k in cached}
            fields["log_scanned"] = True
            return job_id, fields

        try:
            text, scanned = fetch_job_log_result(job)
        except Exception as exc:  # a missing log must not fail the pass
            log.warning("  verdict fetch failed for %s: %s", job.get("name"), exc)
            return job_id, {}

        fields = extract_verdict(text)
        # A log with no verdict is only final once it has stopped growing;
        # marking it early would freeze in an answer we read too soon.
        if scanned and (fields or _log_has_settled(job)):
            fields["log_scanned"] = True
        return job_id, fields

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return {jid: fields for jid, fields in pool.map(one, jobs)}


# ---------------------------------------------------------------------------
# JSONL persistence
# ---------------------------------------------------------------------------

def _record_key(record: dict) -> tuple:
    return (record.get("build_number"), record.get("job_id"))


def load_job_records(path: Path) -> list[dict]:
    """Read jobs.jsonl, tolerating a truncated final line."""
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            log.warning("Skipping malformed line in %s", path)
    return records


def upsert_job_records(path: Path, new_records: list[dict]) -> int:
    """Merge records into jobs.jsonl, keyed on (build_number, job_id).

    Not a pure append: a job seen while running must be replaced once it
    finishes. Rewriting the whole file keeps it sorted and makes the
    collector idempotent — running it twice produces byte-identical output.
    """
    merged = {_record_key(r): r for r in load_job_records(path)}
    for record in new_records:
        merged[_record_key(record)] = record
    ordered = sorted(
        merged.values(),
        key=lambda r: (r.get("build_number") or 0, r.get("label") or ""),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in ordered)
    )
    return len(ordered)


def load_di_results(results_dir: Path) -> list[tuple[int, str, list[TestResult]]]:
    """Load persisted TestResult rows as (build_number, date, results), oldest-first.

    Files are named ``{date}_b{build}_di.jsonl`` — build number is in the name
    because, unlike the nightlies, DI can run several builds in one day.
    """
    entries = []
    if not results_dir.exists():
        return entries
    for path in sorted(results_dir.glob("*_di.jsonl")):
        results = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            d.setdefault("step_id", "")
            results.append(TestResult(**d))
        if results:
            entries.append((results[0].build_number, results[0].date, results))
    entries.sort(key=lambda e: (e[1], e[0]))
    return entries


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def collect(
    days: int,
    output_dir: Path,
    dry_run: bool = False,
    fetch_logs: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Run one collection pass.

    Returns (builds, job_records).
    """
    slug = cfg.PIPELINES[DI_KEY]["slug"]
    builds = fetch_di_builds(days, cache_dir=output_dir / ".cache")
    if not builds:
        log.warning("No DI builds found in the last %d days", days)
        return [], []

    if dry_run:
        for b in builds:
            log.info(
                "  Build #%s  %-9s  %s  %s  %r",
                b.get("number"), b.get("state", ""),
                (b.get("created_at") or "")[:16],
                b.get("branch", ""), (b.get("message") or "")[:50],
            )
        return builds, []

    results_dir = output_dir / "test_results"
    all_records: list[dict] = []
    # Read once, not per build: the file holds every record ever collected.
    cached_by_key = {_record_key(r): r for r in load_job_records(output_dir / "jobs.jsonl")}

    for build in builds:
        build_num = build.get("number")
        if not build.get("jobs"):
            detail = fetch_build_detail(DI_KEY, build_num)
            build.clear()
            build.update(detail)

        jobs = di_jobs(build)
        if not jobs:
            log.info("  Build #%s: no DI steps", build_num)
            continue

        verdicts = (
            _verdicts_for_jobs(jobs, cached_by_key, build_num) if fetch_logs else {}
        )
        records = [job_record(build, j, verdicts.get(j.get("id", ""))) for j in jobs]
        all_records.extend(records)

        date = (build.get("created_at") or "")[:10]
        terminal = [j for j in jobs if j.get("state") in cfg.TERMINAL_STATES]
        results: list[TestResult] = []
        for job in terminal:
            # log_text="" forces the job-level fallback — see module docstring.
            results.extend(parse_job_results(job, build_num, slug, date, log_text=""))

        log.info(
            "  Build #%s (%s): %d steps, %d terminal, %d result rows",
            build_num, date, len(jobs), len(terminal), len(results),
        )

        if results:
            from .ci.reporter import write_test_results
            write_test_results(results, f"{date}_b{build_num}", "di", results_dir)

    total = upsert_job_records(output_dir / "jobs.jsonl", all_records)
    log.info("jobs.jsonl now holds %d records (%d from this pass)", total, len(all_records))
    return builds, all_records


def configure() -> None:
    """Point the shared CI framework at the DI pipeline for this process."""
    from .ci.analyzer import set_default_hardware
    from .di_pipelines import BK_ORG, DI_HARDWARE

    cfg.configure(BK_ORG, DI_PIPELINES)
    # DI labels carry no hardware tag, and the analyzer's fallback for an
    # untagged name is the upstream default "h100" — actively wrong here.
    set_default_hardware(DI_HARDWARE)
