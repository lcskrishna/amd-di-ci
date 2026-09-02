"""Parse pytest results from Buildkite job logs.

Since vLLM CI does not upload JUnit XML artifacts, we extract test results
from the pytest console output found in each job's log.

Extracts:
- Individual FAILED/ERROR test names from 'short test summary info' section
- Aggregate counts from the pytest summary line (e.g., '4 passed, 1 failed in 30s')
- Creates TestResult objects for each test group / individual failure
"""

import logging
import re
from typing import Optional

import requests

from . import config as cfg
from . import ratelimit
from .models import TestResult

log = logging.getLogger(__name__)

# Patterns for parsing pytest output
PYTEST_SUMMARY_RE = re.compile(
    r"=+\s+(.*(?:passed|failed|error).*)\s+in\s+([\d.]+)s"
)
PYTEST_COUNT_RE = re.compile(
    r"(\d+)\s+(passed|failed|error|warning|skipped|deselected|xfailed|xpassed)"
)
FAILED_TEST_RE = re.compile(
    r"^FAILED\s+(\S+?)(?:\s+-\s+(.*))?$"
)
ERROR_TEST_RE = re.compile(
    r"^ERROR\s+(\S+?)(?:\s+-\s+(.*))?$"
)
# ANSI escape and Buildkite timestamp cleanup
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
BK_TS_RE = re.compile(r"_bk;t=\d+")
LOG_TS_RE = re.compile(r"^\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\]\s*")

# Primary physical-node source: the Buildkite agent's ``k8s:node=<node>`` tag,
# present in ``job["agent"]["meta_data"]`` for ~all AMD GPU runners
# (mi250/mi300/mi325/mi355) and available without fetching the log.
K8S_NODE_PREFIX = "k8s:node="

# Fallback for the rare case the agent tag is missing: the physical-node banner
# that AMD MI325 runners print into the log:
#   "=== Pod: buildkite-...-55rwd | Node: chi-mi325x-pod2-032 | Tue Jul 14 ... ==="
# The "Pod: <pod> | Node: <node>" structure disambiguates the physical node from
# unrelated "Node: 0" GPU-topology lines elsewhere in the log.
NODE_BANNER_RE = re.compile(
    r"Pod:\s*\S+\s*\|\s*Node:\s*([A-Za-z0-9._-]+)", re.IGNORECASE
)


def node_from_agent(job: dict) -> str:
    """Return the physical node from a job's Buildkite agent ``k8s:node`` tag.

    The agent ``meta_data`` is embedded in the build JSON both the list and
    detail endpoints already return, so this needs no extra request. Returns ""
    when the job has no agent tags (e.g. CPU-only steps).
    """
    agent = job.get("agent") or {}
    for tag in agent.get("meta_data") or []:
        if isinstance(tag, str) and tag.startswith(K8S_NODE_PREFIX):
            return tag[len(K8S_NODE_PREFIX):].strip()
    return ""


def _clean_line(line: str) -> str:
    """Remove ANSI codes, Buildkite timestamps, and log timestamps."""
    line = ANSI_RE.sub("", line)
    line = BK_TS_RE.sub("", line)
    line = LOG_TS_RE.sub("", line)
    return line.strip()


def extract_node(log_text: Optional[str]) -> str:
    """Return the physical CI agent hostname from a job log, or "" if absent.

    Matches the ``Pod: <pod> | Node: <node>`` banner that AMD MI325 runners emit
    near the start of the log. The search runs against the raw text: the banner
    body is plain ASCII and any leading Buildkite escape codes precede it, so no
    line cleaning is required. Returns the first match; runners print it once.
    """
    if not log_text:
        return ""
    match = NODE_BANNER_RE.search(log_text)
    return match.group(1).strip() if match else ""


# Reusable HTTP session for connection pooling
_session: Optional[requests.Session] = None


def _get_session() -> requests.Session:
    """Get or create a reusable HTTP session."""
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers["Authorization"] = f"Bearer {cfg.BK_TOKEN}"
    return _session


def fetch_job_log_result(job: dict) -> tuple[Optional[str], bool]:
    """Download a job's raw log, reporting whether the log was actually read.

    Returns ``(text, scanned)``. A ``None`` text has four causes — no log URL,
    no token, retries exhausted, or an empty body — and only the last of those
    means we saw the log and it held nothing. Callers that cache "already
    looked at this" need to tell those apart, hence the second value.

    The Buildkite API returns JSON by default with the log in the "content"
    field. We request text/plain for the raw log, falling back to extracting
    from JSON if the response is JSON.
    """
    log_url = job.get("raw_log_url")
    if not log_url:
        return None, False

    if not cfg.BK_TOKEN:
        return None, False

    session = _get_session()
    for attempt in range(1, 4):
        try:
            ratelimit.acquire()
            resp = session.get(
                log_url,
                timeout=60,
                headers={"Accept": "text/plain"},
            )
            ratelimit.observe(resp.headers)
            if resp.status_code == 429:
                import time
                wait = int(resp.headers.get("Retry-After", 5 * attempt))
                log.warning("Rate limited fetching log, waiting %ds", wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            text = resp.text
            # If the response is JSON (API didn't honor Accept: text/plain),
            # extract the "content" field which has the actual log text.
            if text.startswith('{"'):
                try:
                    data = resp.json()
                    if "content" in data:
                        text = data["content"]
                except Exception:
                    pass
            return text, True
        except Exception as e:
            if attempt < 3:
                import time
                time.sleep(2 * attempt)
                continue
            log.warning("Failed to fetch log for job %s: %s", job.get("name"), e)
            return None, False
    return None, False


def fetch_job_log(job: dict) -> Optional[str]:
    """Download the raw log for a Buildkite job."""
    return fetch_job_log_result(job)[0]


def parse_pytest_log(
    log_text: str,
    job_name: str,
    job_id: str,
    step_id: str,
    build_number: int,
    pipeline: str,
    date: str,
) -> list[TestResult]:
    """Parse pytest output from a job log.

    Strategy:
    1. Find the pytest summary line to get aggregate counts
    2. Find 'short test summary info' to get individual FAILED/ERROR test names
    3. For passed tests, create a single summary TestResult per job
       (we can't get individual pass names from the log)
    4. For failed/error tests, create individual TestResults from the summary section

    Returns:
        List of TestResult objects.
    """
    lines = log_text.split("\n")
    clean_lines = [_clean_line(l) for l in lines]

    # Find pytest summary line (search from end)
    counts = {}
    total_duration = 0.0
    for line in reversed(clean_lines):
        m = PYTEST_SUMMARY_RE.search(line)
        if m:
            summary_text = m.group(1)
            total_duration = float(m.group(2))
            for cm in PYTEST_COUNT_RE.finditer(summary_text):
                counts[cm.group(2)] = int(cm.group(1))
            break

    if not counts:
        return []

    results = []
    passed = counts.get("passed", 0)
    failed = counts.get("failed", 0)
    errors = counts.get("error", 0)
    skipped = counts.get("skipped", 0)
    xfailed = counts.get("xfailed", 0)
    xpassed = counts.get("xpassed", 0)
    deselected = counts.get("deselected", 0)

    # Find individual failed/error tests from 'short test summary info'
    failed_tests = []
    in_summary = False
    for line in clean_lines:
        if "short test summary info" in line:
            in_summary = True
            continue
        if in_summary:
            # Summary section ends at the pytest result line
            if PYTEST_SUMMARY_RE.search(line):
                break
            fm = FAILED_TEST_RE.match(line)
            if fm:
                failed_tests.append(("failed", fm.group(1), fm.group(2) or ""))
                continue
            em = ERROR_TEST_RE.match(line)
            if em:
                failed_tests.append(("error", em.group(1), em.group(2) or ""))
                continue

    # Create TestResult for each individually-identified failure
    seen_failures = set()
    for status, test_path, message in failed_tests:
        # test_path is like "tests/test_foo.py::TestClass::test_method"
        # or "tests/test_foo.py::test_method[param]"
        parts = test_path.rsplit("::", 1)
        if len(parts) == 2:
            classname = parts[0]
            name = parts[1]
        else:
            classname = job_name
            name = test_path

        test_id = f"{classname}::{name}"
        seen_failures.add(test_id)

        results.append(TestResult(
            test_id=test_id,
            name=name,
            classname=classname,
            status=status,
            duration_secs=0.0,
            failure_message=message[:500],
            job_name=job_name,
            job_id=job_id,
            step_id=step_id,
            build_number=build_number,
            pipeline=pipeline,
            date=date,
        ))

    # For failures not individually identified, create generic entries
    unaccounted_failures = max(0, failed - len([t for t in failed_tests if t[0] == "failed"]))
    unaccounted_errors = max(0, errors - len([t for t in failed_tests if t[0] == "error"]))
    if unaccounted_failures > 0:
        results.append(TestResult(
            test_id=f"{job_name}::__unidentified_failures__",
            name=f"__unidentified_failures__ ({unaccounted_failures})",
            classname=job_name,
            status="failed",
            duration_secs=0.0,
            failure_message=f"{unaccounted_failures} failures not individually identified in log",
            job_name=job_name,
            job_id=job_id,
            step_id=step_id,
            build_number=build_number,
            pipeline=pipeline,
            date=date,
        ))
    if unaccounted_errors > 0:
        results.append(TestResult(
            test_id=f"{job_name}::__unidentified_errors__",
            name=f"__unidentified_errors__ ({unaccounted_errors})",
            classname=job_name,
            status="error",
            duration_secs=0.0,
            failure_message=f"{unaccounted_errors} errors not individually identified in log",
            job_name=job_name,
            job_id=job_id,
            step_id=step_id,
            build_number=build_number,
            pipeline=pipeline,
            date=date,
        ))

    # Create a summary entry for passed tests (grouped by job)
    if passed > 0:
        results.append(TestResult(
            test_id=f"{job_name}::__passed__",
            name=f"__passed__ ({passed})",
            classname=job_name,
            status="passed",
            duration_secs=total_duration,
            failure_message="",
            job_name=job_name,
            job_id=job_id,
            step_id=step_id,
            build_number=build_number,
            pipeline=pipeline,
            date=date,
        ))

    # Skipped
    if skipped > 0:
        results.append(TestResult(
            test_id=f"{job_name}::__skipped__",
            name=f"__skipped__ ({skipped})",
            classname=job_name,
            status="skipped",
            duration_secs=0.0,
            failure_message="",
            job_name=job_name,
            job_id=job_id,
            step_id=step_id,
            build_number=build_number,
            pipeline=pipeline,
            date=date,
        ))

    # xfailed
    if xfailed > 0:
        results.append(TestResult(
            test_id=f"{job_name}::__xfailed__",
            name=f"__xfailed__ ({xfailed})",
            classname=job_name,
            status="xfailed",
            duration_secs=0.0,
            failure_message="",
            job_name=job_name,
            job_id=job_id,
            step_id=step_id,
            build_number=build_number,
            pipeline=pipeline,
            date=date,
        ))

    return results


def parse_job_results(
    job: dict,
    build_number: int,
    pipeline: str,
    date: str,
    log_text: Optional[str] = None,
) -> list[TestResult]:
    """Parse test results from a single Buildkite job.

    Falls back to job-level status if log parsing fails.

    Args:
        job: Buildkite job dict
        build_number: Build number
        pipeline: Pipeline slug
        date: ISO date
        log_text: Pre-fetched log text (if None, will be fetched)

    Returns:
        List of TestResult objects
    """
    job_name = job.get("name", "unknown")
    job_id = job.get("id", "")
    step_id = (job.get("step") or {}).get("id", "")
    job_state = job.get("state", "unknown")

    if log_text is None:
        log_text = fetch_job_log(job)

    # Physical CI agent hostname, stamped onto every result this job produces so
    # downstream per-agent analytics can join test groups to the box that ran
    # them. Prefer the agent's k8s:node tag (covers all AMD GPU queues, no log
    # needed); fall back to the MI325 log banner. "" when neither is available.
    node = node_from_agent(job) or extract_node(log_text)

    def _stamp(results: list[TestResult]) -> list[TestResult]:
        for r in results:
            r.node = node
        return results

    if log_text:
        results = parse_pytest_log(
            log_text, job_name, job_id, step_id, build_number, pipeline, date
        )
        if results:
            # CRITICAL: If the Buildkite job state is "passed" but the log parser
            # found failures, trust the job state. The log may contain output from
            # retried/subprocess attempts that ultimately succeeded.
            if job_state == "passed":
                has_failures = any(
                    r.status in ("failed", "error")
                    for r in results
                )
                if has_failures:
                    log.info(
                        "    Job %s passed but log has failures — overriding "
                        "to trust Buildkite job state",
                        job_name,
                    )
                    # Remove failure entries, keep passed/skipped
                    results = [
                        r for r in results
                        if r.status not in ("failed", "error")
                    ]
                    # If no passed entry exists, add a job-level pass
                    if not any(r.status == "passed" for r in results):
                        results.append(TestResult(
                            test_id=f"{job_name}::__job_level__",
                            name="__job_level__",
                            classname=job_name,
                            status="passed",
                            duration_secs=0.0,
                            failure_message="",
                            job_name=job_name,
                            job_id=job_id,
                            step_id=step_id,
                            build_number=build_number,
                            pipeline=pipeline,
                            date=date,
                        ))
            # CRITICAL: If the Buildkite job state is "failed" but the log parser
            # found ONLY passes (no failures), the log likely contains output from
            # a retry that passed over the original failure. Trust the job state.
            if job_state in ("failed", "timed_out", "broken"):
                has_failures = any(
                    r.status in ("failed", "error")
                    for r in results
                )
                if not has_failures:
                    log.info(
                        "    Job %s failed but log shows only passes — "
                        "adding job-level failure (retry log may have overwritten original)",
                        job_name,
                    )
                    results.append(TestResult(
                        test_id=f"{job_name}::__unidentified_failures__",
                        name="__unidentified_failures__ (1)",
                        classname=job_name,
                        status="failed",
                        duration_secs=0.0,
                        failure_message=f"Job state: {job_state} (log shows passes but job failed — likely retry output)",
                        job_name=job_name,
                        job_id=job_id,
                        step_id=step_id,
                        build_number=build_number,
                        pipeline=pipeline,
                        date=date,
                    ))
            return _stamp(results)

    # Fallback: create a single TestResult from job state
    # Blocked jobs never ran — skip them entirely (not an error)
    if job_state == "blocked":
        return []

    status_map = {
        "passed": "passed",
        "failed": "failed",
        "timed_out": "error",
        "broken": "error",
        "canceled": "canceled",
    }
    status = status_map.get(job_state, "error")

    return _stamp([TestResult(
        test_id=f"{job_name}::__job_level__",
        name="__job_level__",
        classname=job_name,
        status=status,
        duration_secs=0.0,
        failure_message=f"Job state: {job_state} (no pytest output in log)",
        job_name=job_name,
        job_id=job_id,
        step_id=step_id,
        build_number=build_number,
        pipeline=pipeline,
        date=date,
    )])
