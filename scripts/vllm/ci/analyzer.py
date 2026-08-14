"""Test health analysis: labeling, parity comparison, trend detection.

Core analysis engine for the CI dashboard backend.
"""

import json
import logging
import re
from collections import defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from typing import Optional

import yaml

from . import config as cfg
from .models import BuildSummary, ParityEntry, TestHealth, TestResult

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Label normalization (adapted from vllm_ci_parity.py)
# ---------------------------------------------------------------------------

# Known GPU/hardware patterns in parentheses — matches (H100), (mi325), (2xA100), etc.
# Keeps parentheticals like (Standard), (CPU), (8 GPUs) intact.
_HW_TOKEN = (
    r'(?:\d+\s*[xX\s]\s*)?'                      # optional multiplier: 2x, 4x, "1 ", "2 "
    r'(?:H\d+\w*|A\d+\w*|B\d+\w*|L\d+\w*'       # NVIDIA: H100, A100, B200, L40, H100s
    r'|MI?\d+\w*|mi\d+\w*'                       # AMD: MI300X, mi325, mi355
    r'|GB\d+\w*|GH\d+\w*'                        # NVIDIA arch: GB200, GH200
    r')s?'                                        # optional trailing 's' (e.g., "H100s")
)
# Single-HW tag: (H100), (MI325), (B200) — just a queue identifier, safe to strip
_HW_SINGLE = re.compile(
    r'\s*\(\s*' + _HW_TOKEN + r'\s*\)',
    re.IGNORECASE,
)
# Multi-HW tag: (H100-MI325), (2xH100-2xMI355), (A100-MI325) — test config, keep it
_HW_MULTI = re.compile(
    r'\s*\(\s*'
    + _HW_TOKEN +
    r'(?:\s*[-]\s*' + _HW_TOKEN + r')+'          # one or more dash-separated HW (required)
    r'\s*\)',
    re.IGNORECASE,
)
# Combined pattern: matches both single and multi (used for _extract_hardware)
_HW_PATTERN = re.compile(
    r'\s*\(\s*'
    + _HW_TOKEN +
    r'(?:\s*[-]\s*' + _HW_TOKEN + r')*'          # optional dash-separated additional HW
    r'\s*\)',
    re.IGNORECASE,
)

# Hardware prefixes in Buildkite job names: "mi250_1: ", "mi325_8: ", "gpu_1: "
_JOB_PREFIX_RE = re.compile(
    r'^(mi\d+_\d+|mi\d+|gpu_\d+|amd_\w+):\s*',
    re.IGNORECASE,
)


def _normalize_job_name(name: str) -> str:
    """Normalize a Buildkite job name for cross-pipeline matching.

    Strips:
    - Hardware prefixes like 'mi250_1: ', 'gpu_1: '
    - Hardware tags in parens like (H100), (mi325), (B200-MI355)
    - Trailing '# comment'
    - '%N' parallelism marker
    - Shard indices from %N expansion ONLY (e.g., "LoRA 0" -> "LoRA")
    - Extra whitespace

    Does NOT strip numbers that are part of the actual group name
    (e.g., "Extended Generation 1" stays as-is, "Standard 1: qwen2" stays).

    Adapted from vllm_ci_parity.py normalize_label().
    """
    s = _JOB_PREFIX_RE.sub('', name)
    s = re.sub(r'#.*$', '', s).strip()
    s = re.sub(r'\s*%N\s*$', '', s).strip()
    # Convert SINGLE-HW GPU-count tags to plain GPU count:
    #   (4xH100) → (4 GPUs)        — upstream single-HW with count
    #   (2xB200) → (2 GPUs)        — upstream single-HW with count
    # Multi-HW count tags are KEPT — they are cross-hardware test configs:
    #   (4xH100-4xMI325)           — kept as-is (different test from plain 4 GPUs)
    #   (2xH100-2xMI355)           — kept as-is
    # Tags WITHOUT a count like (H200), (B200) are left as-is — they
    # identify which hardware the test runs on.
    s = re.sub(
        r'\s*\(\s*(\d+)\s*[xX]\s*' + _HW_TOKEN +
        r'\s*\)',
        lambda m: f' ({m.group(1)} GPUs)',
        s,
        flags=re.IGNORECASE,
    )
    # Normalize version-like dots to hyphens (e.g., "Qwen3.5" → "Qwen3-5")
    s = re.sub(r'(\d)\.(\d)', r'\1-\2', s)
    s = re.sub(r'\s+', ' ', s).strip()

    # Only strip trailing shard index for known %N-expanded patterns.
    # Use the global shard bases list (populated from YAML or auto-detected).
    lower = s.lower()
    for base in _SHARD_BASES:
        if lower.startswith(base) and len(lower) > len(base):
            rest = lower[len(base):]
            # Match " N" (bare shard index) at end
            if re.match(r'^\s+\d+\s*$', rest):
                s = s[:len(base)]
                break
    return s.lower()


_PARITY_KEY_OVERRIDES: dict[str, str] = {}


def set_parity_key_overrides(overrides: dict[str, str] | None):
    """Set YAML-derived parity-key overrides for runtime job names.

    Some upstream labels omit GPU counts that are present as YAML metadata
    (``num_devices``), while AMD labels encode the same count in the label
    itself, e.g. ``DeepSeek V2-Lite Accuracy`` vs
    ``DeepSeek V2-Lite Accuracy (4xH100-4xMI300)``.  Runtime Buildkite job
    names no longer carry the YAML fields, so collect_ci loads those fields
    up front and installs normalized-label -> parity-key overrides here.
    """
    global _PARITY_KEY_OVERRIDES
    _PARITY_KEY_OVERRIDES = {}
    if not overrides:
        return
    for label, key in overrides.items():
        if not label or not key:
            continue
        _PARITY_KEY_OVERRIDES[_normalize_job_name(str(label))] = str(key).lower()


def _parity_key_base(name: str) -> str:
    """Normalize for cross-pipeline parity matching without YAML overrides.

    _normalize_job_name keeps multi-HW count tags like (4xH100-4xMI325)
    because they're distinct tests at the display level.  _parity_key
    converts them to (N GPUs) so AMD (4xH100-4xMI325) matches upstream
    (4xH100) — both become (4 GPUs).

    Bare HW tags without counts like (H200), (B200) are stripped — they
    identify queue hardware, not test configuration.

    GPU counts (N GPUs) are KEPT because different counts = different tests.
    """
    s = _normalize_job_name(name)
    # Convert multi-HW count tags to plain GPU count for parity matching:
    #   (4xh100-4xmi325) → (4 GPUs)  — matches upstream (4xH100) → (4 GPUs)
    #   (2xh100-2xmi355) → (2 GPUs)  — matches upstream (2xH100) → (2 GPUs)
    s = re.sub(
        r'\s*\(\s*(\d+)\s*[xX]\s*' + _HW_TOKEN +
        r'(?:\s*[-]\s*\d+\s*[xX]\s*' + _HW_TOKEN + r')+' +
        r'\s*\)',
        lambda m: f' ({m.group(1)} gpus)',
        s,
        flags=re.IGNORECASE,
    )
    # Strip remaining single-HW tags without count: (H200), (B200), etc.
    s = _HW_SINGLE.sub('', s)
    # Strip remaining multi-HW tags without count (no GPU count prefix)
    s = _HW_MULTI.sub('', s)
    # Lowercase for case-insensitive matching: "(4 GPUs)" == "(4 gpus)"
    return re.sub(r'\s+', ' ', s).strip().lower()


def _parity_key(name: str) -> str:
    """Normalize for cross-pipeline parity matching."""
    normalized = _normalize_job_name(name)
    return _PARITY_KEY_OVERRIDES.get(normalized) or _parity_key_base(normalized)


def _parity_family_name(name: str) -> str:
    """Return the canonical family label shared across parity variants.

    This is intentionally hardware-agnostic. For example:
    - ``Distributed Tests (2 GPUs)(H100)``
    - ``Distributed Tests (2xH100-2xMI300)``
    - ``Distributed Tests (2xH100-2xMI355)``

    all map to the same family label: ``distributed tests (2 gpus)``.

    Downstream views use this to collapse one upstream identity that fans out
    across multiple AMD hardware variants without losing the per-variant raw
    rows in ``parity_report.json``.
    """
    return _parity_key(name)


# Shard bases — auto-populated from YAML %N parallelism steps.
# Set at runtime by collect_ci.py via set_shard_bases(), or loaded
# from shard_bases.json as fallback for direct imports/tests.
_SHARD_BASES: list[str] = []


def set_shard_bases(bases: list[str]):
    """Set the shard bases list (called by collect_ci.py after YAML extraction)."""
    global _SHARD_BASES
    _SHARD_BASES = [b.lower() for b in bases]


def _load_shard_bases_fallback():
    """Try to load shard_bases.json from known locations."""
    global _SHARD_BASES
    if _SHARD_BASES:
        return
    from pathlib import Path as _Path
    for p in [
        _Path(__file__).resolve().parent.parent.parent.parent / "data" / "vllm" / "ci" / "shard_bases.json",
        _Path("data/vllm/ci/shard_bases.json"),
    ]:
        if p.exists():
            try:
                _SHARD_BASES = [b.lower() for b in json.loads(p.read_text())]
                return
            except Exception:
                pass


_load_shard_bases_fallback()


# Tests that should be excluded from parity comparison
# (not relevant to AMD GPU testing)
_EXCLUDE_PATTERNS = re.compile(
    r'^(cpu[-\s]|arm\s|ascend\s|intel\s|gh200\s|amd:\s)',
    re.IGNORECASE,
)


def commands_similarity(cmds_a: list[str], cmds_b: list[str]) -> float:
    """Compare two command lists, ignoring env-specific differences.

    Adapted from vllm_ci_parity.py commands_similarity().
    """
    def clean(cmd: str) -> str:
        cmd = re.sub(r'export\s+\w+=\S+', '', cmd).strip()
        cmd = re.sub(r'(CUDA_VISIBLE_DEVICES|HIP_VISIBLE_DEVICES)=\S+\s*', '', cmd)
        # AMD and upstream use the same pytest target with a platform-specific
        # target-suite selector.  Treat the selector as execution metadata,
        # just like the visibility variables above, while retaining other
        # leading assignments that can materially change test coverage.
        cmd = re.sub(
            r'^(?:VLLM_)?TARGET_TEST_SUITE=(?:"[^"]*"|\'[^\']*\'|\S+)\s*',
            '',
            cmd,
        )
        cmd = re.sub(r'--shard-id=\$\$\w+', '--shard-id=N', cmd)
        cmd = re.sub(r'--num-shards=\$\$\w+', '--num-shards=N', cmd)
        return cmd.strip()

    filtered_a = [clean(c) for c in cmds_a if clean(c)]
    filtered_b = [clean(c) for c in cmds_b if clean(c)]

    if not filtered_a and not filtered_b:
        return 1.0
    if not filtered_a or not filtered_b:
        return 0.0

    return SequenceMatcher(None, '\n'.join(filtered_a), '\n'.join(filtered_b)).ratio()


def similarity_color(score: float) -> str:
    """Return a color name for a similarity score (for display/reporting)."""
    if score >= 0.9:
        return "green"
    elif score >= 0.7:
        return "yellow"
    elif score >= 0.5:
        return "orange"
    return "red"


# ---------------------------------------------------------------------------
# Test health labeling
# ---------------------------------------------------------------------------

def _extract_module(test_id: str) -> str:
    """Extract module/area from a test_id like 'tests.models.test_llama::test_foo'."""
    parts = test_id.split("::")
    classname = parts[0] if parts else test_id
    # Use first two dotted segments as module
    segments = classname.split(".")
    if len(segments) >= 2:
        return ".".join(segments[:2])
    return segments[0] if segments else "unknown"


def label_test_health(
    test_id: str,
    history: list[str],
    dates: list[str],
    durations: list[float],
) -> TestHealth:
    """Assign a health label to a test based on its recent history.

    Args:
        test_id: Canonical test identifier
        history: List of statuses (oldest first), e.g. ["passed", "passed", "failed"]
        dates: Corresponding dates for each status
        durations: Corresponding durations for each status

    Returns:
        TestHealth object with computed label and metrics
    """
    appearances = len(history)

    # Use the most recent FLAKY_WINDOW entries for analysis
    window = history[-cfg.FLAKY_WINDOW:]
    window_dates = dates[-cfg.FLAKY_WINDOW:]

    # Count statuses in window
    pass_count = sum(1 for s in window if s in ("passed", "xpassed"))
    fail_count = sum(1 for s in window if s in ("failed", "error"))
    skip_count = sum(1 for s in window if s in ("skipped", "xfailed"))
    active_count = pass_count + fail_count  # exclude skips from rate calc

    # All skipped
    if active_count == 0 and skip_count > 0:
        label = "skipped"
        pass_rate = 0.0
    elif active_count == 0:
        label = "skipped"
        pass_rate = 0.0
    else:
        pass_rate = pass_count / active_count

        if appearances <= 2:
            label = "new_test"
        elif pass_rate >= cfg.FLAKY_MAX_RATE:
            # Check if it was recently fixed (failing before, passing now)
            prior = history[:-cfg.NEW_FAILURE_WINDOW] if len(history) > cfg.NEW_FAILURE_WINDOW else []
            recent = history[-cfg.NEW_FAILURE_WINDOW:]
            prior_had_failures = any(s in ("failed", "error") for s in prior[-cfg.FLAKY_WINDOW:])
            recent_all_pass = all(s in ("passed", "xpassed", "skipped", "xfailed") for s in recent)
            if prior_had_failures and recent_all_pass:
                label = "fixed"
            else:
                label = "passing"
        elif pass_rate <= cfg.FLAKY_MIN_RATE:
            # Check if this is a new failure (was passing before)
            # Look at history BEFORE the current window
            prior = history[:-cfg.FLAKY_WINDOW] if len(history) > cfg.FLAKY_WINDOW else []
            if prior:
                prior_pass = sum(1 for s in prior if s in ("passed", "xpassed"))
                prior_active = sum(1 for s in prior if s in ("passed", "xpassed", "failed", "error"))
                prior_rate = prior_pass / prior_active if prior_active > 0 else 0
                if prior_rate >= cfg.FLAKY_MAX_RATE:
                    label = "new_failure"
                else:
                    label = "failing"
            else:
                label = "failing"
        else:
            label = "flaky"

    # Compute failure streak (consecutive failures from most recent)
    failure_streak = 0
    for s in reversed(history):
        if s in ("failed", "error"):
            failure_streak += 1
        else:
            break

    # First failure date in current streak
    first_failure = None
    if failure_streak > 0:
        idx = len(history) - failure_streak
        if idx < len(dates):
            first_failure = dates[idx]

    # Mean duration (excluding zero/skip)
    valid_durations = [d for d in durations if d > 0]
    mean_dur = sum(valid_durations) / len(valid_durations) if valid_durations else 0.0

    # Compact history for display (last FLAKY_WINDOW entries)
    compact_history = []
    for s in window:
        if s in ("passed", "xpassed"):
            compact_history.append("P")
        elif s in ("failed", "error"):
            compact_history.append("F")
        elif s == "skipped":
            compact_history.append("S")
        elif s == "xfailed":
            compact_history.append("X")
        else:
            compact_history.append("?")

    return TestHealth(
        test_id=test_id,
        label=label,
        pass_rate=pass_rate,
        appearances=appearances,
        last_seen=dates[-1] if dates else "",
        first_failure=first_failure,
        failure_streak=failure_streak,
        history=compact_history,
        module=_extract_module(test_id),
        mean_duration=mean_dur,
    )


# ---------------------------------------------------------------------------
# Build health across all tests
# ---------------------------------------------------------------------------

def compute_all_test_health(
    results_by_build: list[tuple[int, str, list[TestResult]]],
) -> list[TestHealth]:
    """Compute health labels for all tests across multiple builds.

    Args:
        results_by_build: List of (build_number, date, results) tuples,
                          sorted oldest-first.

    Returns:
        List of TestHealth objects.
    """
    # Collect per-test history
    test_history: dict[str, list[str]] = defaultdict(list)
    test_dates: dict[str, list[str]] = defaultdict(list)
    test_durations: dict[str, list[float]] = defaultdict(list)

    for build_num, date, results in results_by_build:
        # Get the status per test for this build (use worst status if duplicates)
        build_tests: dict[str, tuple[str, float]] = {}
        for r in results:
            existing = build_tests.get(r.test_id)
            if existing is None:
                build_tests[r.test_id] = (r.status, r.duration_secs)
            else:
                # Keep the worst status (failed > error > skipped > passed)
                priority = {"failed": 0, "error": 1, "xfailed": 2, "skipped": 3, "xpassed": 4, "passed": 5}
                if priority.get(r.status, 5) < priority.get(existing[0], 5):
                    build_tests[r.test_id] = (r.status, r.duration_secs)

        for test_id, (status, duration) in build_tests.items():
            test_history[test_id].append(status)
            test_dates[test_id].append(date)
            test_durations[test_id].append(duration)

    # Label each test
    health_list = []
    for test_id in sorted(test_history.keys()):
        health = label_test_health(
            test_id,
            test_history[test_id],
            test_dates[test_id],
            test_durations[test_id],
        )
        health_list.append(health)

    return health_list


# ---------------------------------------------------------------------------
# Parity computation
# ---------------------------------------------------------------------------

def compute_parity(
    amd_results: list[TestResult],
    upstream_results: list[TestResult],
) -> dict:
    """Compare AMD vs upstream test results for parity analysis.

    Args:
        amd_results: Test results from the latest AMD nightly
        upstream_results: Test results from the latest upstream nightly

    Returns:
        Parity report dict with summary, per-module breakdown, and details.
    """
    # Build test_id -> best status maps
    def best_status(results: list[TestResult]) -> dict[str, str]:
        status_map = {}
        priority = {"passed": 0, "xpassed": 1, "failed": 2, "error": 3, "skipped": 4, "xfailed": 5}
        for r in results:
            existing = status_map.get(r.test_id)
            if existing is None or priority.get(r.status, 5) < priority.get(existing, 5):
                status_map[r.test_id] = r.status
        return status_map

    amd_map = best_status(amd_results)
    upstream_map = best_status(upstream_results)

    all_tests = set(amd_map.keys()) | set(upstream_map.keys())

    entries = []
    summary = defaultdict(int)
    module_stats = defaultdict(lambda: defaultdict(int))

    for test_id in sorted(all_tests):
        amd_s = amd_map.get(test_id, "missing")
        up_s = upstream_map.get(test_id, "missing")

        if amd_s == "missing":
            category = "upstream_only"
        elif up_s == "missing":
            category = "amd_only"
        elif amd_s in ("passed", "xpassed") and up_s in ("passed", "xpassed"):
            category = "both_pass"
        elif amd_s in ("failed", "error") and up_s in ("failed", "error"):
            category = "both_fail"
        elif amd_s in ("failed", "error") and up_s in ("passed", "xpassed"):
            category = "amd_regression"
        elif amd_s in ("passed", "xpassed") and up_s in ("failed", "error"):
            category = "amd_advantage"
        elif amd_s in ("skipped", "xfailed") and up_s in ("skipped", "xfailed"):
            category = "both_skip"
        else:
            category = "mixed"

        entries.append(ParityEntry(
            test_id=test_id,
            amd_status=amd_s,
            upstream_status=up_s,
            category=category,
        ))

        summary[category] += 1
        module = _extract_module(test_id)
        module_stats[module][category] += 1

    # Parity % = tests passing on both / tests passing on upstream
    upstream_passing = summary.get("both_pass", 0) + summary.get("amd_regression", 0)
    parity_pct = (
        round(summary.get("both_pass", 0) / upstream_passing * 100, 1)
        if upstream_passing > 0 else 0.0
    )

    # Per-module parity
    by_module = {}
    for module, cats in sorted(module_stats.items()):
        mod_up_passing = cats.get("both_pass", 0) + cats.get("amd_regression", 0)
        mod_parity = (
            round(cats.get("both_pass", 0) / mod_up_passing * 100, 1)
            if mod_up_passing > 0 else 100.0
        )
        by_module[module] = {
            "parity_pct": mod_parity,
            **{k: v for k, v in sorted(cats.items())},
        }

    # Job-group-level parity: compare per-job counts between AMD and upstream
    job_group_parity = _compute_job_group_parity(amd_results, upstream_results)

    return {
        "parity_pct": parity_pct,
        "total_tests": len(all_tests),
        "summary": dict(summary),
        "by_module": by_module,
        "job_groups": job_group_parity,
        "details": [e.to_dict() for e in entries],
    }


def _compute_job_group_parity(
    amd_results: list[TestResult],
    upstream_results: list[TestResult],
) -> list[dict]:
    """Compare test counts per job group between AMD and upstream.

    Groups results by job_name (test group) and compares:
    - Total tests, passed, failed, skipped, xfailed, xpassed, errors
    - Duration (total pytest time)
    """
    def _group_counts(results: list[TestResult]) -> dict[str, dict]:
        groups: dict[str, dict] = {}
        for r in results:
            g = groups.setdefault(r.job_name, {
                "total": 0, "passed": 0, "failed": 0, "skipped": 0,
                "xfailed": 0, "xpassed": 0, "error": 0, "duration": 0.0,
            })
            # For summary entries, extract the count from the name
            if r.name.startswith("__passed__"):
                count = _extract_count(r.name)
                g["passed"] += count
                g["total"] += count
                g["duration"] += r.duration_secs
            elif r.name.startswith("__skipped__"):
                count = _extract_count(r.name)
                g["skipped"] += count
                g["total"] += count
            elif r.name.startswith("__xfailed__"):
                count = _extract_count(r.name)
                g["xfailed"] += count
                g["total"] += count
            elif r.name.startswith("__unidentified_failures__"):
                count = _extract_count(r.name)
                g["failed"] += count
                g["total"] += count
            elif r.name.startswith("__unidentified_errors__"):
                count = _extract_count(r.name)
                g["error"] += count
                g["total"] += count
            elif r.name == "__job_level__":
                # Job-level fallback (no pytest output)
                if r.status == "passed":
                    g["passed"] += 1
                elif r.status == "failed":
                    g["failed"] += 1
                elif r.status == "error":
                    g["error"] += 1
                elif r.status == "canceled":
                    g["canceled"] = g.get("canceled", 0) + 1
                g["total"] += 1
            else:
                # Individual test (failures/errors identified by name)
                g[r.status] = g.get(r.status, 0) + 1
                g["total"] += 1
        return groups

    amd_groups = _group_counts(amd_results)
    upstream_groups = _group_counts(upstream_results)

    # Build per-hardware details. Keep AMD and upstream side data separate:
    # parity matching can pair an AMD hardware-specific variant with an
    # upstream sibling whose normalized name is also used by a different AMD
    # variant. A shared map would leak those sibling failures into the wrong
    # hardware row.
    amd_hw_failures: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    amd_hw_canceled: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    amd_hw_all: dict[str, set] = defaultdict(set)
    upstream_hw_failures: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    upstream_hw_canceled: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    upstream_hw_all: dict[str, set] = defaultdict(set)
    amd_failure_names: dict[str, list[str]] = defaultdict(list)
    upstream_failure_names: dict[str, list[str]] = defaultdict(list)
    job_links: dict[str, list[dict]] = defaultdict(list)  # Buildkite job URLs
    amd_seen_hw: dict[str, set] = defaultdict(set)  # track which hw already has a link
    for r in amd_results:
        hw = _extract_hardware(r.job_name)
        norm = _normalize_job_name(r.job_name).strip()
        amd_hw_all[norm].add(hw)
        if r.status in ("failed", "error"):
            count = _extract_count(r.name) if "__unidentified" in r.name else 1
            amd_hw_failures[norm][hw] += count
            # Track individual failure names (not summary entries)
            if not r.name.startswith("__"):
                amd_failure_names[norm].append(r.name)
        elif r.status == "canceled":
            amd_hw_canceled[norm][hw] += 1
        # Track job links for all AMD jobs (one per hw). AMD matrix/table
        # navigation should land on the exact Buildkite step output, which is
        # keyed by ``step_id``. Fall back to the older job URL only if the
        # step UUID is unavailable in the source record.
        if (r.step_id or r.job_id) and hw not in amd_seen_hw[norm]:
            if r.step_id:
                bk_url = (
                    f"https://buildkite.com/vllm/{r.pipeline}/builds/{r.build_number}"
                    f"/steps/canvas?sid={r.step_id}&tab=output"
                )
            else:
                bk_url = (
                    f"https://buildkite.com/vllm/{r.pipeline}/builds/{r.build_number}"
                    f"/steps/canvas?jid={r.job_id}&tab=output"
                )
            job_links[norm].append({"hw": hw, "url": bk_url, "job_name": r.job_name, "side": "amd"})
            amd_seen_hw[norm].add(hw)

    # Collect upstream hardware + job links
    upstream_job_links: dict[str, dict] = {}
    upstream_seen_hw: dict[str, set] = defaultdict(set)
    for r in upstream_results:
        hw = _extract_hardware(r.job_name)
        norm = _normalize_job_name(r.job_name).strip()
        upstream_hw_all[norm].add(hw)
        if r.status in ("failed", "error"):
            count = _extract_count(r.name) if "__unidentified" in r.name else 1
            upstream_hw_failures[norm][hw] += count
            if not r.name.startswith("__"):
                upstream_failure_names[norm].append(r.name)
        elif r.status == "canceled":
            upstream_hw_canceled[norm][hw] += 1
        if r.job_id and hw not in upstream_seen_hw[norm]:
            bk_url = f"https://buildkite.com/vllm/{r.pipeline}/builds/{r.build_number}/steps/canvas?jid={r.job_id}&tab=output"
            job_links[norm].append({"hw": hw, "url": bk_url, "job_name": r.job_name, "side": "upstream"})
            upstream_seen_hw[norm].add(hw)
            if norm not in upstream_job_links:
                upstream_job_links[norm] = {"hw": hw, "url": bk_url, "job_name": r.job_name, "side": "upstream"}

    # Build normalized -> original maps using full normalize_job_name
    # When multiple jobs normalize to the same name (e.g., MoE Test 1-5 -> MoE Test),
    # merge their counts. Uses _parity_key for cross-pipeline matching
    # (strips multi-HW tags and GPU counts) while keeping norm names for display.
    def _build_norm_map(groups):
        norm_to_orig = {}
        merged = {}
        for k, v in groups.items():
            norm = _normalize_job_name(k)
            if norm in merged:
                for field in v:
                    if isinstance(v[field], (int, float)):
                        merged[norm][field] = merged[norm].get(field, 0) + v[field]
            else:
                merged[norm] = dict(v)
                norm_to_orig[norm] = k
        return norm_to_orig, merged

    amd_norm, amd_merged = _build_norm_map(amd_groups)
    up_norm, up_merged = _build_norm_map(upstream_groups)

    # Match AMD and upstream groups using parity keys.
    # Multiple AMD norms can share one parity key (e.g., different GPU
    # count or HW-combo variants). Each AMD norm gets its own entry,
    # matched to the upstream group with the same parity key.
    def _upstream_preference(norm_name: str) -> tuple:
        """Rank upstream variants sharing one parity key.

        Prefer variants with real failures/errors first so family-level parity
        views reflect the most severe live upstream state. If there are no hard
        failures, prefer variants with substantive executed results over
        xfailed/skipped-only siblings.
        """
        g = up_merged.get(norm_name, {})
        failed = g.get("failed", 0) + g.get("error", 0)
        passed = g.get("passed", 0) + g.get("xpassed", 0)
        skipped = g.get("skipped", 0) + g.get("xfailed", 0)
        total = g.get("total", 0)
        return (
            1 if failed > 0 else 0,
            failed,
            1 if passed > 0 else 0,
            passed,
            -skipped,
            total,
            norm_name,
        )

    up_pk_map: dict[str, list[str]] = defaultdict(list)  # parity_key -> upstream norm names
    for n in up_norm:
        up_pk_map[_parity_key(n)].append(n)
    for pk in up_pk_map:
        up_pk_map[pk].sort(key=_upstream_preference, reverse=True)

    all_norms = []
    amd_remap = {}
    up_remap = {}
    used_up_norms = set()  # track exact upstream norm names already surfaced

    # First: add all AMD norms (each gets its own entry)
    for amd_n in sorted(amd_norm.keys()):
        pk = _parity_key(amd_n)
        all_norms.append(amd_n)
        amd_remap[amd_n] = amd_n
        up_candidates = up_pk_map.get(pk, [])
        if amd_n in up_candidates:
            up_n = amd_n
        else:
            up_n = up_candidates[0] if up_candidates else None
        if up_n:
            up_remap[amd_n] = up_n
            used_up_norms.add(up_n)

    # Then: add every upstream variant that was not already surfaced above.
    # This preserves additional upstream siblings that share a family key with
    # AMD rows instead of silently dropping them behind the first-match winner.
    for up_n in sorted(up_norm.keys()):
        if up_n not in used_up_norms:
            all_norms.append(up_n)
            up_remap[up_n] = up_n

    # Filter out non-GPU tests (CPU, Intel, Arm, Ascend, GH200)
    all_norms = [n for n in all_norms if not _EXCLUDE_PATTERNS.match(n)]

    job_parity = []
    for norm_name in all_norms:
        amd_key = amd_remap.get(norm_name, norm_name)
        up_key = up_remap.get(norm_name, norm_name)
        amd_orig = amd_norm.get(amd_key)
        up_orig = up_norm.get(up_key)
        amd_g = amd_merged.get(amd_key, {})
        up_g = up_merged.get(up_key, {})
        family_key = _parity_key(up_orig or amd_orig or norm_name)
        family_name = _parity_family_name(up_orig or amd_orig or norm_name)

        # Merge display hardware from the matching AMD/upstream sides, but keep
        # failure and cancellation counts side-scoped. This prevents a failing
        # AMD sibling such as "V1 e2e (4 GPUs)" from making the passed
        # "V1 e2e (4xH100-4xMI300)" row look failed merely because both match
        # the same upstream parity key.
        merged_hw = set()
        merged_hwf: dict[str, int] = {}
        merged_hwc: dict[str, int] = {}
        if amd_orig:
            merged_hw |= amd_hw_all.get(amd_key, set())
            merged_hwf.update(amd_hw_failures.get(amd_key, {}))
            merged_hwc.update(amd_hw_canceled.get(amd_key, {}))
        if up_orig:
            merged_hw |= upstream_hw_all.get(up_key, set())
            for hw, c in upstream_hw_failures.get(up_key, {}).items():
                merged_hwf[hw] = merged_hwf.get(hw, 0) + c
            for hw, c in upstream_hw_canceled.get(up_key, {}).items():
                merged_hwc[hw] = merged_hwc.get(hw, 0) + c
        merged_links = job_links.get(amd_key, [])
        if up_key != amd_key:
            merged_links = merged_links + job_links.get(up_key, [])
        # Dedup merged links by (hw, side): the parity table shows one cell per
        # (hw, side), so two variants of the same group on the same hardware
        # (e.g. "mi325_4: V1 e2e (4 GPUs)" + "mi325_4: V1 e2e (4xH100-4xMI325)")
        # should produce a single link, not two.
        _seen_link_cells: set = set()
        _deduped_links: list[dict] = []
        for _l in merged_links:
            _k = (_l.get("hw"), _l.get("side"))
            if _k in _seen_link_cells:
                continue
            _seen_link_cells.add(_k)
            _deduped_links.append(_l)
        merged_links = _deduped_links
        merged_failures = []
        if amd_orig:
            merged_failures += amd_failure_names.get(amd_key, [])
        if up_orig:
            merged_failures += upstream_failure_names.get(up_key, [])

        entry = {
            "name": norm_name,
            "family_key": family_key,
            "family_name": family_name,
            "amd_job_name": amd_orig,
            "upstream_job_name": up_orig,
            "amd": amd_g if amd_g else None,
            "upstream": up_g if up_g else None,
            "hardware": sorted(merged_hw),
            "hw_failures": merged_hwf if merged_hwf else None,
            "hw_canceled": merged_hwc if merged_hwc else None,
            "failure_tests": merged_failures[:20],
            "job_links": merged_links,
        }

        # Compute delta
        if amd_g and up_g:
            entry["delta"] = {
                "total": amd_g.get("total", 0) - up_g.get("total", 0),
                "passed": amd_g.get("passed", 0) - up_g.get("passed", 0),
                "failed": amd_g.get("failed", 0) - up_g.get("failed", 0),
                "skipped": amd_g.get("skipped", 0) - up_g.get("skipped", 0),
            }
            entry["status"] = "amd_only" if not up_orig else (
                "upstream_only" if not amd_orig else "both"
            )
        elif amd_g:
            entry["status"] = "amd_only"
        else:
            entry["status"] = "upstream_only"

        job_parity.append(entry)

    return job_parity


def _extract_count(name: str) -> int:
    """Extract count from names like '__passed__ (136)'."""
    import re
    m = re.search(r"\((\d+)\)", name)
    return int(m.group(1)) if m else 1


# ---------------------------------------------------------------------------
# Trend analysis
# ---------------------------------------------------------------------------

def compute_trends(
    build_summaries: list[BuildSummary],
    health_data: list[TestHealth],
) -> dict:
    """Compute failure trends, top offenders, and module health.

    Args:
        build_summaries: Build summaries sorted oldest-first
        health_data: Current health labels for all tests

    Returns:
        Trends dict with top_offenders, new_failures, recently_fixed,
        degrading_modules, and mttf.
    """
    # Top offenders: tests with most failures
    failing_tests = [
        h for h in health_data
        if h.label in ("failing", "new_failure", "flaky")
    ]
    top_offenders = sorted(
        failing_tests,
        key=lambda h: h.failure_streak + (1 - h.pass_rate) * 100,
        reverse=True,
    )[:20]

    # New failures
    new_failures = [h for h in health_data if h.label == "new_failure"]

    # Recently fixed
    recently_fixed = [h for h in health_data if h.label == "fixed"]

    # Degrading modules: aggregate pass rate per module across builds
    module_health = defaultdict(list)
    for h in health_data:
        module_health[h.module].append(h)

    degrading_modules = []
    for module, tests in sorted(module_health.items()):
        pass_rates = [t.pass_rate for t in tests if t.appearances >= 3]
        if not pass_rates:
            continue
        avg_rate = sum(pass_rates) / len(pass_rates)
        failing_count = sum(1 for t in tests if t.label in ("failing", "new_failure"))
        flaky_count = sum(1 for t in tests if t.label == "flaky")
        if failing_count > 0 or flaky_count > 0:
            degrading_modules.append({
                "module": module,
                "avg_pass_rate": round(avg_rate, 4),
                "total_tests": len(tests),
                "failing": failing_count,
                "flaky": flaky_count,
                "passing": sum(1 for t in tests if t.label == "passing"),
            })

    degrading_modules.sort(key=lambda m: m["avg_pass_rate"])

    # MTTF: for fixed tests, estimate days from first_failure to last_seen
    mttf_values = []
    for h in recently_fixed:
        if h.first_failure and h.last_seen:
            try:
                d1 = datetime.fromisoformat(h.first_failure)
                d2 = datetime.fromisoformat(h.last_seen)
                days = (d2 - d1).days
                if days >= 0:
                    mttf_values.append(days)
            except ValueError:
                pass

    mttf = {
        "avg_days": round(sum(mttf_values) / len(mttf_values), 1) if mttf_values else None,
        "median_days": sorted(mttf_values)[len(mttf_values) // 2] if mttf_values else None,
        "count": len(mttf_values),
    }

    # Build pass rate trend
    pass_rate_trend = []
    for bs in build_summaries:
        pass_rate_trend.append({
            "build_number": bs.build_number,
            "date": bs.created_at[:10] if bs.created_at else "",
            "pass_rate": bs.pass_rate,
            "total": bs.total_tests,
            "failed": bs.failed,
        })

    return {
        "top_offenders": [h.to_dict() for h in top_offenders],
        "new_failures": [h.to_dict() for h in new_failures],
        "recently_fixed": [h.to_dict() for h in recently_fixed],
        "degrading_modules": degrading_modules,
        "mttf": mttf,
        "pass_rate_trend": pass_rate_trend,
    }


# ---------------------------------------------------------------------------
# Quarantine
# ---------------------------------------------------------------------------

def load_quarantine(quarantine_path: str) -> dict:
    """Load quarantine/allowlist config from YAML.

    Format:
        quarantine:
          - test_id: "module::test_name"
            reason: "Known issue"
            issue: "https://github.com/..."
            added: "2026-03-01"
            expires: "2026-04-01"
        allowlist:
          - test_id: "module::test_other"
            reason: "Expected AMD failure"
            permanent: true

    Returns:
        Dict with 'quarantine' and 'allowlist' lists.
    """
    try:
        with open(quarantine_path) as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {"quarantine": [], "allowlist": []}

    return {
        "quarantine": data.get("quarantine", []),
        "allowlist": data.get("allowlist", []),
    }


def apply_quarantine(
    health_data: list[TestHealth],
    quarantine_config: dict,
    today: Optional[str] = None,
) -> tuple[list[TestHealth], dict]:
    """Apply quarantine/allowlist labels to health data.

    Quarantined tests are relabeled so they don't count as failures in metrics.

    Args:
        health_data: List of TestHealth objects
        quarantine_config: Output of load_quarantine()
        today: ISO date string for expiry check (default: now)

    Returns:
        Tuple of (updated health_data, quarantine_report)
    """
    if today is None:
        today = datetime.utcnow().strftime("%Y-%m-%d")

    quarantine_ids = set()
    allowlist_ids = set()
    quarantine_details = []
    allowlist_details = []

    for entry in quarantine_config.get("quarantine", []):
        tid = entry.get("test_id", "")
        expires = entry.get("expires")
        if expires and expires < today:
            continue  # expired
        quarantine_ids.add(tid)
        quarantine_details.append(entry)

    for entry in quarantine_config.get("allowlist", []):
        tid = entry.get("test_id", "")
        allowlist_ids.add(tid)
        allowlist_details.append(entry)

    excluded_from_failures = 0
    for h in health_data:
        if h.test_id in quarantine_ids:
            h.label = "quarantined"
            excluded_from_failures += 1
        elif h.test_id in allowlist_ids:
            h.label = "allowlisted"
            excluded_from_failures += 1

    report = {
        "quarantined_count": len([h for h in health_data if h.label == "quarantined"]),
        "allowlisted_count": len([h for h in health_data if h.label == "allowlisted"]),
        "excluded_from_failures": excluded_from_failures,
        "quarantine_entries": quarantine_details,
        "allowlist_entries": allowlist_details,
    }

    return health_data, report


# ---------------------------------------------------------------------------
# Build summary computation
# ---------------------------------------------------------------------------

_HW_FAMILY_RE = re.compile(r'^(mi\d+)_\d+:', re.IGNORECASE)
# Upstream GPU tags in parens: (H100), (B200), (2xH100), (4xA100), (H100-MI250), etc.
_UPSTREAM_HW_RE = re.compile(
    r'\((\d*x?)(H\d+|B\d+|A\d+|L\d+|GH?\d+)(?:\s+\w+)*(?:\s*-\s*[\w\s]+)?\)\s*$',
    re.IGNORECASE,
)


_DEFAULT_HARDWARE = "h100"


def set_default_hardware(hw: str):
    """Set the hardware reported for job names that carry no hardware tag.

    Defaults to ``h100`` because an untagged name in the upstream vLLM
    pipeline means the default NVIDIA queue. A single-hardware pipeline whose
    labels carry no tag (the AMD distributed-inference pipeline, entirely on
    ``amd_mi350_ainic``) sets this at startup so its jobs are not all reported
    as H100. Process-scoped, like ``set_shard_bases``.
    """
    global _DEFAULT_HARDWARE
    _DEFAULT_HARDWARE = (hw or "h100").lower()


def _extract_hardware(job_name: str) -> str:
    """Extract hardware family from job name.

    AMD style: 'mi250_1: Test Name' -> 'mi250'
    Upstream style: 'Test Name (H100)' -> 'h100'
                    'Test Name (2xB200)' -> 'b200'
                    'Test Name (H100-MI250)' -> 'h100'
    No tag (upstream default): -> 'h100' (default NVIDIA queue)
    """
    # AMD prefix
    m = _HW_FAMILY_RE.match(job_name)
    if m:
        return m.group(1).lower()
    # Upstream GPU tag in parens
    m = _UPSTREAM_HW_RE.search(job_name)
    if m:
        return m.group(2).lower()
    # Skip non-GPU platform jobs.
    # Be careful: "(CPU)" at the end of a test name (e.g., "V1 others (CPU)")
    # means CPU codepath testing on GPU hardware, NOT a CPU-only job.
    # Only classify as "cpu" for actual CPU-platform jobs:
    #   - Names starting with "CPU" or "Arm" (e.g., "CPU-Distributed Tests", "Arm CPU Test")
    #   - Specific platform tests: HPU, NPU, Intel GPU, Ascend
    lower = job_name.lower()
    if (lower.startswith("cpu") or lower.startswith("arm ")
            or re.search(r'\bhpu\b|\bascend\b', lower)
            or lower.startswith("intel")):
        return "cpu"
    if lower.startswith("amd:"):
        return "unknown"
    # Default upstream GPU queue is H100
    return _DEFAULT_HARDWARE


def _actual_count(r: TestResult) -> int:
    """Get the actual test count from a TestResult entry.

    Summary entries like '__passed__ (136)' wrap 136 actual tests.
    Individual named tests count as 1.
    """
    if r.name.startswith("__") and "(" in r.name:
        return _extract_count(r.name)
    return 1


def compute_build_summary(
    build: dict,
    test_results: list[TestResult],
    pipeline_key: str,
    previous: Optional[BuildSummary] = None,
    skip_job_patterns: tuple[str, ...] = (),
) -> BuildSummary:
    """Compute a BuildSummary from a build dict and its test results.

    Uses actual test counts extracted from summary entries (e.g.,
    '__passed__ (136)' counts as 136, not 1).
    """
    # Count actual tests, not entries
    passed = 0
    failed = 0
    skipped = 0
    errors = 0
    canceled = 0
    test_groups = len(test_results)  # entry count (old total_tests)

    # Build set of soft-failed job names — failures in these are expected
    # and should not count toward groups_failed
    soft_failed_jobs = set()
    for j in build.get("jobs", []):
        if j.get("soft_failed"):
            soft_failed_jobs.add(j.get("name", ""))

    # Per-hardware breakdown
    hw_counts: dict[str, dict] = {}
    hw_seen_groups: dict[str, set] = defaultdict(set)
    hw_failed_groups: dict[str, set] = defaultdict(set)

    for r in test_results:
        count = _actual_count(r)
        hw = _extract_hardware(r.job_name)

        if hw not in hw_counts:
            hw_counts[hw] = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0, "total": 0, "groups": 0, "groups_failed": 0}

        if r.status in ("passed", "xpassed"):
            passed += count
            hw_counts[hw]["passed"] += count
        elif r.status == "failed":
            failed += count
            hw_counts[hw]["failed"] += count
        elif r.status == "error":
            errors += count
            failed += count
            hw_counts[hw]["errors"] += count
            hw_counts[hw]["failed"] += count
        elif r.status == "canceled":
            canceled += count
            hw_counts[hw].setdefault("canceled", 0)
            hw_counts[hw]["canceled"] += count
        elif r.status in ("skipped", "xfailed"):
            skipped += count
            hw_counts[hw]["skipped"] += count

        hw_counts[hw]["total"] += count

        # Track groups per HW — any failure in any shard marks the group as failed
        # but exclude soft-failed jobs (failures are expected/accepted)
        norm = _normalize_job_name(r.job_name).strip()
        hw_seen_groups[hw].add(norm)
        if r.status in ("failed", "error") and r.job_name not in soft_failed_jobs:
            hw_failed_groups[hw].add(norm)

    # Add group counts to hw_counts
    for hw in hw_counts:
        hw_counts[hw]["groups"] = len(hw_seen_groups.get(hw, set()))
        hw_counts[hw]["groups_failed"] = len(hw_failed_groups.get(hw, set()))

    total = passed + failed + skipped
    ran = passed + failed
    pass_rate = round(passed / ran, 4) if ran > 0 else 0.0

    # Per-hardware pass rates
    for hw, counts in hw_counts.items():
        hw_ran = counts["passed"] + counts["failed"]
        counts["pass_rate"] = round(counts["passed"] / hw_ran, 4) if hw_ran > 0 else 0.0

    # OR-logic test group pass rate
    # Group by normalized name (strip HW prefix), track per-HW pass/fail
    # AND-logic across shards: if ANY shard/result in a group fails, the group fails
    group_hw_status: dict[str, dict[str, bool]] = defaultdict(dict)
    for r in test_results:
        norm = _normalize_job_name(r.job_name).strip()
        hw = _extract_hardware(r.job_name)
        if r.status in ("failed", "error"):
            # Any failure -> mark group as failed on this HW (AND across shards)
            group_hw_status[norm][hw] = False
        elif r.status == "canceled":
            # Canceled jobs should not be counted as passing
            group_hw_status[norm].setdefault(hw, None)
        elif r.status in ("skipped", "xfailed"):
            # Skip-only groups are still observed test groups. Keep them in
            # the denominator without treating them as passing or failing.
            group_hw_status[norm].setdefault(hw, None)
        elif r.status in ("passed", "xpassed"):
            # Only mark as passing if not already failed or canceled
            if group_hw_status[norm].get(hw) is not False:
                group_hw_status[norm][hw] = True

    unique_test_groups = len(group_hw_status)
    groups_passing_or = 0   # passes on ANY hardware
    groups_passing_all = 0  # passes on ALL hardware
    groups_partial = 0      # passes on some, fails on others
    for name, hw_map in group_hw_status.items():
        any_pass = any(hw_map.values())
        all_pass = all(hw_map.values())
        if any_pass:
            groups_passing_or += 1
        if all_pass:
            groups_passing_all += 1
        if any_pass and not all_pass:
            groups_partial += 1

    duration = sum(r.duration_secs for r in test_results)

    # Wall clock from build timestamps
    # For running builds, use now - created_at as elapsed time
    created = build.get("created_at", "")
    finished = build.get("finished_at", "")
    wall_clock = 0.0
    if created:
        try:
            t1 = datetime.fromisoformat(created.replace("Z", "+00:00"))
            if finished:
                t2 = datetime.fromisoformat(finished.replace("Z", "+00:00"))
            else:
                t2 = datetime.now(t1.tzinfo or None)
            wall_clock = (t2 - t1).total_seconds()
        except ValueError:
            pass

    # Job-level stats (count ALL script jobs, including running/waiting)
    jobs = build.get("jobs", [])
    script_jobs = [j for j in jobs if j.get("type") == "script"]

    # Filter out retried jobs (superseded by a retry) so we only count
    # the latest attempt per step.  Buildkite sets ``retried_in_job_id``
    # on the OLD job pointing to its replacement — so any job with this
    # field set is a superseded attempt we should skip.
    latest_jobs = [j for j in script_jobs
                   if not j.get("retried_in_job_id")]

    # For counting *unique* steps (as shown in the Buildkite UI), collapse
    # parallel shards into a single logical step.  We use the step's
    # ``step_key`` (or ``name`` as fallback) so that e.g. 4 shards of
    # "Kernels MoE Test" count as one step.
    def _step_key(j: dict) -> str:
        return j.get("step_key") or j.get("name") or j.get("id", "")

    # Deduplicated step-level counts: group latest_jobs by step, pick the
    # "worst" state per step (failed > passed > running > waiting).
    _step_groups: dict[str, list[dict]] = defaultdict(list)
    for j in latest_jobs:
        _step_groups[_step_key(j)].append(j)

    test_step_groups = {
        key: group
        for key, group in _step_groups.items()
        if not any(
            pattern in str((group[0] if group else {}).get("name") or "").lower()
            for pattern in skip_job_patterns
        )
    }

    def _step_state(group: list[dict]) -> tuple[str, bool]:
        """Return (effective_state, soft_failed) for a group of shard jobs."""
        states = [j.get("state") for j in group]
        soft = any(j.get("soft_failed") for j in group)
        for s in cfg.FAILURE_STATES:
            if s in states:
                return s, soft
        if "passed" in states:
            return "passed", False
        for s in cfg.RUNNING_STATES:
            if s in states:
                return s, False
        for s in cfg.WAITING_STATES:
            if s in states:
                return s, False
        return states[0] if states else "unknown", False

    jobs_passed = 0
    jobs_failed = 0
    jobs_soft_failed = 0
    jobs_running = 0
    jobs_waiting = 0
    for _grp in _step_groups.values():
        st, sf = _step_state(_grp)
        if st == "passed":
            jobs_passed += 1
        elif st in cfg.FAILURE_STATES:
            jobs_failed += 1
            if sf:
                jobs_soft_failed += 1
        elif st in cfg.RUNNING_STATES:
            jobs_running += 1
        elif st in cfg.WAITING_STATES:
            jobs_waiting += 1

    test_jobs_blocked = 0
    for group in test_step_groups.values():
        states = {str(job.get("state") or "").lower() for job in group}
        if states & cfg.BLOCKED_JOB_STATES:
            test_jobs_blocked += 1
    # Only mark as running if jobs are actually in-flight — Buildkite's
    # build.state can lag behind and report "running" long after completion.
    is_running = jobs_running > 0 or jobs_waiting > 0

    # Guard against stale "running" banners: if the build is terminal or was
    # created long enough ago that no human would still consider it in-flight,
    # force is_running False even when the cached API snapshot still shows a
    # lone job stuck in the "running" state.
    build_state = build.get("state", "")
    if build_state in cfg.TERMINAL_STATES:
        is_running = False
    else:
        try:
            _created_raw = build.get("created_at", "")
            if _created_raw:
                _created_dt = datetime.fromisoformat(_created_raw.replace("Z", "+00:00"))
                _age_hrs = (datetime.now(_created_dt.tzinfo) - _created_dt).total_seconds() / 3600
                if _age_hrs >= 18:
                    is_running = False
        except (ValueError, TypeError):
            pass

    # Delta vs previous
    delta = {}
    if previous:
        delta = {
            "total": total - previous.total_tests,
            "passed": passed - previous.passed,
            "failed": failed - previous.failed,
            "pass_rate": round(pass_rate - previous.pass_rate, 4),
        }

    return BuildSummary(
        pipeline=pipeline_key,
        build_number=build.get("number", 0),
        build_url=build.get("web_url", ""),
        branch=build.get("branch", ""),
        commit=build.get("commit", "")[:12],
        created_at=created,
        state=build.get("state", ""),
        total_tests=total,
        passed=passed,
        failed=failed,
        skipped=skipped,
        errors=errors,
        pass_rate=pass_rate,
        duration_secs=round(duration, 1),
        wall_clock_secs=round(wall_clock, 1),
        job_count=len(_step_groups),
        jobs_passed=jobs_passed,
        jobs_failed=jobs_failed,
        jobs_soft_failed=jobs_soft_failed,
        jobs_running=jobs_running,
        jobs_waiting=jobs_waiting,
        test_job_count=len(test_step_groups),
        test_jobs_blocked=test_jobs_blocked,
        has_test_results=bool(test_results),
        is_running=is_running,
        test_groups=test_groups,
        unique_test_groups=unique_test_groups,
        test_groups_passing_or=groups_passing_or,
        test_groups_passing_all=groups_passing_all,
        test_groups_partial=groups_partial,
        by_hardware=hw_counts,
        delta_vs_previous=delta,
    )
