#!/usr/bin/env python3
"""Ingest real perf-eval nightly results from Buildkite artifacts.

The Perf Eval tab is fed by an append-only event log
(``data/vllm/perf_eval/events.jsonl``) that ``collect_perf_eval.py`` folds into
``perf_eval.json``. This collector is what actually *fills* that log from live
data — no webhook receiver required.

The ``vllm/perf-eval`` pipeline uploads its entire ``results/`` tree as
Buildkite artifacts on every build (``artifact_paths: ["results/**/*"]``). Using
only the existing ``BUILDKITE_TOKEN`` (needs Read Builds + Read Artifacts) and
``GITHUB_TOKEN`` (to read the public workload recipes), we:

1. list finished ``perf-eval`` builds on ``main`` in a lookback window;
2. keep the nightly ones (identified by the build message
   ``Nightly run <date>: commit <sha>`` — which also gives us the vLLM commit —
   with ``NIGHTLY=1`` / scheduled-source as fallbacks);
3. download each AMD workload's raw ``bench-*.json`` (perf) and
   ``results_*.json`` (accuracy) artifacts;
4. transform them into the same canonical events ``perf_eval_webhook`` produces
   (reusing its normalizers, AMD filter and metric registry); and
5. append only new events (deduped by build+workload+config/task).

NVIDIA workloads are dropped: only AMD (MIxxx) workloads are kept, matching the
executive AMD-only view. Per-GPU throughput is derived exactly the way the
pipeline's ``ingest_perf.py`` does, using ``tp``/device/precision read from the
workload recipe.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vllm.constants import BK_API_BASE, BK_ORG  # noqa: E402
from vllm.ci.perf_eval_webhook import (  # noqa: E402
    METRIC_META,
    append_event,
    commit_from_image,
    is_amd_workload,
    normalize_eval_payload,
    read_events,
    utcnow_iso,
)
from vllm.ci import ratelimit  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_STORE = ROOT / "data" / "vllm" / "perf_eval" / "events.jsonl"

PIPELINE_SLUG = "perf-eval"
# Public repo that holds the workload recipes (device/tp/precision/bench sizes).
WORKLOAD_REPO = "vllm-project/perf-eval"
AMD_IMAGE_REPO = "vllm/vllm-openai-rocm"

# "Nightly run 2026-06-30: commit 93d8f834dd8acf33eb0e2a75b2711b628cb6e226".
# The date + commit make this the least brittle nightly signal and give us the
# exact vLLM commit for provenance for free.
_NIGHTLY_MSG_RE = re.compile(
    r"nightly\s+run\s+(\d{4}-\d{2}-\d{2}).*?commit\s+([0-9a-f]{7,40})",
    re.IGNORECASE | re.DOTALL,
)
_NIGHTLY_WORD_RE = re.compile(r"\bnightly\b", re.IGNORECASE)

# Artifact paths look like ``results/<workload>/bench-<config>.json`` (perf) and
# ``results/<workload>/<task>/results_*.json`` (accuracy).
_PERF_ARTIFACT_RE = re.compile(r"^results/(?P<wl>[^/]+)/bench-(?P<cfg>.+)\.json$")
_ACC_ARTIFACT_RE = re.compile(r"^results/(?P<wl>[^/]+)/(?P<task>[^/]+)/results_.*\.json$")


# ---------------------------------------------------------------------------
# Pure helpers (no I/O) — unit tested without a live Buildkite/GitHub
# ---------------------------------------------------------------------------

def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def parse_tp(serve_args: str) -> int:
    """Effective parallel degree (TP * DP) from serve_args; defaults to 1.

    Mirrors perf-eval's ``parse_workload.parse_tp`` so per-GPU throughput is
    computed identically to what the pipeline posts to its own dashboard.
    """
    toks = (serve_args or "").split()

    def find(*names: str) -> Optional[int]:
        for i, tok in enumerate(toks):
            if "=" in tok:
                key, _, val = tok.partition("=")
                if key in names:
                    try:
                        return int(val)
                    except ValueError:
                        return None
            elif tok in names and i + 1 < len(toks):
                try:
                    return int(toks[i + 1])
                except ValueError:
                    return None
        return None

    tp = find("--tensor-parallel-size", "-tp", "--tp") or 1
    dp = find("--data-parallel-size", "-dp", "--dp") or 1
    return tp * dp


def precision_from_model(model: str) -> str:
    """Infer a precision tag from the model id (mirrors parse_workload)."""
    name = (model or "").lower()
    for marker in ("fp4", "fp8", "int4", "int8", "bf16", "fp16"):
        if marker in name:
            return marker
    return "bf16"


def workload_entry(data: dict) -> dict:
    """Project a workload recipe into the fields the transform needs."""
    gpu = (data.get("gpu") or "").strip()
    vllm = data.get("vllm") or {}
    bench = data.get("vllm_bench") or {}
    meta = bench.get("metadata") or {}
    model = (vllm.get("model") or "").strip()
    serve_args = vllm.get("serve_args") or ""
    configs = {}
    for cfg in bench.get("configs") or []:
        name = cfg.get("name")
        if not name:
            continue
        configs[str(name)] = {
            "isl": cfg.get("input_len"),
            "osl": cfg.get("output_len"),
            "conc": cfg.get("max_concurrency"),
        }
    return {
        "name": (data.get("name") or "").strip(),
        "gpu": gpu,
        "device": (meta.get("device") or gpu.lower()).strip(),
        "tp": meta.get("tp") if meta.get("tp") is not None else parse_tp(serve_args),
        "precision": (meta.get("precision") or precision_from_model(model)).strip(),
        "model": model,
    }, configs


def nightly_info(build: dict) -> Optional[dict]:
    """Return ``{date, vllm_commit, branch}`` if a build is a nightly, else None.

    Primary signal is the structured build message; ``NIGHTLY=1`` in the build
    env and a scheduled build whose message merely mentions "nightly" are
    accepted as fallbacks (both additionally require the ``main`` branch to
    avoid mislabeling ad-hoc runs).
    """
    branch = (build.get("branch") or "").strip()
    message = build.get("message") or ""
    env = build.get("env") or {}

    date = commit = None
    is_night = False

    match = _NIGHTLY_MSG_RE.search(message)
    if match:
        is_night = True
        date, commit = match.group(1), match.group(2)
    elif branch in {"main", "master"} and _truthy(env.get("NIGHTLY")):
        is_night = True
    elif (
        branch in {"main", "master"}
        and (build.get("source") or "") == "schedule"
        and _NIGHTLY_WORD_RE.search(message)
    ):
        is_night = True

    if not is_night:
        return None

    if not commit:
        commit = (env.get("VLLM_COMMIT") or "").strip() or commit_from_image(
            env.get("VLLM_IMAGE") or ""
        )
    if not date:
        stamp = build.get("created_at") or build.get("finished_at") or ""
        date = stamp[:10] if stamp else ""
    return {"date": date, "vllm_commit": commit, "branch": branch or "main"}


def amd_image(env: dict, vllm_commit: str) -> str:
    """Best-effort ROCm image URI for provenance / AMD detection."""
    img = (env.get("VLLM_IMAGE") or "").strip()
    if img and "rocm" in img.lower():
        return img
    if vllm_commit:
        return f"{AMD_IMAGE_REPO}:nightly-{vllm_commit}"
    return f"{AMD_IMAGE_REPO}:nightly"


def classify_artifact(path: str) -> Optional[tuple[str, str, str]]:
    """Classify an artifact path.

    Returns ``("perf", workload, config)`` for a bench result,
    ``("accuracy", workload, task)`` for an lm-eval/bfcl result, or None for
    anything else (e.g. large ``samples_*.jsonl`` we never download).
    """
    norm = (path or "").strip().lstrip("./")
    m = _PERF_ARTIFACT_RE.match(norm)
    if m:
        return "perf", m.group("wl"), m.group("cfg")
    m = _ACC_ARTIFACT_RE.match(norm)
    if m:
        return "accuracy", m.group("wl"), m.group("task")
    return None


def _to_float(value: Any) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


def transform_perf(raw: dict, *, tp: int) -> dict[str, float]:
    """Turn a raw ``vllm bench serve`` result into canonical per-GPU metrics.

    Mirrors perf-eval's ``ingest_perf.transform``: aggregate throughput is
    divided by ``tp`` for the per-GPU columns, ``*_ms`` latencies are converted
    to seconds, and interactivity (tok/s) is derived from TPOT. Only metrics in
    the shared registry survive.
    """
    tp = max(int(tp or 1), 1)
    total = _to_float(raw.get("total_token_throughput")) or 0.0
    output = _to_float(raw.get("output_throughput")) or 0.0
    metrics: dict[str, float] = {
        "tput_per_gpu": total / tp,
        "output_tput_per_gpu": output / tp,
        "input_tput_per_gpu": (total - output) / tp,
    }
    for key, value in raw.items():
        if not isinstance(key, str) or not key.endswith("_ms"):
            continue
        v = _to_float(value)
        if v is None:
            continue
        base = key[: -len("_ms")]
        metrics[base] = v / 1000.0
        if "tpot" in base:
            metrics[base.replace("tpot", "intvty")] = 1000.0 / v if v else 0.0
    return {k: round(v, 4) for k, v in metrics.items() if k in METRIC_META}


def perf_event(
    raw: dict,
    *,
    entry: dict,
    config: dict,
    identity: dict,
) -> Optional[dict]:
    """Build a canonical ``perf_result`` event from a raw bench artifact."""
    device = entry.get("device") or ""
    image = identity.get("image") or ""
    if not is_amd_workload(image=image, device=device, workload=entry.get("name")):
        return None
    metrics = transform_perf(raw, tp=entry.get("tp") or 1)
    if not metrics:
        return None
    conc = config.get("conc")
    if conc is None:
        conc = raw.get("max_concurrency")
    return {
        "event": "perf_result",
        "received_at": utcnow_iso(),
        "nightly": True,
        "model": (raw.get("model_id") or entry.get("model") or "").strip(),
        "device": device,
        "precision": entry.get("precision") or "",
        "tp": entry.get("tp"),
        "isl": config.get("isl"),
        "osl": config.get("osl"),
        "conc": conc,
        "date": identity.get("date") or "",
        "build_number": identity.get("build_number"),
        "build_url": identity.get("build_url") or "",
        "build_commit": identity.get("build_commit") or "",
        "branch": identity.get("branch") or "main",
        "image": image,
        "vllm_commit": identity.get("vllm_commit") or "",
        "metrics": metrics,
    }


def accuracy_event(results_json: dict, *, workload: str, task: str, entry: dict, identity: dict) -> Optional[dict]:
    """Build a canonical ``accuracy_result`` event from an lm-eval artifact."""
    payload = {
        "kind": "results",
        "workload": workload,
        "task": task,
        "device": entry.get("device") or "",
        "image": identity.get("image") or "",
        "vllm_commit": identity.get("vllm_commit") or "",
        "buildkite_build_number": identity.get("build_number"),
        "buildkite_build_url": identity.get("build_url") or "",
        "buildkite_commit": identity.get("build_commit") or "",
        "buildkite_branch": identity.get("branch") or "main",
        "nightly": True,
        "data": results_json,
    }
    event = normalize_eval_payload(payload)
    if event is None:
        return None
    event["nightly"] = True
    if identity.get("date"):
        event["date"] = identity["date"]
    if not (event.get("model") or "").strip():
        event["model"] = entry.get("model") or workload
    return event


def event_key(event: dict) -> tuple:
    """Stable dedup identity so re-runs never double-append the same result."""
    if event.get("event") == "perf_result":
        return (
            "perf",
            event.get("build_number"),
            (event.get("model") or "").strip(),
            event.get("device"),
            event.get("isl"),
            event.get("osl"),
            event.get("conc"),
        )
    tasks = tuple(sorted((r.get("task"), r.get("metric")) for r in event.get("results") or []))
    return (
        "accuracy",
        event.get("build_number"),
        (event.get("workload") or "").strip(),
        tasks,
    )


# ---------------------------------------------------------------------------
# I/O — Buildkite REST + GitHub raw
# ---------------------------------------------------------------------------

def _bk_get(path: str, token: str, params: Optional[dict] = None):
    ratelimit.acquire()
    resp = requests.get(
        f"{BK_API_BASE}{path}",
        headers={"Authorization": f"Bearer {token}"},
        params=params,
        timeout=30,
    )
    ratelimit.observe(resp.headers)
    if resp.status_code == 429:
        log.warning("Buildkite rate limited on %s", path)
        return []
    resp.raise_for_status()
    return resp.json()


def _bk_paginate(path: str, token: str, params: Optional[dict] = None, max_pages: int = 10):
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


def _bk_download_json(download_url: str, token: str) -> Optional[dict]:
    """Download a JSON artifact. Buildkite redirects to a presigned URL;
    requests drops the auth header on the cross-host hop automatically."""
    try:
        ratelimit.acquire()
        resp = requests.get(
            download_url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=60,
            allow_redirects=True,
        )
        resp.raise_for_status()
        return resp.json()
    except (requests.RequestException, ValueError) as exc:
        log.warning("Failed to download artifact %s: %s", download_url, exc)
        return None


def fetch_workload_map(gh_token: str) -> dict[str, tuple[dict, dict]]:
    """Fetch all workload recipes and index them by their ``name`` field.

    The artifact path uses the recipe's ``name`` (e.g. ``minimax_m2_5-mi355x``),
    which differs from the filename, so we parse every recipe and key on name.
    """
    import yaml

    headers = {"Accept": "application/vnd.github+json"}
    if gh_token:
        headers["Authorization"] = f"Bearer {gh_token}"
    listing = requests.get(
        f"https://api.github.com/repos/{WORKLOAD_REPO}/contents/workloads",
        headers=headers,
        timeout=30,
    )
    listing.raise_for_status()
    out: dict[str, tuple[dict, dict]] = {}
    for item in listing.json():
        name = item.get("name") or ""
        if not name.endswith((".yaml", ".yml")):
            continue
        raw = requests.get(item["download_url"], headers=headers, timeout=30)
        raw.raise_for_status()
        try:
            data = yaml.safe_load(raw.text)
        except yaml.YAMLError as exc:
            log.warning("Skipping unparseable workload %s: %s", name, exc)
            continue
        if not isinstance(data, dict) or not data.get("name"):
            continue
        entry, configs = workload_entry(data)
        out[entry["name"]] = (entry, configs)
    log.info("Loaded %d workload recipes", len(out))
    return out


def collect(store_path: Path, *, days: int, bk_token: str, gh_token: str) -> int:
    """Pull nightly perf-eval artifacts and append new canonical events."""
    existing = read_events(store_path)
    seen = {event_key(e) for e in existing}
    before = len(existing)

    workloads = fetch_workload_map(gh_token)

    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    builds = _bk_paginate(
        f"/organizations/{BK_ORG}/pipelines/{PIPELINE_SLUG}/builds",
        bk_token,
        {"branch": "main", "state": "finished", "created_from": cutoff},
    )
    log.info("Examining %d finished perf-eval builds since %s", len(builds), cutoff)

    appended = 0
    for build in builds:
        night = nightly_info(build)
        if night is None:
            continue
        number = build.get("number")
        identity = {
            "build_number": number,
            "build_url": build.get("web_url") or "",
            "build_commit": (build.get("commit") or ""),  # perf-eval repo commit
            "branch": night["branch"],
            "vllm_commit": night["vllm_commit"],
            "date": build.get("finished_at") or build.get("created_at") or night["date"],
            "image": amd_image(build.get("env") or {}, night["vllm_commit"]),
        }
        artifacts = _bk_paginate(
            f"/organizations/{BK_ORG}/pipelines/{PIPELINE_SLUG}/builds/{number}/artifacts",
            bk_token,
        )
        for art in artifacts:
            kind = classify_artifact(art.get("path") or "")
            if kind is None:
                continue
            _, workload, tail = kind
            if not is_amd_workload(workload=workload):
                continue
            recipe = workloads.get(workload)
            if recipe is None:
                log.warning("No recipe for workload %s (build #%s); skipping", workload, number)
                continue
            entry, configs = recipe
            payload = _bk_download_json(art.get("download_url") or "", bk_token)
            if payload is None:
                continue
            if kind[0] == "perf":
                event = perf_event(
                    payload, entry=entry, config=configs.get(tail, {}), identity=identity
                )
            else:
                event = accuracy_event(
                    payload, workload=workload, task=tail, entry=entry, identity=identity
                )
            if event is None:
                continue
            key = event_key(event)
            if key in seen:
                continue
            append_event(store_path, event)
            seen.add(key)
            appended += 1

    log.info("Appended %d new events (%d -> %d total)", appended, before, before + appended)
    return appended


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=10, help="Lookback window in days (default: 10)")
    parser.add_argument("--store", default=str(DEFAULT_STORE), help="Path to events.jsonl")
    args = parser.parse_args()

    bk_token = os.getenv("BUILDKITE_TOKEN") or ""
    if not bk_token:
        log.error("BUILDKITE_TOKEN not set; cannot pull perf-eval artifacts")
        return 1
    gh_token = os.getenv("GITHUB_TOKEN") or ""

    collect(Path(args.store), days=args.days, bk_token=bk_token, gh_token=gh_token)
    return 0


if __name__ == "__main__":
    sys.exit(main())
