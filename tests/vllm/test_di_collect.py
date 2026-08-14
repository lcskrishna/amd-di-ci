"""DI collector unit tests — no network, fixture dicts only."""

import json

from vllm.di_collect import (
    classify_state,
    di_jobs,
    extract_verdict,
    job_record,
    load_job_records,
    queue_from_job,
    upsert_job_records,
    verdict_for,
)

BUILD = {
    "number": 412,
    "web_url": "https://buildkite.com/vllm/amd-distributed-inference-ci/builds/412",
    "state": "failed",
    "branch": "main",
    "commit": "0123456789abcdef0123",
    "created_at": "2026-08-12T22:04:11.000Z",
}


def _job(**overrides):
    job = {
        "type": "script",
        "id": "job-1",
        "name": "DeepSeek-V3-PD-1P1D-TP8-MoRIIO-proxy",
        "state": "passed",
        "exit_status": 0,
        "web_url": "https://buildkite.com/.../job-1",
        "agent": {"name": "mi350-agent-3"},
        "agent_query_rules": ["queue=amd_mi350_ainic"],
        "runnable_at": "2026-08-12T22:05:00.000Z",
        "started_at": "2026-08-12T22:35:00.000Z",
        "finished_at": "2026-08-12T23:35:00.000Z",
    }
    job.update(overrides)
    return job


# ---------------------------------------------------------------------------
# Verdict mapping
# ---------------------------------------------------------------------------

def test_timed_out_with_zero_exit_is_actually_a_pass():
    # A Buildkite quirk the nightly collector already accounts for. A 120-min
    # DI step is exactly the kind of job that trips it.
    assert verdict_for(_job(state="timed_out", exit_status=0)) == "passed"


def test_timed_out_with_nonzero_exit_stays_a_timeout():
    assert verdict_for(_job(state="timed_out", exit_status=1)) == "timed_out"


def test_soft_failure_is_distinguished_from_failure():
    assert verdict_for(_job(state="failed", exit_status=1)) == "failed"
    assert verdict_for(_job(state="failed", exit_status=1, soft_failed=True)) == "soft_failed"


def test_queued_and_blocked_states():
    assert verdict_for(_job(state="scheduled")) == "waiting"
    assert verdict_for(_job(state="running")) == "running"
    assert verdict_for(_job(state="waiting_failed")) == "blocked"


# ---------------------------------------------------------------------------
# Job filtering
# ---------------------------------------------------------------------------

def test_superseded_retry_is_dropped_so_a_step_appears_once():
    build = dict(BUILD, jobs=[
        _job(id="old", retried_in_job_id="new", state="failed"),
        _job(id="new", state="passed"),
    ])
    assert [j["id"] for j in di_jobs(build)] == ["new"]


def test_non_script_and_infrastructure_steps_are_dropped():
    build = dict(BUILD, jobs=[
        {"type": "waiter", "id": "w", "name": "wait"},
        _job(id="boot", name="bootstrap pipeline"),
        _job(id="real"),
    ])
    assert [j["id"] for j in di_jobs(build)] == ["real"]


def test_the_pipeline_upload_step_is_dropped():
    # Regression: the live label is ":pipeline: Upload AMD distributed
    # inference". The emoji sits between the words, so the "pipeline upload"
    # pattern misses it and the step was landing in the unclassified bucket
    # and inflating every build to 21 steps.
    build = dict(BUILD, jobs=[
        _job(id="up", name=":pipeline: Upload AMD distributed inference"),
        _job(id="real"),
    ])
    assert [j["id"] for j in di_jobs(build)] == ["real"]


def test_queued_steps_are_kept():
    # A step stuck behind concurrency: 2 is the whole point of the queue-wait
    # panel; filtering to terminal jobs here would hide it.
    build = dict(BUILD, jobs=[_job(id="q", state="scheduled")])
    assert len(di_jobs(build)) == 1


# ---------------------------------------------------------------------------
# Record shape
# ---------------------------------------------------------------------------

def test_job_record_carries_grid_coordinates_and_timings():
    r = job_record(BUILD, _job())
    assert r["model"] == "DeepSeek-V3"
    assert r["shape"] == "1P1D"
    assert r["router"] == "proxy"
    assert r["mode"] == "TP8"
    assert r["label_ok"] is True
    assert r["queue_wait_s"] == 1800.0
    assert r["runtime_s"] == 3600.0
    assert r["queue"] == "amd_mi350_ainic"
    assert r["agent_name"] == "mi350-agent-3"
    assert r["commit"] == "0123456789ab"
    assert r["build_number"] == 412


def test_missing_timestamps_yield_none_not_zero():
    r = job_record(BUILD, _job(started_at=None, finished_at=None))
    assert r["queue_wait_s"] is None
    assert r["runtime_s"] is None


def test_unparseable_label_is_recorded_not_dropped():
    r = job_record(BUILD, _job(name="Some Renamed Step"))
    assert r["label_ok"] is False
    assert r["label"] == "Some Renamed Step"


def test_queue_from_job_without_rules():
    assert queue_from_job(_job(agent_query_rules=[])) == ""


# ---------------------------------------------------------------------------
# The SLURM driver's verdict line
# ---------------------------------------------------------------------------

VERDICT_LOG = """\
some earlier output
[slurm-submit] job 4821 finished: state=workload-failed phase=workload exit=1 \
reason=scontrol JobState=FAILED phase=workload
"""


def test_extract_verdict_parses_the_driver_line():
    v = extract_verdict(VERDICT_LOG)
    assert v["slurm_job_id"] == "4821"
    assert v["slurm_state"] == "workload-failed"
    assert v["phase"] == "workload"
    assert v["reason"].startswith("scontrol JobState=FAILED")
    assert v["failure_class"] == "workload"


def test_extract_verdict_takes_the_last_line_when_a_retry_reran():
    log = VERDICT_LOG + (
        "[slurm-submit] job 4830 finished: state=COMPLETED phase=workload exit=0 reason=gate\n"
    )
    v = extract_verdict(log)
    assert v["slurm_job_id"] == "4830"
    assert v["failure_class"] == "ok"


def test_extract_verdict_tolerates_an_empty_reason():
    v = extract_verdict("[slurm-submit] job 9 finished: state=deadline phase=bringup exit=1 reason=")
    assert v["reason"] == ""
    assert v["failure_class"] == "infra"


def test_extract_verdict_on_a_log_without_the_line():
    assert extract_verdict("nothing here") == {}
    assert extract_verdict(None) == {}


def test_failure_classification_separates_cluster_from_product():
    # This distinction is the difference between hitting retry and filing a
    # bug, and Buildkite's red/green throws it away.
    assert classify_state("infra-NODE_FAIL") == "infra"
    assert classify_state("infra-stuck-PENDING") == "infra"
    assert classify_state("preflight-rejected") == "infra"
    assert classify_state("server-failed") == "bringup"
    assert classify_state("bringup-timeout") == "bringup"
    assert classify_state("workload-timeout") == "workload"
    assert classify_state("completed-no-verdict") == "workload"
    assert classify_state("COMPLETED") == "ok"
    assert classify_state("") == "unknown"


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def test_upsert_is_idempotent(tmp_path):
    path = tmp_path / "jobs.jsonl"
    records = [job_record(BUILD, _job(id=f"job-{i}")) for i in range(3)]
    upsert_job_records(path, records)
    first = path.read_text()
    upsert_job_records(path, records)
    assert path.read_text() == first
    assert len(load_job_records(path)) == 3


def test_upsert_replaces_a_running_job_with_its_finished_record(tmp_path):
    path = tmp_path / "jobs.jsonl"
    upsert_job_records(path, [job_record(BUILD, _job(state="running", exit_status=None))])
    upsert_job_records(path, [job_record(BUILD, _job(state="passed", exit_status=0))])
    records = load_job_records(path)
    assert len(records) == 1
    assert records[0]["verdict"] == "passed"


def test_load_skips_a_truncated_final_line(tmp_path):
    path = tmp_path / "jobs.jsonl"
    path.write_text(json.dumps({"build_number": 1, "job_id": "a"}) + "\n{\"partial\"")
    assert len(load_job_records(path)) == 1
