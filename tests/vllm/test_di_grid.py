"""DI grid assembly tests."""

from collections import Counter

from vllm.build_di_grid import MIN_SAMPLES_FOR_RATE, build_grid, expected_cells
from vllm.di_collect import job_record
from vllm.di_labels import parse_label

BUILD = {
    "number": 100,
    "web_url": "https://buildkite.com/vllm/amd-distributed-inference-ci/builds/100",
    "state": "passed",
    "branch": "main",
    "commit": "abcdef123456",
    "created_at": "2026-08-12T22:04:11.000Z",
}

LABEL = "DeepSeek-V3-PD-1P1D-TP8-MoRIIO-proxy"
CELL_ID = parse_label(LABEL).cell_id

# Same model, shape and router as LABEL — only the mode differs.
WIDE_EP_LABEL = "DeepSeek-V3-PD-1P1D-EP8/DP8-WideEP-MoRIIO-proxy"
WIDE_EP_CELL_ID = parse_label(WIDE_EP_LABEL).cell_id


def record(build_number, state="passed", label=LABEL, **overrides):
    build = dict(BUILD, number=build_number)
    job = {
        "type": "script",
        "id": f"job-{build_number}-{label}",
        "name": label,
        "state": state,
        "exit_status": 0 if state == "passed" else 1,
        "agent": {"name": overrides.pop("agent", "mi350-agent-1")},
        "agent_query_rules": ["queue=amd_mi350_ainic"],
        "runnable_at": "2026-08-12T22:05:00.000Z",
        "started_at": "2026-08-12T22:35:00.000Z",
        "finished_at": "2026-08-12T23:35:00.000Z",
    }
    job.update(overrides)
    return job_record(build, job)


def cell_by_id(grid, cell_id):
    return next(c for c in grid["cells"] if c["cell_id"] == cell_id)


# ---------------------------------------------------------------------------
# Shape of the grid
# ---------------------------------------------------------------------------

def test_every_grid_is_enumerated_in_full():
    cells = expected_cells()
    assert Counter(c["grid"] for c in cells) == {"tp8": 20, "wide-ep": 20}


def test_cells_that_never_ran_are_rendered_not_omitted():
    grid = build_grid([])
    assert len(grid["cells"]) == 40
    assert all(c["last_verdict"] == "never_run" for c in grid["cells"])
    assert cell_by_id(grid, CELL_ID)["pass_rate"] is None


def test_tp8_and_wide_ep_cells_do_not_collide():
    # These two differ in mode alone. Any structure keyed on
    # (model, shape, router) folds them into one, which is the bug this
    # whole split exists to prevent.
    grid = build_grid([
        record(1, label=LABEL),
        record(1, state="failed", label=WIDE_EP_LABEL),
    ])
    tp8 = cell_by_id(grid, CELL_ID)
    wide = cell_by_id(grid, WIDE_EP_CELL_ID)
    assert (tp8["grid"], wide["grid"]) == ("tp8", "wide-ep")
    assert (tp8["last_verdict"], wide["last_verdict"]) == ("passed", "failed")
    assert not [c for c in grid["cells"] if c.get("unexpected")]


def test_a_renamed_step_surfaces_as_an_extra_cell():
    # If someone renames a step upstream its records stop matching the
    # enumerated grid. They must not disappear.
    grid = build_grid([record(1, label="DeepSeek-V9-PD-1P1D-TP8-MoRIIO-proxy")])
    extra = [c for c in grid["cells"] if c.get("unexpected")]
    assert len(extra) == 1
    assert extra[0]["model"] == "DeepSeek-V9"


def test_a_step_dead_for_several_builds_is_retired_not_unexpected():
    # The real case: router "toy proxy" ran until build 15 and never again. A
    # step renamed today and one dead for months want opposite volumes on the
    # page, so the same detection has to sort them by recency.
    gone = "DeepSeek-V3-PD-1P1D-TP8-MoRIIO-toy proxy"
    grid = build_grid([record(1, label=gone)] + [record(n) for n in (2, 3, 4)])
    retired = [c for c in grid["cells"] if c.get("retired")]
    assert [c["router"] for c in retired] == ["toy proxy"]
    assert not [c for c in grid["cells"] if c.get("unexpected")]

    # Still running in the newest build: an alert, not history.
    grid = build_grid([record(n, label=gone) for n in (1, 4)] + [record(n) for n in (2, 3)])
    assert [c["router"] for c in grid["cells"] if c.get("unexpected")] == ["toy proxy"]
    assert not [c for c in grid["cells"] if c.get("retired")]


def test_unparseable_labels_go_to_a_visible_bucket():
    grid = build_grid([record(1, label="Renamed Step")])
    assert len(grid["unclassified"]) == 1
    assert grid["unclassified"][0]["label"] == "Renamed Step"


# ---------------------------------------------------------------------------
# Rates and history
# ---------------------------------------------------------------------------

def test_pass_rate_and_history_order():
    records = [record(n, state="passed" if n % 2 else "failed") for n in range(1, 5)]
    cell = cell_by_id(build_grid(records), CELL_ID)
    assert cell["completed"] == 4
    assert cell["passed"] == 2
    assert cell["pass_rate"] == 0.5
    assert [h["build_number"] for h in cell["history"]] == [4, 3, 2, 1]
    assert cell["last_verdict"] == "failed"


def test_a_thin_sample_is_marked_unreportable():
    # 120-minute jobs on a scarce allocation: three samples is normal, and a
    # confident percentage from three samples is a lie.
    thin = build_grid([record(n) for n in range(1, 4)])
    assert cell_by_id(thin, CELL_ID)["rate_is_reportable"] is False
    thick = build_grid([record(n) for n in range(1, MIN_SAMPLES_FOR_RATE + 1)])
    assert cell_by_id(thick, CELL_ID)["rate_is_reportable"] is True


def test_queued_steps_are_excluded_from_the_pass_rate():
    records = [record(1, state="passed"), record(2, state="scheduled")]
    cell = cell_by_id(build_grid(records), CELL_ID)
    assert cell["attempts"] == 2
    assert cell["completed"] == 1
    assert cell["pass_rate"] == 1.0


def test_flip_count_separates_alternating_from_steadily_red():
    alternating = [record(n, state="passed" if n % 2 else "failed") for n in range(1, 7)]
    steady = [record(n, state="failed") for n in range(1, 7)]
    assert cell_by_id(build_grid(alternating), CELL_ID)["flips"] == 5
    assert cell_by_id(build_grid(steady), CELL_ID)["flips"] == 0


def test_median_runtime_and_queue_wait():
    cell = cell_by_id(build_grid([record(1)]), CELL_ID)
    assert cell["median_runtime_s"] == 3600.0
    assert cell["median_queue_wait_s"] == 1800.0


# ---------------------------------------------------------------------------
# Cross-cutting panels
# ---------------------------------------------------------------------------

def test_agent_attribution_ranks_the_worst_agent_first():
    records = [
        record(1, state="failed", agent="bad-node"),
        record(2, state="failed", agent="bad-node"),
        record(3, state="passed", agent="good-node"),
    ]
    agents = build_grid(records)["agents"]
    assert agents[0]["agent_name"] == "bad-node"
    assert agents[0]["failed"] == 2
    assert agents[0]["failure_rate"] == 1.0


def test_failure_classes_count_only_failures():
    records = [record(1, state="failed"), record(2, state="passed")]
    records[0]["failure_class"] = "infra"
    records[1]["failure_class"] = "ok"
    assert build_grid(records)["failure_classes"] == {"infra": 1}


def test_builds_are_listed_newest_first():
    grid = build_grid([record(1), record(5), record(3)])
    assert [b["build_number"] for b in grid["builds"]] == [5, 3, 1]
