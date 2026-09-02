"""Unit tests for scripts/vllm/collect_queue_snapshot.py.

Covers wait-time summary math, queue-metrics seeding, the legacy fallback,
and the ``queue_jobs.json`` side effect the dashboard depends on.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from vllm import collect_queue_snapshot as cqs


class TestWaitSummary:
    def test_empty_returns_nullable_block(self):
        assert cqs._wait_summary([]) == {
            "p50": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
            "avg": None,
        }

    def test_values_rounded_to_one_decimal(self):
        out = cqs._wait_summary([1.0, 2.0, 3.0, 4.0, 5.0])
        assert out == {
            "p50": 3.0,
            "p75": 4.0,
            "p90": 5.0,
            "p95": 5.0,
            "p99": 5.0,
            "max": 5.0,
            "avg": 3.0,
        }


class TestOfficialWaitSummary:
    def test_only_exposes_queue_native_metrics(self):
        assert cqs._wait_summary_from_queue_metrics(
            {
                "min": 60,
                "p50": 120,
                "p95": 900,
                "max": 1200,
            }
        ) == {
            "min": 1.0,
            "p50": 2.0,
            "p95": 15.0,
            "max": 20.0,
        }

    def test_missing_native_metrics_remain_null(self):
        assert cqs._wait_summary_from_queue_metrics({"p50": 120}) == {
            "min": None,
            "p50": 2.0,
            "p95": None,
            "max": None,
        }


class TestRestApiResilience:
    def test_rate_limit_raises_instead_of_looking_like_an_empty_page(self, monkeypatch):
        class RateLimitedResponse:
            status_code = 429
            headers: dict = {}

        monkeypatch.setattr(cqs.requests, "get", lambda *args, **kwargs: RateLimitedResponse())

        with pytest.raises(RuntimeError, match="rate limited"):
            cqs.bk_get("/organizations/vllm/builds", "fake-token")

    def test_rate_limited_legacy_scan_aborts_before_publishing_jobs(self, monkeypatch, tmp_path):
        output = tmp_path / "queue_timeseries.jsonl"
        monkeypatch.setattr(cqs, "OUTPUT", output, raising=False)
        monkeypatch.setattr(
            cqs,
            "fetch_cluster_queue_metrics",
            lambda token: (_ for _ in ()).throw(RuntimeError("metrics unavailable")),
        )
        monkeypatch.setattr(
            cqs,
            "fetch_active_cluster_jobs",
            lambda token, queue_ids_by_key=None: (_ for _ in ()).throw(
                RuntimeError("graphql unavailable")
            ),
        )
        monkeypatch.setattr(
            cqs,
            "bk_get",
            lambda path, token, params=None: (_ for _ in ()).throw(
                RuntimeError("Buildkite REST API rate limited")
            ),
        )

        with pytest.raises(RuntimeError, match="rate limited"):
            cqs.collect_snapshot("fake-token")

        assert not (output.parent / "queue_jobs.json").exists()

    def test_pagination_continues_beyond_five_full_pages(self, monkeypatch):
        requested_pages = []

        def fake_get(path, token, params=None):
            page = (params or {}).get("page", 1)
            requested_pages.append(page)
            return [{"page": page}] * 100 if page <= 6 else []

        monkeypatch.setattr(cqs, "bk_get", fake_get)

        rows = cqs.bk_get_paginated("/organizations/vllm/builds", "fake-token")

        assert len(rows) == 600
        assert requested_pages == list(range(1, 8))

    def test_full_final_page_raises_at_safety_cap(self, monkeypatch):
        monkeypatch.setattr(
            cqs,
            "bk_get",
            lambda path, token, params=None: [{"page": (params or {}).get("page")}] * 100,
        )

        with pytest.raises(RuntimeError, match="pagination safety cap"):
            cqs.bk_get_paginated(
                "/organizations/vllm/builds",
                "fake-token",
                max_pages=2,
            )

    def test_legacy_state_scans_deduplicate_transitioning_jobs(self, monkeypatch):
        def fake_paginated(path, token, params=None, max_pages=None):
            requested_state = (params or {}).get("state")
            job_state = "running" if requested_state == "running" else "scheduled"
            return [
                {
                    "number": 42,
                    "branch": "main",
                    "commit": "abc123def456",
                    "source": "schedule",
                    "pipeline": {"slug": "amd-ci"},
                    "web_url": "https://buildkite.com/vllm/amd-ci/builds/42",
                    "jobs": [
                        {
                            "type": "script",
                            "id": "transitioning-job",
                            "state": job_state,
                            "name": "mi250_1: test",
                            "agent_query_rules": ["queue=amd_mi250_1"],
                            "runnable_at": "2026-08-04T18:00:00Z",
                            "started_at": (
                                "2026-08-04T18:01:00Z" if job_state == "running" else None
                            ),
                        }
                    ],
                }
            ]

        monkeypatch.setattr(cqs, "bk_get_paginated", fake_paginated)

        jobs = cqs._collect_legacy_active_jobs("fake-token")

        assert len(jobs) == 1
        assert jobs[0]["job_uuid"] == "transitioning-job"
        assert jobs[0]["state"] == "RUNNING"


class TestRewriteJobUrl:
    def test_hash_style_converted_to_step_canvas(self):
        url = "https://buildkite.com/vllm/amd-ci/builds/12345#deadbeef-1234-5678-90ab-cdef01234567"
        expected = (
            "https://buildkite.com/vllm/amd-ci/builds/12345/steps/canvas"
            "?jid=deadbeef-1234-5678-90ab-cdef01234567&tab=output"
        )
        assert cqs._rewrite_job_url(url) == expected

    def test_canonical_url_passthrough(self):
        url = "https://buildkite.com/vllm/amd-ci/builds/12345/jobs/abc"
        assert cqs._rewrite_job_url(url) == url

    def test_empty_returns_empty(self):
        assert cqs._rewrite_job_url("") == ""


class TestGraphqlQueueMetrics:
    def test_fetches_official_queue_wait_metrics(self, monkeypatch):
        def fake_graphql(query, token, variables):
            return {
                "organization": {
                    "cluster": {
                        "queues": {
                            "edges": [
                                {
                                    "node": {
                                        "id": "ClusterQueueID",
                                        "key": "amd_mi355_1",
                                        "uuid": "queue-uuid",
                                        "dispatchPaused": False,
                                        "metrics": {
                                            "timestamp": "2026-05-20T12:00:00Z",
                                            "connectedAgentsCount": 8,
                                            "waitingJobsCount": 3,
                                            "runningJobsCount": 5,
                                            "jobsPassedCount": 12,
                                            "jobsFailedCount": 2,
                                            "waitTimeSec": {
                                                "min": 60,
                                                "p50": 120,
                                                "p95": 900,
                                                "max": 1200,
                                            },
                                        },
                                    }
                                }
                            ],
                            "pageInfo": {"hasNextPage": False, "endCursor": None},
                        }
                    }
                }
            }

        monkeypatch.setattr(cqs, "bk_graphql", fake_graphql)

        metrics = cqs.fetch_cluster_queue_metrics("fake-token")

        assert metrics["amd_mi355_1"]["graphql_id"] == "ClusterQueueID"
        assert metrics["amd_mi355_1"]["official_wait"] == {
            "min": 1.0,
            "p50": 2.0,
            "p95": 15.0,
            "max": 20.0,
        }
        assert "jobsPassedCount" in cqs.GRAPHQL_QUEUE_METRICS_Q
        assert "jobsFailedCount" in cqs.GRAPHQL_QUEUE_METRICS_Q
        assert metrics["amd_mi355_1"]["jobs_passed"] == 12
        assert metrics["amd_mi355_1"]["jobs_failed"] == 2
        assert "p99" not in metrics["amd_mi355_1"]["official_wait"]

    def test_native_activity_zero_is_preserved_but_missing_stays_null(self, monkeypatch):
        monkeypatch.setattr(
            cqs,
            "bk_graphql",
            lambda query, token, variables: {
                "organization": {
                    "cluster": {
                        "queues": {
                            "edges": [
                                {
                                    "node": {
                                        "key": "amd_mi250_1",
                                        "metrics": {
                                            "waitingJobsCount": 0,
                                            "runningJobsCount": 0,
                                            "jobsPassedCount": 0,
                                        },
                                    }
                                }
                            ],
                            "pageInfo": {"hasNextPage": False},
                        }
                    }
                }
            },
        )

        row = cqs.fetch_cluster_queue_metrics("fake-token")["amd_mi250_1"]

        assert row["jobs_passed"] == 0
        assert row["jobs_failed"] is None

    def test_fetches_jobs_by_cluster_queue_graphql_id(self, monkeypatch):
        calls = []

        def fake_graphql(query, token, variables):
            calls.append((query, variables))
            return {
                "organization": {
                    "jobs": {
                        "edges": [
                            {
                                "node": {
                                    "uuid": "deadbeef-1234-5678-90ab-cdef01234567",
                                    "state": "SCHEDULED",
                                    "label": "mi355_1: queue check",
                                    "runnableAt": "2026-05-20T12:00:00Z",
                                    "scheduledAt": "2026-05-20T12:00:00Z",
                                    "createdAt": "2026-05-20T11:59:00Z",
                                    "startedAt": None,
                                    "agentQueryRules": [],
                                    "clusterQueue": {"key": "amd_mi355_1"},
                                    "build": {
                                        "number": 123,
                                        "branch": "main",
                                        "commit": "abcdef1234567890",
                                        "url": "https://buildkite.com/vllm/amd-ci/builds/123",
                                    },
                                    "pipeline": {"slug": "amd-ci"},
                                }
                            }
                        ],
                        "pageInfo": {"hasNextPage": False, "endCursor": None},
                    }
                }
            }

        monkeypatch.setattr(cqs, "bk_graphql", fake_graphql)

        jobs = cqs.fetch_active_cluster_jobs("fake-token", {"amd_mi355_1": "ClusterQueueID"})

        assert calls[0][0] == cqs.GRAPHQL_QUEUE_JOBS_Q
        assert "$queue: [ID!]!" in calls[0][0]
        assert "clusterQueue: $queue" in calls[0][0]
        assert calls[0][1]["queue"] == ["ClusterQueueID"]
        assert jobs[0]["queue"] == "amd_mi355_1"
        assert jobs[0]["state"] == "SCHEDULED"


class TestQueueExclusions:
    def test_predicate_is_case_insensitive_and_covers_suffixes(self):
        assert cqs.is_excluded_queue("amd_mi355B")
        assert cqs.is_excluded_queue("AMD_MI355b_8")
        assert cqs.is_excluded_queue("amd_mi355B_future_suffix")
        assert not cqs.is_excluded_queue("amd_mi250_8")
        assert not cqs.is_excluded_queue("amd_mi355_8")
        assert not cqs.is_excluded_queue("amd_mi250_4")
        assert all(not cqs.is_excluded_queue(queue) for queue in cqs.TRACKED_QUEUES)

    def test_amd_scope_includes_cpu_and_excludes_mi355b(self):
        assert cqs.is_amd_queue("AMD_MI250_1")
        assert cqs.is_amd_queue("amd-cpu")
        assert cqs.is_amd_queue("amd_mi250_8")
        assert not cqs.is_amd_queue("amd_mi355B_8")
        assert not cqs.is_amd_queue("gpu_1_queue")


def _history_snapshot(ts: str) -> dict:
    return {
        "ts": ts,
        "queues": {
            "amd_mi250_1": {
                "waiting": 0,
                "running": 0,
                "official_wait": {"p50": None, "p95": None, "max": None},
                "sample_wait": {
                    "available": True,
                    "count": 0,
                    "p50": None,
                    "p75": None,
                    "p90": None,
                    "p95": None,
                    "p99": None,
                    "max": None,
                    "avg": None,
                },
                "current_wait": {
                    "p50": {"value": None, "source": None},
                    "p95": {"value": None, "source": None},
                    "p99": {"value": None, "source": None},
                },
                "p50_wait": None,
                "p50_wait_source": None,
                "p95_wait": None,
                "p95_wait_source": None,
                "p99_wait": None,
                "p99_wait_source": None,
            }
        },
        "total_waiting": 0,
        "total_running": 0,
        "sources": {"counts": "cluster_metrics", "wait_fields": {}},
    }


class TestHistoryPrune:
    def test_drops_pre_reset_but_migrates_old_schema_rows(self, tmp_path):
        path = tmp_path / "queue_timeseries.jsonl"
        path.write_text(
            json.dumps(
                {
                    "ts": "2026-04-20T22:00:00Z",
                    "queues": {"amd_mi250_1": {"waiting": 1}},
                    "total_waiting": 1,
                    "total_running": 0,
                }
            )
            + "\n"
            + json.dumps(
                {
                    "ts": "2026-04-20T23:45:00Z",
                    "queues": {
                        "amd_mi250_1": {
                            "waiting": 2,
                            "running": 3,
                            "connected_agents": 9,
                            "p50_wait": 0,
                            "p75_wait": 0,
                            "p90_wait": 0,
                            "p99_wait": 0,
                            "max_wait": 0,
                            "avg_wait": 0,
                        }
                    },
                    "total_waiting": 2,
                    "total_running": 3,
                    "sources": {"counts": "cluster_metrics", "waits": "cluster_metrics"},
                }
            )
            + "\n"
            + json.dumps(_history_snapshot("2026-04-20T23:46:00Z"))
            + "\n"
        )

        total, kept = cqs.prune_history_file(path, now=datetime(2026, 4, 21, tzinfo=timezone.utc))
        assert total == 3
        assert kept == 2
        lines = [line for line in path.read_text().splitlines() if line.strip()]
        assert len(lines) == 2
        migrated = json.loads(lines[0])
        assert migrated["ts"] == "2026-04-20T23:45:00Z"
        assert (migrated["total_waiting"], migrated["total_running"]) == (2, 3)
        row = migrated["queues"]["amd_mi250_1"]
        assert row["count_source"] == "historical_counts"
        assert row["count_provenance"]["original_source"] == "cluster_metrics"
        assert row["connected_agents"] is None
        assert row["connected_agents_source"] is None
        assert row["official_wait"] == {"p50": None, "p95": None, "max": None}
        assert row["p50_wait"] is None
        assert row["p95_wait"] is None
        assert row["p99_wait"] is None
        assert migrated["sources"]["history_provenance"]["migration"] == (
            "legacy_queue_snapshot_v1_to_v2"
        )

    def test_existing_history_is_not_backfilled_with_verbose_live_evidence(self):
        normalized = cqs.normalize_history_snapshot(_history_snapshot("2026-08-10T12:00:00Z"))

        row = normalized["queues"]["amd_mi250_1"]
        assert "field_provenance" not in row
        assert "wait_sample_reconciliation" not in row
        assert "wait_sample_promotable" not in row
        assert "min" not in row["official_wait"]
        assert "min_wait" not in row
        assert "jobs_passed" not in row
        assert "jobs_failed" not in row
        assert "target_queue_scope" not in normalized
        assert "target" not in normalized["scope_totals"]
        assert "metric_fields" not in normalized["sources"]
        assert "target_queue_scope" not in normalized["sources"]
        assert "native_activity" not in normalized["sources"]

        # Re-reading an old row remains compact and does not gradually accrete
        # fields during every prune or history merge.
        assert cqs.normalize_history_snapshot(normalized) == normalized

    def test_migrated_waits_require_nonzero_sample_and_are_labeled_sampled(self):
        snapshot = {
            "ts": "2026-06-20T12:00:00Z",
            "queues": {
                "amd_mi250_1": {
                    "waiting": 2,
                    "running": 1,
                    "wait_sample_count": 2,
                    "p50_wait": 1.5,
                    "p75_wait": 2.0,
                    "p90_wait": 2.5,
                    "p95_wait": 2.5,
                    "p99_wait": 2.5,
                    "max_wait": 2.5,
                    "avg_wait": 2.0,
                    "wait_source": "cluster_metrics",
                },
                "amd_mi250_2": {
                    "waiting": 0,
                    "running": 4,
                    "wait_sample_count": 0,
                    "p50_wait": 0,
                    "p95_wait": 0,
                    "p99_wait": 0,
                },
            },
            "sources": {"counts": "active_job_scan", "waits": "scheduled_jobs"},
        }

        migrated = cqs.normalize_history_snapshot(snapshot)
        sampled = migrated["queues"]["amd_mi250_1"]
        unsupported = migrated["queues"]["amd_mi250_2"]

        assert sampled["sample_wait"]["count"] == 2
        assert sampled["sample_wait"]["source"] == "historical_scheduled_job_sample"
        assert (sampled["p50_wait"], sampled["p50_wait_source"]) == (1.5, "sample_wait")
        assert (sampled["p99_wait"], sampled["p99_wait_source"]) == (2.5, "sample_wait")
        assert unsupported["sample_wait"]["count"] == 0
        assert unsupported["p50_wait"] is None
        assert unsupported["p95_wait"] is None
        assert unsupported["p99_wait"] is None
        assert migrated["sources"]["official_wait"] == "unavailable"
        remigrated = cqs.normalize_history_snapshot(migrated)
        assert remigrated["sources"]["sampled_wait"] == ("historical_scheduled_job_sample")
        assert (
            remigrated["sources"]["history_provenance"]
            == (migrated["sources"]["history_provenance"])
        )

    def test_migration_removes_excluded_queues_and_recomputes_totals(self):
        snapshot = {
            "ts": "2026-06-20T12:00:00Z",
            "queues": {
                "amd_mi250_1": {"waiting": 1, "running": 2},
                "AMD_MI355b_1": {"waiting": 50, "running": 60},
                "amd_mi355B_future_suffix": {"waiting": 70, "running": 80},
            },
            "total_waiting": 121,
            "total_running": 142,
        }

        migrated = cqs.normalize_history_snapshot(snapshot)

        assert set(migrated["queues"]) == {"amd_mi250_1"}
        assert migrated["total_waiting"] == 1
        assert migrated["total_running"] == 2

    def test_inconsistent_workload_splits_become_unavailable_without_changing_totals(self):
        snapshot = {
            "ts": "2026-06-20T12:00:00Z",
            "queues": {
                "amd_mi250_1": {
                    "waiting": 1,
                    "running": 2,
                    "waiting_by_workload": {"vllm": 0, "omni": 2},
                    "running_by_workload": {"vllm": 1, "omni": 0},
                },
                "amd-cpu": {
                    "waiting": 3,
                    "running": 4,
                    "waiting_by_workload": {"vllm": 3, "omni": 0},
                    "running_by_workload": {"vllm": 4, "omni": 0},
                },
                "gpu_1_queue": {"waiting": 5, "running": 6},
                "AMD_MI355b_8": {"waiting": 100, "running": 100},
            },
            "total_waiting": 109,
            "total_running": 112,
            "sources": {
                "counts": "active_job_scan",
                "active_jobs": "legacy_build_scan",
            },
        }

        migrated = cqs.normalize_history_snapshot(snapshot)
        row = migrated["queues"]["amd_mi250_1"]

        assert row["waiting_by_workload"] is None
        assert row["waiting_by_workload_provenance"] == {
            "available": False,
            "status": "inconsistent",
            "source": "legacy_build_scan",
            "reason": "observed_split_exceeds_queue_total",
            "queue_total": 1,
            "observed_split_total": 2,
            "observed_split": {"omni": 2, "vllm": 0},
        }
        assert row["running_by_workload"] == {"omni": 0, "vllm": 1}
        assert row["running_by_workload_provenance"]["status"] == "partial"
        assert migrated["total_waiting"] == 9
        assert migrated["total_running"] == 12
        assert migrated["scope_totals"]["all"] == {
            "waiting": 9,
            "running": 12,
            "count_source": "historical_counts",
            "count_sources": ["historical_counts"],
            "queue_count": 3,
        }
        assert migrated["scope_totals"]["amd"] == {
            "waiting": 4,
            "running": 6,
            "count_source": "historical_counts",
            "count_sources": ["historical_counts"],
            "queue_count": 2,
        }
        assert "no remainder is assigned" in migrated["sources"]["workload_split_fields"]["rule"]

        remigrated = cqs.normalize_history_snapshot(migrated)
        assert remigrated["queues"]["amd_mi250_1"]["waiting_by_workload"] is None
        assert (
            remigrated["queues"]["amd_mi250_1"]["waiting_by_workload_provenance"]
            == row["waiting_by_workload_provenance"]
        )

    def test_merge_is_deterministic_and_deduplicates_timestamps(self, tmp_path):
        path = tmp_path / "queue_timeseries.jsonl"
        local = _history_snapshot("2026-06-21T00:00:00Z")
        local["merge_marker"] = "local"
        path.write_text(json.dumps(local) + "\n")
        duplicate = _history_snapshot("2026-06-21T00:00:00Z")
        duplicate["merge_marker"] = "incoming"
        incoming = [
            _history_snapshot("2026-06-20T00:00:00Z"),
            duplicate,
        ]

        assert cqs.merge_history_rows(path, incoming) == (2, 2)
        first_write = path.read_text()
        assert cqs.merge_history_rows(path, incoming) == (2, 2)
        assert path.read_text() == first_write
        assert [json.loads(line)["ts"] for line in first_write.splitlines()] == [
            "2026-06-20T00:00:00Z",
            "2026-06-21T00:00:00Z",
        ]
        assert json.loads(first_write.splitlines()[-1])["merge_marker"] == "local"

    def test_drops_rows_older_than_retention_window(self, tmp_path):
        path = tmp_path / "queue_timeseries.jsonl"

        def snapshot(ts: str) -> str:
            return json.dumps(_history_snapshot(ts))

        path.write_text(
            snapshot("2026-05-17T00:00:00Z") + "\n" + snapshot("2026-05-20T00:00:00Z") + "\n"
        )

        total, kept = cqs.prune_history_file(path, now=datetime(2026, 6, 18, tzinfo=timezone.utc))
        assert total == 2
        assert kept == 1
        lines = [line for line in path.read_text().splitlines() if line.strip()]
        assert len(lines) == 1
        assert json.loads(lines[0])["ts"] == "2026-05-20T00:00:00Z"

    def test_archive_compaction_keeps_hourly_peak_and_recent_full_resolution(self, tmp_path):
        path = tmp_path / "queue_timeseries.jsonl"

        def snapshot(ts: str, p95: float, max_wait: float | None = None) -> dict:
            row = _history_snapshot(ts)
            queue = row["queues"]["amd_mi250_1"]
            queue["official_wait"] = {
                "p50": p95 / 2,
                "p95": p95,
                "max": p95 if max_wait is None else max_wait,
            }
            queue["official_wait_source"] = "queue_native_metrics"
            queue["count_source"] = "cluster_metrics"
            return row

        rows = [
            snapshot("2026-06-16T10:05:00Z", 5.0),
            snapshot("2026-06-16T10:25:00Z", 80.0),
            snapshot("2026-06-16T10:55:00Z", 10.0, 100.0),
            snapshot("2026-06-18T11:05:00Z", 1.0),
            snapshot("2026-06-18T11:15:00Z", 2.0),
            snapshot("2026-06-18T11:25:00Z", 3.0),
        ]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

        total, kept = cqs.prune_history_file(
            path, now=datetime(2026, 6, 18, 12, 0, tzinfo=timezone.utc)
        )

        output = [json.loads(line) for line in path.read_text().splitlines()]
        assert total == 6
        assert kept == 4
        assert [row["ts"] for row in output] == [
            "2026-06-16T10:25:00Z",
            "2026-06-18T11:05:00Z",
            "2026-06-18T11:15:00Z",
            "2026-06-18T11:25:00Z",
        ]
        assert output[0]["queues"]["amd_mi250_1"]["p95_wait"] == 80.0

    def test_archive_envelope_preserves_each_queues_peak_timestamp(self):
        def snapshot(ts: str, mi250_p95: float, mi300_p95: float) -> dict:
            row = _history_snapshot(ts)
            base = row["queues"]["amd_mi250_1"]
            queues = {}
            for name, p95 in (
                ("amd_mi250_1", mi250_p95),
                ("amd_mi300_1", mi300_p95),
            ):
                queue = json.loads(json.dumps(base))
                queue["official_wait"] = {"p50": p95 / 2, "p95": p95, "max": p95}
                queue["official_wait_source"] = "queue_native_metrics"
                queue["sample_wait"] = {
                    "available": True,
                    "count": 2,
                    "p50": p95 / 2 + 1,
                    "p95": p95 + 5,
                    "p99": p95 + 8,
                }
                queue["sample_wait_source"] = "cluster_queue_graphql"
                queue["wait_sample_count"] = 2
                queue["count_source"] = "cluster_metrics"
                queues[name] = queue
            row["queues"] = queues
            return row

        compacted = cqs.compact_history_resolution(
            [
                snapshot("2026-06-16T10:05:00Z", 80.0, 10.0),
                snapshot("2026-06-16T10:25:00Z", 20.0, 90.0),
            ],
            datetime(2026, 6, 18, 12, 0, tzinfo=timezone.utc),
        )

        assert len(compacted) == 1
        archived = compacted[0]
        assert archived["history_mode"] == "hourly_queue_wait_peaks"
        mi250 = archived["queues"]["amd_mi250_1"]["archive_wait_peaks"]["p95"]
        mi300 = archived["queues"]["amd_mi300_1"]["archive_wait_peaks"]["p95"]
        assert (mi250["value"], mi250["observed_at"]) == (
            80.0,
            "2026-06-16T10:05:00Z",
        )
        assert (mi300["value"], mi300["observed_at"]) == (
            90.0,
            "2026-06-16T10:25:00Z",
        )
        mi250_sample = archived["queues"]["amd_mi250_1"]["archive_sample_wait_peaks"]["p95"]
        mi300_sample = archived["queues"]["amd_mi300_1"]["archive_sample_wait_peaks"]["p95"]
        assert (mi250_sample["value"], mi250_sample["observed_at"]) == (
            85.0,
            "2026-06-16T10:05:00Z",
        )
        assert (mi300_sample["value"], mi300_sample["observed_at"]) == (
            95.0,
            "2026-06-16T10:25:00Z",
        )
        assert (
            cqs.compact_history_resolution(
                compacted,
                datetime(2026, 6, 18, 12, 0, tzinfo=timezone.utc),
            )
            == compacted
        )

    def test_equal_timestamp_raw_merge_retains_incoming_archive_peaks(self, tmp_path):
        raw = _history_snapshot("2026-06-16T10:25:00Z")
        envelope = json.loads(json.dumps(raw))
        envelope["history_mode"] = "hourly_queue_wait_peaks"
        envelope["archive_bucket_start"] = "2026-06-16T10:00:00Z"
        envelope["queues"]["amd_mi250_1"]["archive_wait_peaks"] = {
            "p95": {
                "value": 80.0,
                "observed_at": "2026-06-16T10:05:00Z",
                "source": "sample_wait",
                "provider": "scheduled_job_scan",
                "sample_count": 4,
            }
        }
        path = tmp_path / "queue_timeseries.jsonl"
        path.write_text(json.dumps(raw) + "\n")

        cqs.merge_history_rows(path, [envelope])

        merged = json.loads(path.read_text())
        assert merged["history_mode"] == "hourly_queue_wait_peaks"
        peak = merged["queues"]["amd_mi250_1"]["archive_wait_peaks"]["p95"]
        assert (peak["value"], peak["observed_at"]) == (
            80.0,
            "2026-06-16T10:05:00Z",
        )


def _active_job(
    queue: str,
    state: str,
    *,
    runnable_at: str | None = None,
    scheduled_at: str | None = None,
    created_at: str | None = None,
    started_at: str | None = None,
    name: str = "mi250_1: foo",
    pipeline: str = "amd-ci",
    branch: str = "main",
    build: int = 100,
    commit: str = "abc123def456",
    build_url: str = "https://buildkite.com/vllm/amd-ci/builds/100",
    job_uuid: str | None = None,
) -> dict:
    return {
        "queue": queue,
        "state": state,
        "name": name,
        "job_uuid": job_uuid
        or "::".join(
            str(value or "") for value in (queue, state, runnable_at, started_at, name, build)
        ),
        "build_url": build_url,
        "pipeline": pipeline,
        "build": build,
        "branch": branch,
        "commit": commit[:12],
        "workload": cqs.classify_workload(pipeline, branch, queue),
        "fork_url": "",
        "source": "",
        "runnable_at": runnable_at,
        "scheduled_at": scheduled_at,
        "created_at": created_at,
        "started_at": started_at,
    }


class _FakeBk:
    """Inject canned Buildkite REST responses keyed by build state."""

    def __init__(self, running_builds=None, scheduled_builds=None):
        self._pages = {"running": [running_builds or []], "scheduled": [scheduled_builds or []]}

    def __call__(self, path, token, params=None):
        params = params or {}
        state = params.get("state", "")
        page = params.get("page", 1)
        pages = self._pages.get(state, [])
        if 1 <= page <= len(pages):
            return pages[page - 1]
        return []


def _build(state_pipeline="amd-ci", branch="main", jobs=None, number=100, commit="abc123def456"):
    return {
        "number": number,
        "branch": branch,
        "commit": commit,
        "source": "ui",
        "pipeline": {"slug": state_pipeline},
        "pull_request": {},
        "web_url": f"https://buildkite.com/vllm/{state_pipeline}/builds/{number}",
        "jobs": jobs or [],
    }


def _job(
    queue,
    state,
    runnable_at=None,
    started_at=None,
    name="mi250_1: foo",
    web_url="",
    job_id=None,
):
    return {
        "id": job_id
        or "::".join(str(value or "") for value in (queue, state, runnable_at, started_at, name)),
        "type": "script",
        "state": state,
        "name": name,
        "web_url": web_url,
        "agent_query_rules": [f"queue={queue}"] if queue else [],
        "runnable_at": runnable_at,
        "scheduled_at": runnable_at,
        "started_at": started_at,
    }


class TestCollectSnapshot:
    def test_tracked_queue_zero_filled_when_empty(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cqs, "OUTPUT", tmp_path / "queue_timeseries.jsonl", raising=False)
        monkeypatch.setattr(cqs, "fetch_cluster_queue_metrics", lambda token: {})
        monkeypatch.setattr(
            cqs, "fetch_active_cluster_jobs", lambda token, queue_ids_by_key=None: []
        )

        snap = cqs.collect_snapshot("fake-token")
        for q in cqs.TRACKED_QUEUES:
            assert q in snap["queues"]
            row = snap["queues"][q]
            assert row["waiting"] == 0
            assert row["running"] == 0
            assert row["p95_wait"] is None
            assert row["p99_wait"] is None
            assert row["official_wait"] == {
                "min": None,
                "p50": None,
                "p95": None,
                "max": None,
            }
            assert row["sample_wait"]["count"] == 0
            assert row["connected_agents"] is None
            assert row["connected_agents_source"] is None
        assert snap["sources"]["agents"] == "unavailable"
        assert snap["sources"]["official_wait"] == "unavailable"
        assert snap["sources"]["counts"] == "active_job_scan"

    def test_job_scan_includes_queues_without_native_counts_but_trusts_native_zero(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(cqs, "OUTPUT", tmp_path / "queue_timeseries.jsonl", raising=False)
        monkeypatch.setattr(
            cqs,
            "TRACKED_QUEUES",
            frozenset({"amd_mi250_1", "amd_mi250_2", "amd_mi300_1", "amd_mi355_1"}),
        )
        monkeypatch.setattr(
            cqs,
            "fetch_cluster_queue_metrics",
            lambda token: {
                "amd_mi250_1": {
                    "graphql_id": "native-zero-id",
                    "counts_available": True,
                    "waiting": 0,
                    "running": 0,
                },
                "amd_mi300_1": {
                    "graphql_id": "missing-counts-id",
                    "counts_available": False,
                    "waiting": 0,
                    "running": 0,
                },
                "amd_mi355_1": {
                    "graphql_id": "native-active-id",
                    "counts_available": True,
                    "waiting": 1,
                    "running": 0,
                },
            },
        )
        captured_queue_ids = []

        def fake_fetch(token, queue_ids_by_key=None):
            captured_queue_ids.append(None if queue_ids_by_key is None else dict(queue_ids_by_key))
            if queue_ids_by_key is None:
                missing = _active_job("amd_mi250_2", "SCHEDULED")
                missing["job_uuid"] = "missing-queue-job"
                duplicate = _active_job("amd_mi300_1", "SCHEDULED")
                duplicate["job_uuid"] = "native-missing-counts-job"
                return [missing, duplicate]
            scoped = _active_job("amd_mi300_1", "SCHEDULED")
            scoped["job_uuid"] = "native-missing-counts-job"
            return [scoped]

        monkeypatch.setattr(cqs, "fetch_active_cluster_jobs", fake_fetch)

        snapshot = cqs.collect_snapshot("fake-token")

        assert captured_queue_ids == [
            {
                "amd_mi300_1": "missing-counts-id",
                "amd_mi355_1": "native-active-id",
            },
            None,
        ]
        assert snapshot["queues"]["amd_mi300_1"]["waiting"] == 1
        assert snapshot["queues"]["amd_mi300_1"]["count_source"] == "active_job_scan"
        assert snapshot["queues"]["amd_mi250_1"]["waiting"] == 0
        assert snapshot["queues"]["amd_mi250_1"]["count_source"] == "cluster_metrics"
        assert snapshot["queues"]["amd_mi250_2"]["waiting"] == 1
        assert snapshot["queues"]["amd_mi250_2"]["sample_wait"]["available"] is True

    def test_excluded_queues_never_reach_rows_jobs_or_totals(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cqs, "OUTPUT", tmp_path / "queue_timeseries.jsonl", raising=False)
        monkeypatch.setattr(
            cqs,
            "fetch_cluster_queue_metrics",
            lambda token: {
                "AMD_MI355b_4": {
                    "counts_available": True,
                    "waiting": 20,
                    "running": 30,
                    "connected_agents": 40,
                    "official_wait": {"p50": 1.0, "p95": 2.0, "max": 3.0},
                },
                "amd_mi250_1": {
                    "counts_available": True,
                    "waiting": 1,
                    "running": 2,
                    "connected_agents": 3,
                },
            },
        )
        monkeypatch.setattr(
            cqs,
            "fetch_active_cluster_jobs",
            lambda token, queue_ids_by_key=None: [
                _active_job("amd_mi355B_future", "SCHEDULED"),
                _active_job("amd_mi250_1", "RUNNING"),
            ],
        )

        snap = cqs.collect_snapshot("fake-token")

        assert all(not cqs.is_excluded_queue(queue) for queue in snap["queues"])
        assert snap["total_waiting"] == 1
        assert snap["total_running"] == 2
        jobs = json.loads((tmp_path / "queue_jobs.json").read_text())
        assert all(
            not cqs.is_excluded_queue(job["queue"]) for job in [*jobs["pending"], *jobs["running"]]
        )

    def test_running_jobs_do_not_inflate_current_wait(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cqs, "OUTPUT", tmp_path / "out.jsonl", raising=False)
        monkeypatch.setattr(cqs, "fetch_cluster_queue_metrics", lambda token: {})
        monkeypatch.setattr(
            cqs,
            "fetch_active_cluster_jobs",
            lambda token, queue_ids_by_key=None: [
                _active_job("amd_mi250_1", "SCHEDULED", runnable_at="2026-04-18T11:55:00Z"),
                _active_job(
                    "amd_mi250_1",
                    "RUNNING",
                    runnable_at="2026-04-18T09:00:00Z",
                    started_at="2026-04-18T11:00:00Z",
                    name="mi250_1: long queue wait",
                ),
            ],
        )

        with patch("vllm.collect_queue_snapshot.datetime") as dt_mock:
            from datetime import datetime, timezone

            dt_mock.now.return_value = datetime(2026, 4, 18, 12, 0, 0, tzinfo=timezone.utc)
            dt_mock.fromisoformat = datetime.fromisoformat
            snap = cqs.collect_snapshot("fake-token")

        row = snap["queues"]["amd_mi250_1"]
        assert row["waiting"] == 1
        assert row["running"] == 1
        assert row["max_wait"] == 5.0
        assert row["p95_wait"] == 5.0
        assert row["p99_wait"] == 5.0
        assert row["avg_wait"] == 5.0
        assert row["sample_wait"] == {
            "available": True,
            "count": 1,
            "p50": 5.0,
            "p75": 5.0,
            "p90": 5.0,
            "p95": 5.0,
            "p99": 5.0,
            "max": 5.0,
            "avg": 5.0,
        }
        assert row["p50_wait_source"] == "sample_wait"
        assert row["p95_wait_source"] == "sample_wait"
        assert row["p99_wait_source"] == "sample_wait"
        assert row["current_wait"] == {
            "p50": {"value": 5.0, "source": "sample_wait"},
            "p95": {"value": 5.0, "source": "sample_wait"},
            "p99": {"value": 5.0, "source": "sample_wait"},
        }

    def test_zombie_jobs_are_excluded_from_analysis_counts(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cqs, "OUTPUT", tmp_path / "out.jsonl", raising=False)
        monkeypatch.setattr(
            cqs,
            "fetch_cluster_queue_metrics",
            lambda token: {
                "amd_mi250_1": {
                    "waiting": 2,
                    "running": 2,
                    "connected_agents": 4,
                    "metrics_ts": "2026-04-20T23:50:00Z",
                }
            },
        )
        monkeypatch.setattr(
            cqs,
            "fetch_active_cluster_jobs",
            lambda token, queue_ids_by_key=None: [
                _active_job("amd_mi250_1", "SCHEDULED", runnable_at="2026-04-20T19:00:00Z"),
                _active_job("amd_mi250_1", "SCHEDULED", runnable_at="2026-04-20T23:45:00Z"),
                _active_job("amd_mi250_1", "RUNNING", started_at="2026-04-20T19:00:00Z"),
                _active_job("amd_mi250_1", "RUNNING", started_at="2026-04-20T23:30:00Z"),
            ],
        )

        with patch("vllm.collect_queue_snapshot.datetime") as dt_mock:
            from datetime import datetime, timezone

            dt_mock.now.return_value = datetime(2026, 4, 20, 23, 50, 0, tzinfo=timezone.utc)
            dt_mock.fromisoformat = datetime.fromisoformat
            snap = cqs.collect_snapshot("fake-token")

        row = snap["queues"]["amd_mi250_1"]
        assert row["waiting"] == 2
        assert row["running"] == 2
        assert row["zombie_waiting"] == 1
        assert row["zombie_running"] == 1
        assert snap["total_zombie_waiting"] == 1
        assert snap["total_zombie_running"] == 1
        assert row["p95_wait"] == 5.0
        assert row["sample_wait"]["count"] == 1
        assert row["sample_wait"]["max"] == 5.0
        assert row["count_source"] == "cluster_metrics"

    def test_cluster_metrics_seed_counts_and_agents(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cqs, "OUTPUT", tmp_path / "out.jsonl", raising=False)
        monkeypatch.setattr(
            cqs,
            "fetch_cluster_queue_metrics",
            lambda token: {
                "amd_mi250_1": {
                    "waiting": 9,
                    "running": 8,
                    "connected_agents": 7,
                    "metrics_ts": "2026-04-18T12:00:00Z",
                    "queue_url": "https://buildkite.com/organizations/vllm/clusters/cluster/queues/q1",
                    "dispatch_paused": False,
                    "official_wait": {
                        "p50": 2.0,
                        "p95": 12.0,
                        "max": 20.0,
                    },
                }
            },
        )
        monkeypatch.setattr(
            cqs,
            "fetch_active_cluster_jobs",
            lambda token, queue_ids_by_key=None: [
                _active_job("amd_mi250_1", "SCHEDULED", runnable_at="2026-04-18T11:55:00Z"),
                _active_job(
                    "amd_mi250_1",
                    "RUNNING",
                    runnable_at="2026-04-18T11:58:00Z",
                    started_at="2026-04-18T11:59:00Z",
                ),
            ],
        )

        with patch("vllm.collect_queue_snapshot.datetime") as dt_mock:
            from datetime import datetime, timezone

            dt_mock.now.return_value = datetime(2026, 4, 18, 12, 0, 0, tzinfo=timezone.utc)
            dt_mock.fromisoformat = datetime.fromisoformat
            snap = cqs.collect_snapshot("fake-token")

        row = snap["queues"]["amd_mi250_1"]
        assert row["waiting"] == 9
        assert row["running"] == 8
        assert row["total"] == 17
        assert row["connected_agents"] == 7
        assert row["connected_agents_source"] == "queue_native_metrics"
        assert row["queue_url"].endswith("/queues/q1")
        assert row["wait_source"] == "cluster_metrics"
        assert row["count_source"] == "cluster_metrics"
        assert row["p50_wait"] == 2.0
        assert row["p95_wait"] == 12.0
        assert row["p99_wait"] is None
        assert row["max_wait"] == 20.0
        assert row["p50_wait_source"] == "official_wait"
        assert row["p95_wait_source"] == "official_wait"
        assert row["wait_sample_expected_count"] == 9
        assert row["wait_sample_complete"] is False
        assert row["p99_wait_source"] is None
        assert row["max_wait_source"] == "official_wait"
        assert row["p75_wait"] is None
        assert row["p75_wait_source"] is None
        assert row["sample_wait"]["p99"] == 5.0
        assert row["wait_sample_reconciliation"]["reason"] == (
            "scheduled_job_scan_below_reference_count"
        )
        assert snap["sources"]["waits"] == "cluster_metrics"
        assert snap["sources"]["official_wait"] == "queue_native_metrics"
        assert snap["sources"]["agents"] == "queue_native_metrics"
        assert row["waiting_by_workload"] == {"vllm": 1, "omni": 0}
        assert row["running_by_workload"] == {"vllm": 1, "omni": 0}

    def test_native_activity_min_wait_provenance_and_mismatch_details(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cqs, "OUTPUT", tmp_path / "out.jsonl", raising=False)
        monkeypatch.setattr(
            cqs,
            "fetch_cluster_queue_metrics",
            lambda token: {
                "amd_mi300_1": {
                    "graphql_id": "mi300-native-id",
                    "counts_available": True,
                    "waiting": 2,
                    "running": 4,
                    "connected_agents": 6,
                    "jobs_passed": 11,
                    "jobs_failed": 3,
                    "metrics_ts": "2026-08-11T20:00:00Z",
                    "official_wait": {"min": 1.0, "p50": 2.0, "p95": 9.0, "max": 12.0},
                }
            },
        )

        def fake_active_jobs(token, queue_ids_by_key=None):
            if queue_ids_by_key is None:
                return []
            return [
                _active_job(
                    "amd_mi300_1",
                    "SCHEDULED",
                    runnable_at="2026-08-11T19:55:00Z",
                )
            ]

        monkeypatch.setattr(cqs, "fetch_active_cluster_jobs", fake_active_jobs)

        with patch("vllm.collect_queue_snapshot.datetime") as dt_mock:
            dt_mock.now.return_value = datetime(2026, 8, 11, 20, 0, 0, tzinfo=timezone.utc)
            dt_mock.fromisoformat = datetime.fromisoformat
            snap = cqs.collect_snapshot("fake-token")

        row = snap["queues"]["amd_mi300_1"]
        # The independent scheduled-job read must never overwrite direct gauges.
        assert (row["waiting"], row["running"], row["connected_agents"]) == (2, 4, 6)
        assert (row["jobs_passed"], row["jobs_failed"]) == (11, 3)
        assert row["min_wait"] == 1.0
        assert row["min_wait_source"] == "official_wait"
        assert row["waiting_source"] == "queue_native_metrics"
        assert row["running_source"] == "queue_native_metrics"
        assert row["jobs_passed_source"] == "queue_native_metrics"
        assert row["jobs_failed_source"] == "queue_native_metrics"
        assert row["official_wait_source"] == "queue_native_metrics"
        assert row["metrics_ts"] == "2026-08-11T20:00:00Z"
        assert "field_provenance" not in row
        assert "official_wait_field_sources" not in row
        metric_fields = snap["sources"]["metric_fields"]
        assert metric_fields["observed_at_field"] == "metrics_ts"
        assert metric_fields["provider_fields"]["jobs_passed"] == (
            "ClusterQueue.metrics.jobsPassedCount"
        )
        assert metric_fields["provider_fields"]["official_wait.min"] == (
            "ClusterQueue.metrics.waitTimeSec.min"
        )

        reconciliation = row["wait_sample_reconciliation"]
        assert reconciliation == {
            "status": "count_mismatch",
            "reason": "scheduled_job_scan_below_reference_count",
            "reference_kind": "queue_native_waiting_jobs_including_observed_zombies",
            "reference_count": 2,
            "observed_count": 1,
            "count_delta": -1,
            "membership_verified": False,
            "native_wait_values_used": False,
        }
        assert row["wait_sample_complete"] is False
        assert row["sample_wait"]["p99"] == 5.0
        assert row["p99_wait"] is None
        assert snap["sources"]["native_activity"] == "queue_native_metrics"
        assert "amd_mi300_1" in snap["target_queue_scope"]["native_activity_queue_ids"]

    def test_target_scope_is_annotated_without_filtering_general_monitoring(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(cqs, "OUTPUT", tmp_path / "out.jsonl", raising=False)
        monkeypatch.setattr(cqs, "fetch_cluster_queue_metrics", lambda token: {})
        monkeypatch.setattr(
            cqs, "fetch_active_cluster_jobs", lambda token, queue_ids_by_key=None: []
        )

        snap = cqs.collect_snapshot("fake-token")

        scope = snap["target_queue_scope"]
        assert scope["queue_ids"] == list(cqs.AMD_METRIC_TARGET_QUEUES)
        assert scope["queue_count"] == 12
        assert scope["families"] == ["MI250", "MI300", "MI355"]
        assert scope["gpu_widths"] == [1, 2, 4, 8]
        assert scope["all_rows_present"] is True
        assert snap["scope_totals"]["target"]["queue_count"] == 12
        assert "gpu_1_queue" in snap["queues"]
        assert "wait_sample_reconciliation" not in snap["queues"]["gpu_1_queue"]
        assert cqs.normalize_history_snapshot(snap) == snap

    def test_official_max_never_becomes_p99_or_sample_only_metrics(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cqs, "OUTPUT", tmp_path / "out.jsonl", raising=False)
        monkeypatch.setattr(
            cqs,
            "fetch_cluster_queue_metrics",
            lambda token: {
                "amd_mi250_1": {
                    "waiting": 9,
                    "running": 8,
                    "connected_agents": 7,
                    "official_wait": {"p50": 2.0, "p95": 12.0, "max": 20.0},
                }
            },
        )
        monkeypatch.setattr(
            cqs, "fetch_active_cluster_jobs", lambda token, queue_ids_by_key=None: []
        )

        snap = cqs.collect_snapshot("fake-token")
        row = snap["queues"]["amd_mi250_1"]

        assert row["official_wait"] == {
            "min": None,
            "p50": 2.0,
            "p95": 12.0,
            "max": 20.0,
        }
        assert row["sample_wait"]["count"] == 0
        assert row["p50_wait"] == 2.0
        assert row["p95_wait"] == 12.0
        assert row["max_wait"] == 20.0
        assert row["p99_wait"] is None
        assert row["p99_wait_source"] is None
        assert row["current_wait"]["p99"] == {"value": None, "source": None}
        assert row["p75_wait"] is None
        assert row["p90_wait"] is None
        assert row["avg_wait"] is None
        assert "sample_wait.p99" in snap["sources"]["wait_fields"]["p99_wait"]

    def test_sample_wait_contains_all_exact_job_percentiles(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cqs, "OUTPUT", tmp_path / "out.jsonl", raising=False)
        monkeypatch.setattr(cqs, "fetch_cluster_queue_metrics", lambda token: {})
        monkeypatch.setattr(
            cqs,
            "fetch_active_cluster_jobs",
            lambda token, queue_ids_by_key=None: [
                _active_job(
                    "amd_mi250_1", "SCHEDULED", runnable_at=f"2026-04-18T11:{minute:02d}:00Z"
                )
                for minute in (59, 58, 57, 56, 55)
            ],
        )

        with patch("vllm.collect_queue_snapshot.datetime") as dt_mock:
            from datetime import datetime, timezone

            dt_mock.now.return_value = datetime(2026, 4, 18, 12, 0, 0, tzinfo=timezone.utc)
            dt_mock.fromisoformat = datetime.fromisoformat
            snap = cqs.collect_snapshot("fake-token")

        row = snap["queues"]["amd_mi250_1"]
        assert row["official_wait"] == {
            "min": None,
            "p50": None,
            "p95": None,
            "max": None,
        }
        assert row["sample_wait"] == {
            "available": True,
            "count": 5,
            "p50": 3.0,
            "p75": 4.0,
            "p90": 5.0,
            "p95": 5.0,
            "p99": 5.0,
            "max": 5.0,
            "avg": 3.0,
        }
        assert row["p50_wait"] == 3.0
        assert row["p95_wait"] == 5.0
        assert row["p99_wait"] == 5.0
        assert row["p50_wait_source"] == "sample_wait"
        assert row["p95_wait_source"] == "sample_wait"
        assert row["p99_wait_source"] == "sample_wait"
        assert row["current_wait"]["p99"] == {"value": 5.0, "source": "sample_wait"}

    def test_any_partial_sample_does_not_replace_queue_native_percentiles(self):
        for expected, sampled in ((2, 1), (20, 19), (243, 242)):
            row = cqs._apply_wait_contract(
                {
                    "waiting": expected,
                    "running": 0,
                    "zombie_waiting": 0,
                    "count_source": "cluster_metrics",
                },
                {"p50": 0.0, "p95": 75.0, "max": 90.0},
                {
                    "available": True,
                    "count": sampled,
                    "p50": 20.0,
                    "p95": 20.0,
                    "p99": 20.0,
                    "max": 20.0,
                },
            )

            assert row["wait_sample_expected_count"] == expected
            assert row["wait_sample_complete"] is False
            assert (row["p50_wait"], row["p50_wait_source"]) == (0.0, "official_wait")
            assert (row["p95_wait"], row["p95_wait_source"]) == (75.0, "official_wait")

    def test_reconciled_job_sample_stays_separate_from_native_percentiles(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(cqs, "OUTPUT", tmp_path / "out.jsonl", raising=False)
        monkeypatch.setattr(
            cqs,
            "fetch_cluster_queue_metrics",
            lambda token: {
                "amd_mi250_1": {
                    "waiting": 2,
                    "running": 0,
                    "connected_agents": 1,
                    "official_wait": {"p50": 2.0, "p95": 12.0, "max": 20.0},
                }
            },
        )
        monkeypatch.setattr(
            cqs,
            "fetch_active_cluster_jobs",
            lambda token, queue_ids_by_key=None: [
                _active_job("amd_mi250_1", "SCHEDULED", runnable_at="2026-04-18T11:55:00Z"),
                _active_job("amd_mi250_1", "SCHEDULED", runnable_at="2026-04-18T11:50:00Z"),
            ],
        )

        with patch("vllm.collect_queue_snapshot.datetime") as dt_mock:
            from datetime import datetime, timezone

            dt_mock.now.return_value = datetime(2026, 4, 18, 12, 0, 0, tzinfo=timezone.utc)
            dt_mock.fromisoformat = datetime.fromisoformat
            snap = cqs.collect_snapshot("fake-token")

        row = snap["queues"]["amd_mi250_1"]
        assert row["wait_sample_expected_count"] == 2
        assert row["wait_sample_complete"] is True
        assert (row["p50_wait"], row["p50_wait_source"]) == (2.0, "official_wait")
        assert (row["p95_wait"], row["p95_wait_source"]) == (12.0, "official_wait")
        assert (row["p99_wait"], row["p99_wait_source"]) == (10.0, "sample_wait")
        assert row["sample_wait"]["p50"] == 10.0
        assert row["sample_wait"]["p95"] == 10.0
        assert row["sample_wait"]["p99"] == 10.0
        assert row["current_wait"] == {
            "p50": {"value": 2.0, "source": "official_wait"},
            "p95": {"value": 12.0, "source": "official_wait"},
            "p99": {"value": 10.0, "source": "sample_wait"},
        }

    def test_sample_max_wins_when_it_exceeds_official_max(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cqs, "OUTPUT", tmp_path / "out.jsonl", raising=False)
        monkeypatch.setattr(
            cqs,
            "fetch_cluster_queue_metrics",
            lambda token: {
                "amd_mi250_1": {
                    "waiting": 1,
                    "running": 0,
                    "connected_agents": 1,
                    "official_wait": {"p50": 0.0, "p95": 0.0, "max": 0.0},
                }
            },
        )
        monkeypatch.setattr(
            cqs,
            "fetch_active_cluster_jobs",
            lambda token, queue_ids_by_key=None: [
                _active_job("amd_mi250_1", "SCHEDULED", runnable_at="2026-04-18T11:42:00Z"),
            ],
        )

        with patch("vllm.collect_queue_snapshot.datetime") as dt_mock:
            from datetime import datetime, timezone

            dt_mock.now.return_value = datetime(2026, 4, 18, 12, 0, 0, tzinfo=timezone.utc)
            dt_mock.fromisoformat = datetime.fromisoformat
            snap = cqs.collect_snapshot("fake-token")

        row = snap["queues"]["amd_mi250_1"]
        assert row["wait_sample_complete"] is True
        assert (row["p50_wait"], row["p50_wait_source"]) == (0.0, "official_wait")
        assert (row["p95_wait"], row["p95_wait_source"]) == (0.0, "official_wait")
        assert row["sample_wait"]["p50"] == 18.0
        assert row["sample_wait"]["p95"] == 18.0
        assert (row["p99_wait"], row["p99_wait_source"]) == (18.0, "sample_wait")
        assert (row["max_wait"], row["max_wait_source"]) == (18.0, "sample_wait")

    def test_complete_sample_wins_over_partially_available_official_percentiles(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(cqs, "OUTPUT", tmp_path / "out.jsonl", raising=False)
        monkeypatch.setattr(
            cqs,
            "fetch_cluster_queue_metrics",
            lambda token: {
                "amd_mi250_1": {
                    "waiting": 1,
                    "running": 0,
                    "connected_agents": 1,
                    "official_wait": {"p50": None, "p95": 12.0, "max": 20.0},
                }
            },
        )
        monkeypatch.setattr(
            cqs,
            "fetch_active_cluster_jobs",
            lambda token, queue_ids_by_key=None: [
                _active_job("amd_mi250_1", "SCHEDULED", runnable_at="2026-04-18T11:55:00Z"),
            ],
        )

        with patch("vllm.collect_queue_snapshot.datetime") as dt_mock:
            from datetime import datetime, timezone

            dt_mock.now.return_value = datetime(2026, 4, 18, 12, 0, 0, tzinfo=timezone.utc)
            dt_mock.fromisoformat = datetime.fromisoformat
            snap = cqs.collect_snapshot("fake-token")

        row = snap["queues"]["amd_mi250_1"]
        assert row["wait_sample_complete"] is True
        assert (row["p50_wait"], row["p50_wait_source"]) == (5.0, "sample_wait")
        assert (row["p95_wait"], row["p95_wait_source"]) == (12.0, "official_wait")
        assert row["sample_wait"]["p95"] == 5.0

    def test_missing_metrics_queues_are_sampled_by_organization_query(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cqs, "OUTPUT", tmp_path / "out.jsonl", raising=False)
        monkeypatch.setattr(
            cqs,
            "fetch_cluster_queue_metrics",
            lambda token: {
                "amd_mi250_1": {
                    "graphql_id": "ClusterQueueID",
                    "waiting": 1,
                    "running": 0,
                    "connected_agents": 1,
                    "official_wait": {"p50": 2.0, "p95": 12.0, "max": 20.0},
                }
            },
        )
        monkeypatch.setattr(
            cqs, "fetch_active_cluster_jobs", lambda token, queue_ids_by_key=None: []
        )

        snap = cqs.collect_snapshot("fake-token")

        sampled = snap["queues"]["amd_mi250_1"]["sample_wait"]
        assert sampled["available"] is True
        assert sampled["count"] == 0
        recovered = next(
            row
            for queue, row in snap["queues"].items()
            if queue != "amd_mi250_1" and row["count_source"] == "active_job_scan"
        )["sample_wait"]
        assert recovered["available"] is True
        assert recovered["count"] == 0
        assert recovered["p99"] is None

    def test_workload_split_from_omni_queue(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cqs, "OUTPUT", tmp_path / "out.jsonl", raising=False)
        monkeypatch.setattr(cqs, "fetch_cluster_queue_metrics", lambda token: {})
        monkeypatch.setattr(
            cqs,
            "fetch_active_cluster_jobs",
            lambda token, queue_ids_by_key=None: [
                _active_job("intel-gpu-omni", "SCHEDULED", runnable_at="2026-04-18T11:59:00Z"),
            ],
        )

        with patch("vllm.collect_queue_snapshot.datetime") as dt_mock:
            from datetime import datetime, timezone

            dt_mock.now.return_value = datetime(2026, 4, 18, 12, 0, 0, tzinfo=timezone.utc)
            dt_mock.fromisoformat = datetime.fromisoformat
            snap = cqs.collect_snapshot("fake-token")
        row = snap["queues"]["intel-gpu-omni"]
        assert row["waiting_by_workload"] == {"vllm": 0, "omni": 1}

    def test_workload_split_from_omni_branch(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cqs, "OUTPUT", tmp_path / "out.jsonl", raising=False)
        monkeypatch.setattr(cqs, "fetch_cluster_queue_metrics", lambda token: {})
        monkeypatch.setattr(
            cqs,
            "fetch_active_cluster_jobs",
            lambda token, queue_ids_by_key=None: [
                _active_job(
                    "amd_mi250_1",
                    "RUNNING",
                    runnable_at="2026-04-18T11:58:00Z",
                    started_at="2026-04-18T12:00:00Z",
                    branch="user/omni-feature",
                ),
            ],
        )

        with patch("vllm.collect_queue_snapshot.datetime") as dt_mock:
            from datetime import datetime, timezone

            dt_mock.now.return_value = datetime(2026, 4, 18, 12, 0, 0, tzinfo=timezone.utc)
            dt_mock.fromisoformat = datetime.fromisoformat
            snap = cqs.collect_snapshot("fake-token")
        row = snap["queues"]["amd_mi250_1"]
        assert row["running_by_workload"] == {"vllm": 0, "omni": 1}

    def test_jobs_without_queue_rule_skipped(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cqs, "OUTPUT", tmp_path / "out.jsonl", raising=False)
        monkeypatch.setattr(cqs, "fetch_cluster_queue_metrics", lambda token: {})
        monkeypatch.setattr(
            cqs,
            "fetch_active_cluster_jobs",
            lambda token, queue_ids_by_key=None: [
                _active_job("", "RUNNING", name="no-queue job"),
            ],
        )

        snap = cqs.collect_snapshot("fake-token")
        assert snap["total_running"] == 0

    def test_legacy_fallback_treats_assigned_as_running(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cqs, "OUTPUT", tmp_path / "out.jsonl", raising=False)
        monkeypatch.setattr(cqs, "fetch_cluster_queue_metrics", lambda token: {})
        monkeypatch.setattr(
            cqs,
            "fetch_active_cluster_jobs",
            lambda token, queue_ids_by_key=None: (_ for _ in ()).throw(RuntimeError("no graphql")),
        )
        monkeypatch.setattr(
            cqs,
            "bk_get",
            _FakeBk(
                running_builds=[
                    _build(
                        jobs=[
                            _job("amd_mi250_1", "scheduled", runnable_at="2026-04-18T11:55:00Z"),
                            _job("amd_mi250_1", "assigned", runnable_at="2026-04-18T11:58:00Z"),
                        ]
                    )
                ]
            ),
        )

        with patch("vllm.collect_queue_snapshot.datetime") as dt_mock:
            from datetime import datetime, timezone

            dt_mock.now.return_value = datetime(2026, 4, 18, 12, 0, 0, tzinfo=timezone.utc)
            dt_mock.fromisoformat = datetime.fromisoformat
            snap = cqs.collect_snapshot("fake-token")

        row = snap["queues"]["amd_mi250_1"]
        assert row["waiting"] == 1
        assert row["running"] == 1

    def test_output_schema_has_required_top_level_keys(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cqs, "OUTPUT", tmp_path / "out.jsonl", raising=False)
        monkeypatch.setattr(cqs, "fetch_cluster_queue_metrics", lambda token: {})
        monkeypatch.setattr(
            cqs, "fetch_active_cluster_jobs", lambda token, queue_ids_by_key=None: []
        )

        snap = cqs.collect_snapshot("fake-token")
        for key in (
            "ts",
            "queues",
            "total_waiting",
            "total_running",
            "total_zombie_waiting",
            "total_zombie_running",
            "sources",
        ):
            assert key in snap
        assert snap["ts"].endswith("Z")
        json.dumps(snap)


class TestJobsJsonSideEffect:
    """``collect_snapshot`` writes ``queue_jobs.json`` as a side effect."""

    def test_jobs_file_written_with_expected_schema(self, monkeypatch, tmp_path):
        out_path = tmp_path / "queue_timeseries.jsonl"
        monkeypatch.setattr(cqs, "OUTPUT", out_path, raising=False)
        monkeypatch.setattr(cqs, "fetch_cluster_queue_metrics", lambda token: {})
        monkeypatch.setattr(
            cqs,
            "fetch_active_cluster_jobs",
            lambda token, queue_ids_by_key=None: [
                _active_job("amd_mi250_1", "SCHEDULED", runnable_at="2026-04-18T11:55:00Z"),
                _active_job("amd_mi250_1", "RUNNING", started_at="2026-04-18T11:58:00Z"),
            ],
        )

        with patch("vllm.collect_queue_snapshot.datetime") as dt_mock:
            from datetime import datetime, timezone

            dt_mock.now.return_value = datetime(2026, 4, 18, 12, 0, 0, tzinfo=timezone.utc)
            dt_mock.fromisoformat = datetime.fromisoformat
            cqs.collect_snapshot("fake-token")
        jobs_file = out_path.parent / "queue_jobs.json"
        assert jobs_file.exists()
        data = json.loads(jobs_file.read_text())
        assert "ts" in data and "pending" in data and "running" in data
        assert len(data["pending"]) == 1
        pending = data["pending"][0]
        for field in ("name", "queue", "wait_min", "url", "workload", "branch", "commit"):
            assert field in pending
        assert pending["state"] == "scheduled"
        assert pending["analysis_excluded"] is False
        assert len(data["running"]) == 1
        running = data["running"][0]
        for field in ("name", "queue", "url", "run_min", "queue_wait_before_start_min"):
            assert field in running
        assert running["state"] == "running"
        assert running["analysis_excluded"] is False
