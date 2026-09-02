"""Token bucket pacing for Buildkite calls.

The bucket is driven by an injected clock, so these run instantly and assert
on the delay it *asked* for rather than on wall time.
"""

from __future__ import annotations

import threading

import pytest

from vllm.ci import ratelimit


class FakeClock:
    """A clock that only advances when the bucket sleeps."""

    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture(autouse=True)
def _fresh_singleton():
    ratelimit.reset_limiter()
    yield
    ratelimit.reset_limiter()


def bucket(rpm=60):
    clock = FakeClock()
    return ratelimit.TokenBucket(rpm, clock=clock.time, sleep=clock.sleep), clock


def test_a_full_bucket_admits_a_burst_without_waiting():
    # The point of a bucket rather than a fixed delay: an idle collector can
    # spend its whole minute at once instead of trickling.
    b, clock = bucket(rpm=60)
    for _ in range(60):
        assert b.acquire() == 0.0
    assert clock.sleeps == []


def test_the_request_past_the_budget_waits_for_one_token():
    b, clock = bucket(rpm=60)
    for _ in range(60):
        b.acquire()
    assert b.acquire() == pytest.approx(1.0)
    assert clock.sleeps == [pytest.approx(1.0)]


def test_sustained_load_settles_at_the_configured_rate():
    # 120 requests at 60 rpm: the first 60 are free, the rest cost a minute.
    b, clock = bucket(rpm=60)
    for _ in range(120):
        b.acquire()
    assert clock.now == pytest.approx(60.0)


def test_tokens_come_back_as_time_passes():
    b, clock = bucket(rpm=60)
    for _ in range(60):
        b.acquire()
    clock.now += 10.0
    for _ in range(10):
        assert b.acquire() == 0.0


def test_the_bucket_does_not_bank_more_than_a_minute_of_idleness():
    # Otherwise an overnight-idle process would open with a burst far past the
    # limit the moment it woke up.
    b, clock = bucket(rpm=60)
    clock.now += 3600.0
    for _ in range(60):
        assert b.acquire() == 0.0
    assert b.acquire() > 0.0


def test_the_server_headline_overrides_our_own_count():
    # The same token is used by other workflows, so our local count is only a
    # lower bound on what has been spent.
    b, clock = bucket(rpm=60)
    b.observe({"RateLimit-User-Remaining": "0"})
    assert b.acquire() > 0.0


def test_a_generous_server_headline_does_not_raise_our_budget():
    b, _ = bucket(rpm=60)
    for _ in range(60):
        b.acquire()
    b.observe({"RateLimit-User-Remaining": "500"})
    assert b.acquire() > 0.0


def test_headers_without_rate_limit_information_are_ignored():
    b, clock = bucket(rpm=60)
    b.observe({})
    b.observe({"RateLimit-User-Remaining": "not-a-number"})
    assert b.acquire() == 0.0


def test_the_org_header_is_used_when_the_user_header_is_absent():
    b, _ = bucket(rpm=60)
    b.observe({"RateLimit-Remaining": "0"})
    assert b.acquire() > 0.0


def test_concurrent_callers_share_one_budget():
    # Threads must pace on the bucket, not convoy on its mutex: if acquire()
    # slept while holding the lock, total admissions would still be right but
    # the wait would serialise. Assert on the count, which is what protects us.
    b, clock = bucket(rpm=60)
    admitted = []
    lock = threading.Lock()

    def worker():
        for _ in range(10):
            if b.acquire() == 0.0:
                with lock:
                    admitted.append(1)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(admitted) <= 60


# ---------------------------------------------------------------- shared state
#
# A workflow job runs its steps as separate processes on one filesystem.
# hourly-master spends ~17 of them on collectors, so an in-memory bucket would
# hand out its whole budget ~17 times per run. These pin the file-backed path
# that stops that.


def shared_bucket(path, clock, rpm=60):
    return ratelimit.TokenBucket(
        rpm, clock=clock.time, sleep=clock.sleep, state_path=str(path)
    )


def test_two_processes_on_one_state_file_share_a_single_budget(tmp_path):
    # The whole point of the file: without it each bucket below would start
    # full and admit 60 apiece.
    clock = FakeClock()
    path = tmp_path / "bk.json"
    first = shared_bucket(path, clock)
    second = shared_bucket(path, clock)

    for _ in range(60):
        first.acquire()
    assert second.acquire() > 0.0


def test_a_later_process_inherits_what_an_earlier_one_spent(tmp_path):
    clock = FakeClock()
    path = tmp_path / "bk.json"
    first = shared_bucket(path, clock)
    for _ in range(50):
        first.acquire()
    del first

    # Stands in for the next collector step starting up.
    second = shared_bucket(path, clock)
    for _ in range(10):
        assert second.acquire() == 0.0
    assert second.acquire() > 0.0


def test_tokens_still_refill_across_processes(tmp_path):
    clock = FakeClock()
    path = tmp_path / "bk.json"
    first = shared_bucket(path, clock)
    for _ in range(60):
        first.acquire()
    clock.now += 30.0
    assert shared_bucket(path, clock).acquire() == 0.0


def test_a_corrupt_state_file_falls_back_to_a_full_bucket(tmp_path):
    # Degrading to a fresh budget is what an in-memory bucket would have done,
    # so a garbled file must not take the collectors down with it.
    path = tmp_path / "bk.json"
    path.write_text("{not json")
    clock = FakeClock()
    assert shared_bucket(path, clock).acquire() == 0.0


def test_an_empty_state_file_is_treated_as_untouched(tmp_path):
    path = tmp_path / "bk.json"
    path.write_text("")
    clock = FakeClock()
    assert shared_bucket(path, clock).acquire() == 0.0


def test_the_server_headline_is_visible_to_sibling_processes(tmp_path):
    clock = FakeClock()
    path = tmp_path / "bk.json"
    shared_bucket(path, clock).observe({"RateLimit-User-Remaining": "0"})
    assert shared_bucket(path, clock).acquire() > 0.0


def test_state_is_written_where_the_runner_shares_it(monkeypatch, tmp_path):
    monkeypatch.delenv("BUILDKITE_RPM_STATE", raising=False)
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    assert ratelimit._state_path() == str(tmp_path / ratelimit.STATE_FILENAME)


def test_an_explicit_state_path_wins_over_the_runner_default(monkeypatch, tmp_path):
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    monkeypatch.setenv("BUILDKITE_RPM_STATE", "/tmp/chosen.json")
    assert ratelimit._state_path() == "/tmp/chosen.json"


def test_a_developer_run_keeps_its_state_in_memory(monkeypatch):
    # No RUNNER_TEMP off-runner, and a local run is one process anyway.
    monkeypatch.delenv("BUILDKITE_RPM_STATE", raising=False)
    monkeypatch.delenv("RUNNER_TEMP", raising=False)
    assert ratelimit._state_path() is None


def test_a_runner_temp_that_does_not_exist_is_not_used(monkeypatch):
    monkeypatch.delenv("BUILDKITE_RPM_STATE", raising=False)
    monkeypatch.setenv("RUNNER_TEMP", "/nonexistent/runner/temp")
    assert ratelimit._state_path() is None


def test_the_limiter_reads_its_budget_from_the_environment(monkeypatch):
    monkeypatch.setenv("BUILDKITE_MAX_RPM", "15")
    assert ratelimit.limiter().rpm == 15


def test_an_unset_or_junk_budget_falls_back_to_the_default(monkeypatch):
    monkeypatch.delenv("BUILDKITE_MAX_RPM", raising=False)
    assert ratelimit.limiter().rpm == ratelimit.DEFAULT_MAX_RPM
    ratelimit.reset_limiter()
    monkeypatch.setenv("BUILDKITE_MAX_RPM", "banana")
    assert ratelimit.limiter().rpm == ratelimit.DEFAULT_MAX_RPM


def test_the_limiter_is_a_singleton():
    assert ratelimit.limiter() is ratelimit.limiter()
