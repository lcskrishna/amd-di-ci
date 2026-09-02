"""Client-side pacing for Buildkite REST calls.

Buildkite enforces 200 requests/minute per organization *and* 50 per
authenticated user, and returns 429 when either trips. Every collector here
runs under one token, so 50 is the binding number. Retrying a 429 without
pacing makes a breach worse, so the bucket sits underneath the retry loops
rather than beside them.

Budget is split statically across workflows via ``BUILDKITE_MAX_RPM``: GitHub
Actions runs each workflow on its own ephemeral VM, so there is no shared state
for a cross-process limiter to coordinate through.

*Within* a workflow the opposite is true. A single job runs its steps
sequentially on one filesystem, and a workflow like hourly-master spends ~17 of
them on separate collector processes. An in-memory bucket would hand each one a
full budget on startup, so the per-workflow number would be enforced ~17 times
over rather than once. When ``RUNNER_TEMP`` is set the bucket therefore keeps
its state in a file under an advisory lock, and every process in the job draws
down the same tokens.
"""

import json
import os
import threading
import time
from contextlib import contextmanager
from typing import Callable, Optional

try:
    import fcntl
except ImportError:  # non-POSIX; fall back to per-process state
    fcntl = None

DEFAULT_MAX_RPM = 10

STATE_FILENAME = "buildkite_ratelimit.json"

_limiter: Optional["TokenBucket"] = None
_limiter_lock = threading.Lock()


class TokenBucket:
    """Rate limiter admitting at most ``rpm`` requests per rolling minute.

    ``clock`` and ``sleep`` are injected so tests can drive it with a fake
    clock instead of monkeypatching ``time``.

    With ``state_path`` the token count lives in that file instead of on the
    instance, so sibling processes share one budget. The default clock is then
    the wall clock rather than ``monotonic``, whose zero point is only
    guaranteed meaningful within a single process.
    """

    def __init__(
        self,
        rpm: int,
        clock: Optional[Callable[[], float]] = None,
        sleep: Callable[[float], None] = time.sleep,
        state_path: Optional[str] = None,
    ):
        self.rpm = max(1, int(rpm))
        self.state_path = state_path if fcntl is not None else None
        self._clock = clock or (time.time if self.state_path else time.monotonic)
        self._sleep = sleep
        self._rate = self.rpm / 60.0
        self._lock = threading.Lock()
        self._tokens = float(self.rpm)
        self._updated = self._clock()

    def _drain(self, n: int, tokens: float, updated: float) -> tuple:
        """Refill for elapsed time, then take ``n`` if they are there.

        Returns ``(tokens, updated, delay)`` with ``delay`` zero when the take
        succeeded. Kept free of I/O and locking so both the in-memory and the
        file-backed path can share it.
        """
        now = self._clock()
        elapsed = max(0.0, now - updated)
        tokens = min(float(self.rpm), tokens + elapsed * self._rate)
        if tokens >= n:
            return tokens - n, now, 0.0
        return tokens, now, (n - tokens) / self._rate

    @contextmanager
    def _locked_state(self):
        """Yield ``[tokens, updated]`` from the state file, exclusively held.

        Mutating the yielded list writes it back on exit. A missing or corrupt
        file is treated as a full bucket: the failure mode is a fresh budget,
        which is what an in-memory bucket would have done anyway.
        """
        # O_RDWR|O_CREAT rather than "a+": append mode forces every write to
        # the end of the file on POSIX, ignoring the seek before the rewrite.
        fd = os.open(self.state_path, os.O_RDWR | os.O_CREAT, 0o600)
        with os.fdopen(fd, "r+") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                fh.seek(0)
                try:
                    raw = json.loads(fh.read() or "{}")
                    state = [float(raw["tokens"]), float(raw["updated"])]
                except (ValueError, KeyError, TypeError):
                    state = [float(self.rpm), self._clock()]
                yield state
                fh.seek(0)
                fh.truncate()
                json.dump({"tokens": state[0], "updated": state[1]}, fh)
                fh.flush()
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

    def _attempt(self, n: int) -> float:
        """One non-blocking pass at taking ``n``. Returns seconds to wait."""
        with self._lock:
            if self.state_path:
                with self._locked_state() as state:
                    state[0], state[1], delay = self._drain(n, state[0], state[1])
                    return delay
            self._tokens, self._updated, delay = self._drain(n, self._tokens, self._updated)
            return delay

    def acquire(self, n: int = 1) -> float:
        """Block until ``n`` tokens are available. Returns seconds waited."""
        waited = 0.0
        while True:
            delay = self._attempt(n)
            if delay <= 0.0:
                return waited
            # Sleeping outside the lock matters: holding it would make N
            # threads convoy on the mutex instead of pacing on the bucket.
            self._sleep(delay)
            waited += delay

    def observe(self, headers) -> None:
        """Reconcile against what Buildkite says is actually left.

        Our own count only sees requests made through this process; the same
        token is in use elsewhere. When the server reports fewer remaining
        than we think we have, believe the server.
        """
        remaining = _int_header(headers, "RateLimit-User-Remaining")
        if remaining is None:
            remaining = _int_header(headers, "RateLimit-Remaining")
        if remaining is None:
            return
        with self._lock:
            if self.state_path:
                with self._locked_state() as state:
                    state[0], state[1], _ = self._drain(0, state[0], state[1])
                    state[0] = min(state[0], float(remaining))
                return
            self._tokens, self._updated, _ = self._drain(0, self._tokens, self._updated)
            self._tokens = min(self._tokens, float(remaining))


def _int_header(headers, name: str) -> Optional[int]:
    try:
        value = headers.get(name)
    except AttributeError:
        return None
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _state_path() -> Optional[str]:
    """Where sibling processes in this job rendezvous, if anywhere.

    ``BUILDKITE_RPM_STATE`` is the explicit override; otherwise the state goes
    in ``RUNNER_TEMP``, which GitHub Actions sets and shares across the steps
    of one job. Unset locally, so a developer run stays in memory.
    """
    explicit = os.getenv("BUILDKITE_RPM_STATE")
    if explicit:
        return explicit
    runner_temp = os.getenv("RUNNER_TEMP")
    if runner_temp and os.path.isdir(runner_temp):
        return os.path.join(runner_temp, STATE_FILENAME)
    return None


def limiter() -> TokenBucket:
    """Process-wide limiter, built on first use.

    ``BUILDKITE_MAX_RPM`` is read here rather than at import time so a
    workflow's env var is honoured regardless of import order.
    """
    global _limiter
    with _limiter_lock:
        if _limiter is None:
            try:
                rpm = int(os.getenv("BUILDKITE_MAX_RPM", "") or DEFAULT_MAX_RPM)
            except ValueError:
                rpm = DEFAULT_MAX_RPM
            _limiter = TokenBucket(rpm, state_path=_state_path())
        return _limiter


def acquire(n: int = 1) -> float:
    return limiter().acquire(n)


def observe(headers) -> None:
    limiter().observe(headers)


def reset_limiter() -> None:
    global _limiter
    with _limiter_lock:
        _limiter = None
