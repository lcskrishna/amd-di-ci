"""Ensure scripts/vllm is importable as 'vllm.ci.*' from tests/vllm/.

Without this, Python resolves 'vllm' to tests/vllm/ (this package)
instead of scripts/vllm/ where the CI modules live.
"""
import sys
from pathlib import Path

import pytest

# Insert scripts/ at the front of sys.path so 'from vllm.ci.models import ...'
# resolves to scripts/vllm/ci/models.py, not tests/vllm/.
_scripts_dir = str(Path(__file__).resolve().parent.parent.parent / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)


@pytest.fixture(autouse=True)
def _unthrottled_buildkite():
    """Tests mock the network, so the Buildkite pacer has nothing to protect.

    Left at its real budget it would sleep between mocked requests and add
    minutes to the suite. ``test_ratelimit`` builds its own buckets and is
    unaffected.
    """
    from vllm.ci import ratelimit

    ratelimit.reset_limiter()
    ratelimit._limiter = ratelimit.TokenBucket(1_000_000)
    yield
    ratelimit.reset_limiter()
