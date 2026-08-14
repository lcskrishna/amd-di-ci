"""Tests that the Buildkite token is ONLY used for vLLM pipelines.

These tests protect against accidental or malicious changes that could
use the BUILDKITE_TOKEN to access non-vLLM Buildkite organizations or
pipelines. If any of these tests fail, the token may be compromised.

CRITICAL: Do not weaken or skip these tests.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS = ROOT / "scripts"
WORKFLOWS = ROOT / ".github" / "workflows"


class TestBuildkiteTokenScope:
    """Ensure BUILDKITE_TOKEN is only used for the vLLM Buildkite org."""

    def test_pipelines_py_only_vllm_org(self):
        """scripts/vllm/pipelines.py must set BK_ORG = 'vllm'."""
        path = SCRIPTS / "vllm" / "pipelines.py"
        text = path.read_text()
        # Extract BK_ORG value
        m = re.search(r'BK_ORG\s*=\s*["\']([^"\']+)["\']', text)
        assert m, "pipelines.py must define BK_ORG"
        assert m.group(1) == "vllm", (
            f"BK_ORG must be 'vllm', got '{m.group(1)}'. "
            "Changing this would use the Buildkite token on a different org!"
        )

    def test_pipelines_py_only_known_slugs(self):
        """Pipeline slugs must be known vLLM pipelines."""
        path = SCRIPTS / "vllm" / "pipelines.py"
        text = path.read_text()
        # Extract all slug values
        slugs = re.findall(r'"slug"\s*:\s*"([^"]+)"', text)
        allowed_slugs = {"amd-ci", "ci"}
        for slug in slugs:
            assert slug in allowed_slugs, (
                f"Unknown pipeline slug '{slug}' in pipelines.py. "
                f"Only {allowed_slugs} are authorized. Adding new slugs "
                "may expose the Buildkite token to unauthorized pipelines."
            )

    def test_collect_analytics_only_vllm_org(self):
        """collect_analytics.py must only access the vLLM org.

        ``BK_ORG`` is defined centrally in ``scripts/vllm/constants.py``; the
        collector imports it from there. This test verifies both sides: the
        import site matches the constants source, and the constants source
        literally says ``"vllm"``.
        """
        path = SCRIPTS / "vllm" / "collect_analytics.py"
        text = path.read_text()
        assert re.search(r'from\s+vllm\.constants\s+import[^\n]*\bBK_ORG\b', text), (
            "collect_analytics.py must import BK_ORG from scripts/vllm/constants.py"
        )
        constants_path = SCRIPTS / "vllm" / "constants.py"
        constants_text = constants_path.read_text()
        m = re.search(r'BK_ORG\s*=\s*["\']([^"\']+)["\']', constants_text)
        assert m, "scripts/vllm/constants.py must define BK_ORG"
        assert m.group(1) == "vllm", (
            f"constants.py BK_ORG must be 'vllm', got '{m.group(1)}'"
        )

    def test_collect_analytics_only_known_pipelines(self):
        """collect_analytics.py pipeline dict must only have known slugs."""
        path = SCRIPTS / "vllm" / "collect_analytics.py"
        text = path.read_text()
        # Find PIPELINES dict keys
        m = re.search(r'PIPELINES\s*=\s*\{([^}]+)\}', text)
        assert m, "collect_analytics.py must define PIPELINES"
        slugs = re.findall(r'"([^"]+)"\s*:', m.group(1))
        allowed_slugs = {"amd-ci", "ci"}
        for slug in slugs:
            assert slug in allowed_slugs, (
                f"Unknown pipeline slug '{slug}' in collect_analytics.py"
            )

    def test_di_pipelines_py_only_vllm_org(self):
        """scripts/vllm/di_pipelines.py must set BK_ORG = 'vllm'."""
        path = SCRIPTS / "vllm" / "di_pipelines.py"
        text = path.read_text()
        m = re.search(r'BK_ORG\s*=\s*["\']([^"\']+)["\']', text)
        assert m, "di_pipelines.py must define BK_ORG"
        assert m.group(1) == "vllm", (
            f"BK_ORG must be 'vllm', got '{m.group(1)}'. "
            "Changing this would use the Buildkite token on a different org!"
        )

    def test_di_pipelines_py_only_known_slugs(self):
        """The DI collector may only reach its own pipeline.

        Mirrors ``test_pipelines_py_only_known_slugs``. The allowlist is the
        point: a second pipeline registry must not become an unguarded path
        for the shared token to reach an arbitrary slug.
        """
        path = SCRIPTS / "vllm" / "di_pipelines.py"
        text = path.read_text()
        slugs = re.findall(r'"slug"\s*:\s*"([^"]+)"', text)
        allowed_slugs = {"amd-distributed-inference-ci"}
        assert slugs, "di_pipelines.py must define at least one pipeline slug"
        for slug in slugs:
            assert slug in allowed_slugs, (
                f"Unknown pipeline slug '{slug}' in di_pipelines.py. "
                f"Only {allowed_slugs} are authorized."
            )

    def test_no_buildkite_token_in_non_vllm_scripts(self):
        """Only scripts/vllm/ and the collect_*.py entry points may reference it."""
        # Both are at root but import from vllm.* — they are vLLM-specific.
        allowed_root_scripts = {"collect_ci.py", "collect_di_ci.py"}
        for py_file in SCRIPTS.rglob("*.py"):
            rel = py_file.relative_to(SCRIPTS)
            if str(rel).startswith("vllm") or rel.name in allowed_root_scripts:
                continue
            text = py_file.read_text()
            assert "BUILDKITE_TOKEN" not in text and "BK_TOKEN" not in text, (
                f"{rel} references Buildkite token but is not a vLLM script. "
                "Only vLLM scripts should use the Buildkite token."
            )

    def test_no_buildkite_token_in_other_framework_workflows(self):
        """No workflow should pass BUILDKITE_TOKEN to non-vLLM scripts."""
        for wf in WORKFLOWS.glob("*.yml"):
            text = wf.read_text()
            if "BUILDKITE_TOKEN" not in text:
                continue
            # If it uses the token, it must only call vllm scripts
            lines = text.split("\n")
            for i, line in enumerate(lines):
                if "BUILDKITE_TOKEN" in line and "secrets." in line:
                    # Find the surrounding step's run command
                    # Look backward for 'run:' to find what script uses this token
                    for j in range(i, max(0, i - 10), -1):
                        if "run:" in lines[j] or "python" in lines[j]:
                            script_line = lines[j].strip()
                            if "python" in script_line:
                                assert "vllm" in script_line or "collect_ci" in script_line or "collect_queue" in script_line, (
                                    f"{wf.name} line {j+1}: BUILDKITE_TOKEN used by non-vLLM script: {script_line}"
                                )
                            break

    def test_buildkite_api_base_is_standard(self):
        """The Buildkite API base URL must be the standard public API."""
        for py_file in (SCRIPTS / "vllm").rglob("*.py"):
            text = py_file.read_text()
            for m in re.finditer(r'(BK_API|BK_API_BASE)\s*=\s*["\']([^"\']+)["\']', text):
                url = m.group(2)
                assert url == "https://api.buildkite.com/v2", (
                    f"{py_file.name}: Buildkite API URL is '{url}', "
                    "expected 'https://api.buildkite.com/v2'. "
                    "Redirecting API calls could expose the token."
                )

    def test_skip_patterns_dont_match_test_groups(self):
        """SKIP_JOB_PATTERNS must not accidentally match real test group names
        from the parity report."""
        import sys
        sys.path.insert(0, str(ROOT / "scripts"))
        from vllm.pipelines import SKIP_JOB_PATTERNS
        parity_path = ROOT / "data" / "vllm" / "ci" / "parity_report.json"
        if not parity_path.exists():
            pytest.skip("no parity data")
        import json
        groups = json.loads(parity_path.read_text()).get("job_groups", [])
        group_names = [g["name"] for g in groups if g.get("amd")]
        for group in group_names:
            lower = group.lower()
            for pattern in SKIP_JOB_PATTERNS:
                assert pattern not in lower, (
                    f"SKIP_JOB_PATTERNS '{pattern}' matches test group '{group}'. "
                    "This causes the group to be silently dropped from collection. "
                    "Make the skip pattern more specific."
                )

    def test_no_token_in_committed_files(self):
        """No Buildkite token value should appear in any committed file."""
        token_pattern = re.compile(r'bkua_[a-f0-9]{40}')
        for ext in ["*.py", "*.js", "*.yml", "*.yaml", "*.json", "*.md"]:
            for f in ROOT.rglob(ext):
                if ".git" in str(f) or "node_modules" in str(f):
                    continue
                try:
                    text = f.read_text()
                    match = token_pattern.search(text)
                    assert not match, (
                        f"CRITICAL: Buildkite token found in {f.relative_to(ROOT)}! "
                        "Tokens must never be committed to code."
                    )
                except UnicodeDecodeError:
                    continue
