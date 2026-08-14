#!/usr/bin/env python3
"""Assemble the standalone DI grid site.

Separate from build_site.py, which publishes the full operations dashboard and
therefore requires the whole data/vllm/ci/** corpus to be present and fresh.
This one needs a Buildkite token and nothing else, so it can be deployed by a
workflow that has no GitHub API credentials.

The file list is an allowlist rather than a copy-tree-minus-exclusions. The
collector's inputs (jobs.jsonl, the raw per-build logs under test_results/) are
not fit for publication, and an allowlist cannot forget to exclude something
that gets added later.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
DATA = ROOT / "data"

# (source, destination-relative-to-site). docs/di.html becomes the site root so
# the Pages URL lands directly on the grid.
ASSETS = [
    (DOCS / "di.html", "index.html"),
    (DOCS / "assets/css/dashboard.css", "assets/css/dashboard.css"),
    (DOCS / "assets/js/utils.js", "assets/js/utils.js"),
    (DOCS / "assets/js/ci-di.js", "assets/js/ci-di.js"),
    (DATA / "vllm/di/grid.json", "data/vllm/di/grid.json"),
]

CACHE_BUST_RE = re.compile(r"\?v=\d+")


def build(out: Path, cache_bust: bool = True) -> None:
    if out.exists():
        shutil.rmtree(out)

    for src, rel in ASSETS:
        if not src.exists():
            raise SystemExit(f"missing required input: {src.relative_to(ROOT)}")
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)

    grid = json.loads((out / "data/vllm/di/grid.json").read_text())
    if not grid.get("cells"):
        raise SystemExit("grid.json has no cells; refusing to publish an empty site")

    if cache_bust:
        index = out / "index.html"
        index.write_text(CACHE_BUST_RE.sub(f"?v={int(time.time())}", index.read_text()))

    # A .nojekyll is required or Pages will not serve paths that Jekyll would
    # treat as special. Nothing here starts with an underscore today, but the
    # cost of being wrong later is a silent 404.
    (out / ".nojekyll").write_text("")

    print(f"built {out} — {len(grid['cells'])} cells, "
          f"{len(grid.get('build_rollup', []))} builds")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default="_site_di", help="output directory")
    ap.add_argument("--no-cache-bust", action="store_true")
    args = ap.parse_args()
    build(ROOT / args.output, cache_bust=not args.no_cache_bust)


if __name__ == "__main__":
    main()
