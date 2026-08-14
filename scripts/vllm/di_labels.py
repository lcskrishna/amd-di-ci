"""Parse Buildkite step labels from the amd-distributed-inference-ci pipeline
into grid coordinates.

Labels look like:

    DeepSeek-V3-PD-1P1D-TP8-MoRIIO-proxy
    DeepSeek-R1-MXFP4-PD-2P2D-TP8-MoRIIO-vllm-router
    DeepSeek-V3-PD-1P1D-EP8/DP8-WideEP-MoRIIO-proxy      (currently disabled)

Model names and router names both contain hyphens, so splitting on "-" does not
work. Anchor instead on the two literal separators that are always present: the
"-PD-" that follows the model, and the transport token that precedes the router.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

# Longest-first so a future "MoRIIOv2" cannot be shadowed by "MoRIIO".
TRANSPORTS = ("MoRIIO", "Mooncake", "NIXL")

_PD_SEP = "-PD-"
_SHAPE = re.compile(r"^(?P<shape>\d+P\d+D)(?:-(?P<rest>.*))?$")
_TP = re.compile(r"\bTP(\d+)\b")
_EP = re.compile(r"\bEP(\d+)\b")
_DP = re.compile(r"\bDP(\d+)\b")
_WIDE_EP = re.compile(r"WideEP", re.IGNORECASE)

UNCLASSIFIED = "unclassified"


@dataclass(frozen=True)
class Cell:
    raw: str
    ok: bool
    model: str = UNCLASSIFIED
    shape: str = UNCLASSIFIED  # "1P1D" / "2P2D"
    tp: int | None = None
    ep: int | None = None
    dp: int | None = None
    wide_ep: bool = False
    transport: str = UNCLASSIFIED
    router: str = UNCLASSIFIED

    @property
    def mode(self) -> str:
        """Parallelism descriptor used as part of the cell identity."""
        if not self.ok:
            return UNCLASSIFIED
        if self.wide_ep:
            parts = []
            if self.ep is not None:
                parts.append(f"EP{self.ep}")
            if self.dp is not None:
                parts.append(f"DP{self.dp}")
            return "/".join(parts) + "-WideEP" if parts else "WideEP"
        return f"TP{self.tp}" if self.tp is not None else UNCLASSIFIED

    @property
    def cell_id(self) -> str:
        if not self.ok:
            return f"{UNCLASSIFIED}:{self.raw}"
        return f"{self.model}|{self.shape}|{self.mode}|{self.transport}|{self.router}"

    def as_dict(self) -> dict:
        d = asdict(self)
        d["mode"] = self.mode
        d["cell_id"] = self.cell_id
        return d


def _unparsed(label: str) -> Cell:
    return Cell(raw=label, ok=False)


def parse_label(label: str) -> Cell:
    """Parse one step label. Never raises; unrecognised labels come back with
    ok=False so they land in a visible bucket instead of being dropped."""
    label = (label or "").strip()
    if _PD_SEP not in label:
        return _unparsed(label)

    model, _, tail = label.partition(_PD_SEP)
    if not model or not tail:
        return _unparsed(label)

    # Split the tail on the transport token: everything left of it describes the
    # parallel shape, everything right of it is the router.
    for transport in TRANSPORTS:
        needle = f"-{transport}-"
        idx = tail.find(needle)
        if idx != -1:
            break
    else:
        return _unparsed(label)

    shape_part = tail[:idx]
    router = tail[idx + len(needle):]
    if not shape_part or not router:
        return _unparsed(label)

    m = _SHAPE.match(shape_part)
    if not m:
        return _unparsed(label)
    shape = m.group("shape")
    rest = m.group("rest") or ""

    tp = int(_TP.search(rest).group(1)) if _TP.search(rest) else None
    ep = int(_EP.search(rest).group(1)) if _EP.search(rest) else None
    dp = int(_DP.search(rest).group(1)) if _DP.search(rest) else None
    wide_ep = bool(_WIDE_EP.search(rest))

    # A shape with no parallelism descriptor at all is not something we know how
    # to place on the grid.
    if tp is None and not wide_ep:
        return _unparsed(label)

    return Cell(
        raw=label,
        ok=True,
        model=model,
        shape=shape,
        tp=tp,
        ep=ep,
        dp=dp,
        wide_ep=wide_ep,
        transport=transport,
        router=router,
    )


if __name__ == "__main__":
    import json
    import sys

    for line in sys.stdin:
        line = line.strip()
        if line:
            print(json.dumps(parse_label(line).as_dict()))
