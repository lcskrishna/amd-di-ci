"""Buildkite pipeline definition for the AMD distributed-inference (DI) CI.

Separate from ``scripts/vllm/pipelines.py`` so the DI collector can be
configured, scheduled, and reasoned about independently of the nightly
AMD/upstream pipelines. Same Buildkite org.

The pipeline runs the disaggregated prefill/decode SLURM tests defined in
``vllm-project/vllm/.buildkite/amd-disagg/pipeline-disagg.yaml``: a grid of
model x shape x router, every step on agent queue ``amd_mi350_ainic``.
"""

BK_ORG = "vllm"

DI_KEY = "di"

DI_PIPELINES = {
    DI_KEY: {
        "slug": "amd-distributed-inference-ci",
        # This is a dedicated pipeline — every build in it is a DI run — so
        # unlike the nightly pipelines there is nothing to filter on. The
        # nightly collector matches a build's commit message against a
        # schedule name; doing that here would silently drop manually
        # triggered builds, which is most of this pipeline's traffic.
        "name_pattern": r".*",
        # None means "do not filter by branch". The pipeline is triggered
        # against whatever branch is under test, not just main.
        "branch": None,
        "display_name": "AMD Distributed Inference",
    },
}

# Hardware is uniform across the pipeline: every step requests this queue.
# Job labels carry no hardware tag, so the analyzer cannot infer it from the
# name and would otherwise fall back to its upstream default of "h100".
DI_QUEUE = "amd_mi350_ainic"
DI_HARDWARE = "mi350"

# Non-test infrastructure steps, matched as substrings of lowercased labels.
# None of the current 30 DI labels contain any of these.
SKIP_JOB_PATTERNS = (
    "bootstrap",
    ":docker:",
    "build image",
    "pipeline upload",
    # The real label is ":pipeline: Upload AMD distributed inference" — the
    # emoji sits between the words, so "pipeline upload" never matches it.
    ":pipeline:",
)

# The 120-minute per-step wall clock from pipeline-disagg.yaml. Rendered as
# the reference line on the runtime trend.
STEP_TIMEOUT_MINS = 120

# Grid axes as defined today. Used to render empty cells: a cell that never
# ran is evidence, and it disappears if the grid is built only from observed
# labels.
MODELS = (
    "DeepSeek-V3",
    "DeepSeek-R1-MXFP4",
    "Kimi-K2.5-MXFP4",
    "Kimi-K2.6-MXFP4",
    "MiniMax-M3-MXFP8",
    "GLM-5.2-FP8",
    # Only the three vllm-router steps exist upstream, so the proxy columns
    # stay empty. That is the coverage gap, not a bug in the enumeration.
    "DeepSeek-V4-Pro",
)
SHAPES = ("1P1D", "2P2D")
ROUTERS = ("proxy", "vllm-router")

# One grid per parallelism family, rendered top to bottom in this order.
#
# Mode is a grid selector, not a global constant: a wide-EP cell shares model,
# shape and router with a TP8 cell and differs only by mode, so a single matrix
# keyed on (model, shape, router) silently collapses the two into one.
#
# Mode also varies *within* the wide-EP grid — expert parallelism scales with
# node count, so 2P2D is EP16/DP16 where 1P1D is EP8/DP8 — which is why each
# shape carries its own mode rather than inheriting one from the grid.
#
# ``mode`` is the label token and the cell key; ``mode_label`` is the shorter
# form the column headers carry. Declared rather than derived from ``mode``,
# because trimming "-WideEP" off the end is the kind of rule that quietly
# mangles the next mode someone adds.
#
# Grids carry the full shape column set even where only some shapes run today.
# The empty half *is* the coverage gap; a narrower table would hide it. The
# same reasoning admits DeepSeek-V4-Pro's empty proxy columns.
#
# ``models`` is optional and defaults to the global MODELS. It exists for
# transports under a narrower rollout than the fleet — NIXL is DeepSeek-V3 only
# — where a full cross-product would be six rows of noise rather than six gaps.
TP8_SHAPES = tuple(
    {"shape": s, "mode": "TP8", "mode_label": "TP8"} for s in SHAPES
)
WIDE_EP_SHAPES = (
    {"shape": "1P1D", "mode": "EP8/DP8-WideEP", "mode_label": "EP8/DP8"},
    # Not defined upstream yet, so this mode string is a prediction. If the step
    # lands under a different name its records will surface in the
    # unexpected-steps panel while this column still reads "never run" — which
    # is the signal to come and fix this line.
    {"shape": "2P2D", "mode": "EP16/DP16-WideEP", "mode_label": "EP16/DP16"},
)

# NIXL runs DeepSeek-V3 only, and only behind vllm-router: the ``proxy`` option
# is the MoRI proxy, which has no NIXL equivalent. So the narrower column set
# here is "not applicable", not an untested gap — unlike the empty shapes
# above, which are.
_NIXL_MODELS = ("DeepSeek-V3",)
_NIXL_ROUTERS = ("vllm-router",)
_NIXL_NOTE = (
    "Pending vLLM PR #62007 (open). Until it merges these cells read "
    "'never run' because the steps do not exist yet, not because they fail."
)

GRIDS = (
    {
        "key": "tp8",
        "transport": "MoRIIO",
        "title": "PD Disagg with TP8 · MoRIIO",
        "shapes": TP8_SHAPES,
        "routers": ROUTERS,
        "note": "The original matrix. 1P1D = 2 nodes, 2P2D = 4 nodes.",
    },
    {
        "key": "wide-ep",
        "transport": "MoRIIO",
        "title": "PD Disagg with Wide EP · MoRIIO",
        "shapes": WIDE_EP_SHAPES,
        "routers": ROUTERS,
        "note": "Live upstream from build 47 (2026-08-21). Expert parallelism "
                "scales with node count, so 2P2D is EP16/DP16; only 1P1D is "
                "defined upstream so far.",
    },
)

# Not rendered yet. vLLM PR #62007 is still open, so these six steps do not
# exist upstream and four grids of "never run" would be noise rather than a
# coverage gap. Append to GRIDS to turn them on once the PR merges:
#
#     GRIDS = GRIDS + NIXL_GRIDS
#
# Until then a NIXL record, if one appears early, lands in the unexpected-steps
# panel with its transport correctly parsed — captured and visible, not folded
# into a MoRIIO cell.
NIXL_GRIDS = (
    {
        "key": "tp8-nixl",
        "transport": "NIXL",
        "title": "PD Disagg with TP8 · NIXL",
        "shapes": TP8_SHAPES,
        "routers": _NIXL_ROUTERS,
        "models": _NIXL_MODELS,
        "note": _NIXL_NOTE + " 2P2D has no NIXL step proposed; that column is a "
                             "real gap.",
    },
    {
        "key": "wide-ep-nixl",
        "transport": "NIXL",
        "title": "PD Disagg with Wide EP · NIXL",
        "shapes": WIDE_EP_SHAPES,
        "routers": _NIXL_ROUTERS,
        "models": _NIXL_MODELS,
        "note": _NIXL_NOTE,
    },
    {
        "key": "tp8-nixl-push",
        "transport": "NIXL-PUSH",
        "title": "PD Disagg with TP8 · NIXL-PUSH",
        "shapes": TP8_SHAPES,
        "routers": _NIXL_ROUTERS,
        "models": _NIXL_MODELS,
        "note": _NIXL_NOTE + " 2P2D has no NIXL-PUSH step proposed; that column "
                             "is a real gap.",
    },
    {
        "key": "wide-ep-nixl-push",
        "transport": "NIXL-PUSH",
        "title": "PD Disagg with Wide EP · NIXL-PUSH",
        "shapes": WIDE_EP_SHAPES,
        "routers": _NIXL_ROUTERS,
        "models": _NIXL_MODELS,
        "note": _NIXL_NOTE,
    },
)

# Transport is a grid property rather than a global constant because a NIXL cell
# shares model, shape, mode and router with its MoRIIO twin and differs only by
# transport — the same collapse the per-grid ``mode`` exists to prevent, one axis
# over.
#
# The dashboard list is derived from GRIDS, in grid order and de-duplicated, so
# enabling NIXL_GRIDS cannot leave the "KV transport" row advertising a transport
# no grid renders. Note this is a narrower set than ``di_labels.TRANSPORTS``,
# which is the parser's vocabulary and must keep recognising NIXL whether or not
# a grid shows it.
TRANSPORTS = tuple(dict.fromkeys(g["transport"] for g in GRIDS))

# Deprecated: the pre-schema-3 scalar. Emitted as ``axes.transport`` for one
# release so a consumer reading the old field does not blank mid-deploy.
TRANSPORT = TRANSPORTS[0]
