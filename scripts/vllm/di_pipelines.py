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
)
SHAPES = ("1P1D", "2P2D")
ROUTERS = ("proxy", "vllm-router")
TRANSPORT = "MoRIIO"

# One grid per parallelism mode, rendered top to bottom in this order.
#
# Mode is a grid selector, not a global constant: a wide-EP cell shares model,
# shape and router with a TP8 cell and differs only by mode, so a single matrix
# keyed on (model, shape, router) silently collapses the two into one.
#
# Both grids carry the full shape x router column set even though wide-EP only
# runs 1P1D today. The empty 2P2D half *is* the coverage gap; a narrower table
# would hide it.
GRIDS = (
    {
        "key": "tp8",
        "title": "Tensor parallel — TP8",
        "mode": "TP8",
        "shapes": SHAPES,
        "routers": ROUTERS,
        "note": "The original matrix. 1P1D = 2 nodes, 2P2D = 4 nodes.",
    },
    {
        "key": "wide-ep",
        "title": "Wide expert parallel — EP8/DP8",
        "mode": "EP8/DP8-WideEP",
        "shapes": SHAPES,
        "routers": ROUTERS,
        "note": "Commented out in pipeline-disagg.yaml until build 47 "
                "(2026-08-21). Only 1P1D is defined upstream so far.",
    },
)
