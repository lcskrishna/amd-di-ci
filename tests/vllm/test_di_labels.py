"""Label-parsing tests against the real DI pipeline definition.

The step label *is* the grid schema: rename a step upstream and its cell
silently moves, so these tests are the tripwire that should fail first.

LIVE_LABELS is the model x shape x router vocabulary from
vllm-project/vllm/.buildkite/amd-disagg/pipeline-disagg.yaml, exercised as a
full cross-product. It is deliberately wider than the upstream step list —
DeepSeek-V4-Pro runs only its three vllm-router steps — because the point is
to prove the parser handles every token combination, not to assert which
combinations exist. UPSTREAM_LABELS below carries the ones that really run.
"""

import pytest

from vllm.di_labels import UNCLASSIFIED, parse_label
from vllm.di_pipelines import MODELS, ROUTERS, SHAPES

LIVE_LABELS = [
    f"{m}-PD-{s}-TP8-MoRIIO-{r}" for s in SHAPES for m in MODELS for r in ROUTERS
]

WIDE_EP_LABEL = "DeepSeek-V3-PD-1P1D-EP8/DP8-WideEP-MoRIIO-proxy"


def test_grid_covers_the_full_model_shape_router_vocabulary():
    assert len(LIVE_LABELS) == len(MODELS) * len(SHAPES) * len(ROUTERS)


@pytest.mark.parametrize("label", LIVE_LABELS)
def test_live_label_fields(label):
    cell = parse_label(label)
    assert cell.ok, f"failed to parse {label}"
    assert cell.model in MODELS
    assert cell.shape in SHAPES
    assert cell.router in ROUTERS
    assert cell.transport == "MoRIIO"
    assert cell.tp == 8
    assert not cell.wide_ep
    assert cell.mode == "TP8"


def test_cell_ids_are_distinct():
    assert len({parse_label(l).cell_id for l in LIVE_LABELS}) == len(LIVE_LABELS)


def test_hyphenated_model_and_router_are_not_split():
    cell = parse_label("DeepSeek-R1-MXFP4-PD-2P2D-TP8-MoRIIO-vllm-router")
    assert cell.model == "DeepSeek-R1-MXFP4"
    assert cell.router == "vllm-router"
    assert cell.shape == "2P2D"


def test_dotted_model_name():
    assert parse_label("Kimi-K2.5-MXFP4-PD-1P1D-TP8-MoRIIO-proxy").model == "Kimi-K2.5-MXFP4"


def test_wide_ep_label_is_not_unclassified():
    cell = parse_label(WIDE_EP_LABEL)
    assert cell.ok, "wide-EP went live at build 47; its labels must parse"
    assert cell.model == "DeepSeek-V3"
    assert cell.shape == "1P1D"
    assert cell.wide_ep
    assert (cell.ep, cell.dp, cell.tp) == (8, 8, None)
    assert cell.mode == "EP8/DP8-WideEP"
    assert cell.router == "proxy"


def test_wide_ep_is_a_distinct_cell_from_its_tp_sibling():
    assert parse_label(WIDE_EP_LABEL).cell_id != parse_label(
        "DeepSeek-V3-PD-1P1D-TP8-MoRIIO-proxy"
    ).cell_id


# ---------------------------------------------------------------------------
# NIXL — vLLM PR #62007
# ---------------------------------------------------------------------------

# The six steps the PR appends, transcribed from its diff of
# pipeline-disagg.yaml, each with the (transport, router) it must parse to.
NIXL_LABELS = [
    ("DeepSeek-V3-PD-1P1D-TP8-NIXL-vllm-router", "NIXL", "vllm-router"),
    ("DeepSeek-V3-PD-1P1D-EP8/DP8-WideEP-NIXL-vllm-router", "NIXL", "vllm-router"),
    ("DeepSeek-V3-PD-2P2D-EP16/DP16-WideEP-NIXL-vllm-router", "NIXL", "vllm-router"),
    ("DeepSeek-V3-PD-1P1D-TP8-NIXL-PUSH-vllm-router", "NIXL-PUSH", "vllm-router"),
    ("DeepSeek-V3-PD-1P1D-EP8/DP8-WideEP-NIXL-PUSH-vllm-router", "NIXL-PUSH", "vllm-router"),
    ("DeepSeek-V3-PD-2P2D-EP16/DP16-WideEP-NIXL-PUSH-vllm-router", "NIXL-PUSH", "vllm-router"),
]


@pytest.mark.parametrize("label,transport,router", NIXL_LABELS)
def test_nixl_labels_parse_to_the_right_transport(label, transport, router):
    cell = parse_label(label)
    assert cell.ok, f"failed to parse {label}"
    assert cell.model == "DeepSeek-V3"
    assert cell.transport == transport
    assert cell.router == router


def test_nixl_push_is_not_shadowed_by_nixl():
    # The regression. "NIXL" is a prefix of "NIXL-PUSH", so a parser that takes
    # the first matching transport reads this label as transport "NIXL" with an
    # invented router "PUSH-vllm-router" — and returns ok=True while doing it,
    # so nothing downstream notices. Match order is the whole fix.
    cell = parse_label("DeepSeek-V3-PD-1P1D-TP8-NIXL-PUSH-vllm-router")
    assert cell.transport == "NIXL-PUSH"
    assert cell.router == "vllm-router"
    assert not cell.router.startswith("PUSH")


def test_nixl_and_moriio_twins_are_distinct_cells():
    # Identical in model, shape, mode and router; transport alone separates
    # them. Any structure keyed without transport folds the two into one.
    nixl = parse_label("DeepSeek-V3-PD-1P1D-TP8-NIXL-vllm-router")
    push = parse_label("DeepSeek-V3-PD-1P1D-TP8-NIXL-PUSH-vllm-router")
    moriio = parse_label("DeepSeek-V3-PD-1P1D-TP8-MoRIIO-vllm-router")
    assert len({nixl.cell_id, push.cell_id, moriio.cell_id}) == 3


def test_v4_pro_model_name_survives_its_hyphens():
    cell = parse_label("DeepSeek-V4-Pro-PD-1P1D-TP8-MoRIIO-vllm-router")
    assert cell.ok
    assert cell.model == "DeepSeek-V4-Pro"
    assert cell.router == "vllm-router"


@pytest.mark.parametrize("label", [
    "",
    "build docker image",
    "DeepSeek-V3-1P1D-TP8-MoRIIO-proxy",   # no -PD-
    "DeepSeek-V3-PD-1P1D-TP8-proxy",       # no known transport
    "DeepSeek-V3-PD-TP8-MoRIIO-proxy",     # no xPyD shape
    "DeepSeek-V3-PD-1P1D-MoRIIO-proxy",    # no parallelism descriptor
    "DeepSeek-V3-PD-1P1D-TP8-MoRIIO-",     # empty router
    None,
])
def test_unparseable_labels_are_flagged_not_dropped(label):
    cell = parse_label(label)
    assert not cell.ok
    assert cell.model == UNCLASSIFIED
    assert cell.cell_id.startswith(UNCLASSIFIED)
