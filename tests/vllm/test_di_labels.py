"""Label-parsing tests against the real DI pipeline definition.

LIVE_LABELS is transcribed from
vllm-project/vllm/.buildkite/amd-disagg/pipeline-disagg.yaml. The step label
*is* the grid schema: rename a step upstream and its cell silently moves, so
these tests are the tripwire that should fail first.
"""

import pytest

from vllm.di_labels import UNCLASSIFIED, parse_label
from vllm.di_pipelines import MODELS, ROUTERS, SHAPES

LIVE_LABELS = [
    f"{m}-PD-{s}-TP8-MoRIIO-{r}" for s in SHAPES for m in MODELS for r in ROUTERS
]

WIDE_EP_LABEL = "DeepSeek-V3-PD-1P1D-EP8/DP8-WideEP-MoRIIO-proxy"


def test_grid_is_twenty_cells():
    assert len(LIVE_LABELS) == 20


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
    assert len({parse_label(l).cell_id for l in LIVE_LABELS}) == 20


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
