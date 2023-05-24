from pathlib import Path

import pytest
from pix_framework.io.bpm_graph import BPMNGraph

assets_dir = Path(__file__).parent.parent / "assets"


@pytest.mark.smoke
@pytest.mark.parametrize("model_path", [(assets_dir / "PurchasingExample.bpmn")])
def test_from_bpmn_path(model_path: Path):
    graph = BPMNGraph.from_bpmn_path(model_path)
    assert graph is not None
    assert graph.starting_event is not None
    assert len(graph.flow_arcs) > 0
