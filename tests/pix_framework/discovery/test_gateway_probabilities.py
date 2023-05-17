from pathlib import Path

import pytest

from pix_framework.discovery.gateway_probabilities import (
    compute_gateway_probabilities,
    GatewayProbabilitiesDiscoveryMethod,
)
from pix_framework.input import read_csv_log
from pix_framework.io.bpm_graph import BPMNGraph
from pix_framework.log_ids import EventLogIDs

assets_dir = Path(__file__).parent.parent / "assets"


@pytest.mark.integration
@pytest.mark.parametrize(
    "args",
    [
        (
            assets_dir / "PurchasingExample.csv.gz",
            assets_dir / "PurchasingExample.bpmn",
        ),
    ],
)
def test_compute_gateway_probabilities(args: tuple):
    log_path = args[0]
    model_path = args[1]

    log_ids = EventLogIDs(
        case="case:concept:name",
        activity="concept:name",
        resource="org:resource",
        start_time="start_timestamp",
        end_time="time:timestamp",
    )

    log = read_csv_log(log_path, log_ids)

    bpmn_graph = BPMNGraph.from_bpmn_path(model_path)

    # Discover with equiprobable
    gateway_probabilities = compute_gateway_probabilities(
        log, log_ids, bpmn_graph, GatewayProbabilitiesDiscoveryMethod.EQUIPROBABLE
    )

    # Assert equiprobable probabilities
    assert gateway_probabilities is not None
    for gateway in gateway_probabilities:
        total_paths = len(gateway.outgoing_paths)
        for path in gateway.outgoing_paths:
            assert path.probability == 1.0 / total_paths

    # Discover
    gateway_probabilities = compute_gateway_probabilities(
        log, log_ids, bpmn_graph, GatewayProbabilitiesDiscoveryMethod.DISCOVERY
    )

    # Assert they add up to one
    assert gateway_probabilities is not None
    for gateway in gateway_probabilities:
        total_paths = len(gateway.outgoing_paths)
        sum_probs = 0.0
        for path in gateway.outgoing_paths:
            assert (
                path.probability != 1.0 / total_paths
            )  # Don't have to hold, but it should in Purchasing Example
            sum_probs += path.probability
        assert sum_probs == 1.0
