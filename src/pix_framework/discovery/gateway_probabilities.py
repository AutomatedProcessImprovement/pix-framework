from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Union

import pandas as pd

from pix_framework.io.bpm_graph import BPMNGraph
from pix_framework.log_ids import EventLogIDs


@dataclass
class PathProbability:
    path_id: str
    probability: float

    @staticmethod
    def from_dict(path_probabilities: dict) -> "PathProbability":
        return PathProbability(
            path_id=path_probabilities["path_id"],
            probability=path_probabilities["value"],
        )

    def to_dict(self):
        """Dictionary compatible with Prosimos."""
        return {"path_id": self.path_id, "value": self.probability}


@dataclass
class GatewayProbabilities:
    """Gateway branching probabilities for Prosimos."""

    gateway_id: str
    outgoing_paths: List[PathProbability]

    @staticmethod
    def from_dict(gateway_probabilities: dict) -> "GatewayProbabilities":
        return GatewayProbabilities(
            gateway_id=gateway_probabilities["gateway_id"],
            outgoing_paths=[
                PathProbability.from_dict(path_probability)
                for path_probability in gateway_probabilities["probabilities"]
            ],
        )

    def to_dict(self):
        """Dictionary compatible with Prosimos."""
        return {
            "gateway_id": self.gateway_id,
            "probabilities": [p.to_dict() for p in self.outgoing_paths],
        }


class GatewayProbabilitiesMethod(str, Enum):
    DISCOVERY = "discovery"
    EQUIPROBABLE = "equiprobable"

    @classmethod
    def from_str(
        cls, value: Union[str, List[str]]
    ) -> Union["GatewayProbabilitiesMethod", List["GatewayProbabilitiesMethod"]]:
        if isinstance(value, str):
            return GatewayProbabilitiesMethod._from_str(value)
        elif isinstance(value, list):
            return [GatewayProbabilitiesMethod._from_str(v) for v in value]

    @classmethod
    def _from_str(cls, value: str) -> "GatewayProbabilitiesMethod":
        if value.lower() == "discovery":
            return cls.DISCOVERY
        elif value.lower() == "equiprobable":
            return cls.EQUIPROBABLE
        else:
            raise ValueError(f"Unknown value {value}")

    def __str__(self):
        if self == GatewayProbabilitiesMethod.DISCOVERY:
            return "discovery"
        elif self == GatewayProbabilitiesMethod.EQUIPROBABLE:
            return "equiprobable"
        return f"Unknown GateManagement {str(self)}"


def compute_gateway_probabilities(
    event_log: pd.DataFrame,
    log_ids: EventLogIDs,
    bpmn_path: Path,
    discovery_method: GatewayProbabilitiesMethod,
) -> List[GatewayProbabilities]:
    # Read BPMN model
    bpmn_graph = BPMNGraph.from_bpmn_path(bpmn_path)
    # Discover gateway probabilities depending on the type
    if discovery_method is GatewayProbabilitiesMethod.EQUIPROBABLE:
        gateway_probabilities = bpmn_graph.compute_equiprobable_gateway_probabilities()
    elif discovery_method is GatewayProbabilitiesMethod.DISCOVERY:
        # Discover the frequency of each gateway branch with replay
        arcs_frequencies = dict()
        for _, events in event_log.groupby(log_ids.case):
            # Transform to list of activity labels
            trace = events.sort_values([log_ids.start_time, log_ids.end_time])[
                log_ids.activity
            ].tolist()
            # Replay updating arc frequencies
            bpmn_graph.replay_trace(trace, arcs_frequencies)
        # Obtain gateway path probabilities based on arc frequencies
        gateway_probabilities = bpmn_graph.discover_gateway_probabilities(
            arcs_frequencies
        )
    else:
        # Error, wrong method
        raise ValueError(
            f"Only GatewayProbabilitiesMethod.DISCOVERY and GatewayProbabilitiesMethod.EQUIPROBABLE are supported, "
            f"got {discovery_method} ({type(discovery_method)})."
        )

    return _translate_to_prosimos_format(gateway_probabilities)


def _translate_to_prosimos_format(gateway_probabilities) -> List[GatewayProbabilities]:
    # Transform to prosimos list format
    prosimos_gateway_probabilities = [
        GatewayProbabilities(
            gateway_id,
            [
                PathProbability(
                    outgoing_node, gateway_probabilities[gateway_id][outgoing_node]
                )
                for outgoing_node in gateway_probabilities[gateway_id]
            ],
        )
        for gateway_id in gateway_probabilities
    ]
    # Return prosimos format
    return prosimos_gateway_probabilities
