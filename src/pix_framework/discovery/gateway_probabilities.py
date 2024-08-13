from dataclasses import dataclass, field
from enum import Enum
from typing import List, Union

import pandas as pd

from pix_framework.io.bpm_graph import BPMNGraph
from pix_framework.io.event_log import EventLogIDs
from pix_framework.discovery.gateway_conditions.gateway_conditions import discover_gateway_conditions
from pix_framework.discovery.gateway_conditions.types import BranchRules, BranchRule


@dataclass
class PathProbability:
    path_id: str
    probability: float
    condition_id: str = None

    @staticmethod
    def from_dict(path_probabilities: dict) -> "PathProbability":
        return PathProbability(
            path_id=path_probabilities["path_id"],
            probability=path_probabilities["value"],
            condition_id=path_probabilities.get("condition_id")
        )

    def to_dict(self):
        """
        Dictionary compatible with Prosimos.
        """
        data = {
            "path_id": self.path_id,
            "value": self.probability
        }
        if self.condition_id is not None:
            data["condition_id"] = self.condition_id
        return data


@dataclass
class GatewayProbabilities:
    """
    Gateway branching probabilities for Prosimos.
    """

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
        """
        Dictionary compatible with Prosimos.
        """
        return {
            "gateway_id": self.gateway_id,
            "probabilities": [p.to_dict() for p in self.outgoing_paths],
        }


class GatewayProbabilitiesDiscoveryMethod(str, Enum):
    """
    Gateway probabilities discovery method. It can be either discovery or equiprobable. Equiprobable assumes that
    all outgoing paths from a gateway have the same probability. Discovery computes the probability of each path
    based on the BPMN model and event log.
    """

    DISCOVERY = "discovery"
    EQUIPROBABLE = "equiprobable"
    # DISCOVERY_WITH_CONDITIONS = "discovery_with_conditions"
    # EQUIPROBABLE_WITH_CONDITIONS = "equiprobable_with_conditions"

    @classmethod
    def from_str(
        cls, value: Union[str, List[str]]
    ) -> Union["GatewayProbabilitiesDiscoveryMethod", List["GatewayProbabilitiesDiscoveryMethod"],]:
        if isinstance(value, str):
            return GatewayProbabilitiesDiscoveryMethod._from_str(value)
        elif isinstance(value, list):
            return [GatewayProbabilitiesDiscoveryMethod._from_str(v) for v in value]

    @classmethod
    def _from_str(cls, value: str) -> "GatewayProbabilitiesDiscoveryMethod":
        if value.lower() == "discovery":
            return cls.DISCOVERY
        elif value.lower() == "equiprobable":
            return cls.EQUIPROBABLE
        # elif value.lower() == "discovery_with_conditions":
        #     return cls.DISCOVERY_WITH_CONDITIONS
        # elif value.lower() == "equiprobable_with_conditions":
        #     return cls.EQUIPROBABLE_WITH_CONDITIONS
        else:
            raise ValueError(f"Unknown value {value}")

    def __str__(self):
        if self == GatewayProbabilitiesDiscoveryMethod.DISCOVERY:
            return "discovery"
        elif self == GatewayProbabilitiesDiscoveryMethod.EQUIPROBABLE:
            return "equiprobable"
        # elif self == GatewayProbabilitiesDiscoveryMethod.DISCOVERY_WITH_CONDITIONS:
        #     return "discovery_with_conditions"
        # elif self == GatewayProbabilitiesDiscoveryMethod.EQUIPROBABLE_WITH_CONDITIONS:
        #     return "equiprobable_with_conditions"
        return f"Unknown GateManagement {str(self)}"


def compute_gateway_probabilities(
    event_log: pd.DataFrame,
    log_ids: EventLogIDs,
    bpmn_graph: BPMNGraph,
    discovery_method: GatewayProbabilitiesDiscoveryMethod = GatewayProbabilitiesDiscoveryMethod.DISCOVERY,
) -> List[GatewayProbabilities]:
    """
    Compute the gateway probabilities for a given event log and BPMN model.
    """
    if discovery_method is GatewayProbabilitiesDiscoveryMethod.EQUIPROBABLE:
        gateway_probabilities = bpmn_graph.compute_equiprobable_gateway_probabilities()
        # branch_rules = None
    elif discovery_method is GatewayProbabilitiesDiscoveryMethod.DISCOVERY:
        gateway_probabilities = discover_gateway_probabilities(bpmn_graph, event_log, log_ids)
        # branch_rules = None
    # elif discovery_method is GatewayProbabilitiesDiscoveryMethod.DISCOVERY_WITH_CONDITIONS:
    #     gateway_probabilities = discover_gateway_probabilities(bpmn_graph, event_log, log_ids)
    #     branch_rules = discover_gateway_conditions(bpmn_graph, event_log, log_ids)
    #     gateway_probabilities = map_conditions_to_flows(gateway_probabilities, branch_rules)
    # elif discovery_method is GatewayProbabilitiesDiscoveryMethod.EQUIPROBABLE_WITH_CONDITIONS:
    #     gateway_probabilities = bpmn_graph.compute_equiprobable_gateway_probabilities()
    #     branch_rules = discover_gateway_conditions(bpmn_graph, event_log, log_ids)
    #     gateway_probabilities = map_conditions_to_flows(gateway_probabilities, branch_rules)
    else:
        raise ValueError(f"Unknown gateway probabilities discovery method: {discovery_method}")

    return _translate_to_prosimos_format(gateway_probabilities)


def discover_gateway_probabilities(bpmn_graph: BPMNGraph, event_log: pd.DataFrame, log_ids: EventLogIDs):
    """
    Discover the frequency of each gateway branch with replay.
    """

    arcs_frequencies = {}

    for _, events in event_log.groupby(log_ids.case):
        trace = events.sort_values([log_ids.start_time, log_ids.end_time])[log_ids.activity].tolist()

        bpmn_graph.replay_trace(trace, arcs_frequencies)

    gateway_probabilities = bpmn_graph.discover_gateway_probabilities(arcs_frequencies)

    return gateway_probabilities


def _translate_to_prosimos_format(gateway_probabilities) -> List[GatewayProbabilities]:
    prosimos_gateway_probabilities = [
        GatewayProbabilities(
            gateway_id,
            [
                PathProbability(outgoing_node, gateway_probabilities[gateway_id][outgoing_node])
                for outgoing_node in gateway_probabilities[gateway_id]
            ],
        )
        for gateway_id in gateway_probabilities
    ]

    return prosimos_gateway_probabilities

# def _translate_to_prosimos_format(gateway_probabilities, branch_rules) -> (List[GatewayProbabilities], List[BranchRules]):
#     prosimos_branch_rules = None
#     prosimos_gateway_probabilities = [
#         GatewayProbabilities(
#             gateway_id,
#             [
#                 PathProbability(outgoing_node, gateway_probabilities[gateway_id][outgoing_node])
#                 for outgoing_node in gateway_probabilities[gateway_id]
#             ],
#         )
#         for gateway_id in gateway_probabilities
#     ]
#
#     if branch_rules is not None:
#         prosimos_branch_rules = [
#             BranchRules(
#                 id=rule["id"],
#                 rules=[
#                     [BranchRule(attribute=cond["attribute"], comparison=cond["comparison"], value=cond["value"])
#                      for cond in rule_group]
#                     for rule_group in rule["rules"]
#                 ]
#             )
#             for rule in branch_rules
#         ]
#
#     return prosimos_gateway_probabilities, prosimos_branch_rules


def map_conditions_to_flows(gateway_probabilities, branch_rules):
    conditions_lookup = {cond['id']: cond for cond in branch_rules}

    for gateway in gateway_probabilities:
        probabilities = gateway['probabilities']

        for prob in probabilities:
            flow_id = prob['path_id']

            if flow_id in conditions_lookup:
                prob['condition_id'] = flow_id

    return gateway_probabilities
