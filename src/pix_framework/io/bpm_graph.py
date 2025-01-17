import copy
import logging
import random
import sys
from collections import deque
from enum import Enum
from pathlib import Path
from typing import List, Tuple
from xml.etree import ElementTree

import numpy as np

BPMN_NAMESPACE_URI = "http://www.omg.org/spec/BPMN/20100524/MODEL"

logger = logging.getLogger(__name__)


class BPMNNodeType(Enum):
    TASK = "TASK"
    START_EVENT = "START-EVENT"
    END_EVENT = ("END-EVENT",)
    EXCLUSIVE_GATEWAY = "EXCLUSIVE-GATEWAY"
    INCLUSIVE_GATEWAY = "INCLUSIVE-GATEWAY"
    PARALLEL_GATEWAY = "PARALLEL-GATEWAY"
    INTERMEDIATE_EVENT = "INTERMEDIATE-EVENT"
    UNDEFINED = "UNDEFINED"


class ElementInfo:
    def __init__(self, element_type: BPMNNodeType, element_id: str, element_name: str):
        self.id = element_id
        self.name = element_name
        self.type = element_type
        self.incoming_flows = list()
        self.outgoing_flows = list()

    def is_split(self):
        return len(self.outgoing_flows) > 1

    def is_join(self):
        return len(self.incoming_flows) > 1

    def is_gateway(self):
        return self.type in [
            BPMNNodeType.EXCLUSIVE_GATEWAY,
            BPMNNodeType.PARALLEL_GATEWAY,
            BPMNNodeType.INCLUSIVE_GATEWAY,
        ]


class ProcessState:
    def __init__(self, bpmn_graph):
        self.arcs_bitset = bpmn_graph.arcs_bitset
        self.tokens = dict()
        self.flow_date = dict()
        self.state_mask = 0
        for flow_arc in bpmn_graph.flow_arcs:
            self.tokens[flow_arc] = 0

    def add_token(self, flow_id):
        if flow_id in self.tokens:
            self.tokens[flow_id] += 1
            self.state_mask |= self.arcs_bitset[flow_id]

    def remove_token(self, flow_id):
        if self.has_token(flow_id):
            self.tokens[flow_id] = 0
            self.state_mask &= ~self.arcs_bitset[flow_id]

    def has_token(self, flow_id):
        return flow_id in self.tokens and self.tokens[flow_id] > 0

    def pending_tokens(self):
        marked_flows = list()
        for flow_id in self.tokens:
            if self.tokens[flow_id] > 0:
                marked_flows.append(flow_id)
        return marked_flows


class BPMNGraph:
    def __init__(self):
        self.starting_event = None
        self.end_event = None
        self.element_info = dict()
        self.from_name = dict()
        self.flow_arcs = dict()
        self.concurrent_enablers = dict()
        self.nodes_bitset = dict()
        self.arcs_bitset = dict()
        self.or_join_pred = dict()  # or_id -> [0 = node predecesors bitset, 1 = predecesors flow arcs]
        self.or_join_conflicting_pred = dict()
        self.decision_successors = dict()
        self.element_probability = None
        self.task_resource_probability = None
        self.closest_distance = None
        self.decision_flows_sortest_path = None

        self._c_trace = None
        self.gateway_states = dict()
        self.current_attributes = dict()

    @staticmethod
    def from_bpmn_path(model_path: Path):
        bpmn_element_ns = {"xmlns": BPMN_NAMESPACE_URI}
        tree = ElementTree.parse(model_path.absolute())
        root = tree.getroot()
        to_extract = {
            "xmlns:task": BPMNNodeType.TASK,
            "xmlns:startEvent": BPMNNodeType.START_EVENT,
            "xmlns:endEvent": BPMNNodeType.END_EVENT,
            "xmlns:exclusiveGateway": BPMNNodeType.EXCLUSIVE_GATEWAY,
            "xmlns:intermediateCatchEvent": BPMNNodeType.INTERMEDIATE_EVENT,
            "xmlns:parallelGateway": BPMNNodeType.PARALLEL_GATEWAY,
            "xmlns:inclusiveGateway": BPMNNodeType.INCLUSIVE_GATEWAY,
        }

        bpmn_graph = BPMNGraph()
        for process in root.findall("xmlns:process", bpmn_element_ns):
            for xmlns_key in to_extract:
                for bpmn_element in process.findall(xmlns_key, bpmn_element_ns):
                    name = (
                        bpmn_element.attrib["name"]
                        if "name" in bpmn_element.attrib and len(bpmn_element.attrib["name"]) > 0
                        else bpmn_element.attrib["id"]
                    )
                    bpmn_graph.add_bpmn_element(
                        bpmn_element.attrib["id"],
                        ElementInfo(to_extract[xmlns_key], bpmn_element.attrib["id"], name),
                    )
            for flow_arc in process.findall("xmlns:sequenceFlow", bpmn_element_ns):
                bpmn_graph.add_flow_arc(
                    flow_arc.attrib["id"],
                    flow_arc.attrib["sourceRef"],
                    flow_arc.attrib["targetRef"],
                )
        bpmn_graph.encode_or_join_predecessors()

        return bpmn_graph

    def capture_gateway_state(self, gateway_id, gateway_type, decision_made, outgoing_flows):
        if gateway_id not in self.gateway_states:
            self.gateway_states[gateway_id] = {
                "type": gateway_type,
                "decisions": [],
                "outgoing_flows": outgoing_flows,
                "attributes": []
            }

        self.gateway_states[gateway_id]["decisions"].append(decision_made)
        self.gateway_states[gateway_id]["attributes"].append(self.current_attributes)

    def set_element_probabilities(self, element_probability, task_resource_probability):
        self.element_probability = element_probability
        self.task_resource_probability = task_resource_probability

    def add_bpmn_element(self, element_id, element_info):
        if element_info.type == BPMNNodeType.START_EVENT:
            self.starting_event = element_id
        if element_info.type == BPMNNodeType.END_EVENT:
            self.end_event = element_id
        self.element_info[element_id] = element_info
        self.from_name[element_info.name] = element_id
        self.nodes_bitset[element_id] = 1 << len(self.element_info)

    def add_flow_arc(self, flow_id, source_id, target_id):
        for node_id in [source_id, target_id]:
            if node_id not in self.element_info:
                self.element_info[node_id] = ElementInfo(BPMNNodeType.UNDEFINED, node_id, node_id)
        self.element_info[source_id].outgoing_flows.append(flow_id)
        self.element_info[target_id].incoming_flows.append(flow_id)
        self.flow_arcs[flow_id] = [source_id, target_id]
        self.arcs_bitset[flow_id] = 1 << len(self.flow_arcs)

    def encode_or_join_predecessors(self):
        for e_id in self.element_info:
            element = self.element_info[e_id]
            if element.type is BPMNNodeType.INCLUSIVE_GATEWAY and element.is_join():
                self.or_join_pred[e_id] = [0, 0]
                self._find_or_conflicting_predecessors(e_id)
                pred_queue = deque([e_id])
                while len(pred_queue) > 0:
                    element = self.element_info[pred_queue.popleft()]
                    for flow_id in element.incoming_flows:
                        prev_id = self.flow_arcs[flow_id][0]
                        if self.or_join_pred[e_id][0] & self.nodes_bitset[prev_id] == 0:
                            pred_queue.append(prev_id)
                        self.or_join_pred[e_id][0] |= self.nodes_bitset[prev_id]
                        if self.flow_arcs[flow_id][1] != e_id:
                            self.or_join_pred[e_id][1] |= self.arcs_bitset[flow_id]
            if element.type in [BPMNNodeType.EXCLUSIVE_GATEWAY, BPMNNodeType.INCLUSIVE_GATEWAY] and element.is_split():
                self._find_decision_successors(element)

    def _find_decision_successors(self, split_info):
        self.decision_successors[split_info.id] = set()
        visited = {split_info.id}
        suc_queue = deque([split_info])
        while suc_queue:
            e_info = suc_queue.popleft()
            for out_flow in e_info.outgoing_flows:
                next_info = self._get_successor(out_flow)
                if next_info.id not in visited:
                    visited.add(next_info.id)
                    next_info = self.element_info[next_info.id]
                    if next_info.type is BPMNNodeType.TASK:
                        self.decision_successors[split_info.id].add(next_info.id)
                    elif next_info.is_gateway():
                        suc_queue.append(next_info)

    def _find_or_conflicting_predecessors(self, or_join_id):
        visited = {or_join_id}
        self.or_join_conflicting_pred[or_join_id] = set()
        for in_flow in self.element_info[or_join_id].incoming_flows:
            self._dfs_from_or_join(or_join_id, in_flow, self._get_predecessor(in_flow), visited)

    def _dfs_from_or_join(self, or_id, flow_id, e_info, visited):
        visited.add(e_info.id)
        if e_info.type in [BPMNNodeType.INCLUSIVE_GATEWAY, BPMNNodeType.EXCLUSIVE_GATEWAY] and e_info.is_split():
            self.or_join_conflicting_pred[or_id].add(e_info.id)
        for in_flow in e_info.incoming_flows:
            prev_info = self._get_predecessor(in_flow)
            if prev_info.id not in visited and prev_info.is_gateway():
                self._dfs_from_or_join(or_id, flow_id, prev_info, visited)

    def _is_enabled(self, e_id, p_state):
        if e_id not in self.element_info:
            return False
        if e_id == self.starting_event:
            return True
        e_info = self.element_info[e_id]
        if e_info.type in [
            BPMNNodeType.TASK,
            BPMNNodeType.END_EVENT,
            BPMNNodeType.PARALLEL_GATEWAY,
        ]:
            for f_arc in e_info.incoming_flows:
                if p_state.tokens[f_arc] < 1:
                    return False
            return True
        elif e_info.type == BPMNNodeType.EXCLUSIVE_GATEWAY:
            for f_arc in e_info.incoming_flows:
                if p_state.tokens[f_arc] > 0:
                    return True
            return False
        elif e_info.type == BPMNNodeType.INCLUSIVE_GATEWAY:
            if e_info.is_split():
                if p_state.has_token(e_info.incoming_flows[0]):
                    return True
                for flow_id in e_info.outgoing_flows:
                    if p_state.has_token(flow_id):
                        return True
                return False
            else:
                count_tokens = 0
                for flow_id in e_info.incoming_flows:
                    if p_state.tokens[flow_id] > 0:
                        count_tokens += 1
                if count_tokens == len(e_info.incoming_flows):
                    return True
                if count_tokens > 0 and self.or_join_pred[e_id][1] & p_state.state_mask == 0:
                    return True
                return False
        return False

    def get_gateway_states(self):
        return self.gateway_states

    def update_process_state(self, e_id, p_state):
        if not self._is_enabled(e_id, p_state):
            return []
        enabled_tasks = list()
        to_execute = [e_id]
        current = 0
        while current < len(to_execute):
            e_info = self.element_info[to_execute[current]]
            for in_flow in e_info.incoming_flows:
                if p_state.tokens[in_flow] > 0:
                    p_state.tokens[in_flow] -= 1
                    p_state.state_mask &= ~self.arcs_bitset[in_flow]
            f_arcs = e_info.outgoing_flows
            if len(f_arcs) > 1:
                if e_info.type is BPMNNodeType.EXCLUSIVE_GATEWAY:
                    f_arcs = [self.element_probability[e_info.id].get_outgoing_flow()]
                else:
                    if e_info.type in [
                        BPMNNodeType.TASK,
                        BPMNNodeType.PARALLEL_GATEWAY,
                        BPMNNodeType.START_EVENT,
                    ]:
                        f_arcs = copy.deepcopy(e_info.outgoing_flows)
                    elif e_info.type is BPMNNodeType.INCLUSIVE_GATEWAY:
                        f_arcs = self.element_probability[e_info.id].get_multiple_flows()
                random.shuffle(f_arcs)
            for f_arc in f_arcs:
                self._find_next(f_arc, p_state, enabled_tasks, to_execute)
            current += 1
        if len(enabled_tasks) > 1:
            random.shuffle(enabled_tasks)
        return enabled_tasks

    def replay_trace(
        self, task_sequence: list, f_arcs_frequency: dict, post_p=True, trace=None
    ) -> Tuple[bool, List[bool], ProcessState]:
        self._c_trace = trace
        task_enabling = list()
        p_state = ProcessState(self)
        fired_tasks = list()
        fired_or_splits = set()
        for flow_id in self.element_info[self.starting_event].outgoing_flows:
            p_state.flow_date[flow_id] = self._c_trace[0].started_at if self._c_trace is not None else None
            p_state.add_token(flow_id)

        if self._c_trace:
            self.update_flow_dates(self.element_info[self.starting_event], p_state, self._c_trace[0].started_at)
        pending_tasks = dict()
        for current_index in range(len(task_sequence)):
            el_id = self.from_name.get(task_sequence[current_index])
            fired_tasks.append(False)

            in_flow = self.element_info[el_id].incoming_flows[0]
            task_enabling.append(p_state.flow_date[in_flow] if in_flow in p_state.flow_date else None)
            if self._c_trace:
                self.update_flow_dates(self.element_info[el_id], p_state,
                                       self._c_trace[current_index].completed_at if self._c_trace is not None else None)

            # NOTE: changing self.try_firing to self.try_firing_alternative
            # to avoid no such element in self.from_name error
            self.try_firing_alternative(
                current_index,
                current_index,
                task_sequence,
                fired_tasks,
                pending_tasks,
                p_state,
                f_arcs_frequency,
                fired_or_splits,
            )
            el_id = self.from_name.get(task_sequence[current_index])
            if el_id is None:  # NOTE: skipping if no such element in self.from_name
                continue
            p_state.add_token(self.element_info[el_id].outgoing_flows[0])
            if current_index in pending_tasks:
                for pending_index in pending_tasks[current_index]:
                    self.try_firing_alternative(
                        pending_index,
                        current_index,
                        task_sequence,
                        fired_tasks,
                        pending_tasks,
                        p_state,
                        f_arcs_frequency,
                        fired_or_splits,
                    )

        # Firing End Event
        enabled_end, or_fired, path_decisions = self._find_enabled_predecessors(
            self.element_info[self.end_event], p_state
        )
        self._fire_enabled_predecessors(
            enabled_end,
            p_state,
            or_fired,
            path_decisions,
            f_arcs_frequency,
            fired_or_splits,
        )
        end_flow = self.element_info[self.end_event].incoming_flows[0]
        if p_state.has_token(end_flow):
            p_state.tokens[end_flow] = 0

        is_correct = True
        for i in range(0, len(task_sequence)):
            if not fired_tasks[i]:
                is_correct = False
                break

        self.check_unfired_or_splits(fired_or_splits, f_arcs_frequency, p_state)
        if post_p:
            self.postprocess_unfired_tasks(task_sequence, fired_tasks, f_arcs_frequency, task_enabling)
        self._c_trace = None
        return is_correct, fired_tasks, p_state.pending_tokens()

    def update_flow_dates(self, e_info: ElementInfo, p_state: ProcessState, last_date):
        visited_elements = set()
        suc_queue = deque([e_info])
        visited_elements.add(e_info.id)
        while suc_queue:
            e_info = suc_queue.popleft()
            for out_flow in e_info.outgoing_flows:
                next_info = self._get_successor(out_flow)
                p_state.flow_date[out_flow] = last_date
                if next_info.is_gateway() and next_info.id not in visited_elements:
                    suc_queue.append(next_info)
                    visited_elements.add(next_info.id)

    def postprocess_unfired_tasks(self, task_sequence: list, fired_tasks: list, f_arcs_frequency: dict, task_enablement: list):
        if self.closest_distance is None:
            self._sort_by_closest_predecesors()
        for i in range(0, len(fired_tasks)):
            if not fired_tasks[i] and task_sequence[i] is not None and self.from_name.get(task_sequence[i]):
                e_info = self.element_info[self.from_name.get(task_sequence[i])]
                fix_from = [
                    self.starting_event,
                    self.closest_distance[e_info.id][self.starting_event],
                ]
                j = i - 1
                while j >= 0:
                    if task_sequence[j] is None or self.from_name.get(task_sequence[j]) is None:
                        j -= 1
                        continue
                    p_info = self.element_info[self.from_name.get(task_sequence[j])]
                    if (
                        p_info.id in self.closest_distance[e_info.id]
                        and self.closest_distance[e_info.id][p_info.id] < fix_from[1]
                    ):
                        fix_from = [
                            p_info.id,
                            self.closest_distance[e_info.id][p_info.id],
                        ]
                        if fix_from[1] == 1:
                            break
                    j -= 1
                if fix_from[0] is not None:
                    if task_enablement[i] is None and self._c_trace:
                        task_enablement[i] = self._c_trace[j].completed_at if j >= 0 else self._c_trace[0].completed_at
                    for flow_id in self.decision_flows_sortest_path[e_info.id][fix_from[0]]:
                        if flow_id not in f_arcs_frequency:
                            f_arcs_frequency[flow_id] = 0
                        f_arcs_frequency[flow_id] += 1

    def _sort_by_closest_predecesors(self):
        self.closest_distance = dict()
        self.decision_flows_sortest_path = dict()
        for e_id in self.element_info:
            self.closest_distance[e_id] = dict()
            pred_seq = dict()
            distance_map = {e_id: 0}
            pred_queue = deque([self.element_info[e_id]])
            while pred_queue:
                e_info = pred_queue.popleft()
                for flow_id in e_info.incoming_flows:
                    pred_info = self._get_predecessor(flow_id)
                    if pred_info.id not in distance_map:
                        pred_seq[pred_info.id] = flow_id
                        dist = distance_map[e_info.id]
                        if pred_info.type in [
                            BPMNNodeType.TASK,
                            BPMNNodeType.START_EVENT,
                        ]:
                            dist += 1
                            self.closest_distance[e_id][pred_info.id] = dist
                        distance_map[pred_info.id] = dist
                        pred_queue.append(pred_info)
            self.decision_flows_sortest_path[e_id] = dict()
            for p_id in self.element_info:
                self.decision_flows_sortest_path[e_id][p_id] = list()
                if p_id is not e_id and p_id in self.closest_distance[e_id]:
                    p_info = self.element_info[p_id]
                    while p_info.id is not e_id:
                        if (
                            p_info.type
                            in [
                                BPMNNodeType.INCLUSIVE_GATEWAY,
                                BPMNNodeType.EXCLUSIVE_GATEWAY,
                            ]
                            and p_info.is_split()
                        ):
                            self.decision_flows_sortest_path[e_id][p_id].append(pred_seq[p_info.id])
                        p_info = self._get_successor(pred_seq[p_info.id])

    def try_firing(
        self,
        task_index,
        from_index,
        task_sequence,
        fired_tasks,
        pending_tasks,
        p_state,
        f_arcs_frequency,
        fired_or_splits,
    ):
        task_info = self.element_info[self.from_name[task_sequence[task_index]]]
        if not p_state.has_token(task_info.incoming_flows[0]):
            enabled_pred, or_fired, path_decisions = self._find_enabled_predecessors(task_info, p_state)
            firing_index = self.find_firing_index(task_index, from_index, task_sequence, path_decisions, enabled_pred)
            if firing_index == from_index:
                self._fire_enabled_predecessors(
                    enabled_pred,
                    p_state,
                    or_fired,
                    path_decisions,
                    f_arcs_frequency,
                    fired_or_splits,
                )
            elif firing_index not in pending_tasks:
                pending_tasks[firing_index] = [task_index]
            else:
                pending_tasks[firing_index].append(task_index)
        if p_state.has_token(task_info.incoming_flows[0]):
            p_state.remove_token(task_info.incoming_flows[0])
            fired_tasks[task_index] = True

    def try_firing_alternative(
        self,
        task_index,
        from_index,
        task_sequence,
        fired_tasks,
        pending_tasks,
        p_state,
        f_arcs_frequency,
        fired_or_splits,
    ):
        el_id = self.from_name.get(task_sequence[task_index])
        if el_id is None:
            return
        task_info = self.element_info[el_id]
        if not p_state.has_token(task_info.incoming_flows[0]):
            enabled_pred, or_fired, path_decisions = self._find_enabled_predecessors(task_info, p_state)
            firing_index = self.find_firing_index(task_index, from_index, task_sequence, path_decisions, enabled_pred)
            if firing_index == from_index:
                self._fire_enabled_predecessors(
                    enabled_pred,
                    p_state,
                    or_fired,
                    path_decisions,
                    f_arcs_frequency,
                    fired_or_splits,
                )
            elif firing_index not in pending_tasks:
                pending_tasks[firing_index] = [task_index]
            else:
                pending_tasks[firing_index].append(task_index)
        if p_state.has_token(task_info.incoming_flows[0]):
            p_state.remove_token(task_info.incoming_flows[0])
            fired_tasks[task_index] = True
            if self._c_trace:
                self.current_attributes = self._c_trace[task_index].attributes

    def closer_enabled_predecessors(
        self,
        e_info,
        flow_id,
        enabled_pred,
        or_firing,
        path_split,
        visited,
        p_state,
        dist,
        min_dist,
    ):
        if self._is_enabled(e_info.id, p_state):
            if dist not in enabled_pred:
                enabled_pred[dist] = list()
            enabled_pred[dist].append([e_info, flow_id])
            min_dist[0] = max(min_dist[0], dist)
            return dist, enabled_pred, or_firing, path_split
        elif e_info.type is BPMNNodeType.INCLUSIVE_GATEWAY and e_info.is_join():
            for in_or in e_info.incoming_flows:
                if p_state.has_token(in_or):
                    or_firing[e_info.id] = dist
                    break
        if e_info.type in [
            BPMNNodeType.INCLUSIVE_GATEWAY,
            BPMNNodeType.EXCLUSIVE_GATEWAY,
        ]:
            path_split[e_info.id] = flow_id
        visited.add(e_info.id)
        if e_info.is_gateway():
            if e_info.type is BPMNNodeType.EXCLUSIVE_GATEWAY and e_info.is_join():
                closer_pred, temp_path, or_f = dict(), dict(), dict()
                c_min = sys.maxsize
                for in_flow in e_info.incoming_flows:
                    pr_info = self._get_predecessor(in_flow)
                    if pr_info.id not in visited:
                        d, e_p, o_f, t_path = self.closer_enabled_predecessors(
                            pr_info,
                            in_flow,
                            dict(),
                            dict(),
                            dict(),
                            visited,
                            p_state,
                            dist + 1,
                            min_dist,
                        )
                        if d < c_min:
                            c_min, closer_pred, or_f, temp_path = d, e_p, o_f, t_path
                for e_id in closer_pred:
                    enabled_pred[e_id] = closer_pred[e_id]
                for e_id in temp_path:
                    path_split[e_id] = temp_path[e_id]
                for e_id in or_f:
                    or_firing[e_id] = dist
                return c_min, enabled_pred, or_firing, path_split
            else:
                c_min = dist if e_info.id in or_firing else sys.maxsize
                for in_flow in e_info.incoming_flows:
                    pred_info = self._get_predecessor(in_flow)
                    if pred_info.id not in visited and pred_info.is_gateway():
                        res = self.closer_enabled_predecessors(
                            pred_info,
                            in_flow,
                            enabled_pred,
                            or_firing,
                            path_split,
                            visited,
                            p_state,
                            dist + 1,
                            min_dist,
                        )
                        c_min = min(res[0], c_min)

                return c_min, enabled_pred, or_firing, path_split
        return sys.maxsize, enabled_pred, or_firing, path_split

    def _find_enabled_predecessors(self, from_task_info, p_state):
        pred_info = self._get_predecessor(from_task_info.incoming_flows[0])
        max_dist = [0]
        closer_pred = self.closer_enabled_predecessors(
            pred_info,
            from_task_info.incoming_flows[0],
            dict(),
            dict(),
            dict(),
            set(),
            p_state,
            0,
            max_dist,
        )
        enabled_pred = deque()
        for i in range(0, max_dist[0] + 1):
            if i in closer_pred[1]:
                for pred_id in closer_pred[1][i]:
                    enabled_pred.appendleft(pred_id)
        return enabled_pred, closer_pred[2], closer_pred[3]

    def find_firing_index(self, task_index, from_index, task_sequence, path_decisions, enabled_pred):
        is_conflicting, conflicting_gateways = self.is_conflicting_task(path_decisions, enabled_pred)
        if is_conflicting:
            firing_index = from_index
            for i in range(from_index + 1, len(task_sequence)):
                if task_sequence[i] != task_sequence[task_index]:
                    for or_id in conflicting_gateways:
                        for split_id in conflicting_gateways[or_id]:
                            t_id = self.from_name[task_sequence[i]]
                            if t_id in self.decision_successors[split_id]:
                                return i
            return firing_index
        return from_index

    def is_conflicting_task(self, path_decisions, enabled_pred):
        conflicting_gateways = dict()
        is_conflicting = False
        for or_id in path_decisions:
            if self.element_info[or_id].type is BPMNNodeType.INCLUSIVE_GATEWAY and self.element_info[or_id].is_join():
                conflicting_gateways[or_id] = set()
                for enabled in enabled_pred:
                    e_info = enabled[0]
                    if e_info.id in self.or_join_conflicting_pred[or_id]:
                        conflicting_gateways[or_id].add(e_info.id)
                    if len(conflicting_gateways[or_id]) > 1:
                        is_conflicting = True
        return is_conflicting, conflicting_gateways

    def _fire_enabled_predecessors(
        self,
        enabled_pred,
        p_state,
        or_firing,
        path_decisions,
        f_arcs_frequency,
        fired_or_split,
    ):
        visited_elements = set()
        if not enabled_pred:
            self.try_firing_or_join(enabled_pred, p_state, or_firing, path_decisions, f_arcs_frequency)
        while enabled_pred:
            [e_info, e_flow] = enabled_pred.popleft()
            if self._is_enabled(e_info.id, p_state):
                visited_elements.add(e_info.id)
                if e_info.type is BPMNNodeType.PARALLEL_GATEWAY:
                    for out_flow in e_info.outgoing_flows:
                        self._update_next(
                            out_flow,
                            enabled_pred,
                            p_state,
                            or_firing,
                            path_decisions,
                            f_arcs_frequency,
                        )
                elif e_info.type is BPMNNodeType.EXCLUSIVE_GATEWAY:
                    self._update_next(
                        e_flow,
                        enabled_pred,
                        p_state,
                        or_firing,
                        path_decisions,
                        f_arcs_frequency,
                    )
                    if e_info.is_split():
                        self.capture_gateway_state(
                            e_info.id,
                            "XOR",
                            [e_flow],
                            e_info.outgoing_flows
                        )
                elif e_info.type is BPMNNodeType.INCLUSIVE_GATEWAY:
                    self._update_next(
                        e_flow,
                        enabled_pred,
                        p_state,
                        or_firing,
                        path_decisions,
                        f_arcs_frequency,
                    )
                    if e_info.is_split():
                        fired_or_split.add(e_info.id)
                        decision_made = [e_flow]

                        for flow_id in e_info.outgoing_flows:
                            if flow_id != e_flow:
                                self._update_next(
                                    flow_id,
                                    enabled_pred,
                                    p_state,
                                    or_firing,
                                    path_decisions,
                                    f_arcs_frequency,
                                    True,
                                )
                                decision_made.append(flow_id)
                        self.capture_gateway_state(
                            e_info.id,
                            "OR",
                            decision_made,
                            e_info.outgoing_flows
                        )

            for in_flow in e_info.incoming_flows:
                p_state.remove_token(in_flow)
            self.try_firing_or_join(enabled_pred, p_state, or_firing, path_decisions, f_arcs_frequency)

    def try_firing_or_join(self, enabled_pred, p_state, or_firing, path_decisions, f_arcs_frequency):
        fired = set()
        or_firing_list = list()
        for or_join_id in or_firing:
            or_firing_list.append(or_join_id)
        for or_join_id in or_firing_list:
            if self._is_enabled(or_join_id, p_state) or not enabled_pred:
                fired.add(or_join_id)
                e_info = self.element_info[or_join_id]
                self._update_next(
                    e_info.outgoing_flows[0],
                    enabled_pred,
                    p_state,
                    or_firing,
                    path_decisions,
                    f_arcs_frequency,
                )
                for in_flow in e_info.incoming_flows:
                    p_state.remove_token(in_flow)
                if enabled_pred:
                    break
                if len(or_firing_list) != len(or_firing):
                    for e_id in or_firing:
                        if e_id not in or_firing_list:
                            or_firing_list.append(e_id)
        for or_id in fired:
            del or_firing[or_id]

    def check_unfired_or_splits(self, or_splits, f_arcs_frequency, p_state):
        for or_id in or_splits:
            for flow_id in self.element_info[or_id].outgoing_flows:
                if p_state.tokens[flow_id] > 0:
                    self.gateway_states[or_id]['decisions'][-1].remove(flow_id)
                    f_arcs_frequency[flow_id] -= p_state.tokens[flow_id]
                    p_state.tokens[flow_id] = 0

    def _update_next(
        self,
        flow_id,
        enabled_pred,
        p_state,
        or_firing,
        path_decisions,
        f_arcs_frequency,
        from_or=False,
    ):
        if flow_id not in f_arcs_frequency:
            f_arcs_frequency[flow_id] = 1
        else:
            f_arcs_frequency[flow_id] += 1
        p_state.add_token(flow_id)
        if not from_or:
            next_info = self._get_successor(flow_id)
            if next_info.type is BPMNNodeType.PARALLEL_GATEWAY and self._is_enabled(next_info.id, p_state):
                enabled_pred.appendleft([next_info, None])
            elif next_info.id in path_decisions:
                if next_info.type is BPMNNodeType.INCLUSIVE_GATEWAY:
                    if next_info.is_split():
                        enabled_pred.appendleft([next_info, path_decisions[next_info.id]])
                    else:
                        if next_info.id not in or_firing:
                            or_firing[next_info.id] = 1
                else:
                    enabled_pred.appendleft([next_info, path_decisions[next_info.id]])

    def _get_predecessor(self, flow_id):
        return self.element_info[self.flow_arcs[flow_id][0]]

    def _get_successor(self, flow_id):
        return self.element_info[self.flow_arcs[flow_id][1]]

    def compute_branching_probability(self, flow_arcs_frequency):
        gateways_branching = dict()
        for e_id in self.element_info:
            if (
                self.element_info[e_id].type in [BPMNNodeType.EXCLUSIVE_GATEWAY, BPMNNodeType.INCLUSIVE_GATEWAY]
                and len(self.element_info[e_id].outgoing_flows) > 1
            ):
                total_frequency = 0
                for flow_id in self.element_info[e_id].outgoing_flows:
                    if flow_id not in flow_arcs_frequency:
                        flow_arcs_frequency[flow_id] = 0
                    total_frequency += flow_arcs_frequency[flow_id]
                flow_arc_probability = dict()
                for flow_id in self.element_info[e_id].outgoing_flows:
                    flow_arc_probability[flow_id] = (
                        flow_arcs_frequency[flow_id] / total_frequency if total_frequency > 0 else 0
                    )
                gateways_branching[e_id] = flow_arc_probability
        return gateways_branching

    def discover_gateway_probabilities(self, flow_arcs_frequency):
        gateways_branching = dict()
        for e_id in self.element_info:
            if (
                self.element_info[e_id].type in [BPMNNodeType.EXCLUSIVE_GATEWAY, BPMNNodeType.INCLUSIVE_GATEWAY]
                and len(self.element_info[e_id].outgoing_flows) > 1
            ):
                (
                    flow_arcs_probability,
                    total_frequency,
                ) = self._calculate_arcs_probabilities(e_id, flow_arcs_frequency)
                # recalculate not only pure zeros, but also low probabilities --- PONER ESTO DE REGRESO
                if min(flow_arcs_probability.values()) <= 0.005:
                    self._recalculate_arcs_probabilities(flow_arcs_frequency, flow_arcs_probability, total_frequency)
                self._check_probabilities(flow_arcs_probability)
                gateways_branching[e_id] = flow_arcs_probability
        return gateways_branching

    def compute_equiprobable_gateway_probabilities(self):
        gateways_branching = dict()
        for e_id in self.element_info:
            if (
                self.element_info[e_id].type in [BPMNNodeType.EXCLUSIVE_GATEWAY, BPMNNodeType.INCLUSIVE_GATEWAY]
                and len(self.element_info[e_id].outgoing_flows) > 1
            ):
                average_probability = 1.0 / len(self.element_info[e_id].outgoing_flows)
                probabilities = dict()
                for flow_id in self.element_info[e_id].outgoing_flows:
                    probabilities[flow_id] = average_probability
                gateways_branching[e_id] = probabilities
        return gateways_branching

    @staticmethod
    def _recalculate_arcs_probabilities(flow_arcs_frequency, flow_arcs_probability, total_frequency):
        # recalculating probabilities because of missing arcs or very small probabilities for them
        arcs_probabilities = np.array(list(flow_arcs_probability.values()))
        valid_probability_threshold = 0.005
        min_probability = 0.01
        number_of_invalid_arcs = (arcs_probabilities <= valid_probability_threshold).sum()
        number_of_valid_arcs = len(flow_arcs_probability) - number_of_invalid_arcs
        if number_of_valid_arcs == 0:  # if all arcs are missing, we make the probability equiprobable
            probability = 1.0 / float(number_of_invalid_arcs)
            for flow_id in flow_arcs_probability:
                flow_arcs_probability[flow_id] = probability
        # FIX THIS CORRECTION BECAUSE IT MAY LEAD TO NEGATIVE PROBABILITIES
        # else:  # otherwise, we set min_probability instead of zero and balance probabilities for valid arcs
        #     valid_probabilities = arcs_probabilities[arcs_probabilities > valid_probability_threshold].sum()
        #     extra_probability = (number_of_invalid_arcs * min_probability) - (1.0 - valid_probabilities)
        #     extra_probability_per_valid_arc = extra_probability / number_of_valid_arcs
        #     for flow_id in flow_arcs_probability:
        #         if flow_arcs_probability[flow_id] <= valid_probability_threshold:
        #             # enforcing the minimum possible probability
        #             probability = min_probability
        #         else:
        #             # balancing valid probabilities
        #             probability = flow_arcs_probability[flow_id] - extra_probability_per_valid_arc
        #         flow_arcs_probability[flow_id] = probability

    @staticmethod
    def _check_probabilities(flow_arcs_probability):
        # checking probabilities sum up to 1.0
        tolerance = 0.001
        probabilities_sum = sum(flow_arcs_probability.values())
        if 1.0 - probabilities_sum >= tolerance:
            logger.warning(f"Sum of outgoing flow arcs does not sum up to {1.0 - tolerance}: {probabilities_sum}")

    def _calculate_arcs_probabilities(self, e_id, flow_arcs_frequency):
        total_frequency = 0
        for flow_id in self.element_info[e_id].outgoing_flows:
            frequency = flow_arcs_frequency.get(flow_id)
            total_frequency += frequency if frequency else 0
        flow_arcs_probability = dict()
        for flow_id in self.element_info[e_id].outgoing_flows:
            frequency = flow_arcs_frequency.get(flow_id)
            if frequency:
                probability = frequency / total_frequency
            else:
                probability = 0
            flow_arcs_probability[flow_id] = probability
        return flow_arcs_probability, total_frequency

    def _find_next(self, f_arc, p_state, enabled_tasks, to_execute):
        p_state.tokens[f_arc] += 1
        p_state.state_mask |= self.arcs_bitset[f_arc]
        next_e = self.flow_arcs[f_arc][1]
        if self._is_enabled(next_e, p_state):
            if self.element_info[next_e].type == BPMNNodeType.TASK:
                enabled_tasks.append(next_e)
            else:
                to_execute.append(next_e)
