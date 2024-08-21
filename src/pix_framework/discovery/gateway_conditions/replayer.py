import sys
from collections import deque
from enum import Enum
import csv
import pandas as pd
import datetime
import pytz


class BPMN(Enum):
    TASK = 'TASK'
    START_EVENT = 'START-EVENT'
    END_EVENT = 'END-EVENT',
    INTERMEDIATE_EVENT = 'INTERMEDIATE_EVENT',
    EXCLUSIVE_GATEWAY = 'EXCLUSIVE-GATEWAY'
    INCLUSIVE_GATEWAY = 'INCLUSIVE-GATEWAY'
    PARALLEL_GATEWAY = 'PARALLEL-GATEWAY'
    EVENT_BASED_GATEWAY = 'EVENT-BASED-GATEWAY'
    UNDEFINED = 'UNDEFINED'

    @classmethod
    def is_event(cls, type):
        if (type in [cls.START_EVENT, cls.END_EVENT, cls.INTERMEDIATE_EVENT]):
            return True
        else:
            return False


class ElementInfo:
    def __init__(self, element_type, element_id, element_name, event_type):
        self.id = element_id
        self.name = element_name
        self.type = element_type
        self.event_type = event_type
        self.incoming_flows = list()
        self.outgoing_flows = list()

    def is_split(self):
        return len(self.outgoing_flows) > 1

    def is_join(self):
        return len(self.incoming_flows) > 1

    def is_gateway(self):
        return self.type in [BPMN.EXCLUSIVE_GATEWAY, BPMN.PARALLEL_GATEWAY, BPMN.INCLUSIVE_GATEWAY, BPMN.EVENT_BASED_GATEWAY]

    def is_start_or_end_event(self):
        return self.type in [BPMN.START_EVENT, BPMN.END_EVENT]

    def is_event(self):
        return self.type in [BPMN.START_EVENT, BPMN.END_EVENT, BPMN.INTERMEDIATE_EVENT]


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

    def remove_all_tokens_on_terminate(self):
        for token in self.tokens:
            self.remove_token(token)

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
        self.end_events_count = 0
        self.element_info = dict()
        self.from_name = dict()
        self.flow_arcs = dict()
        self.nodes_bitset = dict()
        self.arcs_bitset = dict()
        self.or_join_pred = dict()  # or_id -> [0 = node predecesors bitset, 1 = predecesors flow arcs]
        self.or_join_conflicting_pred = dict()
        self.decision_successors = dict()
        self.closest_distance = None
        self.decision_flows_sortest_path = None
        self._c_trace = None
        self.event_distribution = None
        self.last_datetime = dict()

        self.gateway_states = dict()
        self.current_attributes = dict()

    def capture_gateway_state(self, gateway_id, gateway_type, decision_made, outgoing_flows):
        if gateway_id not in self.gateway_states:
            self.gateway_states[gateway_id] = {
                "type": gateway_type,
                "incoming_flows": [],
                "decisions": [],
                "outgoing_flows": outgoing_flows,
                "attributes": []
            }

        self.gateway_states[gateway_id]["decisions"].append(decision_made)
        self.gateway_states[gateway_id]["attributes"].append(self.current_attributes)

    def fire_enabled_predecessors(self, enabled_pred, p_state, or_firing, path_decisions, f_arcs_frequency,
                                  fired_or_split):
        # print("\t\t-->IN PREDECESSORS")
        # print(f"\t\tENABLED: {enabled_pred}")
        visited_elements = set()
        if not enabled_pred:
            # print("\t\tFIRING OR-JOIN")
            self.try_firing_or_join(enabled_pred, p_state, or_firing, path_decisions, f_arcs_frequency)
        while enabled_pred:
            # print(f"\t\tENABLED_PRED:\t{enabled_pred}")
            [e_info, e_flow] = enabled_pred.popleft()
            if self.is_enabled(e_info.id, p_state):
                visited_elements.add(e_info.id)

                if e_info.type == BPMN.PARALLEL_GATEWAY:
                    for out_flow in e_info.outgoing_flows:
                        self._update_next(out_flow, enabled_pred, p_state, or_firing, path_decisions, f_arcs_frequency)

                elif e_info.type == BPMN.EXCLUSIVE_GATEWAY:
                    self._update_next(e_flow, enabled_pred, p_state, or_firing, path_decisions, f_arcs_frequency)
                    if e_info.is_split():
                        self.capture_gateway_state(
                            e_info.id,
                            "XOR",
                            [e_flow],
                            e_info.outgoing_flows
                        )

                elif e_info.type == BPMN.INCLUSIVE_GATEWAY:
                    # print("\t\t\tINCLUSIVE GATEWAY")
                    # print(f"GATEWAY {e_flow}, {enabled_pred}, {or_firing}, {path_decisions}")
                    self._update_next(e_flow, enabled_pred, p_state, or_firing, path_decisions, f_arcs_frequency)
                    if e_info.is_split():
                        # print("\t\t\tAND ITS SPLIT!!!")
                        fired_or_split.add(e_info.id)
                        decision_made = [e_flow]
                        # print(f"\t\t\telement flow id: {decision_made}")

                        for flow_id in e_info.outgoing_flows:
                            # print(f"{e_info.outgoing_flows} => {decision_made} | {flow_id}")
                            # print(f"flow: {flow_id} != {e_flow} => {flow_id != e_flow}")
                            # print(f"{flow_id}\t{enabled_pred}\t{p_state}\t{or_firing}\t{path_decisions}\t{f_arcs_frequency}")
                            if flow_id != e_flow:
                                self._update_next(flow_id, enabled_pred, p_state, or_firing, path_decisions,
                                                  f_arcs_frequency, True)
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

    def try_firing(self, task_index, from_index, task_sequence, fired_tasks, pending_tasks, p_state,
                   f_arcs_frequency, fired_or_splits):
        # print("\t\tTRY FIRING METHOD")
        el_id = self.from_name.get(task_sequence[task_index])
        if el_id is None:
            return
        task_info = self.element_info[el_id]
        # print(f"Task info: {task_info} {task_info.id}\t{task_info.name}\t{task_info.type}\t{task_info.incoming_flows}\t{task_info.outgoing_flows}")

        if not p_state.has_token(task_info.incoming_flows[0]):
            enabled_pred, or_fired, path_decisions = self._find_enabled_predecessors(task_info, p_state)
            # print(f"\tNO TOKEN, NEXT PATH:: {path_decisions}")

            firing_index = self.find_firing_index(task_index, from_index, task_sequence, path_decisions, enabled_pred)
            # print(f"\tFIRING INDEX: {firing_index}")
            if firing_index == from_index:
                self.fire_enabled_predecessors(enabled_pred, p_state, or_fired, path_decisions, f_arcs_frequency,
                                               fired_or_splits)
            elif firing_index not in pending_tasks:
                pending_tasks[firing_index] = [task_index]
            else:
                pending_tasks[firing_index].append(task_index)
        if p_state.has_token(task_info.incoming_flows[0]):
            # print(f"HAS TOKEN: {task_info} {task_info.id}\t{task_info.name}\t{task_info.type}\t{task_info.incoming_flows}\t{task_info.outgoing_flows}")
            p_state.remove_token(task_info.incoming_flows[0])
            fired_tasks[task_index] = True
            self.current_attributes = self._c_trace[task_index].attributes

    def replay_trace(self, task_sequence, f_arcs_frequency, post_p=True, trace=None):
        self._c_trace = trace
        task_enabling = list()
        p_state = ProcessState(self)
        fired_tasks = list()
        fired_or_splits = set()

        for flow_id in self.element_info[self.starting_event].outgoing_flows:
            p_state.flow_date[flow_id] = self._c_trace[0].started_at if self._c_trace is not None else None
            p_state.add_token(flow_id)
        self.update_flow_dates(self.element_info[self.starting_event], p_state, self._c_trace[0].started_at)
        pending_tasks = dict()
        for current_index in range(len(task_sequence)):
            el_id = self.from_name.get(task_sequence[current_index])
            fired_tasks.append(False)
            in_flow = self.element_info[el_id].incoming_flows[0]
            # print(f"Current index: {current_index}\t el_id: {el_id}\t inflow: {in_flow}")
            # print(f_arcs_frequency)

            task_enabling.append(p_state.flow_date[in_flow] if in_flow in p_state.flow_date else None)
            if self._c_trace:
                self.update_flow_dates(self.element_info[el_id], p_state,
                                       self._c_trace[current_index].completed_at if self._c_trace is not None else None)

            # print(f"{current_index} {task_sequence} ===> ({task_sequence[current_index]})")
            self.try_firing(current_index, current_index, task_sequence, fired_tasks, pending_tasks,
                            p_state, f_arcs_frequency, fired_or_splits)
            # print("After firing:")
            # print(f_arcs_frequency)

            if el_id is None:  # NOTE: skipping if no such element in self.from_name
                continue
            p_state.add_token(self.element_info[el_id].outgoing_flows[0])
            if current_index in pending_tasks:
                for pending_index in pending_tasks[current_index]:
                    self.try_firing(pending_index, current_index, task_sequence, fired_tasks, pending_tasks,
                                    p_state, f_arcs_frequency, fired_or_splits)

        # Firing End Event
        # print("CASE ENDED\tCASE ENDED\tCASE ENDED\tCASE ENDED\tCASE ENDED\tCASE ENDED\t\n\n")
        enabled_end, or_fired, path_decisions = self._find_enabled_predecessors(
            self.element_info[self.end_event], p_state)
        self.fire_enabled_predecessors(enabled_end, p_state, or_fired, path_decisions, f_arcs_frequency,
                                       fired_or_splits)
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
        return is_correct, fired_tasks, p_state.pending_tokens(), task_enabling

    def get_gateway_states(self):
        return self.gateway_states

    def check_unfired_or_splits(self, or_splits, f_arcs_frequency, p_state):
        # print(f_arcs_frequency)
        for or_id in or_splits:
            for flow_id in self.element_info[or_id].outgoing_flows:
                if p_state.tokens[flow_id] > 0:
                    # pprint.pprint(self.gateway_states[or_id]['decisions'])
                    # print(f"\t\t\t\t\t\t\t\t DEAD FLOW: {flow_id}")
                    self.gateway_states[or_id]['decisions'][-1].remove(flow_id)
                    f_arcs_frequency[flow_id] -= p_state.tokens[flow_id]
                    p_state.tokens[flow_id] = 0

    def postprocess_unfired_tasks(self, task_sequence: list, fired_tasks: list, f_arcs_frequency: dict,
                                  task_enablement: list):
        if self.closest_distance is None:
            self._sort_by_closest_predecesors()
        task_sequence = [task_name for task_name in task_sequence if task_name in self.from_name]
        for i in range(0, len(fired_tasks)):
            if not fired_tasks[i]:
                e_info = self.element_info[self.from_name.get(task_sequence[i])]
                fix_from = [self.starting_event, self.closest_distance[e_info.id][self.starting_event]]
                j = i - 1
                while j >= 0:
                    p_info = self.element_info[self.from_name.get(task_sequence[j])]
                    if p_info.id in self.closest_distance[e_info.id] and self.closest_distance[e_info.id][p_info.id] < \
                            fix_from[1]:
                        fix_from = [p_info.id, self.closest_distance[e_info.id][p_info.id]]
                        if fix_from[1] == 1:
                            break
                    j -= 1
                if fix_from[0] is not None:
                    if task_enablement[i] is None:
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
                        if pred_info.type in [BPMN.TASK, BPMN.START_EVENT]:
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
                        if p_info.type in [BPMN.INCLUSIVE_GATEWAY, BPMN.EXCLUSIVE_GATEWAY] and p_info.is_split():
                            self.decision_flows_sortest_path[e_id][p_id].append(pred_seq[p_info.id])
                        p_info = self._get_successor(pred_seq[p_info.id])

    def try_firing_or_join(self, enabled_pred, p_state, or_firing, path_decisions, f_arcs_frequency):
        fired = set()
        or_firing_list = list()
        for or_join_id in or_firing:
            or_firing_list.append(or_join_id)
        for or_join_id in or_firing_list:
            if self.is_enabled(or_join_id, p_state) or not enabled_pred:
                fired.add(or_join_id)
                e_info = self.element_info[or_join_id]
                self._update_next(e_info.outgoing_flows[0], enabled_pred, p_state, or_firing, path_decisions,
                                  f_arcs_frequency)
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

    def _update_next(self, flow_id, enabled_pred, p_state, or_firing, path_decisions, f_arcs_frequency, from_or=False):
        if flow_id not in f_arcs_frequency:
            f_arcs_frequency[flow_id] = 1
        else:
            f_arcs_frequency[flow_id] += 1
        p_state.add_token(flow_id)
        if not from_or:
            next_info = self._get_successor(flow_id)
            if next_info.type is BPMN.PARALLEL_GATEWAY and self.is_enabled(next_info.id, p_state):
                enabled_pred.appendleft([next_info, None])
            elif next_info.id in path_decisions:
                if next_info.type is BPMN.INCLUSIVE_GATEWAY:
                    if next_info.is_split():
                        enabled_pred.appendleft([next_info, path_decisions[next_info.id]])
                    else:
                        if next_info.id not in or_firing:
                            or_firing[next_info.id] = 1
                else:
                    enabled_pred.appendleft([next_info, path_decisions[next_info.id]])

    def _find_enabled_predecessors(self, from_task_info, p_state):
        pred_info = self._get_predecessor(from_task_info.incoming_flows[0])
        max_dist = [0]
        closer_pred = self.closer_enabled_predecessors(pred_info, from_task_info.incoming_flows[0], dict(),
                                                       dict(), dict(), set(), p_state, 0,
                                                       max_dist)
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
                            if task_sequence[i] in self.decision_successors[split_id]:
                                return i
            return firing_index
        return from_index

    def is_conflicting_task(self, path_decisions, enabled_pred):
        conflicting_gateways = dict()
        is_conflicting = False
        for or_id in path_decisions:
            if self.element_info[or_id].type is BPMN.INCLUSIVE_GATEWAY and self.element_info[or_id].is_join():
                conflicting_gateways[or_id] = set()
                for enabled in enabled_pred:
                    e_info = enabled[0]
                    if e_info.id in self.or_join_conflicting_pred[or_id]:
                        conflicting_gateways[or_id].add(e_info.id)
                    if len(conflicting_gateways[or_id]) > 1:
                        is_conflicting = True
        return is_conflicting, conflicting_gateways

    def closer_enabled_predecessors(self, e_info, flow_id, enabled_pred, or_firing, path_split, visited, p_state, dist,
                                    min_dist):
        # Verificar que hacer cuando el camino más corto termina en el Start event,
        # pero hay una tarea habilitada en un camino más largo
        if self.is_enabled(e_info.id, p_state):
            # if e_info.type is BPMN.START_EVENT:
            #     dist = len(self.flow_arcs)
            if dist not in enabled_pred:
                enabled_pred[dist] = list()
            enabled_pred[dist].append([e_info, flow_id])
            min_dist[0] = max(min_dist[0], dist)
            return dist, enabled_pred, or_firing, path_split
        elif e_info.type is BPMN.INCLUSIVE_GATEWAY and e_info.is_join():
            for in_or in e_info.incoming_flows:
                if p_state.has_token(in_or):
                    or_firing[e_info.id] = dist
                    break
        if e_info.type in [BPMN.INCLUSIVE_GATEWAY, BPMN.EXCLUSIVE_GATEWAY]:
            path_split[e_info.id] = flow_id
        visited.add(e_info.id)
        if e_info.is_gateway():
            if e_info.type is BPMN.EXCLUSIVE_GATEWAY and e_info.is_join():
                closer_pred, temp_path, or_f = dict(), dict(), dict()
                c_min = sys.maxsize
                for in_flow in e_info.incoming_flows:
                    pr_info = self._get_predecessor(in_flow)
                    if pr_info.id not in visited:
                        d, e_p, o_f, t_path = self.closer_enabled_predecessors(pr_info, in_flow, dict(), dict(), dict(),
                                                                               visited, p_state, dist + 1, min_dist)

                        if d < c_min or (BPMNGraph._has_start_event(closer_pred) and d != sys.maxsize):
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
                        res = self.closer_enabled_predecessors(pred_info, in_flow, enabled_pred, or_firing, path_split,
                                                               visited, p_state, dist + 1, min_dist)
                        c_min = min(res[0], c_min)

                return c_min, enabled_pred, or_firing, path_split
        return sys.maxsize, enabled_pred, or_firing, path_split

    @staticmethod
    def _has_start_event(closer_pred):
        has_start_event = False
        for e_id in closer_pred:
            for e_info in closer_pred[e_id]:
                if e_info[0].type is BPMN.START_EVENT:
                    has_start_event = True
                    break
        return has_start_event

    def _get_predecessor(self, flow_id):
        return self.element_info[self.flow_arcs[flow_id][0]]

    def _get_successor(self, flow_id):
        return self.element_info[self.flow_arcs[flow_id][1]]

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

    def add_flow_arc(self, flow_id, source_id, target_id):
        for node_id in [source_id, target_id]:
            if node_id not in self.element_info:
                self.element_info[node_id] = ElementInfo(BPMN.UNDEFINED, node_id, node_id, None)
        self.element_info[source_id].outgoing_flows.append(flow_id)
        self.element_info[target_id].incoming_flows.append(flow_id)
        self.flow_arcs[flow_id] = [source_id, target_id]
        self.arcs_bitset[flow_id] = (1 << len(self.flow_arcs))

    def is_enabled(self, e_id, p_state):
        if e_id not in self.element_info:
            return False
        if e_id == self.starting_event:
            return True
        e_info = self.element_info[e_id]
        if e_info.type == BPMN.TASK:
            # not enabled in case there is no token before the task
            # we assume that there might be only one incoming flow to the task
            for f_arc in e_info.incoming_flows:
                if p_state.tokens[f_arc] < 1:
                    return False

            return True
        if e_info.type in [BPMN.END_EVENT, BPMN.PARALLEL_GATEWAY, BPMN.INTERMEDIATE_EVENT]:
            for f_arc in e_info.incoming_flows:
                if p_state.tokens[f_arc] < 1:
                    return False
            return True
        elif e_info.type in [BPMN.EXCLUSIVE_GATEWAY, BPMN.EVENT_BASED_GATEWAY]:
            for f_arc in e_info.incoming_flows:
                if p_state.tokens[f_arc] > 0:
                    return True
            return False
        elif e_info.type == BPMN.INCLUSIVE_GATEWAY:
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

    def add_bpmn_element(self, element_id, element_info):
        if element_info.type == BPMN.START_EVENT:
            self.starting_event = element_id
        if element_info.type == BPMN.END_EVENT:
            self.end_event = element_id
            self.end_events_count += 1
        self.element_info[element_id] = element_info
        self.from_name[element_info.name] = element_id
        self.nodes_bitset[element_id] = (1 << len(self.element_info))
        self.last_datetime[element_id] = dict()

    def encode_or_join_predecesors(self):
        for e_id in self.element_info:
            element = self.element_info[e_id]
            if element.type is BPMN.INCLUSIVE_GATEWAY and element.is_join():
                self.or_join_pred[e_id] = [0, 0]
                self._find_or_conflicting_predecesors(e_id)
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
            if element.type in [BPMN.EXCLUSIVE_GATEWAY, BPMN.INCLUSIVE_GATEWAY] and element.is_split():
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
                    if next_info.type is BPMN.TASK:
                        self.decision_successors[split_info.id].add(next_info.id)
                    elif next_info.is_gateway():
                        suc_queue.append(next_info)

    def validate_model(self):
        if self.end_events_count == 0:
            raise InvalidBpmnModelException("At least one end event is required")
        if self.end_events_count > 1:
            raise InvalidBpmnModelException("Temporarily not supporting multiple end events")

    def _find_or_conflicting_predecesors(self, or_join_id):
        visited = {or_join_id}
        self.or_join_conflicting_pred[or_join_id] = set()
        for in_flow in self.element_info[or_join_id].incoming_flows:
            self._dfs_from_or_join(or_join_id, in_flow, self._get_predecessor(in_flow), visited)

    def _dfs_from_or_join(self, or_id, flow_id, e_info, visited):
        visited.add(e_info.id)
        if e_info.type in [BPMN.INCLUSIVE_GATEWAY, BPMN.EXCLUSIVE_GATEWAY] and e_info.is_split():
            self.or_join_conflicting_pred[or_id].add(e_info.id)
        for in_flow in e_info.incoming_flows:
            prev_info = self._get_predecessor(in_flow)
            if prev_info.id not in visited and prev_info.is_gateway():
                self._dfs_from_or_join(or_id, flow_id, prev_info, visited)


def parse_dataframe(df, log_ids, avoid_columns=[]):
    log_traces = list()
    trace_map = dict()

    for index, row in df.iterrows():
        case_id = str(row[log_ids.case])
        if case_id not in trace_map:
            trace_map[case_id] = len(log_traces)
            log_traces.append(CSVTrace(case_id))

        attributes = row.to_dict()

        for col in [log_ids.case, log_ids.activity, log_ids.start_time, log_ids.end_time,
                    log_ids.resource] + avoid_columns:
            attributes.pop(col, None)

        log_traces[trace_map[case_id]].add_event(
            activity=str(row[log_ids.activity]),
            state="start",
            resource=str(row[log_ids.resource]),
            timestamp=pd.to_datetime(row[log_ids.start_time], utc=True),
            attributes=attributes
        )

        log_traces[trace_map[case_id]].add_event(
            activity=str(row[log_ids.activity]),
            state="complete",
            resource=str(row[log_ids.resource]),
            timestamp=pd.to_datetime(row[log_ids.end_time], utc=True),
            attributes=attributes
        )

    for trace in log_traces:
        trace.events.sort(key=lambda x: x["time:timestamp"])

    return log_traces


def parse_csv(log_path, avoid_columns=[]):
    try:
        with open(log_path, mode="r") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            headers = next(csv_reader)  # Read the first row to use as headers
            i_map = process_csv_header(headers)
            log_traces = list()
            trace_map = dict()

            for row in csv_reader:
                case_id = row[i_map["case"]]
                if case_id not in trace_map:
                    trace_map[case_id] = len(log_traces)
                    log_traces.append(CSVTrace(case_id))

                attributes = {headers[index]: value for index, value in enumerate(row)
                              if headers[index] not in i_map
                              and headers[index] not in avoid_columns}

                log_traces[trace_map[case_id]]\
                    .add_event(
                        row[i_map["activity"]],
                        "start",
                        row[i_map["resource"]],
                        pd.to_datetime(row[i_map["start"]], utc=True),
                        attributes
                    )

                log_traces[trace_map[case_id]]\
                    .add_event(
                        row[i_map["activity"]],
                        "complete",
                        row[i_map["resource"]],
                        pd.to_datetime(row[i_map["end"]], utc=True),
                        attributes
                    )

            for trace in log_traces:
                trace.events.sort(key=lambda x: x["time:timestamp"])
            return log_traces
    except IOError as e:
        print(str(e))
        return list()


def process_csv_header(first_row):
    i_map = dict()
    i = 0
    key_words = ["case", "activity", "start", "end", "resource"]
    for key in first_row:
        l_key = key.lower()
        for kw in key_words:
            if kw in l_key:
                i_map[kw] = i
                break
        i += 1

    for key in key_words:
        if key not in i_map:
            raise InvalidLogFileException("%s column missing in the CSV file." % key)

    return i_map


class CSVTrace:
    def __init__(self, case_id):
        self.attributes = {"concept:name": case_id}
        self.events = list()

    def add_event(self, activity, state, resource, timestamp, attributes=None):
        if attributes is None:
            attributes = {}

        self.events.append(
            {
                "concept:name": activity.strip(),
                "elementId": activity.strip(),
                "org:resource": resource.strip(),
                "lifecycle:transition": state.strip(),
                "time:timestamp": timestamp,
                "attributes": attributes
            }
        )

    def __iter__(self):
        return CSVTraceIterator(self.events)


class CSVTraceIterator:
    def __init__(self, events):
        self._events = events
        self._index = -1

    def __next__(self):
        self._index += 1
        if self._index < len(self._events):
            return self._events[self._index]
        raise StopIteration


class EnabledEvent:
    def __init__(self, p_case, p_state, task_id, enabled_at, enabled_datetime, duration_sec = None, is_inter_event = False):
        self.p_case = p_case
        self.p_state = p_state
        self.task_id = task_id
        self.enabled_datetime = enabled_datetime
        self.enabled_at = enabled_at
        self.duration_sec = duration_sec        # filled only in case of event-based gateway
        self.is_inter_event = is_inter_event    # whether the enabled event is the intermediate event


class ProcessInfo:
    def __init__(self):
        self.traces = dict()
        self.resource_profiles = dict()


class TaskEvent:
    def __init__(self, p_case, task_id, resource_id, resource_available_at=None,
        enabled_at=None, enabled_datetime=None, bpm_env=None, num_tasks_in_batch=0, attributes=None):
        self.p_case = p_case  # ID of the current trace, i.e., index of the trace in log_info list
        self.task_id = task_id  # Name of the task related to the current event
        self.type = BPMN.TASK # showing whether it's task or event
        self.resource_id = resource_id  # ID of the resource performing to the event
        self.waiting_time = None
        self.processing_time = None
        self.normalized_waiting = None
        self.normalized_processing = None
        self.worked_intervals = []
        self.attributes = attributes or {}

        if resource_available_at is not None:
            # Time moment in seconds from beginning, i.e., first event has time = 0
            self.enabled_at = enabled_at
            # Datetime of the time-moment calculated from the starting simulation datetime
            self.enabled_datetime = enabled_datetime

            # Time moment in seconds from beginning, i.e., first event has time = 0

            self.started_at = max(resource_available_at, enabled_at)
            # Datetime of the time-moment calculated from the starting simulation datetime
            self.started_datetime = bpm_env.simulation_datetime_from(self.started_at)

            # Ideal duration from the distribution-function if allocate resource doesn't rest
            self.ideal_duration = bpm_env.sim_setup.ideal_task_duration(task_id, resource_id, num_tasks_in_batch)
            # Actual duration adding the resource resting-time according to their calendar
            self.real_duration = bpm_env.sim_setup.real_task_duration(self.ideal_duration, self.resource_id,
                                                                      self.started_datetime, self.worked_intervals)

            # Time moment in seconds from beginning, i.e., first event has time = 0
            self.completed_at = self.started_at + self.real_duration
            # Datetime of the time-moment calculated from the starting simulation datetime
            self.completed_datetime = bpm_env.simulation_datetime_from(self.completed_at)

            # Time of a resource was resting while performing a task (in seconds)
            self.idle_time = self.real_duration - self.ideal_duration
            # Time from an event is enabled until it is started by any resource
            self.waiting_time = self.started_at - self.enabled_at
            self.idle_cycle_time = self.completed_at - self.enabled_at
            self.idle_processing_time = self.completed_at - self.started_at
            self.cycle_time = self.idle_cycle_time - self.idle_time
            self.processing_time = self.idle_processing_time - self.idle_time
        else:
            self.task_name = None
            self.enabled_at = enabled_at
            self.enabled_by = None
            self.started_at = None
            self.completed_at = None
            self.idle_time = None

    @classmethod
    def create_event_entity(cls, c_event: EnabledEvent, ended_at, ended_datetime):
        cls.p_case = c_event.p_case  # ID of the current trace, i.e., index of the trace in log_info list
        cls.task_id = c_event.task_id  # Name of the task related to the current event
        cls.type = BPMN.INTERMEDIATE_EVENT
        cls.enabled_at = c_event.enabled_at
        cls.enabled_datetime = c_event.enabled_datetime
        cls.started_at = c_event.enabled_at
        cls.started_datetime = c_event.enabled_datetime
        cls.completed_at = ended_at
        cls.completed_datetime = ended_datetime
        cls.idle_time = 0.0
        cls.waiting_time = 0.0
        cls.idle_cycle_time = 0.0
        cls.idle_processing_time = 0.0
        cls.cycle_time = 0.0
        cls.processing_time = 0.0

        return cls

    def update_enabling_times(self, enabled_at):
        # what's the use case ?
        if self.started_at is None or enabled_at > self.started_at:
            # print(self.task_id)
            # print(str(enabled_at))
            # print(str(self.started_at))
            # print("--------------------------------------------")
            enabled_at = self.started_at
            # raise Exception("Task ENABLED after STARTED")
        self.enabled_at = enabled_at
        self.waiting_time = (self.started_at - self.enabled_at).total_seconds()
        self.processing_time = (self.completed_at - self.started_at).total_seconds()


class LogEvent:
    def __int__(self, task_id, started_datetime, resource_id):
        self.task_id = task_id
        self.started_datetime = started_datetime
        self.resource_id = resource_id
        self.completed_datetime = None


class Trace:
    def __init__(self, p_case, started_at=datetime.datetime(9999, 12, 31, 23, 59, 59, 999999, pytz.utc)):
        self.p_case = p_case
        self.started_at = started_at
        self.completed_at = started_at
        self.event_list = list()

        self.cycle_time = None
        self.idle_cycle_time = None
        self.processing_time = None
        self.idle_processing_time = None
        self.waiting_time = None
        self.idle_time = None

    def start_event(self, task_id, task_name, started_at, resource_name, attributes=None):
        event_info = TaskEvent(self.p_case, task_id, resource_name, attributes=attributes)
        event_info.task_name = task_name
        event_info.started_at = started_at
        event_index = len(self.event_list)
        self.event_list.append(event_info)
        self.started_at = min(self.started_at, started_at)
        return event_index

    def complete_event(self, event_index, completed_at, idle_time=0, attributes=None):
        if attributes:
            self.event_list[event_index].attributes.update(attributes)
        self.event_list[event_index].completed_at = completed_at
        self.event_list[event_index].idle_time = idle_time
        self.completed_at = max(self.completed_at, self.event_list[event_index].completed_at)
        return self.event_list[event_index]

    def sort_by_completion_date(self, completed_at=False):
        if completed_at:
            self.event_list.sort(key=lambda e_info: e_info.completed_at)
        else:
            self.event_list.sort(key=lambda e_info: e_info.started_at)
        self.started_at = self.event_list[0].started_at
        self.completed_at = self.event_list[len(self.event_list) - 1].completed_at

    def filter_incomplete_events(self):
        filtered_list = list()
        filtered_events = 0
        for ev_info in self.event_list:
            if ev_info.started_at is not None and ev_info.completed_at is not None:
                filtered_list.append(ev_info)
            else:
                filtered_events += 2
        self.event_list = filtered_list
        return filtered_events


class Error(Exception):
    """Base class for other exceptions"""
    pass


class InvalidBpmnModelException(Error):
    """Raised when the provided BPMN model is invalid"""


class InvalidLogFileException(Error):
    """Raised when the provided log file is invalid"""
