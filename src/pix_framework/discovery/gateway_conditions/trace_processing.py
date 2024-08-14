import sys
import pytz
import pandas as pd

from datetime import datetime
from sklearn.preprocessing import LabelEncoder

from pix_framework.discovery.gateway_conditions.helpers import log_time
from pix_framework.discovery.gateway_conditions.replayer import Trace
from pix_framework.io.bpm_graph import BPMNGraph

@log_time
def traces_to_dataframes(gateway_states):
    dataframes = {}

    for gateway_id, gateway_details in gateway_states.items():
        data = []
        outgoing_flows = gateway_details['outgoing_flows']
        decisions = gateway_details['decisions']

        for attributes, decision in zip(gateway_details['attributes'], decisions):
            row = attributes.copy()
            for flow in outgoing_flows:
                row[flow] = flow in decision
            data.append(row)

        df = pd.DataFrame(data)

        for flow in outgoing_flows:
            if flow not in df:
                df[flow] = False

        if 'index' in df.columns:
            df.drop('index', axis=1, inplace=True)
        dataframes[gateway_id] = df

    return dataframes


@log_time
def encode_dataframes(dataframes, prefixes):
    gateway_encoders = {}

    for gateway_id, df in dataframes.items():
        encoded_df = df.copy()
        encoders = {}

        target_columns = [col for col in df.columns if any(col.startswith(prefix) for prefix in prefixes)]

        for col in df.columns:
            if col not in target_columns and (df[col].dtype == 'object' or df[col].dtype == 'bool'):
                if df[col].dtype == bool:
                    encoded_df[col] = encoded_df[col].astype(str)

                encoded_df[col] = encoded_df[col].astype(str)

                all_values = pd.concat([pd.Series(['']), encoded_df[col]]).unique()

                encoder = LabelEncoder()
                encoder.fit(all_values)
                encoded_df[col] = encoder.transform(encoded_df[col])
                encoders[col] = encoder

        dataframes[gateway_id] = encoded_df
        gateway_encoders[gateway_id] = encoders
    return dataframes, gateway_encoders


@log_time
def process_traces(log_traces, bpmn_graph, flow_arcs_frequency):
    completed_events = list()
    total_traces = 0
    task_resource_freq = dict()
    initial_events = dict()
    min_date = None
    task_events = dict()
    observed_task_resources = dict()
    min_max_task_duration = dict()
    total_events = 0
    removed_traces = 0
    removed_events = 0
    gateway_states = []

    num = 0

    for trace in log_traces:
        num += 1
        caseid = trace.attributes["concept:name"]
        total_traces += 1
        started_events = dict()
        trace_info = Trace(caseid)
        initial_events[caseid] = datetime(9999, 12, 31, tzinfo=pytz.UTC)
        for event in trace:
            attributes = event.get("attributes", {})
            total_events += 1
            if is_trace_event_start_or_end(event, bpmn_graph):
                # trace event is a start or end event, we skip it for further parsing
                removed_events += 1
                continue
            if not is_event_in_bpmn_model(event, bpmn_graph):
                # event in the log does not match any task in the BPMN model
                removed_events += 1
                continue

            state = event["lifecycle:transition"].lower()
            timestamp = event["time:timestamp"]
            if min_date is None:
                min_date = timestamp
            min_date = min(min_date, timestamp)

            initial_events[caseid] = min(initial_events[caseid], timestamp)

            task_name = event["concept:name"]

            if task_name not in task_resource_freq:
                task_resource_freq[task_name] = [0, dict()]
                task_events[task_name] = list()
                observed_task_resources[task_name] = set()
                min_max_task_duration[task_name] = [sys.float_info.max, 0]

            if state in ["start", "assign"]:
                started_events[task_name] = trace_info.start_event(task_name, task_name, timestamp, "None",
                                                                   attributes=attributes)
            elif state == "complete":
                if task_name in started_events:
                    c_event = trace_info.complete_event(started_events.pop(task_name), timestamp,
                                                        attributes=attributes)
                    task_events[task_name].append(c_event)
                    completed_events.append(c_event)
                    duration = (c_event.completed_at - c_event.started_at).total_seconds()
                    min_max_task_duration[task_name][0] = min(min_max_task_duration[task_name][0], duration)
                    min_max_task_duration[task_name][1] = max(min_max_task_duration[task_name][1], duration)

        removed_events += trace_info.filter_incomplete_events()
        if len(trace_info.event_list) == 0:
            removed_traces += 1
            continue

        task_sequence = sort_by_completion_times(trace_info)

        bpmn_graph.replay_trace(task_sequence, flow_arcs_frequency, True, trace_info.event_list)
        gateway_states = bpmn_graph.get_gateway_states()

    return gateway_states


def is_trace_event_start_or_end(event, bpmn_graph: BPMNGraph):
    """Check whether the trace event is start or end event"""

    element_id = get_element_id_from_event_info(event, bpmn_graph)

    if element_id == "":
        print("WARNING: Trace event could not be mapped to the BPMN element.")
        return False
    elif element_id in [bpmn_graph.starting_event, bpmn_graph.end_event]:
        return True

    return False


def get_element_id_from_event_info(event, bpmn_graph: BPMNGraph):
    original_element_id = event.get("elementId", "")
    task_name = event.get("concept:name", "")

    if original_element_id != "" and original_element_id != task_name:
        # when log file is in CSV format, then task_name == original_element_id
        # and they both equals to task name
        return original_element_id

    element_id = bpmn_graph.from_name.get(task_name, "")
    return element_id


def is_event_in_bpmn_model(event, bpmn_graph: BPMNGraph):
    """Check whether the task name in the event matches a task in the BPMN process model"""

    return True if event["concept:name"] in bpmn_graph.from_name else False


def sort_by_completion_times(trace_info: Trace):
    trace_info.sort_by_completion_date(False)
    task_sequence = list()
    for e_info in trace_info.event_list:
        task_sequence.append(e_info.task_id)
    return task_sequence