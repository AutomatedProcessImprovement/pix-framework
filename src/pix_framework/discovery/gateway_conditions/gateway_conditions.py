import m5py.main
from pandas import DataFrame
from pix_framework.statistics.distribution import get_best_fitting_distribution

import pandas as pd
import numpy as np
import json
from pandas.api.types import is_numeric_dtype
from pix_framework.io.event_log import EventLogIDs
import time
import functools
import pprint
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.tree import BaseDecisionTree, _tree
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from niaarm.text import Corpus
from niaarm.mine import get_text_rules
from niaarm import Dataset, get_rules
from niapy.algorithms.basic import DifferentialEvolution, ParticleSwarmOptimization
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from scipy.stats import median_abs_deviation
import xml.etree.ElementTree as ET
from sklearn import tree
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from imodels import RuleFitClassifier
from imodels import RuleFitRegressor
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, hamming_loss, f1_score
from sklearn.model_selection import GridSearchCV
from bpmn_parser import parse_simulation_model
from replayer import parse_csv, Trace, parse_dataframe
from replayer import BPMNGraph
import pytz
import sys
from datetime import datetime
from sklearn.preprocessing import MultiLabelBinarizer



from m5py import M5Prime, export_text_m5

import logging

import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

namespaces = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}

DISCRETE_ERROR_RATIO = 0.95
DEFAULT_SAMPLING_SIZE = 1000


def discover_gateway_conditions(bpmn_model_path,
                                event_log_path,
                                log_ids: EventLogIDs,
                                sampling_size: int = DEFAULT_SAMPLING_SIZE):
    avoid_columns = [
        log_ids.case, log_ids.activity, log_ids.start_time,
        log_ids.end_time, log_ids.resource, log_ids.enabled_time
    ]
    bpmn_graph = parse_simulation_model(bpmn_model_path)

    bpmn_model_tree = ET.parse(bpmn_model_path)
    bpmn_model = bpmn_model_tree.getroot()

    event_log = pd.read_csv(event_log_path)
    log_by_case = preprocess_event_log(event_log, log_ids, sampling_size)

    activity_id_map = map_activity_names_to_ids(bpmn_model, namespaces) # Will be used in the future

    print("Parsing log traces")
    start_time = time.time()
    log_traces = parse_dataframe(log_by_case, log_ids, avoid_columns)
    end_time = time.time()
    print(f"Completed parsing in {end_time - start_time} seconds.")

    flow_arcs_frequency = dict()
    gateway_states = process_traces(log_traces, bpmn_graph, flow_arcs_frequency)

    branching_probabilities = discover_joins(gateway_states)

    dataframes = gateways_to_dataframes(gateway_states)

    print("=======================================================")
    print("=======================================================")
    print("=======================================================")
    print("=======================================================")
    print("=======================================================")
    print("=======================================================")

    dataframes, gateway_encoders = encode_dataframes(dataframes)

    for gateway_id, df in dataframes.items():
        print(f"DataFrame for {gateway_id}:")
        print(df, "\n")

    gateway_rules = discover_gateway_models(dataframes)
    display_gateway_rules(gateway_rules)

    filtered_gateway_rules = filter_true_outcomes(gateway_rules)
    display_gateway_rules(filtered_gateway_rules)

    simplified_gateway_rules = simplify_rules(filtered_gateway_rules)
    display_gateway_rules(simplified_gateway_rules)


    pprint.pprint(branching_probabilities)

    branch_rules = convert_to_branch_rules(simplified_gateway_rules)

    pprint.pprint(branch_rules)
    print(flow_arcs_frequency)

    n = calculate_gateway_branching_probabilities(simplified_gateway_rules, flow_arcs_frequency)
    pprint.pprint(n)
    pprint.pprint(branching_probabilities)

    branching_probabilities.extend(n)
    pprint.pprint(branching_probabilities)

    result = {"branching_probabilities": branching_probabilities, "branch_rules": branch_rules}
    pprint.pprint(result)
    return result


def calculate_gateway_branching_probabilities(simplified_gateway_rules, flow_arcs_frequency):
    gateway_branching_probabilities = []

    for gateway_id, flows in simplified_gateway_rules.items():
        probabilities = []
        total_executions = sum(flow_arcs_frequency[flow_id] for flow_id in flows.keys())
        rounding_error = 0.0

        flow_ids = list(flows.keys())
        for i, flow_id in enumerate(flow_ids):
            execution_count = flow_arcs_frequency[flow_id]
            probability = execution_count / total_executions

            if i < len(flow_ids) - 1:
                adjusted_probability = round(probability + rounding_error, 2)
                rounding_error += (probability - adjusted_probability)
            else:
                adjusted_probability = round(1 - sum([float(p['value']) for p in probabilities]), 2)

            probabilities.append({
                "path_id": flow_id,
                "value": str(adjusted_probability),
                "condition_id": flow_id
            })

        gateway_branching_probabilities.append({
            "gateway_id": gateway_id,
            "probabilities": probabilities
        })

    return gateway_branching_probabilities


def convert_to_branch_rules(gateway_analysis_results):
    branch_rules = []

    for gateway_id, flows in gateway_analysis_results.items():
        for flow_id, conditions_list in flows.items():
            formatted_conditions = []

            for condition_set in conditions_list:
                # Check if the outcome is True, then process
                if condition_set[1] == 1:
                    inner_conditions = []
                    for condition in condition_set[0]:
                        attr, operator, value = condition
                        formatted_condition = {"attribute": attr, "comparison": operator, "value": str(value)}
                        inner_conditions.append(formatted_condition)
                    formatted_conditions.append(inner_conditions)

            branch_rule = {
                "id": flow_id,
                "rules": formatted_conditions
            }
            branch_rules.append(branch_rule)

    return branch_rules


def simplify_rules(gateway_analysis_results):
    simplified_results = {}
    for gateway_id, flows in gateway_analysis_results.items():
        simplified_flows = {}
        for flow_id, conditions in flows.items():
            simplified_conditions = []
            for condition_set in conditions:
                condition_rules, outcome = condition_set
                if len(condition_rules) > 1 and outcome == 1:
                    simplified_condition = simplify_rule(condition_rules)
                    if simplified_condition:
                        simplified_conditions.append((simplified_condition, 1))
                else:
                    simplified_conditions.append(condition_set)
            simplified_flows[flow_id] = simplified_conditions
        simplified_results[gateway_id] = simplified_flows
    return simplified_results


def simplify_rule(rules):
    grouped_conditions = {}
    for attr, op, value in rules:
        if attr not in grouped_conditions:
            grouped_conditions[attr] = []
        grouped_conditions[attr].append((op, value))

    simplified_conditions = []
    for attr, ops_values in grouped_conditions.items():
        greater_than_values = [v for op, v in ops_values if op == '>']
        less_than_or_equal_values = [v for op, v in ops_values if op == '<=']
        if greater_than_values:
            max_greater_than = max(greater_than_values)
            simplified_conditions.append((attr, '>', max_greater_than))
        if less_than_or_equal_values:
            min_less_than_or_equal = min(less_than_or_equal_values)
            simplified_conditions.append((attr, '<=', min_less_than_or_equal))

    return simplified_conditions


def filter_true_outcomes(gateway_analysis_results):
    filtered_results = {}
    for gateway_id, flows in gateway_analysis_results.items():
        filtered_flows = {}
        for flow_id, rules in flows.items():
            true_rules = [rule for rule in rules if rule[1] == 1]
            if true_rules:
                filtered_flows[flow_id] = true_rules
        if filtered_flows:
            filtered_results[gateway_id] = filtered_flows
    return filtered_results


def display_gateway_rules(gateway_rules):
    for gateway_id, flows in gateway_rules.items():
        print(f"\nGateway ID: {gateway_id}")
        for flow_id, rules in flows.items():
            print(f"  Flow ID: {flow_id}")
            for rule in rules:
                print(rule)


def extract_rules(tree, feature_names):
    def recurse(node, rules, current_rule):
        if tree.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_names[tree.feature[node]]
            threshold = tree.threshold[node]

            left_rule = current_rule + [(name, "<=", threshold)]
            recurse(tree.children_left[node], rules, left_rule)

            right_rule = current_rule + [(name, ">", threshold)]
            recurse(tree.children_right[node], rules, right_rule)
        else:
            outcome = np.argmax(tree.value[node])
            rules.append((current_rule, outcome))

    rules = []
    recurse(0, rules, [])
    return rules


def discover_gateway_models(dataframes):
    gateway_analysis_results = {}
    for gateway_id, df in dataframes.items():
        print(f"\nProcessing Gateway: {gateway_id}")

        feature_columns = [col for col in df.columns if not col.startswith("Flow_")]
        target_columns = [col for col in df.columns if col.startswith("Flow_")]

        models_results = {}
        for target_col in target_columns:
            features = df[feature_columns]
            target = df[target_col]

            ccp_alphas = np.linspace(0.0001, 0.01, 20)
            tree = DecisionTreeClassifier(random_state=42, max_depth=5)
            grid_search = GridSearchCV(estimator=tree, param_grid={'ccp_alpha': ccp_alphas}, cv=5)
            grid_search.fit(features, target)

            best_tree = grid_search.best_estimator_

            rules = extract_rules(best_tree.tree_, feature_names=best_tree.feature_names_in_)

            models_results[target_col] = rules

        gateway_analysis_results[gateway_id] = models_results

    return gateway_analysis_results


def encode_dataframes(dataframes):
    gateway_encoders = {}

    for gateway_id, df in dataframes.items():
        encoded_df = df.copy()
        encoders = {}

        target_columns = [col for col in df.columns if col.startswith("Flow_")]

        for col in df.columns:
            if col not in target_columns and df[col].dtype == 'object':
                encoder = LabelEncoder()
                encoded_df[col] = encoder.fit_transform(df[col])
                encoders[col] = encoder

        dataframes[gateway_id] = encoded_df
        gateway_encoders[gateway_id] = encoders

    return dataframes, gateway_encoders


def gateways_to_dataframes(gateway_states):
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

        dataframes[gateway_id] = df

    return dataframes


def discover_joins(gateway_states):
    gateway_branching_probabilities = []

    for gateway_id, details in list(gateway_states.items()):
        if len(details['outgoing_flows']) == 1:
            path_id = details['outgoing_flows'][0]
            gateway_branching_probabilities.append({
                "gateway_id": gateway_id,
                "probabilities": [{"path_id": path_id, "value": "1"}]
            })
            del gateway_states[gateway_id]

    return gateway_branching_probabilities


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
        # print(f"Trace #{num}: {trace}")
        num += 1
        caseid = trace.attributes["concept:name"]
        total_traces += 1
        started_events = dict()
        trace_info = Trace(caseid)
        initial_events[caseid] = datetime(9999, 12, 31, tzinfo=pytz.UTC)
        for event in trace:
            # print(event)
            attributes = event.get("attributes", {})  # Assuming attributes are stored under the 'attributes' key
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

        # print(f"\nTrace #{num}: {trace}")
        # for event in trace_info.event_list:
        #     print(event.attributes)

        is_correct, fired_tasks, pending_tokens, _, gateway_states = bpmn_graph.reply_trace(
            task_sequence, flow_arcs_frequency, True, trace_info.event_list
        )

    return gateway_states


def map_activity_names_to_ids(bpmn_model, namespaces):
    activity_id_map = {}
    for task in bpmn_model.findall('.//bpmn:task', namespaces):
        activity_name = task.get('name')
        activity_id = task.get('id')
        activity_id_map[activity_name] = activity_id
    return activity_id_map


def preprocess_event_log(event_log, log_ids, sampling_size=DEFAULT_SAMPLING_SIZE):
    sorted_log = event_log.sort_values(by=log_ids.end_time)

    # Drop unnecessary columns except for essential system columns
    # columns_to_drop = [getattr(log_ids, attr) for attr in vars(log_ids) if getattr(log_ids, attr) in sorted_log.columns]
    # system_columns_to_keep = [log_ids.case, log_ids.activity]
    # columns_to_drop = [x for x in columns_to_drop if x not in system_columns_to_keep]
    #
    # sorted_log = sorted_log.drop(columns=columns_to_drop)

    # Sample the event log by case and fill NaN values
    log_by_case = sample_event_log_by_case(sorted_log, log_ids, sampling_size)
    log_by_case = fill_nans(log_by_case, log_ids)

    string_values = list()
    for col in log_by_case.columns:
        if log_by_case[col].dtype == 'object':
            string_values.append(col)

    log_by_case = scale_data(log_by_case, string_values)

    return log_by_case


def sample_event_log_by_case(event_log: pd.DataFrame, log_ids: EventLogIDs, sampling_size: int = DEFAULT_SAMPLING_SIZE):
    event_log_reset = event_log.reset_index()
    event_log_sorted = event_log_reset.sort_values(by=[log_ids.case, 'index'])

    unique_cases = event_log_sorted[log_ids.case].unique()

    if len(unique_cases) <= sampling_size:
        return event_log_sorted

    step_size = len(unique_cases) // sampling_size
    sampled_cases = unique_cases[::step_size]

    sampled_log = event_log_sorted[event_log_sorted[log_ids.case].isin(sampled_cases)]

    sampled_log = sampled_log.drop(columns=['index'])

    return sampled_log


def fill_nans(log, log_ids):
    def replace_initial_nans(series):
        if is_numeric_dtype(series):
            initial_value = 0
        else:
            initial_value = ''
        first_valid_index = series.first_valid_index()
        if first_valid_index is not None:
            series[:first_valid_index] = series[:first_valid_index].fillna(initial_value)
        return series

    def preprocess_case(case_data):
        return case_data.apply(replace_initial_nans)

    preprocessed_log = (log.groupby(log_ids.case).apply(preprocess_case)).reset_index(drop=True)
    preprocessed_log = preprocessed_log.ffill()
    return preprocessed_log


def scale_data(log, encoded_columns, threshold=1e+37):
    log_scaled = log.copy()
    max_float = np.finfo(np.float32).max
    min_float = np.finfo(np.float32).tiny

    for col in log.columns:
        if col not in encoded_columns and log[col].dtype in ['float64', 'int64']:
            log_scaled.loc[log_scaled[col] > threshold, col] = max_float
            log_scaled.loc[log_scaled[col] < -threshold, col] = min_float

            log_scaled[col] = log_scaled[col].fillna(method='ffill')
    return log_scaled


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

    # TODO: check whether 'from_name' handles duplicated names of elements in the BPMN model
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