import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from pix_framework.io.event_log import EventLogIDs
import time
import pprint
from sklearn.tree import _tree
from sklearn.preprocessing import LabelEncoder
import xml.etree.ElementTree as ET

from bpmn_parser import parse_simulation_model
from replayer import Trace, parse_dataframe
from replayer import BPMNGraph
import pytz
import sys
from datetime import datetime

import optuna
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import logging

import warnings

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

namespaces = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}

DEFAULT_SAMPLING_SIZE = 25000


def log_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper


@log_time
def discover_gateway_conditions(bpmn_model_path,
                                event_log_path,
                                log_ids: EventLogIDs,
                                sampling_size: int = DEFAULT_SAMPLING_SIZE):
    avoid_columns = [
        log_ids.case, log_ids.activity, log_ids.start_time,
        log_ids.end_time, log_ids.resource, log_ids.enabled_time
    ]

    bpmn_graph = parse_simulation_model(bpmn_model_path)
    flow_prefixes = ["Flow_", "edge", "node"]
    branching_probabilities = discover_joins(bpmn_graph)

    event_log = pd.read_csv(event_log_path)
    log_by_case = preprocess_event_log(event_log, log_ids, sampling_size)

    log_traces = parse_dataframe(log_by_case, log_ids, avoid_columns)

    flow_arcs_frequency = dict()
    gateway_states = process_traces(log_traces, bpmn_graph, flow_arcs_frequency)

    dataframes = gateways_to_dataframes(gateway_states)
    dataframes, gateway_encoders = encode_dataframes(dataframes, flow_prefixes)

    true_or_flows = discover_true_or_flows(gateway_states, dataframes, flow_prefixes)
    gateway_rules = discover_gateway_models(dataframes, flow_prefixes)

    filtered_gateway_rules = filter_true_outcomes(gateway_rules)

    simplified_gateway_rules = simplify_rules(filtered_gateway_rules)

    adjusted_and_decoded_rules = adjust_and_decode_conditions(simplified_gateway_rules, gateway_encoders)
    adjusted_and_decoded_rules = simplify_rules(adjusted_and_decoded_rules)

    branch_rules = extract_branch_rules(adjusted_and_decoded_rules)
    branch_rules.extend(true_or_flows)

    branching_probabilities_with_rules = calculate_gateway_branching_probabilities(adjusted_and_decoded_rules, flow_arcs_frequency, branch_rules)
    branching_probabilities.extend(branching_probabilities_with_rules)

    result = {"gateway_branching_probabilities": branching_probabilities, "branch_rules": branch_rules}

    return result


def discover_true_or_flows(gateway_states, dataframes, prefixes):
    branch_rules = []

    for gateway_id, gateway_info in gateway_states.items():
        if gateway_info['type'] == 'OR':
            df = dataframes[gateway_id]

            always_true_columns = [col for col in df.columns if any(col.startswith(prefix) for prefix in prefixes) and df[col].all()]
            non_flow_columns = [col for col in df.columns if not any(col.startswith(prefix) for prefix in prefixes)]

            for col in always_true_columns:
                if non_flow_columns:
                    synthetic_rule = {
                        "id": col,
                        "rules": [[{
                            "attribute": non_flow_columns[0],
                            "comparison": "!=",
                            "value": "WARNING_THIS_CONDITION_ALWAYS_EVALUATES_TO_TRUE"
                        }]]
                    }
                    branch_rules.append(synthetic_rule)
    return branch_rules


def adjust_and_decode_conditions2(conditions, encoders):
    adjusted_conditions = []
    for condition in conditions:
        attribute, operator, raw_value = condition
        if attribute in encoders:
            encoder = encoders[attribute]
            value = float(raw_value)
            if operator == '>':
                decoded_value = encoder.inverse_transform([int(value + 1)])[0]
            else:
                decoded_value = encoder.inverse_transform([int(value)])[0]
            adjusted_conditions.append((attribute, '=', decoded_value))
        else:
            adjusted_conditions.append((attribute, operator, raw_value))
    return adjusted_conditions


def adjust_and_decode_conditions(gateway_rules, gateway_encoders):
    adjusted_rules = {}
    for gateway_id, flows in gateway_rules.items():
        encoders = gateway_encoders.get(gateway_id, {})
        adjusted_rules[gateway_id] = {}
        for flow_id, conditions_outcomes_list in flows.items():
            adjusted_conditions_outcome = []
            for conditions_outcome_pair in conditions_outcomes_list:
                conditions, outcome = conditions_outcome_pair
                adjusted_conditions = adjust_and_decode_conditions2(conditions, encoders)
                adjusted_conditions_outcome.append((adjusted_conditions, outcome))
            adjusted_rules[gateway_id][flow_id] = adjusted_conditions_outcome
    return adjusted_rules


@log_time
def calculate_gateway_branching_probabilities(simplified_gateway_rules, flow_arcs_frequency, branch_rules):
    gateway_branching_probabilities = []
    branch_rules_lookup = {rule['id']: rule for rule in branch_rules}
    for gateway_id, flows in simplified_gateway_rules.items():
        probabilities = []
        total_executions = sum(flow_arcs_frequency.get(flow_id, 0) for flow_id in flows.keys())

        flow_ids = list(flows.keys())
        for i, flow_id in enumerate(flow_ids):
            execution_count = flow_arcs_frequency.get(flow_id, 0)
            probability = execution_count / total_executions if total_executions > 0 else 0

            adjusted_probability = str(round(probability, 2)) if i < len(flow_ids) - 1 else str(round(1 - sum(float(p['value']) for p in probabilities), 2))
            prob_entry = {
                "path_id": flow_id,
                "value": adjusted_probability
            }

            if flow_id in branch_rules_lookup:
                prob_entry["condition_id"] = flow_id

            probabilities.append(prob_entry)

        gateway_branching_probabilities.append({
            "gateway_id": gateway_id,
            "probabilities": probabilities
        })

    return gateway_branching_probabilities


@log_time
def extract_branch_rules(gateway_analysis_results):
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
                    if inner_conditions:  # Add only if there are conditions
                        formatted_conditions.append(inner_conditions)

            # Add branch rule only if there are any formatted conditions
            if formatted_conditions:
                branch_rule = {
                    "id": flow_id,
                    "rules": formatted_conditions
                }
                branch_rules.append(branch_rule)

    return branch_rules


@log_time
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
            simplified_flows[flow_id] = simplified_conditions if simplified_conditions else []
        simplified_results[gateway_id] = simplified_flows
    return simplified_results


def simplify_rule(rules):
    grouped_conditions = {}
    for attr, op, value in rules:
        if attr not in grouped_conditions:
            grouped_conditions[attr] = {'=': set(), '>': [], '<=': []}
        if op == '=':
            grouped_conditions[attr]['='].add(value)
        else:
            grouped_conditions[attr][op].append(value)

    simplified_conditions = []
    for attr, ops_values in grouped_conditions.items():
        for value in ops_values['=']:
            simplified_conditions.append((attr, '=', value))
        if ops_values['>']:
            max_greater_than = max(ops_values['>'])
            simplified_conditions.append((attr, '>', max_greater_than))
        if ops_values['<=']:
            min_less_than_or_equal = min(ops_values['<='])
            simplified_conditions.append((attr, '<=', min_less_than_or_equal))

    return simplified_conditions


@log_time
def filter_true_outcomes(gateway_analysis_results):
    filtered_results = {}
    for gateway_id, flows in gateway_analysis_results.items():
        filtered_flows = {}
        for flow_id, rules in flows.items():
            true_rules = [rule for rule in rules if rule[1] == 1]
            if true_rules:
                filtered_flows[flow_id] = true_rules
            else:
                filtered_flows[flow_id] = []
        if filtered_flows:
            filtered_results[gateway_id] = filtered_flows
    return filtered_results


def display_gateway_rules(gateway_rules):
    for gateway_id, flows in gateway_rules.items():
        print(f"\nGateway ID: {gateway_id}")
        for flow_id, rules in flows.items():
            print(f"\tFlow ID: {flow_id}")
            for rule in rules:
                print(f"\t\t{rule}")


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


def objective(trial, X, y):
    if X.empty or y.empty or X.shape[0] != y.shape[0]:
        return 0

    n_samples = X.shape[0]

    if n_samples < 2:
        return 0

    test_size = 0.5
    if n_samples * test_size < 1 or n_samples * (1 - test_size) < 1:
        return 0

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)

    param = {
        'max_depth': trial.suggest_int('max_depth', 1, 3),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 16),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 16),
        'ccp_alpha': trial.suggest_float('ccp_alpha', 1e-5, 1e-1, log=True),
    }

    model = DecisionTreeClassifier(**param, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    accuracy = accuracy_score(y_val, preds)
    return accuracy


@log_time
def discover_gateway_models(dataframes, prefixes):
    gateway_analysis_results = {}
    for gateway_id, df in dataframes.items():
        print(f"Processing Gateway: {gateway_id}")

        feature_columns = [col for col in df.columns if not any(col.startswith(prefix) for prefix in prefixes)]
        target_columns = [col for col in df.columns if any(col.startswith(prefix) for prefix in prefixes)]

        models_results = {}
        for target_col in target_columns:
            features = df[feature_columns]
            target = df[target_col]

            study = optuna.create_study(direction='maximize')
            study.optimize(lambda trial: objective(trial, features, target), n_trials=100)

            best_params = study.best_params

            best_tree = DecisionTreeClassifier(**best_params, random_state=42)
            best_tree.fit(features, target)

            rules = extract_rules(best_tree.tree_, feature_names=best_tree.feature_names_in_)
            models_results[target_col] = rules

        gateway_analysis_results[gateway_id] = models_results

    return gateway_analysis_results


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

        if 'index' in df.columns:
            df.drop('index', axis=1, inplace=True)
        dataframes[gateway_id] = df

    return dataframes


@log_time
def discover_joins(bpmn_graph):
    gateway_branching_probabilities = []

    for element_id, element_info in bpmn_graph.element_info.items():
        if element_info.is_gateway() and element_info.is_join():
            gateway_branching_probabilities.append({
                "gateway_id": element_id,
                "probabilities": [{"path_id": element_info.outgoing_flows[0], "value": "1"}]
            })

    return gateway_branching_probabilities


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

        is_correct, fired_tasks, pending_tokens, _, gateway_states = bpmn_graph.reply_trace(
            task_sequence, flow_arcs_frequency, True, trace_info.event_list
        )

    return gateway_states


@log_time
def preprocess_event_log(event_log, log_ids, sampling_size=DEFAULT_SAMPLING_SIZE):
    sorted_log = event_log.sort_values(by=log_ids.end_time)

    log_by_case = sample_event_log_by_case(sorted_log, log_ids, sampling_size)
    log_by_case = fill_nans(log_by_case, log_ids)

    string_values = list()
    for col in log_by_case.columns:
        if log_by_case[col].dtype == 'object':
            string_values.append(col)

    log_by_case = scale_data(log_by_case, string_values)
    return log_by_case


@log_time
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


@log_time
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


@log_time
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