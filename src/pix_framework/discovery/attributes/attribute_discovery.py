import logging
import warnings
import pandas as pd

from pix_framework.io.event_log import EventLogIDs

from pix_framework.discovery.attributes.helpers import subtract_lists
from pix_framework.discovery.attributes.preprocessing import preprocess_event_log
from pix_framework.discovery.attributes.case_attributes import discover_case_attributes
from pix_framework.discovery.attributes.ge_discrete_attributes import discover_global_and_event_discrete_attributes
from pix_framework.discovery.attributes.ge_continuous_attributes import discover_global_and_event_continuous_attributes


warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

namespaces = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}


def discover_attributes(event_log: pd.DataFrame,
                        log_ids: EventLogIDs,
                        avoid_columns: list = [],
                        confidence_threshold: float = 1.0):
    default_avoid_columns = [
        log_ids.case, log_ids.activity, log_ids.start_time,
        log_ids.end_time, log_ids.resource, log_ids.enabled_time
    ]
    avoid_columns = default_avoid_columns + avoid_columns

    attributes_to_discover = subtract_lists(event_log.columns, avoid_columns)

    g_dfs, e_dfs, encoders = preprocess_event_log(event_log, avoid_columns, log_ids)

    # CASE ATTRIBUTES
    case_attributes, case_attribute_metrics = discover_case_attributes(e_dfs, attributes_to_discover, encoders, log_ids, confidence_threshold)
    case_attribute_names = [attribute['name'] for attribute in case_attributes]
    attributes_to_discover = subtract_lists(attributes_to_discover, case_attribute_names)

    # DISCRETE GLOBAL AND EVENT ATTRIBUTES
    discrete_results = discover_global_and_event_discrete_attributes(g_dfs, e_dfs, attributes_to_discover, encoders)
    discrete_results = determine_attribute_type_and_best_model(discrete_results, "KS")
    global_discrete_attributes, event_discrete_attributes = format_attribute_results(discrete_results)

    discovered_discrete_attribute_names = get_discovered_attribute_names(global_discrete_attributes, event_discrete_attributes)
    attributes_to_discover = subtract_lists(attributes_to_discover, discovered_discrete_attribute_names)

    # CONTINUOUS GLOBAL AND EVENT ATTRIBUTES
    continuous_results = discover_global_and_event_continuous_attributes(g_dfs, e_dfs, attributes_to_discover)
    continuous_results = determine_attribute_type_and_best_model(continuous_results, "EMD")
    global_continuous_attributes, event_continuous_attributes = format_attribute_results(continuous_results)

    global_attributes = merge_global_attributes(global_discrete_attributes, global_continuous_attributes)

    event_attributes = merge_event_attributes(event_discrete_attributes, event_continuous_attributes)

    return {
        "global_attributes": global_attributes,
        "case_attributes": case_attributes,
        "event_attributes": event_attributes
    }


def determine_attribute_type_and_best_model(discovery_results, comparison_metric):
    for attr_name, attr_data in discovery_results.items():
        best_model_name = None
        best_score = None
        best_type = None

        for model_name, model_data in attr_data['models'].items():
            for attr_type in ['event', 'global']:
                if comparison_metric in model_data['total_scores'][attr_type]:
                    score = model_data['total_scores'][attr_type][comparison_metric]

                    if best_score is None or score < best_score:
                        best_score = score
                        best_model_name = model_name
                        best_type = attr_type

        if best_model_name and best_type:
            discovery_results[attr_name]['best_model'] = best_model_name
            discovery_results[attr_name]['type'] = best_type

    return discovery_results


def format_attribute_results(discovery_results):
    global_attributes = []
    event_attributes_dict = {}

    for attr_name, attr_data in discovery_results.items():
        best_model_name = attr_data.get('best_model')
        attr_type = attr_data.get('type')
        model_data = attr_data['models'][best_model_name]

        if best_model_name == 'Linear Regression' or best_model_name == 'Curve Fitting Update Rules':
            attribute_format = 'expression'
        elif best_model_name == 'M5Prime':
            attribute_format = 'dtree'
        elif best_model_name == 'Curve Fitting Generators':
            attribute_format = 'continuous'
        else:
            attribute_format = 'discrete'

        if attr_type == 'global':
            placeholder_values = {
                "distribution_name": "fix",
                "distribution_params": [{"value": 0}]} if attribute_format in ['continuous', 'expression', 'dtree'] \
                else [{"key": "", "value": 1}]

            global_attribute = {
                "name": attr_name,
                "type": 'continuous' if attribute_format in ['continuous', 'expression', 'dtree'] else 'discrete',
                "values": placeholder_values
            }
            if global_attribute not in global_attributes:
                global_attributes.append(global_attribute)

        for activity_id, activity_data in model_data['activities'].items():
            formula_info = activity_data[attr_type]['formula'] if attr_type in activity_data else None

            if formula_info:
                event_attribute_entry = {
                    "name": attr_name,
                    "type": attribute_format,
                    "values": formula_info
                }

                if activity_id not in event_attributes_dict:
                    event_attributes_dict[activity_id] = []
                event_attributes_dict[activity_id].append(event_attribute_entry)

    event_attributes = [{"event_id": event_id, "attributes": attrs} for event_id, attrs in
                        event_attributes_dict.items()]

    return global_attributes, event_attributes


def get_discovered_attribute_names(global_attrs, event_attrs):
    global_attr_names = [attr['name'] for attr in global_attrs]

    event_attr_names = []
    for event in event_attrs:
        for attr in event['attributes']:
            if attr['name'] not in event_attr_names:
                event_attr_names.append(attr['name'])

    discovered_attr_names = list(set(global_attr_names + event_attr_names))
    return discovered_attr_names


def merge_global_attributes(*global_attributes_groups):
    merged_global_attributes = []
    for attribute_group in global_attributes_groups:
        merged_global_attributes.extend(attribute_group)
    return merged_global_attributes


def merge_event_attributes(*event_attributes_groups):
    merged_events_dict = {}

    def merge_event_list(event_list):
        for event in event_list:
            event_id = event['event_id']
            if event_id not in merged_events_dict:
                merged_events_dict[event_id] = event['attributes']
            else:
                existing_attrs = {attr['name'] for attr in merged_events_dict[event_id]}
                merged_events_dict[event_id].extend(
                    [attr for attr in event['attributes'] if attr['name'] not in existing_attrs]
                )

    for event_group in event_attributes_groups:
        merge_event_list(event_group)

    merged_event_attributes = [{"event_id": event_id, "attributes": attrs} for event_id, attrs in merged_events_dict.items()]
    return merged_event_attributes
