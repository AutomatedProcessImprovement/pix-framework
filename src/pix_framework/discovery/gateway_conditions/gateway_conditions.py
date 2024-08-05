import optuna
import logging
import warnings
import pandas as pd

from pix_framework.io.event_log import EventLogIDs

from helpers import log_time
from replayer import parse_dataframe
from bpmn_parser import parse_simulation_model
from preprocessing import preprocess_event_log
from rules_postprocessing import process_rules
from branching_rules import discover_xor_gateways, discover_or_gateways
from trace_processing import process_traces, traces_to_dataframes, encode_dataframes


warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

namespaces = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}

DEFAULT_SAMPLING_SIZE = 25000


@log_time
def discover_gateway_conditions(bpmn_model_path,
                                event_log_path,
                                log_ids: EventLogIDs,
                                sampling_size: int = DEFAULT_SAMPLING_SIZE,
                                f_score_threshold=0.7):
    flow_arcs_frequency = dict()
    flow_prefixes = ["Flow_", "edge", "node"]
    avoid_columns = [
        log_ids.case, log_ids.activity, log_ids.start_time,
        log_ids.end_time, log_ids.resource, log_ids.enabled_time
    ]

    bpmn_graph = parse_simulation_model(bpmn_model_path)

    event_log = pd.read_csv(event_log_path)
    log_by_case = preprocess_event_log(event_log, log_ids, sampling_size)

    log_traces = parse_dataframe(log_by_case, log_ids, avoid_columns)

    gateway_states = process_traces(log_traces, bpmn_graph, flow_arcs_frequency)

    dataframes = traces_to_dataframes(gateway_states)
    dataframes, encoders = encode_dataframes(dataframes, flow_prefixes)

    xor_rules = discover_xor_gateways(gateway_states, dataframes, flow_prefixes, f_score_threshold)
    or_rules = discover_or_gateways(gateway_states, dataframes, flow_prefixes, f_score_threshold)

    xor_rules = process_rules(xor_rules, encoders)
    or_rules = process_rules(or_rules, encoders)

    xor_rules = format_branch_rules(xor_rules, dataframes, flow_prefixes)
    or_rules = format_branch_rules(or_rules, dataframes, flow_prefixes)

    gateway_probabilities = calculate_gateway_branching_probabilities(gateway_states, flow_arcs_frequency)
    gateway_probabilities = map_conditions_to_probabilities(gateway_probabilities, xor_rules, or_rules)

    branch_rules = xor_rules
    branch_rules.extend(or_rules)

    result = {"gateway_branching_probabilities": gateway_probabilities, "branch_rules": branch_rules}

    return result


def format_branch_rules(gateway_analysis_results, dataframes, prefixes):
    branch_rules = []

    for gateway_id, flows in gateway_analysis_results.items():
        for flow_id, conditions_list in flows.items():
            formatted_conditions = []
            always_true = False

            for condition_set in conditions_list:
                if condition_set[1] == 1 and not condition_set[0]:
                    always_true = True
                    break

                if condition_set[1] == 1:
                    inner_conditions = []
                    for condition in condition_set[0]:
                        attr, operator, value = condition
                        formatted_condition = {"attribute": attr, "comparison": operator, "value": str(value)}
                        inner_conditions.append(formatted_condition)
                    if inner_conditions:
                        formatted_conditions.append(inner_conditions)

            if always_true:
                non_flow_columns = [col for col in dataframes[gateway_id].columns if not any(col.startswith(prefix) for prefix in prefixes)]
                if non_flow_columns:
                    synthetic_rule = {
                        "id": flow_id,
                        "rules": [[{
                            "attribute": non_flow_columns[0],
                            "comparison": "!=",
                            "value": "WARNING_THIS_FLOW_ALWAYS_HAS_BEEN_EXECUTED"
                        }]]
                    }
                    branch_rules.append(synthetic_rule)
            elif formatted_conditions:
                branch_rule = {
                    "id": flow_id,
                    "rules": formatted_conditions
                }
                branch_rules.append(branch_rule)

    return branch_rules


@log_time
def calculate_gateway_branching_probabilities(gateway_states, flow_arcs_frequency):
    gateway_branching_probabilities = []

    for gateway_id, flows in gateway_states.items():
        probabilities = []
        total_executions = sum(flow_arcs_frequency.get(flow_id, 0) for flow_id in flows['outgoing_flows'])

        flow_ids = flows['outgoing_flows']
        for i, flow_id in enumerate(flow_ids):
            execution_count = flow_arcs_frequency.get(flow_id, 0)
            probability = execution_count / total_executions if total_executions > 0 else 0

            adjusted_probability = str(round(probability, 2)) if i < len(flow_ids) - 1 else str(round(1 - sum(float(p['value']) for p in probabilities), 2))
            prob_entry = {
                "path_id": flow_id,
                "value": adjusted_probability
            }

            probabilities.append(prob_entry)

        gateway_branching_probabilities.append({
            "gateway_id": gateway_id,
            "probabilities": probabilities
        })

    return gateway_branching_probabilities


def map_conditions_to_probabilities(gateway_probabilities, xor_conditions, or_conditions):
    conditions_lookup = {cond['id']: cond for cond in xor_conditions + or_conditions}

    for gateway in gateway_probabilities:
        probabilities = gateway['probabilities']

        for prob in probabilities:
            flow_id = prob['path_id']

            if flow_id in conditions_lookup:
                prob['condition_id'] = flow_id

    return gateway_probabilities
