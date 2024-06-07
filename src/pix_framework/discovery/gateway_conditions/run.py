import pandas as pd
from gateway_conditions import discover_gateway_conditions
from pix_framework.io.event_log import EventLogIDs
import pprint
import os
import json
import subprocess
from log_distance_measures.n_gram_distribution import n_gram_distribution_distance
from statistics import mean

PROSIMOS_LOG_IDS = EventLogIDs(
    case="case_id",
    activity="activity",
    enabled_time="enable_time",
    start_time="start_time",
    end_time="end_time",
    resource="resource",
)

BASIC_CONDITIONS = ("D:/_est/PIX_discovery/Experiments/conditions/basic_condition_model.bpmn",
                    "D:/_est/PIX_discovery/Experiments/conditions/basic_condition_config.csv")


def generate_model_csv_tuples(csv_folder_path, range):
    tuples_list = []
    bpmn_paths = {
        'xor': 'D:/_est/PIX_discovery/Experiments/conditions/test2/basic_xor_condition_model.bpmn',
        'or': 'D:/_est/PIX_discovery/Experiments/conditions/test2/basic_or_condition_model.bpmn'
    }

    for i in range:
        for prefix in bpmn_paths:
            csv_path = os.path.join(csv_folder_path, f"_{i}_{prefix}/_{i}.csv")
            if os.path.exists(csv_path):
                tuples_list.append((bpmn_paths[prefix], csv_path))

    return tuples_list


def run_prosimos_simulation(bpmn_path, json_path, total_cases, log_out_path):
    prosimos_project_dir = "D:\\_est\\Prosimos\\Prosimos"

    command = [
        "poetry", "run", "prosimos", "start-simulation",
        "--bpmn_path", bpmn_path,
        "--json_path", json_path,
        "--total_cases", str(total_cases),
        "--log_out_path", log_out_path
    ]

    result = subprocess.run(command, cwd=prosimos_project_dir, capture_output=True, text=True)

    if result.returncode == 0:
        print("Simulation completed successfully.")
        print(result.stdout)
    else:
        print("Error running simulation.")
        print(result.stderr)



def update_and_save_json(event_log_file, results, is_probability=False):
    base_name = os.path.splitext(os.path.basename(event_log_file))[0]
    suffix = '_prob' if is_probability else '_DAS'
    json_file_path = os.path.join(os.path.dirname(event_log_file), base_name + '.json')
    new_json_file_path = os.path.join(os.path.dirname(event_log_file), base_name + '_test' + suffix + '.json')

    updated_content = {}

    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as file:
            content = json.load(file)
            updated_content = content

            updated_content['gateway_branching_probabilities'] = results['gateway_branching_probabilities']
            updated_content['branch_rules'] = results['branch_rules']

            if is_probability:
                updated_content.pop('branch_rules', None)
                for gateway in updated_content['gateway_branching_probabilities']:
                    for probability in gateway['probabilities']:
                        probability.pop('condition_id', None)

    with open(new_json_file_path, 'w') as new_file:
        json.dump(updated_content, new_file, indent=4)

    print(f"Updated JSON saved to {new_json_file_path}")
    return new_json_file_path


def get_log_size(log_path):
    df = pd.read_csv(log_path)
    return df['case_id'].nunique()


def count_string_in_file(file_path, search_string):
    count = 0
    with open(file_path, 'r') as file:
        for line in file:
            if search_string in line:
                count += 1
    return count

def discover_and_print_for_files(method, file_paths, sizes, log_ids):
    all_metrics = {}

    for file_path in file_paths:
        bpmn_model_path, event_log_path = file_path
        base_name = os.path.splitext(os.path.basename(event_log_path))[0]
        simulation_warnings_path = os.path.join(os.path.dirname(event_log_path), "simulation_warnings.txt")
        search_string = "conditions evaluated to the same result"

        if base_name not in all_metrics:
            all_metrics[base_name] = {"condition_metrics": [], "probability_metrics": []}

        train_log_path = event_log_path.replace(".csv", "_train.csv")
        test_log_path = event_log_path.replace(".csv", "_test.csv")

        print(f"\nDISCOVERING {train_log_path}")
        result = method(bpmn_model_path, train_log_path, log_ids=log_ids)

        test_log_size = get_log_size(test_log_path)

        condition_json_file_path = update_and_save_json(event_log_path, result)
        probability_json_file_path = update_and_save_json(event_log_path, result, is_probability=True)

        for i in range(0, 3):
            condition_csv_file_path = condition_json_file_path.rsplit('.', 1)[0] + '.csv'
            probability_csv_file_path = probability_json_file_path.rsplit('.', 1)[0] + '.csv'

            run_prosimos_simulation(bpmn_model_path, condition_json_file_path, test_log_size, condition_csv_file_path)
            warning_count = count_string_in_file(simulation_warnings_path, search_string)
            condition_error_ratio = warning_count / (test_log_size * 3)
            condition_metrics = test_discovered_log(test_log_path, condition_csv_file_path)
            condition_metrics["condition_error_ratio"] = condition_error_ratio

            run_prosimos_simulation(bpmn_model_path, probability_json_file_path, test_log_size, probability_csv_file_path)
            probability_metrics = test_discovered_log(test_log_path, probability_csv_file_path)
            probability_metrics["condition_error_ratio"] = 1

            all_metrics[base_name]["condition_metrics"].append(condition_metrics)
            all_metrics[base_name]["probability_metrics"].append(probability_metrics)

    print("\nFinal Metrics:")
    print_metrics(all_metrics)


def print_metrics(metrics):
    averaged_metrics = {}

    for model, model_metrics in metrics.items():
        avg_cond_metrics = {}
        avg_prob_metrics = {}

        cond_metrics_list = model_metrics['condition_metrics']
        prob_metrics_list = model_metrics['probability_metrics']

        metric_names = set()
        for metrics_dict in cond_metrics_list + prob_metrics_list:
            metric_names.update(metrics_dict.keys())

        for metric_name in metric_names:
            cond_metric_values = [m[metric_name] for m in cond_metrics_list if metric_name in m]
            prob_metric_values = [m[metric_name] for m in prob_metrics_list if metric_name in m]

            if cond_metric_values:
                avg_cond_metrics[f'Cond_{metric_name}'] = mean(cond_metric_values)
            if prob_metric_values:
                avg_prob_metrics[f'Prob_{metric_name}'] = mean(prob_metric_values)

        averaged_metrics[model] = {**avg_cond_metrics, **avg_prob_metrics}

    data_for_df = []

    for model, avg_metrics in averaged_metrics.items():
        row = {'Model': model}
        row.update(avg_metrics)
        data_for_df.append(row)

    df = pd.DataFrame(data_for_df)

    folder_path = 'D:/_est/PIX_discovery/Experiments/conditions/test2'
    csv_filename = 'simulation_metrics.csv'
    csv_path = f'{folder_path}/{csv_filename}'

    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, mode='w', header=True, index=False)

    print(f'DataFrame saved to {csv_path}')


def convert_times(df):
    df['start_time'] = pd.to_datetime(df['start_time'], utc=True)
    df['end_time'] = pd.to_datetime(df['end_time'], utc=True)
    return df


def test_discovered_log(original_log_path, simulated_log_path):
    original_log = pd.read_csv(original_log_path)
    simulated_log = pd.read_csv(simulated_log_path)

    original_log = convert_times(original_log)
    simulated_log = convert_times(simulated_log)

    three_gram_distance = n_gram_distribution_distance(
        original_log, PROSIMOS_LOG_IDS,
        simulated_log, PROSIMOS_LOG_IDS,
        n=3
    )

    return {
        "three_gram_distance": three_gram_distance,
    }

if __name__ == "__main__":
    sizes_to_test = [0]
    files_to_discover = [
        # BASIC_CONDITIONS
    ]
    csv_folder_path = 'D:/_est/PIX_discovery/Experiments/conditions/test2'
    test_range = range(0, 100)

    files_to_discover.extend(generate_model_csv_tuples(csv_folder_path, test_range))

    pprint.pprint(files_to_discover)

    discover_and_print_for_files(discover_gateway_conditions, files_to_discover, sizes_to_test, PROSIMOS_LOG_IDS)




