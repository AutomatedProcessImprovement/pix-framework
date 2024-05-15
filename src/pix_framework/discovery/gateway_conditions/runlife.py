import pandas as pd
from gateway_conditions import discover_gateway_conditions
from pix_framework.io.event_log import EventLogIDs
import pprint
import time
import os
import json
import subprocess
from log_distance_measures.n_gram_distribution import n_gram_distribution_distance
from log_distance_measures.relative_event_distribution import relative_event_distribution_distance
from log_distance_measures.cycle_time_distribution import cycle_time_distribution_distance
from log_distance_measures.config import AbsoluteTimestampType, discretize_to_hour

PROSIMOS_PROJECT_DIR = "D:\\_est\\Prosimos\\Prosimos"

PROSIMOS_LOG_IDS = EventLogIDs(
    case="case_id",
    activity="activity",
    enabled_time="enable_time",
    start_time="start_time",
    end_time="end_time",
    resource="resource",
)

EXPERIMENTS_LOG_IDS = EventLogIDs(
    case="case_id",
    activity="activity",
    enabled_time="enable_time",
    start_time="Start_Time",
    end_time="End_Time",
    resource="resource",
)

Sepsis = (r"D:\_est\PIX_discovery\Experiments\real_life\Sepsis\Sepsis.bpmn",
          r"D:\_est\PIX_discovery\Experiments\real_life\Sepsis\Sepsis.csv")
Trafic = (r"D:\_est\PIX_discovery\Experiments\real_life\Trafic\Trafic.bpmn",
          r"D:\_est\PIX_discovery\Experiments\real_life\Trafic\Trafic.csv")


def convert_times(df, log_ids):
    df[log_ids.start_time] = pd.to_datetime(df[log_ids.start_time], utc=True)
    df[log_ids.end_time] = pd.to_datetime(df[log_ids.end_time], utc=True)
    return df


def test_discovered_log(original_log_path, simulated_log_path):
    original_log = pd.read_csv(original_log_path)
    simulated_log = pd.read_csv(simulated_log_path)

    original_log = convert_times(original_log, EXPERIMENTS_LOG_IDS)
    simulated_log = convert_times(simulated_log, PROSIMOS_LOG_IDS)

    n_gram_distance = n_gram_distribution_distance(
        original_log, EXPERIMENTS_LOG_IDS,
        simulated_log, PROSIMOS_LOG_IDS,
        n=3
    )

    relative_event_distribution = relative_event_distribution_distance(
        original_log, EXPERIMENTS_LOG_IDS,
        simulated_log, PROSIMOS_LOG_IDS,
        discretize_type=AbsoluteTimestampType.BOTH,
        discretize_event=discretize_to_hour
    )

    cycle_time_distribution = cycle_time_distribution_distance(
        original_log, EXPERIMENTS_LOG_IDS,
        simulated_log, PROSIMOS_LOG_IDS,
        bin_size=pd.Timedelta(hours=1)
    )

    return {
        "n_gram_distance": n_gram_distance,
        "relative_event_distribution": relative_event_distribution,
        "cycle_time_distribution": cycle_time_distribution
    }


def run_prosimos_simulation(bpmn_path, json_path, total_cases, log_out_path):
    command = [
        "poetry", "run", "prosimos", "start-simulation",
        "--bpmn_path", bpmn_path,
        "--json_path", json_path,
        "--total_cases", str(total_cases),
        "--log_out_path", log_out_path
    ]

    result = subprocess.run(command, cwd=PROSIMOS_PROJECT_DIR, capture_output=True, text=True)

    if result.returncode == 0:
        print("Simulation completed successfully.")
        print(result.stdout)
    else:
        print("Error running simulation.")
        print(result.stderr)


def count_cases_in_event_log(csv_path, case_id_column):
    event_log = pd.read_csv(csv_path)
    unique_cases = event_log[case_id_column].nunique()
    return unique_cases


def update_DAS_config(config_path, results):
    if os.path.exists(config_path):
        with open(config_path, 'r') as file:
            content = json.load(file)
            updated_content = content

            updated_content['gateway_branching_probabilities'] = results['gateway_branching_probabilities']
            updated_content['branch_rules'] = results['branch_rules']

            pprint.pprint(updated_content['gateway_branching_probabilities'])
            pprint.pprint(updated_content['branch_rules'])

    with open(config_path, 'w') as new_file:
        json.dump(updated_content, new_file, indent=4)


def main(bpmn_path, csv_path, run_experiments=False):
    work_dir = os.path.dirname(bpmn_path)

    DAS_config_path = csv_path.rsplit('.', 1)[0] + '_DAS.json'
    DAS_csv_path = csv_path.rsplit('.', 1)[0] + '_DAS.csv'

    TRAD_config_path = csv_path.rsplit('.', 1)[0] + '.json'
    TRAD_csv_path = csv_path.rsplit('.', 1)[0] + '_TRAD.csv'

    DAS_conditoins = discover_gateway_conditions(bpmn_path, csv_path, EXPERIMENTS_LOG_IDS)
    update_DAS_config(DAS_config_path, DAS_conditoins)

    if run_experiments == False:
        return

    cases_amount = count_cases_in_event_log(csv_path, EXPERIMENTS_LOG_IDS.case)
    run_prosimos_simulation(bpmn_path, DAS_config_path, cases_amount, DAS_csv_path)
    run_prosimos_simulation(bpmn_path, TRAD_config_path, cases_amount, TRAD_csv_path)

    metrics = dict()
    metrics["DAS"] = test_discovered_log(csv_path, DAS_csv_path)
    metrics["TRAD"] = test_discovered_log(csv_path, TRAD_csv_path)

    print_metrics(metrics, work_dir)



def print_metrics(metrics, outputdir):
    data_for_df = []

    cond_metrics = metrics['DAS']
    prob_metrics = metrics['TRAD']

    row = dict()
    for metric_name in cond_metrics.keys():
        row[f'Cond_{metric_name}'] = cond_metrics.get(metric_name, 'N/A')
        row[f'Prob_{metric_name}'] = prob_metrics.get(metric_name, 'N/A')

    data_for_df.append(row)
    df = pd.DataFrame(data_for_df)

    csv_filename = 'simulation_metrics.csv'
    csv_path = f'{outputdir}/{csv_filename}'

    df.to_csv(csv_path, index=False)

    print(f'DataFrame saved to {csv_path}')

if __name__ == "__main__":
    conditions_to_discover = [
        Sepsis,
        Trafic
    ]
    run_experiments = True  # To run simulations and metric calculation after discovery of the

    for bpmn_path, csv_path in conditions_to_discover:
        main(bpmn_path, csv_path, run_experiments)