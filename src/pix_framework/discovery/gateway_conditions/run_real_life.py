import pprint

import pandas as pd
import json
import os
import subprocess
from log_distance_measures.n_gram_distribution import n_gram_distribution_distance
from log_distance_measures.relative_event_distribution import relative_event_distribution_distance
from log_distance_measures.cycle_time_distribution import cycle_time_distribution_distance
from log_distance_measures.config import AbsoluteTimestampType, discretize_to_hour
from pix_framework.io.event_log import EventLogIDs

# Defining log IDs for the original and prosimos logs
ORIGINAL_LOG_IDS = EventLogIDs(
    case="case_id",
    activity="activity",
    enabled_time="enable_time",
    start_time="Start_Time",
    end_time="End_Time",
    resource="resource",
)

PROSIMOS_LOG_IDS = EventLogIDs(
    case="case_id",
    activity="activity",
    enabled_time="enable_time",
    start_time="start_time",
    end_time="end_time",
    resource="resource",
)


def convert_times(df, log_ids):
    df[log_ids.start_time] = pd.to_datetime(df[log_ids.start_time], utc=True)
    df[log_ids.end_time] = pd.to_datetime(df[log_ids.end_time], utc=True)
    return df


def run_prosimos_simulation(bpmn_path, json_path, total_cases, log_out_path):
    prosimos_project_dir = "D:\\_est\\Prosimos\\Prosimos"
    command = [
        "poetry", "run", "prosimos", "start-simulation",
        "--bpmn_path", bpmn_path,
        "--json_path", json_path,
        "--total_cases", str(total_cases),
        "--log_out_path", log_out_path
    ]
    print("STARTING SIMULATION")
    result = subprocess.run(command, cwd=prosimos_project_dir, capture_output=True, text=True)

    if result.returncode == 0:
        print("Simulation completed successfully.")
    else:
        print("Error running simulation.")
        print(result.stderr)



def test_discovered_log(original_log_path, simulated_log_path):
    original_log = pd.read_csv(original_log_path)
    simulated_log = pd.read_csv(simulated_log_path)

    original_log = convert_times(original_log, ORIGINAL_LOG_IDS)
    simulated_log = convert_times(simulated_log, PROSIMOS_LOG_IDS)

    n_gram_distance = n_gram_distribution_distance(
        original_log, ORIGINAL_LOG_IDS,
        simulated_log, PROSIMOS_LOG_IDS,
        n=3
    )

    relative_event_distribution = relative_event_distribution_distance(
        original_log, ORIGINAL_LOG_IDS,
        simulated_log, PROSIMOS_LOG_IDS,
        discretize_type=AbsoluteTimestampType.BOTH,
        discretize_event=discretize_to_hour
    )

    cycle_time_distribution = cycle_time_distribution_distance(
        original_log, ORIGINAL_LOG_IDS,
        simulated_log, PROSIMOS_LOG_IDS,
        bin_size=pd.Timedelta(hours=1)
    )

    return {
        "n_gram_distance": n_gram_distance,
        "relative_event_distribution": relative_event_distribution,
        "cycle_time_distribution": cycle_time_distribution
    }


def main(folder_path, bpmn_model_name, original_log_name, config_files):
    original_log_path = os.path.join(folder_path, original_log_name)
    original_log_df = pd.read_csv(original_log_path)
    total_cases = original_log_df['case_id'].nunique()

    bpmn_model_path = os.path.join(folder_path, bpmn_model_name)
    results = {}

    for file_name in config_files:
        config_file_path = os.path.join(folder_path, f"discovered_{file_name}.json")
        output_file_path = os.path.join(folder_path, f"simulation_output_{file_name}.csv")

        attr_config_file_path = os.path.join(folder_path, f"discovered_{file_name}_attr.json")
        attr_output_file_path = os.path.join(folder_path, f"simulation_output_{file_name}_attr.csv")

        if file_name not in results:
            results[file_name] = {"condition_metrics": [], "probability_metrics": []}


        run_prosimos_simulation(bpmn_model_path, config_file_path, total_cases, output_file_path)
        run_prosimos_simulation(bpmn_model_path, attr_config_file_path, total_cases, attr_output_file_path)

        results[file_name]['probability_metrics'] = test_discovered_log(original_log_path, output_file_path)
        results[file_name]['condition_metrics'] = test_discovered_log(original_log_path, attr_output_file_path)

    pprint.pprint(results)


if __name__ == "__main__":
    folder_path = "D:\\_est\\PIX_discovery\\Experiments\\real_life"
    bpmn_model_name = "BPIC2019_10arc.bpmn"
    original_log_name = "BPIC2019.csv"
    config_files = ["BPIC2019_10arc"]

    main(folder_path, bpmn_model_name, original_log_name, config_files)
