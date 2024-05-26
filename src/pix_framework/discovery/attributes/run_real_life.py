import pandas as pd
from attribute_discovery import discover_attributes
from pix_framework.io.event_log import EventLogIDs
import pprint
import time
import json
import os

REAL_LIFE_LOG_IDS = EventLogIDs(
    case="case_id",
    activity="activity",
    start_time="Start_Time",
    end_time="End_Time",
    resource="resource",
)

BPIC2019 = r"D:\_est\PIX_discovery\Experiments\icpm\real_life\BPIC2019\BPIC2019_train.csv"
SEPSIS = r"D:\_est\PIX_discovery\Experiments\icpm\real_life\Sepsis\Sepsis_train.csv"
TRAFIC = r"D:\_est\PIX_discovery\Experiments\icpm\real_life\Trafic\Trafic_train.csv"

def fetch_and_print_attributes(file_name, method, log_ids):
    start_time = time.time()  # Start time
    event_log = pd.read_csv(file_name)
    results = method(event_log, log_ids=log_ids)
    end_time = time.time()  # End time
    print(f"\n{file_name.split('/')[-1]} execution time: {end_time - start_time:.2f} seconds\n\n\n\n")
    return results


def update_das_config(file_name, attributes):
    config_file_path = file_name.replace('_train.csv', '_DAS.json')

    if not os.path.exists(config_file_path):
        print(f"WARNING: No such file {config_file_path}")

    with open(config_file_path, 'r') as config_file:
        config_data = json.load(config_file)

    config_data.update(attributes)

    with open(config_file_path, 'w') as config_file:
        json.dump(config_data, config_file, indent=4)

    print(f"Updated DAS configuration file: {config_file_path}")

def discover_and_print_for_files(method, file_paths, log_ids):
    for file_name in file_paths:
        print(f"\n\n\n\n\nDISCOVERING {file_name}")
        attributes = fetch_and_print_attributes(file_name, method, log_ids)
        update_das_config(file_name, attributes)


if __name__ == "__main__":
    files_to_discover = [
        BPIC2019,
        SEPSIS,
        TRAFIC
    ]
    discover_and_print_for_files(discover_attributes, files_to_discover, REAL_LIFE_LOG_IDS)
