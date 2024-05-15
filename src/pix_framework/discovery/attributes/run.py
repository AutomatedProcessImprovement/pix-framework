import pandas as pd
from attribute_discovery import discover_attributes
from pix_framework.io.event_log import EventLogIDs
import pprint
import time
import os

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

# Constants for file paths
LOG_PATH_EXAMPLE = "D:/path/file.csv"
EXPERIMENTS_TEST_PATH = "D:/_est/PIX_discovery/Experiments/experiment_log_main.csv"
SEPSIS_PATH = r"D:\_est\PIX_discovery\Experiments\real_life\Sepsis\Sepsis.csv"
TRAFIC_PATH = r"D:\_est\PIX_discovery\Experiments\real_life\Trafic\Trafic.csv"


def fetch_and_print_attributes(file_name, method, log_ids):
    start_time = time.time()

    event_log = pd.read_csv(file_name)
    results = method(event_log, log_ids=log_ids)

    end_time = time.time()
    print(f"\n{file_name.split('/')[-1]} execution time: {end_time - start_time:.2f} seconds\n\n\n\n")


def discover_and_print_for_files(method, file_paths, log_ids):
    for file_name in file_paths:
        print(f"\n\n\n\n\nDISCOVERING {file_name}")
        fetch_and_print_attributes(file_name, method, log_ids)


if __name__ == "__main__":
    files_to_discover = [
        # LOG_PATH_EXAMPLE,
        # EXPERIMENTS_TEST_PATH,
        # SEPSIS_PATH,
        # TRAFIC_PATH
    ]

    discover_and_print_for_files(discover_attributes, files_to_discover, EXPERIMENTS_LOG_IDS)



