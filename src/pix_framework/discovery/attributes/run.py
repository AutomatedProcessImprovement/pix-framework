import pandas as pd
from attribute_discovery import discover_attributes
from pix_framework.io.event_log import EventLogIDs
import pprint
import time
import os

# Defining the log IDs
PROSIMOS_LOG_IDS = EventLogIDs(
    case="case_id",
    activity="activity",
    enabled_time="enable_time",
    start_time="start_time",
    end_time="end_time",
    resource="resource",
)

# Constants for file paths
# CASE_ATTRIBUTES_FILE_PATH = "tests/pix_framework/assets/case_attributes_log.csv"
# EVENT_ATTRIBUTES_FILE_PATH = "tests/pix_framework/assets/event_attributes_log.csv"
# CASE_AND_EVENT_ATTRIBUTES_FILE_PATH = "tests/pix_framework/assets/case_and_event_attributes_log.csv"
# GLOBAL_ATTRIBUTES_FILE_PATH = "tests/pix_framework/assets/global_attributes_log.csv"
# EVENT_AND_GLOBAL_FILE_PATH = "tests/pix_framework/assets/event_and_global_attributes_log.csv"
# MATH_TEST_PATH = "tests/pix_framework/assets/math_log.csv"
# DISCRETE_TEST_PATH = "tests/pix_framework/assets/discrete_log.csv"
EXPERIMENTS_TEST_PATH = "D:/_est/PIX_discovery/Experiments/experiment_log_main.csv"
# MG_EXPERIMENTS_SHORT_TEST_PATH = "D:/_est/PIX_discovery/Experiments/mg_experiment_log_main.csv"
# ME_EXPERIMENTS_SHORT_TEST_PATH = "D:/_est/PIX_discovery/Experiments/me_experiment_log_main.csv"
# SG_EXPERIMENTS_SHORT_TEST_PATH = "D:/_est/PIX_discovery/Experiments/sg_experiment_log_main.csv"
# SE_EXPERIMENTS_SHORT_TEST_PATH = "D:/_est/PIX_discovery/Experiments/se_experiment_log_main.csv"
# C_EXPERIMENTS_SHORT_TEST_PATH = "D:/_est/PIX_discovery/Experiments/c_experiment_log_main.csv"

def fetch_and_print_attributes(file_name, method, log_ids):
        start_time = time.time()  # Start time

        event_log = pd.read_csv(file_name)

        # Fetching the results from method and printing them
        results = method(event_log, log_ids=log_ids)

        end_time = time.time()  # End time
        print(f"\n{file_name.split('/')[-1]} execution time: {end_time - start_time:.2f} seconds\n\n\n\n")


def discover_and_print_for_files(method, file_paths, log_ids):
    for file_name in file_paths:
        print(f"\n\n\n\n\nDISCOVERING {file_name}")
        fetch_and_print_attributes(file_name, method, log_ids)


if __name__ == "__main__":
    files_to_discover = [
        # CASE_ATTRIBUTES_FILE_PATH,
        # EVENT_ATTRIBUTES_FILE_PATH,
        # CASE_AND_EVENT_ATTRIBUTES_FILE_PATH,
        # GLOBAL_ATTRIBUTES_FILE_PATH,
        # EVENT_AND_GLOBAL_FILE_PATH,
        # MATH_TEST_PATH,
        # DISCRETE_TEST_PATH,
        EXPERIMENTS_TEST_PATH,
        # MG_EXPERIMENTS_SHORT_TEST_PATH,
        # ME_EXPERIMENTS_SHORT_TEST_PATH,
        # SG_EXPERIMENTS_SHORT_TEST_PATH,
        # SE_EXPERIMENTS_SHORT_TEST_PATH,
        # C_EXPERIMENTS_SHORT_TEST_PATH
    ]

    discover_and_print_for_files(discover_attributes, files_to_discover, PROSIMOS_LOG_IDS)



