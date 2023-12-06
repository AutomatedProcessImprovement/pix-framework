import pandas as pd
from attributes import discover_attributes
from case_attribute import discover_case_attributes
from pix_framework.io.event_log import EventLogIDs
import pprint
import time

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
CASE_ATTRIBUTES_FILE_PATH = "tests/pix_framework/assets/case_attributes_log.csv"
EVENT_ATTRIBUTES_FILE_PATH = "tests/pix_framework/assets/event_attributes_log.csv"
CASE_AND_EVENT_ATTRIBUTES_FILE_PATH = "tests/pix_framework/assets/case_and_event_attributes_log.csv"
GLOBAL_ATTRIBUTES_FILE_PATH = "tests/pix_framework/assets/global_attributes_log.csv"
EVENT_AND_GLOBAL_FILE_PATH = "tests/pix_framework/assets/event_and_global_attributes_log.csv"
RANDOM_TEST_PATH = "tests/pix_framework/assets/test.csv"

def fetch_and_print_attributes(file_name, method, sizes, log_ids):
    for size in sizes:
        start_time = time.time()  # Start time

        event_log = pd.read_csv(file_name)

        if size > 0:  # Only apply the filtering if size is greater than 0
            subset_cases = event_log.drop_duplicates(subset='case_id').head(size)
            event_log = event_log[event_log['case_id'].isin(subset_cases['case_id'])]

        # Fetching the results from method and printing them
        results = method(event_log, log_ids=log_ids)

        end_time = time.time()  # End time
        print(f"\n{file_name.split('/')[-1]} with head size {size} execution time: {end_time - start_time:.2f} seconds\n\n\n\n")


def discover_and_print_for_files(method, file_paths, sizes, log_ids):
    for file_name in file_paths:
        fetch_and_print_attributes(file_name, method, sizes, log_ids)


if __name__ == "__main__":
    sizes_to_test = [0]
    files_to_discover = [
        CASE_ATTRIBUTES_FILE_PATH,
        EVENT_ATTRIBUTES_FILE_PATH,
        CASE_AND_EVENT_ATTRIBUTES_FILE_PATH,
        GLOBAL_ATTRIBUTES_FILE_PATH,
        EVENT_AND_GLOBAL_FILE_PATH,
        RANDOM_TEST_PATH
    ]


    discover_and_print_for_files(discover_attributes, files_to_discover, sizes_to_test, PROSIMOS_LOG_IDS)



