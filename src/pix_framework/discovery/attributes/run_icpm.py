import pandas as pd
from attribute_discovery import discover_attributes
from pix_framework.io.event_log import EventLogIDs
import time
import os
import glob

# Defining the log IDs
PROSIMOS_LOG_IDS = EventLogIDs(
    case="case_id",
    activity="activity",
    enabled_time="enable_time",
    start_time="start_time",
    end_time="end_time",
    resource="resource",
)

base_dir = r'D:\_est\PIX_discovery\ICPM\assets\data\out'


def create_path_list(base_dir):
    pattern = os.path.join(base_dir, '*_*_[0-9]*.csv')
    path_list = glob.glob(pattern)
    return path_list


def filter_by_number(path_list, number):
    return [path for path in path_list if f'_{number}.csv' in path]


def filter_by_prefix(path_list, prefix):
    return [path for path in path_list if os.path.basename(path).startswith(prefix)]


def fetch_and_print_attributes(file_name, method, log_ids):
        start_time = time.time()
        event_log = pd.read_csv(file_name)
        results = method(event_log, log_ids=log_ids)
        end_time = time.time()  # End time
        print(f"\n{file_name.split('/')[-1]} execution time: {end_time - start_time:.2f} seconds\n\n\n\n")


def discover_and_print_for_files(method, file_paths, log_ids):
    for file_name in file_paths:
        print(f"\n\n\n\n\nDISCOVERING {file_name}")
        fetch_and_print_attributes(file_name, method, log_ids)


if __name__ == "__main__":
    files_to_discover = create_path_list(base_dir)

    # files_to_discover = filter_by_number(files_to_discover, 9)
    # files_to_discover = filter_by_number(files_to_discover, 10)

    files_to_discover = filter_by_prefix(files_to_discover, 'single_global')
    # files_to_discover = filter_by_prefix(files_to_discover, 'multiple_global')
    print(files_to_discover)
    discover_and_print_for_files(discover_attributes, files_to_discover, PROSIMOS_LOG_IDS)



