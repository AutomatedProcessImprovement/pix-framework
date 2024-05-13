import pandas as pd
from attribute_discovery import discover_attributes
from pix_framework.io.event_log import EventLogIDs
import pprint
import time

REAL_LIFE_LOG_IDS = EventLogIDs(
    case="case_id",
    activity="activity",
    start_time="Start_Time",
    end_time="End_Time",
    resource="resource",
)

REAL_LIFE = "D:/_est/PIX_discovery/Experiments/real_life/BPIC2019/BPIC2019.csv"
# REAL_LIFE = "D:/_est/PIX_discovery/Experiments/real_life/Sepsis/Sepsis_log.csv"
# REAL_LIFE = "D:/_est/PIX_discovery/Experiments/real_life/Trafic/Trafic_log.csv"

def fetch_and_print_attributes(file_name, method, sizes, log_ids):
    for size in sizes:
        start_time = time.time()  # Start time

        event_log = pd.read_csv(file_name)

        if size > 0:  # Only apply the filtering if size is greater than 0
            subset_cases = event_log.drop_duplicates(subset=log_ids.case).head(size)
            event_log = event_log[event_log[log_ids.case].isin(subset_cases[log_ids.case])]

        # Fetching the results from method and printing them
        results = method(event_log, log_ids=log_ids)

        end_time = time.time()  # End time
        print(f"\n{file_name.split('/')[-1]} with head size {size} execution time: {end_time - start_time:.2f} seconds\n\n\n\n")


def discover_and_print_for_files(method, file_paths, sizes, log_ids):
    for file_name in file_paths:
        print(f"\n\n\n\n\nDISCOVERING {file_name}")
        fetch_and_print_attributes(file_name, method, sizes, log_ids)


if __name__ == "__main__":
    sizes_to_test = [0]
    files_to_discover = [REAL_LIFE]

    discover_and_print_for_files(discover_attributes, files_to_discover, sizes_to_test, REAL_LIFE_LOG_IDS)
