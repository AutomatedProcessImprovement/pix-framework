import pandas as pd
from gateway_conditions import discover_gateway_conditions
from pix_framework.io.event_log import EventLogIDs
import pprint
import time
import xml.etree.ElementTree as ET


# Defining the log IDs
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

def fetch_and_print_conditions(bpmn_model_file, event_log_file, method, sizes, log_ids):
    for size in sizes:
        start_time = time.time()

        results = method(bpmn_model_file, event_log_file, log_ids=log_ids)

        end_time = time.time()
        print(f"\n{event_log_file.split('/')[-1]} with head size {size} execution time: {end_time - start_time:.2f} seconds\n\n\n\n")


def discover_and_print_for_files(method, file_paths, sizes, log_ids):
    for file_path in file_paths:
        bpmn_model, event_log_file = file_path
        print(f"\n\n\n\n\nDISCOVERING {event_log_file}")
        fetch_and_print_conditions(bpmn_model, event_log_file, method, sizes, log_ids)


if __name__ == "__main__":
    sizes_to_test = [0]
    files_to_discover = [
        BASIC_CONDITIONS
    ]

    discover_and_print_for_files(discover_gateway_conditions, files_to_discover, sizes_to_test, PROSIMOS_LOG_IDS)




