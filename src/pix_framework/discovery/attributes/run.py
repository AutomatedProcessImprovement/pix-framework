import pandas as pd
from attribute_discovery import discover_attributes
from pix_framework.io.event_log import EventLogIDs

PROSIMOS_LOG_IDS = EventLogIDs(
    case="case_id",
    activity="activity",
    enabled_time="enable_time",
    start_time="start_time",
    end_time="end_time",
    resource="resource",
)

EXPERIMENTS_TEST_PATH = "D:/_est/PIX_discovery/Experiments/experiment_log_main.csv"
MG_EXPERIMENTS_SHORT_TEST_PATH = "D:/_est/PIX_discovery/Experiments/mg_experiment_log_main.csv"
ME_EXPERIMENTS_SHORT_TEST_PATH = "D:/_est/PIX_discovery/Experiments/me_experiment_log_main.csv"
SG_EXPERIMENTS_SHORT_TEST_PATH = "D:/_est/PIX_discovery/Experiments/sg_experiment_log_main.csv"
SE_EXPERIMENTS_SHORT_TEST_PATH = "D:/_est/PIX_discovery/Experiments/se_experiment_log_main.csv"
C_EXPERIMENTS_SHORT_TEST_PATH = "D:/_est/PIX_discovery/Experiments/c_experiment_log_main.csv"


def discover_and_print_for_files(method, file_paths, log_ids):
    for file_name in file_paths:
        print(f"DISCOVERING {file_name}")
        event_log = pd.read_csv(file_name)
        results = method(event_log, log_ids=log_ids)


if __name__ == "__main__":
    files_to_discover = [
        EXPERIMENTS_TEST_PATH,
        MG_EXPERIMENTS_SHORT_TEST_PATH,
        ME_EXPERIMENTS_SHORT_TEST_PATH,
        SG_EXPERIMENTS_SHORT_TEST_PATH,
        SE_EXPERIMENTS_SHORT_TEST_PATH,
        C_EXPERIMENTS_SHORT_TEST_PATH
    ]

    discover_and_print_for_files(discover_attributes, files_to_discover, PROSIMOS_LOG_IDS)



