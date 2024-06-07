from attribute_discovery import discover_attributes
from pix_framework.io.event_log import EventLogIDs
import os
import glob

PROSIMOS_LOG_IDS = EventLogIDs(
    case="case_id",
    activity="activity",
    enabled_time="enable_time",
    start_time="start_time",
    end_time="end_time",
    resource="resource",
)

base_dir = r'D:\_est\PIX_discovery\ICPM\assets\data\out'
bpmn = r'D:\_est\PIX_discovery\ICPM\loan_application.bpmn'

def create_path_list(base_dir):
    pattern = os.path.join(base_dir, '*_*_[0-9]*.csv')
    path_list = glob.glob(pattern)
    return path_list


def filter_by_number(path_list, number):
    return [path for path in path_list if f'_{number}.csv' in path]


def filter_by_prefix(path_list, prefix):
    return [path for path in path_list if os.path.basename(path).startswith(prefix)]


if __name__ == "__main__":
    for i in range(1,30):
        files_to_discover = create_path_list(base_dir)

        files_to_discover = filter_by_number(files_to_discover, i)
        # files_to_discover = filter_by_number(files_to_discover, 10)

        # files_to_discover = filter_by_prefix(files_to_discover, 'single_global')
        # files_to_discover = filter_by_prefix(files_to_discover, 'multiple_global')

        print(files_to_discover)
        for log in files_to_discover:
            print(f"\n\n\n\n\nDISCOVERING {log}")
            discover_attributes(bpmn, log, log_ids=PROSIMOS_LOG_IDS)



