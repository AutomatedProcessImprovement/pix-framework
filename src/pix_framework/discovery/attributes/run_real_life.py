import pandas as pd
from attribute_discovery import discover_attributes
from pix_framework.io.event_log import EventLogIDs
import json
import os

REAL_LIFE_LOG_IDS = EventLogIDs(
    case="case_id",
    activity="activity",
    start_time="Start_Time",
    end_time="End_Time",
    resource="resource",
)

BPIC2019 = (r"D:\_est\PIX_discovery\Experiments\icpm\real_life\BPIC2019\BPIC2019.bpmn",
            r"D:\_est\PIX_discovery\Experiments\icpm\real_life\BPIC2019\BPIC2019_train.csv")
SEPSIS = (r"D:\_est\PIX_discovery\Experiments\icpm\real_life\Sepsis\Sepsis.bpmn",
          r"D:\_est\PIX_discovery\Experiments\icpm\real_life\Sepsis\Sepsis_train.csv")
TRAFIC = (r"D:\_est\PIX_discovery\Experiments\icpm\real_life\Trafic\Trafic.bpmn",
          r"D:\_est\PIX_discovery\Experiments\icpm\real_life\Trafic\Trafic_train.csv")


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


if __name__ == "__main__":
    files_to_discover = [
        BPIC2019,
        SEPSIS,
        TRAFIC
    ]
    log_ids = REAL_LIFE_LOG_IDS

    for (bpmn, log) in files_to_discover:
        print(f"\n\n\n\n\nDISCOVERING {log}")
        attributes = discover_attributes(bpmn, log, log_ids=log_ids)
        update_das_config(log, attributes)

