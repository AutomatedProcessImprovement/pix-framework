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

EXPERIMENTS_TEST_PATH = (r"D:\_est\PIX_discovery\Experiments\loan_application.bpmn",
                         r"D:\_est\PIX_discovery\Experiments\experiment_log_main.csv")
MG_EXPERIMENTS_SHORT_TEST_PATH = (r"D:\_est\PIX_discovery\Experiments\loan_application.bpmn",
                                  r"D:\_est\PIX_discovery\Experiments\mg_experiment_log_main.csv")
ME_EXPERIMENTS_SHORT_TEST_PATH = (r"D:\_est\PIX_discovery\Experiments\loan_application.bpmn",
                                  r"D:\_est\PIX_discovery\Experiments\me_experiment_log_main.csv")
SG_EXPERIMENTS_SHORT_TEST_PATH = (r"D:\_est\PIX_discovery\Experiments\loan_application.bpmn",
                                  r"D:\_est\PIX_discovery\Experiments\sg_experiment_log_main.csv")
SE_EXPERIMENTS_SHORT_TEST_PATH = (r"D:\_est\PIX_discovery\Experiments\loan_application.bpmn",
                                  r"D:\_est\PIX_discovery\Experiments\se_experiment_log_main.csv")
C_EXPERIMENTS_SHORT_TEST_PATH = (r"D:\_est\PIX_discovery\Experiments\loan_application.bpmn",
                                 r"D:\_est\PIX_discovery\Experiments\c_experiment_log_main.csv")

if __name__ == "__main__":
    files_to_discover = [
        EXPERIMENTS_TEST_PATH,
        MG_EXPERIMENTS_SHORT_TEST_PATH,
        ME_EXPERIMENTS_SHORT_TEST_PATH,
        SG_EXPERIMENTS_SHORT_TEST_PATH,
        SE_EXPERIMENTS_SHORT_TEST_PATH,
        C_EXPERIMENTS_SHORT_TEST_PATH
    ]
    for (bpmn, log) in files_to_discover:
        print(f"DISCOVERING {log}")
        results = discover_attributes(bpmn, log, log_ids=PROSIMOS_LOG_IDS)




