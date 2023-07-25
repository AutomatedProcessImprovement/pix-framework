from pathlib import Path

from pix_framework.discovery.resource_calendars import CalendarDiscoveryParams, CalendarType
from pix_framework.discovery.resource_model import ResourceModel, discover_resource_model
from pix_framework.input import read_csv_log
from pix_framework.log_ids import PROSIMOS_LOG_IDS

assets_dir = Path(__file__).parent.parent / "assets"


def profile_discover_case_arrival_model_differentiated():
    log_path = assets_dir / "AcademicCredentials_train.csv.gz"
    log_ids = PROSIMOS_LOG_IDS
    log = read_csv_log(log_path, log_ids)

    result = discover_resource_model(
        event_log=log,
        log_ids=log_ids,
        params=CalendarDiscoveryParams(discovery_type=CalendarType.DIFFERENTIATED_BY_RESOURCE),
    )

    assert type(result) is ResourceModel


if __name__ == "__main__":
    profile_discover_case_arrival_model_differentiated()
