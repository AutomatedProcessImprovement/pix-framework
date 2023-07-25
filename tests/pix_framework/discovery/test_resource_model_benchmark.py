from pathlib import Path

import pytest
from pix_framework.discovery.resource_calendars import CalendarDiscoveryParams, CalendarType
from pix_framework.discovery.resource_model import ResourceModel, discover_resource_model
from pix_framework.input import read_csv_log
from pix_framework.log_ids import PROSIMOS_LOG_IDS

assets_dir = Path(__file__).parent.parent / "assets"


@pytest.mark.benchmark(
    min_time=10,
    max_time=20,
)
@pytest.mark.parametrize("log_name", ["AcademicCredentials_train.csv.gz"])
def test_discover_case_arrival_model_differentiated_benchmark(benchmark, log_name):
    log_path = assets_dir / log_name
    log_ids = PROSIMOS_LOG_IDS
    log = read_csv_log(log_path, log_ids)

    # Discover resource model with differentiated resources
    result = benchmark(
        discover_resource_model,
        event_log=log,
        log_ids=log_ids,
        params=CalendarDiscoveryParams(discovery_type=CalendarType.DIFFERENTIATED_BY_RESOURCE),
    )

    # Assert
    assert type(result) is ResourceModel


@pytest.mark.benchmark(
    min_time=10,
    max_time=20,
)
@pytest.mark.parametrize("log_name", ["AcademicCredentials_train.csv.gz"])
def test_discover_case_arrival_model_pool_benchmark(benchmark, log_name):
    log_path = assets_dir / log_name
    log_ids = PROSIMOS_LOG_IDS
    log = read_csv_log(log_path, log_ids)

    # Discover resource model with pooled resources
    result = benchmark(
        discover_resource_model,
        event_log=log,
        log_ids=log_ids,
        params=CalendarDiscoveryParams(discovery_type=CalendarType.DIFFERENTIATED_BY_POOL),
    )

    # Assert
    assert type(result) is ResourceModel
