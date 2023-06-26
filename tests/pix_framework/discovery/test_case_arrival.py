from pathlib import Path

import pytest

from pix_framework.calendar.resource_calendar import RCalendar
from pix_framework.discovery.case_arrival import discover_case_arrival_calendar, discover_inter_arrival_distribution
from pix_framework.input import read_csv_log
from pix_framework.log_ids import APROMORE_LOG_IDS

assets_dir = Path(__file__).parent.parent / "assets"


@pytest.mark.integration
@pytest.mark.parametrize("log_name", ["DifferentiatedCalendars.csv"])
def test_calendar_case_arrival_discover(log_name):
    log_path = assets_dir / log_name
    log_ids = APROMORE_LOG_IDS
    # Read event log
    log = read_csv_log(log_path, log_ids)
    # Discover arrival calendar
    result = discover_case_arrival_calendar(log, log_ids)
    # Assert it exists...
    assert result
    assert type(result) is RCalendar


@pytest.mark.integration
@pytest.mark.parametrize("log_name", ["DifferentiatedCalendars.csv"])
def test_discover_inter_arrival_distribution(log_name):
    log_path = assets_dir / log_name
    log_ids = APROMORE_LOG_IDS
    # Read event log
    log = read_csv_log(log_path, log_ids)
    # Discover inter-arrival distribution
    result = discover_inter_arrival_distribution(log, log_ids)
    # Assert it exists...
    assert result is not None
