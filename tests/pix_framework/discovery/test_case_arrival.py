from pathlib import Path

import pytest

from pix_framework.calendar.resource_calendar import RCalendar
from pix_framework.discovery.case_arrival import discover_case_arrival_calendar, discover_inter_arrival_distribution, \
    discover_case_arrival_model, CaseArrivalModel
from pix_framework.io.event_log import read_csv_log
from pix_framework.io.event_log import APROMORE_LOG_IDS

assets_dir = Path(__file__).parent.parent / "assets"


@pytest.mark.integration
@pytest.mark.parametrize("log_name", ["Arrival_test_fixed.csv", "Arrival_test_normal.csv"])
def test_discover_case_arrival_model(log_name):
    log_path = assets_dir / log_name
    log_ids = APROMORE_LOG_IDS
    # Read event log
    log = read_csv_log(log_path, log_ids)
    # Discover arrival calendar
    result = discover_case_arrival_model(log, log_ids)
    # Assert
    assert type(result) is CaseArrivalModel
    assert type(result.case_arrival_calendar) is RCalendar
    assert result.inter_arrival_times is not None
    assert result.inter_arrival_times["distribution_name"] in ["fix", "norm"]


@pytest.mark.integration
@pytest.mark.parametrize("log_name", ["Arrival_test_fixed.csv"])
def test_discover_case_arrival_calendar_business_hours(log_name):
    log_path = assets_dir / log_name
    log_ids = APROMORE_LOG_IDS
    # Read event log
    log = read_csv_log(log_path, log_ids)
    # Discover arrival calendar
    result = discover_case_arrival_calendar(log, log_ids)
    # Assert
    assert type(result) is RCalendar
    for i in range(5):
        # Mon-Fri from 7am to 3pm
        daily_work_intervals = result.work_intervals[i]
        # Only one interval
        assert len(daily_work_intervals) == 1
        work_interval = daily_work_intervals[0]
        # Starts at 7am
        assert work_interval.start.hour == 7
        assert work_interval.start.minute == 0
        assert work_interval.start.second == 0
        # Ends at 3pm
        assert work_interval.end.hour == 15
        assert work_interval.end.minute == 0
        assert work_interval.end.second == 0
    # Saturday and Sunday empty
    assert len(result.work_intervals[5]) == 0
    assert len(result.work_intervals[6]) == 0


@pytest.mark.integration
@pytest.mark.parametrize("log_name", ["Arrival_test_normal.csv"])
def test_discover_case_arrival_calendar_24_7(log_name):
    log_path = assets_dir / log_name
    log_ids = APROMORE_LOG_IDS
    # Read event log
    log = read_csv_log(log_path, log_ids)
    # Discover arrival calendar
    result = discover_case_arrival_calendar(log, log_ids)
    # Assert
    assert type(result) is RCalendar
    for i in range(7):
        # Mon-Sun 24/7
        daily_work_intervals = result.work_intervals[i]
        # Only one interval
        assert len(daily_work_intervals) == 1
        work_interval = daily_work_intervals[0]
        # Starts at 12am
        assert work_interval.start.hour == 0
        assert work_interval.start.minute == 0
        assert work_interval.start.second == 0
        # Ends at 11:59pm
        assert work_interval.end.hour == 23
        assert work_interval.end.minute == 59
        assert work_interval.end.second == 59


@pytest.mark.integration
@pytest.mark.parametrize("log_name", ["Arrival_test_fixed.csv"])
def test_discover_inter_arrival_distribution_fixed(log_name):
    log_path = assets_dir / log_name
    log_ids = APROMORE_LOG_IDS
    # Read event log
    log = read_csv_log(log_path, log_ids)
    # Discover inter-arrival distribution
    result = discover_inter_arrival_distribution(log, log_ids)
    # Assert
    assert result is not None
    assert result["distribution_name"] == "fix"
    assert result["distribution_params"][0]["value"] == 1800.0


@pytest.mark.integration
@pytest.mark.parametrize("log_name", ["Arrival_test_normal.csv"])
def test_discover_inter_arrival_distribution_normal(log_name):
    log_path = assets_dir / log_name
    log_ids = APROMORE_LOG_IDS
    # Read event log
    log = read_csv_log(log_path, log_ids)
    # Discover inter-arrival distribution
    result = discover_inter_arrival_distribution(log, log_ids)
    # Assert
    assert result is not None
    assert result["distribution_name"] == "norm"
    assert result["distribution_params"][0]["value"] - 2700 < 600  # Less than 10m error in norm mean
