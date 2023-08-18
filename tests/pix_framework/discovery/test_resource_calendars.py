from pathlib import Path

import pytest
from pix_framework.calendar.resource_calendar import RCalendar
from pix_framework.discovery.resource_calendars import (
    CalendarDiscoveryParams,
    CalendarType,
    _discover_resource_calendars_per_profile,
    _discover_undifferentiated_resource_calendar,
    discover_classic_resource_calendars_per_profile,
)
from pix_framework.discovery.resource_profiles import (
    discover_differentiated_resource_profiles,
    discover_pool_resource_profiles,
)
from pix_framework.io.event_log import APROMORE_LOG_IDS, read_csv_log

assets_dir = Path(__file__).parent.parent / "assets"


@pytest.mark.integration
@pytest.mark.parametrize("log_name", ["Resource_profiles_calendar_test.csv"])
def test_discover_resource_calendars_per_profile(log_name):
    log_path = assets_dir / log_name
    log_ids = APROMORE_LOG_IDS
    # Read event log
    log = read_csv_log(log_path, log_ids)
    # Discover profiles
    resource_profiles = discover_differentiated_resource_profiles(event_log=log, log_ids=log_ids)
    # Assert their old calendar IDs
    for resource_profile in resource_profiles:
        for resource in resource_profile.resources:
            assert resource.name in resource.calendar_id
    # Discover resource calendar per profile
    result = discover_classic_resource_calendars_per_profile(
        event_log=log,
        log_ids=log_ids,
        params=CalendarDiscoveryParams(CalendarType.UNDIFFERENTIATED),
        resource_profiles=resource_profiles,
    )
    assert len(result) == 1
    discovered_calendar_id = result[0].calendar_id
    # Assert they're updated
    for resource_profile in resource_profiles:
        for resource in resource_profile.resources:
            assert resource.calendar_id == discovered_calendar_id


@pytest.mark.integration
@pytest.mark.parametrize("log_name", ["Resource_profiles_calendar_test.csv"])
def test_resource_discover_undifferentiated(log_name):
    log_path = assets_dir / log_name
    log_ids = APROMORE_LOG_IDS
    # Read event log
    log = read_csv_log(log_path, log_ids)
    # Discover resource calendar
    result = _discover_undifferentiated_resource_calendar(
        event_log=log, log_ids=log_ids, params=CalendarDiscoveryParams(), calendar_id="Undifferentiated_test"
    )
    # Assert
    assert result
    assert type(result) is RCalendar
    assert result.calendar_id == "Undifferentiated_test"
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
@pytest.mark.parametrize("log_name", ["Resource_profiles_calendar_test.csv"])
def test_resource_discover_per_resource_pool(log_name):
    log_path = assets_dir / log_name
    log_ids = APROMORE_LOG_IDS
    # Read event log
    log = read_csv_log(log_path, log_ids)
    # Discover resource calendar
    resource_profiles = discover_pool_resource_profiles(event_log=log, log_ids=log_ids)
    result = _discover_resource_calendars_per_profile(
        event_log=log, log_ids=log_ids, params=CalendarDiscoveryParams(), resource_profiles=resource_profiles
    )
    # Assert
    assert result
    assert len(result) == 2
    jolyne_jotaro_pool = next(profile.name for profile in resource_profiles if len(profile.resources) > 1)
    for calendar in result:
        assert type(calendar) is RCalendar
        if jolyne_jotaro_pool in calendar.calendar_id:
            # Combined BusinessHours and Evening
            for i in range(5):
                # Mon-Fri from 7am to 8pm
                daily_work_intervals = calendar.work_intervals[i]
                # Only one interval
                assert len(daily_work_intervals) == 1
                work_interval = daily_work_intervals[0]
                # Starts at 7am
                assert work_interval.start.hour == 7
                assert work_interval.start.minute == 0
                assert work_interval.start.second == 0
                # Ends at 8pm
                assert work_interval.end.hour == 20
                assert work_interval.end.minute == 0
                assert work_interval.end.second == 0
            # Saturday and Sunday empty
            assert len(calendar.work_intervals[5]) == 0
            assert len(calendar.work_intervals[6]) == 0
        else:
            # Mon-Sun 24/7
            for i in range(7):
                daily_work_intervals = calendar.work_intervals[i]
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
@pytest.mark.parametrize("log_name", ["Resource_profiles_calendar_test.csv"])
def test_resource_discover_per_resource(log_name):
    log_path = assets_dir / log_name
    log_ids = APROMORE_LOG_IDS
    # Read event log
    log = read_csv_log(log_path, log_ids)
    # Discover resource calendar
    result = _discover_resource_calendars_per_profile(
        event_log=log,
        log_ids=log_ids,
        params=CalendarDiscoveryParams(),
        resource_profiles=discover_differentiated_resource_profiles(event_log=log, log_ids=log_ids),
    )

    # Assert
    assert result
    assert len(result) == 3
    for calendar in result:
        assert type(calendar) is RCalendar
        if "Jotaro" in calendar.calendar_id:
            # Business Hours
            for i in range(5):
                # Mon-Fri from 7am to 3pm
                daily_work_intervals = calendar.work_intervals[i]
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
            assert len(calendar.work_intervals[5]) == 0
            assert len(calendar.work_intervals[6]) == 0
        elif "Jolyne" in calendar.calendar_id:
            # Evening
            for i in range(5):
                # Mon-Fri from 7am to 3pm
                daily_work_intervals = calendar.work_intervals[i]
                # Only one interval
                assert len(daily_work_intervals) == 1
                work_interval = daily_work_intervals[0]
                # Starts at 3pm
                assert work_interval.start.hour == 15
                assert work_interval.start.minute == 0
                assert work_interval.start.second == 0
                # Ends at 8pm
                assert work_interval.end.hour == 20
                assert work_interval.end.minute == 0
                assert work_interval.end.second == 0
            # Saturday and Sunday empty
            assert len(calendar.work_intervals[5]) == 0
            assert len(calendar.work_intervals[6]) == 0
        else:
            assert "Pucci" in calendar.calendar_id
            # Mon-Sun 24/7
            for i in range(7):
                daily_work_intervals = calendar.work_intervals[i]
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
