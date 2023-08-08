from pathlib import Path

import pytest
from pix_framework.calendar.resource_calendar import RCalendar
from pix_framework.discovery.resource_activity_performances import ActivityResourceDistribution
from pix_framework.discovery.resource_calendars import CalendarDiscoveryParams, CalendarType
from pix_framework.discovery.resource_model import ResourceModel, discover_resource_model
from pix_framework.discovery.resource_profiles import ResourceProfile
from pix_framework.io.event_log import APROMORE_LOG_IDS, read_csv_log
from pix_framework.io.event_log import PROSIMOS_LOG_IDS

assets_dir = Path(__file__).parent.parent / "assets"


@pytest.mark.integration
@pytest.mark.parametrize("log_name", ["Resource_profiles_test.csv"])
def test_discover_case_arrival_model_undifferentiated(log_name):
    log_path = assets_dir / log_name
    log_ids = APROMORE_LOG_IDS
    # Read event log
    log = read_csv_log(log_path, log_ids)

    # Discover resource model with undifferentiated resources
    result = discover_resource_model(
        event_log=log, log_ids=log_ids, params=CalendarDiscoveryParams(discovery_type=CalendarType.UNDIFFERENTIATED)
    )
    # Assert
    assert type(result) is ResourceModel
    assert len(result.resource_profiles) == 1
    assert type(result.resource_profiles[0]) is ResourceProfile
    assert result.resource_profiles[0].name == "Undifferentiated_resource_profile"
    assert len(result.resource_calendars) == 1
    assert type(result.resource_calendars[0]) is RCalendar
    assert result.resource_calendars[0].calendar_id == "Undifferentiated_calendar"
    assert len(result.activity_resource_distributions) == 5
    assert type(result.activity_resource_distributions[0]) is ActivityResourceDistribution
    assert (
        len(
            [
                distribution.resource_id
                for activity in result.activity_resource_distributions
                for distribution in activity.activity_resources_distributions
            ]
        )
        == 15
    )


@pytest.mark.integration
@pytest.mark.parametrize("log_name", ["Resource_profiles_test.csv"])
def test_discover_case_arrival_model_24_7(log_name):
    log_path = assets_dir / log_name
    log_ids = APROMORE_LOG_IDS
    # Read event log
    log = read_csv_log(log_path, log_ids)

    # Discover resource model with 24/7 resources
    result = discover_resource_model(
        event_log=log, log_ids=log_ids, params=CalendarDiscoveryParams(discovery_type=CalendarType.DEFAULT_24_7)
    )
    # Assert
    assert type(result) is ResourceModel
    assert len(result.resource_profiles) == 1
    assert type(result.resource_profiles[0]) is ResourceProfile
    assert result.resource_profiles[0].name == "Undifferentiated_resource_profile"
    assert len(result.resource_calendars) == 1
    assert type(result.resource_calendars[0]) is RCalendar
    assert result.resource_calendars[0].calendar_id == "24_7_CALENDAR"
    assert len(result.activity_resource_distributions) == 5
    assert type(result.activity_resource_distributions[0]) is ActivityResourceDistribution
    assert (
        len(
            [
                distribution.resource_id
                for activity in result.activity_resource_distributions
                for distribution in activity.activity_resources_distributions
            ]
        )
        == 15
    )


@pytest.mark.integration
@pytest.mark.parametrize("log_name", ["Resource_profiles_test.csv"])
def test_discover_case_arrival_model_9_5(log_name):
    log_path = assets_dir / log_name
    log_ids = APROMORE_LOG_IDS
    # Read event log
    log = read_csv_log(log_path, log_ids)

    # Discover resource model with 9/5 resources
    result = discover_resource_model(
        event_log=log, log_ids=log_ids, params=CalendarDiscoveryParams(discovery_type=CalendarType.DEFAULT_9_5)
    )
    # Assert
    assert type(result) is ResourceModel
    assert len(result.resource_profiles) == 1
    assert type(result.resource_profiles[0]) is ResourceProfile
    assert result.resource_profiles[0].name == "Undifferentiated_resource_profile"
    assert len(result.resource_calendars) == 1
    assert type(result.resource_calendars[0]) is RCalendar
    assert result.resource_calendars[0].calendar_id == "9_5_CALENDAR"
    assert len(result.activity_resource_distributions) == 5
    assert type(result.activity_resource_distributions[0]) is ActivityResourceDistribution
    assert (
        len(
            [
                distribution.resource_id
                for activity in result.activity_resource_distributions
                for distribution in activity.activity_resources_distributions
            ]
        )
        == 15
    )


@pytest.mark.integration
@pytest.mark.parametrize("log_name", ["Resource_profiles_test.csv"])
def test_discover_case_arrival_model_differentiated(log_name):
    log_path = assets_dir / log_name
    log_ids = APROMORE_LOG_IDS
    # Read event log
    log = read_csv_log(log_path, log_ids)

    # Discover resource model with differentiated resources
    result = discover_resource_model(
        event_log=log,
        log_ids=log_ids,
        params=CalendarDiscoveryParams(discovery_type=CalendarType.DIFFERENTIATED_BY_RESOURCE),
    )
    # Assert
    assert type(result) is ResourceModel
    assert len(result.resource_profiles) == 3
    assert type(result.resource_profiles[0]) is ResourceProfile
    assert len(result.resource_calendars) == 3
    assert type(result.resource_calendars[0]) is RCalendar
    assert len(result.activity_resource_distributions) == 5
    assert type(result.activity_resource_distributions[0]) is ActivityResourceDistribution
    assert (
        len(
            [
                distribution.resource_id
                for activity in result.activity_resource_distributions
                for distribution in activity.activity_resources_distributions
            ]
        )
        == 8
    )


@pytest.mark.integration
@pytest.mark.parametrize("log_name", ["Resource_profiles_test.csv"])
def test_discover_case_arrival_model_pool(log_name):
    log_path = assets_dir / log_name
    log_ids = APROMORE_LOG_IDS
    # Read event log
    log = read_csv_log(log_path, log_ids)

    # Discover resource model with pooled resources
    result = discover_resource_model(
        event_log=log,
        log_ids=log_ids,
        params=CalendarDiscoveryParams(discovery_type=CalendarType.DIFFERENTIATED_BY_POOL),
    )
    # Assert
    assert type(result) is ResourceModel
    assert len(result.resource_profiles) == 2
    assert type(result.resource_profiles[0]) is ResourceProfile
    assert len(result.resource_calendars) == 2
    assert type(result.resource_calendars[0]) is RCalendar
    assert len(result.activity_resource_distributions) == 5
    assert type(result.activity_resource_distributions[0]) is ActivityResourceDistribution
    assert (
        len(
            [
                distribution.resource_id
                for activity in result.activity_resource_distributions
                for distribution in activity.activity_resources_distributions
            ]
        )
        == 8
    )
