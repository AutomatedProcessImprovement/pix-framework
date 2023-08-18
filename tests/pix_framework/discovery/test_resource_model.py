from pathlib import Path

import pandas as pd
import pytest
from pix_framework.calendar.crisp_resource_calendar import RCalendar
from pix_framework.discovery.calendar_discovery_parameters import CalendarDiscoveryParameters, CalendarType
from pix_framework.discovery.resource_activity_performance import ActivityResourceDistribution
from pix_framework.discovery.resource_model import ResourceModel, discover_resource_model
from pix_framework.discovery.resource_profiles import ResourceProfile
from pix_framework.discovery.start_time_estimator.concurrency_oracle import OverlappingConcurrencyOracle
from pix_framework.discovery.start_time_estimator.config import ConcurrencyThresholds
from pix_framework.discovery.start_time_estimator.config import Configuration as StartTimeEstimatorConfiguration
from pix_framework.io.event_log import APROMORE_LOG_IDS, DEFAULT_XES_IDS, EventLogIDs, read_csv_log

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
        event_log=log, log_ids=log_ids, params=CalendarDiscoveryParameters(discovery_type=CalendarType.UNDIFFERENTIATED)
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
        event_log=log, log_ids=log_ids, params=CalendarDiscoveryParameters(discovery_type=CalendarType.DEFAULT_24_7)
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
        event_log=log, log_ids=log_ids, params=CalendarDiscoveryParameters(discovery_type=CalendarType.DEFAULT_9_5)
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
        params=CalendarDiscoveryParameters(discovery_type=CalendarType.DIFFERENTIATED_BY_RESOURCE),
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
        params=CalendarDiscoveryParameters(discovery_type=CalendarType.DIFFERENTIATED_BY_POOL),
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


def test_resource_profiles_complete():
    log_path = assets_dir / "BPIC15_1_processed.csv.gz"
    log = read_csv_log(log_path, DEFAULT_XES_IDS)

    model = discover_resource_model(
        log, DEFAULT_XES_IDS, CalendarDiscoveryParameters(discovery_type=CalendarType.DIFFERENTIATED_BY_RESOURCE)
    )

    # Ensure "9264148" in activity_resource_distributions
    resource_id = "9264148"
    assert any(
        [
            resource_id in [d.resource_id for d in ard.activity_resources_distributions]
            for ard in model.activity_resource_distributions
        ]
    )
    # Ensure "9264148" in resource_profiles
    assert any([resource_id in [r.id for r in rp.resources] for rp in model.resource_profiles])


@pytest.mark.integration
@pytest.mark.parametrize("log_name", ["Resource_profiles_test.csv"])
def test_discover_fuzzy_resource_model(log_name):
    log_path = assets_dir / log_name
    log_ids = APROMORE_LOG_IDS
    log = read_csv_log(log_path, log_ids)
    _add_enabled_times(log, log_ids)

    result = discover_resource_model(
        event_log=log,
        log_ids=log_ids,
        params=CalendarDiscoveryParameters(discovery_type=CalendarType.DIFFERENTIATED_BY_RESOURCE_FUZZY),
    )

    assert len(result.resource_profiles) == 3
    # Check the discovery type flag works and discovers fuzzy calendars with probabilities field present
    assert result.resource_calendars[0].intervals[0].probability == 1.0


def _add_enabled_times(log: pd.DataFrame, log_ids: EventLogIDs):
    configuration = StartTimeEstimatorConfiguration(
        log_ids=log_ids,
        concurrency_thresholds=ConcurrencyThresholds(df=0.75),
        consider_start_times=True,
    )
    # The start times are the original ones, so use overlapping concurrency oracle
    OverlappingConcurrencyOracle(log, configuration).add_enabled_times(log)
