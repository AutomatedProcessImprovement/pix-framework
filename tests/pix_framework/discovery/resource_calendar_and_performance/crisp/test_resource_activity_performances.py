from pathlib import Path

import pytest
from pix_framework.discovery.calendar_discovery_parameters import (
    CalendarDiscoveryParameters,
    CalendarType,
)
from pix_framework.discovery.resource_calendar_and_performance.crisp.discovery import (
    discover_crisp_resource_calendars_per_profile,
)
from pix_framework.discovery.resource_calendar_and_performance.crisp.resource_activity_performance import (
    discover_crisp_activity_resource_distributions,
)
from pix_framework.discovery.resource_profiles import (
    discover_differentiated_resource_profiles,
    discover_pool_resource_profiles,
    discover_undifferentiated_resource_profile,
)
from pix_framework.io.event_log import APROMORE_LOG_IDS, read_csv_log

assets_dir = Path(__file__).parent.parent.parent.parent / "assets"


@pytest.mark.integration
@pytest.mark.parametrize("log_name", ["Resource_profiles_test.csv"])
def test_discover_resource_activity_performances_undifferentiated(log_name):
    log_path = assets_dir / log_name
    log_ids = APROMORE_LOG_IDS
    # Read event log
    log = read_csv_log(log_path, log_ids)

    # Discover resource-activity distributions based on undifferentiated profile and 24/7 calendar
    resource_profiles = [discover_undifferentiated_resource_profile(event_log=log, log_ids=log_ids)]
    resource_calendars = discover_crisp_resource_calendars_per_profile(
        event_log=log,
        log_ids=log_ids,
        params=CalendarDiscoveryParameters(discovery_type=CalendarType.DEFAULT_24_7),
        resource_profiles=resource_profiles,
    )
    activity_resource_distributions = discover_crisp_activity_resource_distributions(
        event_log=log, log_ids=log_ids, resource_profiles=resource_profiles, resource_calendars=resource_calendars
    )
    # Assert
    assert len(activity_resource_distributions) == 5
    for activity_resource_distribution in activity_resource_distributions:
        # Three resources on each activity
        assert len(activity_resource_distribution.activity_resources_distributions) == 3
        # All of them same performance
        assert (
            activity_resource_distribution.activity_resources_distributions[0].distribution
            == activity_resource_distribution.activity_resources_distributions[1].distribution
            == activity_resource_distribution.activity_resources_distributions[2].distribution
        )


@pytest.mark.integration
@pytest.mark.parametrize("log_name", ["Resource_profiles_test.csv"])
def test_discover_resource_activity_performances_differentiated(log_name):
    log_path = assets_dir / log_name
    log_ids = APROMORE_LOG_IDS
    # Read event log
    log = read_csv_log(log_path, log_ids)

    # Discover resource-activity distributions based on undifferentiated profile and 24/7 calendar
    resource_profiles = discover_differentiated_resource_profiles(event_log=log, log_ids=log_ids)
    resource_calendars = discover_crisp_resource_calendars_per_profile(
        event_log=log,
        log_ids=log_ids,
        params=CalendarDiscoveryParameters(discovery_type=CalendarType.DEFAULT_24_7),
        resource_profiles=resource_profiles,
    )
    activity_resource_distributions = discover_crisp_activity_resource_distributions(
        event_log=log, log_ids=log_ids, resource_profiles=resource_profiles, resource_calendars=resource_calendars
    )
    # Assert
    assert len(activity_resource_distributions) == 5
    for activity_resource_distribution in activity_resource_distributions:
        # Expecting Jolyne/Jotaro or Pucci
        if len(activity_resource_distribution.activity_resources_distributions) == 2:
            # Jolyne and Jotaro
            assert activity_resource_distribution.activity_id in ["First-task", "Second-task", "Third-task"]
            for resource_distribution in activity_resource_distribution.activity_resources_distributions:
                assert resource_distribution.resource_id in ["Jotaro-000001", "Jolyne-000001"]
                if resource_distribution.resource_id == "Jotaro-000001":
                    assert resource_distribution.distribution["distribution_name"] == "fix"
                    assert resource_distribution.distribution["distribution_params"][0]["value"] == 3600.0
                else:
                    assert resource_distribution.distribution["distribution_name"] == "fix"
                    assert resource_distribution.distribution["distribution_params"][0]["value"] == 1800.0
        else:
            assert activity_resource_distribution.activity_id in ["Fourth-task", "Fifth-task"]
            assert len(activity_resource_distribution.activity_resources_distributions) == 1
            resource_distribution = activity_resource_distribution.activity_resources_distributions[0]
            assert resource_distribution.resource_id == "Pucci-000001"
            assert resource_distribution.distribution["distribution_name"] == "fix"
            assert resource_distribution.distribution["distribution_params"][0]["value"] == 1800.0


@pytest.mark.integration
@pytest.mark.parametrize("log_name", ["Resource_profiles_test.csv"])
def test_discover_resource_activity_performances_pools(log_name):
    log_path = assets_dir / log_name
    log_ids = APROMORE_LOG_IDS
    # Read event log
    log = read_csv_log(log_path, log_ids)

    # Discover resource-activity distributions based on undifferentiated profile and 24/7 calendar
    resource_profiles = discover_pool_resource_profiles(event_log=log, log_ids=log_ids)
    resource_calendars = discover_crisp_resource_calendars_per_profile(
        event_log=log,
        log_ids=log_ids,
        params=CalendarDiscoveryParameters(discovery_type=CalendarType.DEFAULT_24_7),
        resource_profiles=resource_profiles,
    )
    activity_resource_distributions = discover_crisp_activity_resource_distributions(
        event_log=log, log_ids=log_ids, resource_profiles=resource_profiles, resource_calendars=resource_calendars
    )
    # Assert
    assert len(activity_resource_distributions) == 5
    for activity_resource_distribution in activity_resource_distributions:
        # Expecting Jolyne/Jotaro or Pucci
        if len(activity_resource_distribution.activity_resources_distributions) == 2:
            # Jolyne and Jotaro
            assert activity_resource_distribution.activity_id in ["First-task", "Second-task", "Third-task"]
            # Both of them same performance
            assert (
                activity_resource_distribution.activity_resources_distributions[0].distribution
                == activity_resource_distribution.activity_resources_distributions[1].distribution
            )
        else:
            assert activity_resource_distribution.activity_id in ["Fourth-task", "Fifth-task"]
            assert len(activity_resource_distribution.activity_resources_distributions) == 1
            resource_distribution = activity_resource_distribution.activity_resources_distributions[0]
            assert resource_distribution.resource_id == "Pucci-000001"
            assert resource_distribution.distribution["distribution_name"] == "fix"
            assert resource_distribution.distribution["distribution_params"][0]["value"] == 1800.0
