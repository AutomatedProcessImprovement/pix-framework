from pathlib import Path

import pytest

from pix_framework.discovery.resource_profiles import discover_undifferentiated_resource_profile, \
    discover_differentiated_resource_profiles, discover_pool_resource_profiles
from pix_framework.io.event_log import read_csv_log
from pix_framework.io.event_log import APROMORE_LOG_IDS

assets_dir = Path(__file__).parent.parent / "assets"


@pytest.mark.integration
@pytest.mark.parametrize('log_name', ['Resource_profiles_test.csv'])
def test_resource_profiles_undifferentiated(log_name):
    log_path = assets_dir / log_name
    log_ids = APROMORE_LOG_IDS
    log = read_csv_log(log_path, log_ids)

    # Discover undifferentiated profile keeping log resource names
    undifferentiated_profile = discover_undifferentiated_resource_profile(event_log=log, log_ids=log_ids)
    # Assert discovered profile name is expected
    assert undifferentiated_profile is not None
    assert undifferentiated_profile.name == 'Undifferentiated_resource_profile'
    # Assert the resources are the ones from the log
    log_resources = list(log[log_ids.resource].unique())
    profile_resources = [resource.name for resource in undifferentiated_profile.resources]
    assert sorted(profile_resources) == sorted(log_resources)
    # Assert the resources have all activities assigned to them
    log_activities = sorted(log[log_ids.activity].unique())
    for resource in undifferentiated_profile.resources:
        assert sorted(resource.assigned_tasks) == log_activities

    # Discover undifferentiated profile keeping log resource names
    undifferentiated_profile = discover_undifferentiated_resource_profile(
        event_log=log,
        log_ids=log_ids,
        keep_log_names=False
    )
    # Assert discovered profile name is expected
    assert undifferentiated_profile is not None
    assert undifferentiated_profile.name == 'Undifferentiated_resource_profile'
    # Assert the number of resources in the log is the same as the amount
    num_log_resources = log[log_ids.resource].nunique()
    assert len(undifferentiated_profile.resources) == 1
    assert undifferentiated_profile.resources[0].amount == num_log_resources
    # Assert the resource has all activities assigned
    log_activities = sorted(log[log_ids.activity].unique())
    assert sorted(undifferentiated_profile.resources[0].assigned_tasks) == log_activities


@pytest.mark.integration
@pytest.mark.parametrize('log_name', ['Resource_profiles_test.csv'])
def test_resource_profiles_differentiated(log_name):
    log_path = assets_dir / log_name
    log_ids = APROMORE_LOG_IDS
    log = read_csv_log(log_path, log_ids)
    # Discover differentiated profiles
    differentiated_profiles = discover_differentiated_resource_profiles(event_log=log, log_ids=log_ids)
    # Assert discovered profile name is expected
    assert differentiated_profiles is not None
    # Assert the resources are the ones from the log
    log_resources = list(log[log_ids.resource].unique())
    assert len(differentiated_profiles) == len(log_resources)
    for differentiated_profile in differentiated_profiles:
        # One resource in this profile
        assert len(differentiated_profile.resources) == 1
        # Name is one of the existing resources
        resource = differentiated_profile.resources[0]
        assert resource.name in log_resources
        # They have their activities assigned to them
        log_activities = sorted(log[log[log_ids.resource] == resource.name][log_ids.activity].unique())
        assert sorted(resource.assigned_tasks) == log_activities


@pytest.mark.integration
@pytest.mark.parametrize('log_name', ['Resource_profiles_test.csv'])
def test_resource_profiles_pools(log_name):
    log_path = assets_dir / log_name
    log_ids = APROMORE_LOG_IDS
    log = read_csv_log(log_path, log_ids)
    # Discover differentiated profiles
    pooled_profiles = discover_pool_resource_profiles(event_log=log, log_ids=log_ids)
    # Assert discovered pools is two
    assert pooled_profiles is not None
    assert len(pooled_profiles) == 2
    # Assert the resources are the ones from the log
    log_resources = list(log[log_ids.resource].unique())
    assert len(
        [resource.name for pool_profile in pooled_profiles for resource in pool_profile.resources]
    ) == len(log_resources)
    # Analyze each pool individually
    for pool_profile in pooled_profiles:
        # Expecting two pools:
        if len(pool_profile.resources) == 2:
            # Jotaro and Jolyne
            log_activities = sorted(
                log[log[log_ids.resource].isin(["Jotaro-000001", "Jolyne-000001"])][log_ids.activity].unique()
            )
            for resource in pool_profile.resources:
                assert resource.name in ["Jotaro-000001", "Jolyne-000001"]
                assert sorted(resource.assigned_tasks) == log_activities
        else:
            # Pucci
            assert len(pool_profile.resources) == 1
            resource = pool_profile.resources[0]
            assert resource.name == "Pucci-000001"
            log_activities = sorted(
                log[log[log_ids.resource] == "Pucci-000001"][log_ids.activity].unique()
            )
            assert sorted(resource.assigned_tasks) == log_activities
