from dataclasses import dataclass
from typing import List, Optional, Union

import pandas as pd

from pix_framework.discovery.resource_calendar_and_performance.crisp.resource_calendar import RCalendar
from pix_framework.discovery.resource_calendar_and_performance.fuzzy.resource_calendar import FuzzyResourceCalendar
from pix_framework.discovery.resource_calendar_and_performance.calendar_discovery_parameters import (
    CalendarDiscoveryParameters,
    CalendarType,
)
from pix_framework.discovery.resource_calendar_and_performance.fuzzy.discovery import (
    discovery_fuzzy_resource_calendars_and_performances,
)
from pix_framework.discovery.resource_calendar_and_performance.resource_activity_performance import (
    ActivityResourceDistribution,
)
from pix_framework.discovery.resource_calendar_and_performance.crisp.discovery import (
    discover_crisp_resource_calendars_per_profile,
)
from pix_framework.discovery.resource_calendar_and_performance.crisp.resource_activity_performance import (
    discover_crisp_activity_resource_distributions,
)
from pix_framework.discovery.resource_profiles import (
    ResourceProfile,
    discover_differentiated_resource_profiles,
    discover_pool_resource_profiles,
    discover_undifferentiated_resource_profile,
)
from pix_framework.io.event_log import EventLogIDs


@dataclass
class ResourceModel:
    """
    Simulation model parameters containing the resource profiles, their calendars and their performance per activity.
    """

    resource_profiles: List[ResourceProfile]
    resource_calendars: Union[List[RCalendar], List[FuzzyResourceCalendar]]
    activity_resource_distributions: List[ActivityResourceDistribution]

    def to_dict(self) -> dict:
        return {
            "resource_profiles": [resource_profile.to_dict() for resource_profile in self.resource_profiles],
            "resource_calendars": [calendar.to_dict() for calendar in self.resource_calendars],
            "task_resource_distribution": [
                activity_resources.to_dict() for activity_resources in self.activity_resource_distributions
            ],
        }

    @staticmethod
    def from_dict(resource_model: dict) -> "ResourceModel":
        if len(resource_model["resource_calendars"]) > 0:
            if "workload_ratio" in resource_model["resource_calendars"][0]:
                calendars = [
                    FuzzyResourceCalendar.from_dict(calendar_dict)
                    for calendar_dict in resource_model["resource_calendars"]
                ]
            else:
                calendars = [
                    RCalendar.from_dict(calendar_dict)
                    for calendar_dict in resource_model["resource_calendars"]
                ]
        else:
            calendars = []

        return ResourceModel(
            resource_profiles=[
                ResourceProfile.from_dict(resource_profile) for resource_profile in resource_model["resource_profiles"]
            ],
            resource_calendars=calendars,
            activity_resource_distributions=[
                ActivityResourceDistribution.from_dict(activity_resource_distribution)
                for activity_resource_distribution in resource_model["task_resource_distribution"]
            ],
        )


def discover_resource_model(
    event_log: pd.DataFrame,
    log_ids: EventLogIDs,
    params: CalendarDiscoveryParameters,
    provided_profiles: Optional[List[ResourceProfile]] = None,
) -> ResourceModel:
    """
    Discover resource model parameters composed by the resource profiles, their calendars, and the resource-activity
    duration distributions.

    :param event_log: event log to discover the resource profiles, calendars, and performances from.
    :param log_ids: column IDs of the event log.
    :param params: parameters for the calendar discovery composed of the calendar type (default 24/7,
    default 9/5 undifferentiated, differentiates, or pools), and, if needed, the parameters for their discovery.
    :param provided_profiles: list of provided resource profiles to use instead of discovering them (mainly for
    performance issues when discovering pools repeatedly).

    :return: class with the resource profiles, their calendars, and the resource-activity duration distributions.
    """
    if params.discovery_type == CalendarType.DIFFERENTIATED_BY_RESOURCE_FUZZY:
        assert (
            log_ids.enabled_time in event_log.columns and not event_log[log_ids.enabled_time].isna().any()
        ), "Enabled time must be present in the event log for fuzzy calendars discovery"

    if provided_profiles is None:
        resource_profiles = _discover_resource_profiles(params.discovery_type, event_log, log_ids)
        assert len(resource_profiles) > 0, "No resource profiles found"
    else:
        resource_profiles = provided_profiles  # Skipping resource profile discovery, using provided

    if params.discovery_type == CalendarType.DIFFERENTIATED_BY_RESOURCE_FUZZY:
        # Fuzzy resource calendars and activity-resource distribution discovery
        default_granularity = 15
        resource_calendars, activity_resource_distributions = discovery_fuzzy_resource_calendars_and_performances(
            log=event_log,
            log_ids=log_ids,
            granularity=params.granularity or default_granularity,
            fuzzy_angle=params.fuzzy_angle
        )
    else:
        # Crisp resource calendars and activity-resource distribution discovery
        resource_calendars = discover_crisp_resource_calendars_per_profile(
            event_log, log_ids, params, resource_profiles
        )
        activity_resource_distributions = discover_crisp_activity_resource_distributions(
            event_log, log_ids, resource_profiles, resource_calendars
        )
    assert len(resource_calendars) > 0, "No resource calendars found"
    assert len(activity_resource_distributions) > 0, "No activity resource distributions found"

    return ResourceModel(
        resource_profiles=resource_profiles,
        resource_calendars=resource_calendars,
        activity_resource_distributions=activity_resource_distributions,
    )


def _discover_resource_profiles(
    calendar_type: CalendarType,
    event_log: pd.DataFrame,
    log_ids: EventLogIDs,
) -> List[ResourceProfile]:
    if calendar_type in [
        CalendarType.DEFAULT_24_7,
        CalendarType.DEFAULT_9_5,
        CalendarType.UNDIFFERENTIATED,
    ]:
        resource_profiles = [discover_undifferentiated_resource_profile(event_log, log_ids)]
    elif calendar_type in [CalendarType.DIFFERENTIATED_BY_RESOURCE, CalendarType.DIFFERENTIATED_BY_RESOURCE_FUZZY]:
        resource_profiles = discover_differentiated_resource_profiles(event_log, log_ids)
    elif calendar_type in [CalendarType.DIFFERENTIATED_BY_POOL]:
        resource_profiles = discover_pool_resource_profiles(event_log, log_ids)
    else:
        raise ValueError(f"Unknown calendar discovery type: {calendar_type}")
    return resource_profiles
