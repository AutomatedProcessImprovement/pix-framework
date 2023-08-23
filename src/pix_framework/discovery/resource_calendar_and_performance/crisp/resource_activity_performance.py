from typing import List

import pandas as pd

from pix_framework.calendar.availability import absolute_unavailability_intervals_within
from pix_framework.discovery.resource_calendar_and_performance.crisp.resource_calendar import RCalendar
from pix_framework.discovery.resource_calendar_and_performance.resource_activity_performance import ActivityResourceDistribution, ResourceDistribution
from pix_framework.discovery.resource_profiles import ResourceProfile
from pix_framework.io.event_log import EventLogIDs
from pix_framework.statistics.distribution import get_best_fitting_distribution


def discover_crisp_activity_resource_distributions(
    event_log: pd.DataFrame,
    log_ids: EventLogIDs,
    resource_profiles: List[ResourceProfile],
    resource_calendars: List[RCalendar],
) -> List[ActivityResourceDistribution]:
    """
    Discover the performance (activity duration) for each resource profile in [resource_profiles]. Treats
    each resource profile as a pool with shared performance (i.e., all the resources of a profile will
    have the same performance for an activity A, computed with the durations of the executions of A performed
    by any resource in that profile).

    :param event_log: event log to discover the activity durations from.
    :param log_ids: column IDs of the event log.
    :param resource_profiles: list of resource profiles with their ID and resources.
    :param resource_calendars: list of calendars containing their ID and working intervals.

    :return: list of duration distribution per activity and resource.
    """
    # Go over each resource profile, computing the corresponding activity durations
    activity_resource_distributions = {}
    for resource_profile in resource_profiles:
        assert (
            len(resource_profile.resources) > 0
        ), "Trying to compute activity performance of an empty resource profile."
        # Get the calendar of the resource profile
        calendar_id = resource_profile.resources[0].calendar_id
        calendar = next(calendar for calendar in resource_calendars if calendar.calendar_id == calendar_id)
        # Get the list of resources of this profile and the activities assigned to them
        resources = [resource.id for resource in resource_profile.resources]
        assigned_activities = resource_profile.resources[0].assigned_tasks
        # Filter the log with activities performed by them
        filtered_event_log = event_log[
            event_log[log_ids.activity].isin(assigned_activities) & event_log[log_ids.resource].isin(resources)
        ]
        # For each assigned activity
        for activity_label, events in filtered_event_log.groupby(log_ids.activity):
            # Get their durations
            durations = compute_activity_durations_without_off_duty(events, log_ids, calendar)
            # Compute duration distribution
            duration_distribution = get_best_fitting_distribution(durations).to_prosimos_distribution()
            # Recover activity-resource distribution for this activity or create a new one
            activity_resource_distribution = activity_resource_distributions.get(
                activity_label, ActivityResourceDistribution(activity_label, [])
            )
            # Append distribution to the durations of this activity (per resource)
            for resource in resources:
                activity_resource_distribution.activity_resources_distributions += [
                    ResourceDistribution(resource, duration_distribution)
                ]
            # Add/Update resource distributions of this activity
            activity_resource_distributions[activity_label] = activity_resource_distribution
    # Return list of activity-resource performances
    return list(activity_resource_distributions.values())


def compute_activity_durations_without_off_duty(
    events: pd.DataFrame,
    log_ids: EventLogIDs,
    calendar: RCalendar,
) -> List[float]:
    """
    Returns activity durations without off-duty time.
    """
    # Compute the calendar-aware duration of each event
    calendar_aware_durations = []
    for start, end in events[[log_ids.start_time, log_ids.end_time]].values.tolist():
        # Recover off-duty periods based on given calendar
        unavailable_periods = absolute_unavailability_intervals_within(
            start=start,
            end=end,
            schedule=calendar,
        )
        # Compute total off-duty duration
        unavailable_time = sum(
            [
                (unavailable_period.end - unavailable_period.start).total_seconds()
                for unavailable_period in unavailable_periods
            ]
        )
        # Compute raw duration and subtract off-duty periods
        calendar_aware_durations += [(end - start).total_seconds() - unavailable_time]
    # Return durations without off-duty time
    return calendar_aware_durations
