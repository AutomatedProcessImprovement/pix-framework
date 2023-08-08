import enum
from dataclasses import dataclass, field

from pix_framework.io.event_log import DEFAULT_CSV_IDS, EventLogIDs


class ReEstimationMethod(enum.Enum):
    SET_INSTANT = 1
    MODE = 2
    MEDIAN = 3
    MEAN = 4


class OutlierStatistic(enum.Enum):
    MODE = 1
    MEDIAN = 2
    MEAN = 3


class ConcurrencyOracleType(enum.Enum):
    DEACTIVATED = 1
    DF = 2
    ALPHA = 3
    HEURISTICS = 4
    OVERLAPPING = 5


class ResourceAvailabilityType(enum.Enum):
    SIMPLE = 1  # Consider all the events that each resource performs
    WITH_CALENDAR = 2  # Future possibility considering also the resource calendars and non-working days


@dataclass
class ConcurrencyThresholds:
    df: float = 0.9
    l2l: float = 0.9
    l1l: float = 0.9


@dataclass
class Configuration:
    """Class storing the configuration parameters for the start time estimation.

    Attributes:
        log_ids                     Identifiers for each key element (e.g. executed activity or resource).
        concurrency_oracle_type     Concurrency oracle to use (e.g. heuristics miner's concurrency oracle).
        resource_availability_type  Resource availability engine to use (e.g. using resource calendars).
        missing_resource            String to identify the events with missing resource (it is avoided in
                                    the resource availability calculation).
        re_estimation_method        Method (e.g. median) to re-estimate the start times that couldn't be
                                    estimated due to lack of resource availability and causal predecessors.
        bot_resources               Set of resource IDs corresponding bots, in order to set their events as
                                    instant.
        instant_activities          Set of instantaneous activities, in order to set their events as instant.
        concurrency_thresholds      Thresholds for the concurrency oracle. The three thresholds [df], [l1l],
                                    and [l2l] are used in the Heuristics oracle. In the overlapping oracle,
                                    only [df] is used.
        reuse_current_start_times   Do not estimate the start times of those activities with already recorded
                                    start time (caution, the instant activities and bot resources will still
                                    be set as instant).
        consider_start_times        Consider start times when checking for the enabled time of an activity in
                                    the concurrency oracle, if 'true', do not consider the events which end
                                    time is after the start time of the current activity instance, they overlap
                                    so no causality between them. In the case of the resource availability, if
                                    'true', search the availability as the previous end before the start of the
                                    current activity, not its end.
        outlier_statistic           Statistic (e.g. median) to calculate the most typical duration from the
                                    distribution of each activity durations to consider and re-estimate the
                                    outlier events which estimated duration is higher.
        outlier_threshold           Threshold to control outliers, those events with estimated durations over
        working_schedules           Dictionary with the resources as key and the working calendars (RCalendar)
                                    as value.
    """

    log_ids: EventLogIDs = field(default_factory=lambda: DEFAULT_CSV_IDS)
    concurrency_oracle_type: ConcurrencyOracleType = ConcurrencyOracleType.HEURISTICS
    resource_availability_type: ResourceAvailabilityType = ResourceAvailabilityType.SIMPLE
    missing_resource: str = "NOT_SET"
    re_estimation_method: ReEstimationMethod = ReEstimationMethod.MEDIAN
    bot_resources: set = field(default_factory=set)
    instant_activities: set = field(default_factory=set)
    concurrency_thresholds: ConcurrencyThresholds = field(default_factory=lambda: ConcurrencyThresholds())
    reuse_current_start_times: bool = False
    consider_start_times: bool = False
    outlier_statistic: OutlierStatistic = OutlierStatistic.MEDIAN
    outlier_threshold: float = float("nan")
    working_schedules: dict = field(default_factory=dict)
