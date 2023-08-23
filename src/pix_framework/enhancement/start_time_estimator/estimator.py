import math
from statistics import mode

import numpy as np
import pandas as pd

from pix_framework.enhancement.concurrency_oracle import (
    AlphaConcurrencyOracle,
    DeactivatedConcurrencyOracle,
    DirectlyFollowsConcurrencyOracle,
    HeuristicsConcurrencyOracle,
    OverlappingConcurrencyOracle,
)
from pix_framework.enhancement.resource_availability import (
    CalendarResourceAvailability,
    SimpleResourceAvailability,
)
from .config import (
    ConcurrencyOracleType,
    Configuration,
    OutlierStatistic,
    ReEstimationMethod,
    ResourceAvailabilityType,
)


class StartTimeEstimator:
    def __init__(self, event_log: pd.DataFrame, config: Configuration):
        # Set event log
        self.event_log = event_log
        # Set configuration
        self.config = config
        # Set log IDs to ease access within class
        self.log_ids = config.log_ids
        # Set concurrency oracle
        if self.config.concurrency_oracle_type == ConcurrencyOracleType.DEACTIVATED:
            self.concurrency_oracle = DeactivatedConcurrencyOracle(self.config)
        elif self.config.concurrency_oracle_type == ConcurrencyOracleType.DF:
            self.concurrency_oracle = DirectlyFollowsConcurrencyOracle(self.event_log, self.config)
        elif self.config.concurrency_oracle_type == ConcurrencyOracleType.ALPHA:
            self.concurrency_oracle = AlphaConcurrencyOracle(self.event_log, self.config)
        elif self.config.concurrency_oracle_type == ConcurrencyOracleType.HEURISTICS:
            self.concurrency_oracle = HeuristicsConcurrencyOracle(self.event_log, self.config)
        elif self.config.concurrency_oracle_type == ConcurrencyOracleType.OVERLAPPING:
            self.concurrency_oracle = OverlappingConcurrencyOracle(self.event_log, self.config)
        else:
            raise ValueError("No concurrency oracle defined!")
        # Set resource availability
        if self.config.resource_availability_type == ResourceAvailabilityType.SIMPLE:
            self.resource_availability = SimpleResourceAvailability(self.event_log, self.config)
        elif self.config.resource_availability_type == ResourceAvailabilityType.WITH_CALENDAR:
            self.resource_availability = CalendarResourceAvailability(self.event_log, self.config)
        else:
            raise ValueError("No resource availability defined!")

    def estimate(self, replace_recorded_start_times: bool = False) -> pd.DataFrame:
        """
        Estimate the start times of each activity instance in the event log based on the resource availability and enablement times with
        the configuration defined in the parameters.

        :param replace_recorded_start_times:    If 'true', replace the start time column with the estimated start
                                                times, if 'false', the estimation is placed in its own column.

        :return: A copy of the event log with the estimated start time, the resource availability time, and the enablement time for each
        activity instance.
        """
        # Copy self event log to allow lunching this method many times
        event_log = self.event_log.copy()
        # Compute resource availability time if not already in the log
        if self.log_ids.available_time not in event_log.columns:
            self.resource_availability.add_resource_availability_times(event_log)
        # Compute enablement time if not already in the log
        if self.log_ids.enabled_time not in event_log.columns:
            self.concurrency_oracle.add_enabled_times(event_log)
        # Assign estimated start timestamps
        event_log[self.log_ids.estimated_start_time] = event_log[
            [self.log_ids.available_time, self.log_ids.enabled_time]
        ].max(axis=1, skipna=True, numeric_only=False)
        # Reuse current start times as estimation if the option is enabled
        if self.config.reuse_current_start_times:
            event_log.loc[~pd.isna(event_log[self.log_ids.start_time]), self.log_ids.estimated_start_time] = event_log[
                self.log_ids.start_time
            ]
        # Re-estimate as instant those activities declared as instant
        event_log.loc[
            event_log[self.log_ids.activity].isin(self.config.instant_activities), self.log_ids.estimated_start_time
        ] = event_log[self.log_ids.end_time]
        # Re-estimate start time of those events with an estimated duration over the threshold
        if not math.isnan(self.config.outlier_threshold):
            self._re_estimate_durations_over_threshold(event_log)
        # Fix start time of those events for which it could not be estimated (with pd.NaT)
        if self.config.re_estimation_method == ReEstimationMethod.SET_INSTANT:
            self._set_instant_non_estimated_start_times(event_log)
        else:
            self._re_estimate_non_estimated_start_times(event_log)
        # If replacement to true, set estimated as start times
        if replace_recorded_start_times:
            event_log[self.log_ids.start_time] = event_log[self.log_ids.estimated_start_time]
            event_log.drop([self.log_ids.estimated_start_time], axis=1, inplace=True)
        # Return estimated event log
        return event_log

    def _re_estimate_durations_over_threshold(self, event_log: pd.DataFrame):
        # Get only events with estimated start time
        estimated_events = event_log[~pd.isna(event_log[self.log_ids.estimated_start_time])]
        # For each event, if the duration is over the threshold, set the defined statistic
        for activity, events in estimated_events.groupby(self.log_ids.activity):
            statistic_duration = self._apply_statistic(
                events[self.log_ids.end_time] - events[self.log_ids.estimated_start_time]
            )
            duration_limit = self.config.outlier_threshold * statistic_duration
            event_log.loc[
                (event_log[self.log_ids.activity] == activity)
                & (~pd.isna(event_log[self.log_ids.estimated_start_time]))
                & ((event_log[self.log_ids.end_time] - event_log[self.log_ids.estimated_start_time]) > duration_limit),
                self.log_ids.estimated_start_time,
            ] = (
                event_log[self.log_ids.end_time] - duration_limit
            )

    def _set_instant_non_estimated_start_times(self, event_log: pd.DataFrame):
        # Identify events with non_estimated as start time
        # and set their duration to instant
        event_log.loc[
            pd.isna(event_log[self.log_ids.estimated_start_time]), self.log_ids.estimated_start_time
        ] = event_log[self.log_ids.end_time]

    def _re_estimate_non_estimated_start_times(self, event_log: pd.DataFrame):
        # Get only events with estimated start time
        estimated_events = event_log[~pd.isna(event_log[self.log_ids.estimated_start_time])]
        # For each event, if the duration is over the threshold, set the defined statistic
        for activity, events in estimated_events.groupby(self.log_ids.activity):
            durations = (events[self.log_ids.end_time] - events[self.log_ids.estimated_start_time]).values
            statistic_duration = self._get_activity_duration(durations)
            event_log.loc[
                (event_log[self.log_ids.activity] == activity) & pd.isna(event_log[self.log_ids.estimated_start_time]),
                self.log_ids.estimated_start_time,
            ] = (
                event_log[self.log_ids.end_time] - statistic_duration
            )
        # Set remaining non estimated activity instances to instant (those of activities with no estimated time)
        event_log[self.log_ids.estimated_start_time].fillna(event_log[self.log_ids.end_time], inplace=True)

    def _get_activity_duration(self, durations):
        if self.config.re_estimation_method == ReEstimationMethod.MODE:
            return mode(durations)
        elif self.config.re_estimation_method == ReEstimationMethod.MEDIAN:
            return np.median(durations)
        elif self.config.re_estimation_method == ReEstimationMethod.MEAN:
            return np.mean(durations)
        else:
            raise ValueError("Unselected re-estimation method for events with non-estimated start time!")

    def _apply_statistic(self, durations):
        if self.config.outlier_statistic == OutlierStatistic.MODE:
            return mode(durations)
        elif self.config.outlier_statistic == OutlierStatistic.MEDIAN:
            return np.median(durations)
        elif self.config.outlier_statistic == OutlierStatistic.MEAN:
            return np.mean(durations)
        else:
            raise ValueError("Unselected outlier statistic for events with estimated duration over the established!")
