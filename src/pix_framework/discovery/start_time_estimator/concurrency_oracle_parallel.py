from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from pix_framework.io.event_log import EventLogIDs
from .concurrency_oracle_original import get_enabling_activity_instance


def add_enabled_times(
    event_log: pd.DataFrame,
    log_ids: EventLogIDs,
    concurrency: dict,
    set_nat_to_first_event: bool = False,
    include_enabling_activity: bool = False,
    consider_start_times: bool = False,
):
    indexes, enabled_times, enabling_activities = [], [], []

    with ProcessPoolExecutor() as executor:
        handles = [
            executor.submit(
                _estimate_enabled_time_per_trace,
                trace=trace,
                log_ids=log_ids,
                concurrency=concurrency,
                set_nat_to_first_event=set_nat_to_first_event,
                include_enabling_activity=include_enabling_activity,
                consider_start_times=consider_start_times,
            )
            for case_id, trace in event_log.groupby(log_ids.case)
        ]
        for handle in handles:
            indexes_, enabled_times_, enabling_activities_ = handle.result()
            indexes += indexes_
            enabled_times += enabled_times_
            enabling_activities += enabling_activities_

    # Set all trace enabled times (and enabling activities if necessary) at once
    if include_enabling_activity:
        event_log.loc[indexes, log_ids.enabling_activity] = enabling_activities
    event_log.loc[indexes, log_ids.enabled_time] = enabled_times
    event_log[log_ids.enabled_time] = pd.to_datetime(event_log[log_ids.enabled_time], utc=True)


def _estimate_enabled_time_per_trace(
    trace: pd.DataFrame,
    log_ids: EventLogIDs,
    concurrency: dict,
    set_nat_to_first_event: bool = False,
    include_enabling_activity: bool = False,
    consider_start_times: bool = False,
):
    indexes, enabled_times, enabling_activities = [], [], []

    # Compute trace start time
    if log_ids.start_time in trace:
        trace_start_time = min(trace[log_ids.start_time].min(), trace[log_ids.end_time].min())
    else:
        trace_start_time = trace[log_ids.end_time].min()

    # Get the enabling activity of each event
    for index, event in trace.iterrows():
        indexes += [index]
        enabling_activity_instance = get_enabling_activity_instance(
            trace=trace,
            event=event,
            log_ids=log_ids,
            consider_start_times=consider_start_times,
            concurrency=concurrency,
        )
        # Store enabled time
        if not enabling_activity_instance.empty:
            # Use computed value
            enabling_activity_label = enabling_activity_instance[log_ids.activity]
            enabled_times += [enabling_activity_instance[log_ids.end_time]]
        else:
            # No enabling activity, use trace start or NA
            enabling_activity_label = pd.NA
            enabled_times += [pd.NaT] if set_nat_to_first_event else [trace_start_time]
        # Store enabled activity label if necessary
        if include_enabling_activity:
            enabling_activities += [enabling_activity_label]

    return indexes, enabled_times, enabling_activities
