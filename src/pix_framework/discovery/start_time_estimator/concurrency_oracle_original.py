import pandas as pd

from pix_framework.io.event_log import EventLogIDs


def add_enabled_times(
    event_log: pd.DataFrame,
    log_ids: EventLogIDs,
    concurrency: dict,
    set_nat_to_first_event: bool = False,
    include_enabling_activity: bool = False,
    consider_start_times: bool = False,
):
    # For each trace in the log, estimate the enabled time of its events
    indexes, enabled_times, enabling_activities = [], [], []

    for case_id, trace in event_log.groupby(log_ids.case):
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

    # Set all trace enabled times (and enabling activities if necessary) at once
    if include_enabling_activity:
        event_log.loc[indexes, log_ids.enabling_activity] = enabling_activities
    event_log.loc[indexes, log_ids.enabled_time] = enabled_times
    event_log[log_ids.enabled_time] = pd.to_datetime(event_log[log_ids.enabled_time], utc=True)


def get_enabling_activity_instance(trace, event, log_ids, consider_start_times, concurrency) -> pd.Series:
    # Get the list of previous end times
    event_end_time = event[log_ids.end_time]
    event_start_time = event[log_ids.start_time]
    event_activity = event[log_ids.activity]
    previous_end_times = trace[
        (trace[log_ids.end_time] < event_end_time)  # i) previous to the current one;
        & (
            (not consider_start_times)
            or (trace[log_ids.end_time] <= event_start_time)  # ii) if parallel check is activated,
        )
        & (~trace[log_ids.activity].isin(concurrency[event_activity]))  # not overlapping;  # iii) with no concurrency;
    ][log_ids.end_time]

    # Get enabling activity instance or NA if none
    if not previous_end_times.empty:
        enabling_activity_instance = trace.loc[previous_end_times.idxmax()]
    else:
        enabling_activity_instance = pd.Series()

    return enabling_activity_instance
