from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import pandas as pd
import polars as pl

from pix_framework.io.event_log import EventLogIDs


def add_enabled_times(
    event_log: pd.DataFrame,
    log_ids: EventLogIDs,
    concurrency: dict,
    set_nat_to_first_event: bool = False,
    include_enabling_activity: bool = False,
    consider_start_times: bool = False,
):
    """
    Optimized version of add_enabled_times that uses Polars.
    """
    # For each trace in the log, estimate the enabled time of its events
    indexes, enabled_times, enabling_activities = [], [], []

    event_log["index"] = event_log.index
    event_log_rs = pl.from_pandas(event_log)
    with ProcessPoolExecutor() as executor:
        handles = [
            executor.submit(
                _estimate_enabled_time_per_trace_rs,
                trace=trace,
                log_ids=log_ids,
                concurrency=concurrency,
                set_nat_to_first_event=set_nat_to_first_event,
                include_enabling_activity=include_enabling_activity,
                consider_start_times=consider_start_times,
            )
            for _, trace in event_log_rs.groupby(log_ids.case)
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
    event_log.drop(columns=["index"], inplace=True)


def _estimate_enabled_time_per_trace_rs(
    trace: pl.DataFrame,
    log_ids: EventLogIDs,
    concurrency: dict,
    set_nat_to_first_event: bool = False,
    include_enabling_activity: bool = False,
    consider_start_times: bool = False,
):
    indexes, enabled_times, enabling_activities = [], [], []

    # Compute trace start time
    if log_ids.start_time in trace:
        trace_start_time = trace.select(
            pl.min(pl.col(log_ids.start_time).min(), pl.col(log_ids.end_time).min())
        ).to_series()[0]
    else:
        trace_start_time = trace.select(pl.col(log_ids.end_time).min()).to_series()[0]

    # Get the enabling activity of each event
    for event in trace.iter_rows(named=True):
        indexes += [event["index"]]
        enabling_activity_instance = _enabling_activity_instance_rs(
            trace=trace,
            event=event,
            log_ids=log_ids,
            concurrency=concurrency,
            consider_start_times=consider_start_times,
        )
        # Store enabled time
        if enabling_activity_instance:
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


def _enabling_activity_instance_rs(
    trace: pl.DataFrame,
    event: dict,
    log_ids: EventLogIDs,
    concurrency: dict,
    consider_start_times: bool = False,
) -> Optional[dict]:
    # Get the list of previous end times
    event_end_time = event[log_ids.end_time]
    event_start_time = event[log_ids.start_time]
    event_activity = event[log_ids.activity]
    enabling_activity_instance = (
        trace.filter(
            (pl.col(log_ids.end_time) < event_end_time)  # i) previous to the current one;
            & (
                (not consider_start_times) | (pl.col(log_ids.end_time) <= event_start_time)
            )  # ii) if parallel check is activated,
            & (pl.col(log_ids.activity).is_in(concurrency[event_activity]).is_not())  # iii) with no concurrency;
        )
        .sort(log_ids.end_time, descending=True)
        .select(log_ids.end_time, log_ids.activity)
        .to_dicts()
    )

    if len(enabling_activity_instance) > 0:
        return enabling_activity_instance[0]
    else:
        return None
