from concurrent.futures import ProcessPoolExecutor
from typing import Optional, Callable, Union

import pandas as pd
import polars as pl

from pix_framework.io.event_log import EventLogIDs


def get_enabling_activity_instance(
    log_ids,
    concurrency,
    consider_start_times,
    trace,
    event,
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


def add_enabled_times(
    event_log: pd.DataFrame,
    log_ids: EventLogIDs,
    set_nat_to_first_event: bool = False,
    include_enabling_activity: bool = False,
    get_enabling_activity_instance_fn=Callable[[pd.DataFrame, pd.Series], pd.Series],
    use_polars: bool = False,
):
    """
    Optimized version of add_enabled_times that uses Polars.
    """
    if use_polars:
        # Polars doesn't use the index, so we need to store it as a column to use for the final update
        event_log["index"] = event_log.index
        # Use Polars DataFrame that has similar API, at least "groupby" is the same
        event_log_rs = pl.from_pandas(event_log)

    indexes, enabled_times, enabling_activities = _collect_enabled_times_and_indices(
        enabling_activity_instance_fn=get_enabling_activity_instance_fn,
        event_log=event_log_rs if use_polars else event_log,
        include_enabling_activity=include_enabling_activity,
        log_ids=log_ids,
        set_nat_to_first_event=set_nat_to_first_event,
        use_polars=use_polars,
    )

    # Update all trace enabled times (and enabling activities if necessary) at once
    if include_enabling_activity:
        event_log.loc[indexes, log_ids.enabling_activity] = enabling_activities
    event_log.loc[indexes, log_ids.enabled_time] = enabled_times
    event_log[log_ids.enabled_time] = pd.to_datetime(event_log[log_ids.enabled_time], utc=True)

    # Drop the index column added at the beginning
    if use_polars:
        event_log.drop(columns=["index"], inplace=True)


def _collect_enabled_times_and_indices(
    enabling_activity_instance_fn: Callable[[pd.DataFrame, pd.Series], pd.Series],
    event_log: Union[pd.DataFrame, pl.DataFrame],
    log_ids: EventLogIDs,
    include_enabling_activity: bool,
    set_nat_to_first_event: bool,
    use_polars: bool = False,
) -> tuple[list[int], list[pd.Timestamp], list[Optional[str]]]:
    indexes, enabled_times, enabling_activities = [], [], []

    # For each trace in the log, estimate the enabled time of its events
    with ProcessPoolExecutor() as executor:
        handles = [
            executor.submit(
                _find_enabled_instance_polars if use_polars else _find_enabled_instance_original,
                trace=trace,
                log_ids=log_ids,
                set_nat_to_first_event=set_nat_to_first_event,
                include_enabling_activity=include_enabling_activity,
                enabling_activity_instance_fn=enabling_activity_instance_fn,
            )
            for _, trace in event_log.groupby(log_ids.case)
        ]
        for handle in handles:
            indexes_, enabled_times_, enabling_activities_ = handle.result()
            indexes += indexes_
            enabled_times += enabled_times_
            enabling_activities += enabling_activities_

    return indexes, enabled_times, enabling_activities


def _find_enabled_instance_original(
    trace: pd.DataFrame,
    log_ids: EventLogIDs,
    set_nat_to_first_event: bool = False,
    include_enabling_activity: bool = False,
    enabling_activity_instance_fn=Callable[[pd.DataFrame, pd.Series, EventLogIDs, bool, dict], pd.Series],
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
        enabling_activity_instance = enabling_activity_instance_fn(trace, event)
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


def _find_enabled_instance_polars(
    trace: pl.DataFrame,
    log_ids: EventLogIDs,
    set_nat_to_first_event: bool = False,
    include_enabling_activity: bool = False,
    enabling_activity_instance_fn=Callable[[pd.DataFrame, pd.Series], pd.Series],
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
        enabling_activity_instance = enabling_activity_instance_fn(
            trace=trace,
            event=event,
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
