import logging
from dataclasses import dataclass
from itertools import groupby
from typing import List, Dict, Optional

import pandas as pd

from pix_framework.log_ids import EventLogIDs


@dataclass
class _CustomLogRecord:
    event_id: int
    timestamp: pd.Timestamp
    lifecycle_type: str  # start or complete
    resource: str


@dataclass
class _AuxiliaryLogRecord:
    event_id: int
    timestamp: pd.Timestamp
    adjusted_duration_s: float


def adjust_durations(
    log: pd.DataFrame,
    log_ids: EventLogIDs,
    verbose=False,
) -> pd.DataFrame:
    """
    Changes end timestamps for multitasking events without changing the overall resource utilization.
    """
    metrics_before = None
    if verbose:
        metrics_before = _resource_metrics(log, log_ids)

    resources = log[log_ids.resource].unique()

    for resource in resources:
        _adjust_duration_for_resource(log, log_ids, resource)

    if verbose and metrics_before is not None:
        metrics = _resource_metrics(log, log_ids)
        logging.info(
            f"Utilization before ({metrics_before['utilization']}) equals the one after ({metrics['utilization']}): ",
            metrics_before["utilization"] == metrics["utilization"],
        )
        logging.info(
            f"Resource events before ({metrics_before['number_of_events']}) equal the one after ({metrics['number_of_events']}):",
            metrics_before["number_of_events"] == metrics["number_of_events"],
        )

    return log


def _resource_metrics(log: pd.DataFrame, log_ids: EventLogIDs) -> dict:
    """
    Calculates resource utilization for each resource in the log.
    """
    resources = log[log_ids.resource].unique()
    utilization = {}
    number_of_events = {}
    for resource in resources:
        events = log[log[log_ids.resource] == resource]
        number_of_events[resource] = len(events)
        max_end = events[log_ids.end_time].max()
        min_start = events[log_ids.start_time].min()
        end_timestamps = events[log_ids.end_time]
        start_timestamps = events[log_ids.start_time]
        result = ((end_timestamps - start_timestamps) / (max_end - min_start)).sum()
        utilization[resource] = result
    return {"utilization": utilization, "number_of_events": number_of_events}


def _adjust_duration_for_resource(
    log: pd.DataFrame, log_ids: EventLogIDs, resource: str
):
    resource_events = log[log[log_ids.resource] == resource]
    data = _make_custom_records(resource_events, log, log_ids)
    aux_log = _make_auxiliary_log(data)
    _update_end_timestamps(aux_log, log, log_ids)


def _make_aux_log(log, log_ids, resource):
    resource_events = log[log[log_ids.resource] == resource]
    data = _make_custom_records(resource_events, log, log_ids)
    return _make_auxiliary_log(data)


def _make_custom_records(
    resource_events: pd.DataFrame, log: pd.DataFrame, log_ids: EventLogIDs
):
    """
    Prepares records for the Sweep Line algorithm.
    """
    data = []

    for i, event in resource_events.iterrows():
        start_timestamp = event[log_ids.start_time]
        end_timestamp = event[log_ids.end_time]
        resource = event[log_ids.resource]
        event_id = log.index[i]
        if start_timestamp == end_timestamp:  # filter out instant events
            continue
        start_item = _CustomLogRecord(
            event_id=event_id,
            timestamp=start_timestamp,
            lifecycle_type="start",
            resource=resource,
        )
        end_item = _CustomLogRecord(
            event_id=event_id,
            timestamp=end_timestamp,
            lifecycle_type="complete",
            resource=resource,
        )
        data.extend([start_item, end_item])

    return data


def _make_auxiliary_log(data: List[_CustomLogRecord]) -> List[_AuxiliaryLogRecord]:
    """
    Adjusts duration for multitasking resources.
    """
    active_set: Dict[int, Optional[_CustomLogRecord]] = {}
    previous_time_s: float = 0
    aux_log = []
    data = sorted(data, key=lambda item: item.timestamp)

    for record in data:
        current_time_s = record.timestamp.timestamp()
        adjusted_duration = 0
        active_set_len = len(active_set)
        if active_set_len > 0:
            adjusted_duration = (current_time_s - previous_time_s) / active_set_len
        aux_log.extend(
            _AuxiliaryLogRecord(
                event_id=e_id,
                timestamp=active_set[e_id].timestamp,
                adjusted_duration_s=adjusted_duration,
            )
            for e_id in active_set
        )
        previous_time_s = record.timestamp.timestamp()
        if record.lifecycle_type == "start":
            active_set[record.event_id] = record
        else:
            del active_set[record.event_id]

    return aux_log


def _update_end_timestamps(
    records: List[_AuxiliaryLogRecord], log: pd.DataFrame, log_ids: EventLogIDs
) -> pd.DataFrame:
    """
    Modifies end timestamp according to the adjusted durations.
    """
    # group-by below works only on sorted data
    records = sorted(records, key=lambda record: record.event_id)

    for event_id, group in groupby(records, lambda record: record.event_id):
        duration = sum(map(lambda record: record.adjusted_duration_s, group))
        log.at[event_id, log_ids.end_time] = log.loc[event_id][
            log_ids.start_time
        ] + pd.Timedelta(duration, unit="s")

    return log
