from typing import Optional

import pandas as pd

from pix_framework.log_ids import EventLogIDs


def read_csv_log(
    log_path,
    log_ids: EventLogIDs,
    missing_resource: Optional[str] = "NOT_SET",
    sort=True,
) -> pd.DataFrame:
    """
    Read an event log from a CSV file given the column IDs in [log_ids]. Set the enabled_time, start_time, and end_time columns to date,
    set the NA resource cells to [missing_value] if not None, and sort by [end, start, enabled].

    :param log_path: path to the CSV log file.
    :param log_ids: IDs of the columns of the event log.
    :param missing_resource: string to set as NA value for the resource column (not set if None).
    :param sort: if true, sort event log by start, end, enabled (if available).

    :return: the read event log,
    """
    # Read log
    event_log = pd.read_csv(log_path)
    # Set case id as object
    event_log = event_log.astype({log_ids.case: object})
    # Fix missing resources (don't do it if [missing_resources] is set to None)
    if missing_resource:
        if log_ids.resource not in event_log.columns:
            event_log[log_ids.resource] = missing_resource
        else:
            event_log[log_ids.resource].fillna(missing_resource, inplace=True)
    # Set resource type to string if numeric
    if log_ids.resource in event_log.columns:
        event_log[log_ids.resource] = event_log[log_ids.resource].apply(str)
    # Convert timestamp value to pd.Timestamp (setting timezone to UTC)
    event_log[log_ids.end_time] = pd.to_datetime(event_log[log_ids.end_time], utc=True)
    if log_ids.start_time in event_log.columns:
        event_log[log_ids.start_time] = pd.to_datetime(
            event_log[log_ids.start_time], utc=True
        )
    if log_ids.enabled_time in event_log.columns:
        event_log[log_ids.enabled_time] = pd.to_datetime(
            event_log[log_ids.enabled_time], utc=True
        )
    # Sort by end time
    if sort:
        if (
            log_ids.start_time in event_log.columns
            and log_ids.enabled_time in event_log.columns
        ):
            event_log = event_log.sort_values(
                [log_ids.start_time, log_ids.end_time, log_ids.enabled_time]
            )
        elif log_ids.start_time in event_log.columns:
            event_log = event_log.sort_values([log_ids.start_time, log_ids.end_time])
        else:
            event_log = event_log.sort_values(log_ids.end_time)
    # Return parsed event log
    return event_log
