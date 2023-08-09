from dataclasses import dataclass, fields
from typing import Optional

import pandas as pd


@dataclass
class EventLogIDs:
    # General
    case: str = "case"  # Case ID
    activity: str = "activity"  # Activity label
    resource: str = "resource"  # Resource who performed this activity instance
    start_time: str = "start_time"  # Start time of the activity instance
    end_time: str = "end_time"  # End time of the activity instance
    # Start time estimator
    enabled_time: str = "enabled_time"  # Enablement time of the activity instance
    enabling_activity: str = "enabling_activity"  # Label of the activity instance enabling the current one
    available_time: str = (
        "available_time"  # Last availability time of the resource who performed this activity instance
    )
    estimated_start_time: str = "estimated_start_time"  # Estimated start time of the activity instance
    batch_id: str = "batch_instance_id"  # ID of the batch instance this activity instance belongs to, if any
    batch_type: str = "batch_instance_type"  # Type of the batch instance this activity instance belongs to, if any

    @staticmethod
    def from_dict(config: dict) -> "EventLogIDs":
        return EventLogIDs(**config)

    def to_dict(self) -> dict:
        return {attr.name: getattr(self, attr.name) for attr in fields(self.__class__)}


DEFAULT_CSV_IDS = EventLogIDs(
    case="case_id",
    activity="Activity",
    enabled_time="enabled_time",
    start_time="start_time",
    end_time="end_time",
    available_time="available_time",
    estimated_start_time="estimated_start_time",
    resource="Resource",
    batch_id="batch_instance_id",
    batch_type="batch_instance_type",
)
DEFAULT_XES_IDS = EventLogIDs(
    case="case:concept:name",
    activity="concept:name",
    enabled_time="time:enabled",
    start_time="start_timestamp",  # Compatibility with PM4PY
    end_time="time:timestamp",
    available_time="time:available",
    estimated_start_time="time:estimated_start",
    resource="org:resource",
)
APROMORE_LOG_IDS = EventLogIDs(
    case="Case_ID",
    activity="Activity",
    start_time="Start_Time",
    end_time="End_Time",
    resource="Resource",
)
PROSIMOS_LOG_IDS = EventLogIDs(
    case="case_id",
    activity="activity",
    enabled_time="enabled_time",
    start_time="start_time",
    end_time="end_time",
    resource="resource",
)


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
    event_log[log_ids.end_time] = pd.to_datetime(event_log[log_ids.end_time], utc=True, format="ISO8601")
    if log_ids.start_time in event_log.columns:
        event_log[log_ids.start_time] = pd.to_datetime(event_log[log_ids.start_time], utc=True, format="ISO8601")
    if log_ids.enabled_time in event_log.columns:
        event_log[log_ids.enabled_time] = pd.to_datetime(event_log[log_ids.enabled_time], utc=True, format="ISO8601")
    # Sort by end time
    if sort:
        if log_ids.start_time in event_log.columns and log_ids.enabled_time in event_log.columns:
            event_log = event_log.sort_values([log_ids.start_time, log_ids.end_time, log_ids.enabled_time])
        elif log_ids.start_time in event_log.columns:
            event_log = event_log.sort_values([log_ids.start_time, log_ids.end_time])
        else:
            event_log = event_log.sort_values(log_ids.end_time)
    # Return parsed event log
    return event_log


def split_log_training_validation_trace_wise(
    event_log: pd.DataFrame, log_ids: EventLogIDs, training_percentage: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the traces of [event_log] into two separated event logs (one for training and the other for validation). Split full traces in
    order to achieve an approximate proportion of [training_percentage] events in the training set.

    :param event_log:           event log to split.
    :param log_ids:             IDs for the columns of the event log.
    :param training_percentage: percentage of events (approx) to retain in the training data.

    :return: a tuple with two datasets, the training and the validation ones.
    """
    # Sort event log
    sorted_event_log = event_log.sort_values([log_ids.start_time, log_ids.end_time])
    # Take first trace until the number of events is [training_percentage] * total size
    total_events = len(event_log)
    training_case_ids = []
    training_full = False
    # Go over the case IDs (sorted by start and end time of its events)
    for case_id in sorted_event_log[log_ids.case].unique():
        # The first traces until the size limit is met goes to the training set
        if not training_full:
            training_case_ids += [case_id]
            training_full = len(event_log[event_log[log_ids.case].isin(training_case_ids)]) >= (
                training_percentage * total_events
            )
    # Return the two splits
    return (
        event_log[event_log[log_ids.case].isin(training_case_ids)],
        event_log[~event_log[log_ids.case].isin(training_case_ids)],
    )


def split_log_training_validation_event_wise(
    event_log: pd.DataFrame,
    log_ids: EventLogIDs,
    training_percentage: float,
    sort: bool = True,
    remove_partial_traces_from_validation: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the traces of [event_log] into two separated event logs (one for training and the other for validation). Split event-wise retaining the
    first [training_percentage] of events in the training set, and the remaining ones in the validation set.

    :param event_log:                               event log to split.
    :param log_ids:                                 IDs for the columns of the event log.
    :param training_percentage:                     percentage of events to retain in the training data.
    :param sort:                                    if true, sort events in the log by start+end (if start available) or by end (otherwise).
    :param remove_partial_traces_from_validation    if true, remove from validation set the traces that has been split being some event in
                                                    training and some events in validation.

    :return: a tuple with two datasets, the training and the validation ones.
    """
    # Sort if needed
    if sort:
        keys = [log_ids.start_time, log_ids.end_time] if log_ids.start_time in event_log.columns else [log_ids.end_time]
        sorted_event_log = event_log.sort_values(keys)
    else:
        sorted_event_log = event_log
    # Get the event splitting train and validation
    num_train_events = int(len(event_log) * training_percentage)
    last_training_event = sorted_event_log.head(num_train_events).iloc[-1]
    # Split the log based on the timestamp of the splitting event
    if log_ids.start_time in event_log.columns:
        training_log = event_log[event_log[log_ids.start_time] <= last_training_event[log_ids.start_time]]
        validation_log = event_log[event_log[log_ids.start_time] > last_training_event[log_ids.start_time]]
    else:
        training_log = event_log[event_log[log_ids.end_time] <= last_training_event[log_ids.end_time]]
        validation_log = event_log[event_log[log_ids.end_time] > last_training_event[log_ids.end_time]]
    # Remove from validation incomplete traces if needed
    if remove_partial_traces_from_validation:
        training_cases = training_log[log_ids.case].unique()
        validation_log = validation_log[~validation_log[log_ids.case].isin(training_cases)]
    # Return the two splits
    return training_log, validation_log
