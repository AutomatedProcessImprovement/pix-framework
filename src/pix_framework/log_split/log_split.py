import pandas as pd

from pix_framework.log_ids import EventLogIDs


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
            training_full = len(
                event_log[event_log[log_ids.case].isin(training_case_ids)]
            ) >= (training_percentage * total_events)
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
        keys = (
            [log_ids.start_time, log_ids.end_time]
            if log_ids.start_time in event_log.columns
            else [log_ids.end_time]
        )
        sorted_event_log = event_log.sort_values(keys)
    else:
        sorted_event_log = event_log
    # Get the event splitting train and validation
    num_train_events = int(len(event_log) * training_percentage)
    last_training_event = sorted_event_log.head(num_train_events).iloc[-1]
    # Split the log based on the timestamp of the splitting event
    if log_ids.start_time in event_log.columns:
        training_log = event_log[
            event_log[log_ids.start_time] <= last_training_event[log_ids.start_time]
        ]
        validation_log = event_log[
            event_log[log_ids.start_time] > last_training_event[log_ids.start_time]
        ]
    else:
        training_log = event_log[
            event_log[log_ids.end_time] <= last_training_event[log_ids.end_time]
        ]
        validation_log = event_log[
            event_log[log_ids.end_time] > last_training_event[log_ids.end_time]
        ]
    # Remove from validation incomplete traces if needed
    if remove_partial_traces_from_validation:
        training_cases = training_log[log_ids.case].unique()
        validation_log = validation_log[
            ~validation_log[log_ids.case].isin(training_cases)
        ]
    # Return the two splits
    return training_log, validation_log
