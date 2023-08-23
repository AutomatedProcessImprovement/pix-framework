import random

import numpy as np
import pandas as pd

from pix_framework.io.event_log import EventLogIDs


def _compute_features_table(
    event_log: pd.DataFrame,
    batched_instances: pd.DataFrame,
    log_ids: EventLogIDs,
    num_batch_ready_negative_events: int = 2,
    num_batch_enabled_negative_events: int = 2,
) -> pd.DataFrame:
    """
    Create a DataFrame with the features of the batch-related events, classifying them into events that activate the batch and events
    that does not activate the batch.

    :param event_log:                           full event log with the batch information already discovered.
    :param batched_instances:                   batched activity instances to extract the features out of them.
    :param log_ids:                             mapping with the IDs of each column in the dataset.
    :param num_batch_ready_negative_events:     number of non-firing instants in between the batch enablement and firing.
    :param num_batch_enabled_negative_events:   number of non-firing instants from the enablement times of each case in the batch.
    :return: A Dataframe with the features of the events activating a batch.
    """
    # Register firing feature for each single activity that is not executed as a batch?
    # - I was thinking of this option, but kinda discarded it. It could be good to also take as observations the firing of the
    #   batched activity as a single activity (not batched) if it is executed some times individually. In this way we could
    #   discover rules like "try to batch it but meanwhile the WT is not more than 1 day", so single activities could be fired
    #   not as part of a batch because the WT is 1 day.
    # - However, in order to be fruitful we have to assume that all the executions (individual and batched) are planned to be
    #   part of a batch, but the individual ones got fired because the firing rule was activated before accumulating more than
    #   1 instance.
    # - The single instances could be executed individually just because the resource wanted, and not related to the firing rule
    #   being activated. Thus, consider them without knowing if they were thought to be a batch could hinder the rules discovery.
    # Register features for each batch instance
    features = []
    for key, batch_instance in batched_instances.groupby([log_ids.batch_id]):
        batch_instance_start = batch_instance[log_ids.start_time].min()
        # Get features of the instant activating the batch instance
        features += [
            _get_features(event_log, batch_instance_start, batch_instance, 1, log_ids)  # Batch fired at this instant
        ]
        # Get features of non-activating instants
        non_activating_instants = []
        # 1 - X events in between the ready time of the batch
        batch_instance_enabled = batch_instance[log_ids.enabled_time].max()
        non_activating_instants += pd.date_range(
            start=batch_instance_enabled, end=batch_instance_start, periods=num_batch_ready_negative_events + 2
        )[1:-1].tolist()
        # 2 - Instants per enablement time of each case
        enable_times = [instant for instant in batch_instance[log_ids.enabled_time] if instant < batch_instance_start]
        non_activating_instants += random.sample(
            enable_times, min(len(enable_times), num_batch_enabled_negative_events)
        )
        # 3 - Obtain the features per instant
        for instant in non_activating_instants:
            if instant < batch_instance_start:
                # Discard the batch cases enabled after the current instant, and then calculate the features of the remaining cases.
                features += [
                    _get_features(
                        event_log, instant, batch_instance[batch_instance[log_ids.enabled_time] <= instant], 0, log_ids
                    )
                ]
    # Transform duration to seconds
    features_table = pd.DataFrame(data=features)
    features_table["instant"] = features_table["instant"].astype(np.int64) / 10**9
    features_table["batch_ready_wt"] = features_table["batch_ready_wt"].apply(lambda t: t.total_seconds())
    features_table["batch_max_wt"] = features_table["batch_max_wt"].apply(lambda t: t.total_seconds())
    # features_table['max_cycle_time'] = features_table['max_cycle_time'].apply(lambda t: t.total_seconds())
    # Return table
    return features_table


def _get_features(
    event_log: pd.DataFrame, instant: pd.Timestamp, batch_instance: pd.DataFrame, outcome: int, log_ids: EventLogIDs
) -> dict:
    """
    Get the features to discover activation rules of a specific instant [instant] in a batch instance [batch_instance].

    :param event_log:       event log with all the activity instances.
    :param instant:         instant of the event to register.
    :param batch_instance:  DataFrame with the activity instances of the batch instance.
    :param outcome:         integer indicating the outcome of this instant, 1 if the batch is fired, 0 if not.
    :param log_ids:         mapping with the IDs of each column in the dataset.

    :return: a dict with the features of this batch instance.
    """
    batch_id = batch_instance[log_ids.batch_id].iloc[0]
    batch_type = batch_instance[log_ids.batch_type].iloc[0]
    activity = batch_instance[log_ids.activity].iloc[0]
    resource = batch_instance[log_ids.resource].iloc[0]
    batch_size = len(batch_instance)
    batch_ready_wt = instant - batch_instance[log_ids.enabled_time].max()
    batch_max_wt = instant - batch_instance[log_ids.enabled_time].min()
    case_ids = batch_instance[log_ids.case].unique()
    # max_cycle_time = (instant - event_log[event_log[log_ids.case].isin(case_ids)][log_ids.start_time].min())
    week_day = instant.day_of_week
    # day_of_month = instant.day
    daily_hour = instant.hour
    # minute_of_day = instant.minute
    # Return the features dict
    return {
        log_ids.batch_id: batch_id,
        log_ids.batch_type: batch_type,
        log_ids.activity: activity,
        log_ids.resource: resource,
        "instant": instant,
        "batch_size": batch_size,
        "batch_ready_wt": batch_ready_wt,
        "batch_max_wt": batch_max_wt,
        # 'max_cycle_time': max_cycle_time,
        "week_day": week_day,
        # 'day_of_month': day_of_month,
        "daily_hour": daily_hour,
        # 'minute': minute_of_day,
        "outcome": outcome,
    }
