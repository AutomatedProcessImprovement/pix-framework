import pandas as pd
from numpy import mean

from pix_framework.io.event_log import EventLogIDs

from .discovery import discover_batches
from .features_table import _compute_features_table
from .rules import _get_rules, _parse_rules


def discover_batch_processing_and_characteristics(
    event_log: pd.DataFrame,
    log_ids: EventLogIDs,
    batch_min_size: int = 2,
    max_sequential_gap: pd.Timedelta = pd.Timedelta(0),
    resource_aware: bool = False,
) -> list:
    """
    Discover, from [event_log], the activities being processed as a batch, and the characteristics of the batches:
        - The activity being executed.
        - The resources involved in the batch processing
        - The type of batch (most common if more than one)
        - The frequency of that activity occurring as part of a batch
        - The distribution of batch sizes, i.e., for each size, the number of activity instances executed
        as a batch of that size.
        - The distribution of the scaling factor of the duration, i.e., for each batch size, the scaling
        factor of the duration of the activity instances processed in that batch. For example, if the activity
        is processed in a 2-size batch, each activity instance lasts x0.7 what it lasts executed individually.
        - The firing rules that better describe the start of the batch.

    :param event_log:           event log to discover the batches and their characteristics.
    :param log_ids:             mapping with the IDs of each column in the dataset.
    :param batch_min_size:      (for discovery) minimum number of activity instances for a batch to be considered as such.
    :param max_sequential_gap:  (for discovery) maximum time gap (with no processing) between the processing of an activity
                                instance and the next one to be considered as a batch.
    :param resource_aware:      (for characteristics extraction) if True, take into the account both the resource and the
                                executed activity for the characteristics discovery.
    :return: a list with the characteristics of each discovered batch.

    """
    # Discover batch behavior
    batched_event_log = discover_batches(
        event_log=event_log, log_ids=log_ids, batch_min_size=batch_min_size, max_sequential_gap=max_sequential_gap
    )
    # Get the characteristics of each bach
    batch_characteristics = discover_batch_characteristics(
        event_log=batched_event_log, log_ids=log_ids, resource_aware=resource_aware
    )
    # Return characteristics
    return batch_characteristics


def discover_batch_characteristics(event_log: pd.DataFrame, log_ids: EventLogIDs, resource_aware: bool = False) -> list:
    """
    Get the characteristics of the batches present in in [event_log].

    :param event_log:       event log with the batch information already discovered.
    :param log_ids:         mapping with the IDs of each column in the dataset.
    :param resource_aware:  if True, take into the account both the resource and the executed activity
                            for the rules discovery.
    :return: a list with the characteristics of each batch.
    """
    # Prepare datasets based on the type
    if resource_aware:
        keys = [log_ids.activity, log_ids.resource]
    else:
        keys = [log_ids.activity]
    # Calculate features per batch
    batches = []
    for group_key, grouped_instances in event_log.groupby(keys):
        batched_grouped_instances = grouped_instances[~pd.isna(grouped_instances[log_ids.batch_id])]
        # If the activity is executed as a batch any time
        if len(batched_grouped_instances) > 0:
            # Get the batch size distribution and batch frequency
            size_distribution = _get_size_distribution(grouped_instances, log_ids)
            batch_frequency = (sum(size_distribution.values()) - size_distribution[1]) / sum(size_distribution.values())
            # Get the batch duration distribution
            duration_distribution = _get_duration_distribution(grouped_instances, log_ids)
            # Get the features table of the instances in this group
            features_table = _compute_features_table(
                event_log=event_log, batched_instances=batched_grouped_instances, log_ids=log_ids
            ).drop([log_ids.batch_id, log_ids.batch_type, log_ids.resource, log_ids.activity, "instant"], axis=1)
            # Get the activation rules
            firing_rules = {}
            if len(features_table["outcome"].unique()) > 1:
                discovered_rules = _get_rules(features_table, "outcome")
                if len(discovered_rules) > 0:
                    firing_rules["confidence"] = discovered_rules["confidence"]
                    firing_rules["support"] = discovered_rules["support"]
                    firing_rules["rules"] = _parse_rules(discovered_rules["model"])
                # Create batch dictionary
                batches += [
                    {
                        "activity": grouped_instances[log_ids.activity].iloc[0],
                        "resources": list(batched_grouped_instances[log_ids.resource].unique()),
                        "type": batched_grouped_instances[log_ids.batch_type].mode().iloc[0],
                        "batch_frequency": batch_frequency,
                        "size_distribution": size_distribution,
                        "duration_distribution": duration_distribution,
                        "firing_rules": firing_rules,
                    }
                ]
    return batches


def _get_size_distribution(event_log: pd.DataFrame, log_ids: EventLogIDs) -> dict:
    """
    Get, for each observed batch size (1 meaning not batched), the number of activity instances executed in batches of that size.

    :param event_log:       event log with the activity instances of the same activity, or of the same activity and performed by the
                            same result (if [resource_aware] is true.
    :param log_ids:         mapping with the IDs of each column in the dataset.

    :return: a dict with the batch size as keys, and the number of activity instances executed in batches of that size as values.
    """
    sizes = {}
    # For each batched execution, increase one the count of their size
    batched_executions = event_log[~pd.isna(event_log[log_ids.batch_id])]
    for batch_id, events in batched_executions.groupby([log_ids.batch_id]):
        batch_size = len(events)
        sizes[batch_size] = sizes.get(batch_size, 0) + len(events)
    # Add count of single executions
    sizes[1] = len(event_log) - len(batched_executions)
    # Return size distribution
    return sizes


def _get_duration_distribution(event_log: pd.DataFrame, log_ids: EventLogIDs) -> dict:
    """
    Get the distribution of scale factors for the duration of the batched activity, depending on the number of instances batched. For,
    example, an activity can last x0.9 when is executed in a batch of two, and x0.8 if it is executed in a batch of three.

    :param event_log:       event log with the activity instances of the same activity, or of the same activity and performed by the
                            same result (if [resource_aware] is true.
    :param log_ids:         mapping with the IDs of each column in the dataset.

    :return: a dict with the batch size as keys, and the scale factor for the duration of the activity
    instances executed in batches of that size as values.
    """
    # Copy log to edit
    event_log_copy = event_log.copy()
    # Set activity duration as new column
    event_log_copy["duration"] = event_log_copy[log_ids.end_time] - event_log_copy[log_ids.start_time]
    # Save durations of no batched activity instances
    no_batched_durations = list(event_log_copy[pd.isna(event_log_copy[log_ids.batch_id])]["duration"])
    # For each batch size, record its activity duration
    batched_durations = {}
    batched_executions = event_log_copy[~pd.isna(event_log_copy[log_ids.batch_id])]
    for batch_id, events in batched_executions.groupby([log_ids.batch_id]):
        batch_size = len(events)
        batched_durations[batch_size] = batched_durations.get(batch_size, []) + list(events["duration"])
    # Compute scale factor of mean value
    durations = {}
    if len(no_batched_durations) > 0:
        mean_no_batched = mean(no_batched_durations)
        for size in batched_durations.keys():
            durations[size] = mean(batched_durations[size]) / mean_no_batched
    else:
        print("WARNING! No non-batched executions to learn duration scaling factor, setting 1.0 as default.")
        for size in batched_durations.keys():
            durations[size] = 1.0
        pass
    # Return durations
    return durations
