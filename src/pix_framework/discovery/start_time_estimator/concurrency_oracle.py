from collections import Counter
from typing import Optional

import pandas as pd

from pix_framework.io.event_log import EventLogIDs

from .config import Configuration
from .utils import zip_with_next


class ConcurrencyOracle:
    def __init__(self, concurrency: dict, config: Configuration):
        # Dict with the concurrency: self.concurrency[A] = set of activities concurrent with A
        self.concurrency = concurrency
        # Configuration parameters
        self.config = config
        # Set log IDs to ease access within class
        self.log_ids = config.log_ids

    def enabled_since(self, trace, event) -> pd.Timestamp:
        # Get enabling activity instance or NA if none
        enabling_activity_instance = self.enabling_activity_instance(trace, event)
        # Return enabling activity instance
        return (
            enabling_activity_instance[self.log_ids.end_time]
            if isinstance(enabling_activity_instance, pd.Series)
            else pd.NaT
        )

    def enabling_activity_instance(self, trace, event) -> Optional[pd.Series]:
        # Get the list of previous end times
        previous_end_times = trace[
            (trace[self.log_ids.end_time] < event[self.log_ids.end_time])
            & (  # i) previous to the current one;
                (not self.config.consider_start_times)
                or (  # ii) if parallel check is activated,
                    trace[self.log_ids.end_time] <= event[self.log_ids.start_time]
                )
            )
            & (  # not overlapping;
                ~trace[self.log_ids.activity].isin(self.concurrency[event[self.log_ids.activity]])
            )  # iii) with no concurrency;
        ][self.log_ids.end_time]
        # Get enabling activity instance or NA if none
        enabling_activity_instance = trace.loc[previous_end_times.idxmax()] if len(previous_end_times) > 0 else None
        # Return enabling activity instance
        return enabling_activity_instance

    def add_enabled_times(
        self, event_log: pd.DataFrame, set_nat_to_first_event: bool = False, include_enabling_activity: bool = False
    ):
        """
        Add the enabled time of each activity instance to the received event log based on the concurrency relations established in the
        class instance (extracted from the event log passed to the instantiation). For the first event on each trace, set the start of the
        trace as value.

        :param event_log:                   event log to add the enabled time information to.
        :param set_nat_to_first_event:      if False, use the start of the trace as enabled time for the activity instances with no previous
                                            activity enabling them, otherwise use pd.NaT.
        :param include_enabling_activity:   if True, add a column with the label of the activity enabling the current one.
        """
        # For each trace in the log, estimate the enabled time of its events
        indexes, enabled_times, enabling_activities = [], [], []
        for case_id, trace in event_log.groupby(self.log_ids.case):
            # Compute trace start time
            if self.log_ids.start_time in trace:
                trace_start_time = min(trace[self.log_ids.start_time].min(), trace[self.log_ids.end_time].min())
            else:
                trace_start_time = trace[self.log_ids.end_time].min()
            # Get the enabling activity of each event
            for index, event in trace.iterrows():
                indexes += [index]
                enabling_activity_instance = self.enabling_activity_instance(trace, event)
                # Store enabled time
                if isinstance(enabling_activity_instance, pd.Series):
                    # Use computed value
                    enabling_activity_label = enabling_activity_instance[self.log_ids.activity]
                    enabled_times += [enabling_activity_instance[self.log_ids.end_time]]
                else:
                    # No enabling activity, use trace start or NA
                    enabling_activity_label = pd.NA
                    enabled_times += [pd.NaT] if set_nat_to_first_event else [trace_start_time]
                # Store enabled activity label if necessary
                if include_enabling_activity:
                    enabling_activities += [enabling_activity_label]
        # Set all trace enabled times (and enabling activities if necessary) at once
        if include_enabling_activity:
            event_log.loc[indexes, self.log_ids.enabling_activity] = enabling_activities
        event_log.loc[indexes, self.log_ids.enabled_time] = enabled_times
        event_log[self.log_ids.enabled_time] = pd.to_datetime(event_log[self.log_ids.enabled_time], utc=True)


class DeactivatedConcurrencyOracle(ConcurrencyOracle):
    def __init__(self, config: Configuration):
        # Super (with empty concurrency)
        super(DeactivatedConcurrencyOracle, self).__init__({}, config)

    def enabled_since(self, trace, event) -> pd.Timestamp:
        return pd.NaT

    def enabling_activity_instance(self, trace, event) -> Optional[pd.Series]:
        return None


class DirectlyFollowsConcurrencyOracle(ConcurrencyOracle):
    def __init__(self, event_log: pd.DataFrame, config):
        # Default with no concurrency (all directly-follows relations)
        activities = event_log[config.log_ids.activity].unique()
        concurrency = {activity: set() for activity in activities}
        # Super
        super(DirectlyFollowsConcurrencyOracle, self).__init__(concurrency, config)


class AlphaConcurrencyOracle(ConcurrencyOracle):
    def __init__(self, event_log: pd.DataFrame, config: Configuration):
        # Alpha concurrency
        # Initialize dictionary for directly-follows relations df_relations[A][B] = number of times B following A
        df_relations = _get_df_relations(event_log, config.log_ids)
        # Create concurrency if there is a directly-follows relation in both directions
        concurrency = {}
        activities = event_log[config.log_ids.activity].unique()
        for act_a in activities:
            concurrency[act_a] = set()
            for act_b in activities:
                if act_a != act_b and act_a in df_relations.get(act_b, []) and act_b in df_relations.get(act_a, []):
                    # Concurrency relation AB, add it to A
                    concurrency[act_a].add(act_b)
        # Super
        super(AlphaConcurrencyOracle, self).__init__(concurrency, config)


def _get_df_relations(event_log: pd.DataFrame, log_ids: EventLogIDs) -> dict:
    # Initialize dictionary for directly-follows relations df_relations[A][B] = number of times B following A
    df_relations = {activity: {} for activity in event_log[log_ids.activity].unique()}
    # Fill dictionary with directly-follows relations
    for key, trace in event_log.groupby(log_ids.case):
        for (i, current_event), (j, future_event) in zip_with_next(trace.iterrows()):
            current_activity = current_event[log_ids.activity]
            future_activity = future_event[log_ids.activity]
            # Store df relation
            df_relations[current_activity][future_activity] = df_relations[current_activity].get(future_activity, 0) + 1
    return df_relations


class HeuristicsConcurrencyOracle(ConcurrencyOracle):
    def __init__(self, event_log: pd.DataFrame, config: Configuration):
        # Heuristics concurrency
        activities = event_log[config.log_ids.activity].unique()
        # Get matrices for:
        # - Directly-follows relations: df_count[A][B] = number of times B following A
        # - Directly-follows dependency values: df_dependency[A][B] = value of certainty that there is a df-relation between A and B
        # - Length-2 loop values: l2l_dependency[A][B] = value of certainty that there is a l2l relation between A and B (A-B-A)
        (df_count, df_dependency, l2l_dependency) = _get_heuristics_matrices(event_log, activities, config)
        # Create concurrency if there is a directly-follows relation in both directions
        concurrency = {}
        for act_a in activities:
            concurrency[act_a] = set()
            for act_b in activities:
                if (
                    act_a != act_b
                    and df_count[act_a].get(act_b, 0) > 0  # They are not the same activity
                    and df_count[act_b].get(act_a, 0) > 0  # 'B' follows 'A' at least once
                    and l2l_dependency[act_a].get(act_b, 0)  # 'A' follows 'B' at least once
                    < config.concurrency_thresholds.l2l
                    and abs(df_dependency[act_a].get(act_b, 0))  # 'A' and 'B' are not a length 2 loop
                    < config.concurrency_thresholds.df
                ):  # The df relations are weak
                    # Concurrency relation AB, add it to A
                    concurrency[act_a].add(act_b)
        # Super
        super(HeuristicsConcurrencyOracle, self).__init__(concurrency, config)


def _get_heuristics_matrices(event_log: pd.DataFrame, activities: list, config: Configuration) -> (dict, dict, dict):
    # Initialize dictionary for directly-follows relations df_count[A][B] = number of times B following A
    df_count = {activity: {} for activity in activities}
    # Initialize dictionary for length 2 loops
    l2l_count = {activity: {} for activity in activities}
    # Count directly-follows and l2l relations
    for key, trace in event_log.groupby(config.log_ids.case):
        previous_activity = None
        # Iterate the events of the trace in pairs: (e1, e2), (e2, e3), (e3, e4)...
        for (i, current_event), (j, future_event) in zip_with_next(trace.iterrows()):
            current_activity = current_event[config.log_ids.activity]
            future_activity = future_event[config.log_ids.activity]
            # Store df relation
            df_count[current_activity][future_activity] = df_count[current_activity].get(future_activity, 0) + 1
            # Process l2l
            if previous_activity:
                # Increase value if there is a length 2 loop (A-B-A)
                if previous_activity == future_activity:
                    l2l_count[previous_activity][current_activity] = (
                        l2l_count[previous_activity].get(current_activity, 0) + 1
                    )
            # Save previous activity
            previous_activity = current_activity
    # Fill df and l1l dependency matrices
    df_dependency = {activity: {} for activity in activities}
    l1l_dependency = {activity: 0 for activity in activities}
    for act_a in activities:
        for act_b in activities:
            if act_a != act_b:
                # Process directly follows dependency value A -> B
                ab = df_count[act_a].get(act_b, 0)
                ba = df_count[act_b].get(act_a, 0)
                df_dependency[act_a][act_b] = (ab - ba) / (ab + ba + 1)
            else:
                # Process length 1 loop value
                aa = df_count[act_a].get(act_a, 0)
                l1l_dependency[act_a] = aa / (aa + 1)
    # Fill l2l dependency matrix
    l2l_dependency = {activity: {} for activity in activities}
    for act_a in activities:
        for act_b in activities:
            if (
                act_a != act_b
                and l1l_dependency[act_a] < config.concurrency_thresholds.l1l
                and l1l_dependency[act_b] < config.concurrency_thresholds.l1l
            ):
                # Process directly follows dependency value A -> B
                aba = l2l_count[act_a].get(act_b, 0)
                bab = l2l_count[act_b].get(act_a, 0)
                l2l_dependency[act_a][act_b] = (aba + bab) / (aba + bab + 1)
            else:
                l2l_dependency[act_a][act_b] = 0
    # Return matrices with dependency values
    return df_count, df_dependency, l2l_dependency


class OverlappingConcurrencyOracle(ConcurrencyOracle):
    def __init__(self, event_log: pd.DataFrame, config: Configuration):
        # Get the activity labels
        activities = set(event_log[config.log_ids.activity])
        # Get matrix with the frequency of each activity happening overlapping with the rest and in directly-follows order
        overlapping_relations = _get_overlapping_matrix(event_log, activities, config)
        # Create concurrency if the overlapping relations is higher than the threshold specifies
        concurrency = {activity: set() for activity in activities}
        already_checked = set()
        for act_a in activities:
            # Store as already checked to avoid redundant checks
            already_checked.add(act_a)
            # Get the number of occurrences of A per case
            occurrences_a = Counter(event_log[event_log[config.log_ids.activity] == act_a][config.log_ids.case])
            for act_b in activities - already_checked:
                # Get the number of occurrences of B per case
                occurrences_b = Counter(event_log[event_log[config.log_ids.activity] == act_b][config.log_ids.case])
                # Compute number of times they co-occur
                co_occurrences = sum(
                    [
                        occurrences_a[case_id] * occurrences_b[case_id]
                        for case_id in set(list(occurrences_a.keys()) + list(occurrences_b.keys()))
                    ]
                )
                # Check if the proportion of overlapping occurrences is higher than the established threshold
                if co_occurrences > 0:
                    overlapping_ratio = overlapping_relations[act_a].get(act_b, 0) / co_occurrences
                    if overlapping_ratio >= config.concurrency_thresholds.df:
                        # Concurrency relation AB, add it
                        concurrency[act_a].add(act_b)
                        concurrency[act_b].add(act_a)
        # Set flag to consider start times also when individually checking enabled time
        config.consider_start_times = True
        # Super
        super(OverlappingConcurrencyOracle, self).__init__(concurrency, config)


def _get_overlapping_matrix(event_log: pd.DataFrame, activities: set, config: Configuration) -> dict:
    # Initialize dictionary for overlapping relations df_count[A][B] = number of times B overlaps with A
    overlapping_relations = {activity: {} for activity in activities}
    # Count overlapping relations
    for key, trace in event_log.groupby(config.log_ids.case):
        # Iterate the events of the trace
        for i, event in trace.iterrows():
            current_activity = event[config.log_ids.activity]
            # Get labels of overlapping activity instances
            overlapping_labels = trace[
                (
                    (event[config.log_ids.start_time] < trace[config.log_ids.start_time])
                    & (  # The current event starts while the other
                        trace[config.log_ids.start_time] < event[config.log_ids.end_time]
                    )
                )
                | (  # is being executed; OR
                    (event[config.log_ids.start_time] < trace[config.log_ids.end_time])
                    & (  # the current event ends while the other
                        trace[config.log_ids.end_time] < event[config.log_ids.end_time]
                    )
                )
                | (  # is being executed; OR
                    (trace[config.log_ids.start_time] <= event[config.log_ids.start_time])
                    & (event[config.log_ids.end_time] <= trace[config.log_ids.end_time])  # the other event starts and
                    & (event[config.log_ids.activity] != trace[config.log_ids.activity])  # ends within the current one,
                )  # and it's not the current one.
            ][config.log_ids.activity]
            for overlapping_activity in overlapping_labels:
                overlapping_relations[current_activity][overlapping_activity] = (
                    overlapping_relations[current_activity].get(overlapping_activity, 0) + 1
                )
    # Return matrix with dependency values
    return overlapping_relations
