import pandas as pd

from collections import defaultdict, Counter


class MultitaskInfo:
    def __init__(self):
        self.total_tasks = 0
        self.multitask_freq = dict()
        self.max_freq = 0

    def check_started_event(self, current_active_events_count: int):
        if current_active_events_count not in self.multitask_freq:
            self.multitask_freq[current_active_events_count] = 0
        self.multitask_freq[current_active_events_count] += 1
        self.max_freq = max(self.max_freq, current_active_events_count)
        self.total_tasks += 1


def calculate_multitasking(event_log: pd.DataFrame):
    return _calculate_staircase_probabilities(event_log)


def _calculate_staircase_probabilities(event_log: pd.DataFrame):
    resource_multitask_info = _calculate_task_intersections(event_log)
    multitask_probabilities = dict()

    for resource in resource_multitask_info:
        multitask_info = resource_multitask_info[resource]

        multitask_probabilities[resource] = [0.0] * multitask_info.max_freq
        cumul_freq = 0
        for n in range(multitask_info.max_freq - 1, 0, -1):
            cumul_freq += multitask_info.multitask_freq[n] if n in multitask_info.multitask_freq else 0
            multitask_probabilities[resource][n] = cumul_freq / multitask_info.total_tasks

    return multitask_probabilities


def _calculate_task_intersections(event_log: pd.DataFrame):
    resource_multitask_info = dict()
    for resource, group in event_log.groupby('Resource'):
        resource_multitask_info[resource] = MultitaskInfo()
        events = []

        for index, row in group.iterrows():
            events.append((row['start_time'], 'start', index))
            events.append((row['end_time'], 'end', index))

        # Sort the events by time
        events.sort(key=lambda x: x[0])

        active_tasks = set()
        for time, event_type, index in events:
            if event_type == 'start':
                active_tasks.add(index)
                resource_multitask_info[resource].check_started_event(len(active_tasks))
            else:
                active_tasks.remove(index)

    return resource_multitask_info
