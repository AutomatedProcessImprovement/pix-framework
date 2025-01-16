from datetime import datetime, timedelta
from enum import Enum

import pandas as pd


class MultiType(Enum):
    GLOBAL = 1
    LOCAL = 2


class MultitaskInfo:
    def __init__(self):
        self.total_tasks = 0
        self.multitask_freq = dict()
        self.max_multitask = 0

    def check_started_event(self, current_active_events_count: int, tasks_count: int = 1):
        if current_active_events_count not in self.multitask_freq:
            self.multitask_freq[current_active_events_count] = 0
        self.multitask_freq[current_active_events_count] += 1
        self.max_multitask = max(self.max_multitask, current_active_events_count)
        self.total_tasks += tasks_count


class GranuleMultitaskInfo:
    def __init__(self, i_size: int = 60):
        self.i_size = i_size
        self.granularity_info = self.init_weekly_granules()

    def init_weekly_granules(self):
        weekly_interval_info = dict()
        for wd in range(0, 7):
            weekly_interval_info[wd] = [MultitaskInfo() for _ in range(1440 // self.i_size)]
        return weekly_interval_info

    def check_granule(self, wd, gr, parallel_events_count):
        self.granularity_info[wd][gr].check_started_event(parallel_events_count, parallel_events_count)

    def set_total(self, wd, gr, total_tasks):
        self.granularity_info[wd][gr].total_tasks = total_tasks


# class IntervalMultiTask:
#     def __init__(self, i_size=60):
#         self.i_size = i_size
#         self.interval_freq = self.init_weekly_intervals(True)
#         self.max_interval_freq = self.init_weekly_intervals(True)
#         self.multi_interval_freq = self.init_weekly_intervals(False)
#
#     def init_weekly_intervals(self, is_total):
#         weekly_interval_freq = {}
#         for i in range(0, 7):
#             if is_total:
#                 weekly_interval_freq[i] = [0] * (1440 // self.i_size)
#             else:
#                 weekly_interval_freq[i] = [dict() for _ in range(1440 // self.i_size)]
#         return weekly_interval_freq
#
#     def check_granule(self, wd, gr, freq):
#         self.interval_freq[wd][gr] += 1
#         if freq not in self.multi_interval_freq[wd][gr]:
#             self.multi_interval_freq[wd][gr][freq] = 0
#         self.multi_interval_freq[wd][gr][freq] += 1
#         self.max_interval_freq[wd][gr] = max(self.max_interval_freq[wd][gr], freq)
#
#     def temporal_check_valid_frequencies(self):
#         for wd in self.interval_freq:
#             for gr in range(0, len(self.interval_freq[wd])):
#                 cumul = 0
#                 for simul in self.multi_interval_freq[wd][gr]:
#                     cumul += self.multi_interval_freq[wd][gr][simul]
#                 if cumul != self.interval_freq[wd][gr]:
#                     return False
#         return True


def calculate_multitasking(event_log: pd.DataFrame, m_type: MultiType = MultiType.GLOBAL, i_size: int = 60):
    workloads = (event_log['resource'].value_counts() / len(event_log)).to_dict()
    return (_calculate_global_staircase_probabilities(event_log), workloads) if m_type == MultiType.GLOBAL \
        else (_calculate_local_staircase_probabilities(event_log, i_size), workloads)


def _calculate_global_staircase_probabilities(event_log: pd.DataFrame):
    resource_multitask_info = _calculate_task_intersections(event_log)
    multitask_probabilities = dict()

    for resource in resource_multitask_info:
        multitask_info = resource_multitask_info[resource]

        multitask_probabilities[resource] = _calculate_decrease_probability(multitask_info.max_multitask,
                                                                            multitask_info.multitask_freq,
                                                                            multitask_info.total_tasks)
    return multitask_probabilities


def _calculate_local_staircase_probabilities(event_log: pd.DataFrame, i_size: int = 60):
    resource_multitask_info = _calculate_interval_intersection(event_log, i_size)
    probabilities = dict()

    for resource in resource_multitask_info:
        multi_inf = resource_multitask_info[resource]
        probabilities[resource] = _init_local_probability_intervals(i_size)

        for wd in range(0, 7):
            for gr in range(0, len(multi_inf.granularity_info[wd])):
                r_info = multi_inf.granularity_info[wd][gr]
                probabilities[resource][wd][gr] = _calculate_decrease_probability(r_info.max_multitask,
                                                                                  r_info.multitask_freq,
                                                                                  r_info.total_tasks,
                                                                                  True)
    return probabilities


def _calculate_decrease_probability(max_freq: int, obs_freq: dict, total_obs: int, is_local: bool = False):
    result = [0.0] * (max_freq + 1)
    cumul_freq = 0
    for n in range(max_freq, 0, -1):
        mult = n if is_local else 1
        cumul_freq += obs_freq[n] * mult if n in obs_freq else 0
        result[n] = cumul_freq / total_obs if total_obs > 0 else 0.0
    return result


def _init_local_probability_intervals(i_size: int):
    gr_multitask = dict()
    for wd in range(0, 7):
        gr_multitask[wd] = _day_granules(i_size)
    return gr_multitask


def _calculate_task_intersections(event_log: pd.DataFrame):
    resource_multitask_info = dict()
    for resource, group in event_log.groupby('resource'):
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


def _calculate_interval_intersection(event_log: pd.DataFrame, i_size):
    resource_multitask_info = dict()
    for resource, group in event_log.groupby('resource'):
        daily_freqs = dict()
        weekday_freqs = _init_weekly_freq(i_size)
        to_weekday = dict()
        for index, row in group.iterrows():
            _update_day_intervals(daily_freqs, weekday_freqs, to_weekday,
                                  pd.to_datetime(row['start_time']), pd.to_datetime(row['end_time']), i_size)

        resource_multitask_info[resource] = _compute_weekly_freqs(daily_freqs, weekday_freqs, to_weekday, i_size)
    return resource_multitask_info


def _update_day_intervals(daily_freqs, weekday_freqs, to_weekday, start_dt, end_dt, i_size):
    c_dt = start_dt
    max_grl = _total_day_granules(i_size)
    c_grl = _current_granule(c_dt, i_size)

    while c_dt <= end_dt:
        date_str = str(c_dt.date())
        wd = c_dt.weekday()
        if date_str not in daily_freqs:
            daily_freqs[date_str] = _day_granules(i_size)
            to_weekday[date_str] = wd

        daily_freqs[date_str][c_grl] += 1
        weekday_freqs[wd][c_grl] += 1

        c_dt += timedelta(minutes=i_size)
        c_grl = (c_grl + 1) % max_grl


def _compute_weekly_freqs(daily_freqs: dict, weekday_freqs: dict, to_weekday: dict, i_size: int):
    interval_multitask = GranuleMultitaskInfo(i_size)

    for day_str in daily_freqs:
        wd = to_weekday[day_str]
        for gr in range(0, len(daily_freqs[day_str])):
            interval_multitask.check_granule(wd, gr, daily_freqs[day_str][gr])
    # # For testing remove, it seems it is correct, the error must be in the calculation of probabilities
    # for wd in range(0, 7):
    #     to_check = interval_multitask.granularity_info[wd]
    #     for gr in range(0, 24):
    #         cumul = 0
    #         for fr in to_check[gr].multitask_freq:
    #             cumul += fr * to_check[gr].multitask_freq[fr]
    #         if cumul != to_check[gr].total_tasks:
    #             print("hola")

    # for wd in weekday_freqs:
    #     for gr in range(0, len(weekday_freqs[wd])):
    #         interval_multitask.set_total(wd, gr, weekday_freqs[wd][gr])
    return interval_multitask


def _current_granule(dt: datetime, i_size: int):
    return (60 * dt.hour + dt.minute) // i_size


def _day_granules(i_size: int):
    return [0] * (1440 // i_size)


def _total_day_granules(i_size: int):
    return 1440 // i_size


def _init_weekly_freq(i_size):
    weekly_freq = dict()
    for wd in range(0, 7):
        weekly_freq[wd] = [0] * (1440 // i_size)
    return weekly_freq


