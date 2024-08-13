import numpy as np
import pandas as pd

from pix_framework.discovery.gateway_conditions.helpers import log_time
from pandas.api.types import is_numeric_dtype
from pix_framework.io.event_log import EventLogIDs


@log_time
def preprocess_event_log(event_log, log_ids, sampling_size):
    sorted_log = event_log.sort_values(by=log_ids.end_time)

    log_by_case = sample_event_log_by_case(sorted_log, log_ids, sampling_size)
    log_by_case = fill_nans(log_by_case, log_ids)

    string_values = list()
    for col in log_by_case.columns:
        if log_by_case[col].dtype == 'object':
            string_values.append(col)

    log_by_case = scale_data(log_by_case, string_values)
    return log_by_case


@log_time
def sample_event_log_by_case(event_log: pd.DataFrame, log_ids: EventLogIDs, sampling_size):
    event_log_reset = event_log.reset_index()
    event_log_sorted = event_log_reset.sort_values(by=[log_ids.case, 'index'])

    unique_cases = event_log_sorted[log_ids.case].unique()

    if len(unique_cases) <= sampling_size:
        return event_log_sorted

    step_size = len(unique_cases) // sampling_size
    sampled_cases = unique_cases[::step_size]

    sampled_log = event_log_sorted[event_log_sorted[log_ids.case].isin(sampled_cases)]
    sampled_log = sampled_log.drop(columns=['index'])

    return sampled_log


@log_time
def fill_nans(log, log_ids):
    def replace_initial_nans(series):
        if is_numeric_dtype(series):
            initial_value = 0
        else:
            initial_value = ''
        first_valid_index = series.first_valid_index()
        if first_valid_index is not None:
            series[:first_valid_index] = series[:first_valid_index].fillna(initial_value)
        return series

    def preprocess_case(case_data):
        return case_data.apply(replace_initial_nans)

    preprocessed_log = (log.groupby(log_ids.case).apply(preprocess_case)).reset_index(drop=True)
    preprocessed_log = preprocessed_log.ffill()
    return preprocessed_log


@log_time
def scale_data(log, avoid_columns, threshold=1e+37):
    log_scaled = log.copy()
    max_float = np.finfo(np.float32).max
    min_float = np.finfo(np.float32).tiny

    for col in log.columns:
        if col not in avoid_columns and is_numeric_dtype(log[col]):
            if (log[col] > threshold).any() or (log[col] < -threshold).any():
                log_scaled.loc[log_scaled[col] > threshold, col] = max_float
                log_scaled.loc[log_scaled[col] < -threshold, col] = min_float

    return log_scaled
