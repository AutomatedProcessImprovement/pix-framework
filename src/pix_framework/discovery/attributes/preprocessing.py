import pandas as pd
import numpy as np

from helpers import log_time
from sklearn.preprocessing import LabelEncoder

DEFAULT_SAMPLING_SIZE = 25000


@log_time
def preprocess_event_log(event_log, log_ids, sampling_size=DEFAULT_SAMPLING_SIZE):
    sorted_log = event_log.sort_values(by=log_ids.end_time)
    columns_to_drop = [getattr(log_ids, attr) for attr in vars(log_ids) if getattr(log_ids, attr) in sorted_log.columns]
    system_columns_to_keep = [log_ids.case, log_ids.activity]
    columns_to_drop = [x for x in columns_to_drop if x not in system_columns_to_keep]

    g_log = sample_until_case_end(sorted_log, log_ids, sampling_size)
    g_log.drop(columns=columns_to_drop, inplace=True)
    g_log = fill_nans(g_log, log_ids, False)

    e_log = sample_event_log_by_case(sorted_log, log_ids, sampling_size)
    e_log.drop(columns=columns_to_drop, inplace=True)
    e_log = fill_nans(e_log, log_ids, True)

    empty_series = pd.Series([''])
    encoders = {}
    for col in g_log.columns:

        if g_log[col].dtype == 'object':
            g_log[col] = g_log[col].astype(str)
            e_log[col] = e_log[col].astype(str)

            all_values = pd.concat([empty_series, g_log[col], e_log[col]]).unique()

            encoder = LabelEncoder()
            encoder.fit(all_values)

            g_log[col] = encoder.transform(g_log[col])
            e_log[col] = encoder.transform(e_log[col])

            encoders[col] = encoder

    g_log = scale_data(g_log, encoders.keys())
    e_log = scale_data(e_log, encoders.keys())

    return g_log, e_log, encoders


@log_time
def sample_event_log_by_case(event_log, log_ids, sampling_size=DEFAULT_SAMPLING_SIZE):
    event_log_copy = event_log.copy(deep=True)
    event_log_reset = event_log_copy.reset_index(drop=True)
    event_log_sorted = event_log_reset.sort_values(by=[log_ids.case, log_ids.end_time])

    unique_cases = event_log_sorted[log_ids.case].unique()

    if len(unique_cases) <= sampling_size:
        return event_log_sorted

    step_size = max(len(unique_cases) // sampling_size, 1)
    sampled_cases = unique_cases[::step_size]

    sampled_log = event_log_sorted[event_log_sorted[log_ids.case].isin(sampled_cases)]

    return sampled_log


@log_time
def sample_until_case_end(event_log, log_ids, sampling_size=DEFAULT_SAMPLING_SIZE):
    event_log_copy = event_log.copy(deep=True)
    event_log_copy.sort_values(log_ids.end_time, inplace=True)
    unique_cases = event_log_copy[log_ids.case].unique()
    if len(unique_cases) <= sampling_size:
        return event_log_copy

    nth_case_id = unique_cases[sampling_size - 1]
    last_index_of_nth_case = event_log_copy[event_log_copy[log_ids.case] == nth_case_id].index[-1]
    return event_log_copy.loc[:last_index_of_nth_case]


@log_time
def fill_nans(log, log_ids, is_event_log=False):
    numeric_columns = log.select_dtypes(include=['number']).columns
    object_columns = log.select_dtypes(include=['object']).columns

    if is_event_log:
        for case_id, group in log.groupby(log_ids.case):
            log.loc[group.index, numeric_columns] = group[numeric_columns].fillna(method='ffill', axis=0).fillna(0)
            log.loc[group.index, object_columns] = group[object_columns].fillna(method='ffill', axis=0).fillna('')
    else:
        log[numeric_columns] = log[numeric_columns].fillna(method='ffill', axis=0).fillna(0)
        log[object_columns] = log[object_columns].fillna(method='ffill', axis=0).fillna('')

    return log


@log_time
def scale_data(log, encoded_columns):
    log_scaled = log.copy()
    max_float32 = np.finfo(np.float32).max
    min_float32 = -max_float32
    tiny_float32 = np.finfo(np.float32).tiny

    for col in log.columns:
        if log[col].dtype == 'object' or col in encoded_columns:
            continue

        if np.issubdtype(log[col].dtype, np.number):
            log_scaled[col] = np.where((log[col] > 0) & (log[col] < tiny_float32), tiny_float32, log[col])
            log_scaled[col] = np.where((log[col] < 0) & (log[col] > -tiny_float32), -tiny_float32, log[col])

            log_scaled[col] = np.clip(log_scaled[col], min_float32, max_float32)

    return log_scaled

