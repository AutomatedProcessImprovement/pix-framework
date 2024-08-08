import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from pix_framework.discovery.attributes.helpers import log_time

DEFAULT_SAMPLING_SIZE = 25000


@log_time
def preprocess_event_log(event_log, avoid_columns, log_ids):
    sorted_log = event_log.sort_values(by=log_ids.end_time)
    encoders = extract_encoders(event_log, avoid_columns)

    # g_log = sample_until_case_end(sorted_log, log_ids, sampling_size)
    # e_log = sample_event_log_by_case(sorted_log, log_ids, sampling_size)

    g_log = sorted_log.copy()
    e_log = sorted_log.copy()

    g_OBS = process_global_attributes(g_log, avoid_columns, encoders.keys(), log_ids)
    e_OBS = process_event_attributes(e_log, avoid_columns, encoders.keys(), log_ids)

    g_dfs = convert_obs_to_dataframe(g_OBS, log_ids, encoders)
    e_dfs = convert_obs_to_dataframe(e_OBS, log_ids, encoders)

    g_dfs = scale_dataframes(g_dfs, encoders.keys())
    e_dfs = scale_dataframes(e_dfs, encoders.keys())

    g_dfs = calculate_difference(g_dfs)
    e_dfs = calculate_difference(e_dfs)

    return g_dfs, e_dfs, encoders


def calculate_difference(dfs):
    for attribute, activity_dfs in dfs.items():
        for activity, df in activity_dfs.items():
            df[f'difference'] = df['current'] - df['previous']
    return dfs


def extract_encoders(event_log, avoid_columns):
    encoders = {}
    for col in event_log.columns:
        if col not in avoid_columns:
            if not np.issubdtype(event_log[col].dtype, np.number):
                unique_values = [''] + event_log[col].dropna().unique().tolist()
                le = LabelEncoder()
                le.fit(unique_values)
                encoders[col] = le
    return encoders


@log_time
def convert_obs_to_dataframe(obs, log_ids, encoders):
    dict = {}
    for (activity, attribute), changes in obs.items():
        df = pd.DataFrame(changes, columns=['previous', 'current', log_ids.case])

        if attribute in encoders:
            encoder = encoders[attribute]
            df['previous'] = df['previous'].apply(lambda x: encoder.transform([x])[0] if pd.notna(x) else -1)
            df['current'] = df['current'].apply(lambda x: encoder.transform([x])[0] if pd.notna(x) else -1)

        if attribute not in dict:
            dict[attribute] = {}
        dict[attribute][activity] = df
    return dict


def is_starting(event):
    return event['type'] == 'START'


def is_completing(event):
    return event['type'] == 'END'


@log_time
def process_global_attributes(event_log, avoid_columns, encoded_columns, log_ids):
    OBS = {}
    T_split = []

    for _, trace in event_log.groupby(log_ids.case):
        for _, event in trace.iterrows():
            T_split.append({'activity': event[log_ids.activity], 'type': 'START', 'time': event[log_ids.start_time]})
            T_split.append({'activity': event[log_ids.activity], 'type': 'END', 'time': event[log_ids.end_time], **event})

    T_split.sort(key=lambda e: e['time'])

    for attribute in event_log.columns:
        if attribute in avoid_columns:
            continue

        is_categorical = attribute in encoded_columns
        C_V = '' if is_categorical else 0
        V_init = {}

        for event in T_split:
            activity = event['activity']
            if is_starting(event):
                V_init[activity] = C_V
            elif is_completing(event):
                next_value = event[attribute] if attribute in event and pd.notnull(event[attribute]) else V_init[activity]
                C_V = next_value

                if (activity, attribute) not in OBS:
                    OBS[(activity, attribute)] = []
                OBS[(activity, attribute)].append((V_init[activity], next_value, event[log_ids.case]))
                V_init[activity] = next_value

    return OBS


@log_time
def process_event_attributes(event_log, avoid_columns, encoded_columns, log_ids):
    OBS = {}
    for _, trace in event_log.groupby(log_ids.case):
        T_split = []

        for _, event in trace.iterrows():
            T_split.append({'activity': event[log_ids.activity], 'type': 'START', 'time': event[log_ids.start_time]})
            T_split.append({'activity': event[log_ids.activity], 'type': 'END', 'time': event[log_ids.end_time], **event})

        T_split.sort(key=lambda e: e['time'])

        for attribute in event_log.columns:
            if attribute in avoid_columns:
                continue

            is_categorical = attribute in encoded_columns
            C_V = '' if is_categorical else 0
            V_init = {}

            for event in T_split:
                activity = event['activity']
                if is_starting(event):
                    V_init[activity] = C_V
                elif is_completing(event):
                    next_value = event[attribute] if attribute in event and pd.notnull(event[attribute]) else V_init[activity]
                    C_V = next_value
                    if (activity, attribute) not in OBS:
                        OBS[(activity, attribute)] = []

                    OBS[(activity, attribute)].append((V_init[activity], next_value, event[log_ids.case]))
                    V_init[activity] = next_value
    return OBS


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
def scale_dataframes(dfs, encoded_columns):
    scaled_dfs = {}
    for attribute, activity_dfs in dfs.items():
        if attribute not in encoded_columns:
            scaled_activity_dfs = {}
            for activity, df in activity_dfs.items():
                df_scaled = scale_data(df, encoded_columns)
                scaled_activity_dfs[activity] = df_scaled
            scaled_dfs[attribute] = scaled_activity_dfs
        else:
            scaled_dfs[attribute] = activity_dfs
    return scaled_dfs


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

