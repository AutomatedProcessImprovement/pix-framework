import pandas as pd

from pix_framework.discovery.attributes.helpers import log_time


@log_time
def extract_features(log, attributes_to_discover, log_ids, is_event_log=False):
    if attributes_to_discover is None:
        relevant_cols = [col for col in log.columns if col not in [log_ids.case, log_ids.activity]]
    else:
        relevant_cols = [col for col in attributes_to_discover if col in log.columns]

    if is_event_log:
        features_list = []

        for _, case_data in log.groupby(log_ids.case):
            previous_values = case_data[relevant_cols].shift(1).fillna(0)

            diffs = case_data[relevant_cols].diff().fillna(case_data[relevant_cols])

            case_features = previous_values.rename(columns=lambda x: 'prev_' + x)
            case_features = case_features.assign(**diffs.rename(columns=lambda x: 'diff_' + x))

            case_features[log_ids.case] = case_data[log_ids.case].values
            case_features[log_ids.activity] = case_data[log_ids.activity].values

            features_list.append(case_features)

        features = pd.concat(features_list, ignore_index=True)

    else:
        previous_values = log[relevant_cols].shift(1).fillna(0)
        diffs = log[relevant_cols].diff().fillna(log[relevant_cols])

        features = previous_values.rename(columns=lambda x: 'prev_' + x)
        features = features.assign(**diffs.rename(columns=lambda x: 'diff_' + x))

        features[log_ids.case] = log[log_ids.case].values
        features[log_ids.activity] = log[log_ids.activity].values

    return features
