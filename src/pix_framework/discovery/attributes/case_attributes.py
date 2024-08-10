import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from pix_framework.discovery.attributes.helpers import log_time
from pix_framework.discovery.attributes.metrics import calculate_continuous_metrics, calculate_discrete_metrics
from pix_framework.statistics.distribution import get_best_fitting_distribution


def combine_dataframes(dfs):
    combined_dataframes = {}
    for attribute, activity_dfs in dfs.items():
        combined_dataframes[attribute] = pd.concat(activity_dfs.values(), ignore_index=True)
    return combined_dataframes


def check_case_attribute_confidence(df, log_ids):
    unique_value_counts = df.groupby(log_ids.case)['current'].nunique()
    total_cases = unique_value_counts.size
    cases_with_single_value = (unique_value_counts == 1).sum()
    return cases_with_single_value / total_cases


def get_most_frequent_value(series):
    return series.value_counts().idxmax()


@log_time
def discover_case_attributes(e_dfs, attributes_to_discover, encoders, log_ids, confidence_threshold=1.0):
    case_attributes = []
    metrics = {}

    combined_dfs = combine_dataframes(e_dfs)
    for attribute in attributes_to_discover:
        if check_case_attribute_confidence(combined_dfs[attribute], log_ids) < confidence_threshold:
            continue

        is_discrete = attribute in encoders.keys()
        df_train, df_test = train_test_split(combined_dfs[attribute], test_size=0.5, random_state=42)

        X_train = df_train.groupby(log_ids.case)['current'].apply(get_most_frequent_value)
        X_test = df_test.groupby(log_ids.case)['current'].apply(get_most_frequent_value)

        if is_discrete:
            attr, attr_metrics = discover_discrete_case_attribute(X_train, X_test, encoders[attribute], attribute)
        else:
            attr, attr_metrics = discover_continuous_case_attribute(X_train, X_test, attribute)

        case_attributes.append(attr)
        metrics[attribute] = attr_metrics

    return case_attributes, metrics


def discover_discrete_case_attribute(X_train, X_test, encoder, attribute):
    X_train_decoded = encoder.inverse_transform(X_train)
    X_test_decoded = encoder.inverse_transform(X_test)
    unique_values = encoder.classes_

    value_probs = [np.mean(X_train_decoded == val) for val in unique_values]
    y_pred = np.random.choice(unique_values, size=len(X_test_decoded), p=value_probs)

    metrics = calculate_discrete_metrics(X_test_decoded, y_pred, unique_values, encoder)

    value_distribution_raw = {val: np.mean(X_train_decoded == val) for val in unique_values}
    value_distribution = {key: value for key, value in value_distribution_raw.items() if value > 0}

    attribute_info = {
        "name": attribute,
        "type": "discrete",
        "values": [{"key": key, "value": value} for key, value in value_distribution.items()]
    }

    return attribute_info, metrics


def discover_continuous_case_attribute(X_train, X_test, attribute):
    best_distribution = get_best_fitting_distribution(X_train.tolist())
    y_true = X_test.values
    y_pred = np.random.normal(loc=np.mean(y_true), scale=np.std(y_true), size=len(y_true))
    metrics = calculate_continuous_metrics(y_true, y_pred)

    attribute_info = {
        "name": attribute,
        "type": "continuous",
        "values": best_distribution.to_prosimos_distribution()
    }

    return attribute_info, metrics
