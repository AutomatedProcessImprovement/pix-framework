from helpers import log_time
from pix_framework.statistics.distribution import get_best_fitting_distribution
from sklearn.model_selection import train_test_split
import numpy as np
from metrics import calculate_continuous_metrics, calculate_discrete_metrics


@log_time
def discover_case_attributes(e_log, attributes_to_discover, encoders, log_ids, confidence_threshold):
    case_attributes = []
    metrics = {}

    e_log_train, e_log_test = train_test_split(e_log, test_size=0.3, random_state=42)

    for attribute in attributes_to_discover:
        if not check_case_attribute_confidence(e_log, attribute, log_ids, confidence_threshold):
            continue

        is_discrete = attribute in encoders.keys()
        X_train, X_test = e_log_train[attribute], e_log_test[attribute]

        if is_discrete:
            attr, attr_metrics = discover_discrete_attribute(X_train, X_test, attribute, encoders)
        else:
            attr, attr_metrics = discover_continuous_case_attribute(X_train, X_test, attribute)

        case_attributes.append(attr)
        metrics[attribute] = attr_metrics

    return case_attributes, metrics


def check_case_attribute_confidence(e_log, attribute, log_ids, confidence_threshold):
    group_counts = e_log.groupby(log_ids.case)[attribute].apply(lambda x: (x == x.iloc[0]).sum())
    case_lengths = e_log.groupby(log_ids.case).size()
    confidences = group_counts / case_lengths
    return confidences.mean() >= confidence_threshold


def discover_discrete_attribute(X_train, X_test, attribute, encoders):
    if attribute in encoders:
        le = encoders[attribute]
        X_train_decoded = le.inverse_transform(X_train)
        X_test_decoded = le.inverse_transform(X_test)
        unique_values = le.classes_
    else:
        X_train_decoded = X_train
        X_test_decoded = X_test
        unique_values = np.unique(X_train)

    value_probs = [np.mean(X_train_decoded == val) for val in unique_values]
    y_pred = np.random.choice(unique_values, size=len(X_test_decoded), p=value_probs)

    metrics = calculate_discrete_metrics(X_test_decoded, y_pred, unique_values, encoders, attribute)

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


@log_time
def _handle_case_attributes(e_log, attributes_to_discover, log_ids, confidence_threshold, encoders):
    case_attributes = []

    for attribute in attributes_to_discover:
        is_discrete = attribute in encoders.keys()

        if is_discrete:
            e_log_decoded = e_log.copy()
            e_log_decoded[attribute] = encoders[attribute].inverse_transform(e_log[attribute])
        else:
            e_log_decoded = e_log

        group_counts = e_log_decoded.groupby(log_ids.case)[attribute].apply(lambda x: (x == x.iloc[0]).sum())
        case_lengths = e_log_decoded.groupby(log_ids.case).size()
        confidences = group_counts / case_lengths

        if confidences.mean() < confidence_threshold:
            continue

        if is_discrete:
            unique_values = e_log_decoded[attribute].unique()
            values = {value: 0 for value in unique_values}
            for case_id, case_data in e_log_decoded.groupby(log_ids.case):
                main_attribute = case_data[attribute].iloc[0]
                values[main_attribute] += 1

            num_cases = len(e_log_decoded[log_ids.case].unique())
            case_attributes.append({
                "name": attribute,
                "type": "discrete",
                "values": [{"key": value, "value": values[value] / num_cases} for value in values if values[value] > 0]
            })
        else:
            data = [case_data[attribute].iloc[0] for case_id, case_data in e_log_decoded.groupby(log_ids.case)]
            best_distribution = get_best_fitting_distribution(data)
            case_attributes.append({
                "name": attribute,
                "type": "continuous",
                "values": best_distribution.to_prosimos_distribution()
            })

    return case_attributes
