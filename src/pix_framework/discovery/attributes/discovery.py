import pandas as pd
import numpy as np
from helpers import log_time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from pix_framework.statistics.distribution import get_best_fitting_distribution
from m5py import M5Prime
from data_filtering import filter_consecutive_edge_values_global, filter_consecutive_edge_values_event
from metrics import calculate_continuous_metrics, get_metrics_by_type
from data_filtering import filter_attribute_columns


def discover_global_and_event_continuous_attributes(e_log, g_log, e_log_features, g_log_features, attributes_to_discover, log_ids):
    results = {}
    metrics_keys = get_metrics_by_type("continuous")
    model_functions = {
        'Linear Regression': linear_regression_analysis,
        'Curve Fitting': curve_fitting_analysis,
        'M5Prime': m5prime_analysis
    }

    for attr in attributes_to_discover:
        print(f"=========================== {attr} ===========================")
        attr_results = {'models': {
            model_name: {
                'total_scores': {
                    'event': {key: 0 for key in metrics_keys},
                    'global': {key: 0 for key in metrics_keys}
                }, 'activities': {}} for model_name in model_functions.keys()}}

        unique_activities = e_log[log_ids.activity].unique()

        e_attr_log = filter_attribute_columns(e_log, e_log_features, attr, log_ids)
        g_attr_log = filter_attribute_columns(g_log, g_log_features, attr, log_ids)

        for activity in unique_activities:
            e_log_activity = e_attr_log[e_attr_log[log_ids.activity] == activity]
            e_diff_metric_count = e_log_activity[f'diff_{attr}'].abs().sum()

            if e_diff_metric_count > 0:
                X = e_log_activity[[f'prev_{attr}']]
                X_reshaped = X.values.reshape(-1, 1) if isinstance(X, pd.Series) else X
                Y = e_log_activity[attr]
                case_ids = e_log_activity[log_ids.case].values

                if len(Y) <= 5:
                    print("Too few samples for a meaningful train/test split. Using full dataset for both.")
                    X_train = X_test = X_reshaped
                    Y_train = Y_test = Y
                    case_ids_train = case_ids_test = case_ids
                else:
                    X_train, X_test, Y_train, Y_test, case_ids_train, case_ids_test = \
                        train_test_split(X_reshaped, Y, case_ids, test_size=0.2, random_state=42)

                X_train_filtered, Y_train_filtered, case_ids_train_filtered = \
                    filter_consecutive_edge_values_event(X_train, Y_train, case_ids_train)

                for model_name, model_function in model_functions.items():
                    event_metrics, event_formula = model_function(X_train_filtered, X_test, Y_train_filtered, Y_test, attr)
                    update_model_results(attr_results, model_name, 'event', activity, event_metrics, event_formula, metrics_keys)

        for activity in unique_activities:
            g_log_activity = g_attr_log[g_attr_log[log_ids.activity] == activity]
            g_diff_metric_count = g_log_activity[f'diff_{attr}'].abs().sum()
            if g_diff_metric_count > 0:
                X = g_log_activity[[f'prev_{attr}']]
                X_reshaped = X.values.reshape(-1, 1) if isinstance(X, pd.Series) else X
                Y = g_log_activity[attr]

                if len(Y) <= 5:
                    X_train = X_test = X_reshaped
                    Y_train = Y_test = Y
                else:
                    X_train, X_test, Y_train, Y_test = train_test_split(X_reshaped, Y, test_size=0.2, random_state=42)
                X_train_filtered, Y_train_filtered = filter_consecutive_edge_values_global(X_train, Y_train)

                for model_name, model_function in model_functions.items():
                    global_metrics, global_formula = model_function(X_train_filtered, X_test, Y_train_filtered, Y_test, attr)
                    update_model_results(attr_results, model_name, 'global', activity, global_metrics, global_formula, metrics_keys)

        results[attr] = attr_results

    return results


def update_model_results(attr_results, model_name, log_type, activity, metrics, formula, metrics_keys):
    if metrics is None:
        print(f"No metrics to update for model {model_name}, activity {activity}.")
        attr_results['models'][model_name]['activities'].setdefault(activity, {})[log_type] = {
            'metrics': 'No metrics due to model fitting failure',
            'formula': formula
        }
        return

    for key in metrics_keys:
        attr_results['models'][model_name]['total_scores'][log_type][key] += metrics[key]
    attr_results['models'][model_name]['activities'].setdefault(activity, {})[log_type] = {
        'metrics': metrics,
        'formula': formula
    }


def linear_regression_analysis(X_train, X_test, y_train, y_test, attribute):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = calculate_continuous_metrics(y_test, y_pred)

    coef_str = " + ".join([f"{coef:.4f}*{attribute}" for i, coef in enumerate(model.coef_)])
    formula = f"{coef_str} + {model.intercept_:.4f}"

    return metrics, formula


def curve_fitting_analysis(X_train, X_test, y_train, y_test, attribute):
    try:
        combined_data = np.concatenate([y_train, y_test])

        combined_data_flattened = combined_data.flatten()

        best_distribution = get_best_fitting_distribution(combined_data_flattened, filter_outliers=True)
        y_pred = best_distribution.generate_sample(len(y_test))

        metrics = calculate_continuous_metrics(y_test, y_pred)

        distribution_info = best_distribution.to_prosimos_distribution()

        return metrics, distribution_info
    except Exception as e:
        distribution_info = None
        error_metrics = {
            'MSE': float('inf'),
            'MAE': float('inf'),
            'MedAD': float('inf'),
            'EMD': float('inf')
        }
        return error_metrics, distribution_info


def apply_threshold(X, y, threshold):
    """Applies a threshold to filter out values from X and y."""
    valid_indices = (y >= -threshold) & (y <= threshold)
    return X[valid_indices], y[valid_indices]


def m5prime_analysis(X_train, X_test, y_train, y_test, attribute):
    print("M5PRIME")
    model = M5Prime(use_smoothing=True, use_pruning=True)

    X_combined = np.vstack((X_train, X_test))
    y_combined = np.concatenate((y_train, y_test))

    # Initial threshold, M5Prime cannot handle values above it
    initial_threshold = 1.9770910581033629e+37
    threshold = initial_threshold

    success = False
    while not success:
        try:
            X_adjusted, y_adjusted = apply_threshold(X_combined, y_combined, threshold)

            if len(y_adjusted) < 5:
                print(f"Not enough samples to perform split after applying threshold {threshold}.")
                return None, None

            X_train_adjusted, X_test_adjusted, y_train_adjusted, y_test_adjusted = train_test_split(
                X_adjusted, y_adjusted, test_size=0.2, random_state=42)

            model.fit(X_train_adjusted, y_train_adjusted)
            success = True
        except ValueError as e:
            print(f"Adjusting threshold down from {threshold} due to error: {str(e)}")
            threshold *= 0.9

            if threshold < 1e+30:
                print("Threshold adjusted too low. Model fitting is not feasible with current data.")
                return None, None

    # After successful fitting, predict and calculate metrics
    y_pred = model.predict(X_test_adjusted)
    metrics = calculate_continuous_metrics(y_test_adjusted, y_pred)

    formula = model.as_pretty_text()

    return metrics, formula


@log_time
def discover_fixed_global_attributes(event_log, attributes_to_discover, confidence_threshold, encoders):
    global_attributes = []
    for attribute in attributes_to_discover:
        max_frequency = event_log[attribute].value_counts(dropna=False).iloc[0] / len(event_log)

        if max_frequency >= confidence_threshold:
            attribute_value = event_log[attribute].iloc[0]

            if attribute in encoders.keys():
                decoded_attribute_value = encoders[attribute].inverse_transform([attribute_value])[0]
                global_attributes.append({
                    "name": attribute,
                    "type": "discrete",
                    "values": [{"key": decoded_attribute_value, "value": 1.0}]
                })
            else:
                global_attributes.append({
                    "name": attribute,
                    "type": "continuous",
                    "values": {
                        "distribution_name": "fix",
                        "distribution_params": [{"value": attribute_value}]
                    }
                })
    return global_attributes
