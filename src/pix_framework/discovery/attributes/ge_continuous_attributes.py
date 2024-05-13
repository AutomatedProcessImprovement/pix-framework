import pprint

from helpers import log_time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from pix_framework.statistics.distribution import get_best_fitting_distribution
from m5py import M5Prime
from metrics import calculate_continuous_metrics, get_metrics_by_type


@log_time
def discover_global_and_event_continuous_attributes(g_dfs, e_dfs, attributes_to_discover):
    results = {}
    metrics_keys = get_metrics_by_type("continuous")
    model_functions = {
        'Linear Regression': linear_regression_analysis,
        'Curve Fitting Generators': curve_fitting_generators_analysis,
        'M5Prime': m5prime_analysis,
        'Curve Fitting Update Rules': curve_fitting_update_rules_analysis
    }

    def process_attributes(dfs, attr_type):
        for activity, df in dfs.items():
            activity_difference = df['difference'].abs().sum()
            if activity_difference == 0:
                continue
            for model_name, model_function in model_functions.items():
                metrics, formula = model_function(df, attr)
                update_model_results(attr_results, model_name, attr_type, activity, metrics, formula, metrics_keys)

    for attr in attributes_to_discover:
        print(f"=========================== {attr} (Continuous) ===========================")
        attr_results = {'models': {
            model_name: {
                'total_scores': {
                    'event': {key: 0 for key in metrics_keys},
                    'global': {key: 0 for key in metrics_keys}
                }, 'activities': {}} for model_name in model_functions.keys()}}

        process_attributes(g_dfs[attr], 'global')
        process_attributes(e_dfs[attr], 'event')
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


def linear_regression_analysis(df, attribute):
    try:
        X_train, X_test, y_train, y_test = train_test_split(df[['previous']], df['current'], test_size=0.5, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        metrics = calculate_continuous_metrics(y_test, y_pred)

        coef_str = " + ".join([f"{coef:.4f}*{attribute}" for i, coef in enumerate(model.coef_)])
        formula = f"{coef_str} + {model.intercept_:.4f}"

        return metrics, formula
    except Exception as e:
        error_metrics = {metric: float('inf') for metric in calculate_continuous_metrics([0], [0]).keys()}
        return error_metrics, None


def curve_fitting_generators_analysis(df, attribute):
    try:
        X_train, X_test, y_train, y_test = train_test_split(df['previous'], df['current'], test_size=0.5, random_state=42)

        y_train_flattened = y_train.values.flatten()
        y_test_flattened = y_test.values.flatten()

        best_distribution = get_best_fitting_distribution(y_train_flattened, filter_outliers=True)

        y_pred_flattened = best_distribution.generate_sample(len(y_test_flattened))

        metrics = calculate_continuous_metrics(y_test_flattened, y_pred_flattened)

        distribution_info = best_distribution.to_prosimos_distribution()

        return metrics, distribution_info
    except Exception as e:
        distribution_info = None
        error_metrics = {metric: float('inf') for metric in calculate_continuous_metrics([0], [0]).keys()}
        return error_metrics, distribution_info


def curve_fitting_update_rules_analysis(df, attribute):
    try:
        train, test = train_test_split(df['difference'], test_size=0.5, random_state=42)

        difference_distribution = get_best_fitting_distribution(train, filter_outliers=False)

        pred = difference_distribution.generate_sample(len(test))

        metrics = calculate_continuous_metrics(test, pred)

        formula = f"{attribute} + {difference_distribution}"

        return metrics, formula
    except Exception as e:
        print(e)
        distribution_info = None
        error_metrics = {metric: float('inf') for metric in calculate_continuous_metrics([0], [0]).keys()}
        return error_metrics, distribution_info


def m5prime_analysis(df, attribute):
    try:
        df.reset_index(drop=True, inplace=True)

        X_train, X_test, y_train, y_test = train_test_split(df['previous'], df['current'], test_size=0.5, random_state=42)

        X_train = X_train.values.reshape(-1, 1)
        X_test = X_test.values.reshape(-1, 1)

        y_train.reset_index(drop=True, inplace=True)
        y_test.reset_index(drop=True, inplace=True)

        model = M5Prime(use_smoothing=True, use_pruning=True)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        metrics = calculate_continuous_metrics(y_test, y_pred)

        formula = model.as_pretty_text()

        return metrics, formula
    except Exception as e:
        formula = None
        error_metrics = {metric: float('inf') for metric in calculate_continuous_metrics([0], [0]).keys()}
        return error_metrics, formula


@log_time
def discover_fixed_global_attributes(dfs, attributes_to_discover, confidence_threshold, encoders):
    global_attributes = []
    for attribute in attributes_to_discover:
        if attribute not in dfs:
            continue

        dfs_activities = dfs[attribute]

        value_counts = {}
        total_count = 0
        for activity, df in dfs_activities.items():
            for value, count in df['current'].value_counts(dropna=False).items():
                if value in value_counts:
                    value_counts[value] += count
                else:
                    value_counts[value] = count
                total_count += count

        max_frequency_value = max(value_counts, key=value_counts.get)
        max_frequency = value_counts[max_frequency_value] / total_count

        if max_frequency >= confidence_threshold:
            if attribute in encoders:
                decoded_attribute_value = encoders[attribute].inverse_transform([max_frequency_value])[0]
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
                        "distribution_params": [{"value": max_frequency_value}]
                    }
                })
    return global_attributes
