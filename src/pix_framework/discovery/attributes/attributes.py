import m5py.main
from pandas import DataFrame
from pix_framework.statistics.distribution import get_best_fitting_distribution

import pandas as pd
import numpy as np
import json
from pandas.api.types import is_numeric_dtype
from pix_framework.io.event_log import EventLogIDs
import time
import functools
import pprint
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_text
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.tree import BaseDecisionTree, _tree
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from niaarm.text import Corpus
from niaarm.mine import get_text_rules
from niaarm import Dataset, get_rules
from niapy.algorithms.basic import DifferentialEvolution, ParticleSwarmOptimization
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from scipy.stats import median_abs_deviation

from m5py import M5Prime, export_text_m5

import logging

import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DISCRETE_ERROR_RATIO = 0.95
DEFAULT_SAMPLING_SIZE = 2500

metrics_dict = {
    'MSE': None,  # Mean Squared Error
    'MAE': None,  # Mean Absolute Error
    'MedAD': None,  # Median Absolute Deviation
    'AIC': None,  # Akaike Information Criterion
}


def sample_event_log_by_case(event_log: pd.DataFrame,
                             log_ids: EventLogIDs,
                             sampling_size: int = DEFAULT_SAMPLING_SIZE) -> pd.DataFrame:
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


def sample_until_case_end(event_log, log_ids, sampling_size=DEFAULT_SAMPLING_SIZE):
    unique_cases = event_log[log_ids.case].unique()
    if len(unique_cases) <= sampling_size:
        return event_log

    nth_case_id = unique_cases[sampling_size - 1]
    last_index_of_nth_case = event_log[event_log[log_ids.case] == nth_case_id].index[-1]
    return event_log.loc[:last_index_of_nth_case]


def fill_nans(log, log_ids, is_event_log=False):
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

    if is_event_log:
        preprocessed_log = (log.groupby(log_ids.case).apply(preprocess_case)).reset_index(drop=True)
    else:
        preprocessed_log = log.apply(replace_initial_nans)

    preprocessed_log = preprocessed_log.ffill()
    return preprocessed_log


def extract_features(log, log_ids, is_event_log=False):
    features = pd.DataFrame()

    if is_event_log:
        for case_id, case_data in log.groupby(log_ids.case):
            initial_values = {col: 0 if is_numeric_dtype(case_data[col]) else '' for col in case_data.columns}
            case_features = case_data.shift(1).fillna(initial_values)

            features = pd.concat([features, case_features])

    else:
        initial_values = {col: 0 if is_numeric_dtype(log[col]) else '' for col in log.columns}
        features = log.shift(1).fillna(initial_values)

    features = features.drop(columns=[log_ids.case])
    features = features.rename(columns=lambda x: 'prev_' + x)

    return features


def scale_data(log, encoded_columns, threshold=1e+37):
    log_scaled = log.copy()
    max_float = np.finfo(np.float32).max
    min_float = np.finfo(np.float32).tiny

    for col in log.columns:
        if col not in encoded_columns and log[col].dtype in ['float64', 'int64']:
            log_scaled.loc[log_scaled[col] > threshold, col] = max_float
            log_scaled.loc[log_scaled[col] < -threshold, col] = min_float

            log_scaled[col] = log_scaled[col].fillna(method='ffill')
    return log_scaled


def preprocess_event_log(event_log, log_ids, sampling_size=DEFAULT_SAMPLING_SIZE):
    sorted_log = event_log.sort_values(by=log_ids.end_time)

    columns_to_drop = [getattr(log_ids, attr) for attr in vars(log_ids) if getattr(log_ids, attr) in sorted_log.columns]
    system_columns_to_keep = [log_ids.case, log_ids.activity]
    columns_to_drop = [x for x in columns_to_drop if x not in system_columns_to_keep]

    sorted_log = sorted_log.drop(columns=columns_to_drop)

    g_log = sample_until_case_end(sorted_log, log_ids, sampling_size)
    g_log = fill_nans(g_log, log_ids, False)

    e_log = sample_event_log_by_case(sorted_log, log_ids, sampling_size)
    e_log = fill_nans(e_log, log_ids, True)

    empty_series = pd.Series([''])
    encoded_columns = []
    encoder = LabelEncoder()
    for col in g_log.columns:

        if g_log[col].dtype == 'object':
            g_log[col] = g_log[col].astype(str)
            e_log[col] = e_log[col].astype(str)

            all_values = pd.concat([empty_series, g_log[col], e_log[col]]).unique()
            encoder.fit(all_values)

            g_log[col] = encoder.transform(g_log[col])
            e_log[col] = encoder.transform(e_log[col])
            encoded_columns.append(col)

    g_log = g_log.iloc[1:].reset_index(drop=True)
    e_log = e_log.iloc[1:].reset_index(drop=True)

    g_log = scale_data(g_log, encoded_columns)
    e_log = scale_data(e_log, encoded_columns)

    e_log_features = extract_features(e_log, log_ids, is_event_log=True)
    g_log_features = extract_features(g_log, log_ids, is_event_log=False)

    return e_log, e_log_features, g_log, g_log_features, encoder, encoded_columns


def discover_attributes(event_log: pd.DataFrame,
                        log_ids: EventLogIDs,
                        avoid_columns: list = None,
                        confidence_threshold: float = 1.0,
                        sampling_size: int = DEFAULT_SAMPLING_SIZE):
    if avoid_columns is None:
        avoid_columns = [
            log_ids.case, log_ids.activity, log_ids.start_time,
            log_ids.end_time, log_ids.resource, log_ids.enabled_time
        ]
    all_attribute_columns = subtract_lists(event_log.columns, avoid_columns)

    e_log, e_log_features, g_log, g_log_features, encoder, encoded_columns = preprocess_event_log(event_log, log_ids, sampling_size)

    # GLOBAL FIXED ATTRIBUTES
    global_attributes = _handle_fixed_global_attributes(e_log, all_attribute_columns, confidence_threshold, encoder, encoded_columns)
    global_fixed_attribute_names = [attribute['name'] for attribute in global_attributes]
    attributes_to_discover = subtract_lists(all_attribute_columns, global_fixed_attribute_names)
    print(f"Discovered global fixed attributes: {global_fixed_attribute_names}")

    # CASE ATTRIBUTES
    case_attributes = _handle_case_attributes(e_log, attributes_to_discover, log_ids, confidence_threshold, encoder, encoded_columns)
    case_attribute_names = [attribute['name'] for attribute in case_attributes]
    attributes_to_discover = subtract_lists(attributes_to_discover, case_attribute_names)
    print(f"Discovered case attributes: {case_attribute_names}")
    print(f"Attributes left to discover: {attributes_to_discover}")

    event_attributes = []
    model_results = {}

    print("LINEAR MODEL ANALYSIS")
    model_results['Linear'] = classify_and_generate_formula(
        e_log, g_log,
        e_log_features, g_log_features,
        attributes_to_discover, log_ids,
        linear_regression_analysis
    )

    print("CURVE MODEL ANALYSIS")
    model_results['Curve'] = classify_and_generate_formula(
        e_log, g_log,
        e_log_features, g_log_features,
        attributes_to_discover, log_ids,
        curve_fitting_analysis
    )

    print("M5PY MODEL ANALYSIS")
    model_results['M5PY'] = classify_and_generate_formula(
        e_log, g_log,
        e_log_features, g_log_features,
        attributes_to_discover, log_ids,
        m5prime_analysis
    )

    print_results_table(model_results)

    # print("------------------------------------------------")
    # print("------------------ GLOBAL ------------------")
    # pprint.pprint(global_attributes)
    # print("------------------ CASE ------------------")
    # pprint.pprint(case_attributes)
    # print("------------------ EVENT ------------------")
    # pprint.pprint(event_attributes)
    # print(f"\n\nAttribute columns left to discover: {attributes_to_discover}")
    #
    # return {
    #     "global_attributes": global_attributes,
    #     "case_attributes": case_attributes,
    #     "event_attributes": event_attributes
    # }


def print_results_table(model_results):
    metrics_keys = ['MSE', 'MAE', 'MedAD', 'AIC']
    header = f"{'Model':<15} {'Attribute':<30} " + \
             " ".join([f"{f'g_{metric}':<15}" for metric in metrics_keys]) + " " + \
             " ".join([f"{f'e_{metric}':<15}" for metric in metrics_keys])
    print(header)

    for model_name, attrs_results in model_results.items():
        for attr, attr_data in attrs_results.items():
            total_scores = attr_data['total_scores']
            global_values = " ".join([f"{total_scores['global'][metric]:<15.5}" for metric in metrics_keys])
            event_values = " ".join([f"{total_scores['event'][metric]:<15.5}" for metric in metrics_keys])
            row = f"{model_name:<15} {attr:<30} {global_values} {event_values}"
            print(row)


def classify_and_generate_formula(e_log, g_log, e_log_features, g_log_features, attributes_to_discover, log_ids, prediction_method):
    results = {}
    metrics_keys = ['MSE', 'MAE', 'MedAD', 'AIC']

    for attr in attributes_to_discover:
        print(f"Analyzing attribute: {attr}")

        attr_results = {
            'total_scores': {
                'event': {key: 0 for key in metrics_keys},
                'global': {key: 0 for key in metrics_keys}},
            'activities': {}
        }
        unique_activities = e_log[log_ids.activity].unique()

        for activity in unique_activities:
            # print(f"Analyzing activity: {activity}")

            # Filter data for current activity
            e_log_activity = e_log[e_log[log_ids.activity] == activity]
            g_log_activity = g_log[g_log[log_ids.activity] == activity]

            # Merge with features
            X_event = e_log_features.loc[e_log_activity.index][["prev_" + attr]]
            y_event = e_log_activity[attr]
            X_global = g_log_features.loc[g_log_activity.index][["prev_" + attr]]
            y_global = g_log_activity[attr]

            event_metrics, event_formula = prediction_method(X_event, y_event)
            global_metrics, global_formula = prediction_method(X_global, y_global)

            for key in metrics_keys:
                attr_results['total_scores']['event'][key] += event_metrics[key]
                attr_results['total_scores']['global'][key] += global_metrics[key]

            attr_results['activities'][activity] = {
                'event': {'metrics': event_metrics, 'formula': event_formula},
                'global': {'metrics': global_metrics, 'formula': global_formula}
            }

        results[attr] = attr_results

    return results



def calculate_aic(y_true, y_pred, num_params):
    n = len(y_true)
    RSS = np.sum((y_true - y_pred) ** 2)
    return n * np.log(RSS / n) + 2 * num_params


def model_function(X, *params):
    if X.ndim == 1:  # Reshape to handle 1 dim feature array
        X = X.reshape(-1, 1)

    y = np.zeros_like(X[:, 0])
    for i, param in enumerate(params):
        if i < X.shape[1]:  # Handle 1 dim feature array
            y += param * X[:, i] ** (len(params) - i - 1)
        else:
            y += param * (len(params) - i - 1)
    return y


def curve_fitting_analysis(X, y):
    X_2d = X.values if hasattr(X, 'values') else np.array(X)
    y_1d = y.values.ravel() if hasattr(y, 'values') else np.array(y)

    num_params = 3
    params, _ = curve_fit(model_function, X_2d, y_1d, p0=np.ones(num_params), maxfev=5000)

    y_pred = model_function(X_2d, *params)
    metrics = {
        'MSE': np.mean((y_1d - y_pred) ** 2),
        'MAE': mean_absolute_error(y_1d, y_pred),
        'MedAD': median_abs_deviation(y_1d - y_pred),
        'AIC': calculate_aic(y_1d, y_pred, num_params)
    }

    formula_dict = {f"x^{num_params-i}": param for i, param in enumerate(params, 1)}
    return metrics, formula_dict


def linear_regression_analysis(X, y):
    model = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        'MSE': mean_squared_error(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'MedAD': median_abs_deviation(y_test - y_pred),
        'AIC': calculate_aic(y_test, y_pred, len(model.coef_) + 1)
    }

    formula = ' + '.join([f'{coef:.3f}*{col}' for coef, col in zip(model.coef_, X.columns)]) + f' + {model.intercept_:.3f}'
    # print(formula)

    return metrics, formula


def m5prime_analysis(X, y):
    model = M5Prime(use_smoothing=True, use_pruning=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        'MSE': mean_squared_error(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'MedAD': median_abs_deviation(y_test - y_pred),
        'AIC': calculate_aic(y_test, y_pred, len(model.get_params()))
    }

    formula = model.as_pretty_text()
    # print(formula)
    return metrics, formula


def _handle_fixed_global_attributes(event_log_sorted, columns, confidence_threshold, encoder, encoded_columns):
    global_attributes = []
    for attribute in columns:
        max_frequency = event_log_sorted[attribute].value_counts(dropna=False).iloc[0] / len(event_log_sorted)

        if max_frequency >= confidence_threshold:
            attribute_value = event_log_sorted[attribute].iloc[0]

            if attribute in encoded_columns:
                decoded_attribute_value = encoder.inverse_transform([attribute_value])[0]
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


def _handle_case_attributes(e_log, attributes_to_discover, log_ids, confidence_threshold, encoder, encoded_columns):
    case_attributes = []

    for attribute in attributes_to_discover:
        is_discrete = attribute in encoded_columns

        if is_discrete:
            e_log_decoded = e_log.copy()
            e_log_decoded[attribute] = encoder.inverse_transform(e_log[attribute])
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


def subtract_lists(main_list, subtract_list):
    return [item for item in main_list if item not in subtract_list]
