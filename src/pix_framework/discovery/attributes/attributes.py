from pix_framework.statistics.distribution import get_best_fitting_distribution

import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from pix_framework.io.event_log import EventLogIDs
import pprint
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

from m5py import M5Prime

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DISCRETE_ERROR_RATIO = 0.95
DEFAULT_SAMPLING_SIZE = 2500


def preprocess_event_log(event_log: pd.DataFrame, attributes_to_discover, log_ids: EventLogIDs):
    processed_log = event_log.copy()
    label_encoders = {}

    le_case = LabelEncoder()
    processed_log[log_ids.case] = le_case.fit_transform(processed_log[log_ids.case])
    label_encoders[log_ids.case] = le_case

    le_activity = LabelEncoder()
    processed_log[log_ids.activity] = le_activity.fit_transform(processed_log[log_ids.activity])
    label_encoders[log_ids.activity] = le_activity

    for attr in attributes_to_discover:
        first_valid_index = processed_log[attr].first_valid_index()
        if first_valid_index is not None:
            processed_log.loc[:first_valid_index, attr] = processed_log.loc[first_valid_index, attr]

        # Forward fill NaNs in the middle for numeric and categorical attributes
        processed_log[attr] = processed_log[attr].ffill()

        # Encoding string columns with LabelEncoder
        if processed_log[attr].dtype == 'O':
            le = LabelEncoder()
            processed_log[attr] = le.fit_transform(processed_log[attr])
            label_encoders[attr] = le  # Store the label encoder for this attribute

    return processed_log, label_encoders


def sample_average_case_size(event_log: pd.DataFrame,
                             log_ids: any,
                             sampling_size: int) -> pd.DataFrame:
    avg_activities_per_case = event_log.groupby(log_ids.case).size().mean()
    rows_to_retrieve = int(avg_activities_per_case * sampling_size)
    return event_log.iloc[:rows_to_retrieve]


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


def sample_until_case_end(event_log: pd.DataFrame,
                          log_ids: EventLogIDs,
                          sampling_size: int = DEFAULT_SAMPLING_SIZE) -> pd.DataFrame:
    unique_cases = event_log[log_ids.case].unique()
    if len(unique_cases) <= sampling_size:
        return event_log

    nth_case_id = unique_cases[sampling_size - 1]
    last_index_of_nth_case = event_log[event_log[log_ids.case] == nth_case_id].index[-1]
    return event_log.loc[:last_index_of_nth_case]


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

    event_log, labels = preprocess_event_log(event_log, all_attribute_columns, log_ids)

    sampled_log_until_case_end = sample_until_case_end(event_log, log_ids, sampling_size)
    # sampled_log_average_case_size = sample_average_case_size(event_log, log_ids, sampling_size)
    sampled_log_by_case = sample_event_log_by_case(event_log, log_ids, sampling_size)
    grouped_sample_log = sampled_log_by_case.groupby(log_ids.case)

    global_attributes = _handle_fixed_global_attributes(sampled_log_by_case, all_attribute_columns, confidence_threshold)
    global_fixed_attribute_names = [attribute['name'] for attribute in global_attributes]
    attributes_to_discover = subtract_lists(all_attribute_columns, global_fixed_attribute_names)

    case_attributes = _handle_case_attributes(sampled_log_by_case, grouped_sample_log, log_ids, attributes_to_discover,
                                              confidence_threshold)
    case_attribute_names = [attribute['name'] for attribute in case_attributes]
    attributes_to_discover = subtract_lists(attributes_to_discover, case_attribute_names)

    event_attributes = []

    # DISCOVER EVENT AND GLOBAL ATTRIBUTES

    features_list = [log_ids.case, log_ids.activity, log_ids.enabled_time] + all_attribute_columns
    # featured_event_log = sampled_log_average_case_size[features_list]
    featured_event_log = sampled_log_until_case_end[features_list]

    print("=== LINEAR === LINEAR === LINEAR === LINEAR === LINEAR === ")
    classification, formulas = classify_and_generate_formula(featured_event_log, sampled_log_by_case, attributes_to_discover, log_ids)
    print(formulas)
    print(classification)

    # decoded_formulas = decode_activity_formulas(formulas, labels[log_ids.activity])
    # pprint.pprint(decoded_formulas)

    print("\n\n=== CURVE === CURVE === CURVE === CURVE === CURVE === ")
    results, formulas = classify_and_generate_formula_curve_fit(featured_event_log, sampled_log_by_case, attributes_to_discover, log_ids)

    print(results)
    pprint.pprint(formulas)

    print("\n\n=== M5PY === M5PY === M5PY === M5PY === M5PY === ")
    classify_attributes(featured_event_log, sampled_log_by_case, attributes_to_discover, log_ids)

    # DISCOVER EVENT AND GLOBAL ATTRIBUTES

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


def polynomial_curve(x, *coefficients):
    x = np.array(x, dtype=float)

    polynomial_value = np.zeros_like(x, dtype=float)
    for i, coef in enumerate(coefficients):
        polynomial_value += coef * (x ** i)
    return polynomial_value


def fit_and_extract_formula(X, y, degree=2):
    X = np.array(X).flatten()
    initial_guess = [1] * (degree + 1)

    coef = curve_fit(polynomial_curve, X, y, p0=initial_guess)
    y_pred = polynomial_curve(X, *coef[0])

    r_squared = r2_score(y, y_pred)

    formula = " + ".join(f"{c:.3f}*x^{i}" for i, c in enumerate(coef[0]))
    return formula, r_squared


def classify_and_generate_formula_curve_fit(event_log, event_log_by_case, attributes_to_discover, log_ids, degree=2):
    results = {}
    formulas = {}

    for attr in attributes_to_discover:
        event_data = event_log_by_case[[log_ids.activity, attr]].dropna()
        X_event = event_data[log_ids.activity]
        y_event = event_data[attr]
        event_formula, event_r_squared = fit_and_extract_formula(X_event, y_event, degree)

        global_data = event_log[[log_ids.activity, attr]].dropna()
        X_global = global_data[log_ids.activity]
        y_global = global_data[attr]
        global_formula, global_r_squared = fit_and_extract_formula(X_global, y_global, degree)

        classification = "event" if event_r_squared > global_r_squared else "global"
        results[attr] = classification
        formulas[attr] = {"event": {"formula": event_formula, "r_squared": event_r_squared},
                          "global": {"formula": global_formula, "r_squared": global_r_squared}}
        print(f"{attr} classified as {classification}. Event R²: {event_r_squared}, Global R²: {global_r_squared}")

    return results, formulas


def decode_activity_formulas(formulas, activity_encoder):
    decoded_formulas = {}

    for attr, data in formulas.items():
        decoded_formulas[attr] = {}
        for model_type, formula_data in data.items():
            decoded_formula = {}
            for encoded_activity, coef in formula_data['formula'].items():
                activity_name = activity_encoder.inverse_transform([int(encoded_activity)])[0]
                decoded_formula[activity_name] = coef
            decoded_formulas[attr][model_type] = {
                "formula": decoded_formula,
                "intercept": formula_data["intercept"]
            }

    return decoded_formulas


def classify_and_generate_formula(event_log, event_log_by_case, attributes_to_discover, log_ids):
    results = {}
    formulas = {}

    for attr in attributes_to_discover:
        print(f"Analyzing {attr}")

        event_data = event_log_by_case[[log_ids.case, log_ids.activity, attr]].dropna()

        X_event = pd.get_dummies(event_data[log_ids.activity], drop_first=False)
        y_event = event_data[attr]

        global_data = event_log[[log_ids.activity, attr]].dropna()
        X_global = pd.get_dummies(global_data[log_ids.activity], drop_first=False)
        y_global = global_data[attr]

        # Classification based on model performance
        print("Event formula")
        event_score, event_formula_dict, event_intercept = linear_regression_analysis(X_event, y_event)

        print("Global formula")
        global_score, global_formula_dict, global_intercept = linear_regression_analysis(X_global, y_global)

        classification = "event" if event_score <= global_score else "global"
        print(f"{attr} \t\t EVENT SCORE: {event_score} \t\t GLOBAL SCORE: {global_score}")
        results[attr] = classification
        formulas[attr] = {
            "event": {"formula": event_formula_dict, "intercept": event_intercept},
            "global": {"formula": global_formula_dict, "intercept": global_intercept}
        }

    return results, formulas


def linear_regression_analysis(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = mean_squared_error(y_test, y_pred)

    formula = ' + '.join([f'{coef:.3f}*{col}' for coef, col in zip(model.coef_, X.columns)]) + f' + {model.intercept_:.3f}'
    print(f"---> {formula}")

    formula_dict = {col: coef for col, coef in zip(X.columns, model.coef_)}
    intercept = model.intercept_

    return score, formula_dict, intercept


def train_and_evaluate_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    model = M5Prime(use_smoothing=False, use_pruning=False)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    score = mean_squared_error(y_test, y_pred)

    return score, model


def prepare_global_data(event_log, attribute, log_ids, n_lags=3):
    for lag in range(1, n_lags + 1):
        event_log[f'{attribute}_lag_{lag}'] = event_log.groupby(log_ids.case)[attribute].shift(lag)

    for lag in range(1, n_lags + 1):
        event_log[f'{attribute}_diff_{lag}'] = event_log[attribute] - event_log[f'{attribute}_lag_{lag}']

    event_log[f'{attribute}_cumsum'] = event_log.groupby(log_ids.case)[attribute].cumsum()

    activity_dummies = pd.get_dummies(event_log[log_ids.activity], prefix='activity', drop_first=True)
    event_log = pd.concat([event_log, activity_dummies], axis=1)

    return event_log.dropna()


def classify_attributes(event_log, event_log_by_case, attributes_to_discover, log_ids):
    results = {}

    for attribute in attributes_to_discover:
        # print(f"Discovering {attribute}")
        # print(event_log)
        # event_log[log_ids.enabled_time] = pd.to_datetime(event_log[log_ids.enabled_time])
        # print(event_log)

        prepared_data = prepare_global_data(event_log, attribute, log_ids)
        event_data = event_log_by_case[[log_ids.case, log_ids.activity, attribute]].dropna()
        global_data = prepared_data[[log_ids.case, log_ids.activity] + [col for col in prepared_data.columns if attribute in col]].dropna()


        X_event = pd.get_dummies(event_data[log_ids.activity], drop_first=False)
        y_event = event_data[attribute]
        event_score, model_event = train_and_evaluate_model(X_event, y_event)

        X_global = global_data.drop(columns=[log_ids.case, log_ids.activity, attribute])
        y_global = global_data[attribute]
        global_score, model_global = train_and_evaluate_model(X_global, y_global)

        print(model_global.as_pretty_text())


        print(f"{attribute} EVENT: {event_score} or GLOBAL: {global_score}")
        # print("\n\n\n\n\n\n\n\n\n")


        if global_score < event_score:
            results[attribute] = "global"
        else:
            results[attribute] = "event"

    pprint.pprint(results)


def _handle_fixed_global_attributes(event_log_sorted, columns, confidence_threshold):
    global_attributes = []
    for attribute in columns:
        max_frequency = event_log_sorted[attribute].value_counts(dropna=False).iloc[0] / len(event_log_sorted)

        if max_frequency >= confidence_threshold:
            attribute_value = event_log_sorted[attribute].iloc[0]

            if is_numeric_dtype(attribute_value):
                global_attributes.append({
                    "name": attribute,
                    "type": "continuous",
                    "values": {
                        "distribution_name": "fix",
                        "distribution_params": [{"value": attribute_value}]
                    }
                })
            else:
                global_attributes.append({
                    "name": attribute,
                    "type": "discrete",
                    "values": [{"key": attribute_value, "value": 1.0}]
                })
    return global_attributes


def _handle_case_attributes(event_log, grouped_by_case, log_ids, columns, confidence_threshold):
    case_attributes = []

    for attribute in columns:
        attribute_distribution = event_log.groupby([log_ids.case, attribute]).size()
        is_discrete = _is_discrete_variable(event_log[attribute])

        group_counts = grouped_by_case[attribute].apply(lambda x: (x == x.iloc[0]).sum())
        case_lengths = grouped_by_case.size()
        confidences = group_counts / case_lengths

        if confidences.mean() < confidence_threshold:
            continue

        if is_discrete:
            unique_values = event_log[attribute].unique()
            values = {value: 0 for value in unique_values}
            for case_id, frequencies in attribute_distribution.groupby(level=0):
                main_attribute = frequencies.idxmax()[1]
                values[main_attribute] += 1

            num_cases = len(event_log[log_ids.case].unique())
            case_attributes.append({
                "name": attribute,
                "type": "discrete",
                "values": [{"key": value, "value": values[value] / num_cases} for value in values if values[value] > 0]
            })
        else:
            data = [frequencies.idxmax()[1] for case_id, frequencies in attribute_distribution.groupby(level=0)]
            best_distribution = get_best_fitting_distribution(data)
            case_attributes.append({
                "name": attribute,
                "type": "continuous",
                "values": best_distribution.to_prosimos_distribution()
            })

    return case_attributes


def _is_discrete_variable(values: pd.Series) -> bool:
    return not is_numeric_dtype(values)


def subtract_lists(main_list, subtract_list):
    return [item for item in main_list if item not in subtract_list]
