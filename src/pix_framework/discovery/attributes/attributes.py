from pix_framework.statistics.distribution import get_best_fitting_distribution

import pandas as pd
import numpy as np
import json
from pandas.api.types import is_numeric_dtype
from pix_framework.io.event_log import EventLogIDs
import time
import functools
import pprint
from m5py import M5Prime
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor, export_text

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DISCRETE_ERROR_RATIO = 0.95
DEFAULT_SAMPLING_SIZE = 2500


def time_it(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        timings[func.__name__] = elapsed_time
        return result

    return wrapper


@time_it
def sample_average_case_size(event_log: pd.DataFrame,
                             log_ids: any,
                             sampling_size: int) -> pd.DataFrame:
    avg_activities_per_case = event_log.groupby(log_ids.case).size().mean()
    rows_to_retrieve = int(avg_activities_per_case * sampling_size)
    return event_log.iloc[:rows_to_retrieve]


@time_it
def sample_event_log_by_case(event_log: pd.DataFrame,
                             log_ids: EventLogIDs,
                             sampling_size: int = DEFAULT_SAMPLING_SIZE) -> pd.DataFrame:
    unique_cases = event_log[log_ids.case].unique()

    if len(unique_cases) <= sampling_size:
        return event_log

    step_size = len(unique_cases) // sampling_size
    sampled_cases = unique_cases[::step_size]

    return event_log[event_log[log_ids.case].isin(sampled_cases)]


@time_it
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
    global timings
    timings = {}

    sampled_log_until_case_end = sample_until_case_end(event_log, log_ids, sampling_size)
    sampled_log_average_case_size = sample_average_case_size(event_log, log_ids, sampling_size)
    sampled_log_by_case = sample_event_log_by_case(event_log, log_ids, sampling_size)
    grouped_sample_log = sampled_log_by_case.groupby(log_ids.case)

    if avoid_columns is None:
        avoid_columns = [
            log_ids.case, log_ids.activity, log_ids.start_time,
            log_ids.end_time, log_ids.resource, log_ids.enabled_time
        ]
    all_attribute_columns = subtract_lists(sampled_log_by_case.columns, avoid_columns)

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
    featured_event_log = sampled_log_average_case_size[features_list]

    # _do_smart_things(featured_event_log, attributes_to_discover, log_ids)
    classify_attributes(featured_event_log, attributes_to_discover, log_ids)
    # DISCOVER EVENT AND GLOBAL ATTRIBUTES

    # print("------------------ TIMINGS ------------------")
    # for section, duration in timings.items():
    #     print(f"{section}: {duration:.4f} seconds")
    # print("------------------------------------------------")
    # print("------------------ GLOBAL ------------------")
    # pprint.pprint(global_attributes)
    # print("------------------ EVENT ------------------")
    # pprint.pprint(event_attributes)
    print(f"Attribute columns left to discover: {attributes_to_discover}")

    return {
        "global_attributes": global_attributes,
        "case_attributes": case_attributes,
        "event_attributes": event_attributes
    }


def classify_attributes(event_log, attributes_to_discover, log_ids):
    print("=============== CLASSIFY ===============")

    event_attributes = []
    global_attributes = []

    # Convert datetime columns
    for time_col in [log_ids.enabled_time]:
        event_log[time_col] = pd.to_datetime(event_log[time_col], utc=True)

    # Encode categorical columns if any
    label_encoders = {}
    for col in event_log.columns:
        if event_log[col].dtype == 'object':
            le = LabelEncoder()
            event_log[col] = le.fit_transform(event_log[col])
            label_encoders[col] = le

    for attr in attributes_to_discover:
        if attr in event_log.columns:
            X = event_log.drop(columns=attr)
            y = event_log[attr]

            # Train model and get scores (optional)
            model = M5Prime(use_smoothing=False, use_pruning=False)
            model.fit(X, y)
            model_scores = cross_val_score(model, X, y, cv=10)
            print(model_scores)

            # if model_score.mean() > threshold:
            #     event_attributes.append(attr)
            # else:
            #     global_attributes.append(attr)

    print(f"Event Attributes: {event_attributes}")
    print(f"Global Attributes: {global_attributes}")
    print("=============================================")

# Not able to FIT M5PY
# def classify_attributes(event_log, attributes_to_discover, log_ids):
#     print("=============== CLASSIFY ===============")
#     event_log_copy = event_log.copy()
#
#     # Convert datetime columns
#     event_log_copy[log_ids.enabled_time] = pd.to_datetime(event_log_copy[log_ids.enabled_time], utc=True)
#
#     # Encode string columns
#     le = LabelEncoder()
#     str_cols = event_log_copy.select_dtypes(include=['object']).columns.tolist()
#     for col in str_cols:
#         event_log_copy[col] = le.fit_transform(event_log_copy[col])
#
#     # Forward fill NaN values for each case
#     event_log_copy.sort_values(by=[log_ids.case, log_ids.enabled_time], inplace=True)
#     event_log_copy = event_log_copy.groupby(log_ids.case).apply(lambda group: group.ffill()).reset_index(drop=True)
#
#     feature_data = []
#
#     for attribute in attributes_to_discover:
#         attr_data = event_log_copy.groupby(log_ids.case)[attribute].agg(['mean', 'var', 'nunique', 'count'])
#         attr_data['consistency_ratio'] = attr_data['nunique'] / attr_data['count']
#         attr_data['percentage_of_NaNs'] = event_log_copy[attribute].isna().sum() / len(event_log_copy)
#         feature_data.append(attr_data.mean().values.tolist())
#
#     feature_df = pd.DataFrame(feature_data, columns=['mean', 'var', 'nunique', 'count', 'consistency_ratio', 'percentage_of_NaNs'])
#
#     # Use KMeans to get pseudo-labels
#     kmeans = KMeans(n_clusters=2, random_state=0).fit(feature_df)
#     pseudo_labels = kmeans.labels_
#
#     # Use M5Prime to generate rules based on features and pseudo-labels
#     model = M5Prime(use_smoothing=False)
#     # model.fit(feature_df)
#     model.fit(feature_df, y=pseudo_labels)
#     rules = model.get_rules()
#
#     # For the sake of this example, let's assume that rules generated help to classify into two groups.
#     # Depending on the nature and complexity of rules, further implementation might be needed.
#     global_attributes = [attr for idx, attr in enumerate(attributes_to_discover) if rules[idx] == 0]  # Or some condition based on rules
#     event_attributes = [attr for idx, attr in enumerate(attributes_to_discover) if rules[idx] == 1]  # Or some condition based on rules
#
#     print(f"Generated Rules: {rules}")
#     print(f"Event Attributes: {event_attributes}")
#     print(f"Global Attributes: {global_attributes}")
#     print("=============================================")


# def classify_attributes(event_log, attributes_to_discover, log_ids):
#     print("=============== CLASSIFY ===============")
#
#     # Convert datetime columns
#     event_log[log_ids.enabled_time] = pd.to_datetime(event_log[log_ids.enabled_time], utc=True)
#
#     # Encode string columns
#     le = LabelEncoder()
#     str_cols = event_log.select_dtypes(include=['object']).columns.tolist()
#     for col in str_cols:
#         event_log[col] = le.fit_transform(event_log[col])
#
#     # Forward fill NaN values for each case
#     event_log.sort_values(by=[log_ids.case, log_ids.enabled_time])
#     print(event_log.head())
#     event_log = event_log.groupby(log_ids.case).ffill().reset_index()
#     print(event_log.head())
#
#     feature_data = []
#
#     for attribute in attributes_to_discover:
#         # Extract data related to the current attribute
#         attr_data = event_log.groupby(log_ids.case)[attribute].agg(['mean', 'var', 'nunique', 'count'])
#         # attr_data.fillna(0, inplace=True)
#
#         # 'Consistency Ratio': Ratio of number of unique values to the count of events
#         attr_data['consistency_ratio'] = attr_data['nunique'] / attr_data['count']
#
#         # 'Percentage of NaNs'
#         attr_data['percentage_of_NaNs'] = event_log[attribute].isna().sum() / len(event_log)
#
#         feature_data.append(attr_data.mean().values.tolist())
#
#     # Scaling the features to make them suitable for clustering
#     scaler = MinMaxScaler()
#     scaled_features = scaler.fit_transform(feature_data)
#
#     # Using KMeans clustering
#     kmeans = KMeans(n_clusters=2, random_state=0).fit(scaled_features)
#     labels = kmeans.labels_
#
#     # Assuming cluster-0 is for global attributes and cluster-1 is for event attributes
#     global_attributes = [attr for idx, attr in enumerate(attributes_to_discover) if labels[idx] == 0]
#     event_attributes = [attr for idx, attr in enumerate(attributes_to_discover) if labels[idx] == 1]
#
#     print(f"Event Attributes: {event_attributes}")
#     print(f"Global Attributes: {global_attributes}")
#     print("=============================================")


# def classify_attributes(event_log, attributes_to_discover, log_ids):
#     print("=============== CLASSIFY ===============")
#
#     # Convert datetime columns
#     for time_col in [log_ids.enabled_time]:
#         event_log[time_col] = pd.to_datetime(event_log[time_col], utc=True)
#
#     # Encode string columns
#     le = LabelEncoder()
#     str_cols = event_log.select_dtypes(include=['object']).columns.tolist()
#     for col in str_cols:
#         event_log[col] = le.fit_transform(event_log[col])
#
#     # Impute NaN values for the entire dataframe
#     event_log = event_log.fillna(-999999)
#
#     event_attributes = []
#     global_attributes = []
#
#     feature_data = []
#     for attribute in attributes_to_discover:
#         std_within_case = event_log.groupby(log_ids.case)[attribute].std()
#         unique_ratio_within_case = event_log.groupby(log_ids.case)[attribute].nunique() / \
#                                    event_log.groupby(log_ids.case)[attribute].count()
#
#         feature_data.append([std_within_case.mean(), unique_ratio_within_case.mean()])
#
#     # For demonstration purposes, we're synthetically generating labels using previous thresholds
#     # 1 represents global attributes and 0 represents event attributes
#     labels = [1 if row[1] > 0.8 else 0 for row in feature_data]
#
#     model = DecisionTreeRegressor()
#     model.fit(feature_data, labels)
#
#     predictions = model.predict(feature_data)
#
#     for attribute, pred in zip(attributes_to_discover, predictions):
#         if pred > 0.5:
#             global_attributes.append(attribute)
#         else:
#             event_attributes.append(attribute)
#
#     print(f"Event Attributes: {event_attributes}")
#     print(f"Global Attributes: {global_attributes}")
#     print("=============================================")


# def classify_attributes(event_log, attributes_to_discover, log_ids):
#     print("=============== CLASSIFY ===============")
#
#     # Encode 'activity' and 'case_id' columns
#     le = LabelEncoder()
#     event_log[log_ids.activity] = le.fit_transform(event_log[log_ids.activity])
#     event_log[log_ids.case] = le.fit_transform(event_log[log_ids.case])
#     event_log[log_ids.enabled_time] = pd.to_datetime(event_log[log_ids.enabled_time], utc=True)
#
#     str_cols = event_log.select_dtypes(include=['object']).columns.tolist()
# 
#     for col in str_cols:
#         event_log[col] = le.fit_transform(event_log[col])
#
#     event_log = event_log.fillna(event_log.median(numeric_only=True))
#
#     event_attributes = []
#     global_attributes = []
#
#     for attribute in attributes_to_discover:
#         # Calculate within-case variability
#         within_case_std = event_log.groupby(log_ids.case)[attribute].std().mean()
#
#         # Calculate across-case variability
#         across_case_std = event_log[attribute].std()
#
#         # Time correlation (using Pearson correlation coefficient)
#         time_correlation = event_log[log_ids.enabled_time].corr(event_log[attribute], method='pearson')
#
#         # Decide if it's an event or global attribute based on calculated metrics
#         # The thresholds here are hypothetical and might need adjustments
#         if within_case_std > 0.5 or abs(time_correlation) > 0.5:
#             event_attributes.append(attribute)
#         elif across_case_std < 0.1:
#             global_attributes.append(attribute)
#
#     print(f"Event Attributes: {event_attributes}")
#     print(f"Global Attributes: {global_attributes}")
#     print("=============================================")

@time_it
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


@time_it
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


# @time_it
# def _handle_event_attribute(attribute, event_log, log_ids):
#     event_attribute_list = []
#
#     for activity_name, group in event_log.groupby(log_ids.activity):
#         filtered_group = group.dropna(subset=[attribute])
#
#         if len(filtered_group) == 0:
#             continue
#
#         attribute_data = {
#             "name": attribute,
#             "values": []
#         }
#
#         if _is_discrete_variable(filtered_group[attribute]):
#             attribute_data["type"] = "discrete"
#             total = len(filtered_group)
#             value_counts = filtered_group[attribute].value_counts()
#             aggregated_values = {key: value / total for key, value in value_counts.items()}
#
#             for key, value in aggregated_values.items():
#                 attribute_data["values"].append({
#                     "key": key,
#                     "value": value
#                 })
#
#         else:
#             attribute_data["type"] = "continuous"
#             # This is based on your specific requirements for continuous variables
#             data_distribution = get_best_fitting_distribution(filtered_group[attribute])
#             attribute_data["values"] = data_distribution.to_prosimos_distribution()
#
#         if attribute_data["values"]:  # Check if there are any qualifying values
#             # Check if the activity has been added already
#             existing_event_attr = next((x for x in event_attribute_list if x["event_id"] == activity_name), None)
#             if existing_event_attr:
#                 existing_event_attr["attributes"].append(attribute_data)
#             else:
#                 event_attribute_list.append({
#                     "event_id": activity_name,
#                     "attributes": [attribute_data]
#                 })
#
#     return event_attribute_list


def _is_discrete_variable(values: pd.Series) -> bool:
    return not is_numeric_dtype(values)


def _is_numeric(value):
    return isinstance(value, (int, float, complex))


def subtract_lists(main_list, subtract_list):
    return [item for item in main_list if item not in subtract_list]


def log_pretty_json(json_object):
    formatted_json_string = json.dumps(json_object, indent=4)
    print(formatted_json_string)
