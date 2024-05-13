import pprint

import numpy as np
from sklearn.model_selection import train_test_split
from case_attributes import discover_discrete_attribute
from metrics import get_metrics_by_type
from helpers import log_time
from metrics import calculate_discrete_metrics


@log_time
def discover_global_and_event_discrete_attributes(g_dfs, e_dfs, encoders):
    results = {}
    metrics_keys = get_metrics_by_type("discrete")
    model_functions = {
        'Frequency Analysis': frequency_analysis,
        'State Transition Matrix': state_transition_matrix_analysis
    }
    attributes_to_discover = encoders.keys()

    def process_attributes(dfs, attr_type, encoders):
        for activity, df in dfs.items():
            activity_difference = df['difference'].abs().sum()
            if activity_difference == 0:
                continue
            for model_name, model_function in model_functions.items():
                metrics, attribute_info = model_function(df, attr, encoders)
                update_model_results(attr_results, model_name, attr_type, activity, metrics, attribute_info, metrics_keys)

    for attr in attributes_to_discover:
        attr_results = {'models': {
            model_name: {
                'total_scores': {
                    'event': {key: 0 for key in metrics_keys},
                    'global': {key: 0 for key in metrics_keys}
                }, 'activities': {}} for model_name in model_functions.keys()}}

        process_attributes(g_dfs[attr], 'global', encoders)
        process_attributes(e_dfs[attr], 'event', encoders)
        results[attr] = attr_results
    return results


def frequency_analysis(df, attribute, encoders):
    train, test = train_test_split(df['current'], test_size=0.5, random_state=42)

    unique_values, counts = np.unique(train, return_counts=True)
    probabilities = counts / counts.sum()

    # Check if the probabilities sum up to 1 and adjust if necessary
    if not np.isclose(probabilities.sum(), 1.0):
        probabilities /= probabilities.sum()  # Scale probabilities to sum up to 1

    predictions = np.random.choice(unique_values, size=len(test), p=probabilities)
    decoded_predictions = encoders[attribute].inverse_transform(predictions)
    decoded_test_data = encoders[attribute].inverse_transform(test.values)

    metrics = calculate_discrete_metrics(decoded_test_data, decoded_predictions, unique_values, encoders, attribute)
    value_distribution = {str(encoders[attribute].inverse_transform([value])[0]): prob for value, prob in zip(unique_values, probabilities)}

    attribute_frequencies = [{"key": key, "value": value} for key, value in value_distribution.items()]

    return metrics, attribute_frequencies

def state_transition_matrix_analysis(df, attribute, encoders):
    encoder = encoders[attribute]

    # Split the data into training and testing sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Create a transition matrix
    num_states = len(encoder.classes_)
    transition_matrix = np.zeros((num_states, num_states))

    # Calculate the transition probabilities
    for previous, current in zip(train_df['previous'], train_df['current']):
        transition_matrix[previous, current] += 1

    # Normalize the transition probabilities within each row
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    transition_matrix = np.divide(transition_matrix, row_sums, out=np.zeros_like(transition_matrix), where=row_sums != 0)

    # Check for NaN values and replace them with 0 if they exist
    transition_matrix = np.nan_to_num(transition_matrix)

    # Format the attribute_info dictionary with the nested structure and decoded keys
    transition_matrix_out = {}
    for i in range(num_states):
        from_state = encoder.inverse_transform([i])[0]
        to_states = {
            encoder.inverse_transform([j])[0]: transition_matrix[i, j]
            for j in range(num_states) if transition_matrix[i, j] > 0
        }
        if to_states:  # Only include non-empty rows
            transition_matrix_out[from_state] = to_states

    # Use the transition matrix to predict the next state
    predictions = []
    for previous in test_df['previous']:
        probs = transition_matrix[previous]
        # Check if the sum of probabilities is 0
        if probs.sum() == 0:
            print(f"Warning: No transitions from state {previous}. Skipping prediction and metrics calculation.")
            continue
        # Check for NaN values in probs before making a prediction
        if np.isnan(probs).any():
            print(f"Warning: NaN values found in probabilities for state {previous}. Replacing with uniform probabilities.")
            probs = np.ones(num_states) / num_states  # Replace with uniform probabilities
        next_state = np.random.choice(range(num_states), p=probs)
        predictions.append(next_state)

    # Calculate metrics using the test data and predictions
    unique_values = encoder.classes_
    metrics = calculate_discrete_metrics(test_df['current'], predictions, unique_values, encoders, attribute)

    pprint.pprint(transition_matrix_out)

    return metrics, transition_matrix_out


# def state_transition_matrix_analysis(df, attribute, encoders):
#     encoder = encoders[attribute]
#
#     # Split the data into training and testing sets
#     train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
#
#     # Create a transition matrix
#     num_states = len(encoder.classes_)
#     transition_matrix = np.zeros((num_states, num_states))
#
#     # Calculate the transition probabilities
#     for previous, current in zip(train_df['previous'], train_df['current']):
#         transition_matrix[previous, current] += 1
#
#     # Normalize the transition probabilities within each row
#     row_sums = transition_matrix.sum(axis=1, keepdims=True)
#     transition_matrix = np.divide(transition_matrix, row_sums, out=np.zeros_like(transition_matrix), where=row_sums != 0)
#
#     # Check for NaN values and replace them with 0 if they exist
#     transition_matrix = np.nan_to_num(transition_matrix)
#
#     # Format the attribute_info dictionary with the nested structure and decoded keys
#     transition_matrix_out = {}
#     for i in range(num_states):
#         from_state = encoder.inverse_transform([i])[0]
#         to_states = {
#             encoder.inverse_transform([j])[0]: transition_matrix[i, j]
#             for j in range(num_states) if transition_matrix[i, j] > 0
#         }
#         if to_states:  # Only include non-empty rows
#             transition_matrix_out[from_state] = to_states
#
#     # Use the transition matrix to predict the next state
#     predictions = []
#     for previous in test_df['previous']:
#         probs = transition_matrix[previous]
#         # Check for NaN values in probs before making a prediction
#         if np.isnan(probs).any():
#             print(f"Warning: NaN values found in probabilities for state {previous}. Replacing with uniform probabilities.")
#             probs = np.ones(num_states) / num_states  # Replace with uniform probabilities
#         pprint.pprint(probs)
#         next_state = np.random.choice(range(num_states), p=probs)
#         predictions.append(next_state)
#
#     # Calculate metrics using the test data and predictions
#     unique_values = encoder.classes_
#     metrics = calculate_discrete_metrics(test_df['current'], predictions, unique_values, encoders, attribute)
#
#     pprint.pprint(transition_matrix_out)
#
#     return metrics, transition_matrix_out

# def discover_global_and_event_discrete_attributes(e_log, g_log, e_log_features, g_log_features, attributes_to_discover, encoders, log_ids):
#     results = {}
#     for attr in attributes_to_discover:
#         if attr not in encoders.keys():
#
#             continue
#
#         print(f"=========================== {attr} (Discrete) ===========================")
#         attr_results = {}
#
#         unique_activities = np.unique(np.concatenate((e_log[log_ids.activity].values, g_log[log_ids.activity].values)))
#
#         perform_model_discovery(unique_activities, e_log, 'event', attr, encoders, attr_results, log_ids)
#         perform_model_discovery(unique_activities, g_log, 'global', attr, encoders, attr_results, log_ids)
#
#         aggregate_metrics(attr_results)
#         results[attr] = attr_results
#
#     return results


def perform_model_discovery(unique_activities, attr_log, log_type, attr, encoders, attr_results, log_ids):
    models = {
        'Frequency Analysis': discrete_frequency_analysis,
    }
    metrics_keys = get_metrics_by_type("discrete")

    if 'models' not in attr_results:
        attr_results['models'] = {}
        for model_name in models.keys():
            attr_results['models'][model_name] = {
                'total_scores': {
                    'event': {k: 0 for k in metrics_keys},
                    'global': {k: 0 for k in metrics_keys}
                }, 'activities': {}}

    for activity in unique_activities:
        log_activity = attr_log[attr_log[log_ids.activity] == activity]
        diff_metric_count = log_activity[f'diff_{attr}'].abs().sum()
        if diff_metric_count > 0:
            for model_name, model_function in models.items():
                attr_info, metrics = model_function(log_activity, attr, encoders)
                update_model_results(attr_results, model_name, log_type, activity, metrics, attr_info, metrics_keys)


def discrete_frequency_analysis(log_activity, attr, encoders):
    if len(log_activity) <= 5:
        print("Too few samples for a meaningful train/test split. Using full dataset for both.")
        e_log_train = log_activity
        e_log_test = log_activity
    else:
        e_log_train, e_log_test = train_test_split(log_activity, test_size=0.2, random_state=42)

    X_train = e_log_train[attr]
    X_test = e_log_test[attr]
    return discover_discrete_attribute(X_train, X_test, attr, encoders)


def aggregate_metrics(attr_results):
    metrics_keys = get_metrics_by_type("discrete")

    for model_name in attr_results['models']:
        for log_type in ['event', 'global']:
            total_scores = {key: 0 for key in metrics_keys}
            activities = attr_results['models'][model_name]['activities']

            for activity, activity_data in activities.items():
                if log_type in activity_data:
                    for key in metrics_keys:
                        if key in activity_data[log_type]['metrics']:
                            total_scores[key] += activity_data[log_type]['metrics'][key]

            num_activities = len(activities)
            if num_activities > 0:
                for key in metrics_keys:
                    total_scores[key] /= num_activities

            attr_results['models'][model_name]['total_scores'][log_type] = total_scores


def update_model_results(attr_results, model_name, log_type, activity, metrics, formula, metrics_keys):
    num_activities = len(attr_results['models'][model_name]['activities'])

    for key in metrics_keys:
        if key in metrics:
            if num_activities > 0:
                attr_results['models'][model_name]['total_scores'][log_type][key] += \
                    metrics[key] / num_activities
            else:
                attr_results['models'][model_name]['total_scores'][log_type][key] = metrics[key]

    attr_results['models'][model_name]['activities'] \
        .setdefault(activity, {})[log_type] = {
        'metrics': metrics,
        'formula': formula
    }
