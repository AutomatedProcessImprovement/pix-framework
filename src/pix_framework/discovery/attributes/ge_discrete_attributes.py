import numpy as np

from sklearn.model_selection import train_test_split

from pix_framework.discovery.attributes.helpers import log_time
from pix_framework.discovery.attributes.metrics import get_metrics_by_type, calculate_discrete_metrics, update_model_results


@log_time
def discover_global_and_event_discrete_attributes(g_dfs, e_dfs, attributes_to_discover, encoders):
    results = {}
    metrics_keys = get_metrics_by_type("discrete")
    model_functions = {
        'Frequency Analysis': frequency_analysis,
        'State Transition Matrix': state_transition_matrix_analysis
    }

    def process_attributes(dfs, attr_type, encoder):
        for activity, df in dfs.items():
            activity_difference = df['difference'].abs().sum()
            if activity_difference == 0:
                continue
            for model_name, model_function in model_functions.items():
                metrics, attribute_info = model_function(df, encoder)
                update_model_results(attr_results, model_name, attr_type, activity, metrics, attribute_info, metrics_keys)

    filtered_attributes = [attr for attr in attributes_to_discover if attr in encoders]

    for attr in filtered_attributes:
        attr_results = {'models': {
            model_name: {
                'total_scores': {
                    'event': {key: 0 for key in metrics_keys},
                    'global': {key: 0 for key in metrics_keys}
                }, 'activities': {}} for model_name in model_functions.keys()}}

        process_attributes(g_dfs[attr], 'global', encoders[attr])
        process_attributes(e_dfs[attr], 'event', encoders[attr])
        results[attr] = attr_results

    return results


def frequency_analysis(df, encoder):
    train, test = train_test_split(df['current'], test_size=0.5, random_state=42)

    unique_values, counts = np.unique(train, return_counts=True)
    decoded_unique_values = encoder.inverse_transform(unique_values)

    probabilities = counts / counts.sum()

    if not np.isclose(probabilities.sum(), 1.0):
        probabilities /= probabilities.sum()

    predictions = np.random.choice(unique_values, size=len(test), p=probabilities)
    decoded_predictions = encoder.inverse_transform(predictions)
    decoded_test = encoder.inverse_transform(test.values)

    metrics = calculate_discrete_metrics(decoded_test, decoded_predictions, decoded_unique_values, encoder)
    value_distribution = {str(encoder.inverse_transform([value])[0]): prob for value, prob in zip(unique_values, probabilities)}

    attribute_frequencies = [{"key": key, "value": value} for key, value in value_distribution.items()]

    return metrics, attribute_frequencies


def state_transition_matrix_analysis(df, encoder):
    train_df, test_df = train_test_split(df, test_size=0.5, random_state=42)

    num_states = len(encoder.classes_)
    transition_matrix = np.zeros((num_states, num_states))

    for previous, current in zip(train_df['previous'], train_df['current']):
        transition_matrix[previous, current] += 1

    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    transition_matrix = np.divide(transition_matrix, row_sums, out=np.zeros_like(transition_matrix), where=row_sums != 0)

    transition_matrix = np.nan_to_num(transition_matrix)

    transition_matrix_out = {}
    for i in range(num_states):
        from_state = encoder.inverse_transform([i])[0]
        to_states = {
            encoder.inverse_transform([j])[0]: transition_matrix[i, j]
            for j in range(num_states) if transition_matrix[i, j] > 0
        }
        if to_states:
            transition_matrix_out[from_state] = to_states

    predictions = []
    for previous in test_df['previous']:
        probs = transition_matrix[previous]
        if probs.sum() == 0:
            # print(f"Warning: No transitions from state {previous}. Skipping prediction and metrics calculation.")
            continue

        if np.isnan(probs).any():
            # print(f"Warning: NaN values found in probabilities for state {previous}. Replacing with uniform probabilities.")
            probs = np.ones(num_states) / num_states
        next_state = np.random.choice(range(num_states), p=probs)
        predictions.append(next_state)

    unique_values = np.unique(df['current'])
    decoded_unique_values = encoder.inverse_transform(unique_values)

    decoded_test_data = encoder.inverse_transform(test_df['current'].values)
    decoded_predictions = encoder.inverse_transform(predictions)

    metrics = calculate_discrete_metrics(decoded_test_data, decoded_predictions, decoded_unique_values, encoder)

    return metrics, transition_matrix_out


# def update_model_results(attr_results, model_name, log_type, activity, metrics, formula, metrics_keys):
#     num_activities = len(attr_results['models'][model_name]['activities'])
#
#     for key in metrics_keys:
#         if key in metrics:
#             if num_activities > 0:
#                 attr_results['models'][model_name]['total_scores'][log_type][key] += \
#                     metrics[key] / num_activities
#             else:
#                 attr_results['models'][model_name]['total_scores'][log_type][key] = metrics[key]
#
#     attr_results['models'][model_name]['activities'] \
#         .setdefault(activity, {})[log_type] = {
#         'metrics': metrics,
#         'formula': formula
#     }
