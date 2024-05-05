import numpy as np
from data_filtering import filter_attribute_columns
from sklearn.model_selection import train_test_split
from case_attributes import discover_discrete_attribute
from metrics import get_metrics_by_type


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

def discover_global_and_event_discrete_attributes(e_log, g_log, e_log_features, g_log_features, attributes_to_discover, encoders, log_ids):
    results = {}
    for attr in attributes_to_discover:
        if attr not in encoders.keys():

            continue

        print(f"=========================== {attr} (Discrete) ===========================")
        attr_results = {}

        unique_activities = np.unique(np.concatenate((e_log[log_ids.activity].values, g_log[log_ids.activity].values)))

        e_attr_log = filter_attribute_columns(e_log, e_log_features, attr, log_ids)
        g_attr_log = filter_attribute_columns(g_log, g_log_features, attr, log_ids)

        perform_model_discovery(unique_activities, e_attr_log, 'event', attr, encoders, attr_results, log_ids)
        perform_model_discovery(unique_activities, g_attr_log, 'global', attr, encoders, attr_results, log_ids)

        aggregate_metrics(attr_results)
        results[attr] = attr_results

    return results


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
