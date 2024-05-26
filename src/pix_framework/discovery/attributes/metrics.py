import numpy as np
from sklearn.metrics import mean_squared_error, median_absolute_error
from scipy.stats import wasserstein_distance, ks_2samp

discrete_metrics = [
    'KS',   # KS_Statistic
    'KSPV'  # KS_p_value
]

continuous_metrics = [
    'MSE',   # Mean Squared Error
    'MedAD', # Median Absolute Deviation
    'EMD'    # Earth Mover's Distance
]


def get_metrics_by_type(metric_type):
    if metric_type == 'continuous':
        return continuous_metrics
    elif metric_type == 'discrete':
        return discrete_metrics
    else:
        raise ValueError("Invalid metric type. Choose 'continuous' or 'discrete'.")


def calculate_discrete_metrics(actual, predicted, unique_values, encoder):
    if '' in encoder.classes_ and '' not in unique_values:
        unique_values = np.append(unique_values, '')

    actual_counts = np.array([np.sum(actual == value) for value in unique_values])
    predicted_counts = np.array([np.sum(predicted == value) for value in unique_values])

    ks_statistic, ks_pvalue = ks_2samp(actual_counts, predicted_counts)

    metrics = {
        'KS': ks_statistic,
        'KSPV': ks_pvalue
    }

    return metrics


def calculate_continuous_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    emd = wasserstein_distance(y_true, y_pred)

    metrics = {
        'MSE': mse,
        'MedAD': medae,
        'EMD': emd
    }
    return metrics


def update_model_results(attr_results, model_name, log_type, activity, metrics, formula, metrics_keys):
    if metrics is None:
        print(f"No metrics to update for model {model_name}, activity {activity}.")
        attr_results['models'][model_name]['activities'].setdefault(activity, {})[log_type] = {
            'metrics': 'No metrics due to model fitting failure',
            'formula': formula
        }
        return

    num_activities = len(attr_results['models'][model_name]['activities'])

    for key in metrics_keys:
        if key in metrics:
            if num_activities > 0:
                attr_results['models'][model_name]['total_scores'][log_type][key] += metrics[key]
            else:
                attr_results['models'][model_name]['total_scores'][log_type][key] = metrics[key]

            attr_results['models'][model_name]['total_scores'][log_type].setdefault(f'{key}_values', []).append(metrics[key])

    attr_results['models'][model_name]['activities'].setdefault(activity, {})[log_type] = {
        'metrics': metrics,
        'formula': formula
    }

    for key in metrics_keys:
        values = attr_results['models'][model_name]['total_scores'][log_type].get(f'{key}_values', [])
        if values:
            mean = np.mean(values)
            deviation = np.std(values)
            attr_results['models'][model_name]['total_scores'][log_type][f'{key}_avg'] = mean
            attr_results['models'][model_name]['total_scores'][log_type][f'{key}_dev'] = deviation
