import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, mutual_info_score
from scipy.stats import entropy, chi2_contingency, wasserstein_distance, ks_2samp
from scipy.spatial.distance import jensenshannon

discrete_metrics = [
    'CSS',  # Chi_Square_Statistic
    'PV',   # p_value
    'ENT',  # Entropy
    'MI',   # Mutual_Information
    'JSD',  # Jensen_Shannon_Divergence
    'KLD',  # KL_Divergence
    'KS',   # KS_Statistic
    'KSPV'  # KS_p_value
]

continuous_metrics = [
    'MSE',   # Mean Squared Error
    'MAE',   # Mean Absolute Error
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


def calculate_discrete_metrics(actual, predicted, unique_values, encoders, attribute):
    if '' in encoders[attribute].classes_ and '' not in unique_values:
        unique_values = np.append(unique_values, '')

    contingency_table = np.zeros((len(unique_values), 2))
    for i, value in enumerate(unique_values):
        contingency_table[i, 0] = np.sum(actual == value)
        contingency_table[i, 1] = np.sum(predicted == value)

    if np.any(contingency_table == 0):
        contingency_table += 1

    chi2, p, dof, _ = chi2_contingency(contingency_table, correction=False)

    actual_counts = np.array([np.sum(actual == value) for value in unique_values])
    predicted_counts = np.array([np.sum(predicted == value) for value in unique_values])
    actual_prob = actual_counts / np.sum(actual_counts)
    predicted_prob = predicted_counts / np.sum(predicted_counts)
    actual_entropy = entropy(actual_prob)
    mutual_info = mutual_info_score(None, None, contingency=contingency_table)
    js_divergence = jensenshannon(actual_prob, predicted_prob, base=2) ** 2

    kl_divergence = calculate_kl_divergence(actual_prob, predicted_prob)
    ks_statistic, ks_pvalue = ks_2samp(actual_counts, predicted_counts)

    metrics = {
        'CSS': chi2,
        'PV': p,
        'ENT': actual_entropy,
        'MI': mutual_info,
        'JSD': js_divergence,
        'KLD': kl_divergence,
        'KS': ks_statistic,
        'KSPV': ks_pvalue
    }

    return metrics


def calculate_kl_divergence(p, q):
    """
    Calculate Kullback-Leibler Divergence. Ensure no zero probabilities.
    """
    p = np.where(p == 0, 1e-9, p)
    q = np.where(q == 0, 1e-9, q)
    return np.sum(p * np.log(p / q))


def calculate_continuous_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    emd = wasserstein_distance(y_true, y_pred)

    metrics = {
        'MSE': mse,
        'MAE': mae,
        'MedAD': medae,
        'EMD': emd
    }
    return metrics
