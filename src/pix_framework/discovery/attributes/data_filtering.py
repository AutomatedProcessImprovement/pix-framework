import numpy as np
import pandas as pd


def filter_consecutive_edge_values_event(X_train, y_train, case_ids):
    if isinstance(X_train, pd.DataFrame) or isinstance(X_train, pd.Series):
        X_train = X_train.to_numpy()
    if isinstance(y_train, pd.Series):
        y_train = y_train.to_numpy()
    if isinstance(case_ids, pd.Series):
        case_ids = case_ids.to_numpy()

    max_float32 = np.finfo(np.float32).max
    min_float32 = -max_float32
    tiny_float32 = np.finfo(np.float32).tiny
    edge_values = {max_float32, min_float32, tiny_float32, -tiny_float32}

    to_keep = np.ones(len(y_train), dtype=bool)
    last_seen_edge_value = {}

    for i in range(len(y_train)):
        current_value = y_train[i]
        current_case_id = case_ids[i]

        if i == 0 or current_case_id != case_ids[i - 1]:
            last_seen_edge_value.clear()

        if current_value in edge_values:
            if last_seen_edge_value.get(current_case_id) == current_value:
                to_keep[i] = False
            else:
                last_seen_edge_value[current_case_id] = current_value

    X_filtered = X_train[to_keep]
    y_filtered = y_train[to_keep]
    case_ids_filtered = case_ids[to_keep]

    return X_filtered, y_filtered, case_ids_filtered


def filter_consecutive_edge_values_global(X, y):
    # Ensure X and y are numpy arrays
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        X = X.to_numpy()
    if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
        y = y.to_numpy()

    # Ensure y is one-dimensional for comparison
    if y.ndim > 1 and y.shape[1] == 1:
        y = y.ravel()  # Convert y to a one-dimensional array if it's a single-column 2D array

    max_float32 = np.finfo(np.float32).max
    min_float32 = -max_float32
    tiny_float32 = np.finfo(np.float32).tiny
    edge_values = {max_float32, min_float32, tiny_float32, -tiny_float32}

    to_keep = np.ones(len(y), dtype=bool)

    for i in range(1, len(y)):
        is_edge_value = y[i] in edge_values
        is_consecutive_edge = is_edge_value and y[i] == y[i - 1]

        if is_consecutive_edge:
            to_keep[i] = False

    X_filtered = X[to_keep] if X.ndim > 1 else X[to_keep]
    y_filtered = y[to_keep]

    return X_filtered, y_filtered


def filter_attribute_columns(log, log_features, attr, log_ids):
    filtered_log = log[[log_ids.case, log_ids.activity, attr]]
    expected_feature_names = [f"prev_{attr}", f"diff_{attr}"]

    feature_cols = [col for col in log_features.columns if col in expected_feature_names]

    filtered_log_features = log_features[feature_cols]
    filtered_log = filtered_log.reset_index(drop=True)
    filtered_log_features = filtered_log_features.reset_index(drop=True)
    return pd.concat([filtered_log, filtered_log_features], axis=1)

