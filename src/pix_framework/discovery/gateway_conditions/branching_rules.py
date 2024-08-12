import optuna
import numpy as np

from pix_framework.discovery.gateway_conditions.helpers import log_time
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, _tree

optuna.logging.set_verbosity(optuna.logging.WARNING)


@log_time
def discover_xor_gateways(gateway_states, dataframes, prefixes, f_score_threshold):
    xor_rules = {}

    for gateway_id, gateway_info in gateway_states.items():
        if gateway_info['type'] == 'OR':
            continue

        print(f"Processing XOR Gateway: {gateway_id}")
        if gateway_id not in dataframes:
            continue

        df = dataframes[gateway_id]

        feature_columns = [col for col in df.columns if not any(col.startswith(prefix) for prefix in prefixes)]
        target_columns = [col for col in df.columns if any(col.startswith(prefix) for prefix in prefixes)]

        if not feature_columns or not target_columns:
            continue

        df['target'] = create_single_target(df[target_columns])

        features = df[feature_columns]
        target = df['target']

        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, features, target), n_trials=100)

        best_params = study.best_params
        best_tree = DecisionTreeClassifier(**best_params, random_state=42)
        best_tree.fit(features, target)

        preds = best_tree.predict(features)
        f_score = f1_score(target, preds, average='weighted')

        if f_score <= f_score_threshold:
            continue

        rules = extract_xor_rules(best_tree, features.columns)

        if gateway_id not in xor_rules:
            xor_rules[gateway_id] = {}

        for rule_conditions, outcome in rules:
            if outcome not in xor_rules[gateway_id]:
                xor_rules[gateway_id][outcome] = []
            xor_rules[gateway_id][outcome].append((rule_conditions, 1))

    return xor_rules


@log_time
def discover_or_gateways(gateway_states, dataframes, prefixes, f_score_threshold):
    or_rules = {}

    for gateway_id, gateway_info in gateway_states.items():
        if gateway_info['type'] == 'XOR':
            continue

        print(f"Processing OR Gateway: {gateway_id}")
        if gateway_id not in dataframes:
            continue

        df = dataframes[gateway_id]

        feature_columns = [col for col in df.columns if not any(col.startswith(prefix) for prefix in prefixes)]
        target_columns = [col for col in df.columns if any(col.startswith(prefix) for prefix in prefixes)]

        if not feature_columns or not target_columns:
            continue

        for target_col in target_columns:
            features = df[feature_columns]
            target = df[target_col]

            study = optuna.create_study(direction='maximize')
            study.optimize(lambda trial: objective(trial, features, target), n_trials=100)

            best_params = study.best_params
            best_tree = DecisionTreeClassifier(**best_params, random_state=42)
            best_tree.fit(features, target)

            preds = best_tree.predict(features)
            f_score = f1_score(target, preds, average='weighted')

            if f_score <= f_score_threshold:
                continue

            rules = extract_or_rules(best_tree, best_tree.feature_names_in_)

            if gateway_id not in or_rules:
                or_rules[gateway_id] = {}

            for rule_conditions, outcome in rules:
                if outcome:
                    if target_col not in or_rules[gateway_id]:
                        or_rules[gateway_id][target_col] = []
                    or_rules[gateway_id][target_col].append((rule_conditions, 1))

    return or_rules


def objective(trial, X, y):
    if X.empty or y.empty or X.shape[0] != y.shape[0]:
        return 0

    n_samples = X.shape[0]

    if n_samples < 10:
        return 0

    test_size = 0.5
    if n_samples * test_size < 1 or n_samples * (1 - test_size) < 1:
        return 0

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)

    param = {
        'max_depth': trial.suggest_int('max_depth', 1, 5),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 16),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 16),
        'ccp_alpha': trial.suggest_float('ccp_alpha', 1e-5, 1e-1, log=True),
    }

    model = DecisionTreeClassifier(**param, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)

    accuracy = f1_score(y_val, preds, average='weighted')
    return accuracy


def extract_xor_rules(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    rules = []

    def recurse(node, current_rule):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            left_rule = current_rule + [(name, "<=", threshold)]
            recurse(tree_.children_left[node], left_rule)

            right_rule = current_rule + [(name, ">", threshold)]
            recurse(tree_.children_right[node], right_rule)
        else:
            outcome_idx = np.argmax(tree_.value[node])
            outcome = tree.classes_[outcome_idx]
            rules.append((current_rule, outcome))

    recurse(0, [])
    return rules


def extract_or_rules(tree, feature_names):
    tree_ = tree.tree_

    def recurse(node, rules, current_rule):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_names[tree_.feature[node]]
            threshold = tree_.threshold[node]

            left_rule = current_rule + [(name, "<=", threshold)]
            recurse(tree_.children_left[node], rules, left_rule)

            right_rule = current_rule + [(name, ">", threshold)]
            recurse(tree_.children_right[node], rules, right_rule)
        else:
            class_probabilities = tree_.value[node]
            predicted_class_index = np.argmax(class_probabilities)
            predicted_class = tree.classes_[predicted_class_index]
            outcome = bool(predicted_class)

            if current_rule:
                rules.append((current_rule, outcome))
            else:
                rules.append(([], outcome))

    rules = []
    recurse(0, rules, [])
    return rules


def create_single_target(df):
    def get_first_true(row):
        for col in df.columns:
            if row[col]:
                return col
        return None
    return df.apply(get_first_true, axis=1)

