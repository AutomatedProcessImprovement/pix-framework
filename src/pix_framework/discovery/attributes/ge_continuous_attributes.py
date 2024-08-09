from m5py import M5Prime
from m5py.main import LinRegLeafModel
from sklearn.tree import _tree
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from pix_framework.statistics.distribution import get_best_fitting_distribution
from pix_framework.discovery.attributes.helpers import log_time
from pix_framework.discovery.attributes.metrics import calculate_continuous_metrics, get_metrics_by_type, update_model_results


@log_time
def discover_global_and_event_continuous_attributes(g_dfs, e_dfs, attributes_to_discover):
    results = {}
    metrics_keys = get_metrics_by_type("continuous")
    model_functions = {
        'Linear Regression': linear_regression_analysis,
        'Curve Fitting Generators': curve_fitting_generators_analysis,
        'M5Prime': m5prime_analysis,
        'Curve Fitting Update Rules': curve_fitting_update_rules_analysis
    }

    def process_attributes(dfs, attr_type):
        for activity, df in dfs.items():
            activity_difference = df['difference'].abs().sum()
            if activity_difference == 0:
                continue
            for model_name, model_function in model_functions.items():
                metrics, formula = model_function(df, attr)
                update_model_results(attr_results, model_name, attr_type, activity, metrics, formula, metrics_keys)

    for attr in attributes_to_discover:
        attr_results = {'models': {
            model_name: {
                'total_scores': {
                    'event': {key: 0 for key in metrics_keys},
                    'global': {key: 0 for key in metrics_keys}
                }, 'activities': {}} for model_name in model_functions.keys()}}

        process_attributes(g_dfs[attr], 'global')
        process_attributes(e_dfs[attr], 'event')
        results[attr] = attr_results

    return results


def linear_regression_analysis(df, attribute):
    try:
        X_train, X_test, y_train, y_test = train_test_split(df[['previous']], df['current'], test_size=0.5, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        metrics = calculate_continuous_metrics(y_test, y_pred)

        coef_str = " + ".join([f"{coef}*{attribute}" for i, coef in enumerate(model.coef_)])
        formula = f"{coef_str} + {model.intercept_}"

        return metrics, formula
    except Exception as e:
        error_metrics = {metric: float('inf') for metric in calculate_continuous_metrics([0], [0]).keys()}
        return error_metrics, None


def curve_fitting_generators_analysis(df, attribute):
    try:
        X_train, X_test, y_train, y_test = train_test_split(df['previous'], df['current'], test_size=0.5, random_state=42)

        y_train_flattened = y_train.values.flatten()
        y_test_flattened = y_test.values.flatten()

        best_distribution = get_best_fitting_distribution(y_train_flattened, filter_outliers=True)

        y_pred_flattened = best_distribution.generate_sample(len(y_test_flattened))

        metrics = calculate_continuous_metrics(y_test_flattened, y_pred_flattened)

        distribution_info = best_distribution.to_prosimos_distribution()

        return metrics, distribution_info
    except Exception as e:
        distribution_info = None
        error_metrics = {metric: float('inf') for metric in calculate_continuous_metrics([0], [0]).keys()}
        return error_metrics, distribution_info


def curve_fitting_update_rules_analysis(df, attribute):
    try:
        train, test = train_test_split(df['difference'], test_size=0.5, random_state=42)

        difference_distribution = get_best_fitting_distribution(train, filter_outliers=False)

        pred = difference_distribution.generate_sample(len(test))

        metrics = calculate_continuous_metrics(test, pred)

        formula = f"{attribute} + {difference_distribution.to_simple_function_call()}"

        return metrics, formula
    except Exception as e:
        distribution_info = None
        error_metrics = {metric: float('inf') for metric in calculate_continuous_metrics([0], [0]).keys()}
        return error_metrics, distribution_info


def m5prime_analysis(df, attribute):
    try:
        df.reset_index(drop=True, inplace=True)

        X_train, X_test, y_train, y_test = train_test_split(df['previous'], df['current'], test_size=0.5, random_state=42)

        X_train = X_train.values.reshape(-1, 1)
        X_test = X_test.values.reshape(-1, 1)

        y_train.reset_index(drop=True, inplace=True)
        y_test.reset_index(drop=True, inplace=True)

        model = M5Prime(use_smoothing=True, use_pruning=True)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        metrics = calculate_continuous_metrics(y_test, y_pred)

        formula = optimize_conditions(m5prime_to_json(model, [attribute]))

        return metrics, formula
    except Exception as e:
        formula = None
        error_metrics = {metric: float('inf') for metric in calculate_continuous_metrics([0], [0]).keys()}
        return error_metrics, formula


def optimize_conditions(rules):
    optimized_rules = []

    for conditions, formula in rules:
        if not conditions:
            optimized_rules.append([True, formula])
            continue

        min_condition = None
        max_condition = None

        for condition in conditions:
            feature, operator, value = condition.split()
            value = float(value)

            if operator == '<=':
                if not max_condition or value < max_condition[2]:
                    max_condition = (feature, operator, value)
            elif operator == '>':
                if not min_condition or value > min_condition[2]:
                    min_condition = (feature, operator, value)

        optimized_conditions = []
        if min_condition:
            optimized_conditions.append(f'{min_condition[0]} {min_condition[1]} {min_condition[2]}')
        if max_condition:
            optimized_conditions.append(f'{max_condition[0]} {max_condition[1]} {max_condition[2]}')

        optimized_rules.append([optimized_conditions, formula])

    return optimized_rules


def extract_rules(tree, node_models, feature_names, node_id=0, conditions=None):
    if conditions is None:
        conditions = []
    rules = []

    left_child = tree.children_left[node_id]
    right_child = tree.children_right[node_id]

    if left_child == _tree.TREE_LEAF:
        leaf_model = node_models[node_id]
        if isinstance(leaf_model, LinRegLeafModel):
            coef = leaf_model.model.coef_
            intercept = leaf_model.model.intercept_
            formula = ' + '.join([f'{feature_names[i]} * {coef[i]}' for i in range(len(coef))])
            if not formula:
                formula = f'{feature_names[0]} * 0.0'
            formula += f' + {intercept}'
            rules.append([conditions, formula])
        else:
            intercept = tree.value[node_id][0][0]
            formula = f'{feature_names[0]} * 0.0 + {intercept}'
            rules.append([conditions, formula])
    else:
        # Decision node
        feature = feature_names[tree.feature[node_id]]
        threshold = tree.threshold[node_id]

        left_conditions = conditions + [f'{feature} <= {threshold}']
        right_conditions = conditions + [f'{feature} > {threshold}']

        rules.extend(extract_rules(tree, node_models, feature_names, left_child, left_conditions))
        rules.extend(extract_rules(tree, node_models, feature_names, right_child, right_conditions))

    return rules


def m5prime_to_json(model, feature_names):
    tree = model.tree_
    return extract_rules(tree, model.node_models, feature_names)