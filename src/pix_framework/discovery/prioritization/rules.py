import copy
import re

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, _tree


def discover_prioritization_rules(data: pd.DataFrame, outcome: str) -> list:
    """
    Discover, incrementally, rules to set the priority level of an activity instance in such a way that; when two activity instances are
    waiting to be executed (enabled), the one with the highest priority goes first. To do this, first discover the cases in the event log
    that have been prioritized (an activity enabled after another got executed first). Then, discover the rules (based on their attributes)
    that best describe the observed prioritizations.

    :param data:    pd.DataFrame with the observations of delayed and prioritized activity instances. The two activity instances
                    of the same prioritization (e.g. a specific instance of A prioritized over a specific instance of B) share
                    the same index in the DataFrame.
    :param outcome: ID of the column with the variable to predict (1 positive, 0 negative).

    :return: a list of dicts with the priority level and the corresponding rules.
    """
    # Create empty list for the incremental models
    models = []
    # Get the data we'll be using in each iteration
    filtered_data = pd.get_dummies(data)
    dummy_columns = {
        column: list(data[column].unique()) for column in data.columns if column not in filtered_data.columns
    }
    # Extract rules level by level
    continue_search = True
    while continue_search:
        # Discover a new model for the current observations
        model = _get_rules(filtered_data, outcome)
        # If any rule has been discovered
        if len(model) > 0:
            # Reverse the one hot encoding and save model for this priority level
            parsed_model = _reverse_one_hot_encoding(model, dummy_columns, filtered_data)
            models += [parsed_model]
            # Remove all observations covered by these rules (also negative ones)
            predictions = np.array(_predict(model, filtered_data.drop([outcome], axis=1)))
            true_positive_indexes = filtered_data[(filtered_data[outcome] == 1) & predictions].index
            filtered_data = filtered_data.loc[filtered_data.index.difference(true_positive_indexes)]
            # If no more prioritizations pending end search
            if len(filtered_data[filtered_data[outcome] == 1]) == 0:
                continue_search = False
        else:
            # If no rules have been discovered, end search
            continue_search = False
    # Create empty list for priority levels
    priority_levels = []
    current_lvl = 1
    for model in models:
        parsed_model = model
        priority_levels += [{"priority_level": current_lvl, "rules": parsed_model}]
        current_lvl += 1
    # Return list of level rules
    return priority_levels


def _get_rules(data: pd.DataFrame, outcome: str) -> list:
    """
    Discover one rule that lead to the positive outcome in the observations passed as argument in [data]. To do this, it uses a decision
    tree classifier to discover a rule 5 times, and gets the one with the highest confidence.

    :param data:    pd.DataFrame with one observation per row.
    :param outcome: ID of the column with the variable to predict (1 positive, 0 negative).
    :return: the discovered rules with the highest confidence.
    """
    # Discover 5 times and get the one with more confidence
    best_confidence = 0
    best_rules = []
    for i in range(5):
        # Train new model to extract 1 rule
        new_model = DecisionTreeClassifier()
        new_model.fit(data[[column for column in data.columns if column is not outcome]], data[outcome])
        best_rules = _tree_to_best_rules(new_model, [column for column in data.columns if column is not outcome])
        # If any rule has been discovered
        if len(best_rules) > 0:
            # Measure confidence
            predictions = _predict(best_rules, data.drop([outcome], axis=1))
            true_positives = [p and a for (p, a) in zip(predictions, data[outcome])]
            confidence = sum(true_positives) / sum(predictions)
            # Retain if it's better than the previous one
            if confidence > best_confidence:
                best_confidence = confidence
                best_rules = best_rules
    # Return the best one, or None if no rules found in any iteration
    return best_rules


def _tree_to_best_rules(tree, feature_names) -> list:
    # Extract tree structure
    tree_ = tree.tree_
    # Get the feature used in each non-leaf node
    feature_name = [feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!" for i in tree_.feature]
    # Go depth-first over the rules storing the best one
    best_rule = {"impurity": 1.0, "sample_size": 0, "rules": []}
    missing_nodes = [0]
    current_rules = [[]]
    while len(missing_nodes) > 0:
        current_node = missing_nodes.pop()
        current_rule = current_rules.pop()
        if tree_.feature[current_node] != _tree.TREE_UNDEFINED:
            # Decision node: add rule and keep search through each child
            name = feature_name[current_node]
            threshold = tree_.threshold[current_node]
            # Add ID and rule for left and right children
            missing_nodes += [tree_.children_left[current_node], tree_.children_right[current_node]]
            current_rules += [
                current_rule + [{'attribute': name, 'comparison': '<=', 'value': threshold}],
                current_rule + [{'attribute': name, 'comparison': '>', 'value': threshold}]
            ]
        else:
            # Leaf node
            current_impurity = tree_.impurity[current_node]
            current_sample_sizes = tree_.value[current_node][0]  # Number of positive samples
            # If it is the best leaf node, save it
            if current_sample_sizes[0] < current_sample_sizes[1] and (  # Less samples with negative outcome
                current_impurity < best_rule["impurity"]
                or (current_impurity == best_rule["impurity"] and current_sample_sizes[1] > best_rule["sample_size"])
            ):
                best_rule["impurity"] = current_impurity
                best_rule["sample_size"] = current_sample_sizes[1]
                best_rule["rules"] = _summarize_rules(current_rule)
    # Return best rules (wrapped in a list)
    return [best_rule["rules"]]


def _summarize_rules(rules: list) -> list:
    filtered_rules = []
    # Merge rules by same feature
    attributes = {rule["attribute"] for rule in rules}
    for attribute in attributes:
        operators = {rule['comparison'] for rule in rules if rule['attribute'] == attribute}
        if len(operators) > 1:
            # Add interval rule
            filtered_rules += [{
                'attribute': attribute,
                'comparison': 'in',
                'value': "({},{}]".format(
                    max([rule['value'] for rule in rules if rule['attribute'] == attribute and rule['comparison'] == ">"]),
                    min([rule['value'] for rule in rules if rule['attribute'] == attribute and rule['comparison'] == "<="])
                )
            }]
        else:
            # Add single rule
            operator = operators.pop()
            values = [rule['value'] for rule in rules if rule['attribute'] == attribute]
            filtered_rules += [{
                'attribute': attribute,
                'comparison': operator,
                'value': str(min(values)) if operator == "<=" else str(max(values))
            }]
    # Return rules
    return filtered_rules


def _predict(rules: list, data: pd.DataFrame) -> list:
    predictions = []
    # Predict each observation
    for index, observation in data.iterrows():
        prediction = False
        for ruleset in rules:
            if _fulfill_ruleset(ruleset, observation):
                prediction = True
        predictions += [prediction]
    # Return predictions
    return predictions


def _fulfill_ruleset(rules: list, observation: pd.Series):
    fulfills = True
    for rule in rules:
        values = [float(value) for value in re.findall(r"[\d.]+", rule["value"])]
        if (
                (rule['comparison'] == "<=" and observation[rule['attribute']] > values[0]) or
                (rule['comparison'] == ">" and observation[rule['attribute']] <= values[0]) or
                (rule['comparison'] == "in" and observation[rule['attribute']] <= values[0]) or
                (rule['comparison'] == "in" and observation[rule['attribute']] > values[1])
        ):
            fulfills = False
    return fulfills


def _reverse_one_hot_encoding(model: list, dummy_columns: dict, data: pd.DataFrame) -> list:
    # Deep copy of the list
    new_model = copy.deepcopy(model)
    # Correct the dummy columns removing the values that are no longer present in the dataset
    new_dummy_columns = {}
    for column in dummy_columns:
        new_values = [value for value in dummy_columns[column] if (data["{}_{}".format(column, value)] == 1).any()]
        new_dummy_columns[column] = new_values
    # For each ruleset (list of rules combined by ANDs such as all must be fulfilled)
    for ruleset in new_model:
        # Process each ruleset individually
        _reverse_one_hot_encoding_ruleset(ruleset, new_dummy_columns)
    # Return parsed rules
    return new_model


def _reverse_one_hot_encoding_ruleset(ruleset: list, dummy_columns: dict):
    # Initialize
    dummy_map = {  # Dict with dummified name as key, and pair with column+value as value
        "{}_{}".format(column, value): (column, value) for column in dummy_columns for value in dummy_columns[column]
    }
    diff_than_attributes = {  # Empty list for each attribute to store the assigned values
        attribute: [] for attribute in dummy_columns
    }
    equal_to_attributes = []  # Attributes with a rule '='
    rules_to_remove = []  # Indices of the rules to remove because of redundancy
    # Parse each rule in the ruleset
    for rule in ruleset:
        if rule['attribute'] in dummy_map:
            (orig_name, orig_value) = dummy_map[rule['attribute']]
            rule['attribute'] = orig_name
            rule['value'] = orig_value
            if rule['comparison'] == ">":
                rule['comparison'] = "="
                equal_to_attributes += [orig_name]
            else:
                rule['comparison'] = "!="
                diff_than_attributes[orig_name] += [orig_value]
    # Remove rules with '!=' if there's also a rule with '='
    for attribute in set(equal_to_attributes):
        rules_to_remove += [  # Get the index of the rules of this attribute with "!=" comparison
            index
            for index, rule in enumerate(ruleset)
            if rule['attribute'] == attribute and rule['comparison'] == "!="
        ]
    # Check if any categorical rule with N possible values got N-1 times '!=' (meaning it's '=' to the missing value).
    for attribute in diff_than_attributes:
        if (
            attribute not in equal_to_attributes
            and len(diff_than_attributes[attribute]) > 0
            and len(diff_than_attributes[attribute]) == len(dummy_columns[attribute]) - 1
        ):
            # Attribute has N possible values, and N-1 rules saying 'different from', simplify it
            new_rule = {
                'attribute': attribute,
                'comparison': "=",
                'value': [  # Get the missing value in all "!=" rules for this attribute
                    value
                    for value in dummy_columns[attribute]
                    if value not in diff_than_attributes[attribute]
                ][0]  # Get the first element (there should be only one)
            }
            # Get the index of the rules of this attribute with "!=" comparison
            rules_to_remove += [index for index, rule in enumerate(ruleset) if rule['attribute'] == attribute]
            # Add new rule to ruleset
            ruleset += [new_rule]
    # Remove redundant rules
    for i in sorted(rules_to_remove, reverse=True):
        del ruleset[i]
