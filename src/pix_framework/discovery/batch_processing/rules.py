import pandas as pd
import wittgenstein as lw


def _get_rules(
        data: pd.DataFrame,
        outcome: str,
        min_rule_support: float = 0.25,
        max_rules: int = 3
) -> dict:
    """
    Discover the rules that lead to the positive outcome in the observations passed as argument in [data].
    In this function, the support is measured only with positive outcome observations, instead of all
    observations. In this way, a rule with 50% support explains 50% of positive outcome observations,
    and a rule with 100% explains all of them.

    :param data:                pd.DataFrame with one observation per row.
    :param outcome              ID of the column with the variable to predict (1 positive, 0 negative).
    :param min_rule_support:    Minimum individual support for the discovered activation rules.
    :param max_rules:           Maximum number of activation rules to extract from a batch.
    :return: a dict with the RIPPER model, its confidence, and its support.
    """
    # Create empty model and data copy
    ripper_model = None
    filtered_data = data.copy()
    # Extract rules one by one
    continue_search = True
    while continue_search:
        # Train new model to extract 1 rule
        new_model = lw.RIPPER(max_rules=2)
        new_model.fit(filtered_data, class_feat=outcome)
        # If any rule has been discovered
        if len(new_model.ruleset_.rules) > 0:
            # Measure support
            predictions = new_model.predict(filtered_data.drop([outcome], axis=1))
            true_positives = [
                p and a
                for (p, a) in zip(predictions, filtered_data[outcome])
            ]
            support = sum(true_positives) / sum(data[outcome])  # hacked support to only consider positive outcomes
            if support >= min_rule_support:
                # If the support is enough, add it to the model and remove its positive cases
                if ripper_model:
                    ripper_model.add_rule(new_model.ruleset_.rules[0])
                else:
                    ripper_model = new_model
                # Retain only non
                filtered_data = filtered_data[[not prediction for prediction in predictions]]
            else:
                # If support is not enough, end search
                continue_search = False
        else:
            # If no rules have been discovered, end search
            continue_search = False
        if ripper_model and len(ripper_model.ruleset_.rules) >= max_rules:
            # If enough rules have been discovered, end search
            continue_search = False

    if ripper_model:
        predictions = ripper_model.predict(data.drop([outcome], axis=1))
        true_positives = [
            p and a
            for (p, a) in zip(predictions, data[outcome])
        ]
        return {
            'model': ripper_model,
            'confidence': sum(true_positives) / sum(predictions),
            'support': sum(true_positives) / sum(data[outcome])  # hacked support to only consider positive outcomes
        }
    else:
        return {}


def _parse_rules(model) -> list:
    """
    Transform the rules from a RIPPER model into a list of sublists (OR of ANDs), where the rule is fulfilled when one
    of the sublists (OR) have all its rules met (AND).

    :param model: RIPPER model to transform.
    :return: list of sublists with the rules.
    """
    rules = []
    # Go over the rules transforming them
    for ruleset in model.ruleset_.rules:
        # For each set of rules (sublist)
        sublist = []
        for condition in ruleset.conds:
            if type(condition.val) is not str:
                # Single number
                operator = "="
                value = str(condition.val)
            elif "<" in condition.val:
                # Lower than
                operator = "<="
                value = condition.val.replace("<", "")
            elif ">" in condition.val:
                # Greater than
                operator = ">="
                value = condition.val.replace(">", "")
            else:
                # Interval
                indexes = [i for i, char in enumerate(condition.val) if i > 0 and char == "-" and condition.val[i - 1] != 'e']
                if len(indexes) == 1:
                    index = indexes[0]
                    operator = "in"
                    value = [condition.val[:index], condition.val[index + 1:]]
                else:
                    print("Error parsing interval '{}', couldn't find the separating character '-'.".format(condition.val))
                    operator = "null"
                    value = "null"
            sublist += [{'attribute': condition.feature, 'comparison': operator, 'value': value}]
        # Add sublist of rules to complete
        rules += [sublist]
    # Return the rules
    return rules
