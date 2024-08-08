def process_rules(rules, encoders):
    true_rules = filter_true_outcomes(rules)
    simplified_rules = simplify_rules(true_rules)
    expanded_rules = expand_and_decode_categorical_values(simplified_rules, encoders)
    return expanded_rules


def filter_true_outcomes(gateway_analysis_results):
    filtered_results = {}
    for gateway_id, flows in gateway_analysis_results.items():
        filtered_flows = {}
        for flow_id, rules in flows.items():
            true_rules = [rule for rule in rules if rule[1] == 1]
            if true_rules:
                filtered_flows[flow_id] = true_rules
            else:
                filtered_flows[flow_id] = []
        if filtered_flows:
            filtered_results[gateway_id] = filtered_flows
    return filtered_results


def simplify_rules(gateway_analysis_results):
    simplified_results = {}
    for gateway_id, flows in gateway_analysis_results.items():
        simplified_flows = {}
        for flow_id, conditions in flows.items():
            simplified_conditions = []
            for condition_set in conditions:
                condition_rules, outcome = condition_set
                if len(condition_rules) > 1 and outcome == 1:
                    simplified_condition = simplify_rule(condition_rules)
                    if simplified_condition:
                        simplified_conditions.append((simplified_condition, 1))
                else:
                    simplified_conditions.append(condition_set)
            simplified_flows[flow_id] = simplified_conditions if simplified_conditions else []
        simplified_results[gateway_id] = simplified_flows
    return simplified_results


def simplify_rule(rules):
    grouped_conditions = {}
    for attr, op, value in rules:
        if attr not in grouped_conditions:
            grouped_conditions[attr] = {'=': set(), '>': [], '<=': []}
        if op == '=':
            grouped_conditions[attr]['='].add(value)
        else:
            grouped_conditions[attr][op].append(value)

    simplified_conditions = []
    for attr, ops_values in grouped_conditions.items():
        for value in ops_values['=']:
            simplified_conditions.append((attr, '=', value))
        if ops_values['>']:
            max_greater_than = max(ops_values['>'])
            simplified_conditions.append((attr, '>', max_greater_than))
        if ops_values['<=']:
            min_less_than_or_equal = min(ops_values['<='])
            simplified_conditions.append((attr, '<=', min_less_than_or_equal))

    return simplified_conditions


def expand_and_decode_categorical_values(gateway_analysis_results, encoders):
    expanded_results = {}
    for gateway_id, flows in gateway_analysis_results.items():
        expanded_flows = {}
        for flow_id, conditions in flows.items():
            expanded_conditions = []
            for condition_set in conditions:
                condition_rules, outcome = condition_set
                expanded_rules = expand_and_decode_rule(condition_rules, encoders.get(gateway_id, {}))
                expanded_conditions.extend([(rule, outcome) for rule in expanded_rules])
            expanded_flows[flow_id] = expanded_conditions
        expanded_results[gateway_id] = expanded_flows
    return expanded_results


def expand_and_decode_rule(rules, encoders):
    expanded_rules = []
    base_rule = []
    discrete_conditions = {}

    for rule in rules:
        attr, op, value = rule
        if attr in encoders:
            if attr not in discrete_conditions:
                discrete_conditions[attr] = {'=': set(), '>': [], '<=': []}
            if op == '=':
                discrete_conditions[attr]['='].add(int(value))
            elif op == '>':
                discrete_conditions[attr]['>'].append(int(value))
            elif op == '<=':
                discrete_conditions[attr]['<='].append(int(value))
        else:
            base_rule.append(rule)

    for attr, ops_values in discrete_conditions.items():
        if ops_values['=']:
            ops_values['='] = [v for v in ops_values['='] if v != 0]
        if ops_values['>']:
            ops_values['>'] = [v for v in ops_values['>'] if v != 0]
        if ops_values['<=']:
            ops_values['<='] = [v for v in ops_values['<='] if v != 0]

        valid_values = set()
        if ops_values['>'] and ops_values['<=']:
            max_greater_than = max(ops_values['>'])
            min_less_than_or_equal = min(ops_values['<='])
            valid_values.update(range(max_greater_than + 1, min_less_than_or_equal + 1))
        else:
            if ops_values['>']:
                valid_values.update(range(max(ops_values['>']) + 1, len(encoders[attr].classes_)))
            if ops_values['<=']:
                valid_values.update(range(1, min(ops_values['<=']) + 1))
        if ops_values['=']:
            valid_values.update(ops_values['='])

        decoded_values = encoders[attr].inverse_transform(list(valid_values))
        for value in decoded_values:
            expanded_rules.append(base_rule + [(attr, '=', value)])

    if not expanded_rules:
        return [rules]

    return expanded_rules
