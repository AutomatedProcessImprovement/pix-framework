import pandas as pd

from prioritization_discovery.rules import discover_prioritization_rules, _reverse_one_hot_encoding


def test_discover_prioritization_rules():
    # Given a set of prioritizations
    prioritizations = pd.DataFrame(
        data=[
            ["B", 0],
            ["B", 0],
            ["B", 0],
            ["B", 0],
            ["B", 0],
            ["B", 0],
            ["C", 1],
            ["C", 1],
            ["C", 1],
            ["C", 1],
            ["C", 1],
            ["C", 1],
        ],
        index=[0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5],
        columns=["Activity", "outcome"],
    )
    # Discover their rules
    prioritization_rules = discover_prioritization_rules(prioritizations, "outcome")
    # Assert the rules
    assert prioritization_rules == [
        {"priority_level": 1, "rules": [[{"attribute": "Activity", "condition": "=", "value": "C"}]]}
    ]


def test_discover_prioritization_rules_with_extra_attribute():
    # Given a set of prioritizations
    prioritizations = pd.DataFrame(
        data=[
            ["A", 500, 0],
            ["B", 500, 0],
            ["A", 500, 1],
            ["B", 100, 0],
            ["B", 100, 0],
            ["B", 100, 0],
            ["B", 100, 0],
            ["B", 100, 0],
            ["B", 500, 1],
            ["B", 1000, 1],
            ["B", 1000, 1],
            ["C", 100, 0],
            ["C", 500, 1],
            ["C", 500, 1],
            ["A", 1000, 1],
            ["C", 1000, 1],
        ],
        index=[0, 1, 2, 2, 3, 4, 5, 6, 4, 0, 3, 7, 6, 7, 1, 5],
        columns=["Activity", "loan_amount", "outcome"],
    )
    # Discover their rules
    prioritization_rules = discover_prioritization_rules(prioritizations, "outcome")
    # Assert the rules
    assert (
            prioritization_rules
            == prioritization_rules
            == [
                {"priority_level": 1, "rules": [[{"attribute": "loan_amount", "condition": ">", "value": "750.0"}]]},
                {"priority_level": 2, "rules": [[{"attribute": "loan_amount", "condition": ">", "value": "300.0"}]]},
            ]
    )


def test_discover_prioritization_rules_with_double_and_condition():
    # Given a set of prioritizations
    data = [
        [400, "high", 0],
        [1100, "high", 1],
        [1000, "low", 0],
        [1000, "high", 1],
        [1100, "low", 0],
        [1010, "high", 1],
        [500, "high", 0],
        [1300, "high", 1],
        [1300, "low", 0],
        [1100, "high", 1],
        [800, "high", 0],
        [1800, "high", 1],
        [510, "high", 0],
        [2000, "high", 1],
        [520, "low", 0],
        [900, "low", 1],
        [400, "high", 0],
        [700, "low", 1],
        [600, "low", 0],
        [800, "low", 1],
    ]
    indices = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9]
    # Multiply by 100 the observations to have enough population
    resize = 100
    data = data * resize
    num_indices = len(set(indices))
    indices = [index + (num_indices * i) for i in range(resize) for index in indices]
    # Create dataframe with observations
    prioritizations = pd.DataFrame(data=data, index=indices, columns=["loan_amount", "importance", "outcome"])
    # Discover their rules
    prioritization_rules = discover_prioritization_rules(prioritizations, "outcome")
    # Assert the rules
    assert sort_rules(prioritization_rules) == sort_rules([
        {
            "priority_level": 1,
            "rules": [
                [
                    {"attribute": "loan_amount", "condition": ">", "value": "900.0"},
                    {"attribute": "importance", "condition": "=", "value": "high"},
                ]
            ],
        },
        {"priority_level": 2, "rules": [[{"attribute": "loan_amount", "condition": ">", "value": "650.0"}]]},
    ])


def test_discover_prioritization_rules_inverted():
    # Given a set of prioritizations
    data = [
        [1300, "low", 0],
        [500, "low", 1],
        [1100, "low", 0],
        [400, "low", 1],
        [700, "high", 0],
        [400, "low", 1],
        [2000, "low", 0],
        [510, "low", 1],
        [900, "high", 0],
        [520, "high", 1],
        [800, "high", 0],
        [600, "high", 1],
        [1800, "low", 0],
        [800, "low", 1],
        [1000, "low", 0],
        [1000, "high", 1],
        [1010, "low", 0],
        [1100, "high", 1],
        [1100, "low", 0],
        [1300, "high", 1],
    ]
    indices = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9]
    # Multiply by 10 the observations to have enough population
    resize = 100
    data = data * resize
    num_indices = len(set(indices))
    indices = [index + (num_indices * i) for i in range(resize) for index in indices]
    # Create dataframe with observations
    prioritizations = pd.DataFrame(data=data, index=indices, columns=["loan_amount", "importance", "outcome"])
    # Discover their rules
    prioritization_rules = discover_prioritization_rules(prioritizations, "outcome")
    # Assert the rules
    assert prioritization_rules == [
        {"priority_level": 1, "rules": [[{"attribute": "loan_amount", "condition": "<=", "value": "650.0"}]]},
        {"priority_level": 2, "rules": [[{"attribute": "importance", "condition": "=", "value": "high"}]]},
        {"priority_level": 3, "rules": [[{"attribute": "loan_amount", "condition": "<=", "value": "1300.0"}]]},
    ]


def test__reverse_one_hot_encoding():
    # Check redundancy removal when there is one rule with '=' and others with '!=' for the same attribute
    assert _reverse_one_hot_encoding(
        model=[
            [
                {"attribute": "urgency_high", "condition": ">", "value": "0.5"},
                {"attribute": "urgency_low", "condition": "<=", "value": "0.5"},
            ]
        ],
        dummy_columns={"urgency": ["low", "medium", "high"]},
        data=pd.DataFrame(
            {
                "urgency_low": [1, 0, 0, 0, 0, 0, 1, 1, 0],
                "urgency_medium": [0, 0, 0, 0, 0, 0, 0, 0, 0],
                "urgency_high": [0, 1, 1, 1, 1, 1, 0, 0, 1],
                "amount": [100, 50, 10, 2, 40, 54, 23, 28, 54],
            }
        ),
    ) == [[{"attribute": "urgency", "condition": "=", "value": "high"}]]
    # Check redundancy removal when there is 5 possible values for an attribute and 4 rules with '!='
    assert _reverse_one_hot_encoding(
        model=[
            [
                {"attribute": "urgency_low_medium", "condition": "<=", "value": "0.5"},
                {"attribute": "urgency_medium", "condition": "<=", "value": "0.5"},
                {"attribute": "urgency_medium_high", "condition": "<=", "value": "0.5"},
                {"attribute": "urgency_high", "condition": "<=", "value": "0.5"},
            ]
        ],
        dummy_columns={"urgency": ["low", "low_medium", "medium", "medium_high", "high"]},
        data=pd.DataFrame(
            {
                "urgency_low": [0, 0, 0, 0, 1, 1, 1, 1, 1],
                "urgency_low_medium": [0, 1, 0, 0, 0, 0, 0, 0, 0],
                "urgency_medium": [0, 0, 1, 0, 0, 0, 0, 0, 0],
                "urgency_medium_high": [0, 0, 0, 1, 0, 0, 0, 0, 0],
                "urgency_high": [1, 0, 0, 0, 0, 0, 0, 0, 0],
                "amount": [100, 50, 10, 2, 40, 54, 23, 28, 54],
            }
        ),
    ) == [[{"attribute": "urgency", "condition": "=", "value": "low"}]]
    # Check redundancy removal when there is 4 possible values (in the filtered data) for an attribute and 3 rules with '!='
    assert _reverse_one_hot_encoding(
        model=[
            [
                {"attribute": "urgency_medium", "condition": "<=", "value": "0.5"},
                {"attribute": "urgency_medium_high", "condition": "<=", "value": "0.5"},
                {"attribute": "urgency_high", "condition": "<=", "value": "0.5"},
            ]
        ],
        dummy_columns={"urgency": ["low", "low_medium", "medium", "medium_high", "high"]},
        data=pd.DataFrame(
            {
                "urgency_low": [0, 0, 0, 0, 1, 1, 1, 1, 1],
                "urgency_low_medium": [0, 0, 0, 0, 0, 0, 0, 0, 0],
                "urgency_medium": [0, 0, 1, 0, 0, 0, 0, 0, 0],
                "urgency_medium_high": [0, 0, 0, 1, 0, 0, 0, 0, 0],
                "urgency_high": [1, 1, 0, 0, 0, 0, 0, 0, 0],
                "amount": [100, 50, 10, 2, 40, 54, 23, 28, 54],
            }
        ),
    ) == [[{"attribute": "urgency", "condition": "=", "value": "low"}]]


def sort_rules(rules):
    sorted_by_level = sorted(rules, key=lambda x: x["priority_level"])
    sorted_by_rules_attribute = [
        {"priority_level": rule["priority_level"], "rules": sorted(rule["rules"], key=lambda x: x[0]["attribute"])}
        for rule in sorted_by_level
    ]
    return sorted_by_rules_attribute
