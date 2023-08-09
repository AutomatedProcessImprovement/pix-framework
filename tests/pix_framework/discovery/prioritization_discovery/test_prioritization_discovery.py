from pathlib import Path

import pandas as pd
from pix_framework.discovery.prioritization_discovery.discovery import (
    _discover_prioritized_instances,
    _split_to_individual_observations,
    discover_priority_rules,
)
from pix_framework.io.event_log import DEFAULT_CSV_IDS

assets_dir = Path(__file__).parent / "assets"


def test_discover_prioritized_instances():
    # Read event log
    event_log = pd.read_csv(assets_dir / "event_log_1.csv")
    event_log[DEFAULT_CSV_IDS.enabled_time] = pd.to_datetime(event_log[DEFAULT_CSV_IDS.enabled_time], utc=True)
    event_log[DEFAULT_CSV_IDS.start_time] = pd.to_datetime(event_log[DEFAULT_CSV_IDS.start_time], utc=True)
    event_log[DEFAULT_CSV_IDS.end_time] = pd.to_datetime(event_log[DEFAULT_CSV_IDS.end_time], utc=True)
    # Discover prioritization
    attributes = [DEFAULT_CSV_IDS.activity]
    prioritizations = _discover_prioritized_instances(event_log, attributes)
    prioritizations.sort_values(["Activity"], inplace=True)
    assert prioritizations.equals(
        pd.DataFrame(
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
    )


def test_discover_prioritized_instances_with_extra_attribute():
    # Read event log
    event_log = pd.read_csv(assets_dir / "event_log_2.csv")
    event_log[DEFAULT_CSV_IDS.enabled_time] = pd.to_datetime(event_log[DEFAULT_CSV_IDS.enabled_time], utc=True)
    event_log[DEFAULT_CSV_IDS.start_time] = pd.to_datetime(event_log[DEFAULT_CSV_IDS.start_time], utc=True)
    event_log[DEFAULT_CSV_IDS.end_time] = pd.to_datetime(event_log[DEFAULT_CSV_IDS.end_time], utc=True)
    # Discover prioritization
    attributes = [DEFAULT_CSV_IDS.activity, "loan_amount"]
    prioritizations = _discover_prioritized_instances(event_log, attributes)
    prioritizations.sort_values(["Activity", "loan_amount", "outcome"], inplace=True)
    assert prioritizations.equals(
        pd.DataFrame(
            data=[
                ["A", 500, 0],
                ["A", 500, 0],
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
                ["C", 1000, 1],
                ["C", 1000, 1],
            ],
            index=[0, 1, 2, 2, 3, 4, 5, 6, 4, 0, 3, 7, 6, 7, 1, 5],
            columns=["Activity", "loan_amount", "outcome"],
        )
    )


def test__split_to_individual_observations():
    # Create simple prioritizations with only the activity
    prioritizations = pd.DataFrame(
        [["B", "C"], ["B", "C"], ["B", "C"], ["B", "C"], ["B", "C"], ["B", "C"]],
        columns=["delayed_Activity", "prioritized_Activity"],
    )
    # Split the prioritizations to the individual delayed/prioritized instances
    prioritized_instances = _split_to_individual_observations(
        prioritizations, ["delayed_Activity"], ["prioritized_Activity"], "outcome"
    )
    # Assert that the split was done correctly, even maintaining the indexes
    assert prioritized_instances.equals(
        pd.DataFrame(
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
    )


def test__split_to_individual_observations_with_extra_attribute():
    # Create simple prioritizations with only the activity
    prioritizations = pd.DataFrame(
        [
            ["A", 500, "B", 1000],
            ["A", 500, "C", 1000],
            ["B", 100, "A", 500],
            ["B", 100, "B", 500],
            ["B", 100, "B", 1000],
            ["B", 100, "C", 500],
            ["B", 100, "C", 1000],
            ["C", 100, "C", 500],
        ],
        columns=["delayed_Activity", "delayed_loan_amount", "prioritized_Activity", "prioritized_loan_amount"],
    )
    # Split the prioritizations to the individual delayed/prioritized instances
    prioritized_instances = _split_to_individual_observations(
        prioritizations,
        ["delayed_Activity", "delayed_loan_amount"],
        ["prioritized_Activity", "prioritized_loan_amount"],
        "outcome",
    )
    # Assert that the split was done correctly, even maintaining the indexes
    assert prioritized_instances.equals(
        pd.DataFrame(
            data=[
                ["A", 500, 0],
                ["A", 500, 0],
                ["B", 100, 0],
                ["B", 100, 0],
                ["B", 100, 0],
                ["B", 100, 0],
                ["B", 100, 0],
                ["C", 100, 0],
                ["B", 1000, 1],
                ["C", 1000, 1],
                ["A", 500, 1],
                ["B", 500, 1],
                ["B", 1000, 1],
                ["C", 500, 1],
                ["C", 1000, 1],
                ["C", 500, 1],
            ],
            index=[0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7],
            columns=["Activity", "loan_amount", "outcome"],
        )
    )


def test_discover_priority_rules_naive():
    # Read event log
    event_log = pd.read_csv(assets_dir / "event_log_3.csv")
    event_log[DEFAULT_CSV_IDS.enabled_time] = pd.to_datetime(event_log[DEFAULT_CSV_IDS.enabled_time], utc=True)
    event_log[DEFAULT_CSV_IDS.start_time] = pd.to_datetime(event_log[DEFAULT_CSV_IDS.start_time], utc=True)
    event_log[DEFAULT_CSV_IDS.end_time] = pd.to_datetime(event_log[DEFAULT_CSV_IDS.end_time], utc=True)
    # Discover prioritization
    attributes = ["urgency"]
    # Get priority levels and their rules
    prioritization_levels = discover_priority_rules(event_log, attributes)
    # Assert expected levels and rules
    assert prioritization_levels == [
        {"priority_level": 1, "rules": [[{"attribute": "urgency", "comparison": "=", "value": "high"}]]},
        {"priority_level": 2, "rules": [[{"attribute": "urgency", "comparison": "=", "value": "medium"}]]},
    ]
