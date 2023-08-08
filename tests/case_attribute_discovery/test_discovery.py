import pandas as pd

from case_attribute_discovery.config import DEFAULT_CSV_IDS
from case_attribute_discovery.discovery import discover_case_attributes


def test_discover_case_attributes_discrete():
    # Create custom dataframe
    event_log = pd.DataFrame(
        data=[
            {"case_id": "c1", "Activity": "Start", "end_time": 2, "case_att_1": "THE WORLD", "case_att_2": "DIO"},
            {
                "case_id": "c1",
                "Activity": "Do something",
                "end_time": 4,
                "case_att_1": "THE WORLD",
                "case_att_2": "DIO",
            },
            {
                "case_id": "c1",
                "Activity": "Do another something",
                "end_time": 6,
                "case_att_1": "THE WORLD",
                "case_att_2": "DIO",
            },
            {"case_id": "c1", "Activity": "End", "end_time": 8, "case_att_1": "THE WORLD", "case_att_2": "DIO"},
            {"case_id": "c2", "Activity": "Start", "end_time": 2, "case_att_1": "THE WORLD", "case_att_2": "DIO"},
            {
                "case_id": "c2",
                "Activity": "Do something",
                "end_time": 4,
                "case_att_1": "THE WORLD",
                "case_att_2": "DIO",
            },
            {
                "case_id": "c2",
                "Activity": "Do another something",
                "end_time": 6,
                "case_att_1": "THE WORLD",
                "case_att_2": "DIO",
            },
            {
                "case_id": "c2",
                "Activity": "End",
                "start_time": 7,
                "end_time": 8,
                "case_att_1": "THE WORLD",
                "case_att_2": "DIO",
            },
            {"case_id": "c3", "Activity": "Start", "end_time": 2, "case_att_1": "THE WORLD", "case_att_2": "Avdol"},
            {
                "case_id": "c3",
                "Activity": "Do something",
                "end_time": 4,
                "case_att_1": "THE WORLD",
                "case_att_2": "Avdol",
            },
            {
                "case_id": "c3",
                "Activity": "Do another something",
                "end_time": 6,
                "case_att_1": "THE WORLD",
                "case_att_2": "Avdol",
            },
            {
                "case_id": "c3",
                "Activity": "End",
                "end_time": 8,
                "case_att_1": "THE WORLD",
                "case_att_2": "Avdol's father",
            },
            {
                "case_id": "c4",
                "Activity": "Start",
                "end_time": 2,
                "case_att_1": "STAR PLATINUM",
                "case_att_2": "Jotaro",
            },
            {
                "case_id": "c4",
                "Activity": "Do something",
                "end_time": 4,
                "case_att_1": "STAR PLATINUM",
                "case_att_2": "Jotaro",
            },
            {
                "case_id": "c4",
                "Activity": "Do another something",
                "end_time": 6,
                "case_att_1": "STAR PLATINUM",
                "case_att_2": "Jotaro",
            },
            {"case_id": "c4", "Activity": "End", "end_time": 8, "case_att_1": "STAR PLATINUM", "case_att_2": "Jotaro"},
            {
                "case_id": "c5",
                "Activity": "Start",
                "end_time": 2,
                "case_att_1": "STAR PLATINUM",
                "case_att_2": "Jotaro",
            },
            {
                "case_id": "c5",
                "Activity": "Do something",
                "end_time": 4,
                "case_att_1": "STAR PLATINUM",
                "case_att_2": "Jotaro",
            },
            {
                "case_id": "c5",
                "Activity": "Do another something",
                "end_time": 6,
                "case_att_1": "STAR PLATINUM",
                "case_att_2": "Jotaro",
            },
            {"case_id": "c5", "Activity": "End", "end_time": 8, "case_att_1": "STAR PLATINUM", "case_att_2": "Jotaro"},
        ]
    )
    # Get case attributes
    case_attributes = discover_case_attributes(event_log, DEFAULT_CSV_IDS)
    # Check they are the expected ones
    assert case_attributes == [
        {
            "name": "case_att_1",
            "type": "discrete",
            "values": [{"key": "THE WORLD", "probability": 0.6}, {"key": "STAR PLATINUM", "probability": 0.4}],
        }
    ]
    # Get case attributes with noise (allow up to an average 10% of different attribute values in the traces)
    case_attributes = discover_case_attributes(event_log, DEFAULT_CSV_IDS, confidence_threshold=0.9)
    # Check they are the expected ones
    assert case_attributes == [
        {
            "name": "case_att_1",
            "type": "discrete",
            "values": [{"key": "THE WORLD", "probability": 0.6}, {"key": "STAR PLATINUM", "probability": 0.4}],
        },
        {
            "name": "case_att_2",
            "type": "discrete",
            "values": [
                {"key": "DIO", "probability": 0.4},
                {"key": "Avdol", "probability": 0.2},
                {"key": "Jotaro", "probability": 0.4},
            ],
        },
    ]


def test_discover_case_attributes_continuous():
    # Create custom dataframe
    event_log = pd.DataFrame(
        data=[
            {"case_id": "c1", "Activity": "Start", "end_time": 2, "case_att_1": 1.0, "case_att_2": 35},
            {"case_id": "c1", "Activity": "Do something", "end_time": 4, "case_att_1": 1.0, "case_att_2": 35},
            {"case_id": "c1", "Activity": "Do another something", "end_time": 6, "case_att_1": 1.0, "case_att_2": 35},
            {"case_id": "c1", "Activity": "End", "end_time": 8, "case_att_1": 1.0, "case_att_2": 35},
            {"case_id": "c2", "Activity": "Start", "end_time": 2, "case_att_1": 1.0, "case_att_2": 35},
            {"case_id": "c2", "Activity": "Do something", "end_time": 4, "case_att_1": 1.0, "case_att_2": 35},
            {"case_id": "c2", "Activity": "Do another something", "end_time": 6, "case_att_1": 1.0, "case_att_2": 35},
            {"case_id": "c2", "Activity": "End", "end_time": 8, "case_att_1": 1.0, "case_att_2": 35},
            {"case_id": "c3", "Activity": "Start", "end_time": 2, "case_att_1": 1.0, "case_att_2": 35},
            {"case_id": "c3", "Activity": "Do something", "end_time": 4, "case_att_1": 1.0, "case_att_2": 35},
            {"case_id": "c3", "Activity": "Do another something", "end_time": 6, "case_att_1": 1.0, "case_att_2": 35},
            {"case_id": "c3", "Activity": "End", "start_time": 7, "end_time": 8, "case_att_1": 1.0, "case_att_2": 36},
            {"case_id": "c4", "Activity": "Start", "end_time": 2, "case_att_1": 1.0, "case_att_2": 35},
            {"case_id": "c4", "Activity": "Do something", "end_time": 4, "case_att_1": 1.0, "case_att_2": 35},
            {"case_id": "c4", "Activity": "Do another something", "end_time": 6, "case_att_1": 1.0, "case_att_2": 35},
            {"case_id": "c4", "Activity": "End", "end_time": 8, "case_att_1": 1.0, "case_att_2": 35},
            {"case_id": "c5", "Activity": "Start", "end_time": 2, "case_att_1": 1.0, "case_att_2": 35},
            {"case_id": "c5", "Activity": "Do something", "end_time": 4, "case_att_1": 1.0, "case_att_2": 35},
            {"case_id": "c5", "Activity": "Do another something", "end_time": 6, "case_att_1": 1.0, "case_att_2": 35},
            {"case_id": "c5", "Activity": "End", "end_time": 8, "case_att_1": 1.0, "case_att_2": 35},
        ]
    )
    # Get case attributes
    case_attributes = discover_case_attributes(event_log, DEFAULT_CSV_IDS)
    # Check they are the expected ones
    assert case_attributes == [
        {
            "name": "case_att_1",
            "type": "continuous",
            "values": {"distribution_name": "fix", "distribution_params": [{"value": 1.0}]},
        }
    ]
    # Get case attributes with noise (allow up to an average 10% of different attribute values in the traces)
    case_attributes = discover_case_attributes(event_log, DEFAULT_CSV_IDS, confidence_threshold=0.9)
    # Check they are the expected ones
    assert case_attributes == [
        {
            "name": "case_att_1",
            "type": "continuous",
            "values": {"distribution_name": "fix", "distribution_params": [{"value": 1.0}]},
        },
        {
            "name": "case_att_2",
            "type": "continuous",
            "values": {"distribution_name": "fix", "distribution_params": [{"value": 35}]},
        },
    ]
