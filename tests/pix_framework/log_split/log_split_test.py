import pandas as pd

from pix_framework.log_ids import DEFAULT_CSV_IDS
from pix_framework.log_split.log_split import (
    split_log_training_validation_event_wise,
    split_log_training_validation_trace_wise,
)


def test_split_log_training_validation_event_wise():
    # Create event log mock
    data = pd.DataFrame(
        {
            "case_id": [
                "0",
                "0",
                "0",
                "1",
                "0",
                "1",
                "2",
                "0",
                "1",
                "2",
                "1",
                "2",
                "3",
                "1",
                "2",
                "3",
                "2",
                "3",
                "3",
                "3",
            ],
            "start_time": [1, 2, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 9, 10],
            "end_time": [6, 2, 3, 5, 4, 6, 7, 8, 9, 8, 6, 1, 2, 4, 3, 2, 5, 32, 13, 25],
        }
    )
    # Split it in 50-50 without removing partial traces from validation
    train, test = split_log_training_validation_event_wise(data, DEFAULT_CSV_IDS, 0.5)
    # Assert expected result
    assert train.equals(data.head(10))
    assert test.equals(data.tail(10))
    # Split it in 90-10 without removing partial traces from validation
    train, test = split_log_training_validation_event_wise(data, DEFAULT_CSV_IDS, 0.9)
    # Assert expected result
    assert train.equals(data.head(18))
    assert test.equals(data.tail(2))
    # Split it in 50-50 removing partial traces from validation
    train, test = split_log_training_validation_event_wise(
        data, DEFAULT_CSV_IDS, 0.5, remove_partial_traces_from_validation=True
    )
    # Assert expected result
    assert train.equals(data.head(10))
    assert test.equals(data[data["case_id"] == "3"])


def test_split_log_training_validation_trace_wise():
    # Create event log mock
    data = pd.DataFrame(
        {
            "case_id": [
                "0",
                "0",
                "0",
                "1",
                "0",
                "1",
                "2",
                "0",
                "1",
                "2",
                "1",
                "2",
                "3",
                "1",
                "2",
                "3",
                "2",
                "3",
                "3",
                "3",
            ],
            "start_time": [1, 2, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 9, 10],
            "end_time": [6, 2, 3, 5, 4, 6, 7, 8, 9, 8, 6, 1, 2, 4, 3, 2, 5, 32, 13, 25],
        }
    )
    # Split it in 50-50
    train, test = split_log_training_validation_trace_wise(data, DEFAULT_CSV_IDS, 0.5)
    # Assert expected result
    assert train.equals(data[data["case_id"].isin(["0", "1"])])
    assert test.equals(data[data["case_id"].isin(["2", "3"])])
    # Split it in 90-10 without removing partial traces from validation
    train, test = split_log_training_validation_trace_wise(data, DEFAULT_CSV_IDS, 0.7)
    # Assert expected result
    assert train.equals(data[data["case_id"].isin(["0", "1", "2"])])
    assert test.equals(data[data["case_id"].isin(["3"])])
