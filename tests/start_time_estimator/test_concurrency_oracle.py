from datetime import datetime
from pathlib import Path

import pandas as pd
from pix_framework.io.event_log import read_csv_log
from start_time_estimator.concurrency_oracle import (
    AlphaConcurrencyOracle,
    DeactivatedConcurrencyOracle,
    DirectlyFollowsConcurrencyOracle,
    HeuristicsConcurrencyOracle,
    OverlappingConcurrencyOracle,
    _get_overlapping_matrix,
)
from start_time_estimator.config import ConcurrencyThresholds, Configuration

assets_dir = Path(__file__).parent / "assets"


def test_deactivated_concurrency_oracle():
    config = Configuration()
    concurrency_oracle = DeactivatedConcurrencyOracle(config)
    # The configuration for the algorithm is the passed
    assert concurrency_oracle.config == config
    # Empty set as concurrency by default
    assert concurrency_oracle.concurrency == {}
    # The concurrency option is deactivated, so always return pd.NaT
    assert pd.isna(concurrency_oracle.enabled_since(None, datetime.now()))
    assert not concurrency_oracle.enabling_activity_instance(None, datetime.now())
    # There is no concurrency, so always enabled since the last event finished
    assert pd.isna(concurrency_oracle.enabled_since(None, pd.Timestamp("2012-11-07T10:00:00.000+02:00")))
    assert not concurrency_oracle.enabling_activity_instance(None, pd.Timestamp("2012-11-07T10:00:00.000+02:00"))
    # pd.NaT as the enablement time of the first event in the trace
    assert pd.isna(concurrency_oracle.enabled_since(None, pd.Timestamp("2006-07-20T22:03:11.000+02:00")))
    assert not concurrency_oracle.enabling_activity_instance(None, pd.Timestamp("2006-07-20T22:03:11.000+02:00"))


def test_no_concurrency_oracle():
    config = Configuration()
    event_log = read_csv_log(assets_dir / "test_event_log_1.csv", config.log_ids, config.missing_resource)
    concurrency_oracle = DirectlyFollowsConcurrencyOracle(event_log, config)
    # No concurrency by default
    assert concurrency_oracle.concurrency == {
        "A": set(),
        "B": set(),
        "C": set(),
        "D": set(),
        "E": set(),
        "F": set(),
        "G": set(),
        "H": set(),
        "I": set(),
    }
    # The configuration for the algorithm is the passed
    assert concurrency_oracle.config == config
    # There is no concurrency, so always enabled since the last event finished
    first_trace = event_log[event_log[config.log_ids.case] == "trace-01"]
    assert (
        concurrency_oracle.enabled_since(first_trace, first_trace.iloc[4])
        == first_trace.iloc[3][config.log_ids.end_time]
    )
    assert concurrency_oracle.enabling_activity_instance(first_trace, first_trace.iloc[4]).equals(first_trace.iloc[3])
    # There is no concurrency, so always enabled since the last event finished
    third_trace = event_log[event_log[config.log_ids.case] == "trace-03"]
    assert (
        concurrency_oracle.enabled_since(third_trace, third_trace.iloc[3])
        == third_trace.iloc[2][config.log_ids.end_time]
    )
    assert concurrency_oracle.enabling_activity_instance(third_trace, third_trace.iloc[3]).equals(third_trace.iloc[2])
    # pd.NaT as the enablement time of the first event in the trace
    fourth_trace = event_log[event_log[config.log_ids.case] == "trace-04"]
    assert pd.isna(concurrency_oracle.enabled_since(fourth_trace, fourth_trace.iloc[0]))
    assert not concurrency_oracle.enabling_activity_instance(fourth_trace, fourth_trace.iloc[0])


def test_alpha_concurrency_oracle():
    config = Configuration()
    event_log = read_csv_log(assets_dir / "test_event_log_1.csv", config.log_ids, config.missing_resource)
    concurrency_oracle = AlphaConcurrencyOracle(event_log, config)
    # Concurrency between the activities that appear both one before the other
    assert concurrency_oracle.concurrency == {
        "A": set(),
        "B": set(),
        "C": {"D"},
        "D": {"C"},
        "E": set(),
        "F": set(),
        "G": set(),
        "H": set(),
        "I": set(),
    }
    # The configuration for the algorithm is the passed
    assert concurrency_oracle.config == config
    # Enabled since the previous event when there is no concurrency
    first_trace = event_log[event_log[config.log_ids.case] == "trace-01"]
    assert (
        concurrency_oracle.enabled_since(first_trace, first_trace.iloc[6])
        == first_trace.iloc[5][config.log_ids.end_time]
    )
    assert concurrency_oracle.enabling_activity_instance(first_trace, first_trace.iloc[6]).equals(first_trace.iloc[5])
    # Enabled since the previous event when there is no concurrency
    third_trace = event_log[event_log[config.log_ids.case] == "trace-03"]
    assert (
        concurrency_oracle.enabled_since(third_trace, third_trace.iloc[5])
        == third_trace.iloc[4][config.log_ids.end_time]
    )
    assert concurrency_oracle.enabling_activity_instance(third_trace, third_trace.iloc[5]).equals(third_trace.iloc[4])
    # Enabled since its causal input for an event when the previous one is concurrent
    second_trace = event_log[event_log[config.log_ids.case] == "trace-02"]
    assert (
        concurrency_oracle.enabled_since(second_trace, second_trace.iloc[3])
        == second_trace.iloc[1][config.log_ids.end_time]
    )
    assert concurrency_oracle.enabling_activity_instance(second_trace, second_trace.iloc[3]).equals(
        second_trace.iloc[1]
    )
    # Enabled since its causal input for an event when the previous one is concurrent
    fourth_trace = event_log[event_log[config.log_ids.case] == "trace-04"]
    assert (
        concurrency_oracle.enabled_since(fourth_trace, fourth_trace.iloc[3])
        == fourth_trace.iloc[1][config.log_ids.end_time]
    )
    assert concurrency_oracle.enabling_activity_instance(fourth_trace, fourth_trace.iloc[3]).equals(
        fourth_trace.iloc[1]
    )
    # pd.NaT as the enablement time of the first event in the trace
    assert pd.isna(concurrency_oracle.enabled_since(fourth_trace, fourth_trace.iloc[0]))
    assert not concurrency_oracle.enabling_activity_instance(fourth_trace, fourth_trace.iloc[0])


def test_heuristics_concurrency_oracle_simple():
    config = Configuration()
    event_log = read_csv_log(assets_dir / "test_event_log_1.csv", config.log_ids, config.missing_resource)
    concurrency_oracle = HeuristicsConcurrencyOracle(event_log, config)
    # Concurrency between the activities that appear both one before the other
    assert concurrency_oracle.concurrency == {
        "A": set(),
        "B": set(),
        "C": {"D"},
        "D": {"C"},
        "E": set(),
        "F": set(),
        "G": set(),
        "H": set(),
        "I": set(),
    }
    # The configuration for the algorithm is the passed
    assert concurrency_oracle.config == config
    # Enabled since the previous event when there is no concurrency
    first_trace = event_log[event_log[config.log_ids.case] == "trace-01"]
    assert (
        concurrency_oracle.enabled_since(first_trace, first_trace.iloc[6])
        == first_trace.iloc[5][config.log_ids.end_time]
    )
    assert concurrency_oracle.enabling_activity_instance(first_trace, first_trace.iloc[6]).equals(first_trace.iloc[5])
    # Enabled since the previous event when there is no concurrency
    third_trace = event_log[event_log[config.log_ids.case] == "trace-03"]
    assert (
        concurrency_oracle.enabled_since(third_trace, third_trace.iloc[5])
        == third_trace.iloc[4][config.log_ids.end_time]
    )
    assert concurrency_oracle.enabling_activity_instance(third_trace, third_trace.iloc[5]).equals(third_trace.iloc[4])
    # Enabled since its causal input for an event when the previous one is concurrent
    second_trace = event_log[event_log[config.log_ids.case] == "trace-02"]
    assert (
        concurrency_oracle.enabled_since(second_trace, second_trace.iloc[3])
        == second_trace.iloc[1][config.log_ids.end_time]
    )
    assert concurrency_oracle.enabling_activity_instance(second_trace, second_trace.iloc[3]).equals(
        second_trace.iloc[1]
    )
    # Enabled since its causal input for an event when the previous one is concurrent
    fourth_trace = event_log[event_log[config.log_ids.case] == "trace-04"]
    assert (
        concurrency_oracle.enabled_since(fourth_trace, fourth_trace.iloc[3])
        == fourth_trace.iloc[1][config.log_ids.end_time]
    )
    assert concurrency_oracle.enabling_activity_instance(fourth_trace, fourth_trace.iloc[3]).equals(
        fourth_trace.iloc[1]
    )
    # pd.NaT as the enablement time of the first event in the trace
    assert pd.isna(concurrency_oracle.enabled_since(fourth_trace, fourth_trace.iloc[0]))
    assert not concurrency_oracle.enabling_activity_instance(fourth_trace, fourth_trace.iloc[0])


def test_add_enable_times():
    config = Configuration()
    event_log = read_csv_log(assets_dir / "test_event_log_1.csv", config.log_ids, config.missing_resource)
    concurrency_oracle = HeuristicsConcurrencyOracle(event_log, config)
    concurrency_oracle.add_enabled_times(event_log, set_nat_to_first_event=True, include_enabling_activity=True)
    # Enabled since the previous event when there is no concurrency
    first_trace = event_log[event_log[config.log_ids.case] == "trace-01"]
    assert first_trace.iloc[6][config.log_ids.enabled_time] == first_trace.iloc[5][config.log_ids.end_time]
    assert first_trace.iloc[6][config.log_ids.enabling_activity] == first_trace.iloc[5][config.log_ids.activity]
    # Enabled since the previous event when there is no concurrency
    third_trace = event_log[event_log[config.log_ids.case] == "trace-03"]
    assert third_trace.iloc[5][config.log_ids.enabled_time] == third_trace.iloc[4][config.log_ids.end_time]
    assert third_trace.iloc[5][config.log_ids.enabling_activity] == third_trace.iloc[4][config.log_ids.activity]
    # Enabled since its causal input for an event when the previous one is concurrent
    second_trace = event_log[event_log[config.log_ids.case] == "trace-02"]
    assert second_trace.iloc[3][config.log_ids.enabled_time] == second_trace.iloc[1][config.log_ids.end_time]
    assert second_trace.iloc[3][config.log_ids.enabling_activity] == second_trace.iloc[1][config.log_ids.activity]
    # Enabled since its causal input for an event when the previous one is concurrent
    fourth_trace = event_log[event_log[config.log_ids.case] == "trace-04"]
    assert fourth_trace.iloc[3][config.log_ids.enabled_time] == fourth_trace.iloc[1][config.log_ids.end_time]
    assert fourth_trace.iloc[3][config.log_ids.enabling_activity] == fourth_trace.iloc[1][config.log_ids.activity]
    # pd.NaT as the enablement time of the first event in the trace
    assert pd.isna(fourth_trace.iloc[0][config.log_ids.enabled_time])
    assert pd.isna(fourth_trace.iloc[0][config.log_ids.enabling_activity])


def test_heuristics_concurrency_oracle_multi_parallel():
    config = Configuration()
    event_log = read_csv_log(assets_dir / "test_event_log_3.csv", config.log_ids, config.missing_resource)
    concurrency_oracle = HeuristicsConcurrencyOracle(event_log, config)
    # The configuration for the algorithm is the passed
    assert concurrency_oracle.config == config
    # Concurrency between the activities that appear both one before the other
    assert concurrency_oracle.concurrency == {
        "A": set(),
        "B": set(),
        "C": {"D", "F", "G"},
        "D": {"C", "E"},
        "E": {"D", "F", "G"},
        "F": {"C", "E"},
        "G": {"C", "E"},
        "H": set(),
        "I": set(),
    }


def test_heuristics_concurrency_oracle_multi_parallel_noise():
    config = Configuration()
    event_log = read_csv_log(assets_dir / "test_event_log_3_noise.csv", config.log_ids, config.missing_resource)
    concurrency_oracle = HeuristicsConcurrencyOracle(event_log, config)
    # The configuration for the algorithm is the passed
    assert concurrency_oracle.config == config
    # Concurrency between the activities that appear both one before the other
    assert concurrency_oracle.concurrency == {
        "A": set(),
        "B": set(),
        "C": {"D", "F", "G"},
        "D": {"C", "E"},
        "E": {"D", "F", "G"},
        "F": {"C", "E"},
        "G": {"C", "E"},
        "H": set(),
        "I": set(),
    }
    # Increasing the thresholds so the directly-follows relations and the length-2 loops
    # detection only detect when the relation happens all the times the activities appear.
    config = Configuration(concurrency_thresholds=ConcurrencyThresholds(df=1.0, l2l=1.0))
    concurrency_oracle = HeuristicsConcurrencyOracle(event_log, config)
    # The configuration for the algorithm is the passed
    assert concurrency_oracle.config == config
    # Concurrency between the activities that appear both one before the other
    assert concurrency_oracle.concurrency == {
        "A": set(),
        "B": set(),
        "C": {"D", "F", "G"},
        "D": {"C", "E"},
        "E": {"D", "F", "G"},
        "F": {"C", "E"},
        "G": {"C", "E"},
        "H": {"I"},
        "I": {"H"},
    }


def test_overlapping_concurrency_oracle_simple():
    config = Configuration()
    event_log = read_csv_log(assets_dir / "test_event_log_6.csv", config.log_ids, config.missing_resource)
    # Get concurrency relations with threshold of 75% (C || D)
    config.concurrency_thresholds.df = 0.75
    concurrency_oracle = OverlappingConcurrencyOracle(event_log, config)
    assert concurrency_oracle.concurrency == {"A": set(), "B": set(), "C": {"D"}, "D": {"C"}, "E": set(), "F": set()}
    # Get concurrency relations with threshold of 100% (none)
    config.concurrency_thresholds.df = 1.0
    concurrency_oracle = OverlappingConcurrencyOracle(event_log, config)
    assert concurrency_oracle.concurrency == {"A": set(), "B": set(), "C": set(), "D": set(), "E": set(), "F": set()}


def test__get_overlapping_matrix():
    config = Configuration()
    event_log = read_csv_log(assets_dir / "test_event_log_6.csv", config.log_ids, config.missing_resource)
    overlapping_relations = _get_overlapping_matrix(event_log, ["A", "B", "C", "D", "E", "F"], config)
    # The configuration for the algorithm is the passed
    assert overlapping_relations == {
        "A": dict(),
        "B": dict(),
        "C": {"D": 4, "E": 1},
        "D": {"C": 4, "E": 1},
        "E": {"C": 1, "D": 1},
        "F": dict(),
    }
