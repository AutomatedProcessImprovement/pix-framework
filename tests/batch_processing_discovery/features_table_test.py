from pathlib import Path

import pandas as pd
from batch_processing_discovery.config import DEFAULT_CSV_IDS
from batch_processing_discovery.features_table import _compute_features_table, _get_features

assets_dir = Path(__file__).parent / "assets"


def test__compute_features_table():
    # Read input event log
    event_log = pd.read_csv(assets_dir / "event_log_4.csv")
    event_log[DEFAULT_CSV_IDS.enabled_time] = pd.to_datetime(event_log[DEFAULT_CSV_IDS.enabled_time], utc=True)
    event_log[DEFAULT_CSV_IDS.start_time] = pd.to_datetime(event_log[DEFAULT_CSV_IDS.start_time], utc=True)
    event_log[DEFAULT_CSV_IDS.end_time] = pd.to_datetime(event_log[DEFAULT_CSV_IDS.end_time], utc=True)
    event_log[DEFAULT_CSV_IDS.batch_id] = event_log[DEFAULT_CSV_IDS.batch_id].astype("Int64")
    # Compute features table
    features_table = _compute_features_table(
        event_log=event_log,
        batched_instances=event_log[~pd.isna(event_log[DEFAULT_CSV_IDS.batch_id])],
        log_ids=DEFAULT_CSV_IDS,
    )
    # Assert features
    # 1 positive observation per batch
    assert len(features_table[features_table["outcome"] == 1]) == 4
    # 4 negative observations for the sequential with Wbatch_ready_wt, 2 negative for
    # the sequential with no Wbatch_ready_wt, and 0 for the concurrent with no WTs
    assert len(features_table[features_table["outcome"] == 0]) == 10
    assert len(features_table) == 14
    # Check that each of the positive observations are there
    positive_observations = pd.DataFrame(
        [
            {
                DEFAULT_CSV_IDS.batch_id: 0,
                DEFAULT_CSV_IDS.batch_type: "Sequential",
                DEFAULT_CSV_IDS.activity: "A",
                DEFAULT_CSV_IDS.resource: "Jonathan",
                "instant": pd.Timestamp("2021-01-01T09:30:00+00:00").value / 10**9,
                "batch_size": 3,
                "batch_ready_wt": pd.Timedelta(seconds=1800).total_seconds(),
                "batch_max_wt": pd.Timedelta(hours=1).total_seconds(),
                # 'max_cycle_time': pd.Timedelta(hours=1, seconds=1800).total_seconds(),
                "week_day": 4,
                # 'day_of_month': 1,
                "daily_hour": 9,
                # 'minute': 30,
                "outcome": 1,
            },
            {
                DEFAULT_CSV_IDS.batch_id: 1,
                DEFAULT_CSV_IDS.batch_type: "Sequential",
                DEFAULT_CSV_IDS.activity: "C",
                DEFAULT_CSV_IDS.resource: "Jolyne",
                "instant": pd.Timestamp("2021-01-01T14:00:00+00:00").value / 10**9,
                "batch_size": 3,
                "batch_ready_wt": pd.Timedelta(hours=1, seconds=1800).total_seconds(),
                "batch_max_wt": pd.Timedelta(hours=2, seconds=1800).total_seconds(),
                # 'max_cycle_time': pd.Timedelta(hours=6).total_seconds(),
                "week_day": 4,
                # 'day_of_month': 1,
                "daily_hour": 14,
                # 'minute': 00,
                "outcome": 1,
            },
            {
                DEFAULT_CSV_IDS.batch_id: 2,
                DEFAULT_CSV_IDS.batch_type: "Concurrent",
                DEFAULT_CSV_IDS.activity: "E",
                DEFAULT_CSV_IDS.resource: "Jonathan",
                "instant": pd.Timestamp("2021-01-01T16:00:00+00:00").value / 10**9,
                "batch_size": 3,
                "batch_ready_wt": pd.Timedelta(0).total_seconds(),
                "batch_max_wt": pd.Timedelta(0).total_seconds(),
                # 'max_cycle_time': pd.Timedelta(hours=8).total_seconds(),
                "week_day": 4,
                # 'day_of_month': 1,
                "daily_hour": 16,
                # 'minute': 00,
                "outcome": 1,
            },
            {
                DEFAULT_CSV_IDS.batch_id: 3,
                DEFAULT_CSV_IDS.batch_type: "Sequential",
                DEFAULT_CSV_IDS.activity: "F",
                DEFAULT_CSV_IDS.resource: "Joseph",
                "instant": pd.Timestamp("2021-01-01T17:00:00+00:00").value / 10**9,
                "batch_size": 3,
                "batch_ready_wt": pd.Timedelta(0).total_seconds(),
                "batch_max_wt": pd.Timedelta(seconds=1800).total_seconds(),
                # 'max_cycle_time': pd.Timedelta(hours=9).total_seconds(),
                "week_day": 4,
                # 'day_of_month': 1,
                "daily_hour": 17,
                # 'minute': 00,
                "outcome": 1,
            },
        ]
    )
    assert features_table[features_table["outcome"] == 1].reset_index(drop=True).equals(positive_observations)
    # Check that the negative observations are there
    neg_features_batch_0 = features_table[
        (features_table["outcome"] == 0) & (features_table[DEFAULT_CSV_IDS.batch_id] == 0)
    ]
    assert (
        neg_features_batch_0["instant"]
        .isin(
            [
                pd.Timestamp("2021-01-01T08:30:00+00:00").value / 10**9,
                pd.Timestamp("2021-01-01T08:45:00+00:00").value / 10**9,
                pd.Timestamp("2021-01-01T09:00:00+00:00").value / 10**9,
                pd.Timestamp("2021-01-01T09:10:00+00:00").value / 10**9,
                pd.Timestamp("2021-01-01T09:20:00+00:00").value / 10**9,
            ]
        )
        .all()
    )
    neg_features_batch_1 = features_table[
        (features_table["outcome"] == 0) & (features_table[DEFAULT_CSV_IDS.batch_id] == 1)
    ]
    assert (
        neg_features_batch_1["instant"]
        .isin(
            [
                pd.Timestamp("2021-01-01T11:30:00+00:00").value / 10**9,
                pd.Timestamp("2021-01-01T12:00:00+00:00").value / 10**9,
                pd.Timestamp("2021-01-01T12:30:00+00:00").value / 10**9,
                pd.Timestamp("2021-01-01T13:00:00+00:00").value / 10**9,
                pd.Timestamp("2021-01-01T13:30:00+00:00").value / 10**9,
            ]
        )
        .all()
    )
    neg_features_batch_2 = features_table[
        (features_table["outcome"] == 0) & (features_table[DEFAULT_CSV_IDS.batch_id] == 2)
    ]
    assert len(neg_features_batch_2) == 0
    neg_features_batch_3 = features_table[
        (features_table["outcome"] == 0) & (features_table[DEFAULT_CSV_IDS.batch_id] == 3)
    ]
    assert (
        neg_features_batch_3["instant"]
        .isin(
            [
                pd.Timestamp("2021-01-01T16:30:00+00:00").value / 10**9,
                pd.Timestamp("2021-01-01T16:45:00+00:00").value / 10**9,
                pd.Timestamp("2021-01-01T17:00:00+00:00").value / 10**9,
            ]
        )
        .all()
    )


def test__get_features():
    # Read input event log
    event_log = pd.read_csv(assets_dir / "event_log_4.csv")
    event_log[DEFAULT_CSV_IDS.enabled_time] = pd.to_datetime(event_log[DEFAULT_CSV_IDS.enabled_time], utc=True)
    event_log[DEFAULT_CSV_IDS.start_time] = pd.to_datetime(event_log[DEFAULT_CSV_IDS.start_time], utc=True)
    event_log[DEFAULT_CSV_IDS.end_time] = pd.to_datetime(event_log[DEFAULT_CSV_IDS.end_time], utc=True)
    event_log[DEFAULT_CSV_IDS.batch_id] = event_log[DEFAULT_CSV_IDS.batch_id].astype("Int64")
    # Assert the features of the start of a batch
    features = _get_features(
        event_log=event_log,
        instant=pd.Timestamp("2021-01-01T09:30:00+00:00"),
        batch_instance=event_log[event_log[DEFAULT_CSV_IDS.batch_id] == 0],
        outcome=1,
        log_ids=DEFAULT_CSV_IDS,
    )
    assert features == {
        DEFAULT_CSV_IDS.batch_id: 0,
        DEFAULT_CSV_IDS.batch_type: "Sequential",
        DEFAULT_CSV_IDS.activity: "A",
        DEFAULT_CSV_IDS.resource: "Jonathan",
        "instant": pd.Timestamp("2021-01-01T09:30:00+00:00"),
        "batch_size": 3,
        "batch_ready_wt": pd.Timedelta(seconds=1800),
        "batch_max_wt": pd.Timedelta(hours=1),
        # 'max_cycle_time': pd.Timedelta(hours=1, seconds=1800),
        "week_day": 4,
        # 'day_of_month': 1,
        "daily_hour": 9,
        # 'minute': 30,
        "outcome": 1,
    }
    # Assert the features of the enabling instant in the middle of the accumulation
    features = _get_features(
        event_log=event_log,
        instant=pd.Timestamp("2021-01-01T08:45:00+00:00"),
        batch_instance=event_log[
            (event_log[DEFAULT_CSV_IDS.batch_id] == 0)
            & (event_log[DEFAULT_CSV_IDS.enabled_time] <= pd.Timestamp("2021-01-01T08:45:00+00:00"))
        ],
        outcome=0,
        log_ids=DEFAULT_CSV_IDS,
    )
    assert features == {
        DEFAULT_CSV_IDS.batch_id: 0,
        DEFAULT_CSV_IDS.batch_type: "Sequential",
        DEFAULT_CSV_IDS.activity: "A",
        DEFAULT_CSV_IDS.resource: "Jonathan",
        "instant": pd.Timestamp("2021-01-01T08:45:00+00:00"),
        "batch_size": 2,
        "batch_ready_wt": pd.Timedelta(0),
        "batch_max_wt": pd.Timedelta(seconds=900),
        # 'max_cycle_time': pd.Timedelta(seconds=2700),
        "week_day": 4,
        # 'day_of_month': 1,
        "daily_hour": 8,
        # 'minute': 45,
        "outcome": 0,
    }
    # Assert the features of the enabling instant in the middle of the batch ready
    features = _get_features(
        event_log=event_log,
        instant=pd.Timestamp("2021-01-01T13:00:00+00:00"),
        batch_instance=event_log[event_log[DEFAULT_CSV_IDS.batch_id] == 1],
        outcome=0,
        log_ids=DEFAULT_CSV_IDS,
    )
    assert features == {
        DEFAULT_CSV_IDS.batch_id: 1,
        DEFAULT_CSV_IDS.batch_type: "Sequential",
        DEFAULT_CSV_IDS.activity: "C",
        DEFAULT_CSV_IDS.resource: "Jolyne",
        "instant": pd.Timestamp("2021-01-01T13:00:00+00:00"),
        "batch_size": 3,
        "batch_ready_wt": pd.Timedelta(seconds=1800),
        "batch_max_wt": pd.Timedelta(hours=1, seconds=1800),
        # 'max_cycle_time': pd.Timedelta(hours=5),
        "week_day": 4,
        # 'day_of_month': 1,
        "daily_hour": 13,
        # 'minute': 00,
        "outcome": 0,
    }
    # Assert the features of the first enabling instant
    features = _get_features(
        event_log=event_log,
        instant=pd.Timestamp("2021-01-01T16:30:00+00:00"),
        batch_instance=event_log[(event_log[DEFAULT_CSV_IDS.batch_id] == 3) & (event_log[DEFAULT_CSV_IDS.case] == 0)],
        outcome=0,
        log_ids=DEFAULT_CSV_IDS,
    )
    assert features == {
        DEFAULT_CSV_IDS.batch_id: 3,
        DEFAULT_CSV_IDS.batch_type: "Sequential",
        DEFAULT_CSV_IDS.activity: "F",
        DEFAULT_CSV_IDS.resource: "Joseph",
        "instant": pd.Timestamp("2021-01-01T16:30:00+00:00"),
        "batch_size": 1,
        "batch_ready_wt": pd.Timedelta(0),
        "batch_max_wt": pd.Timedelta(0),
        # 'max_cycle_time': pd.Timedelta(hours=8, seconds=1800),
        "week_day": 4,
        # 'day_of_month': 1,
        "daily_hour": 16,
        # 'minute': 30,
        "outcome": 0,
    }
