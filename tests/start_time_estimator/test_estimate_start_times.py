from datetime import timedelta
from pathlib import Path

import pandas as pd
from pix_framework.io.event_log import read_csv_log
from start_time_estimator.config import (
    ConcurrencyOracleType,
    Configuration,
    OutlierStatistic,
    ReEstimationMethod,
    ResourceAvailabilityType,
)
from start_time_estimator.estimator import StartTimeEstimator

assets_dir = Path(__file__).parent / "assets"


def test_estimate_start_times_only_resource():
    config = Configuration(
        re_estimation_method=ReEstimationMethod.SET_INSTANT,
        concurrency_oracle_type=ConcurrencyOracleType.DEACTIVATED,
        resource_availability_type=ResourceAvailabilityType.SIMPLE,
    )
    event_log = read_csv_log(assets_dir / "test_event_log_1.csv", config.log_ids, config.missing_resource)
    # Estimate start times
    start_time_estimator = StartTimeEstimator(event_log, config)
    extended_event_log = start_time_estimator.estimate()
    # Traces
    first_trace = extended_event_log[extended_event_log[config.log_ids.case] == "trace-01"]
    second_trace = extended_event_log[extended_event_log[config.log_ids.case] == "trace-02"]
    third_trace = extended_event_log[extended_event_log[config.log_ids.case] == "trace-03"]
    fourth_trace = extended_event_log[extended_event_log[config.log_ids.case] == "trace-04"]
    # The start time of initial events is their end time (instant events)
    assert first_trace.iloc[0][config.log_ids.estimated_start_time] == first_trace.iloc[0][config.log_ids.end_time]
    assert fourth_trace.iloc[0][config.log_ids.estimated_start_time] == fourth_trace.iloc[0][config.log_ids.end_time]
    # The start time of all other events is the availability of the resource (concurrency deactivated)
    assert second_trace.iloc[3][config.log_ids.estimated_start_time] == first_trace.iloc[2][config.log_ids.end_time]
    assert third_trace.iloc[3][config.log_ids.estimated_start_time] == third_trace.iloc[1][config.log_ids.end_time]
    assert fourth_trace.iloc[3][config.log_ids.estimated_start_time] == third_trace.iloc[2][config.log_ids.end_time]
    assert fourth_trace.iloc[4][config.log_ids.estimated_start_time] == second_trace.iloc[4][config.log_ids.end_time]
    assert first_trace.iloc[2][config.log_ids.estimated_start_time] == fourth_trace.iloc[3][config.log_ids.end_time]


def test_estimate_start_times_instant():
    config = Configuration(
        re_estimation_method=ReEstimationMethod.SET_INSTANT,
        concurrency_oracle_type=ConcurrencyOracleType.DF,
        resource_availability_type=ResourceAvailabilityType.SIMPLE,
        reuse_current_start_times=True,
    )
    event_log = read_csv_log(assets_dir / "test_event_log_1.csv", config.log_ids, config.missing_resource)
    # Set one start timestamp manually
    event_log[config.log_ids.start_time] = pd.NaT
    manually_added_timestamp = pd.Timestamp("2006-11-07 12:33:00+02:00")
    event_log.loc[
        (event_log[config.log_ids.case] == "trace-01") & (event_log[config.log_ids.activity] == "C"),
        config.log_ids.start_time,
    ] = manually_added_timestamp
    event_log[config.log_ids.start_time] = pd.to_datetime(event_log[config.log_ids.start_time], utc=True)
    # Estimate start times
    start_time_estimator = StartTimeEstimator(event_log, config)
    extended_event_log = start_time_estimator.estimate()
    # The start time of initial events is the first timestamp of their trace end time (instant events)
    first_trace = extended_event_log[extended_event_log[config.log_ids.case] == "trace-01"]
    assert first_trace.iloc[0][config.log_ids.estimated_start_time] == first_trace.iloc[0][config.log_ids.end_time]
    fourth_trace = extended_event_log[extended_event_log[config.log_ids.case] == "trace-04"]
    assert fourth_trace.iloc[0][config.log_ids.estimated_start_time] == fourth_trace.iloc[0][config.log_ids.end_time]
    # The start time of an event with its resource free but immediately
    # following its previous one is the end time of the previous one.
    second_trace = extended_event_log[extended_event_log[config.log_ids.case] == "trace-02"]
    assert second_trace.iloc[3][config.log_ids.estimated_start_time] == second_trace.iloc[2][config.log_ids.end_time]
    third_trace = extended_event_log[extended_event_log[config.log_ids.case] == "trace-03"]
    assert third_trace.iloc[3][config.log_ids.estimated_start_time] == third_trace.iloc[2][config.log_ids.end_time]
    # The start time of an event enabled for a long time but with its resource
    # busy in other activities is the end time of its resource's last activity.
    assert fourth_trace.iloc[3][config.log_ids.estimated_start_time] == third_trace.iloc[2][config.log_ids.end_time]
    assert fourth_trace.iloc[4][config.log_ids.estimated_start_time] == second_trace.iloc[4][config.log_ids.end_time]
    # The event with predefined start time was not predicted
    assert first_trace.iloc[2][config.log_ids.estimated_start_time] == manually_added_timestamp


def test_bot_resources_and_instant_activities():
    config = Configuration(
        re_estimation_method=ReEstimationMethod.SET_INSTANT,
        concurrency_oracle_type=ConcurrencyOracleType.DF,
        resource_availability_type=ResourceAvailabilityType.SIMPLE,
        bot_resources={"Marcus"},
        instant_activities={"H", "I"},
    )
    event_log = read_csv_log(assets_dir / "test_event_log_1.csv", config.log_ids, config.missing_resource)
    # Estimate start times
    start_time_estimator = StartTimeEstimator(event_log, config)
    extended_event_log = start_time_estimator.estimate()
    # The events performed by bot resources, or being instant activities are instant
    second_trace = extended_event_log[extended_event_log[config.log_ids.case] == "trace-02"]
    fourth_trace = extended_event_log[extended_event_log[config.log_ids.case] == "trace-04"]
    assert second_trace.iloc[2][config.log_ids.estimated_start_time] == second_trace.iloc[2][config.log_ids.end_time]
    assert fourth_trace.iloc[6][config.log_ids.estimated_start_time] == fourth_trace.iloc[6][config.log_ids.end_time]
    assert fourth_trace.iloc[7][config.log_ids.estimated_start_time] == fourth_trace.iloc[7][config.log_ids.end_time]
    # The start time of initial events (with no bot resources nor instant activities) is the end time (instant events)
    assert second_trace.iloc[0][config.log_ids.estimated_start_time] == second_trace.iloc[0][config.log_ids.end_time]
    assert fourth_trace.iloc[0][config.log_ids.estimated_start_time] == fourth_trace.iloc[0][config.log_ids.end_time]
    # The start time of an event (no bot resource nor instant activity) with its resource
    # free but immediately following its previous one is the end time of the previous one.
    second_trace = extended_event_log[extended_event_log[config.log_ids.case] == "trace-02"]
    assert second_trace.iloc[3][config.log_ids.estimated_start_time] == second_trace.iloc[2][config.log_ids.end_time]
    third_trace = extended_event_log[extended_event_log[config.log_ids.case] == "trace-03"]
    assert third_trace.iloc[3][config.log_ids.estimated_start_time] == third_trace.iloc[2][config.log_ids.end_time]
    # The start time of an event (no bot resource nor instant activity) enabled for a long time
    # but with its resource busy in other activities is the end time of its resource's last activity.
    assert fourth_trace.iloc[3][config.log_ids.estimated_start_time] == third_trace.iloc[2][config.log_ids.end_time]
    assert fourth_trace.iloc[4][config.log_ids.estimated_start_time] == second_trace.iloc[4][config.log_ids.end_time]


def test_repair_activities_with_duration_over_threshold():
    config = Configuration(
        re_estimation_method=ReEstimationMethod.MEDIAN,
        concurrency_oracle_type=ConcurrencyOracleType.DF,
        resource_availability_type=ResourceAvailabilityType.SIMPLE,
        outlier_statistic=OutlierStatistic.MEDIAN,
        outlier_threshold=1.6,
    )
    event_log = read_csv_log(assets_dir / "test_event_log_1.csv", config.log_ids, config.missing_resource)
    # Estimate start times
    start_time_estimator = StartTimeEstimator(event_log, config)
    extended_event_log = start_time_estimator.estimate()
    # The start time of an event (with duration under the threshold) with its resource
    # free but immediately following its previous one is the end time of the previous one.
    second_trace = extended_event_log[extended_event_log[config.log_ids.case] == "trace-02"]
    assert second_trace.iloc[3][config.log_ids.estimated_start_time] == second_trace.iloc[2][config.log_ids.end_time]
    # The start time of an event (with duration under the threshold) enabled for a long time
    # but with its resource busy in other activities is the end time of its resource's last activity.
    third_trace = extended_event_log[extended_event_log[config.log_ids.case] == "trace-03"]
    fourth_trace = extended_event_log[extended_event_log[config.log_ids.case] == "trace-04"]
    assert fourth_trace.iloc[3][config.log_ids.estimated_start_time] == third_trace.iloc[2][config.log_ids.end_time]
    assert fourth_trace.iloc[4][config.log_ids.estimated_start_time] == second_trace.iloc[4][config.log_ids.end_time]
    # The events with estimated durations over the threshold where re-estimated
    first_trace = extended_event_log[extended_event_log[config.log_ids.case] == "trace-01"]
    assert first_trace.iloc[1][config.log_ids.estimated_start_time] == first_trace.iloc[1][
        config.log_ids.end_time
    ] - timedelta(minutes=49.6)
    assert third_trace.iloc[2][config.log_ids.estimated_start_time] == third_trace.iloc[2][
        config.log_ids.end_time
    ] - timedelta(minutes=11.2)
    assert first_trace.iloc[6][config.log_ids.estimated_start_time] == first_trace.iloc[6][
        config.log_ids.end_time
    ] - timedelta(minutes=38.4)


def test_estimate_start_times_mode():
    config = Configuration(
        re_estimation_method=ReEstimationMethod.MODE,
        concurrency_oracle_type=ConcurrencyOracleType.DF,
        resource_availability_type=ResourceAvailabilityType.SIMPLE,
    )
    event_log = read_csv_log(assets_dir / "test_event_log_1.csv", config.log_ids, config.missing_resource)
    # Estimate start times
    start_time_estimator = StartTimeEstimator(event_log, config)
    extended_event_log = start_time_estimator.estimate()
    # The start time of initial events is the most frequent duration
    third_trace = extended_event_log[extended_event_log[config.log_ids.case] == "trace-03"]
    first_trace = extended_event_log[extended_event_log[config.log_ids.case] == "trace-01"]
    assert first_trace.iloc[0][config.log_ids.estimated_start_time] == first_trace.iloc[0][config.log_ids.end_time] - (
        third_trace.iloc[0][config.log_ids.end_time] - third_trace.iloc[0][config.log_ids.estimated_start_time]
    )
    second_trace = extended_event_log[extended_event_log[config.log_ids.case] == "trace-02"]
    assert second_trace.iloc[0][config.log_ids.estimated_start_time] == second_trace.iloc[0][
        config.log_ids.end_time
    ] - (third_trace.iloc[0][config.log_ids.end_time] - third_trace.iloc[0][config.log_ids.estimated_start_time])


def test_replace_recorded_start_times_with_estimation():
    config = Configuration(
        re_estimation_method=ReEstimationMethod.MODE,
        concurrency_oracle_type=ConcurrencyOracleType.DF,
        resource_availability_type=ResourceAvailabilityType.SIMPLE,
    )
    event_log = read_csv_log(assets_dir / "test_event_log_1.csv", config.log_ids, config.missing_resource)
    # Estimate start times
    start_time_estimator = StartTimeEstimator(event_log, config)
    extended_event_log = start_time_estimator.estimate(replace_recorded_start_times=True)
    # The start time of initial events is the most frequent duration
    third_trace = extended_event_log[extended_event_log[config.log_ids.case] == "trace-03"]
    first_trace = extended_event_log[extended_event_log[config.log_ids.case] == "trace-01"]
    assert first_trace.iloc[0][config.log_ids.start_time] == first_trace.iloc[0][config.log_ids.end_time] - (
        third_trace.iloc[0][config.log_ids.end_time] - third_trace.iloc[0][config.log_ids.start_time]
    )
    second_trace = extended_event_log[extended_event_log[config.log_ids.case] == "trace-02"]
    assert second_trace.iloc[0][config.log_ids.start_time] == second_trace.iloc[0][config.log_ids.end_time] - (
        third_trace.iloc[0][config.log_ids.end_time] - third_trace.iloc[0][config.log_ids.start_time]
    )
    assert config.log_ids.estimated_start_time not in extended_event_log.columns


def test_set_instant_non_estimated_start_times():
    config = Configuration(
        re_estimation_method=ReEstimationMethod.SET_INSTANT,
        concurrency_oracle_type=ConcurrencyOracleType.DF,
        resource_availability_type=ResourceAvailabilityType.SIMPLE,
    )
    event_log = read_csv_log(assets_dir / "test_event_log_2.csv", config.log_ids, config.missing_resource)
    event_log[config.log_ids.estimated_start_time] = pd.to_datetime(
        event_log[config.log_ids.estimated_start_time], utc=True
    )
    # Estimate start times
    start_time_estimator = StartTimeEstimator(event_log, config)
    start_time_estimator._set_instant_non_estimated_start_times(event_log)
    extended_event_log = start_time_estimator.event_log
    # The start time of non-estimated events is the end time (instant events)
    first_trace = extended_event_log[extended_event_log[config.log_ids.case] == "trace-01"]
    assert first_trace.iloc[0][config.log_ids.estimated_start_time] == first_trace.iloc[0][config.log_ids.end_time]
    assert first_trace.iloc[2][config.log_ids.estimated_start_time] == first_trace.iloc[2][config.log_ids.end_time]
    second_trace = extended_event_log[extended_event_log[config.log_ids.case] == "trace-02"]
    assert second_trace.iloc[1][config.log_ids.estimated_start_time] == second_trace.iloc[1][config.log_ids.end_time]


def test_set_mode_non_estimated_start_times():
    config = Configuration(
        re_estimation_method=ReEstimationMethod.MODE,
        concurrency_oracle_type=ConcurrencyOracleType.DF,
        resource_availability_type=ResourceAvailabilityType.SIMPLE,
    )
    event_log = read_csv_log(assets_dir / "test_event_log_2.csv", config.log_ids, config.missing_resource)
    event_log[config.log_ids.estimated_start_time] = pd.to_datetime(
        event_log[config.log_ids.estimated_start_time], utc=True
    )
    # Estimate start times
    start_time_estimator = StartTimeEstimator(event_log, config)
    start_time_estimator._re_estimate_non_estimated_start_times(event_log)
    extended_event_log = start_time_estimator.event_log
    # The start time of non-estimated events is the most frequent duration
    first_trace = extended_event_log[extended_event_log[config.log_ids.case] == "trace-01"]
    assert first_trace.iloc[0][config.log_ids.estimated_start_time] == (
        first_trace.iloc[0][config.log_ids.end_time] - timedelta(minutes=15)
    )
    second_trace = extended_event_log[extended_event_log[config.log_ids.case] == "trace-02"]
    assert second_trace.iloc[1][config.log_ids.estimated_start_time] == (
        second_trace.iloc[1][config.log_ids.end_time] - timedelta(minutes=30)
    )
    # The start time of a non-estimated event of an activity with no durations is instant
    assert first_trace.iloc[2][config.log_ids.estimated_start_time] == first_trace.iloc[2][config.log_ids.end_time]
    assert second_trace.iloc[2][config.log_ids.estimated_start_time] == second_trace.iloc[2][config.log_ids.end_time]


def test_set_mean_non_estimated_start_times():
    config = Configuration(
        re_estimation_method=ReEstimationMethod.MEAN,
        concurrency_oracle_type=ConcurrencyOracleType.DF,
        resource_availability_type=ResourceAvailabilityType.SIMPLE,
    )
    event_log = read_csv_log(assets_dir / "test_event_log_2.csv", config.log_ids, config.missing_resource)
    event_log[config.log_ids.estimated_start_time] = pd.to_datetime(
        event_log[config.log_ids.estimated_start_time], utc=True
    )
    # Estimate start times
    start_time_estimator = StartTimeEstimator(event_log, config)
    start_time_estimator._re_estimate_non_estimated_start_times(event_log)
    extended_event_log = start_time_estimator.event_log
    # The start time of non-estimated events is the most frequent duration
    first_trace = extended_event_log[extended_event_log[config.log_ids.case] == "trace-01"]
    assert first_trace.iloc[0][config.log_ids.estimated_start_time] == first_trace.iloc[0][
        config.log_ids.end_time
    ] - timedelta(minutes=13)
    second_trace = extended_event_log[extended_event_log[config.log_ids.case] == "trace-02"]
    assert second_trace.iloc[1][config.log_ids.estimated_start_time] == second_trace.iloc[1][
        config.log_ids.end_time
    ] - timedelta(minutes=24.5)
    # The start time of a non-estimated event of an activity with no durations is instant
    assert first_trace.iloc[2][config.log_ids.estimated_start_time] == first_trace.iloc[2][config.log_ids.end_time]
    assert second_trace.iloc[2][config.log_ids.estimated_start_time] == second_trace.iloc[2][config.log_ids.end_time]


def test_set_median_non_estimated_start_times():
    config = Configuration(
        re_estimation_method=ReEstimationMethod.MEDIAN,
        concurrency_oracle_type=ConcurrencyOracleType.DF,
        resource_availability_type=ResourceAvailabilityType.SIMPLE,
    )
    event_log = read_csv_log(assets_dir / "test_event_log_2.csv", config.log_ids, config.missing_resource)
    event_log[config.log_ids.estimated_start_time] = pd.to_datetime(
        event_log[config.log_ids.estimated_start_time], utc=True
    )
    # Estimate start times
    start_time_estimator = StartTimeEstimator(event_log, config)
    start_time_estimator._re_estimate_non_estimated_start_times(event_log)
    extended_event_log = start_time_estimator.event_log
    # The start time of non-estimated events is the most frequent duration
    first_trace = extended_event_log[extended_event_log[config.log_ids.case] == "trace-01"]
    assert first_trace.iloc[0][config.log_ids.estimated_start_time] == (
        first_trace.iloc[0][config.log_ids.end_time] - timedelta(minutes=13.5)
    )
    second_trace = extended_event_log[extended_event_log[config.log_ids.case] == "trace-02"]
    assert second_trace.iloc[1][config.log_ids.estimated_start_time] == (
        second_trace.iloc[1][config.log_ids.end_time] - timedelta(minutes=25)
    )
    # The start time of a non-estimated event of an activity with no durations is instant
    assert first_trace.iloc[2][config.log_ids.estimated_start_time] == first_trace.iloc[2][config.log_ids.end_time]
    assert second_trace.iloc[2][config.log_ids.estimated_start_time] == second_trace.iloc[2][config.log_ids.end_time]


def test_get_activity_duration():
    durationsA = [timedelta(2), timedelta(2), timedelta(4), timedelta(6), timedelta(7), timedelta(9)]
    durationsB = [timedelta(2), timedelta(2), timedelta(4), timedelta(8)]
    durationsC = [timedelta(2), timedelta(2), timedelta(3)]
    # MEAN
    config = Configuration(
        re_estimation_method=ReEstimationMethod.MEAN,
        concurrency_oracle_type=ConcurrencyOracleType.DF,
        resource_availability_type=ResourceAvailabilityType.SIMPLE,
    )
    event_log = read_csv_log(assets_dir / "test_event_log_2.csv", config.log_ids, config.missing_resource)
    start_time_estimator = StartTimeEstimator(event_log, config)
    assert start_time_estimator._get_activity_duration(durationsA) == timedelta(5)
    assert start_time_estimator._get_activity_duration(durationsB) == timedelta(4)
    assert start_time_estimator._get_activity_duration(durationsC) == timedelta(days=2, hours=8)
    # MEDIAN
    config = Configuration(
        re_estimation_method=ReEstimationMethod.MEDIAN,
        concurrency_oracle_type=ConcurrencyOracleType.DF,
        resource_availability_type=ResourceAvailabilityType.SIMPLE,
    )
    event_log = read_csv_log(assets_dir / "test_event_log_2.csv", config.log_ids, config.missing_resource)
    start_time_estimator = StartTimeEstimator(event_log, config)
    assert start_time_estimator._get_activity_duration(durationsA) == timedelta(5)
    assert start_time_estimator._get_activity_duration(durationsB) == timedelta(3)
    assert start_time_estimator._get_activity_duration(durationsC) == timedelta(2)
    # MODE
    config = Configuration(
        re_estimation_method=ReEstimationMethod.MODE,
        concurrency_oracle_type=ConcurrencyOracleType.DF,
        resource_availability_type=ResourceAvailabilityType.SIMPLE,
    )
    event_log = read_csv_log(assets_dir / "test_event_log_2.csv", config.log_ids, config.missing_resource)
    start_time_estimator = StartTimeEstimator(event_log, config)
    assert start_time_estimator._get_activity_duration(durationsA) == timedelta(2)
    assert start_time_estimator._get_activity_duration(durationsB) == timedelta(2)
    assert start_time_estimator._get_activity_duration(durationsC) == timedelta(2)
