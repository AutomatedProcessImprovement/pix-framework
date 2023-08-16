from pathlib import Path

import pandas as pd
import pytest

from pix_framework.discovery.start_time_estimator.concurrency_oracle import OverlappingConcurrencyOracle, Mode
from pix_framework.discovery.start_time_estimator.config import (
    ConcurrencyThresholds,
)
from pix_framework.discovery.start_time_estimator.config import (
    Configuration as StartTimeEstimatorConfiguration,
)
from pix_framework.io.event_log import EventLogIDs, DEFAULT_CSV_IDS, read_csv_log

assets_dir = Path(__file__).parent / "assets"


def _add_enabled_times(log: pd.DataFrame, log_ids: EventLogIDs):
    configuration = StartTimeEstimatorConfiguration(
        log_ids=log_ids,
        concurrency_thresholds=ConcurrencyThresholds(df=0.75),
        consider_start_times=True,
    )
    OverlappingConcurrencyOracle(log, configuration).add_enabled_times(log, mode=Mode.ORIGINAL)


def _add_enabled_times_parallel(log: pd.DataFrame, log_ids: EventLogIDs):
    configuration = StartTimeEstimatorConfiguration(
        log_ids=log_ids,
        concurrency_thresholds=ConcurrencyThresholds(df=0.75),
        consider_start_times=True,
    )
    OverlappingConcurrencyOracle(log, configuration, optimized=True).add_enabled_times(log, mode=Mode.PARALLEL)


def _add_enabled_times_parallel_polars(log: pd.DataFrame, log_ids: EventLogIDs):
    configuration = StartTimeEstimatorConfiguration(
        log_ids=log_ids,
        concurrency_thresholds=ConcurrencyThresholds(df=0.75),
        consider_start_times=True,
    )
    OverlappingConcurrencyOracle(log, configuration, optimized=True).add_enabled_times(log, mode=Mode.PARALLEL_POLARS)


test_data = [
    {
        "log_path": assets_dir / "test_event_log_3_noise.csv",
        "log_ids": DEFAULT_CSV_IDS,
    }
]

test_data_simple = [
    {
        "log_path": assets_dir / "test_event_log_4.csv",
        "log_ids": DEFAULT_CSV_IDS,
    }
]


@pytest.mark.parametrize(
    "test_data_",
    test_data,
    ids=[item["log_path"].stem for item in test_data],
)
def test_enabled_times_discovery(test_data_):
    log_ids = test_data_["log_ids"]
    log = read_csv_log(test_data_["log_path"], log_ids)

    _add_enabled_times(log, log_ids)

    assert log[log_ids.enabled_time].isna().sum() == 0


@pytest.mark.parametrize(
    "test_data_",
    test_data,
    ids=[item["log_path"].stem for item in test_data],
)
def test_enabled_times_discovery_parallel(test_data_):
    log_ids = test_data_["log_ids"]
    log = read_csv_log(test_data_["log_path"], log_ids)

    _add_enabled_times_parallel(log, log_ids)

    assert log[log_ids.enabled_time].isna().sum() == 0


@pytest.mark.parametrize(
    "test_data_",
    test_data,
    ids=[item["log_path"].stem for item in test_data],
)
def test_enabled_times_discovery_parallel_optimized(test_data_):
    log_ids = test_data_["log_ids"]
    log = read_csv_log(test_data_["log_path"], log_ids)

    _add_enabled_times_parallel_polars(log, log_ids)

    assert log[log_ids.enabled_time].isna().sum() == 0


@pytest.mark.parametrize(
    "test_data_",
    test_data,
    ids=[item["log_path"].stem for item in test_data],
)
def test_enabled_times_discovery_correctness(test_data_):
    log_ids = test_data_["log_ids"]
    log = read_csv_log(test_data_["log_path"], log_ids)
    log_a = log.copy(deep=True)
    log_b = log.copy(deep=True)
    log_c = log.copy(deep=True)

    _add_enabled_times(log_a, log_ids)
    _add_enabled_times_parallel(log_b, log_ids)
    _add_enabled_times_parallel_polars(log_c, log_ids)

    assert log_a.equals(log_b)
    assert log_a.equals(log_c)


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "test_data_",
    test_data,
    ids=[item["log_path"].stem for item in test_data],
)
def test_enabled_times_discovery_benchmark(benchmark, test_data_):
    log_ids = test_data_["log_ids"]
    log = read_csv_log(test_data_["log_path"], log_ids)

    benchmark(_add_enabled_times, log, log_ids)

    assert log[log_ids.enabled_time].isna().sum() == 0


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "test_data_",
    test_data,
    ids=[item["log_path"].stem for item in test_data],
)
def test_enabled_times_discovery_parallel_benchmark(benchmark, test_data_):
    log_ids = test_data_["log_ids"]
    log = read_csv_log(test_data_["log_path"], log_ids)

    benchmark(_add_enabled_times_parallel, log, log_ids)

    assert log[log_ids.enabled_time].isna().sum() == 0


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "test_data_",
    test_data,
    ids=[item["log_path"].stem for item in test_data],
)
def test_enabled_times_discovery_parallel_optimized_benchmark(benchmark, test_data_):
    log_ids = test_data_["log_ids"]
    log = read_csv_log(test_data_["log_path"], log_ids)

    benchmark(_add_enabled_times_parallel_polars, log, log_ids)

    assert log[log_ids.enabled_time].isna().sum() == 0
