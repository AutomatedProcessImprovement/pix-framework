from pathlib import Path

import pandas as pd
import pytest

from pix_framework.discovery.start_time_estimator.concurrency_oracle import OverlappingConcurrencyOracle
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
    OverlappingConcurrencyOracle(log, configuration).add_enabled_times(log)


test_data = [
    {
        "log_path": assets_dir / "test_event_log_3_noise.csv",
        "log_ids": DEFAULT_CSV_IDS,
    },
]


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
