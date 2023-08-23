from pathlib import Path

import pandas as pd
import pytest
from pix_framework.discovery.resource_calendar_and_performance.fuzzy.discovery import (
    discovery_fuzzy_resource_calendars_and_performances,
)
from pix_framework.enhancement.concurrency_oracle import OverlappingConcurrencyOracle
from pix_framework.enhancement.start_time_estimator.config import ConcurrencyThresholds
from pix_framework.enhancement.start_time_estimator.config import Configuration as StartTimeEstimatorConfiguration
from pix_framework.io.event_log import PROSIMOS_LOG_IDS, EventLogIDs, read_csv_log

assets_dir = Path(__file__).parent / "assets"

test_data = [
    {
        "log_path": assets_dir / "csv_logs/fuzzy_calendars_log.csv.gz",
        "error_threshold": 0.95,
    },
    {
        "log_path": assets_dir / "csv_logs/LoanApp_simplified.csv.gz",
        "error_threshold": 0.05,
    },
]


@pytest.mark.parametrize(
    "test_data",
    test_data,
    ids=[item["log_path"].name for item in test_data],
)
def test_fuzzy_calendar_discovery_from_df(test_data):
    """
    Checks if the fuzzy discovery technique doesn't discover any fuzziness in the classic log (LoanApp_simplified.csv),
    and if it discovers fuzziness in the fuzzy log (fuzzy_calendars_log.csv).
    """
    log_path = test_data["log_path"]
    log = read_csv_log(log_path, PROSIMOS_LOG_IDS)
    _add_enabled_times(log, PROSIMOS_LOG_IDS)

    # avoiding the applicant with low workload
    log = log[log[PROSIMOS_LOG_IDS.resource] != "Applicant-000001"]

    # discover fuzzy calendars
    result, _ = discovery_fuzzy_resource_calendars_and_performances(
        log=log,
        log_ids=PROSIMOS_LOG_IDS,
        angle=0.0,
    )

    # calculate error
    numerator = 0
    denominator = 0
    for calendar in result:
        for interval in calendar.intervals:
            if interval.probability < 0.8:
                numerator += 1
            denominator += 1
    error = numerator / denominator

    # check error
    assert error <= test_data["error_threshold"], f"Error: {error} > {test_data['error_threshold']}"


def _add_enabled_times(log: pd.DataFrame, log_ids: EventLogIDs):
    configuration = StartTimeEstimatorConfiguration(
        log_ids=log_ids,
        concurrency_thresholds=ConcurrencyThresholds(df=0.75),
        consider_start_times=True,
    )
    # The start times are the original ones, so use overlapping concurrency oracle
    OverlappingConcurrencyOracle(log, configuration).add_enabled_times(log)
