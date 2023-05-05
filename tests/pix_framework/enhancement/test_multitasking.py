from pathlib import Path

import pytest

from pix_framework.enhancement.multitasking import adjust_durations
from pix_framework.input import read_csv_log
from pix_framework.log_ids import DEFAULT_XES_IDS

assets_dir = Path(__file__).parent.parent / "assets"


@pytest.mark.integration
@pytest.mark.parametrize(
    "event_log, expected_values",
    [
        ("MultitaskingSynthetic.csv.gz", [330.0, 870.0]),
        ("MultitaskingSynthetic2.csv.gz", [600.0, 1140.0]),
        ("MultitaskingSynthetic3.csv.gz", [5.0, 2.5, 2.5]),
    ],
)
def test_adjust_durations_synthetic(event_log, expected_values):
    log_path = assets_dir / event_log
    log = read_csv_log(log_path, DEFAULT_XES_IDS)

    result = adjust_durations(log, DEFAULT_XES_IDS, verbose=True)

    assert result is not None
    for i in range(len(expected_values)):
        duration = result.iloc[i]["time:timestamp"] - result.iloc[i]["start_timestamp"]
        assert duration.total_seconds() == expected_values[i]


@pytest.mark.smoke
@pytest.mark.parametrize(
    "event_log",
    [
        "ConsultaDataMining201618.csv.gz",
    ],
)
def test_adjust_durations_real_smoke(event_log):
    log_path = assets_dir / event_log
    log = read_csv_log(log_path, DEFAULT_XES_IDS)
    result = adjust_durations(log, DEFAULT_XES_IDS, verbose=False)
    assert result is not None
