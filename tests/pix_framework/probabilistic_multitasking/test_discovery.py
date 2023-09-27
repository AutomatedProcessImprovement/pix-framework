from pathlib import Path
import pandas as pd

from pix_framework.discovery.probabilistic_multitasking.discovery import calculate_multitasking
from pix_framework.io.event_log import DEFAULT_CSV_IDS

assets_dir = Path(__file__).parent.parent / "assets/multitasking"


def test_discover_multitasking():
    # event_log = pd.read_csv(assets_dir / "Application_to_Approval_Government_Agency.csv")
    probabilities = calculate_multitasking(pd.read_csv(assets_dir / "Application_to_Approval_Government_Agency.csv"))

    valid_p = True
    for resource in probabilities:
        for p in probabilities[resource]:
            if p > 1.0:
                valid_p = False
                break
        if not valid_p:
            break

    assert valid_p


def _read_event_log(log_path: Path):
    event_log = pd.read_csv(log_path)
    if DEFAULT_CSV_IDS.enabled_time in event_log:
        event_log[DEFAULT_CSV_IDS.enabled_time] = pd.to_datetime(event_log[DEFAULT_CSV_IDS.enabled_time], utc=True)
    event_log[DEFAULT_CSV_IDS.start_time] = pd.to_datetime(event_log[DEFAULT_CSV_IDS.start_time], utc=True)
    event_log[DEFAULT_CSV_IDS.end_time] = pd.to_datetime(event_log[DEFAULT_CSV_IDS.end_time], utc=True)
    return event_log
