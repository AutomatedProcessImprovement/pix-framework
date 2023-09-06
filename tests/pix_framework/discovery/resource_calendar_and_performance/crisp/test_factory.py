from pathlib import Path

import pandas as pd
import pytest
from pix_framework.discovery.resource_calendar_and_performance.crisp.factory import CalendarFactory

assets_dir = Path(__file__).parent.parent.parent.parent / "assets"


@pytest.mark.integration
def test_calendar_module():
    log_path = assets_dir / "PurchasingExample.csv"

    df = pd.read_csv(log_path)
    df["start_timestamp"] = pd.to_datetime(df["start_timestamp"], utc=True)
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], utc=True)
    calendar_factory = CalendarFactory(15)

    for _, row in df.iterrows():
        resource = row["org:resource"]
        activity = row["concept:name"]
        start_timestamp = row["start_timestamp"].to_pydatetime()
        end_timestamp = row["time:timestamp"].to_pydatetime()
        calendar_factory.check_date_time(resource, activity, start_timestamp)
        calendar_factory.check_date_time(resource, activity, end_timestamp)

    calendar_candidates = calendar_factory.build_weekly_calendars(0.1, 0.7, 0.4)

    calendar = {}
    for resource_id in calendar_candidates:
        if calendar_candidates[resource_id] is not None:
            calendar[resource_id] = calendar_candidates[resource_id].intervals_to_json()

    assert len(calendar) > 0
    assert "Kim Passa" in calendar
