from dataclasses import dataclass


@dataclass
class EventLogIDs:
    case: str = "case_id"
    activity: str = "Activity"
    start_time: str = "start_time"
    end_time: str = "end_time"
    resource: str = "Resource"


DEFAULT_CSV_IDS = EventLogIDs(
    case="case_id", activity="Activity", start_time="start_time", end_time="end_time", resource="Resource"
)
