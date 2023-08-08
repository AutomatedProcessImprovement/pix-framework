from dataclasses import dataclass


@dataclass
class EventLogIDs:
    case: str = "case_id"  # ID of the case instance of the process (trace)
    activity: str = "Activity"  # Name of the executed activity in this activity instance
    start_time: str = "start_time"  # Timestamp in which this activity instance started
    end_time: str = "end_time"  # Timestamp in which this activity instance ended
    resource: str = "Resource"  # ID of the resource that executed this activity instance
    enabled_time: str = "enabled_time"  # Enable time of this activity instance


DEFAULT_CSV_IDS = EventLogIDs(
    case="case_id",
    activity="Activity",
    start_time="start_time",
    end_time="end_time",
    resource="Resource",
    enabled_time="enabled_time",
)

# Prefixes for internal use in prioritization discovery
PRIORITIZED_PREFIX = "prioritized"
DELAYED_PREFIX = "delayed"
