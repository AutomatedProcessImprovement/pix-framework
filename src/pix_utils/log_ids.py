from dataclasses import dataclass


@dataclass
class EventLogIDs:
    # General
    case: str = 'case'  # Case ID
    activity: str = 'activity'  # Activity label
    resource: str = 'resource'  # Resource who performed this activity instance
    start_time: str = 'start_timestamp'  # Start time of the activity instance
    end_time: str = 'end_timestamp'  # End time of the activity instance
    # Start time estimator
    enabled_time: str = 'enabled_time'  # Enablement time of the activity instance
    enabling_activity: str = 'enabling¨_activity'  # Label of the activity instance enabling the current one
    available_time: str = 'available_time'  # Last availability time of the resource who performed this activity instance
    estimated_start_time: str = 'estimated_start_time'  # Estimated start time of the activity instance


DEFAULT_CSV_IDS = EventLogIDs(case='case_id',
                              activity='Activity',
                              enabled_time='enabled_time',
                              start_time='start_time',
                              end_time='end_time',
                              available_time='available_time',
                              estimated_start_time='estimated_start_time',
                              resource='Resource')

DEFAULT_XES_IDS = EventLogIDs(case='case:concept:name',
                              activity='concept:name',
                              enabled_time='time:enabled',
                              start_time='time:start',
                              end_time='time:timestamp',
                              available_time='time:available',
                              estimated_start_time='time:estimated_start',
                              resource='org:resource')
