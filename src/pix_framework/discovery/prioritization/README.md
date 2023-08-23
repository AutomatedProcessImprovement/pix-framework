# Prioritization Discovery

Python algorithm to discover, from an event log, the case priority levels and the rules that classify a process case in its corresponding
level. For example, the cases of a process can belong to three priority levels (low, medium, high), where the activity instances of cases
with high priority are executed before than activity instances of cases with low priority (when both of them are enabled at the same time).

## Example of use

```python
import pandas as pd

from pix_framework.discovery.prioritization.discovery import discover_priority_rules
from pix_framework.io.event_log import DEFAULT_CSV_IDS

# Read event log
event_log = pd.read_csv("path_to_event_log.csv")
event_log[DEFAULT_CSV_IDS.enabled_time] = pd.to_datetime(event_log[DEFAULT_CSV_IDS.enabled_time], utc=True)
event_log[DEFAULT_CSV_IDS.start_time] = pd.to_datetime(event_log[DEFAULT_CSV_IDS.start_time], utc=True)
event_log[DEFAULT_CSV_IDS.end_time] = pd.to_datetime(event_log[DEFAULT_CSV_IDS.end_time], utc=True)
# Get priority levels and their rules
case_attributes = discover_priority_rules(
    event_log=event_log,
    attributes=['loan_amount', 'client_type']  # Case attributes to consider in the rule discovery
)
```

To see a more detailed example of use, and the format of the output, you can check this
[test file](https://github.com/AutomatedProcessImprovement/prioritization-discovery/blob/45e1aa561a84d8ab16b02469683aa0183f1ac8ca/tests/discovery_test.py#L149).

## No enabled time available

To identify which activity instances have been prioritized over others, the information of the enabled time has to be available in the event
log. In case it is not available, consider using [this Python library](https://github.com/AutomatedProcessImprovement/start-time-estimator)
to estimate them.
