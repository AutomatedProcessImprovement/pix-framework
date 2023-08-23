# Case Attribute Discovery

Python package to discover case attributes from an event log and their value distribution (stochastic if discrete, probability distribution
if continuous).

## Example of use

```python
import pandas as pd

from pix_framework.discovery.case_attribute.discovery import discover_case_attributes
from pix_framework.io.event_log import DEFAULT_CSV_IDS

# Read event log
event_log = pd.read_csv("path_to_event_log.csv")

# Simple call
case_attributes = discover_case_attributes(
    event_log=event_log,
    log_ids=DEFAULT_CSV_IDS
)

# Call specifying the columns to not take into account for case attribute analysis
case_attributes = discover_case_attributes(
    event_log=event_log,
    log_ids=DEFAULT_CSV_IDS,
    avoid_columns=[
        DEFAULT_CSV_IDS.case, DEFAULT_CSV_IDS.activity,
        DEFAULT_CSV_IDS.start_time, DEFAULT_CSV_IDS.end_time
    ]
)

# Call specifying a confidence (or noise) threshold to allow a certain noise 
# in the variability of the attribute along the trace: 
#  - For each trace, the confidence of the most frequent value is computed (i.e. 
#  the % of activity instances from that trace with that same value). For example, 
#  a trace with 8 activity instances with 'amount'=100 and 2 with 'amount'=150 
#  will have a confidence of 0.8.
#  - The average confidence in all traces must be higher or equal to the specified
#  one to consider that column a case attribute.

case_attributes = discover_case_attributes(
    event_log=event_log,
    log_ids=DEFAULT_CSV_IDS,
    confidence_threshold=0.9
)
```

To see a more detailed example of use, and the format of the output, you can check this
[test file](https://github.com/AutomatedProcessImprovement/case-attribute-discovery/blob/964a5840325b1f8e8436c7004d5ab09b78b335d2/tests/discovery_test.py#L40).
