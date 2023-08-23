# Batch Processing Discovery

This technique takes as input an event log (pd.DataFrame) recording the execution of the activities of a process with enabled, start and end
timestamps, as well as the resource who performed it, and discovers which activity instances have been executed in a batch, and the
characteristics of this batch processing.

The discovered characteristics are, for each batch processing:

- The **activity** being executed.
- The **resources** involved in this batch processing.
- The **type of batch** processing (sequential, concurrent or parallel). In case of more than one type, the most common.
- The **frequency** of that activity occurring as part of a batch.
- The **distribution of batch sizes**, i.e., for each size, the number of activity instances executed as a batch with that size.
- The **distribution of durations**, i.e., for each batch size, the scaling factor of the duration of the activity instances processed in
  that batch. For example, if the activity is processed in a 2-size batch, each activity instance lasts x0.7 what it lasts executed
  individually.
- The **firing rules** that better describe the start of the batch.

## Requirements

- **Python v3.9.5+**
- **PIP v21.1.2+**
- Python dependencies: The packages listed in `requirements.txt`.

## Basic Usage

Here we provide a simple example of use with default configuration (see
[function documentation](https://github.com/AutomatedProcessImprovement/batch-processing-discovery/blob/main/src/batch_processing_discovery/batch_characteristics.py)
for more parameters):

```python
import pandas as pd

from pix_framework.discovery.batch_processing.batch_characteristics import discover_batch_processing_and_characteristics
from pix_framework.io.event_log import DEFAULT_CSV_IDS

# Read event log
event_log = pd.read_csv("path/to/event/log.csv.gz")
# Discover batch processing activities and their characteristics
batch_characteristics = discover_batch_processing_and_characteristics(
    event_log=event_log,
    log_ids=DEFAULT_CSV_IDS
)
```

### Discover only batch processing behavior

In case of being interested only in discovering batch processing behavior, the following example applies (see
[function documentation](https://github.com/AutomatedProcessImprovement/batch-processing-discovery/blob/main/src/batch_processing_discovery/discovery.py)
for more parameters):

```python
import pandas as pd

from pix_framework.discovery.batch_processing.discovery import discover_batches
from pix_framework.io.event_log import DEFAULT_CSV_IDS

# Read event log
event_log = pd.read_csv("path/to/event/log.csv.gz")
# Discover batch processing activities and their characteristics
batched_event_log = discover_batches(
    event_log=event_log,
    log_ids=DEFAULT_CSV_IDS
)
```

### Get batch characteristics with already set batch processing behavior

In case of being interested only in getting the batch characteristics, based on an event log with already set batch behavior, the following
example applies (see
[function documentation](https://github.com/AutomatedProcessImprovement/batch-processing-discovery/blob/main/src/batch_processing_discovery/batch_characteristics.py)
for more parameters):

```python
import pandas as pd

from pix_framework.discovery.batch_processing.batch_characteristics import discover_batch_characteristics
from pix_framework.io.event_log import DEFAULT_CSV_IDS

# Read event log
event_log = pd.read_csv("path/to/event/log_with_batch_info.csv.gz")
# Discover batch processing activities and their characteristics
batch_characteristics = discover_batch_characteristics(
    event_log=event_log,
    log_ids=DEFAULT_CSV_IDS
)
```

## ** No enabled time available

In case of not enabled time available in the event log, consider
using [this Python library](https://github.com/AutomatedProcessImprovement/start-time-estimator) to estimate them.