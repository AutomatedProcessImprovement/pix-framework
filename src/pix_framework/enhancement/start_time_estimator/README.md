# Start Time Estimator

Python implementation of the start time estimation technique presented in the paper "Repairing Activity Start Times to Improve Business
Process Simulation", by David Chapela-Campa and Marlon Dumas.

The technique takes as input an event log (pd.DataFrame) recording the execution of the activities of a process (including resource
information), and produces a version of that event log with estimated start times for each activity instance.

## Requirements

- **Python v3.9.5+**
- **PIP v21.1.2+**
- Python dependencies: The packages listed in `requirements.txt`.

## Basic Usage

Check [main file](https://github.com/AutomatedProcessImprovement/start-time-estimator/blob/main/processing/main.py) for an example of a
simple execution,
and [config file](https://github.com/AutomatedProcessImprovement/start-time-estimator/blob/main/src/start_time_estimator/config.py) for an
explanation of the configuration parameters.

### Examples

Here we provide a simple example of use with the default configuration, followed by different custom configurations to run all the versions
of the technique.

```python
from pix_framework.enhancement.start_time_estimator.config import Configuration
from pix_framework.enhancement.start_time_estimator.estimator import StartTimeEstimator
from pix_framework.io.event_log import read_csv_log

# Set up default configuration
configuration = Configuration()
# Read event log
event_log = read_csv_log(
    log_path="path/to/event/log.csv.gz",
    log_ids=configuration.log_ids,
    sort=True  # Sort log by end time (warning this might alter the order of the events sharing end time)
)
# Estimate start times
extended_event_log = StartTimeEstimator(event_log, configuration).estimate()
```

The column IDs for the CSV file can be customized so the implementation works correctly with them:

```python
from pix_framework.enhancement.start_time_estimator.config import Configuration
from pix_framework.io.event_log import EventLogIDs

# Set up custom configuration
configuration = Configuration(
    log_ids=EventLogIDs(
        case="case",
        activity="task",
        start_time="start",
        end_time="end",
        resource="resource"
    )
)
```

#### Configuration of the proposed approach

With no outlier threshold and using the Median as the statistic to re-estimate the activity instances that couldn't be estimated:

```python
from pix_framework.enhancement.start_time_estimator.config import ConcurrencyOracleType, ReEstimationMethod, ResourceAvailabilityType
from pix_framework.enhancement.start_time_estimator.config import Configuration

# Set up custom configuration
configuration = Configuration(
    concurrency_oracle_type=ConcurrencyOracleType.HEURISTICS,
    re_estimation_method=ReEstimationMethod.MEDIAN,
    resource_availability_type=ResourceAvailabilityType.SIMPLE
)
```

With no outlier threshold and using the Mode as the statistic to re-estimate the activity instances that couldn't be estimated:

```python
from pix_framework.enhancement.start_time_estimator.config import ConcurrencyOracleType, ReEstimationMethod, ResourceAvailabilityType
from pix_framework.enhancement.start_time_estimator.config import Configuration

# Set up custom configuration
configuration = Configuration(
    concurrency_oracle_type=ConcurrencyOracleType.HEURISTICS,
    re_estimation_method=ReEstimationMethod.MODE,
    resource_availability_type=ResourceAvailabilityType.SIMPLE
)
```

Customize the thresholds for the concurrency detection:

```python
from pix_framework.enhancement.start_time_estimator.config import ConcurrencyOracleType, ReEstimationMethod
from pix_framework.enhancement.start_time_estimator.config import ResourceAvailabilityType, ConcurrencyThresholds
from pix_framework.enhancement.start_time_estimator.config import Configuration

# Set up custom configuration
configuration = Configuration(
    concurrency_oracle_type=ConcurrencyOracleType.HEURISTICS,
    concurrency_thresholds=ConcurrencyThresholds(df=0.6, l2l=0.6),
    re_estimation_method=ReEstimationMethod.MODE,
    resource_availability_type=ResourceAvailabilityType.SIMPLE
)
```

Add an outlier threshold of 200% and set the Mode to calculate the most typical duration too:

```python
from pix_framework.enhancement.start_time_estimator.config import ConcurrencyOracleType, ReEstimationMethod
from pix_framework.enhancement.start_time_estimator.config import ResourceAvailabilityType, OutlierStatistic
from pix_framework.enhancement.start_time_estimator.config import Configuration

# Set up custom configuration
configuration = Configuration(
    concurrency_oracle_type=ConcurrencyOracleType.HEURISTICS,
    re_estimation_method=ReEstimationMethod.MODE,
    resource_availability_type=ResourceAvailabilityType.SIMPLE,
    outlier_statistic=OutlierStatistic.MODE,
    outlier_threshold=2.0
)
```

Specify *bot resources* (perform the activities instantly) and *instant activities*:

```python
from pix_framework.enhancement.start_time_estimator.config import ConcurrencyOracleType, ReEstimationMethod
from pix_framework.enhancement.start_time_estimator.config import ResourceAvailabilityType
from pix_framework.enhancement.start_time_estimator.config import Configuration

# Set up custom configuration
configuration = Configuration(
    concurrency_oracle_type=ConcurrencyOracleType.HEURISTICS,
    re_estimation_method=ReEstimationMethod.MODE,
    resource_availability_type=ResourceAvailabilityType.SIMPLE,
    bot_resources={"SYSTEM", "BOT_001"},
    instant_activities={"Automatic Validation", "Send Notification"}
)
```

#### Configuration with a simpler concurrency oracle (Alpha Miner's) for the Enablement Time calculation

```python
from pix_framework.enhancement.start_time_estimator.config import ConcurrencyOracleType, ReEstimationMethod
from pix_framework.enhancement.start_time_estimator.config import ResourceAvailabilityType
from pix_framework.enhancement.start_time_estimator.config import Configuration

# Set up custom configuration
configuration = Configuration(
    concurrency_oracle_type=ConcurrencyOracleType.ALPHA,
    re_estimation_method=ReEstimationMethod.MODE,
    resource_availability_type=ResourceAvailabilityType.SIMPLE
)
```

#### Configuration with no concurrency oracle for the Enablement Time calculation (i.e. assuming directly-follows relations)

```python
from pix_framework.enhancement.start_time_estimator.config import ConcurrencyOracleType, ReEstimationMethod
from pix_framework.enhancement.start_time_estimator.config import ResourceAvailabilityType
from pix_framework.enhancement.start_time_estimator.config import Configuration

# Set up custom configuration
configuration = Configuration(
    concurrency_oracle_type=ConcurrencyOracleType.DF,
    re_estimation_method=ReEstimationMethod.MODE,
    resource_availability_type=ResourceAvailabilityType.SIMPLE
)
```

#### Configuration only taking into account the Resource Availability Time

```python
from pix_framework.enhancement.start_time_estimator.config import ConcurrencyOracleType, ReEstimationMethod
from pix_framework.enhancement.start_time_estimator.config import ResourceAvailabilityType
from pix_framework.enhancement.start_time_estimator.config import Configuration

# Set up custom configuration
configuration = Configuration(
    concurrency_oracle_type=ConcurrencyOracleType.DEACTIVATED,
    re_estimation_method=ReEstimationMethod.MODE,
    resource_availability_type=ResourceAvailabilityType.SIMPLE
)
```

## Individual Enablement Time Calculation

This package can be used too to calculate the enablement time (and the enabling activity) of the activity instances of an event log, without
the need to calculate the resource availability and estimate the start times. A simple example can be found here:

```python
from pix_framework.enhancement.concurrency_oracle import HeuristicsConcurrencyOracle
from pix_framework.enhancement.start_time_estimator.config import Configuration
from pix_framework.io.event_log import DEFAULT_CSV_IDS, read_csv_log

# Set up default configuration
configuration = Configuration(
    log_ids=DEFAULT_CSV_IDS,  # Custom the column IDs with this parameter
    consider_start_times=True  # Consider real parallelism if the start times are available
)
# Read event log
event_log = read_csv_log(
    log_path="path/to/event/log.csv.gz",
    log_ids=configuration.log_ids,
    sort=True  # Sort log by end time (warning this might alter the order of the events sharing end time)
)
# Instantiate desired concurrency oracle
concurrency_oracle = HeuristicsConcurrencyOracle(event_log, configuration)
# concurrency_oracle = AlphaConcurrencyOracle(event_log, configuration)
# concurrency_oracle = DirectlyFollowsConcurrencyOracle(event_log, configuration)
# Add enablement times to the event log
concurrency_oracle.add_enabled_times(
    event_log,
    set_nat_to_first_event=False,  # Whether to set NaT or the start trace to the events with no enabling activities.
    include_enabling_activity=True  # Whether to include the label of the enabling activity in a new column or not.
)
```

**Warning:** If the event log contains start times, set the parameter *consider_start_times* to *true*. This parameter allows the enablement
time calculator to know that it can trust the start times of the event log to discard those activity instances that are being executed in
parallel to the current one as a possible causal predecessor.

For example: if activity *A* always preceedes activity *B*, i.e. there are no concurrency, an execution of *A* can be a causal predecessor
of an execution of *B* (meaning this that *A* can enable *B*). Nevertheless, if the start times are available and there is an activity
instance of *B* which starts before the end of *A*, *A* does not enable *B* in that case.

If *consider_start_times* is set to *true*, the estimator consider the start time information in this way, if it is set to *false*, only the
end times will be considered.
