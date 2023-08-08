import os
from dataclasses import dataclass
from pathlib import Path


def get_project_dir() -> Path:
    return Path(os.path.dirname(__file__)).parent.parent


@dataclass
class EventLogIDs:
    """
    Batches nomenclature:
     - Batch: a group of activities that are executed in a batch (e.g. A, B and C), independently of the batch type, they can even be
     executed following a type in some cases and following another type in others.
     - Batch instance: an instance of a batch grouping more than one execution of the batch (i.e. A, B and C executed in a batch in three
     traces).
     - Batch case: the batch activity instances of only one trace (e.g. A, B and C executed in a batch in trace 01).

     A 'batch' contains many 'batch instances' in an event log (all the executions of the batch activities following a batch behavior), and
     each batch instance contains at least two 'batch cases' (it is executed at least in two traces).
    """
    case: str = 'case_id'  # ID of the case instance of the process (trace)
    activity: str = 'Activity'  # Name of the executed activity in this activity instance
    start_time: str = 'start_time'  # Timestamp in which this activity instance started
    end_time: str = 'end_time'  # Timestamp in which this activity instance ended
    resource: str = 'Resource'  # ID of the resource that executed this activity instance
    enabled_time: str = 'enabled_time'  # Enable time of this activity instance
    batch_id: str = 'batch_instance_id'  # ID of the batch instance this activity instance belongs to, if any
    batch_type: str = 'batch_instance_type'  # Type of the batch instance this activity instance belongs to, if any


DEFAULT_CSV_IDS = EventLogIDs(case='case_id',
                              activity='Activity',
                              start_time='start_time',
                              end_time='end_time',
                              resource='Resource',
                              enabled_time='enabled_time',
                              batch_id='batch_instance_id',
                              batch_type='batch_instance_type')


@dataclass
class BatchType:
    parallel: str = "Parallel"
    sequential: str = "Sequential"
    concurrent: str = "Concurrent"
