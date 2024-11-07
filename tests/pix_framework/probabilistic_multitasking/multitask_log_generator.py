import random
import csv
from datetime import datetime, timedelta
from pathlib import Path

assets_dir = Path(__file__).parent.parent / "assets/multitasking"


def is_within_working_hours(c_time: datetime):
    """ Check if the given time is within working hours. """
    morning_start = c_time.replace(hour=8, minute=0, second=0, microsecond=0)
    morning_end = c_time.replace(hour=12, minute=0, second=0, microsecond=0)
    afternoon_start = c_time.replace(hour=13, minute=0, second=0, microsecond=0)
    afternoon_end = c_time.replace(hour=17, minute=0, second=0, microsecond=0)

    return morning_start <= c_time < morning_end or afternoon_start <= c_time < afternoon_end


def next_working_time(c_time):
    """ Return the next start time of a working period. """
    if c_time.weekday() >= 5:  # Saturday or Sunday
        # Move to next Monday
        c_time += timedelta(days=(7 - c_time.weekday()))
        return c_time.replace(hour=8, minute=0, second=0, microsecond=0)

    # During weekday
    morning_end = c_time.replace(hour=12, minute=0, second=0, microsecond=0)
    afternoon_end = c_time.replace(hour=17, minute=0, second=0, microsecond=0)

    if c_time < morning_end:
        return c_time.replace(hour=13, minute=0, second=0, microsecond=0)
    elif c_time < afternoon_end:
        # Move to next day
        return c_time.replace(hour=8, minute=0, second=0, microsecond=0) + timedelta(days=1)
    else:
        # Move to next day
        return c_time.replace(hour=8, minute=0, second=0, microsecond=0) + timedelta(
            days=(1 if c_time.weekday() < 4 else 3))


def generate_sequence_process_log(r_count: int,
                                  t_count: int,
                                  t_duration: float,
                                  p_cases_count: int,
                                  inter_arrival_time: float,
                                  out_log_file_path: Path):
    # Initialize the event log
    event_log = []

    # Create a list of tasks
    tasks = ["Task_" + str(i) for i in range(1, t_count + 1)]

    # Starting time for the first task (assuming starting on a Monday at 8:00)
    c_case_start_time = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0, day=1)
    if c_case_start_time.weekday() != 0:
        c_case_start_time += timedelta(days=(7 - c_case_start_time.weekday()))

    # Assign tasks to resources and process instances sequentially
    p_case = 1
    while p_case <= p_cases_count:
        start_time = c_case_start_time
        for task_id in tasks:
            resource_id = "Resource_" + str(random.randint(1, r_count))  # Assign a random resource

            # If current time is outside working hours, move to next working time (not necessary - just double check)
            if not is_within_working_hours(start_time):
                start_time = next_working_time(start_time)

            # Calculate end time for the task
            end_time = start_time + timedelta(minutes=t_duration)
            # If end time is outside working hours, adjust end time and start time of next task
            if not is_within_working_hours(end_time) or end_time.day != start_time.day:
                start_time = next_working_time(start_time)
                end_time = start_time + timedelta(minutes=t_duration)

            # Log entry for the task
            event_log.append({
                'case_id': p_case,
                'activity': task_id,
                'resource': resource_id,
                'start_time': start_time.strftime("%Y-%m-%d %H:%M:%S"),
                'end_time': end_time.strftime("%Y-%m-%d %H:%M:%S")
            })
            start_time = end_time

        # Update start time for the next process instance
        c_case_start_time = c_case_start_time + timedelta(minutes=inter_arrival_time)
        if not is_within_working_hours(c_case_start_time):
            c_case_start_time = next_working_time(c_case_start_time)
        p_case += 1

    # Write to CSV file
    with open(out_log_file_path, 'w', newline='') as file:
        f_writer = csv.DictWriter(file, fieldnames=['case_id', 'activity', 'resource', 'start_time', 'end_time'])
        f_writer.writeheader()
        for t_case in event_log:
            f_writer.writerow(t_case)


def parse_datetime(datetime_str):
    return datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")


def calculate_parallel_tasks(csv_file_path: Path):
    tasks_by_resource = {}

    # Read and parse the event log
    with open(csv_file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            resource = row['resource']
            start_time = parse_datetime(row['start_time'])
            end_time = parse_datetime(row['end_time'])

            if resource not in tasks_by_resource:
                tasks_by_resource[resource] = []

            tasks_by_resource[resource].append((start_time, end_time))

    # Calculate max and min parallel tasks for each resource
    parallel_tasks_stats = {}
    for resource, tasks in tasks_by_resource.items():
        # Sort tasks by start time
        tasks.sort(key=lambda x: x[0])

        # List to track parallel tasks
        parallel_tasks = []

        # Track max and min values
        max_parallel = 0
        min_parallel = float('inf')

        for task in tasks:
            start, end = task
            # Remove tasks that ended before the current task's start time
            parallel_tasks = [t for t in parallel_tasks if t[1] > start]

            # Add the current task
            parallel_tasks.append(task)

            # Update max and min counts
            max_parallel = max(max_parallel, len(parallel_tasks))
            min_parallel = min(min_parallel, len(parallel_tasks))

        parallel_tasks_stats[resource] = {'max': max_parallel, 'min': min_parallel}

    return parallel_tasks_stats


def main():
    generate_sequence_process_log(5, 7, 240, 1000, 30, assets_dir / "sequential.csv")
    parallel_tasks_stats = calculate_parallel_tasks(assets_dir / "sequential.csv")
    print(parallel_tasks_stats)


if __name__ == "__main__":
    main()
