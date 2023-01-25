# -------------------- Calendar class and utils ------------------- #
# The main structures have been copied and simplified from Prosimos #
# project (https://github.com/AutomatedProcessImprovement/Prosimos/blob/main/bpdfr_simulation_engine/resource_calendar.py).
# ----------------------------------------------------------------- #

import pandas as pd

str_week_days = {"MONDAY": 0, "TUESDAY": 1, "WEDNESDAY": 2, "THURSDAY": 3, "FRIDAY": 4, "SATURDAY": 5, "SUNDAY": 6}
int_week_days = {0: "MONDAY", 1: "TUESDAY", 2: "WEDNESDAY", 3: "THURSDAY", 4: "FRIDAY", 5: "SATURDAY", 6: "SUNDAY"}

conversion_table = {
    'WEEKS': 604800,
    'DAYS': 86400,
    'HOURS': 3600,
    'MINUTES': 60,
    'SECONDS': 1
}


class RCalendar:
    def __init__(self, calendar_id: str):
        self.calendar_id = calendar_id
        self.work_intervals = {i: list() for i in range(0, 7)}

    def to_json(self) -> list:
        # Create empty list
        items = []
        # Insert calendar for each week day
        for i in range(0, 7):
            if len(self.work_intervals[i]) > 0:
                for interval in self.work_intervals[i]:
                    items.append({
                        'from': int_week_days[i],
                        'to': int_week_days[i],
                        "beginTime": str(interval.start.time()),
                        "endTime": str(interval.end.time())
                    })
        # Return list with working schedule
        return items


class Interval:
    def __init__(self, start: pd.Timestamp, end: pd.Timestamp):
        self.start = start
        if end < start and end.hour == 0 and end.minute == 0:
            end.replace(hour=23, minute=59, second=59, microsecond=999)
        self.end = end
        self.duration = (end - start).total_seconds()


def get_last_available_timestamp(start: pd.Timestamp, end: pd.Timestamp, schedule: RCalendar) -> pd.Timestamp:
    """
    Get the timestamp [last_available] within the interval from [start] to [end] (i.e. [start] <= [last_available] <= [end]) such that
    the interval from [last_available] to [end] is the largest and all of it is in the working hours in the calendar [schedule].

    For example, for [start] = 09:30, [end] = 14:00, and a [schedule] of every week day from 06:00 to 09:00, and from 10:00 to 16:00. The
    [last_available] would be 10:00.

    :param start:       start of the interval where to search for the point since when all time is part of the working schedule.
    :param end:         end of the interval where to search for the point since when all time is part of the working schedule.
    :param schedule:    RCalendar with the weekly working schedule.

    :return: The earliest point within the interval from [start] to [end] since which all the time is part of working hours defined in the
             resource calendar [schedule].
    """
    # Get the latest working period previous to the end of the interval
    last_available = end
    found = False
    while not found:
        day_intervals = schedule.work_intervals[last_available.weekday()]
        for interval in reversed(day_intervals):
            # Move interval to current day
            interval_start = interval.start.replace(day=last_available.day, month=last_available.month, year=last_available.year)
            interval_end = interval.end.replace(day=last_available.day, month=last_available.month, year=last_available.year)
            if interval_end < last_available:
                # The last available is later than the end of the current working interval
                if (last_available - interval_end) > pd.Timedelta(seconds=2):
                    # Non-working time gap previous to last_available, search finished
                    found = True
                    # Correct jump to previous day if needed
                    if (
                            last_available.hour == 23 and
                            last_available.minute == 59 and
                            last_available.second == 59 and
                            last_available.microsecond == 999999
                    ):
                        last_available = last_available + pd.Timedelta(microseconds=1)
                else:
                    # No non-working time gap, move to the start of this working interval and continue
                    last_available = last_available.replace(
                        hour=interval_start.hour,
                        minute=interval_start.minute,
                        second=interval_start.second,
                        microsecond=interval_start.microsecond
                    )
            elif interval_start <= last_available <= interval_end:
                # The last available timestamp is within the current interval
                last_available = last_available.replace(
                    hour=interval_start.hour,
                    minute=interval_start.minute,
                    second=interval_start.second,
                    microsecond=interval_start.microsecond
                )
        if not found:
            start_of_day = last_available.replace(hour=00, minute=00, second=00, microsecond=0)
            if (last_available - start_of_day) > pd.Timedelta(seconds=2):
                # Non-working interval between last_available and the start of the day
                found = True
            else:
                # Move to previous day at 23:59:59.999999
                last_available = (last_available - pd.Timedelta(days=1)).replace(hour=23, minute=59, second=59, microsecond=999999)
        # If last_available moved previously to the start of the queried interval
        if last_available <= start:
            # Stop and set to the start of the queried interval
            found = True
            last_available = start
    # Return last available timestamp
    return last_available
