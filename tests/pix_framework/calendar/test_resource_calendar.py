from pathlib import Path

import pandas as pd
import pytz
from pix_framework.calendar.availability import (
    absolute_unavailability_intervals_within,
    get_last_available_timestamp,
)
from pix_framework.calendar.crisp_resource_calendar import Interval, RCalendar

_def_tz = pytz.timezone("UTC")


def test_calendar_add_calendar_item_non_overlapping():
    # Create empty calendar
    working_calendar = RCalendar("test")
    # Add non-overlapping intervals through method
    working_calendar.add_calendar_item("MONDAY", "MONDAY", "10:00:00", "14:00:00")
    working_calendar.add_calendar_item("WEDNESDAY", "WEDNESDAY", "10:00:00", "14:00:00")
    working_calendar.add_calendar_item("WEDNESDAY", "WEDNESDAY", "16:00:00", "20:00:00")
    working_calendar.add_calendar_item("FRIDAY", "FRIDAY", "16:00:00", "20:00:00")
    # Assert it
    morning_shift = Interval(pd.Timestamp("10:00:00"), pd.Timestamp("14:00:00"))
    evening_shift = Interval(pd.Timestamp("16:00:00"), pd.Timestamp("20:00:00"))
    assert working_calendar.work_intervals[0] == [morning_shift]
    assert working_calendar.work_intervals[2] == [morning_shift, evening_shift]
    assert working_calendar.work_intervals[4] == [evening_shift]


def test_calendar_add_calendar_item_overlapping():
    # Reset calendar
    working_calendar = RCalendar("test")
    # Add overlapping intervals through method
    working_calendar.add_calendar_item("WEDNESDAY", "WEDNESDAY", "10:00:00", "14:00:00")
    working_calendar.add_calendar_item("WEDNESDAY", "WEDNESDAY", "14:00:00", "20:00:00")
    # Assert it
    shift = Interval(pd.Timestamp("10:00:00"), pd.Timestamp("20:00:00"))
    assert working_calendar.work_intervals[2] == [shift]


def test_calendar_add_calendar_item_multiple_days():
    # Reset calendar
    working_calendar = RCalendar("test")
    # Add overlapping intervals through method
    working_calendar.add_calendar_item("TUESDAY", "THURSDAY", "10:00:00", "14:00:00")
    # Assert it
    shift = Interval(pd.Timestamp("10:00:00"), pd.Timestamp("14:00:00"))
    assert working_calendar.work_intervals[1] == [shift]
    assert working_calendar.work_intervals[2] == [shift]
    assert working_calendar.work_intervals[3] == [shift]


def test_get_last_available_timestamp_24_7():
    # Create working calendar with 24/7
    working_calendar = RCalendar("test")
    working_calendar.work_intervals[0] = [
        Interval(pd.Timestamp("2023-01-25T00:00:00"), pd.Timestamp("2023-01-25T23:59:59"))
    ]
    working_calendar.work_intervals[1] = [
        Interval(pd.Timestamp("2023-01-25T00:00:00"), pd.Timestamp("2023-01-25T23:59:59"))
    ]
    working_calendar.work_intervals[2] = [
        Interval(pd.Timestamp("2023-01-25T00:00:00"), pd.Timestamp("2023-01-25T23:59:59"))
    ]
    working_calendar.work_intervals[3] = [
        Interval(pd.Timestamp("2023-01-25T00:00:00"), pd.Timestamp("2023-01-25T23:59:59"))
    ]
    working_calendar.work_intervals[4] = [
        Interval(pd.Timestamp("2023-01-25T00:00:00"), pd.Timestamp("2023-01-25T23:59:59"))
    ]
    working_calendar.work_intervals[5] = [
        Interval(pd.Timestamp("2023-01-25T00:00:00"), pd.Timestamp("2023-01-25T23:59:59"))
    ]
    working_calendar.work_intervals[6] = [
        Interval(pd.Timestamp("2023-01-25T00:00:00"), pd.Timestamp("2023-01-25T23:59:59"))
    ]
    # Assert that last available timestamp of any interval is its start
    start = pd.Timestamp("2020-01-20T14:15:17+00:00")
    end = pd.Timestamp("2020-01-20T19:21:10+00:00")
    assert get_last_available_timestamp(start, end, working_calendar) == start
    start = pd.Timestamp("2020-01-20T14:15:17+00:00")
    end = pd.Timestamp("2020-01-25T11:47:13+00:00")
    assert get_last_available_timestamp(start, end, working_calendar) == start
    start = pd.Timestamp("2020-01-20T14:15:17+00:00")
    end = pd.Timestamp("2020-01-25T23:59:59+00:00")
    assert get_last_available_timestamp(start, end, working_calendar) == start
    start = pd.Timestamp("2020-01-20T14:15:17+00:00")
    end = pd.Timestamp("2020-01-25T00:00:00+00:00")
    assert get_last_available_timestamp(start, end, working_calendar) == start
    start = pd.Timestamp("2020-01-20T14:15:17+00:00")
    assert get_last_available_timestamp(start, start, working_calendar) == start


def test_get_last_available_timestamp_all_interval_within():
    # Create working calendar with one single interval in the current day
    working_calendar = RCalendar("test")
    working_calendar.work_intervals[1] = [
        Interval(pd.Timestamp("2023-01-25T10:00:00"), pd.Timestamp("2023-01-25T18:00:00"))
    ]
    # Assert that last available timestamp of any interval within the working hours is the start of the interval
    start = pd.Timestamp("2020-01-21T14:15:17+00:00")
    end = pd.Timestamp("2020-01-21T15:21:10+00:00")
    assert get_last_available_timestamp(start, end, working_calendar) == start
    start = pd.Timestamp("2020-01-21T16:11:08+00:00")
    end = pd.Timestamp("2020-01-21T18:00:00+00:00")
    assert get_last_available_timestamp(start, end, working_calendar) == start
    start = pd.Timestamp("2020-01-21T11:11:08+00:00")
    end = pd.Timestamp("2020-01-21T11:11:08+00:00")
    assert get_last_available_timestamp(start, end, working_calendar) == start


def test_get_last_available_timestamp_within_non_24_7():
    # Create working calendar with one single interval in the current day
    working_calendar = RCalendar("test")
    working_calendar.work_intervals[1] = [
        Interval(pd.Timestamp("2023-01-25T10:00:00"), pd.Timestamp("2023-01-25T18:00:00"))
    ]
    # Assert that last available timestamp of any interval with the end within the working hours is the start of the working hours
    start = pd.Timestamp("2020-01-21T04:15:17+00:00")
    end = pd.Timestamp("2020-01-21T14:21:10+00:00")
    assert get_last_available_timestamp(start, end, working_calendar) == pd.Timestamp("2020-01-21T10:00:00+00:00")
    start = pd.Timestamp("2020-01-21T06:11:08+00:00")
    end = pd.Timestamp("2020-01-21T18:00:00+00:00")
    assert get_last_available_timestamp(start, end, working_calendar) == pd.Timestamp("2020-01-21T10:00:00+00:00")
    start = pd.Timestamp("2020-01-21T06:11:08+00:00")
    end = pd.Timestamp("2020-01-21T10:00:00+00:00")
    assert get_last_available_timestamp(start, end, working_calendar) == pd.Timestamp("2020-01-21T10:00:00+00:00")

    # Create working calendar with two intervals in the current day
    working_calendar = RCalendar("test")
    working_calendar.work_intervals[1] = [
        Interval(pd.Timestamp("2023-01-25T08:00:00"), pd.Timestamp("2023-01-25T14:00:00")),
        Interval(pd.Timestamp("2023-01-25T16:00:00"), pd.Timestamp("2023-01-25T20:00:00")),
    ]
    # Assert that last available timestamp of any interval with the end within the first working hour interval is the start
    start = pd.Timestamp("2020-01-21T04:15:17+00:00")
    end = pd.Timestamp("2020-01-21T10:21:10+00:00")
    assert get_last_available_timestamp(start, end, working_calendar) == pd.Timestamp("2020-01-21T08:00:00+00:00")
    start = pd.Timestamp("2020-01-21T06:11:08+00:00")
    end = pd.Timestamp("2020-01-21T11:00:00+00:00")
    assert get_last_available_timestamp(start, end, working_calendar) == pd.Timestamp("2020-01-21T08:00:00+00:00")
    start = pd.Timestamp("2020-01-21T05:11:08+00:00")
    end = pd.Timestamp("2020-01-21T12:00:00+00:00")
    assert get_last_available_timestamp(start, end, working_calendar) == pd.Timestamp("2020-01-21T08:00:00+00:00")

    # Create working calendar with one single interval in the beginning of the current day and two in the previous one
    working_calendar = RCalendar("test")
    working_calendar.work_intervals[0] = [
        Interval(pd.Timestamp("2023-01-25T10:00:00"), pd.Timestamp("2023-01-25T14:00:00")),
        Interval(pd.Timestamp("2023-01-25T16:00:00"), pd.Timestamp("2023-01-25T20:00:00")),
    ]
    working_calendar.work_intervals[1] = [
        Interval(pd.Timestamp("2023-01-25T00:00:00"), pd.Timestamp("2023-01-25T08:00:00"))
    ]
    # Assert that last available timestamp of any interval with the end within the working hours is the start of the working hours
    start = pd.Timestamp("2020-01-20T14:15:17+00:00")
    end = pd.Timestamp("2020-01-21T06:21:10+00:00")
    assert get_last_available_timestamp(start, end, working_calendar) == pd.Timestamp("2020-01-21T00:00:00+00:00")
    start = pd.Timestamp("2020-01-20T00:00:00+00:00")
    end = pd.Timestamp("2020-01-21T04:12:04+00:00")
    assert get_last_available_timestamp(start, end, working_calendar) == pd.Timestamp("2020-01-21T00:00:00+00:00")
    start = pd.Timestamp("2020-01-20T00:00:00+00:00")
    end = pd.Timestamp("2020-01-21T08:00:00+00:00")
    assert get_last_available_timestamp(start, end, working_calendar) == pd.Timestamp("2020-01-21T00:00:00+00:00")


def test_get_last_available_timestamp_without():
    # Create working calendar with two intervals in the current day
    working_calendar = RCalendar("test")
    working_calendar.work_intervals[1] = [
        Interval(pd.Timestamp("2023-01-25T10:00:00"), pd.Timestamp("2023-01-25T14:00:00")),
        Interval(pd.Timestamp("2023-01-25T16:00:00"), pd.Timestamp("2023-01-25T20:00:00")),
    ]
    # Assert that last available timestamp of any interval with the end out of any working interval is the end
    start = pd.Timestamp("2020-01-21T04:15:17+00:00")
    end = pd.Timestamp("2020-01-21T14:21:10+00:00")
    assert get_last_available_timestamp(start, end, working_calendar) == end
    start = pd.Timestamp("2020-01-18T06:11:08+00:00")
    end = pd.Timestamp("2020-01-21T15:00:00+00:00")
    assert get_last_available_timestamp(start, end, working_calendar) == end
    start = pd.Timestamp("2020-01-21T12:11:08+00:00")
    end = pd.Timestamp("2020-01-21T14:35:12+00:00")
    assert get_last_available_timestamp(start, end, working_calendar) == end


def test_absolute_unavailability_intervals_within():
    # Create working calendar with 3 working periods from Monday to Friday
    working_calendar = RCalendar("test")
    daily_calendar = [
        Interval(pd.Timestamp("2023-01-25T05:00:00"), pd.Timestamp("2023-01-25T12:00:00")),
        Interval(pd.Timestamp("2023-01-25T14:00:00"), pd.Timestamp("2023-01-25T18:00:00")),
        Interval(pd.Timestamp("2023-01-25T20:00:00"), pd.Timestamp("2023-01-25T22:00:00")),
    ]
    working_calendar.work_intervals[0] = daily_calendar
    working_calendar.work_intervals[1] = daily_calendar
    working_calendar.work_intervals[2] = daily_calendar
    working_calendar.work_intervals[3] = daily_calendar
    working_calendar.work_intervals[4] = daily_calendar
    # Non-working periods between the middle of the first turn, to the middle of the last one, are the two slots between working periods
    assert absolute_unavailability_intervals_within(
        start=pd.Timestamp("2023-01-11T06:10:46+00:00"),
        end=pd.Timestamp("2023-01-11T21:35:11+00:00"),
        schedule=working_calendar,
    ) == [
        Interval(
            pd.Timestamp("2023-01-11T12:00:00+00:00"),
            pd.Timestamp("2023-01-11T14:00:00+00:00"),
        ),
        Interval(
            pd.Timestamp("2023-01-11T18:00:00+00:00"),
            pd.Timestamp("2023-01-11T20:00:00+00:00"),
        ),
    ]
    # Non-working periods withing the weekend is all the search interval
    assert absolute_unavailability_intervals_within(
        start=pd.Timestamp("2023-01-06T23:10:46+00:00"),
        end=pd.Timestamp("2023-01-08T19:35:11+00:00"),
        schedule=working_calendar,
    ) == [
        Interval(
            pd.Timestamp("2023-01-06T23:10:46+00:00"),
            pd.Timestamp("2023-01-06T23:59:59.999999+00:00"),
        ),
        Interval(
            pd.Timestamp("2023-01-07T00:00:00+00:00"),
            pd.Timestamp("2023-01-07T23:59:59.999999+00:00"),
        ),
        Interval(
            pd.Timestamp("2023-01-08T00:00:00+00:00"),
            pd.Timestamp("2023-01-08T19:35:11+00:00"),
        ),
    ]
    # Non-working periods of more than one week
    assert absolute_unavailability_intervals_within(
        start=pd.Timestamp("2023-02-06T03:10:46+00:00"),
        end=pd.Timestamp("2023-02-15T21:35:11+00:00"),
        schedule=working_calendar,
    ) == [
        # Monday
        Interval(
            pd.Timestamp("2023-02-06T03:10:46+00:00"),
            pd.Timestamp("2023-02-06T05:00:00+00:00"),
        ),
        Interval(
            pd.Timestamp("2023-02-06T12:00:00+00:00"),
            pd.Timestamp("2023-02-06T14:00:00+00:00"),
        ),
        Interval(
            pd.Timestamp("2023-02-06T18:00:00+00:00"),
            pd.Timestamp("2023-02-06T20:00:00+00:00"),
        ),
        Interval(
            pd.Timestamp("2023-02-06T22:00:00+00:00"),
            pd.Timestamp("2023-02-06T23:59:59.999999+00:00"),
        ),
        # Tuesday
        Interval(
            pd.Timestamp("2023-02-07T00:00:00+00:00"),
            pd.Timestamp("2023-02-07T05:00:00+00:00"),
        ),
        Interval(
            pd.Timestamp("2023-02-07T12:00:00+00:00"),
            pd.Timestamp("2023-02-07T14:00:00+00:00"),
        ),
        Interval(
            pd.Timestamp("2023-02-07T18:00:00+00:00"),
            pd.Timestamp("2023-02-07T20:00:00+00:00"),
        ),
        Interval(
            pd.Timestamp("2023-02-07T22:00:00+00:00"),
            pd.Timestamp("2023-02-07T23:59:59.999999+00:00"),
        ),
        # Wednesday
        Interval(
            pd.Timestamp("2023-02-08T00:00:00+00:00"),
            pd.Timestamp("2023-02-08T05:00:00+00:00"),
        ),
        Interval(
            pd.Timestamp("2023-02-08T12:00:00+00:00"),
            pd.Timestamp("2023-02-08T14:00:00+00:00"),
        ),
        Interval(
            pd.Timestamp("2023-02-08T18:00:00+00:00"),
            pd.Timestamp("2023-02-08T20:00:00+00:00"),
        ),
        Interval(
            pd.Timestamp("2023-02-08T22:00:00+00:00"),
            pd.Timestamp("2023-02-08T23:59:59.999999+00:00"),
        ),
        # Thursday
        Interval(
            pd.Timestamp("2023-02-09T00:00:00+00:00"),
            pd.Timestamp("2023-02-09T05:00:00+00:00"),
        ),
        Interval(
            pd.Timestamp("2023-02-09T12:00:00+00:00"),
            pd.Timestamp("2023-02-09T14:00:00+00:00"),
        ),
        Interval(
            pd.Timestamp("2023-02-09T18:00:00+00:00"),
            pd.Timestamp("2023-02-09T20:00:00+00:00"),
        ),
        Interval(
            pd.Timestamp("2023-02-09T22:00:00+00:00"),
            pd.Timestamp("2023-02-09T23:59:59.999999+00:00"),
        ),
        # Friday
        Interval(
            pd.Timestamp("2023-02-10T00:00:00+00:00"),
            pd.Timestamp("2023-02-10T05:00:00+00:00"),
        ),
        Interval(
            pd.Timestamp("2023-02-10T12:00:00+00:00"),
            pd.Timestamp("2023-02-10T14:00:00+00:00"),
        ),
        Interval(
            pd.Timestamp("2023-02-10T18:00:00+00:00"),
            pd.Timestamp("2023-02-10T20:00:00+00:00"),
        ),
        Interval(
            pd.Timestamp("2023-02-10T22:00:00+00:00"),
            pd.Timestamp("2023-02-10T23:59:59.999999+00:00"),
        ),
        # Saturday
        Interval(
            pd.Timestamp("2023-02-11T00:00:00+00:00"),
            pd.Timestamp("2023-02-11T23:59:59.999999+00:00"),
        ),
        # Sunday
        Interval(
            pd.Timestamp("2023-02-12T00:00:00+00:00"),
            pd.Timestamp("2023-02-12T23:59:59.999999+00:00"),
        ),
        # Monday
        Interval(
            pd.Timestamp("2023-02-13T00:00:00+00:00"),
            pd.Timestamp("2023-02-13T05:00:00+00:00"),
        ),
        Interval(
            pd.Timestamp("2023-02-13T12:00:00+00:00"),
            pd.Timestamp("2023-02-13T14:00:00+00:00"),
        ),
        Interval(
            pd.Timestamp("2023-02-13T18:00:00+00:00"),
            pd.Timestamp("2023-02-13T20:00:00+00:00"),
        ),
        Interval(
            pd.Timestamp("2023-02-13T22:00:00+00:00"),
            pd.Timestamp("2023-02-13T23:59:59.999999+00:00"),
        ),
        # Tuesday
        Interval(
            pd.Timestamp("2023-02-14T00:00:00+00:00"),
            pd.Timestamp("2023-02-14T05:00:00+00:00"),
        ),
        Interval(
            pd.Timestamp("2023-02-14T12:00:00+00:00"),
            pd.Timestamp("2023-02-14T14:00:00+00:00"),
        ),
        Interval(
            pd.Timestamp("2023-02-14T18:00:00+00:00"),
            pd.Timestamp("2023-02-14T20:00:00+00:00"),
        ),
        Interval(
            pd.Timestamp("2023-02-14T22:00:00+00:00"),
            pd.Timestamp("2023-02-14T23:59:59.999999+00:00"),
        ),
        # Wednesday
        Interval(
            pd.Timestamp("2023-02-15T00:00:00+00:00"),
            pd.Timestamp("2023-02-15T05:00:00+00:00"),
        ),
        Interval(
            pd.Timestamp("2023-02-15T12:00:00+00:00"),
            pd.Timestamp("2023-02-15T14:00:00+00:00"),
        ),
        Interval(
            pd.Timestamp("2023-02-15T18:00:00+00:00"),
            pd.Timestamp("2023-02-15T20:00:00+00:00"),
        ),
    ]
