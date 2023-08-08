from pathlib import Path

import pandas as pd
from pix_framework.calendar.resource_calendar import Interval, RCalendar
from pix_framework.io.event_log import read_csv_log
from start_time_estimator.config import Configuration
from start_time_estimator.resource_availability import CalendarResourceAvailability, SimpleResourceAvailability

assets_dir = Path(__file__).parent / "assets"


def test_simple_resource_availability():
    config = Configuration()
    event_log = read_csv_log(assets_dir / "test_event_log_1.csv", config.log_ids, config.missing_resource)
    resource_availability = SimpleResourceAvailability(event_log, config)
    # The configuration for the algorithm is the passed
    assert resource_availability.config == config
    # All the resources have been loaded
    assert set(resource_availability.performed_events.keys()) == {"Marcus", "Dominic", "Anya"}
    # The availability of the resource is the timestamp of its previous executed event
    third_trace = event_log[event_log[config.log_ids.case] == "trace-03"]
    first_trace = event_log[event_log[config.log_ids.case] == "trace-01"]
    assert (
        resource_availability.available_since("Marcus", third_trace.iloc[4])
        == first_trace.iloc[4][config.log_ids.end_time]
    )
    # The availability of the resource is the timestamp of its previous executed event
    artificial_event = {config.log_ids.end_time: pd.Timestamp("2006-11-07T17:00:00.000+02:00")}
    assert (
        resource_availability.available_since("Dominic", artificial_event)
        == first_trace.iloc[2][config.log_ids.end_time]
    )
    # The availability of the resource is the pd.NaT for the first event of the resource
    fourth_trace = event_log[event_log[config.log_ids.case] == "trace-04"]
    assert pd.isna(resource_availability.available_since("Anya", fourth_trace.iloc[0]))
    # The missing resource is always available (pd.NaT)
    artificial_event = {config.log_ids.end_time: pd.Timestamp("2006-11-07T10:00:00.000+02:00")}
    assert pd.isna(resource_availability.available_since(config.missing_resource, artificial_event))
    artificial_event = {config.log_ids.end_time: pd.Timestamp("2006-11-09T10:00:00.000+02:00")}
    assert pd.isna(resource_availability.available_since(config.missing_resource, artificial_event))


def test_simple_resource_availability_bot_resources():
    config = Configuration(bot_resources={"Marcus", "Dominic"})
    event_log = read_csv_log(assets_dir / "test_event_log_1.csv", config.log_ids, config.missing_resource)
    resource_availability = SimpleResourceAvailability(event_log, config)
    # The configuration for the algorithm is the passed
    assert resource_availability.config == config
    # All the resources have been loaded
    assert set(resource_availability.performed_events.keys()) == {"Anya"}
    # The availability of a bot resource is the same timestamp as checked
    first_trace = event_log[event_log[config.log_ids.case] == "trace-01"]
    assert (
        resource_availability.available_since("Marcus", first_trace.iloc[4])
        == first_trace.iloc[4][config.log_ids.end_time]
    )
    # The availability of a bot resource is the same timestamp as checked
    artificial_event = {config.log_ids.end_time: pd.Timestamp("2006-11-07T17:00:00.000+02:00")}
    assert resource_availability.available_since("Dominic", artificial_event) == pd.Timestamp(
        "2006-11-07T17:00:00.000+02:00"
    )
    # The availability of the resource is pd.NaT for the first event of the resource
    fourth_trace = event_log[event_log[config.log_ids.case] == "trace-04"]
    assert pd.isna(resource_availability.available_since("Anya", fourth_trace.iloc[0]))


def test_simple_resource_availability_considering_start_times():
    config = Configuration(consider_start_times=True)
    event_log = read_csv_log(assets_dir / "test_event_log_4.csv", config.log_ids, config.missing_resource)
    resource_availability = SimpleResourceAvailability(event_log, config)
    # The availability of a resource considers the recorded start times
    second_trace = event_log[event_log[config.log_ids.case] == "trace-02"]
    fourth_trace = event_log[event_log[config.log_ids.case] == "trace-04"]
    assert (
        resource_availability.available_since("Marcus", fourth_trace.iloc[1])
        == second_trace.iloc[1][config.log_ids.end_time]
    )
    # The availability of a resource considers the recorded start times but if equals its ok
    fifth_trace = event_log[event_log[config.log_ids.case] == "trace-05"]
    assert (
        resource_availability.available_since("Marcus", fifth_trace.iloc[0])
        == fourth_trace.iloc[2][config.log_ids.end_time]
    )


def test_calendar_resource_availability():
    working_calendar = RCalendar("test")
    working_calendar.work_intervals[0] = [
        Interval(pd.Timestamp("2023-01-23 08:00:00+00:00"), pd.Timestamp("2023-01-23 14:00:00+00:00")),
        Interval(pd.Timestamp("2023-01-23 16:00:00+00:00"), pd.Timestamp("2023-01-23 20:00:00+00:00")),
    ]
    config = Configuration(working_schedules={"Marcus": working_calendar, "Dominic": working_calendar})
    event_log = read_csv_log(assets_dir / "test_event_log_5.csv", config.log_ids, config.missing_resource)
    resource_availability = CalendarResourceAvailability(event_log, config)
    # The configuration for the algorithm is the passed
    assert resource_availability.config == config
    # All the resources have been loaded
    assert set(resource_availability.performed_events.keys()) == {"Marcus", "Dominic"}
    # The availability of the resource is the timestamp of its previous executed event
    first_trace = event_log[event_log[config.log_ids.case] == "trace-01"]
    second_trace = event_log[event_log[config.log_ids.case] == "trace-02"]
    assert (
        resource_availability.available_since("Marcus", first_trace.iloc[2])
        == first_trace.iloc[0][config.log_ids.end_time]
    )
    assert (
        resource_availability.available_since("Marcus", second_trace.iloc[3])
        == second_trace.iloc[1][config.log_ids.end_time]
    )
    # The availability of the resource is the timestamp of its previous executed event
    artificial_event = {config.log_ids.end_time: pd.Timestamp("2022-01-03 11:30:00+00:00")}
    assert (
        resource_availability.available_since("Marcus", artificial_event)
        == first_trace.iloc[0][config.log_ids.end_time]
    )
    # The availability of the resource is the end of the last non-working period for the first event of the resource
    artificial_event = {config.log_ids.end_time: pd.Timestamp("2022-01-03 09:00:00+00:00")}
    assert resource_availability.available_since("Marcus", artificial_event) == pd.Timestamp(
        "2022-01-03 08:00:00+00:00"
    )
    assert resource_availability.available_since("Marcus", first_trace.iloc[0]) == pd.Timestamp(
        "2022-01-03 08:00:00+00:00"
    )
    # The missing resource is always available (pd.NaT)
    artificial_event = {config.log_ids.end_time: pd.Timestamp("2022-01-03 12:00:00+00:00")}
    assert pd.isna(resource_availability.available_since(config.missing_resource, artificial_event))


def test_calendar_resource_availability_bot_resources():
    working_calendar = RCalendar("test")
    working_calendar.work_intervals[0] = [
        Interval(pd.Timestamp("2023-01-23 08:00:00+00:00"), pd.Timestamp("2023-01-23 14:00:00+00:00")),
        Interval(pd.Timestamp("2023-01-23 16:00:00+00:00"), pd.Timestamp("2023-01-23 20:00:00+00:00")),
    ]
    config = Configuration(bot_resources={"Dominic"}, working_schedules={"Marcus": working_calendar})
    event_log = read_csv_log(assets_dir / "test_event_log_5.csv", config.log_ids, config.missing_resource)
    resource_availability = CalendarResourceAvailability(event_log, config)
    # The configuration for the algorithm is the passed
    assert resource_availability.config == config
    # All the resources have been loaded
    assert set(resource_availability.performed_events.keys()) == {"Marcus"}
    # The availability of a bot resource is the same timestamp as checked
    first_trace = event_log[event_log[config.log_ids.case] == "trace-01"]
    assert (
        resource_availability.available_since("Dominic", first_trace.iloc[4])
        == first_trace.iloc[4][config.log_ids.end_time]
    )
    # The availability of the resource is the end of the last non-working period for the first event of the resource
    artificial_event = {config.log_ids.end_time: pd.Timestamp("2022-01-03 09:00:00+00:00")}
    assert resource_availability.available_since("Marcus", artificial_event) == pd.Timestamp(
        "2022-01-03 08:00:00+00:00"
    )
    assert resource_availability.available_since("Marcus", first_trace.iloc[0]) == pd.Timestamp(
        "2022-01-03 08:00:00+00:00"
    )
