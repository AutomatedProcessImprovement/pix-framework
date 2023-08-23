from dataclasses import dataclass

import pandas as pd

from pix_framework.discovery.resource_calendar_and_performance.crisp.resource_calendar import (
    int_week_days,
    str_week_days,
)


class FuzzyInterval:
    _from_day: int
    _to_day: int
    _start_time: pd.Timestamp
    _end_time: pd.Timestamp
    probability: float

    def __init__(
        self, from_day: int, to_day: int, start_time: pd.Timestamp, end_time: pd.Timestamp, probability: float
    ):
        self._from_day = from_day
        self._to_day = to_day
        self._start_time = start_time
        self._end_time = end_time
        self.probability = probability

    @property
    def from_day(self):
        return int_week_days[self._from_day]

    @property
    def to_day(self):
        return int_week_days[self._to_day]

    @property
    def start_time(self):
        return self._start_time.strftime("%H:%M:%S")

    @property
    def end_time(self):
        return self._end_time.strftime("%H:%M:%S")

    def to_prosimos(self) -> dict:
        return {
            "from": self.from_day,
            "to": self.to_day,
            "beginTime": self.start_time,
            "endTime": self.end_time,
            "probability": self.probability,
        }

    @staticmethod
    def from_prosimos(interval: dict) -> "FuzzyInterval":
        return FuzzyInterval(
            from_day=str_week_days[interval["from"]],
            to_day=str_week_days[interval["to"]],
            start_time=pd.Timestamp(interval["beginTime"]),
            end_time=pd.Timestamp(interval["endTime"]),
            probability=interval["probability"],
        )


@dataclass
class FuzzyResourceCalendar:
    resource_id: str
    resource_name: str
    intervals: list[FuzzyInterval]
    workloads: list[FuzzyInterval]

    def to_dict(self) -> dict:  # NOTE: for compatibility with RCalendar that uses to_dict instead of to_prosimos
        return self.to_prosimos()

    @staticmethod  # NOTE: for compatibility with RCalendar that uses to_dict instead of to_prosimos
    def from_dict(calendar: dict) -> "FuzzyResourceCalendar":
        return FuzzyResourceCalendar.from_prosimos(calendar)

    def to_prosimos(self):
        return {
            "id": self.resource_id,
            "name": self.resource_id,
            "time_periods": [interval.to_prosimos() for interval in self.intervals],
            "workload_ratio": [interval.to_prosimos() for interval in self.workloads],
        }

    @staticmethod
    def from_prosimos(calendar: dict) -> "FuzzyResourceCalendar":
        return FuzzyResourceCalendar(
            resource_id=calendar["id"],
            resource_name=calendar["name"],
            intervals=[FuzzyInterval.from_prosimos(i) for i in calendar["time_periods"]],
            workloads=[FuzzyInterval.from_prosimos(i) for i in calendar["workload_ratio"]],
        )
