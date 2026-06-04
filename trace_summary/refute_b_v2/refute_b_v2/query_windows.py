"""OpenRCA query parsing helpers shared by eval scripts."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import re


@dataclass(frozen=True)
class QueryWindow:
    start: datetime
    end: datetime
    failure_count: int

    @property
    def start_ts(self) -> int:
        return int(self.start.timestamp())

    @property
    def end_ts(self) -> int:
        return int(self.end.timestamp())

    @property
    def date_keys(self) -> list[str]:
        keys = []
        day = self.start.date()
        while day <= self.end.date():
            keys.append(day.strftime("%Y_%m_%d"))
            day += timedelta(days=1)
        return keys


def parse_failure_count(instruction: str) -> int:
    text = instruction.lower()
    if re.search(r"\b(two|2)\s+(?:system\s+)?failures\b", text) or "there were two" in text:
        return 2
    return 1


def parse_query_window(instruction: str) -> QueryWindow:
    dates = [int(x) for x in re.findall(r"March (\d{1,2}), 2021", instruction)]
    times = re.findall(r"(\d{1,2}:\d{2})", instruction)
    if not dates or len(times) < 2:
        raise ValueError(f"cannot parse query window: {instruction}")
    d1 = dates[0]
    d2 = dates[1] if len(dates) > 1 else d1
    h1, m1 = (int(x) for x in times[0].split(":"))
    h2, m2 = (int(x) for x in times[1].split(":"))
    start = datetime(2021, 3, d1, h1, m1)
    end = datetime(2021, 3, d2, h2, m2)
    if end <= start:
        end = datetime(2021, 3, d1, h2, m2) + timedelta(days=1)
    return QueryWindow(start, end, parse_failure_count(instruction))
