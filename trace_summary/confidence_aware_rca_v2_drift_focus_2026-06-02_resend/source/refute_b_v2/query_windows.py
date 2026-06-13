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


MONTH_NAMES = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}


def parse_query_window(instruction: str) -> QueryWindow:
    date_matches = re.findall(r"(January|February|March|April|May|June|July|August|September|October|November|December) (\d{1,2}), (\d{4})", instruction)
    times = re.findall(r"(\d{1,2}:\d{2})", instruction)
    if not date_matches or len(times) < 2:
        raise ValueError(f"cannot parse query window: {instruction}")
    m1, d1, y1 = date_matches[0]
    m2, d2, y2 = date_matches[1] if len(date_matches) > 1 else date_matches[0]
    h1, n1 = (int(x) for x in times[0].split(":"))
    h2, n2 = (int(x) for x in times[1].split(":"))
    start = datetime(int(y1), MONTH_NAMES[m1], int(d1), h1, n1)
    end = datetime(int(y2), MONTH_NAMES[m2], int(d2), h2, n2)
    if end <= start:
        end = datetime(int(y1), MONTH_NAMES[m1], int(d1), h2, n2) + timedelta(days=1)
    return QueryWindow(start, end, parse_failure_count(instruction))
