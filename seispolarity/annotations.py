from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Iterable, Iterator, List, Optional

import pandas as pd
from obspy import UTCDateTime


class PolarityLabel(str, Enum):
    UP = "U"
    DOWN = "D"
    UNKNOWN = "N"


@dataclass(order=True)
class Pick:
    trace_id: str
    time: UTCDateTime
    confidence: Optional[float] = None
    phase: Optional[str] = None
    polarity: Optional[PolarityLabel] = None
    window: Optional[tuple[UTCDateTime, UTCDateTime]] = None
    extra: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "time": self.time.datetime if isinstance(self.time, UTCDateTime) else self.time,
            "confidence": self.confidence,
            "phase": self.phase,
            "polarity": self.polarity.value if isinstance(self.polarity, PolarityLabel) else self.polarity,
            "window_start": self.window[0].datetime if self.window else None,
            "window_end": self.window[1].datetime if self.window else None,
            **self.extra,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Pick":
        payload = payload.copy()
        pol = payload.pop("polarity", None)
        if pol is not None:
            try:
                payload["polarity"] = PolarityLabel(pol)
            except ValueError:
                payload["polarity"] = pol
        time_val = payload.get("time")
        if isinstance(time_val, datetime):
            payload["time"] = UTCDateTime(time_val)
        return cls(**payload)

    def __str__(self) -> str:  # pragma: no cover - human friendly only
        parts = [self.trace_id, str(self.time)]
        if self.phase:
            parts.append(self.phase)
        if self.polarity:
            parts.append(self.polarity.value if isinstance(self.polarity, PolarityLabel) else str(self.polarity))
        if self.confidence is not None:
            parts.append(f"p={self.confidence:.3f}")
        return " ".join(parts)


class PickList(List[Pick]):
    def __init__(self, picks: Iterable[Pick] | None = None):
        super().__init__(picks or [])

    def select(
        self,
        trace_id: str | None = None,
        min_confidence: float | None = None,
        phase: str | None = None,
    ) -> "PickList":
        filtered: list[Pick] = []
        for pick in self:
            if trace_id and trace_id not in pick.trace_id:
                continue
            if min_confidence is not None and (
                pick.confidence is None or pick.confidence < min_confidence
            ):
                continue
            if phase is not None and pick.phase != phase:
                continue
            filtered.append(pick)
        return PickList(filtered)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([p.to_row() for p in self])

    def __iter__(self) -> Iterator[Pick]:
        return super().__iter__()

    def __str__(self) -> str:  # pragma: no cover - human friendly only
        if not self:
            return "PickList(0)"
        head = ", ".join(str(p) for p in self[:3])
        tail = "" if len(self) <= 3 else f", ... {len(self) - 3} more"
        return f"PickList({len(self)}): {head}{tail}"


@dataclass
class PolarityOutput:
    picks: PickList
    raw: object | None = None
