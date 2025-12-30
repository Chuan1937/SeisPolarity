from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import obspy


@dataclass
class WaveformRecord:
    trace_id: str
    path: Path
    starttime: obspy.UTCDateTime | None = None
    endtime: obspy.UTCDateTime | None = None

    def load(self) -> obspy.Stream:
        """Load the waveform as an ObsPy stream."""
        return obspy.read(str(self.path))


class WaveformSource(ABC):
    @abstractmethod
    def __len__(self) -> int:  # pragma: no cover - thin wrapper
        raise NotImplementedError

    @abstractmethod
    def records(self) -> Iterator[WaveformRecord]:
        raise NotImplementedError

    def head(self, n: int = 5) -> list[WaveformRecord]:
        return [rec for _, rec in zip(range(n), self.records())]


class LocalDirectorySource(WaveformSource):
    """Simple directory-backed waveform source (e.g., miniSEED files)."""

    def __init__(self, root: str | Path, pattern: str = "*.mseed"):
        self.root = Path(root)
        self.pattern = pattern
        self._cache: list[WaveformRecord] | None = None

    def _scan(self) -> list[WaveformRecord]:
        records: list[WaveformRecord] = []
        for path in self.root.rglob(self.pattern):
            trace_id = path.stem
            records.append(WaveformRecord(trace_id=trace_id, path=path))
        return records

    def records(self) -> Iterator[WaveformRecord]:
        if self._cache is None:
            self._cache = self._scan()
        yield from self._cache

    def __len__(self) -> int:
        if self._cache is None:
            self._cache = self._scan()
        return len(self._cache)


class StreamBatch:
    """A small wrapper to track stream metadata alongside the data."""

    def __init__(self, stream: obspy.Stream, source: WaveformRecord | None = None):
        self.stream = stream
        self.source = source

    @property
    def trace_id(self) -> str | None:
        return self.source.trace_id if self.source else None


def load_streams(source: WaveformSource, limit: int | None = None) -> list[StreamBatch]:
    batches: list[StreamBatch] = []
    for idx, rec in enumerate(source.records()):
        if limit is not None and idx >= limit:
            break
        batches.append(StreamBatch(rec.load(), rec))
    return batches
