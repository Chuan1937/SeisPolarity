from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import obspy
import torch

from seispolarity.annotations import Pick, PickList, PolarityLabel, PolarityOutput


class BasePolarityModel(ABC):
    """Abstract base class for polarity pickers."""

    def __init__(self, name: str, sample_rate: float | None = None, n_components: int = 3):
        self.name = name
        self.sample_rate = sample_rate
        self.n_components = n_components
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

    @abstractmethod
    def forward_tensor(self, tensor: torch.Tensor, **kwargs) -> Any:
        """Model forward pass for a batch tensor."""

    @abstractmethod
    def build_picks(self, raw_output: Any, **kwargs) -> PickList:
        """Convert raw model output to picks with polarity labels."""

    def preprocess(self, stream: obspy.Stream) -> torch.Tensor:
        traces = []
        for tr in stream:
            data = tr.data.astype(np.float32)
            if self.sample_rate and tr.stats.sampling_rate != self.sample_rate:
                raise ValueError(
                    f"Stream sample rate {tr.stats.sampling_rate} does not match model {self.sample_rate}"
                )
            traces.append(data)
        arr = np.stack(traces, axis=0)
        tensor = torch.from_numpy(arr).float().unsqueeze(0)  # shape: (1, C, T)
        return tensor.to(self.device)

    def annotate(self, stream: obspy.Stream, **kwargs) -> PolarityOutput:
        tensor = self.preprocess(stream)
        raw = self.forward_tensor(tensor, **kwargs)
        picks = self.build_picks(raw, **kwargs)
        return PolarityOutput(picks=picks, raw=raw)

    def to(self, device: str | torch.device) -> "BasePolarityModel":
        self.device = torch.device(device)
        if isinstance(self, torch.nn.Module):
            torch.nn.Module.to(self, self.device)
        return self

    def _move(self, device: torch.device) -> None:
        # Subclasses can override; default no-op.
        return


class ThresholdPostprocessor:
    """Utility to convert probability traces into discrete picks."""

    def __init__(self, threshold: float = 0.5, polarity_threshold: float = 0.3):
        self.threshold = threshold
        self.polarity_threshold = polarity_threshold

    def __call__(self, probs: np.ndarray, trace_id: str, times: np.ndarray) -> PickList:
        picks = PickList()
        above = probs >= self.threshold
        if not above.any():
            return picks

        # Find local maxima over probability curve
        idxs = np.where(above)[0]
        for idx in idxs:
            polarity = None
            if probs.ndim > 1 and probs.shape[0] > 1:
                pol_up = probs[0, idx]
                pol_down = probs[1, idx]
                if pol_up >= self.polarity_threshold or pol_down >= self.polarity_threshold:
                    polarity = PolarityLabel.UP if pol_up >= pol_down else PolarityLabel.DOWN
            picks.append(
                Pick(
                    trace_id=trace_id,
                    time=obspy.UTCDateTime(times[idx]),
                    confidence=float(probs[idx] if probs.ndim == 1 else probs[:, idx].max()),
                    polarity=polarity,
                )
            )
        return picks
