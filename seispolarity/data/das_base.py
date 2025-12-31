import seispolarity.util
from .base import WaveformBenchmarkDataset

import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Union, Optional


class DASBenchmarkDataset(WaveformBenchmarkDataset):
    """
    Base class for DAS benchmark datasets.
    DAS 基准数据集的基类。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_waveforms(
        self,
        indices: Union[List[int], int],
        sampling_rate: Optional[float] = None,
        channels: Optional[Union[List[int], int]] = None,
    ) -> np.array:
        """
        Retrieves waveforms for a given set of indices.
        检索给定索引集的波形。

        :param indices: List of indices or single index (索引列表或单个索引)
        :param sampling_rate: Target sampling rate. If None, the native sampling rate is used. (目标采样率。如果为 None，则使用本机采样率。)
        :param channels: List of channels or single channel. If None, all channels are returned. (通道列表或单个通道。如果为 None，则返回所有通道。)
        :return: Waveforms as numpy array. Shape: (samples, channels, time) (作为 numpy 数组的波形。形状：(样本, 通道, 时间))
        """
        if isinstance(indices, int):
            indices = [indices]

        if isinstance(channels, int):
            channels = [channels]

        with h5py.File(self.path / "waveforms.hdf5", "r") as f:
            grp = f["data"]
            waveforms = []
            for idx in indices:
                trace_name = self.metadata.iloc[idx]["trace_name"]
                bucket = self.metadata.iloc[idx]["trace_name_original"]
                
                # Handle bucketing if applicable
                if "bucket" in self.metadata.columns:
                    bucket = self.metadata.iloc[idx]["bucket"]
                    dset = grp[bucket][trace_name]
                else:
                    dset = grp[trace_name]

                if channels is None:
                    data = dset[:]
                else:
                    data = dset[channels]

                waveforms.append(data)

        waveforms = np.stack(waveforms)

        if sampling_rate is not None:
            # Resampling logic would go here. 
            # For now, we assume the user handles resampling or the data is already at the correct rate.
            # SeisPolarity uses obspy or scipy for resampling.
            pass

        return waveforms
