from __future__ import annotations

import copy
import logging
import warnings
import inspect
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional, List, Union, Tuple, Literal
from urllib.parse import urljoin

import h5py
import numpy as np
import pandas as pd
import scipy.signal
import matplotlib.pyplot as plt
from tqdm import tqdm

import seispolarity.util as util

# Configure logging
logger = logging.getLogger("seispolarity")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Default configuration
CONFIG = {
    "dimension_order": "NCW",
    "component_order": "ZNE",
    "cache_data_root": Path.home() / ".seispolarity" / "datasets",
    "remote_data_root": "https://huggingface.co/datasets/chuanjun1978/Seismic-AI-Data/resolve/main/",
}

def pad_packed_sequence(seq: List[np.ndarray]) -> np.ndarray:
    """
    Packs a list of arrays into one array by adding a new first dimension and padding where necessary.
    将一组数组打包成一个数组，通过添加一个新的第一维并在必要时进行填充。

    :param seq: List of numpy arrays (数组列表)
    :return: Combined arrays (组合后的数组)
    """
    if not seq:
        return np.array([])
        
    max_size = np.array([max([x.shape[i] for x in seq]) for i in range(seq[0].ndim)])

    new_seq = []
    for i, elem in enumerate(seq):
        d = max_size - np.array(elem.shape)
        if (d != 0).any():
            pad = [(0, d_dim) for d_dim in d]
            new_seq.append(np.pad(elem, pad, "constant", constant_values=0))
        else:
            new_seq.append(elem)

    return np.stack(new_seq, axis=0)

class LoadingContext:
    """
    The LoadingContext is a dict of pointers to the hdf5 files for the chunks.
    It is an easy way to manage opening and closing of file pointers when required.
    LoadingContext 是一个包含块的 hdf5 文件指针的字典。
    它是一种在需要时管理文件指针打开和关闭的简便方法。
    """

    def __init__(self, chunks, waveform_paths):
        self.chunk_dict = {
            chunk: waveform_path for chunk, waveform_path in zip(chunks, waveform_paths)
        }
        self.file_pointers = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for file in self.file_pointers.values():
            file.close()
        self.file_pointers = {}

    def __getitem__(self, chunk):
        if chunk not in self.chunk_dict:
            raise KeyError(f'Unknown chunk "{chunk}"')

        if chunk not in self.file_pointers:
            self.file_pointers[chunk] = h5py.File(self.chunk_dict[chunk], "r")
        return self.file_pointers[chunk]

class WaveformDataset:
    """
    This class is the base class for waveform datasets.
    """

    def __init__(
        self,
        path=None,
        name=None,
        dimension_order=None,
        component_order=None,
        sampling_rate=None,
        cache=None,
        chunks=None,
        missing_components="pad",
        metadata_cache=False,
        **kwargs,
    ):
        """
        Initialize the dataset.
        初始化数据集。
        """
        if name is None:
            self._name = "Unnamed dataset"
        else:
            self._name = name

        self.cache = cache
        self._path = path
        if self.path is None:
            raise ValueError("Path can not be None")
            
        self._chunks = chunks
        if chunks is not None:
            self._chunks = sorted(chunks)
            self._validate_chunks(path, self._chunks)

        self._missing_components = None
        self._trace_identification_warning_issued = False

        self._dimension_order = None
        self._dimension_mapping = None
        self._component_order = None
        self._component_mapping = None
        self._metadata_lookup = None
        self._chunks_with_paths_cache = None
        self.sampling_rate = sampling_rate

        self._verify_dataset()

        metadatas = []
        for chunk, metadata_path, _ in zip(*self._chunks_with_paths()):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

                tmp_metadata = pd.read_csv(
                    metadata_path,
                    dtype={
                        "trace_sampling_rate_hz": float,
                        "trace_dt_s": float,
                        "trace_component_order": str,
                    },
                )
            tmp_metadata["trace_chunk"] = chunk
            if tmp_metadata.get("source_origin_time") is not None:
                tmp_metadata.source_origin_time = pd.to_datetime(
                    tmp_metadata.source_origin_time
                )
            metadatas.append(tmp_metadata)
        
        if metadatas:
            self._metadata = pd.concat(metadatas)
            self._metadata.reset_index(inplace=True)
        else:
            self._metadata = pd.DataFrame()

        self._data_format = self._read_data_format()

        self._unify_sampling_rate()
        self._unify_component_order()
        self._build_trace_name_to_idx_dict()

        self.dimension_order = dimension_order
        self.component_order = component_order
        self.missing_components = missing_components
        self.metadata_cache = metadata_cache

        self._waveform_cache = defaultdict(dict)
        self.grouping = None

    def _validate_chunks(self, path, chunks):
        """
        Validate that the requested chunks exist in the dataset.
        验证请求的块是否存在于数据集中。
        """
        available = self.available_chunks(path)
        if any(chunk not in available for chunk in chunks):
            raise ValueError(
                f"The dataset does not contain the following chunks: "
                f"{[chunk for chunk in chunks if chunk not in available]}"
            )

    def __str__(self):
        """
        String representation of the dataset.
        数据集的字符串表示。
        """
        return f"{self._name} - {len(self)} traces"

    def copy(self):
        """
        Create a copy of the dataset.
        创建数据集的副本。
        """
        other = copy.copy(self)
        other._metadata = self._metadata.copy()
        other._waveform_cache = defaultdict(dict)
        for key in self._waveform_cache.keys():
            other._waveform_cache[key] = copy.copy(self._waveform_cache[key])
        return other

    @property
    def metadata(self):
        """
        Metadata of the dataset as pandas DataFrame.
        数据集的元数据（pandas DataFrame）。
        """
        return self._metadata
    
    @metadata.setter
    def metadata(self, value):
        self._metadata = value

    @property
    def name(self):
        """
        Name of the dataset.
        数据集名称。
        """
        return self._name

    @property
    def cache(self):
        """
        Get or set the cache strategy.
        获取或设置缓存策略。
        """
        return self._cache

    @cache.setter
    def cache(self, cache):
        if cache not in ["full", "trace", None]:
            raise ValueError(
                f"Unknown cache strategy '{cache}'. Allowed values are 'full', 'trace' and None."
            )
        self._cache = cache

    @property
    def metadata_cache(self):
        """
        Get or set metadata cache strategy.
        获取或设置元数据缓存策略。
        """
        return self._metadata_cache

    @metadata_cache.setter
    def metadata_cache(self, val):
        self._metadata_cache = val
        self._rebuild_metadata_cache()

    @property
    def path(self):
        """
        Path to the dataset.
        数据集路径。
        """
        if self._path is None:
            raise ValueError("Path is None. Can't create data set without a path.")
        return Path(self._path)

    @property
    def data_format(self):
        """
        Data format dictionary.
        数据格式字典。
        """
        return dict(self._data_format)

    @property
    def dimension_order(self):
        """
        Get or set the dimension order (e.g., 'NCW').
        获取或设置维度顺序（例如 'NCW'）。
        """
        return self._dimension_order

    @dimension_order.setter
    def dimension_order(self, value):
        if value is None:
            value = CONFIG["dimension_order"]

        self._dimension_mapping = self._get_dimension_mapping(
            "N" + self._data_format["dimension_order"], value
        )
        self._dimension_order = value

    @property
    def missing_components(self):
        """
        Get or set strategy for missing components.
        获取或设置缺失分量的处理策略。
        """
        return self._missing_components

    @missing_components.setter
    def missing_components(self, value):
        if value not in ["pad", "copy", "ignore"]:
            raise ValueError(
                f"Unknown missing components strategy '{value}'. "
                f"Allowed values are 'pad', 'copy' and 'ignore'."
            )

        self._missing_components = value
        self.component_order = self.component_order

    @property
    def component_order(self):
        """
        Get or set the component order (e.g., 'ZNE').
        获取或设置分量顺序（例如 'ZNE'）。
        """
        return self._component_order

    @component_order.setter
    def component_order(self, value):
        if value is None:
            value = CONFIG["component_order"]
            logger.warning(f"Output component_order not specified, defaulting to '{value}'.")

        if self.missing_components is not None:
            self._component_mapping = {}
            if "trace_component_order" in self.metadata.columns:
                for source_order in self.metadata["trace_component_order"].unique():
                    self._component_mapping[source_order] = self._get_component_mapping(
                        source_order, value
                    )
            else:
                # Fallback if trace_component_order is missing but data_format has it
                source_order = self.data_format.get("component_order", "ZNE")
                self._component_mapping[source_order] = self._get_component_mapping(
                    source_order, value
                )

        self._component_order = value

    @property
    def grouping(self):
        """
        Get or set the grouping parameter.
        获取或设置分组参数。
        """
        return self._grouping

    @grouping.setter
    def grouping(self, value):
        self._grouping = value

        if value is None:
            self._groups = None
            self._groups_to_trace_idx = None
        else:
            self._metadata.reset_index(inplace=True, drop=True)
            self._groups_to_trace_idx = self.metadata.groupby(value).groups
            self._groups = list(self._groups_to_trace_idx.keys())
            self._groups_to_group_idx = {
                group: i for i, group in enumerate(self._groups)
            }

    @property
    def groups(self):
        """
        List of groups.
        组列表。
        """
        return copy.copy(self._groups)

    @property
    def chunks(self):
        """
        List of chunks in the dataset.
        数据集中的块列表。
        """
        if self._chunks is None:
            self._chunks = self.available_chunks(self.path)
        return self._chunks

    @staticmethod
    def available_chunks(path):
        """
        Determine available chunks in the dataset path.
        确定数据集路径中的可用块。
        """
        path = Path(path)
        chunks_path = path / "chunks"
        if chunks_path.is_file():
            with open(chunks_path, "r") as f:
                chunks = [x for x in f.read().split("\n") if x.strip()]
            if len(chunks) == 0:
                logger.warning("Found empty chunks file. Using chunk detection from file names.")
        else:
            chunks = []

        if len(chunks) == 0:
            if (path / "waveforms.hdf5").is_file():
                chunks = [""]
            else:
                metadata_files = set(
                    [
                        x.name[8:-4]
                        for x in path.iterdir()
                        if x.name.startswith("metadata") and x.name.endswith(".csv")
                    ]
                )
                waveform_files = set(
                    [
                        x.name[9:-5]
                        for x in path.iterdir()
                        if x.name.startswith("waveforms") and x.name.endswith(".hdf5")
                    ]
                )
                chunks = metadata_files & waveform_files
                chunks = list(chunks)

        return sorted(chunks)

    def _rebuild_metadata_cache(self):
        """
        Rebuild the metadata cache.
        重建元数据缓存。
        """
        if self.metadata_cache:
            self._metadata_lookup = list(
                self._metadata.apply(lambda x: x.to_dict(), axis=1)
            )
        else:
            self._metadata_lookup = None

    def _unify_sampling_rate(self, eps=1e-4):
        """
        Unify sampling rate across the dataset.
        统一数据集中的采样率。
        """
        if "trace_sampling_rate_hz" in self.metadata.columns:
            if "trace_dt_s" in self.metadata.columns:
                if np.any(np.isnan(self.metadata["trace_sampling_rate_hz"].values)):
                    mask = np.isnan(self.metadata["trace_sampling_rate_hz"].values)
                    self._metadata.loc[mask, "trace_sampling_rate_hz"] = (
                        1 / self.metadata.loc[mask, "trace_dt_s"]
                    )
        elif "trace_dt_s" in self.metadata.columns:
            self.metadata["trace_sampling_rate_hz"] = 1 / self.metadata["trace_dt_s"]
        elif "sampling_rate" in self.data_format:
            self._metadata["trace_sampling_rate_hz"] = self.data_format["sampling_rate"]
        else:
            logger.warning("Sampling rate not specified in data set.")
            self._metadata["trace_sampling_rate_hz"] = np.nan

    def _get_component_mapping(self, source, target):
        """
        Get mapping from source component order to target component order.
        获取从源分量顺序到目标分量顺序的映射。
        """
        if (isinstance(source, float) and np.isnan(source)) or (
            (isinstance(source, list) or isinstance(source, str)) and not len(source)
        ):
            raise ValueError("Component order not set for (parts of) the dataset.")

        source = list(source)
        target = list(target)

        mapping = []
        for t in target:
            if t in source:
                mapping.append(source.index(t))
            else:
                if self.missing_components == "pad":
                    mapping.append(len(source))
                elif self.missing_components == "copy":
                    mapping.append(0)
                else:
                    pass
        return mapping

    @staticmethod
    def _get_dimension_mapping(source, target):
        """
        Get mapping from source dimension order to target dimension order.
        获取从源维度顺序到目标维度顺序的映射。
        """
        source = list(source)
        target = list(target)
        try:
            mapping = [source.index(t) for t in target]
        except ValueError:
            raise ValueError(
                f"Could not determine mapping {source} -> {target}."
            )
        return mapping

    def _chunks_with_paths(self):
        """
        Get chunks with their metadata and waveform paths.
        获取块及其元数据和波形路径。
        """
        if self._chunks_with_paths_cache is None:
            metadata_paths = []
            waveform_paths = []
            for chunk in self.chunks:
                if chunk == "":
                    metadata_paths.append(self.path / "metadata.csv")
                    waveform_paths.append(self.path / "waveforms.hdf5")
                else:
                    metadata_paths.append(self.path / f"metadata_{chunk}.csv")
                    waveform_paths.append(self.path / f"waveforms_{chunk}.hdf5")
            
            self._chunks_with_paths_cache = (
                self.chunks,
                metadata_paths,
                waveform_paths,
            )
        return self._chunks_with_paths_cache

    def _verify_dataset(self):
        """
        Verify that all chunks have corresponding metadata and waveform files.
        验证所有块都有对应的元数据和波形文件。
        """
        for chunk, metadata_path, waveform_path in zip(*self._chunks_with_paths()):
            if not metadata_path.is_file():
                raise FileNotFoundError(f"Missing metadata file for chunk '{chunk}'")
            if not waveform_path.is_file():
                raise FileNotFoundError(f"Missing waveforms file for chunk '{chunk}'")

    def _read_data_format(self):
        """
        Read data format from waveform files.
        从波形文件中读取数据格式。
        """
        data_format = None
        for waveform_file in self._chunks_with_paths()[2]:
            with h5py.File(waveform_file, "r") as f_wave:
                try:
                    g_data_format = f_wave["data_format"]
                    tmp_data_format = {
                        key: g_data_format[key][()] for key in g_data_format.keys()
                    }
                except KeyError:
                    tmp_data_format = {}

            if data_format is None:
                data_format = tmp_data_format

        for key in data_format.keys():
            if isinstance(data_format[key], bytes):
                data_format[key] = data_format[key].decode()

        if "dimension_order" not in data_format:
            data_format["dimension_order"] = "CW"

        return data_format

    def _unify_component_order(self):
        """
        Unify component order across the dataset.
        统一数据集中的分量顺序。
        """
        if "component_order" in self.data_format:
            if "trace_component_order" not in self.metadata.columns:
                self._metadata["trace_component_order"] = self.data_format["component_order"]
        else:
            if "trace_component_order" not in self.metadata.columns:
                logger.warning("Component order not specified. Keeping original components.")

    def get_group_size(self, idx):
        """
        Returns the number of samples in a group

        :param idx: Group index
        :type idx: int
        :return: Size of the group
        :rtype: int
        """
        self._verify_grouping_defined()
        group = self._groups[idx]
        idx = self._groups_to_trace_idx[group]
        return len(idx)

    def get_group_samples(self, idx, **kwargs):
        """
        Returns the waveforms and metadata for each member of a group.
        For details see :py:func:`get_sample`.

        :param idx: Group index
        :type idx: int
        :param kwargs: Kwargs passed to :py:func:`get_sample`
        :return: List of waveforms, list of metadata dicts
        """
        return self._get_group_internal(idx, return_metadata=True, **kwargs)

    def get_group_waveforms(self, idx, **kwargs):
        """
        Returns the waveforms for each member of a group.
        For details see :py:func:`get_sample`.

        :param idx: Group index
        :type idx: int
        :param kwargs: Kwargs passed to :py:func:`get_sample`
        :return: List of waveforms
        """
        return self._get_group_internal(idx, return_metadata=False, **kwargs)

    def _get_group_internal(self, idx, return_metadata, sampling_rate=None):
        self._verify_grouping_defined()

        group = self._groups[idx]
        idx = self._groups_to_trace_idx[group]

        if self._metadata_lookup is None:
            metadata = self.metadata.iloc[idx].to_dict("list")
        else:
            metadata = WaveformDataset._pack_metadata(
                [self._metadata_lookup[i] for i in idx]
            )

        sampling_rate = self._get_sample_unify_sampling_rate(metadata, sampling_rate)

        waveforms = self._get_waveforms_from_load_metadata(
            metadata, sampling_rate, pack=False
        )

        self._calculate_trace_npts_group(metadata, waveforms)

        if return_metadata:
            return waveforms, metadata
        else:
            return waveforms

    def _calculate_trace_npts_group(self, metadata, waveforms):
        dimension_order = list(self.dimension_order)
        del dimension_order[dimension_order.index("N")]
        sample_dimension = dimension_order.index("W")
        metadata["trace_npts"] = [wv.shape[sample_dimension] for wv in waveforms]

    @staticmethod
    def _pack_metadata(metadata):
        """
        Reformats a list of dict into a dict of lists. Assumes identical keys in all dicts!
        """

        return {key: [m[key] for m in metadata] for key in metadata[0].keys()}

    def _verify_grouping_defined(self):
        """
        Check if grouping is defined and raises and error otherwise
        """
        if self.grouping is None:
            raise ValueError(
                "Groups need to be defined first by assigning a value to grouping."
            )

    def get_group_idx_from_params(self, params):
        """
        Returns the index of the group identified by the params.

        :param params: The parameters identifying the group. For a single grouping parameter, this argument will be a
                       single value. Otherwise this argument needs to be a tuple of keys.
        :return: Index of the group
        :rtype: int
        """
        self._verify_grouping_defined()

        if params in self._groups_to_group_idx:
            return self._groups_to_group_idx[params]
        else:
            raise KeyError("The dataset does not contain the requested group.")

    def get_idx_from_trace_name(self, trace_name, chunk=None, dataset=None):
        """
        Returns the index of a trace with given trace_name, chunk and dataset.
        Chunk and dataset parameters are optional, but might be necessary to uniquely identify traces for
        chunked datasets or for :py:class:`MultiWaveformDataset`.
        The method will issue a warning *the first time* a non-uniquely identifiable trace is requested.
        If no matching key is found, a `KeyError` is raised.

        :param trace_name: Trace name as in metadata["trace_name"]
        :type trace_name: str
        :param chunk: Trace chunk as in metadata["trace_chunk"]. If None this key will be ignored.
        :type chunk: None
        :param dataset: Trace dataset as in metadata["trace_dataset"]. Only for :py:class:`MultiWaveformDataset`.
                        If None this key will be ignored.
        :type dataset: None
        :return: Index of the sample
        """
        dict_key = "name"
        search_key = [trace_name]
        if chunk is not None:
            dict_key += "_chunk"
            search_key.append(chunk)
        if dataset is not None:
            dict_key += "_dataset"
            search_key.append(dataset)

        search_key = tuple(search_key)

        if not self._trace_identification_warning_issued and len(
            self._trace_name_to_idx[dict_key]
        ) != len(self.metadata):
            logger.warning(
                f"Traces can not uniformly be identified using {dict_key.replace('_', ', ')}. "
                '"get_idx_from_trace_name" will return only one possible matching trace.'
            )
            self._trace_identification_warning_issued = True

        if search_key in self._trace_name_to_idx[dict_key]:
            return self._trace_name_to_idx[dict_key][search_key]
        else:
            raise KeyError("The dataset does not contain the requested trace.")

    def _build_trace_name_to_idx_dict(self):
        """
        Builds mapping of trace_names to idx.
        """
        self._trace_name_to_idx = {}
        self._trace_name_to_idx["name"] = {
            (trace_name,): i for i, trace_name in enumerate(self.metadata["trace_name"])
        }
        self._trace_name_to_idx["name_chunk"] = {
            trace_info: i
            for i, trace_info in enumerate(
                zip(self.metadata["trace_name"], self.metadata["trace_chunk"])
            )
        }
        if "trace_dataset" in self.metadata.columns:
            self._trace_name_to_idx["name_dataset"] = {
                trace_info: i
                for i, trace_info in enumerate(
                    zip(self.metadata["trace_name"], self.metadata["trace_dataset"])
                )
            }
            self._trace_name_to_idx["name_chunk_dataset"] = {
                trace_info: i
                for i, trace_info in enumerate(
                    zip(
                        self.metadata["trace_name"],
                        self.metadata["trace_chunk"],
                        self.metadata["trace_dataset"],
                    )
                )
            }
        else:
            self._trace_name_to_idx["name_dataset"] = {}
            self._trace_name_to_idx["name_chunk_dataset"] = {}

        self._trace_identification_warning_issued = False

    def preload_waveforms(self, pbar=False):
        """
        Preload all waveforms into memory.
        将所有波形预加载到内存中。
        """
        if self.cache is None:
            logger.warning("Skipping preload, as cache is disabled.")
            return

        chunks, metadata_paths, waveforms_path = self._chunks_with_paths()
        with LoadingContext(chunks, waveforms_path) as context:
            iterator = zip(self._metadata["trace_name"], self._metadata["trace_chunk"])
            if pbar:
                iterator = tqdm(
                    iterator, total=len(self._metadata), desc="Preloading waveforms"
                )

            for trace_name, chunk in iterator:
                self._get_single_waveform(trace_name, chunk, context=context)

    def filter(self, mask, inplace=True):
        """
        Filter the dataset using a boolean mask.
        使用布尔掩码过滤数据集。
        """
        if inplace:
            self._metadata = self._metadata[mask]
            self._evict_cache()
            self._build_trace_name_to_idx_dict()
            self._rebuild_metadata_cache()
            self.grouping = self.grouping
            return self
        else:
            other = self.copy()
            other.filter(mask, inplace=True)
            return other

    # NOTE: lat/lon columns are specified to enhance generalisability as naming convention may
    # change between datasets and users may also want to filter as a function of  receivers/sources
    def region_filter(self, domain, lat_col, lon_col, inplace=True):
        """
        Filtering of dataset based on predefined region or geometry.
        See also convenience functions region_filter_[source|receiver].

        :param domain: The domain filter
        :type domain: obspy.core.fdsn.mass_downloader.domain:
        :param lat_col: Name of latitude coordinate column
        :type lat_col: str
        :param lon_col: Name of longitude coordinate column
        :type lon_col: str
        :param inplace: Inplace filtering, default to true. See also :py:func:`~WaveformDataset.filter`.
        :type inplace: bool
        :return: None if inplace=True, otherwise the filtered dataset.
        """

        def check_domain(metadata):
            return domain.is_in_domain(metadata[lat_col], metadata[lon_col])

        mask = self.metadata.apply(check_domain, axis=1)
        self.filter(mask, inplace=inplace)

    def region_filter_source(self, domain, inplace=True):
        """
        Convenience method for region filtering by source location.
        """
        self.region_filter(
            domain,
            lat_col="source_latitude_deg",
            lon_col="source_longitude_deg",
            inplace=inplace,
        )

    def region_filter_receiver(self, domain, inplace=True):
        """
        Convenience method for region filtering by receiver location.
        """
        self.region_filter(
            domain,
            lat_col="station_latitude_deg",
            lon_col="station_longitude_deg",
            inplace=inplace,
        )

    def plot_map(self, res="110m", connections=False, **kwargs):
        """
        Plots the dataset onto a map using the Mercator projection. Requires a cartopy installation.

        :param res: Resolution for cartopy features, defaults to 110m.
        :type res: str, optional
        :param connections: If true, plots lines connecting sources and stations. Defaults to false.
        :type connections: bool, optional
        :param kwargs: Plotting kwargs that will be passed to matplotlib plot. Args need to be prefixed with
                       `sta_`, `ev_` and `conn_` to address stations, events or connections.
        :return: A figure handle for the created figure.
        """
        fig = plt.figure(figsize=(15, 10))
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Plotting the data set requires cartopy. "
                "Please install cartopy, e.g., using conda."
            )

        ax = fig.add_subplot(111, projection=ccrs.Mercator())

        ax.coastlines(res)
        land_50m = cfeature.NaturalEarthFeature(
            "physical", "land", res, edgecolor="face", facecolor=cfeature.COLORS["land"]
        )
        ax.add_feature(land_50m)

        def prefix_dict(kws, prefix):
            return {
                k[len(prefix) :]: v
                for k, v in kws.items()
                if k[: len(prefix)] == prefix
            }

        lines_kws = {
            "marker": "",
            "linestyle": "-",
            "color": "grey",
            "alpha": 0.5,
            "linewidth": 0.5,
        }
        lines_kws.update(prefix_dict(kwargs, "conn_"))

        station_kws = {"marker": "^", "color": "k", "linestyle": "", "ms": 10}
        station_kws.update(prefix_dict(kwargs, "sta_"))

        event_kws = {"marker": ".", "color": "r", "linestyle": ""}
        event_kws.update(prefix_dict(kwargs, "ev_"))

        # Plot connecting lines
        if connections:
            station_source_pairs = self.metadata[
                [
                    "station_longitude_deg",
                    "station_latitude_deg",
                    "source_longitude_deg",
                    "source_latitude_deg",
                ]
            ].values
            for row in station_source_pairs:
                ax.plot(
                    [row[0], row[2]],
                    [row[1], row[3]],
                    transform=ccrs.Geodetic(),
                    **lines_kws,
                )

        # Plot stations
        station_locations = np.unique(
            self.metadata[["station_longitude_deg", "station_latitude_deg"]].values,
            axis=0,
        )
        ax.plot(
            station_locations[:, 0],
            station_locations[:, 1],
            transform=ccrs.PlateCarree(),
            **station_kws,
        )

        # Plot events
        source_locations = np.unique(
            self.metadata[["source_longitude_deg", "source_latitude_deg"]].values,
            axis=0,
        )
        ax.plot(
            source_locations[:, 0],
            source_locations[:, 1],
            transform=ccrs.PlateCarree(),
            **event_kws,
        )

        # Gridlines
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
        gl.top_labels = False
        gl.left_labels = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

        fig.suptitle(self.name)

        return fig

    def get_split(self, split):
        """
        Get a specific split of the dataset.
        获取数据集的特定划分。
        """
        if "split" not in self.metadata.columns:
            raise ValueError("Split requested but no split defined in metadata")
        mask = (self.metadata["split"] == split).values
        return self.filter(mask, inplace=False)

    def train(self):
        """
        Get the training split.
        获取训练集。
        """
        return self.get_split("train")

    def dev(self):
        """
        Get the development split.
        获取开发集。
        """
        return self.get_split("dev")

    def test(self):
        """
        Get the test split.
        获取测试集。
        """
        return self.get_split("test")

    def train_dev_test(self):
        """
        Get training, development, and test splits.
        获取训练集、开发集和测试集。
        """
        return self.train(), self.dev(), self.test()

    def _evict_cache(self):
        """
        Evict waveforms from cache that are no longer in the metadata.
        从缓存中驱逐不再在元数据中的波形。
        """
        existing_keys = defaultdict(set)
        if self.cache == "full":
            block_names = self._metadata["trace_name"].apply(lambda x: x.split("$")[0])
            chunks = self._metadata["trace_chunk"]
            for chunk, block in zip(chunks, block_names):
                existing_keys[chunk].add(block)
        elif self.cache == "trace":
            trace_names = self._metadata["trace_name"]
            chunks = self._metadata["trace_chunk"]
            for chunk, trace in zip(chunks, trace_names):
                existing_keys[chunk].add(trace)

        for chunk in self._waveform_cache.keys():
            delete_keys = []
            for key in self._waveform_cache[chunk].keys():
                if key not in existing_keys[chunk]:
                    delete_keys.append(key)
            for key in delete_keys:
                del self._waveform_cache[chunk][key]

    def __getitem__(self, item):
        """
        Get metadata column.
        获取元数据列。
        """
        if not isinstance(item, str):
            raise TypeError("Can only use strings to access metadata parameters")
        return self._metadata[item]

    def __len__(self):
        """
        Return the number of traces.
        返回迹线数量。
        """
        return len(self._metadata)

    def get_sample(self, idx, sampling_rate=None):
        """
        Get a sample (waveform and metadata) by index.
        通过索引获取样本（波形和元数据）。
        """
        if self._metadata_lookup is None:
            metadata = self.metadata.iloc[idx].to_dict()
        else:
            metadata = copy.deepcopy(self._metadata_lookup[idx])

        sampling_rate = self._get_sample_unify_sampling_rate(metadata, sampling_rate)
        load_metadata = {k: [v] for k, v in metadata.items()}
        waveforms = self._get_waveforms_from_load_metadata(load_metadata, sampling_rate)

        batch_dimension = list(self.dimension_order).index("N")
        waveforms = np.squeeze(waveforms, axis=batch_dimension)

        dimension_order = list(self.dimension_order)
        del dimension_order[dimension_order.index("N")]
        sample_dimension = dimension_order.index("W")
        metadata["trace_npts"] = waveforms.shape[sample_dimension]

        return waveforms, metadata

    def _get_sample_unify_sampling_rate(
        self, metadata: dict[str, Any], sampling_rate: Optional[float]
    ):
        """
        Unify sampling rate for a single sample.
        统一单个样本的采样率。
        """
        if sampling_rate is None:
            sampling_rate = self.sampling_rate
        if sampling_rate is not None:
            source_sampling_rate = metadata["trace_sampling_rate_hz"]
            if np.isnan(source_sampling_rate).any():
                raise ValueError("Tried resampling trace with unknown sampling rate.")
            else:
                resampling_factor = sampling_rate / np.asarray(source_sampling_rate)
                for key in metadata.keys():
                    if key.endswith("_sample"):
                        try:
                            metadata[key] = (
                                np.asarray(metadata[key]) * resampling_factor
                            )
                        except TypeError:
                            pass

                metadata["trace_source_sampling_rate_hz"] = np.asarray(
                    source_sampling_rate
                )
                metadata["trace_sampling_rate_hz"] = (
                    np.ones_like(metadata["trace_sampling_rate_hz"]) * sampling_rate
                )
                metadata["trace_dt_s"] = 1.0 / metadata["trace_sampling_rate_hz"]
        else:
            metadata["trace_source_sampling_rate_hz"] = np.asarray(
                metadata["trace_sampling_rate_hz"]
            )

        return sampling_rate

    def get_waveforms(self, idx=None, mask=None, sampling_rate=None):
        """
        Get waveforms for specified indices or mask.
        获取指定索引或掩码的波形。
        """
        squeeze = False
        if idx is not None:
            if mask is not None:
                raise ValueError("Mask can not be used jointly with idx.")
            if not isinstance(idx, Iterable):
                idx = [idx]
                squeeze = True

            if self._metadata_lookup is None:
                load_metadata = self._metadata.iloc[idx].to_dict("list")
            else:
                load_metadata = [self._metadata_lookup[i] for i in idx]
                load_metadata = self._pack_metadata(load_metadata)
        else:
            if mask is not None:
                load_metadata = self._metadata[mask].to_dict("list")
            else:
                load_metadata = self._metadata.to_dict("list")

        if sampling_rate is None:
            sampling_rate = self.sampling_rate

        load_metadata["trace_source_sampling_rate_hz"] = load_metadata[
            "trace_sampling_rate_hz"
        ]
        waveforms = self._get_waveforms_from_load_metadata(load_metadata, sampling_rate)

        if squeeze:
            batch_dimension = list(self.dimension_order).index("N")
            waveforms = np.squeeze(waveforms, axis=batch_dimension)

        return waveforms

    def _get_waveforms_from_load_metadata(
        self, load_metadata, sampling_rate, pack=True
    ):
        """
        Load waveforms based on metadata.
        根据元数据加载波形。
        """
        waveforms = {}
        chunks, metadata_paths, waveforms_path = self._chunks_with_paths()

        segments = [
            (trace_name, chunk, float(trace_sampling_rate), trace_component_order)
            for trace_name, chunk, trace_sampling_rate, trace_component_order in zip(
                load_metadata["trace_name"],
                load_metadata["trace_chunk"],
                load_metadata["trace_source_sampling_rate_hz"],
                load_metadata["trace_component_order"],
            )
        ]

        with LoadingContext(chunks, waveforms_path) as context:
            for segment in segments:
                if segment in waveforms:
                    continue
                trace_name, chunk, trace_sampling_rate, trace_component_order = segment
                waveforms[segment] = self._get_single_waveform(
                    trace_name,
                    chunk,
                    context=context,
                    target_sampling_rate=sampling_rate,
                    source_sampling_rate=trace_sampling_rate,
                    source_component_order=trace_component_order,
                )

        if pack:
            waveforms = [waveforms[segment] for segment in segments]
            waveforms = pad_packed_sequence(waveforms)
            waveforms = waveforms.transpose(*self._dimension_mapping)
        else:
            waveforms = {
                k: self._transpose_single_waveform(wv) for k, wv in waveforms.items()
            }
            waveforms = [waveforms[segment] for segment in segments]

        return waveforms

    def _transpose_single_waveform(self, wv: np.ndarray) -> np.ndarray:
        """
        Transpose a single waveform to match dimension order.
        转置单个波形以匹配维度顺序。
        """
        squeeze_axis = self.dimension_order.index("N")
        wv = np.expand_dims(wv, 0)
        wv = wv.transpose(*self._dimension_mapping)
        wv = np.squeeze(wv, squeeze_axis)
        return wv

    @staticmethod
    def _pack_metadata(metadata):
        """
        Pack a list of metadata dictionaries into a single dictionary of lists.
        将元数据字典列表打包成一个列表字典。
        """
        return {key: [m[key] for m in metadata] for key in metadata[0].keys()}

    def _get_single_waveform(
        self,
        trace_name,
        chunk,
        context,
        target_sampling_rate=None,
        source_sampling_rate=None,
        source_component_order=None,
    ):
        """
        Get a single waveform from the file or cache.
        从文件或缓存中获取单个波形。
        """
        trace_name = str(trace_name)

        if trace_name in self._waveform_cache[chunk]:
            waveform = self._waveform_cache[chunk][trace_name]
        else:
            if trace_name.find("$") != -1:
                block_name, location = trace_name.split("$")
            else:
                block_name, location = trace_name, ":"

            location = self._parse_location(location)

            if block_name in self._waveform_cache[chunk]:
                waveform = self._waveform_cache[chunk][block_name][location]
            else:
                g_data = context[chunk]["data"]
                block = g_data[block_name]
                if self.cache == "full":
                    block = block[()]
                    self._waveform_cache[chunk][block_name] = block
                    waveform = block[location]
                else:
                    waveform = block[location]
                    if self.cache == "trace":
                        self._waveform_cache[chunk][trace_name] = waveform

        if target_sampling_rate is not None:
            if np.isnan(source_sampling_rate):
                raise ValueError("Tried resampling trace with unknown sampling rate.")
            else:
                waveform = self._resample(
                    waveform, target_sampling_rate, source_sampling_rate
                )

        if source_component_order is not None:
            component_dimension = list(self._data_format["dimension_order"]).index("C")
            component_mapping = self._component_mapping[source_component_order]

            if waveform.shape[component_dimension] == max(component_mapping):
                pad = []
                for i in range(waveform.ndim):
                    if i == component_dimension:
                        pad += [(0, 1)]
                    else:
                        pad += [(0, 0)]
                waveform = np.pad(waveform, pad, "constant", constant_values=0)

            waveform = waveform.take(component_mapping, axis=component_dimension)

        return waveform

    @staticmethod
    def _parse_location(location):
        """
        Parse location string into slices.
        将位置字符串解析为切片。
        """
        location = location.replace(" ", "")
        slices = []
        dim_slices = location.split(",")

        def int_or_none(s):
            if s == "":
                return None
            else:
                return int(s)

        for dim_slice in dim_slices:
            parts = dim_slice.split(":")
            if len(parts) == 1:
                idx = int_or_none(parts[0])
                slices.append(idx)
            elif len(parts) == 2:
                start = int_or_none(parts[0])
                stop = int_or_none(parts[1])
                slices.append(slice(start, stop))
            elif len(parts) == 3:
                start = int_or_none(parts[0])
                stop = int_or_none(parts[1])
                step = int_or_none(parts[2])
                slices.append(slice(start, stop, step))
            else:
                raise ValueError(f"Invalid location string {location}")

        return tuple(slices)

    def _resample(self, waveform, target_sampling_rate, source_sampling_rate, eps=1e-4):
        """
        Resample waveform to target sampling rate.
        将波形重采样到目标采样率。
        """
        try:
            sample_axis = list(self._data_format["dimension_order"]).index("W")
        except (KeyError, ValueError):
            sample_axis = None

        if 1 - eps < target_sampling_rate / source_sampling_rate < 1 + eps:
            return waveform
        else:
            if sample_axis is None:
                raise ValueError(
                    "Trace can not be resampled because of missing or incorrect dimension order."
                )

            if waveform.shape[sample_axis] == 0:
                return waveform

            if (source_sampling_rate % target_sampling_rate) < eps:
                q = int(source_sampling_rate // target_sampling_rate)
                return scipy.signal.decimate(waveform, q, axis=sample_axis)
            else:
                num = int(
                    waveform.shape[sample_axis]
                    * target_sampling_rate
                    / source_sampling_rate
                )
                return scipy.signal.resample(waveform, num, axis=sample_axis)

class AbstractBenchmarkDataset(ABC):
    """
    Base class for benchmark datasets.
    基准数据集的基类。
    """
    _files: list[str] = []

    def __init__(
        self,
        chunks: Optional[list[str]] = None,
        citation: Optional[str] = None,
        license: Optional[str] = None,
        force: bool = False,
        wait_for_file: bool = False,
        repository_lookup: bool = False,
        compile_from_source: bool = False,
        download_kwargs: dict[str, Any] = None,
        **kwargs,
    ):
        self._name = self._name_internal()
        self._citation = citation
        self._license = license
        self.path.mkdir(exist_ok=True, parents=True)

        if download_kwargs is None:
            download_kwargs = {}

        if chunks is None:
            chunks = self.available_chunks(force=force, wait_for_file=wait_for_file)
        else:
            if any(chunk not in self.available_chunks() for chunk in chunks):
                raise ValueError(
                    f"The dataset does not contain the following chunks: "
                    f"{[chunk for chunk in chunks if chunk not in self.available_chunks()]}"
                )

        for chunk in chunks:
            def download_callback(files):
                chunk_str = f'Chunk "{chunk}" of ' if chunk != "" else ""
                logger.warning(
                    f"{chunk_str}Dataset {self.name} not in cache."
                )
                successful_repository_download = False
                if repository_lookup:
                    logger.warning(
                        "Trying to download preprocessed version from remote repository."
                    )
                    try:
                        self._download_preprocessed(files, chunk=chunk)
                        successful_repository_download = True
                    except ValueError:
                        pass

                if not successful_repository_download and compile_from_source:
                    logger.warning(
                        f"{chunk_str}Dataset {self.name} not in remote repository. "
                        f"Starting download and conversion from source."
                    )

                    self._download_dataset_wrapper(
                        files, chunk=chunk, **download_kwargs
                    )

            files = [self.path / file.replace("$CHUNK", chunk) for file in self._files]

            util.callback_if_uncached(
                files, download_callback, force=force, wait_for_file=wait_for_file
            )

        super().__init__(chunks=chunks, **kwargs)

    @property
    def citation(self):
        return self._citation

    @property
    def license(self):
        return self._license

    @classmethod
    def _path_internal(cls):
        return Path(CONFIG["cache_data_root"], cls._name_internal().lower())

    @property
    def path(self):
        return self._path_internal()

    @classmethod
    def _name_internal(cls):
        return cls.__name__

    @property
    def name(self):
        return self._name_internal()

    @classmethod
    def _remote_path(cls):
        return urljoin(CONFIG["remote_data_root"], cls._name_internal())

    @classmethod
    def available_chunks(cls, force: bool = False, wait_for_file: bool = False):
        test_file = cls._files[0]

        if (cls._path_internal() / test_file.replace("$CHUNK", "")).is_file():
            chunks = [""]
        else:
            def chunks_callback(file):
                remote_chunks_path = os.path.join(cls._remote_path(), "chunks")
                try:
                    util.download_http(
                        remote_chunks_path, file, progress_bar=False, precheck_timeout=0
                    )
                except ValueError:
                    logger.info("Found no remote chunk file. Progressing.")

            chunks_path = cls._path_internal() / "chunks"
            util.callback_if_uncached(
                chunks_path,
                chunks_callback,
                force=force,
                wait_for_file=wait_for_file,
            )

            if chunks_path.is_file():
                with open(chunks_path, "r") as f:
                    chunks = [x for x in f.read().split("\n") if x.strip()]
            else:
                chunks = [""]

        return chunks

    def _download_preprocessed(self, output_files: list[Path], chunk: str):
        self.path.mkdir(parents=True, exist_ok=True)

        remote_base = self._remote_path()
        for file_name, output_file in zip(self._files, output_files):
            file_name = file_name.replace("$CHUNK", chunk)
            remote_file_path = os.path.join(remote_base, file_name)

            util.download_http(
                remote_file_path, output_file, desc=f"Downloading {file_name}"
            )

    def _download_dataset_wrapper(self, files: list[Path], chunk: str, **kwargs):
        tmp_download_args = self.add_chunk_to_download_args(chunk, kwargs)
        self._download_dataset(files, **tmp_download_args)

    def add_chunk_to_download_args(self, chunk, kwargs):
        download_dataset_parameters = inspect.signature(
            self._download_dataset
        ).parameters
        if "chunk" not in download_dataset_parameters and chunk != "":
            raise ValueError(
                "Data set seems not to support chunking, but chunk provided."
            )
        tmp_download_args = copy.copy(kwargs)
        if "chunk" in download_dataset_parameters:
            tmp_download_args["chunk"] = chunk
        return tmp_download_args

    def _download_dataset(self, files: list[Path], chunk: str, **kwargs):
        raise NotImplementedError(
            "This dataset does not implement a conversion from source."
        )


class WaveformBenchmarkDataset(AbstractBenchmarkDataset, WaveformDataset, ABC):
    """
    Base class for waveform benchmark datasets.
    波形基准数据集的基类。
    """
    _files = ["waveforms$CHUNK.hdf5", "metadata$CHUNK.csv"]

    def _download_dataset_wrapper(self, files, chunk, **kwargs):
        tmp_download_args = self.add_chunk_to_download_args(chunk, kwargs)
        with WaveformDataWriter(*files) as writer:
            self._download_dataset(writer, **tmp_download_args)

    @abstractmethod
    def _download_dataset(self, writer, chunk, **kwargs):
        raise NotImplementedError(
            "This dataset does not implement a conversion from source."
        )

class BenchmarkDataset(WaveformBenchmarkDataset):
    """
    Deprecated alias for WaveformBenchmarkDataset.
    """
    def __init__(self, *args, **kwargs):
        logger.warning(
            "The class BenchmarkDataset is deprecated. Use WaveformBenchmarkDataset instead."
        )
        super().__init__(*args, **kwargs)

class Bucketer(ABC):
    @abstractmethod
    def get_bucket(self, metadata, waveform):
        return ""

class GeometricBucketer(Bucketer):
    def __init__(self, minbucket, factor, splits=True, track_channels=False, axis=-1):
        self.minbucket = minbucket
        self.factor = factor
        self.splits = splits
        self.track_channels = track_channels
        self.axis = axis

    def get_bucket(self, metadata, waveform):
        n_samples = waveform.shape[self.axis]
        if n_samples < self.minbucket:
            bucket_id = 0
        else:
            bucket_id = int(np.log(n_samples / self.minbucket) / np.log(self.factor)) + 1
        
        bucket = str(bucket_id)
        
        if self.splits and "split" in metadata:
            bucket += "_" + str(metadata["split"])
            
        if self.track_channels:
            shape = list(waveform.shape)
            del shape[self.axis]
            bucket += "_" + str(shape)
            
        return bucket

class WaveformDataWriter:
    """
    Writes waveform datasets in SeisPolarity format.
    """
    def __init__(self, waveforms_path, metadata_path, bucketer=None):
        self.waveforms_path = Path(waveforms_path)
        self.metadata_path = Path(metadata_path)
        self.bucketer = bucketer
        
        self._metadata = []
        self._waveforms_file = h5py.File(self.waveforms_path, "w")
        self._data_group = self._waveforms_file.create_group("data")
        self._data_format_group = self._waveforms_file.create_group("data_format")
        
        # Default data format
        self.data_format = {
            "dimension_order": "CW",
            "component_order": "ZNE",
            "sampling_rate": 100,
            "measurement": "velocity",
            "unit": "counts",
        }
        
        self._trace_count = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if self._metadata:
            df = pd.DataFrame(self._metadata)
            df.to_csv(self.metadata_path, index=False)
        
        for key, value in self.data_format.items():
            if key not in self._data_format_group:
                self._data_format_group.create_dataset(key, data=value)
                
        self._waveforms_file.close()

    def add_trace(self, metadata, waveform):
        bucket = ""
        if self.bucketer:
            bucket = self.bucketer.get_bucket(metadata, waveform)
            
        if bucket not in self._data_group:
            self._data_group.create_group(bucket)
            
        trace_name = metadata.get("trace_name", f"trace_{self._trace_count}")
        metadata["trace_name"] = trace_name
        
        # We use trace_name as block_name
        self._data_group.create_dataset(trace_name, data=waveform)
        
        self._metadata.append(metadata)
        self._trace_count += 1

    def set_total(self, total):
        pass 

try:
    from typing import TypedDict, NotRequired
except ImportError:
    try:
        from typing_extensions import TypedDict, NotRequired
    except ImportError:
        # Fallback if typing_extensions is not installed
        from typing import TypedDict, Optional
        NotRequired = Optional

class EventParameters(TypedDict):
    index: NotRequired[int]
    split: NotRequired[Literal["train", "dev", "test"]]

    source_id: NotRequired[str]
    source_origin_time: NotRequired[str]
    source_origin_uncertainty_sec: NotRequired[float]
    source_latitude_deg: NotRequired[float]
    source_latitude_uncertainty_km: NotRequired[float]
    source_longitude_deg: NotRequired[float]
    source_longitude_uncertainty_km: NotRequired[float]
    source_depth_km: NotRequired[float]
    source_depth_uncertainty_km: NotRequired[float]

    source_magnitude: NotRequired[float]
    source_magnitude_uncertainty: NotRequired[float]
    source_magnitude_type: NotRequired[str]
    source_magnitude_author: NotRequired[str]

    source_focal_mechanism_t_azimuth: NotRequired[float]
    source_focal_mechanism_t_plunge: NotRequired[float]
    source_focal_mechanism_t_length: NotRequired[float]
    source_focal_mechanism_p_azimuth: NotRequired[float]
    source_focal_mechanism_p_plunge: NotRequired[float]
    source_focal_mechanism_p_length: NotRequired[float]
    source_focal_mechanism_n_azimuth: NotRequired[float]
    source_focal_mechanism_n_plunge: NotRequired[float]
    source_focal_mechanism_n_length: NotRequired[float]


class TraceParameters(TypedDict):
    trace_name: NotRequired[str]

    path_back_azimuth_deg: NotRequired[float]
    station_network_code: NotRequired[str]
    station_code: NotRequired[str]
    trace_channel: NotRequired[str]
    station_location_code: NotRequired[str]
    station_latitude_deg: NotRequired[float]
    station_longitude_deg: NotRequired[float]
    station_elevation_m: NotRequired[float]

    trace_sampling_rate_hz: NotRequired[float]
    trace_completeness: NotRequired[float]
    trace_has_spikes: NotRequired[bool]
    trace_start_time: NotRequired[str]

    trace_component_order: NotRequired[str]

class PolarityDataset(WaveformDataset):
    """
    A wrapper around WaveformDataset to support polarity filtering.
    该类是 WaveformDataset 的一个包装器，用于支持极性过滤。
    """
    def __init__(self, path: Union[str, Path], **kwargs):
        super().__init__(path=path, **kwargs)

    def filter_polarity(self, column: str = "trace_polarity", valid_values: Optional[List[str]] = None):
        """
        Filter the dataset to include only traces with valid polarity labels.
        过滤数据集，仅包含具有有效极性标签的迹线。
        """
        if column not in self.metadata.columns:
            candidates = [c for c in self.metadata.columns if "polarity" in c]
            if candidates:
                logger.info(f"Column '{column}' not found. Using '{candidates[0]}' instead.")
                column = candidates[0]
            else:
                logger.warning(f"No polarity column found in metadata. Available columns: {self.metadata.columns}")
                return self

        original_len = len(self)
        if valid_values:
            self.metadata = self.metadata[self.metadata[column].isin(valid_values)]
        else:
            self.metadata = self.metadata[self.metadata[column].notna()]
        
        logger.info(f"Filtered dataset for polarity. {len(self)}/{original_len} traces remain.")
        return self


class MultiWaveformDataset:
    """
    A :py:class:`MultiWaveformDataset` is an ordered collection of :py:class:`WaveformDataset`.
    It exposes mostly the same API as a single :py:class:`WaveformDataset`.

    The constructor checks for compatibility of `dimension_order`, `component_order` and `sampling_rate`.
    The caching strategy of each contained dataset is left unmodified,
    but a warning is issued if different caching schemes are found.

    :param datasets: List of :py:class:`WaveformDataset`.
                     The constructor will create a copy of each dataset using the
                     :py:func:`WaveformDataset.copy` method.
    """

    def __init__(self, datasets):
        if not isinstance(datasets, list) or not all(
            isinstance(x, WaveformDataset) for x in datasets
        ):
            raise TypeError(
                "MultiWaveformDataset expects a list of WaveformDataset as input."
            )

        if len(datasets) == 0:
            raise ValueError("MultiWaveformDatasets need to have at least one member.")

        self._datasets = [dataset.copy() for dataset in datasets]
        self._metadata = pd.concat(x.metadata for x in datasets)

        # Identify dataset
        self._metadata["trace_dataset"] = sum(
            ([dataset.name] * len(dataset) for dataset in self.datasets), []
        )
        self._metadata.reset_index(inplace=True, drop=True)

        self._trace_identification_warning_issued = (
            False  # Traced whether warning for trace name was issued already
        )

        self._homogenize_dataformat(datasets)
        self._grouping = None
        self._homogenize_grouping(datasets)
        self._build_trace_name_to_idx_dict()

    def __add__(self, other):
        if isinstance(other, WaveformDataset):
            return MultiWaveformDataset(self.datasets + [other])
        elif isinstance(other, MultiWaveformDataset):
            return MultiWaveformDataset(self.datasets + other.datasets)
        else:
            raise TypeError(
                "Can only add WaveformDataset and MultiWaveformDataset to MultiWaveformDataset."
            )

    def _homogenize_dataformat(self, datasets):
        """
        Checks if the output data format options agree.
        In case of mismatches, warnings are issued and the format is reset.
        """
        has_split = ["split" in dataset.metadata.columns for dataset in datasets]
        if (
            np.sum(has_split) % len(datasets) != 0
        ):  # Check if all or no dataset has a split
            logger.warning(
                "Combining datasets with and without split. "
                "get_split and all derived methods will never return any samples from "
                "the datasets without split."
            )
        if not self._test_attribute_equal(datasets, "cache"):
            logger.warning(
                "Found inconsistent caching strategies. "
                "This does not cause an error, but is usually unintended."
            )
        if not self._test_attribute_equal(datasets, "sampling_rate"):
            logger.warning(
                "Found mismatching sampling rates between datasets. "
                "Setting sampling rate to None, i.e., deactivating automatic resampling. "
                "You can change the sampling rate for all datasets through "
                "the sampling_rate property."
            )
            self.sampling_rate = None

        if self.sampling_rate is None and len(self) > 0:
            sr = self.metadata["trace_sampling_rate_hz"].values
            q = sr / sr[0]
            if not np.allclose(q, 1):
                logger.warning(
                    "Data set contains mixed sampling rate, but no sampling rate was specified for the dataset."
                    "get_waveforms will return mixed sampling rate waveforms."
                )

        if not self._test_attribute_equal(datasets, "component_order"):
            logger.warning(
                "Found inconsistent component orders. "
                f"Using component order from first dataset ({self.datasets[0].component_order})."
            )
            self.component_order = self.datasets[0].component_order

        if not self._test_attribute_equal(datasets, "dimension_order"):
            logger.warning(
                "Found inconsistent dimension orders. "
                f"Using dimension order from first dataset ({self.datasets[0].dimension_order})."
            )
            self.dimension_order = self.datasets[0].dimension_order

        if not self._test_attribute_equal(datasets, "missing_components"):
            logger.warning(
                "Found inconsistent missing_components. "
                f"Using missing_components from first dataset ({self.datasets[0].missing_components})."
            )
            self.missing_components = self.datasets[0].missing_components

    def _homogenize_grouping(self, datasets):
        groupings = [dataset.grouping for dataset in datasets]
        if any(grouping != groupings[0] for grouping in groupings):
            logger.warning(
                "Found inconsistent groupings. Setting grouping to None."
            )
            self.grouping = None
        else:
            self.grouping = groupings[0]

    @property
    def grouping(self):
        """
        The grouping parameters for the dataset.
        Grouping allows to access metadata and waveforms
        jointly from a set of traces with a common metadata parameter.
        This can for example be used to access all waveforms belong to one event
        and building event based models.
        Setting the grouping parameter defines the output of
        :py:attr:`~groups` and the associated methods.
        `grouping` can be either a single string or a list of strings.
        Each string must be a column in the metadata.
        By default, the grouping is None.
        """
        return self._grouping

    @grouping.setter
    def grouping(self, value):
        self._grouping = value
        if value is None:
            self._groups = None
            self._groups_to_trace_idx = None
        else:
            self._groups_to_trace_idx = self.metadata.groupby(value).groups
            self._groups = list(self._groups_to_trace_idx.keys())
            self._groups_to_group_idx = {
                group: i for i, group in enumerate(self._groups)
            }

    @property
    def groups(self):
        """
        The list of groups as defined by the :py:attr:`~grouping` or `None` if :py:attr:`~grouping` is `None`.
        """
        return copy.copy(
            self._groups
        )  # Return a copy to make the internal groups immutable

    @property
    def datasets(self):
        """
        Datasets contained in MultiWaveformDataset.
        """
        return list(self._datasets)

    @property
    def metadata(self):
        """
        Metadata of the dataset as pandas DataFrame.
        """
        return self._metadata

    @property
    def sampling_rate(self):
        """
        Get or set sampling rate for output
        """
        return self.datasets[0].sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, sampling_rate):
        for dataset in self.datasets:
            dataset.sampling_rate = sampling_rate

    @property
    def metadata_cache(self):
        return self.datasets[0]._metadata_cache

    @metadata_cache.setter
    def metadata_cache(self, val):
        for dataset in self.datasets:
            dataset.metadata_cache = val

    @property
    def dimension_order(self):
        """
        Get or set dimension order for output
        """
        return self.datasets[0].dimension_order

    @dimension_order.setter
    def dimension_order(self, value):
        for dataset in self.datasets:
            dataset.dimension_order = value

    @property
    def missing_components(self):
        """
        Get or set strategy for missing components
        """
        return self.datasets[0].missing_components

    @missing_components.setter
    def missing_components(self, value):
        for dataset in self.datasets:
            dataset.missing_components = value

    @property
    def component_order(self):
        """
        Get or set component order
        """
        return self.datasets[0].component_order

    @component_order.setter
    def component_order(self, value):
        for dataset in self.datasets:
            dataset.component_order = value

    @property
    def cache(self):
        """
        Get or set cache strategy
        """
        if self._test_attribute_equal(self.datasets, "cache"):
            return self.datasets[0].cache
        else:
            return "Inconsistent"

    @cache.setter
    def cache(self, value):
        for dataset in self.datasets:
            dataset.cache = value

    @staticmethod
    def _test_attribute_equal(datasets, attribute):
        """
        Checks whether the given attribute is equal in all datasets.

        :param datasets: List of WaveformDatasets
        :param attribute: attribute as string
        :return: True if attribute is identical in all datasets, false otherwise
        """
        attribute_list = [dataset.__getattribute__(attribute) for dataset in datasets]
        return all(x == attribute_list[0] for x in attribute_list)

    def __getitem__(self, item):
        """
        Only accepts string inputs. Returns respective column from metadata
        """
        if not isinstance(item, str):
            raise TypeError("Can only use strings to access metadata parameters")
        return self.metadata[item]

    def __len__(self):
        return len(self.metadata)

    def get_sample(self, idx, *args, **kwargs):
        """
        Wraps :py:func:`WaveformDataset.get_sample`

        :param idx: Index of the sample
        :param args: passed to parent function
        :param kwargs: passed to parent function
        :return: Return value of parent function
        """
        dataset_idx, local_idx = self._resolve_idx(idx)
        return self.datasets[dataset_idx].get_sample(local_idx, *args, **kwargs)

    def get_waveforms(self, idx=None, mask=None, **kwargs):
        """
        Collects waveforms and returns them as an array.

        :param idx: Idx or list of idx to obtain waveforms for
        :type idx: int, list[int]
        :param mask: Binary mask on the metadata, indicating which traces should be returned.
                     Can not be used jointly with idx.
        :type mask: np.ndarray[bool]
        :param kwargs: Passed to :py:func:`WaveformDataset.get_waveforms`
        :return: Waveform array with dimensions ordered according to dimension_order e.g. default 'NCW'
                 (number of traces, number of components, record samples). If the number record samples
                 varies between different entries, all entries are padded to the maximum length.
        :rtype: np.ndarray
        """
        squeeze = False
        waveforms = []
        if idx is not None:
            if mask is not None:
                raise ValueError("Mask can not be used jointly with idx.")
            if not isinstance(idx, Iterable):
                idx = [idx]
                squeeze = True

            for i in idx:
                dataset_idx, local_idx = self._resolve_idx(i)
                waveforms.append(
                    self.datasets[dataset_idx].get_waveforms(idx=[local_idx], **kwargs)
                )
        else:
            if mask is None:
                mask = np.ones(len(self), dtype=bool)

            submasks = self._split_mask(mask)

            for submask, dataset in zip(submasks, self.datasets):
                if submask.any():
                    waveforms.append(dataset.get_waveforms(mask=submask, **kwargs))

        if self.missing_components == "ignore":
            # Check consistent number of components
            component_dimension = list(self.dimension_order).index("C")
            n_components = np.array([x.shape[component_dimension] for x in waveforms])
            if (n_components[0] != n_components).any():
                raise ValueError(
                    "Requested traces with mixed number of components. "
                    "Change missing_components or request traces separately."
                )

        batch_dimension = list(self.dimension_order).index("N")
        waveforms = self._pad_pack_along_axis(waveforms, axis=batch_dimension)

        if squeeze:
            waveforms = np.squeeze(waveforms, axis=batch_dimension)

        return waveforms

    @staticmethod
    def _pad_pack_along_axis(seq, axis):
        """
        Concatenate arrays along axis. In each but the given axis, all input arrays are padded with zeros
        to the maximum size of any input array.

        :param seq: List of arrays
        :param axis: Axis along which to concatenate
        :return:
        """
        max_size = np.array(
            [max([x.shape[i] for x in seq]) for i in range(seq[0].ndim)]
        )

        new_seq = []
        for i, elem in enumerate(seq):
            d = max_size - np.array(elem.shape)
            if (d != 0).any():
                pad = [(0, d_dim) for d_dim in d]
                pad[axis] = (0, 0)
                new_seq.append(np.pad(elem, pad, "constant", constant_values=0))
            else:
                new_seq.append(elem)

        return np.concatenate(new_seq, axis=axis)

    def filter(self, mask, inplace=True):
        """
        Filters dataset, similar to :py:func:`WaveformDataset.filter`.

        :param mask: Boolean mask to apple to metadata.
        :type mask: masked-array
        :param inplace: If true, filter inplace.
        :type inplace: bool
        :return: None if filter=true, otherwise the filtered dataset.
        """
        submasks = self._split_mask(mask)
        if inplace:
            for submask, dataset in zip(submasks, self.datasets):
                dataset.filter(submask, inplace=True)
            # Calculate new metadata
            self._metadata = pd.concat(x.metadata for x in self.datasets)
            self._build_trace_name_to_idx_dict()
            self.grouping = self.grouping  # Rebuild groups

        else:
            return MultiWaveformDataset(
                [
                    dataset.filter(submask, inplace=False)
                    for submask, dataset in zip(submasks, self.datasets)
                ]
            )

    def _resolve_idx(self, idx):
        """
        Translates an index into the dataset index and the index within the dataset

        :param idx: Index of the sample
        :return: Dataset index, index within the dataset
        """
        borders = np.cumsum([len(x) for x in self.datasets])
        if idx < 0:
            idx += len(self)

        if idx >= len(self) or idx < 0:
            raise IndexError("Sample index out out range.")

        dataset_idx = np.argmax(idx < borders)
        local_idx = (
            idx - borders[dataset_idx] + len(self.datasets[dataset_idx])
        )  # Resolve the negative indexing

        return dataset_idx, local_idx

    def _split_mask(self, mask):
        """
        Split one mask for the full dataset into several masks for each subset

        :param mask: Mask for the full dataset
        :return: List of masks, one for each dataset
        """
        if not len(mask) == len(self):
            raise ValueError("Mask does not match dataset.")

        masks = []
        p = 0
        for dataset in self.datasets:
            masks.append(mask[p : p + len(dataset)])
            p += len(dataset)

        return masks

    def preload_waveforms(self, *args, **kwargs):
        """
        Calls :py:func:`WaveformDataset.preload_waveforms` for all member datasets with the provided arguments.
        """
        for dataset in self.datasets:
            dataset.preload_waveforms(*args, **kwargs)

    def _get_group_internal(self, idx, return_metadata, sampling_rate=None):
        """
        This function does *not* use the metadata_lookup.
        """
        self._verify_grouping_defined()

        group = self._groups[idx]
        idx = self._groups_to_trace_idx[group]

        metadata = self.metadata.iloc[idx].to_dict("list")

        sampling_rate = self._get_sample_unify_sampling_rate(metadata, sampling_rate)

        lookups = defaultdict(list)
        for query_idx, i in enumerate(idx):
            dataset_idx, _ = self._resolve_idx(i)
            lookups[dataset_idx].append(query_idx)

        waveforms_pre = {}
        for dataset_idx, query_idx in lookups.items():
            sub_metadata = {k: np.asarray(v)[query_idx] for k, v in metadata.items()}
            waveforms_pre[dataset_idx] = self.datasets[
                dataset_idx
            ]._get_waveforms_from_load_metadata(sub_metadata, sampling_rate, pack=False)
        waveforms_pre = {k: iter(v) for k, v in waveforms_pre.items()}

        waveforms = []
        for i in idx:
            dataset_idx, local_idx = self._resolve_idx(i)
            waveforms.append(next(waveforms_pre[dataset_idx]))

        self._calculate_trace_npts_group(metadata, waveforms)

        if return_metadata:
            return waveforms, metadata
        else:
            return waveforms

    # Copy compatible parts from WaveformDataset
    region_filter = WaveformDataset.region_filter
    region_filter_source = WaveformDataset.region_filter_source
    region_filter_receiver = WaveformDataset.region_filter_receiver
    plot_map = WaveformDataset.plot_map
    get_split = WaveformDataset.get_split
    train = WaveformDataset.train
    dev = WaveformDataset.dev
    test = WaveformDataset.test
    train_dev_test = WaveformDataset.train_dev_test
    _build_trace_name_to_idx_dict = WaveformDataset._build_trace_name_to_idx_dict
    get_idx_from_trace_name = WaveformDataset.get_idx_from_trace_name
    get_group_idx_from_params = WaveformDataset.get_group_idx_from_params
    _verify_grouping_defined = WaveformDataset._verify_grouping_defined
    get_group_waveforms = WaveformDataset.get_group_waveforms
    get_group_samples = WaveformDataset.get_group_samples
    get_group_size = WaveformDataset.get_group_size
    _get_sample_unify_sampling_rate = WaveformDataset._get_sample_unify_sampling_rate
    _calculate_trace_npts_group = WaveformDataset._calculate_trace_npts_group
