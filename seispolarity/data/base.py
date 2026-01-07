from __future__ import annotations
import copy
import logging
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional, List, Union, Tuple, Literal
from urllib.parse import urljoin
import os

import h5py
import numpy as np
import pandas as pd
import scipy.signal
from tqdm import tqdm

import seispolarity.util as util
from seispolarity.util import pad_packed_sequence

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

class LoadingContext:
    """
    The LoadingContext is a dict of pointers to the hdf5 files for the chunks.
    It is an easy way to manage opening and closing of file pointers when required.
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
    This class is the base class for waveform datasets provided by SeisPolarity.
    It handles loading metadata (CSV) and waveforms (HDF5).
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
                
                # Check for empty metadata files
                if metadata_path.stat().st_size == 0:
                     logger.warning(f"Metadata file {metadata_path} is empty, skipping.")
                     continue

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
        available = self.available_chunks(path)
        if any(chunk not in available for chunk in chunks):
            # Only raise if we are sure it's missing (files don't exist)
            # Users might be defining chunks that are about to be downloaded
            pass 

    def __str__(self):
        return f"{self._name} - {len(self)} traces"

    def copy(self):
        other = copy.copy(self)
        other._metadata = self._metadata.copy()
        other._waveform_cache = defaultdict(dict)
        for key in self._waveform_cache.keys():
            other._waveform_cache[key] = copy.copy(self._waveform_cache[key])
        return other

    @property
    def metadata(self):
        return self._metadata
    
    @metadata.setter
    def metadata(self, value):
        self._metadata = value

    @property
    def name(self):
        return self._name

    @property
    def cache(self):
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
        return self._metadata_cache

    @metadata_cache.setter
    def metadata_cache(self, val):
        self._metadata_cache = val
        self._rebuild_metadata_cache()

    @property
    def path(self):
        if self._path is None:
            raise ValueError("Path is None. Can't create data set without a path.")
        return Path(self._path)

    @property
    def data_format(self):
        return dict(self._data_format)

    @property
    def dimension_order(self):
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
        return self._component_order

    @component_order.setter
    def component_order(self, value):
        if value is None:
            value = CONFIG["component_order"]
            # logger.warning(f"Output component_order not specified, defaulting to '{value}'.")

        if self.missing_components is not None:
            self._component_mapping = {}
            if "trace_component_order" in self.metadata.columns:
                for source_order in self.metadata["trace_component_order"].unique():
                    self._component_mapping[source_order] = self._get_component_mapping(
                        source_order, value
                    )
            else:
                source_order = self.data_format.get("component_order", "ZNE")
                self._component_mapping[source_order] = self._get_component_mapping(
                    source_order, value
                )

        self._component_order = value
        
    @property
    def grouping(self):
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
        return copy.copy(self._groups)

    @property
    def chunks(self):
        if self._chunks is None:
            self._chunks = self.available_chunks(self.path)
        return self._chunks

    @staticmethod
    def available_chunks(path):
        path = Path(path)
        chunks_path = path / "chunks"
        if chunks_path.is_file():
            with open(chunks_path, "r") as f:
                chunks = [x for x in f.read().split("\n") if x.strip()]
        else:
            chunks = []
        
        if not chunks:
            if (path / "waveforms.hdf5").is_file():
                chunks = [""]
            else:
                try:
                    metadata_files = set(
                        [x.name[8:-4] for x in path.iterdir()
                            if x.name.startswith("metadata") and x.name.endswith(".csv")]
                    )
                    waveform_files = set(
                        [x.name[9:-5] for x in path.iterdir()
                            if x.name.startswith("waveforms") and x.name.endswith(".hdf5")]
                    )
                    chunks = sorted(list(metadata_files & waveform_files))
                except FileNotFoundError:
                    chunks = []

        return sorted(chunks)

    def _rebuild_metadata_cache(self):
        if self.metadata_cache:
            self._metadata_lookup = list(self._metadata.apply(lambda x: x.to_dict(), axis=1))
        else:
            self._metadata_lookup = None

    def _unify_sampling_rate(self, eps=1e-4):
        if "trace_sampling_rate_hz" in self.metadata.columns:
            if "trace_dt_s" in self.metadata.columns:
                mask = np.isnan(self.metadata["trace_sampling_rate_hz"].values)
                if np.any(mask):
                    self._metadata.loc[mask, "trace_sampling_rate_hz"] = (
                        1 / self.metadata.loc[mask, "trace_dt_s"]
                    )
        elif "trace_dt_s" in self.metadata.columns:
            self.metadata["trace_sampling_rate_hz"] = 1 / self.metadata["trace_dt_s"]
        elif "sampling_rate" in self.data_format:
            self._metadata["trace_sampling_rate_hz"] = self.data_format["sampling_rate"]
        else:
            # logger.warning("Sampling rate not specified in data set, setting to NaN.")
            self._metadata["trace_sampling_rate_hz"] = np.nan

    def _get_component_mapping(self, source, target):
        if (isinstance(source, float) and np.isnan(source)) or not source:
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
        return mapping

    @staticmethod
    def _get_dimension_mapping(source, target):
        source = list(source)
        target = list(target)
        try:
            mapping = [source.index(t) for t in target]
        except ValueError:
             raise ValueError(f"Could not determine mapping {source} -> {target}.")
        return mapping

    def _chunks_with_paths(self):
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
            
            self._chunks_with_paths_cache = (self.chunks, metadata_paths, waveform_paths)
        return self._chunks_with_paths_cache

    def _verify_dataset(self):
        for chunk, metadata_path, waveform_path in zip(*self._chunks_with_paths()):
            if not metadata_path.is_file():
                # Allow missing files if we are about to download them in subclass
                pass
            if not waveform_path.is_file():
                pass

    def _read_data_format(self):
        data_format = None
        # Only read if files actually exist
        paths = self._chunks_with_paths()[2]
        existing_paths = [p for p in paths if p.is_file()]
        
        if not existing_paths:
             return {"dimension_order": "CW"}

        for waveform_file in existing_paths:
            with h5py.File(waveform_file, "r") as f_wave:
                try:
                    g_data_format = f_wave.get("data_format", {})
                    # Handle both cases: dict read or group read
                    if hasattr(g_data_format, "keys"):
                        tmp_data_format = {k: g_data_format[k][()] for k in g_data_format.keys()}
                    else:
                        tmp_data_format = {}
                except Exception:
                    tmp_data_format = {}
            if data_format is None:
                data_format = tmp_data_format
        
        if data_format:
            for key in data_format.keys():
                if isinstance(data_format[key], bytes):
                    data_format[key] = data_format[key].decode()
        else:
            data_format = {}

        if "dimension_order" not in data_format:
            data_format["dimension_order"] = "CW"
        return data_format

    def _unify_component_order(self):
        if "component_order" in self.data_format:
            if "trace_component_order" not in self.metadata.columns:
                self._metadata["trace_component_order"] = self.data_format["component_order"]

    def _build_trace_name_to_idx_dict(self):
        self._trace_name_to_idx = {}
        if "trace_name" in self.metadata.columns:
            self._trace_name_to_idx["name"] = {
                (trace_name,): i for i, trace_name in enumerate(self.metadata["trace_name"])
            }
        else:
            self._trace_name_to_idx["name"] = {}
        self._trace_identification_warning_issued = False

    def preload_waveforms(self, pbar=False):
        if self.cache is None:
            logger.warning("Skipping preload, as cache is disabled.")
            return

        chunks, metadata_paths, waveforms_path = self._chunks_with_paths()
        with LoadingContext(chunks, waveforms_path) as context:
            iterator = zip(self._metadata["trace_name"], self._metadata["trace_chunk"])
            if pbar:
                iterator = tqdm(iterator, total=len(self._metadata), desc="Preloading waveforms")

            for trace_name, chunk in iterator:
                self._get_single_waveform(trace_name, chunk, context=context)

    def filter(self, mask, inplace=True):
        if inplace:
            self._metadata = self._metadata[mask]
            self._evict_cache()
            self._build_trace_name_to_idx_dict()
            self._rebuild_metadata_cache()
            self.grouping = self.grouping # triggers regrouping
            return self
        else:
            other = self.copy()
            other.filter(mask, inplace=True)
            return other

    def get_split(self, split):
        if "split" not in self.metadata.columns:
            raise ValueError("Split requested but no split defined in metadata")
        mask = (self.metadata["split"] == split).values
        return self.filter(mask, inplace=False)

    def train(self): return self.get_split("train")
    def dev(self): return self.get_split("dev")
    def test(self): return self.get_split("test")

    def _evict_cache(self):
        pass

    def __getitem__(self, item):
        if not isinstance(item, str):
            raise TypeError("Can only use strings to access metadata parameters")
        return self._metadata[item]

    def __len__(self):
        return len(self._metadata)

    def get_sample(self, idx, sampling_rate=None):
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

    def _get_sample_unify_sampling_rate(self, metadata: dict, sampling_rate: Optional[float]):
        if sampling_rate is None:
            sampling_rate = self.sampling_rate
        
        if sampling_rate is not None:
            source_sr = metadata["trace_sampling_rate_hz"]
            if np.isnan(source_sr).any():
                raise ValueError("Unknown sampling rate.")
            
            metadata["trace_source_sampling_rate_hz"] = np.asarray(source_sr)
            metadata["trace_sampling_rate_hz"] = np.ones_like(metadata["trace_sampling_rate_hz"]) * sampling_rate
            metadata["trace_dt_s"] = 1.0 / metadata["trace_sampling_rate_hz"]
        else:
            metadata["trace_source_sampling_rate_hz"] = np.asarray(metadata["trace_sampling_rate_hz"])
        
        return sampling_rate

    def _get_waveforms_from_load_metadata(self, load_metadata, sampling_rate, pack=True):
        waveforms = {}
        chunks, _, waveforms_path = self._chunks_with_paths()

        segments = [
            (trace_name, chunk, float(trace_sr), trace_co)
            for trace_name, chunk, trace_sr, trace_co in zip(
                load_metadata["trace_name"],
                load_metadata["trace_chunk"],
                load_metadata["trace_source_sampling_rate_hz"],
                load_metadata["trace_component_order"],
            )
        ]

        with LoadingContext(chunks, waveforms_path) as context:
            for segment in segments:
                if segment in waveforms: continue
                trace_name, chunk, trace_sr, trace_co = segment
                waveforms[segment] = self._get_single_waveform(
                    trace_name, chunk, context,
                    target_sampling_rate=sampling_rate,
                    source_sampling_rate=trace_sr,
                    source_component_order=trace_co
                )

        if pack:
            waveforms = [waveforms[segment] for segment in segments]
            waveforms = pad_packed_sequence(waveforms)
            waveforms = waveforms.transpose(*self._dimension_mapping)
        else:
            waveforms = [waveforms[segment] for segment in segments]
        
        return waveforms

    def _get_single_waveform(self, trace_name, chunk, context, target_sampling_rate=None, source_sampling_rate=None, source_component_order=None):
        trace_name = str(trace_name)
        
        # Caching logic
        if trace_name in self._waveform_cache[chunk]:
            waveform = self._waveform_cache[chunk][trace_name]
        else:
            if "$" in trace_name:
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

        # Resampling
        if target_sampling_rate is not None and not np.isnan(source_sampling_rate):
             waveform = self._resample(waveform, target_sampling_rate, source_sampling_rate)

        # Component ordering
        if source_component_order is not None and self._data_format.get("dimension_order"):
             dim_order = self._data_format["dimension_order"]
             if "C" in dim_order:
                 comp_dim = dim_order.index("C")
                 comp_mapping = self._component_mapping[source_component_order]
                 
                 if waveform.shape[comp_dim] == max(comp_mapping):
                     pad = [(0,0)] * waveform.ndim
                     pad[comp_dim] = (0, 1)
                     waveform = np.pad(waveform, pad, "constant")
                 
                 waveform = waveform.take(comp_mapping, axis=comp_dim)

        return waveform

    @staticmethod
    def _parse_location(location):
        location = location.replace(" ", "")
        slices = []
        for dim_slice in location.split(","):
            parts = dim_slice.split(":")
            if len(parts) == 1:
                slices.append(int(parts[0]) if parts[0] else None)
            elif len(parts) == 2:
                slices.append(slice(int(parts[0]) if parts[0] else None, int(parts[1]) if parts[1] else None))
            elif len(parts) == 3:
                slices.append(slice(int(parts[0]) if parts[0] else None, int(parts[1]) if parts[1] else None, int(parts[2]) if parts[2] else None))
        return tuple(slices)

    def _resample(self, waveform, target, source, eps=1e-4):
        try:
            sample_axis = self._data_format["dimension_order"].index("W")
        except: return waveform

        if abs(target/source - 1) < eps:
            return waveform
        
        if (source % target) < eps:
            q = int(source // target)
            return scipy.signal.decimate(waveform, q, axis=sample_axis)
        else:
            num = int(waveform.shape[sample_axis] * target / source)
            return scipy.signal.resample(waveform, num, axis=sample_axis)


class WaveformBenchmarkDataset(WaveformDataset, ABC):
    """
    Base class for benchmark datasets which simply need to download
    pre-processed data from a remote location if not present.
    """
    _files: list[str] = ["waveforms$CHUNK.hdf5", "metadata$CHUNK.csv"]

    def __init__(
        self,
        chunks: Optional[list[str]] = None,
        citation: Optional[str] = None,
        license: Optional[str] = None,
        force: bool = False,
        wait_for_file: bool = False,
        **kwargs,
    ):
        self._name = self.__class__.__name__
        self._citation = citation
        self._license = license
        self.path.mkdir(exist_ok=True, parents=True)

        if chunks is None:
            chunks = self.available_chunks(force=force)

        for chunk in chunks:
             files_needed = [self.path / f.replace("$CHUNK", chunk) for f in self._files]
             if force or any(not p.exists() for p in files_needed):
                 self._download_chunk(chunk, files_needed)

        super().__init__(chunks=chunks, path=self.path, name=self._name, **kwargs)

    @property
    def path(self):
        return CONFIG["cache_data_root"] / self._name.lower()

    @classmethod
    def _remote_path(cls):
         return urljoin(CONFIG["remote_data_root"], cls.__name__)

    def available_chunks(self, force=False):
        # Check if local files exist using parent logic
        local_chunks = WaveformDataset.available_chunks(self.path)
        if local_chunks:
            return local_chunks
        # Default to empty chunk if not found (to trigger download of default files)
        return [""]

    def _download_chunk(self, chunk, output_files):
        logger.info(f"Downloading chunk '{chunk}' for {self._name}...")
        remote_base = self._remote_path()
        for template, output_file in zip(self._files, output_files):
             fname = template.replace("$CHUNK", chunk)
             url = f"{remote_base}/{fname}"
             util.download_http(url, output_file)

class MultiWaveformDataset:
    pass
