"""
DiTing数据集处理模块。
继承自WaveformDataset，专门处理DiTing数据格式。
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, List
from pathlib import Path

from seispolarity.data.base import WaveformDataset

logger = logging.getLogger(__name__)

class DiTingDataset(WaveformDataset):
    """
    DiTing专用的WaveformDataset子类。
    
    处理DiTing数据格式：
    - X: 波形数据 (N, 128)
    - Y: 极性标签 ('U', 'D', 'X')
    - Z: 清晰度标签 ('E', 'I', 'K')
    - p_pick: P波到时索引 (相对位置，约64)
    
    注意：DiTing数据已经过增强处理，U、D、X三类平衡。
    """
    
    def __init__(self, h5_path: str, 
                 preload: bool = False, 
                 allowed_labels: Optional[List[int]] = None,
                 **kwargs):
        """
        初始化DiTing数据集。
        
        Args:
            h5_path: HDF5文件路径
            preload: 是否预加载到内存
            allowed_labels: 允许的标签列表 [0, 1, 2] 对应 [U, D, X]
            **kwargs: 传递给WaveformDataset的额外参数
        """
        # DiTing特定参数
        diting_kwargs = {
            "path": h5_path,
            "name": "DiTing",
            "preload": preload,
            "allowed_labels": allowed_labels,
            "sampling_rate": 100,  # DiTing增强后是100Hz
            "component_order": "Z",  # DiTing是单分量
            # DiTing使用标准键名：X, Y, Z, p_pick
            "data_key": "X",
            "label_key": "Y",
            "clarity_key": "Z",
            "pick_key": "p_pick",
            "metadata_keys": []  # DiTing不需要额外的元数据键
        }
        
        # 合并用户提供的kwargs
        diting_kwargs.update(kwargs)
        
        # 调用父类初始化
        super().__init__(**diting_kwargs)
        
        # DiTing特定属性
        self._label_map = {'U': 0, 'D': 1, 'X': 2}
        self._reverse_label_map = {0: 'U', 1: 'D', 2: 'X'}
        
        logger.info(f"DiTingDataset initialized with {len(self)} samples")
    
    def _load_metadata(self):
        """
        重写元数据加载方法，处理DiTing的字符标签。
        """
        # 调用父类方法加载基本元数据
        metadata = super()._load_metadata()
        
        if metadata.empty:
            # 如果没有CSV元数据，从HDF5创建虚拟元数据
            if self._path and Path(self._path).is_file():
                import h5py
                with h5py.File(self._path, 'r') as f:
                    if self.label_key in f:
                        N = f[self.label_key].shape[0]
                        # 创建虚拟元数据
                        meta_dict = {
                            'trace_chunk': [''] * N,
                            'label': self.convert_labels(f[self.label_key][:])
                        }
                        
                        # 添加清晰度标签
                        if self.clarity_key in f:
                            meta_dict['clarity'] = f[self.clarity_key][:]
                        
                        # 添加P波到时
                        if self.pick_key in f:
                            meta_dict['p_pick'] = f[self.pick_key][:]
                        
                        metadata = pd.DataFrame(meta_dict)
        
        return metadata
    
    def _get_single_item(self, idx):
        """
        重写单个样本获取方法，添加DiTing特定元数据。
        """
        waveform, metadata = super()._get_single_item(idx)
        
        # 添加DiTing特定元数据
        if 'clarity' in self._metadata.columns:
            metadata['clarity'] = self._metadata.iloc[idx]['clarity']
        
        if 'p_pick' in self._metadata.columns:
            metadata['p_pick'] = self._metadata.iloc[idx]['p_pick']
        
        # 添加原始字符标签（使用基类的通用方法）
        if 'label' in metadata:
            num_label = metadata['label']
            char_labels = self.convert_labels_to_chars([num_label])
            metadata['label_char'] = char_labels[0] if char_labels else 'X'
        
        return waveform, metadata
    
    def get_dataloader(self, batch_size=32, num_workers=0, shuffle=False, 
                       window_p0=None, window_len=None):
        """
        重写DataLoader创建方法，处理DiTing的128点数据。
        
        DiTing数据已经是128点，不需要窗口化。
        如果指定了window_p0和window_len，会进行相应的处理。
        """
        # DiTing数据默认不需要窗口化
        if window_p0 is None and window_len is None:
            # 使用父类的DataLoader，但跳过窗口化
            return super().get_dataloader(
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=shuffle,
                window_p0=0,  # 不裁剪
                window_len=128  # 保持原长度
            )
        else:
            # 使用指定的窗口参数
            return super().get_dataloader(
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=shuffle,
                window_p0=window_p0,
                window_len=window_len
            )
    
    @property
    def label_map(self):
        """获取标签映射字典"""
        return self._label_map
    
    @property
    def reverse_label_map(self):
        """获取反向标签映射字典"""
        return self._reverse_label_map


class SCSNForDiTingDataset(WaveformDataset):
    """
    SCSN数据集适配器，使其与DiTing数据格式兼容。
    
    处理：
    1. 添加Z标签（全部设为'K'）
    2. 添加p_pick标签（全部设为300）
    3. 将600点数据裁剪到128点（以p_pick=300为中心）
    """
    
    def __init__(self, h5_path: str, 
                 preload: bool = False, 
                 allowed_labels: Optional[List[int]] = None,
                 **kwargs):
        """
        初始化SCSN适配器数据集。
        
        Args:
            h5_path: SCSN HDF5文件路径
            preload: 是否预加载到内存
            allowed_labels: 允许的标签列表 [0, 1, 2] 对应 [U, D, X]
            **kwargs: 传递给WaveformDataset的额外参数
        """
        # SCSN特定参数
        scsn_kwargs = {
            "path": h5_path,
            "name": "SCSN_For_DiTing",
            "preload": preload,
            "allowed_labels": allowed_labels,
            "sampling_rate": 100,  # SCSN采样率
            "component_order": "Z",  # SCSN是单分量
            # SCSN使用标准键名：X, Y
            "data_key": "X",
            "label_key": "Y",
            "clarity_key": None,  # SCSN没有清晰度标签
            "pick_key": None,     # SCSN没有p_pick标签
            "metadata_keys": []   # SCSN不需要额外的元数据键
        }
        
        # 合并用户提供的kwargs
        scsn_kwargs.update(kwargs)
        
        # 调用父类初始化
        super().__init__(**scsn_kwargs)
        
        # SCSN特定属性
        self._label_map = {0: 'U', 1: 'D', 2: 'X'}
        self._reverse_label_map = {'U': 0, 'D': 1, 'X': 2}
        
        logger.info(f"SCSNForDiTingDataset initialized with {len(self)} samples")
    
    def _load_metadata(self):
        """
        重写元数据加载方法，添加SCSN的Z标签和p_pick。
        """
        # 调用父类方法加载基本元数据
        metadata = super()._load_metadata()
        
        if metadata.empty:
            # 如果没有CSV元数据，从HDF5创建虚拟元数据
            if self._path and Path(self._path).is_file():
                import h5py
                with h5py.File(self._path, 'r') as f:
                    if self.label_key in f:
                        N = f[self.label_key].shape[0]
                        # 创建虚拟元数据
                        meta_dict = {
                            'trace_chunk': [''] * N,
                            'label': f[self.label_key][:]  # SCSN标签已经是数字
                        }
                        
                        # 添加清晰度标签（全部设为'K'）
                        meta_dict['clarity'] = np.full(N, 'K', dtype='S1')
                        
                        # 添加P波到时（全部设为300）
                        meta_dict['p_pick'] = np.full(N, 300.0, dtype=np.float32)
                        
                        metadata = pd.DataFrame(meta_dict)
        
        return metadata
    
    def _get_single_item(self, idx):
        """
        重写单个样本获取方法，裁剪SCSN数据到128点。
        """
        waveform, metadata = super()._get_single_item(idx)
        
        # SCSN数据裁剪：从600点裁剪到128点（以p_pick=300为中心）
        if waveform.shape[1] == 600:  # SCSN原始长度
            p_pick = 300  # SCSN的P波理论位置
            half_window = 64  # 128/2
            
            # 计算裁剪范围
            start_idx = p_pick - half_window
            end_idx = p_pick + half_window
            
            # 处理边界情况
            if start_idx < 0:
                # 左侧填充
                padded = np.pad(waveform[0, :end_idx], (abs(start_idx), 0), 'constant')
                waveform = padded.reshape(1, -1)
            elif end_idx > waveform.shape[1]:
                # 右侧填充
                padded = np.pad(waveform[0, start_idx:], (0, end_idx - waveform.shape[1]), 'constant')
                waveform = padded.reshape(1, -1)
            else:
                waveform = waveform[:, start_idx:end_idx]
            
            # 确保长度是128
            if waveform.shape[1] != 128:
                if waveform.shape[1] < 128:
                    waveform = np.pad(waveform, ((0, 0), (0, 128 - waveform.shape[1])), 'constant')
                else:
                    waveform = waveform[:, :128]
        
        # 添加SCSN特定元数据
        if 'clarity' in self._metadata.columns:
            metadata['clarity'] = self._metadata.iloc[idx]['clarity']
        
        if 'p_pick' in self._metadata.columns:
            metadata['p_pick'] = self._metadata.iloc[idx]['p_pick']
        
        # 添加原始字符标签（使用基类的通用方法）
        if 'label' in metadata:
            num_label = metadata['label']
            char_labels = self.convert_labels_to_chars([num_label])
            metadata['label_char'] = char_labels[0] if char_labels else 'X'
        
        return waveform, metadata
    
    def get_dataloader(self, batch_size=32, num_workers=0, shuffle=False, 
                       window_p0=None, window_len=None):
        """
        重写DataLoader创建方法，SCSN已经裁剪到128点，不需要窗口化。
        """
        # SCSN数据已经裁剪到128点，不需要窗口化
        if window_p0 is None and window_len is None:
            # 使用父类的DataLoader，但跳过窗口化
            return super().get_dataloader(
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=shuffle,
                window_p0=0,  # 不裁剪
                window_len=128  # 保持原长度
            )
        else:
            # 使用指定的窗口参数
            return super().get_dataloader(
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=shuffle,
                window_p0=window_p0,
                window_len=window_len
            )


def create_combined_dataset(diting_path, scsn_path, preload=False, allowed_labels=None):
    """
    创建DiTing和SCSN的合并数据集。
    
    Args:
        diting_path: DiTing HDF5文件路径
        scsn_path: SCSN HDF5文件路径
        preload: 是否预加载到内存
        allowed_labels: 允许的标签列表 [0, 1, 2] 对应 [U, D, X]
        
    Returns:
        MultiWaveformDataset: 合并的数据集
    """
    from seispolarity.data.base import MultiWaveformDataset
    
    # 创建DiTing数据集
    diting_ds = DiTingDataset(
        h5_path=diting_path,
        preload=preload,
        allowed_labels=allowed_labels
    )
    
    # 创建SCSN适配器数据集（裁剪到128点，添加Z标签）
    scsn_ds = SCSNForDiTingDataset(
        h5_path=scsn_path,
        preload=preload,
        allowed_labels=allowed_labels
    )
    
    # 合并数据集
    combined = MultiWaveformDataset([diting_ds, scsn_ds])
    
    logger.info(f"创建合并数据集: DiTing({len(diting_ds)}) + SCSN({len(scsn_ds)}) = {len(combined)}")
    
    return combined
